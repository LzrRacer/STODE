#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors # For explicit color mapping
import sys
import traceback
from tqdm import tqdm

# --- Setup Project Paths and PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.vae import VAEWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze simulated trajectories: sample timepoints, decode, cluster, and find DEGs.")
    parser.add_argument("--simulation_dir", type=str, required=True, help="Directory containing simulation outputs.")
    parser.add_argument("--simulation_type", type=str, default="backward", choices=["forward", "backward"])
    parser.add_argument("--model_load_path", type=str, required=True, help="Path to trained system model.")
    parser.add_argument("--config_train_load_path", type=str, required=True, help="Path to training configuration JSON.")
    parser.add_argument("--original_adata_path", type=str, required=True, help="Path to original AnnData for gene names, etc.")
    
    parser.add_argument("--target_bio_times", type=float, nargs='*', default=None, help="Specific biological timepoints to sample.")
    parser.add_argument("--num_timepoints_to_sample_fallback", type=int, default=5, help="Number of timepoints to sample if target_bio_times not given.")
    
    # NEW: Clustering method selection
    parser.add_argument("--clustering_method", type=str, default="kmeans", choices=["kmeans", "annotation"], 
                       help="Method for clustering: 'kmeans' for de novo clustering, 'annotation' to use observed annotations")
    parser.add_argument("--annotation_key", type=str, default="annotation", 
                       help="Key in original_adata.obs to use for annotation-based clustering (used when clustering_method='annotation')")
    
    parser.add_argument("--num_clusters", type=int, default=8, help="Number of K-Means clusters (used when clustering_method='kmeans').")
    parser.add_argument("--n_top_degs", type=int, default=10, help="Number of top DEGs for lists and plots.")
    parser.add_argument("--spatial_plot_spot_size", type=float, default=10, help="Spot size for spatial plots.")
    
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for analysis results.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()

def assign_clusters_by_annotation(adata_decoded, original_adata, annotation_key, vae_model, device):
    """
    Assign clusters based on observed cell type annotations.
    Maps simulated cells to observed annotation clusters using VAE latent space similarity.
    
    Args:
        adata_decoded: AnnData with simulated/decoded cells
        original_adata: Original AnnData with annotations
        annotation_key: Key for annotations in original_adata.obs
        vae_model: Pre-loaded VAE model
        device: PyTorch device
    """
    print(f"Using annotation-based clustering with key: '{annotation_key}'")
    
    # Check if annotation key exists
    if annotation_key not in original_adata.obs.columns:
        raise ValueError(f"Annotation key '{annotation_key}' not found in original data obs columns: {list(original_adata.obs.columns)}")
    
    print("Computing latent representations for original annotated data...")
    original_expr = torch.tensor(original_adata.X.astype(np.float32)).to(device)
    with torch.no_grad():
        original_latent, _ = vae_model.vae.encoder(original_expr)
    original_latent = original_latent.cpu().numpy()
    
    # Get annotation centroids in latent space
    annotation_centroids = {}
    unique_annotations = original_adata.obs[annotation_key].unique()
    
    for ann in unique_annotations:
        ann_mask = original_adata.obs[annotation_key] == ann
        if ann_mask.sum() > 0:
            ann_latent = original_latent[ann_mask]
            annotation_centroids[ann] = ann_latent.mean(axis=0)
    
    print(f"Found {len(annotation_centroids)} annotation clusters: {list(annotation_centroids.keys())}")
    
    # Assign simulated cells to nearest annotation centroid
    print("Computing latent representations for simulated data...")
    sim_expr = torch.tensor(adata_decoded.X.astype(np.float32)).to(device)
    with torch.no_grad():
        sim_latent, _ = vae_model.vae.encoder(sim_expr)
    sim_latent = sim_latent.cpu().numpy()
    
    # Find nearest centroid for each simulated cell
    cluster_assignments = []
    centroid_names = list(annotation_centroids.keys())
    centroid_coords = np.array([annotation_centroids[name] for name in centroid_names])
    
    print("Assigning simulated cells to annotation clusters...")
    for i in tqdm(range(sim_latent.shape[0]), desc="Assigning cells", leave=False):
        distances = np.linalg.norm(centroid_coords - sim_latent[i], axis=1)
        nearest_idx = np.argmin(distances)
        cluster_assignments.append(centroid_names[nearest_idx])
    
    # Add cluster assignments to adata_decoded
    adata_decoded.obs['annotation_cluster'] = pd.Categorical(cluster_assignments)
    cluster_key = 'annotation_cluster'
    
    print(f"Assigned simulated cells to annotation clusters. Distribution:")
    print(adata_decoded.obs[cluster_key].value_counts().sort_index())
    
    return cluster_key


def assign_clusters_by_kmeans(adata_decoded, num_clusters):
    """Assign clusters using K-means clustering"""
    print(f"Using K-means clustering with {num_clusters} clusters")
    
    if adata_decoded.n_obs >= num_clusters:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        adata_decoded.obs['kmeans_cluster'] = kmeans.fit_predict(adata_decoded.X).astype(str)
    else: 
        adata_decoded.obs['kmeans_cluster'] = '0'
    adata_decoded.obs['kmeans_cluster'] = adata_decoded.obs['kmeans_cluster'].astype('category')
    cluster_key = 'kmeans_cluster'
    
    print(f"K-means clustering complete. Distribution:")
    print(adata_decoded.obs[cluster_key].value_counts().sort_index())
    
    return cluster_key

# --- NEW FUNCTION START ---
def generate_presentation_heatmap(adata_gdeg, cluster_key, deg_heatmaps_dir, n_top_clusters=10, n_top_genes=5):
    """
    Generates a simplified, presentation-friendly heatmap.
    
    It selects the N most abundant clusters and the top M DEGs for each of them,
    then creates a matrix plot.
    
    Args:
        adata_gdeg: AnnData object with global DEG results in .uns['rank_genes_global'].
        cluster_key: The key in .obs for cluster labels.
        deg_heatmaps_dir: Path object for the output directory.
        n_top_clusters: Number of most abundant clusters to include.
        n_top_genes: Number of top genes to select for each cluster.
    """
    print("\n--- Generating Presentation-Friendly Heatmap ---")
    
    if 'rank_genes_global' not in adata_gdeg.uns:
        print("  ✗ Skipping: 'rank_genes_global' not found in AnnData object.")
        return

    # 1. Find the top N most abundant clusters
    cluster_counts = adata_gdeg.obs[cluster_key].value_counts()
    top_clusters = cluster_counts.nlargest(n_top_clusters).index.tolist()
    print(f"  - Found {len(cluster_counts)} total clusters. Selecting top {len(top_clusters)} for the plot.")
    print(f"    Top clusters: {top_clusters}")

    # 2. Get the top M genes for these specific clusters
    presentation_gene_list = []
    rank_genes_df = adata_gdeg.uns['rank_genes_global']['names']
    
    for cluster_id in top_clusters:
        if cluster_id in rank_genes_df.dtype.fields:
            genes = rank_genes_df[cluster_id][:n_top_genes]
            presentation_gene_list.extend(list(genes))
        else:
            print(f"    - Warning: Cluster '{cluster_id}' not found in DEG results.")

    # 3. Create a unique list of genes to plot
    unique_presentation_genes = pd.unique(presentation_gene_list).tolist()
    
    if not unique_presentation_genes:
        print("  ✗ Skipping: No genes found for the selected top clusters.")
        return
        
    # Ensure genes are actually in the AnnData object
    unique_presentation_genes = [g for g in unique_presentation_genes if g in adata_gdeg.var_names]
    print(f"  - Plotting {len(unique_presentation_genes)} unique top genes.")

    # 4. Create a subset of the AnnData object with only the top clusters
    adata_subset = adata_gdeg[adata_gdeg.obs[cluster_key].isin(top_clusters)].copy()
    # Ensure the categorical order is correct for plotting
    adata_subset.obs[cluster_key] = adata_subset.obs[cluster_key].cat.reorder_categories(top_clusters, ordered=True)

    # 5. Generate and save the matrix plot
    try:
        print("  - Generating presentation matrixplot...")
        fig_size = (max(8, 0.4 * len(unique_presentation_genes)), max(6, 0.5 * len(top_clusters)))
        
        sc.pl.matrixplot(adata_subset, 
                      var_names=unique_presentation_genes, 
                      groupby=cluster_key, 
                      cmap='viridis', 
                      dendrogram=True, 
                      standard_scale='var', 
                      use_raw=False, 
                      show=False, 
                      swap_axes=False, 
                      figsize=fig_size)
        
        ax = plt.gca()
        plt.setp(ax.get_xticklabels(), rotation=90, ha='right', va='center')
        plt.title(f"Top {n_top_genes} DEGs for Top {len(top_clusters)} Clusters")
        plt.tight_layout()
        
        output_filename = deg_heatmaps_dir / f"_global_presentation_heatmap_top{n_top_clusters}clusters_{n_top_genes}genes.png"
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Presentation heatmap saved successfully to:\n    {output_filename}")

    except Exception as e:
        print(f"  ✗ Error generating presentation matrixplot: {e}")
        plt.close()
# --- NEW FUNCTION END ---


def perform_global_deg_analysis(adata_decoded, cluster_key, args, deg_heatmaps_dir):
    """Fixed global DEG analysis"""
    print(f"Performing global DEG analysis and heatmap (all sampled times) into {deg_heatmaps_dir}...")
    adata_gdeg = adata_decoded.copy()
    
    if adata_gdeg.obs[cluster_key].nunique() >= 2:
        try:
            print("  Running global differential expression analysis...")
            sc.tl.rank_genes_groups(adata_gdeg, cluster_key, method='wilcoxon', 
                                    n_genes=max(args.n_top_degs * len(adata_gdeg.obs[cluster_key].cat.categories), 100), 
                                    key_added="rank_genes_global", pts=True, use_raw=False)
            
            if 'rank_genes_global' in adata_gdeg.uns:
                print("  Generating global rank genes heatmap...")
                
                try:
                    # Let scanpy create its own figure
                    fig_size = (12, max(8, 0.5 * args.n_top_degs * len(adata_gdeg.obs[cluster_key].cat.categories)))
                    sc.pl.rank_genes_groups_heatmap(
                        adata_gdeg, n_genes=args.n_top_degs, groupby=cluster_key, 
                        key="rank_genes_global", show=False,  # No ax parameter
                        vmin=-2.5, vmax=2.5, cmap='bwr', dendrogram=True, show_gene_labels=True,
                        figsize=fig_size
                    )
                    plt.tight_layout()
                    plt.savefig(deg_heatmaps_dir / "_global_rank_genes_heatmap.png", dpi=150, bbox_inches='tight')
                    plt.close()
                    print("    ✓ Global rank genes heatmap saved successfully.")
                except Exception as e_heatmap:
                    print(f"    ✗ Error generating rank genes heatmap: {e_heatmap}")
                    plt.close()

                # Generate expression matrixplot
                global_deg_gene_list = []
                for cl_id in sorted(adata_gdeg.obs[cluster_key].cat.categories):
                    if cl_id in adata_gdeg.uns['rank_genes_global']['names'].dtype.fields:
                        genes = adata_gdeg.uns['rank_genes_global']['names'][cl_id][:args.n_top_degs]
                        global_deg_gene_list.extend(list(genes))
            
                unique_global_degs = pd.unique(np.array(global_deg_gene_list)) if global_deg_gene_list else []
                if len(unique_global_degs) > 0:
                    unique_global_degs_present = [g for g in unique_global_degs if g in adata_gdeg.var_names]
                    if unique_global_degs_present:
                        print(f"    Generating global expression matrixplot for {len(unique_global_degs_present)} genes...")
                        
                        try:
                            fig_size = (max(12, 0.4*len(unique_global_degs_present)), max(8, 0.6*len(adata_gdeg.obs[cluster_key].cat.categories)))
                            sc.pl.matrixplot(adata_gdeg, var_names=unique_global_degs_present, 
                                          groupby=cluster_key, 
                                          cmap='viridis', dendrogram=True, standard_scale='var', 
                                          use_raw=False, show=False, swap_axes=False, figsize=fig_size)
                            
                            ax = plt.gca()
                            plt.setp(ax.get_xticklabels(), rotation=90, ha='right', va='center')
                            plt.tight_layout()
                            plt.savefig(deg_heatmaps_dir / "_global_expression_matrixplot.png", 
                                      dpi=150, bbox_inches='tight')
                            plt.close()
                            print("    ✓ Global expression matrixplot saved successfully.")
                        except Exception as e_matrix:
                            print(f"    ✗ Error generating global matrixplot: {e_matrix}")
                            plt.close()

                # --- NEW CODE: CALL THE PRESENTATION PLOT FUNCTION ---
                # This will generate the focused plot with top 10 cell types and top 5 genes.
                generate_presentation_heatmap(
                    adata_gdeg=adata_gdeg,
                    cluster_key=cluster_key,
                    deg_heatmaps_dir=deg_heatmaps_dir,
                    n_top_clusters=10,
                    n_top_genes=5
                )
                # --- END NEW CODE ---
                            
            print("  ✓ Global DEG analysis completed successfully.")
        except Exception as e_global_deg: 
            print(f"  ✗ Error in global DEG analysis: {e_global_deg}")
            traceback.print_exc()
    else: 
        print("  Skipping global DEG: Not enough distinct clusters.")

def perform_per_timepoint_deg_analysis(adata_decoded, cluster_key, sampled_bio_times, args, deg_heatmaps_dir):
    """Fixed per-timepoint DEG analysis that handles missing clusters properly"""
    print("Performing per-timepoint DEG analysis and heatmaps...")
    all_deg_results_per_time = []
    
    for sampled_time_idx, bio_time in enumerate(sampled_bio_times):
        adata_t = adata_decoded[adata_decoded.obs['sim_time_point_idx_in_sample'] == sampled_time_idx].copy()
        
        print(f"\n  === Time {bio_time:.3f} ===")
        print(f"  Total cells: {adata_t.n_obs}")
        
        # Check which clusters are actually present at this timepoint
        present_clusters = adata_t.obs[cluster_key].value_counts()
        print(f"  Present clusters: {dict(present_clusters.sort_index())}")
        
        # Filter to clusters with sufficient cells
        min_cells_per_cluster = 5
        sufficient_clusters = present_clusters[present_clusters >= min_cells_per_cluster]
        
        print(f"  Clusters with ≥{min_cells_per_cluster} cells: {len(sufficient_clusters)}")
        
        if len(sufficient_clusters) < 2:
            print(f"  ✗ Skipping DEG analysis: need ≥2 clusters with sufficient cells")
            continue
        
        # Create filtered dataset with only sufficient clusters
        valid_cluster_ids = sufficient_clusters.index.tolist()
        adata_t_filtered = adata_t[adata_t.obs[cluster_key].isin(valid_cluster_ids)].copy()
        
        # CRITICAL FIX: Only use cluster categories that are actually present
        # Don't force global categories that don't exist at this timepoint
        present_categories = sorted(adata_t_filtered.obs[cluster_key].unique())
        adata_t_filtered.obs[cluster_key] = pd.Categorical(
            adata_t_filtered.obs[cluster_key], 
            categories=present_categories,  # Only present categories
            ordered=False
        )
        
        print(f"  Final clusters for analysis: {present_categories}")
        print(f"  Filtered dataset: {adata_t_filtered.n_obs} cells")

        try:
            print(f"  Running differential expression analysis...")
            sc.tl.rank_genes_groups(adata_t_filtered, cluster_key, method='wilcoxon', 
                                    n_genes=min(adata_t_filtered.n_vars, 500),  # Limit genes for efficiency
                                    key_added="rank_genes_degs_t", 
                                    pts=True, use_raw=False)
            
            # Collect top DEGs
            time_deg_gene_list = []
            if 'rank_genes_degs_t' in adata_t_filtered.uns:
                for cl_id_t in present_categories:  # Only check present categories
                     if cl_id_t in adata_t_filtered.uns['rank_genes_degs_t']['names'].dtype.fields:
                        genes_t = adata_t_filtered.uns['rank_genes_degs_t']['names'][cl_id_t][:args.n_top_degs]
                        time_deg_gene_list.extend(list(genes_t))
            
            unique_time_degs = pd.unique(np.array(time_deg_gene_list)) if time_deg_gene_list else []
            
            if len(unique_time_degs) > 0:
                unique_time_degs_present = [g for g in unique_time_degs if g in adata_t_filtered.var_names]
                if unique_time_degs_present:
                    print(f"  Generating matrixplot for {len(unique_time_degs_present)} genes...")
                    
                    try:
                        fig_size = (max(12, 0.4*len(unique_time_degs_present)), 
                                  max(8, 0.6*len(present_categories)))
                        
                        sc.pl.matrixplot(adata_t_filtered, var_names=unique_time_degs_present, 
                                      groupby=cluster_key, 
                                      cmap='viridis', dendrogram=True, standard_scale='var', 
                                      use_raw=False, show=False, swap_axes=False, figsize=fig_size)
                        
                        ax = plt.gca()
                        plt.setp(ax.get_xticklabels(), rotation=90, ha='right', va='center')
                        plt.tight_layout()
                        
                        output_filename = f"_time{bio_time:.2f}_expression_matrixplot.png".replace('.','p')
                        plt.savefig(deg_heatmaps_dir / output_filename, dpi=150, bbox_inches='tight')
                        plt.close()
                        print(f"    ✓ Saved: {output_filename}")
                    except Exception as e_matrix:
                        print(f"    ✗ Error generating matrixplot: {e_matrix}")
                        plt.close()
            
            # Collect DEG results for CSV
            if 'rank_genes_degs_t' in adata_t_filtered.uns: 
                for cl_id in present_categories:  # Only present categories
                    if cl_id not in adata_t_filtered.uns['rank_genes_degs_t']['names'].dtype.fields: 
                        continue
                    deg_df = pd.DataFrame({
                        'names': adata_t_filtered.uns['rank_genes_degs_t']['names'][cl_id],
                        'scores': adata_t_filtered.uns['rank_genes_degs_t']['scores'][cl_id],
                        'pvals': adata_t_filtered.uns['rank_genes_degs_t']['pvals'][cl_id],
                        'pvals_adj': adata_t_filtered.uns['rank_genes_degs_t']['pvals_adj'][cl_id],
                        'logfoldchanges': adata_t_filtered.uns['rank_genes_degs_t']['logfoldchanges'][cl_id],
                    }).head(args.n_top_degs)
                    deg_df['sim_bio_time'] = bio_time
                    deg_df['cluster'] = cl_id
                    all_deg_results_per_time.append(deg_df)
            
            print(f"  ✓ DEG analysis completed successfully")
                    
        except Exception as e_deg_t: 
            print(f"  ✗ Error in DEG analysis: {e_deg_t}")
            traceback.print_exc()

    return all_deg_results_per_time

def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    spatial_cluster_dir = output_dir / "spatial_cluster_plots_per_timepoint"
    spatial_cluster_dir.mkdir(exist_ok=True)
    deg_heatmaps_dir = output_dir / "deg_heatmaps"
    deg_heatmaps_dir.mkdir(exist_ok=True, parents=True) 

    sc.settings.figdir = output_dir 
    sc.settings.autoshow = False
    sc.settings.autosave = False

    # --- MODIFIED: Correctly load ragged trajectory data ---
    sim_dir = Path(args.simulation_dir)
    times_file = "biological_times.npy" if args.simulation_type == "forward" else "biological_times_backward.npy"
    traj_file_npz = "state_trajectories.npz" if args.simulation_type == "forward" else "state_trajectories_backward.npz"
    
    try:
        sim_bio_times = np.load(sim_dir / times_file)
        
        # Load the .npz file containing the dictionary of arrays
        trajectory_npz = np.load(sim_dir / traj_file_npz)
        # Sort keys to ensure correct time order (e.g., 'arr_0', 'arr_1', ..., 'arr_10', ...)
        sorted_keys = sorted(trajectory_npz.keys(), key=lambda k: int(k.split('_')[1]))
        sim_state_traj_list = [trajectory_npz[key] for key in sorted_keys]

    except FileNotFoundError:
        print(f"Error: Sim files not found in {sim_dir} for type '{args.simulation_type}'. Expected .npz trajectory. Exiting.")
        return
    print(f"Loaded {args.simulation_type} simulation: {len(sim_state_traj_list)} timepoints.")
    # --- END OF MODIFIED LOADING ---

    # Load configuration and original data
    with open(args.config_train_load_path, 'r') as f: 
        train_cfg = json.load(f)
    adata_orig = sc.read_h5ad(args.original_adata_path)
    spatial_dim = train_cfg.get('spatial_dim', 2)
    latent_dim = train_cfg['vae_latent_dim']

    # Load VAE model
    vae = VAEWrapper(adata_orig.n_vars, 
                     [int(d) for d in train_cfg['vae_hidden_dims'].split(',')],
                     latent_dim, 
                     train_cfg.get('vae_dropout_rate', 0.1)).to(device).float()
    checkpoint = torch.load(args.model_load_path, map_location=device)
    vae.load_state_dict(checkpoint['vae_state_dict'])
    vae.eval()
    print("VAE model loaded.")

    # Sample timepoints (this logic remains the same, but `sampled_indices` now refers to list indices)
    if args.target_bio_times and len(args.target_bio_times) > 0:
        print(f"Attempting to sample specific target biological times: {args.target_bio_times}")
        sampled_indices_dict = {} 
        for target_t in args.target_bio_times:
            closest_idx = np.argmin(np.abs(sim_bio_times - target_t))
            if closest_idx not in sampled_indices_dict:
                 sampled_indices_dict[closest_idx] = sim_bio_times[closest_idx]
        sorted_unique_indices = sorted(sampled_indices_dict.keys())
        sampled_indices = np.array(sorted_unique_indices)
        sampled_bio_times = np.array([sampled_indices_dict[k] for k in sorted_unique_indices])
        print(f"  Actual closest unique simulated times found (sorted by sim index): {np.round(sampled_bio_times,3)}")
    else:
        num_to_sample = min(args.num_timepoints_to_sample_fallback, len(sim_state_traj_list))
        sampled_indices = np.linspace(0, len(sim_state_traj_list) - 1, num_to_sample, dtype=int)
        sampled_bio_times = sim_bio_times[sampled_indices]
    
    if len(sampled_indices) == 0: 
        print("Error: No timepoints sampled.")
        return
    
    print(f"Processing {len(sampled_bio_times)} timepoints: {np.round(sampled_bio_times, 2)}")
    
    # --- MODIFIED: Iterative construction of AnnData components ---
    # --- CORRECTED ITERATIVE AnnData CONSTRUCTION ---
    all_decoded_expr_list = []
    all_spatial_coords_list = []
    all_obs_list = []
    global_cell_idx = 0

    for i, sim_idx in enumerate(tqdm(sampled_indices, desc="Building AnnData")):
        bio_time = sim_bio_times[sim_idx]
        state_array_t = sim_state_traj_list[sim_idx]
        n_cells_t = state_array_t.shape[0]
        
        if n_cells_t == 0:
            continue
            
        latent_states_t = torch.tensor(state_array_t[:, spatial_dim:], dtype=torch.float32).to(device)
        with torch.no_grad():
            decoded_expr_t = vae.decode(latent_states_t).cpu().numpy()
        all_decoded_expr_list.append(decoded_expr_t)
        
        spatial_coords_t = state_array_t[:, :spatial_dim]
        all_spatial_coords_list.append(spatial_coords_t)
        
        obs_df_t = pd.DataFrame({
            'sim_time_point_idx_in_sample': i,
            'sim_bio_time': bio_time,
        }, index=[f"cell_{j}" for j in range(global_cell_idx, global_cell_idx + n_cells_t)])
        all_obs_list.append(obs_df_t)
        global_cell_idx += n_cells_t

    if not all_decoded_expr_list:
        print("Error: No cells found in the sampled timepoints. Exiting.")
        return
        
    all_decoded_expr = np.concatenate(all_decoded_expr_list, axis=0)
    all_spatial_coords = np.concatenate(all_spatial_coords_list, axis=0)
    obs_dataframe = pd.concat(all_obs_list)
    
    var_dataframe = pd.DataFrame(index=adata_orig.var_names)
    adata_decoded = sc.AnnData(X=all_decoded_expr, obs=obs_dataframe, var=var_dataframe, obsm={'spatial': all_spatial_coords})
    print(f"Created AnnData for decoded expression: {adata_decoded.shape}")

    # CLUSTERING STEP - Choose method based on args
    if args.clustering_method == "annotation":
        try:
            cluster_key = assign_clusters_by_annotation(adata_decoded, adata_orig, args.annotation_key, vae, device)
        except Exception as e:
            print(f"Warning: Annotation-based clustering failed: {e}")
            print("Falling back to k-means clustering")
            cluster_key = assign_clusters_by_kmeans(adata_decoded, args.num_clusters)
    else:  # kmeans
        cluster_key = assign_clusters_by_kmeans(adata_decoded, args.num_clusters)
    
    cluster_categories_global = adata_decoded.obs[cluster_key].cat.categories.tolist()
    num_global_clusters = len(cluster_categories_global)
    print(f"Using clustering method: {args.clustering_method}, cluster key: {cluster_key}")
    
    # Set up colors for clusters
    final_cluster_colors = None
    if num_global_clusters > 0:
        palette_cmap_name = 'tab10'
        if num_global_clusters > 10: palette_cmap_name = 'tab20'
        if num_global_clusters > 20:
            try:
                import distinctipy
                temp_colors = distinctipy.get_colors(num_global_clusters)
                final_cluster_colors = [matplotlib.colors.to_hex(c) for c in temp_colors]
            except ImportError:
                print("distinctipy not installed; using fallback (tab20 cycling) for >20 clusters.")
        
        if not final_cluster_colors:
            cmap_obj = plt.colormaps[palette_cmap_name] 
            final_cluster_colors = [matplotlib.colors.to_hex(cmap_obj(i % cmap_obj.N)) for i in range(num_global_clusters)]
        
        if final_cluster_colors:
             adata_decoded.uns[f'{cluster_key}_colors'] = final_cluster_colors

    # Create cluster time composition plots
    cluster_time_counts = adata_decoded.obs.groupby(['sim_bio_time', cluster_key], observed=False).size().unstack(fill_value=0)
    ordered_cols_for_plot = [cat for cat in cluster_categories_global if cat in cluster_time_counts.columns]
    if not ordered_cols_for_plot and not cluster_time_counts.empty: 
        ordered_cols_for_plot = cluster_time_counts.columns.tolist()

    cluster_time_percentages = cluster_time_counts[ordered_cols_for_plot].apply(lambda x: x / x.sum() * 100 if x.sum() > 0 else x, axis=1)
    fig_ctp, ax_ctp = plt.subplots(figsize=(max(8, 1.0 * len(sampled_bio_times)), 7)) 
    
    plot_colors_for_bar = None
    if f'{cluster_key}_colors' in adata_decoded.uns and final_cluster_colors and len(final_cluster_colors) >= len(ordered_cols_for_plot):
         try:
            plot_colors_for_bar = [final_cluster_colors[cluster_categories_global.index(col)] for col in ordered_cols_for_plot]
         except (ValueError, IndexError): plot_colors_for_bar = None
    cluster_time_percentages.plot(kind='bar', stacked=True, ax=ax_ctp, color=plot_colors_for_bar)
    ax_ctp.set_title(f'Percentage of Cells per {args.clustering_method.title()} Cluster Over Simulated Time')
    ax_ctp.set_xlabel('Simulated Biological Time'); ax_ctp.set_ylabel('Percentage of Cells (%)')
    xtick_labels = [f"{t:.2f}" for t in sorted(cluster_time_percentages.index.get_level_values('sim_bio_time').unique())]
    ax_ctp.set_xticklabels(xtick_labels, rotation=45, ha='right')
    ax_ctp.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1) 
    plt.tight_layout(rect=[0, 0, 0.85, 1]); fig_ctp.savefig(output_dir / f"cluster_proportions_over_time_{args.clustering_method}.png", dpi=150); plt.close(fig_ctp)
    print(f"Saved cluster proportions plot to {output_dir / f'cluster_proportions_over_time_{args.clustering_method}.png'}")

    # Spatial plots
    print(f"Plotting spatial distribution of {args.clustering_method} clusters per timepoint into {spatial_cluster_dir}...")
    if 'spatial' in adata_decoded.obsm and adata_decoded.obsm['spatial'].shape[0] > 0:
        xmin, ymin = adata_decoded.obsm['spatial'][:,0].min(), adata_decoded.obsm['spatial'][:,1].min()
        xmax, ymax = adata_decoded.obsm['spatial'][:,0].max(), adata_decoded.obsm['spatial'][:,1].max()
        xpad = (xmax-xmin)*0.05 if (xmax-xmin) > 1e-6 else 0.1; ypad = (ymax-ymin)*0.05 if (ymax-ymin) > 1e-6 else 0.1
    else: xmin,ymin,xmax,ymax,xpad,ypad = -1,-1,1,1,0.1,0.1

    for sampled_time_idx, bio_time in enumerate(sampled_bio_times):
        adata_t_spatial = adata_decoded[adata_decoded.obs['sim_time_point_idx_in_sample'] == sampled_time_idx].copy()
        if adata_t_spatial.n_obs == 0: continue
        
        adata_t_spatial.obs[cluster_key] = pd.Categorical(
            adata_t_spatial.obs[cluster_key], categories=cluster_categories_global, ordered=False
        )
        if f'{cluster_key}_colors' in adata_decoded.uns:
             adata_t_spatial.uns[f'{cluster_key}_colors'] = adata_decoded.uns[f'{cluster_key}_colors']

        fig_spatial_t, ax_spatial_t = plt.subplots(figsize=(8, 7))
        sc.pl.embedding(adata_t_spatial, basis="spatial", color=cluster_key, 
                        s=args.spatial_plot_spot_size, 
                        title=f"{args.clustering_method.title()} Clusters at Time {bio_time:.2f}", show=False, frameon=False,
                        legend_loc=None, ax=ax_spatial_t, 
                        palette=adata_t_spatial.uns.get(f'{cluster_key}_colors', None))
        ax_spatial_t.set_xlim(xmin - xpad, xmax + xpad)
        ax_spatial_t.set_ylim(ymin - ypad, ymax + ypad)
        ax_spatial_t.set_aspect('equal', adjustable='box')
        
        if f'{cluster_key}_colors' in adata_t_spatial.uns and adata_t_spatial.obs[cluster_key].nunique() > 0:
            present_categories = adata_t_spatial.obs[cluster_key].cat.categories[
                adata_t_spatial.obs[cluster_key].cat.categories.isin(adata_t_spatial.obs[cluster_key].unique())]
            handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=5, label=cat,
                                  markerfacecolor=adata_t_spatial.uns[f'{cluster_key}_colors'][cluster_categories_global.index(cat)]) 
                       for cat in present_categories if cat in cluster_categories_global]
            if handles: 
                ax_spatial_t.legend(handles=handles, title='Cluster', bbox_to_anchor=(1.05, 1), 
                                    loc='upper left', borderaxespad=0., fontsize='small')
        
        fig_spatial_t.tight_layout(rect=[0, 0, 0.80, 1])
        fig_spatial_t.savefig(spatial_cluster_dir / f"spatial_{args.clustering_method}_clusters_time_{bio_time:.2f}.png".replace('.', 'p'), dpi=120)
        plt.close(fig_spatial_t) 
    print("Finished plotting spatial clusters.")

    # Global DEG Analysis
    perform_global_deg_analysis(adata_decoded, cluster_key, args, deg_heatmaps_dir)

    # Per-Time DEG Analysis  
    all_deg_results_per_time = perform_per_timepoint_deg_analysis(
        adata_decoded, cluster_key, sampled_bio_times, args, deg_heatmaps_dir
    )

    if all_deg_results_per_time:
        final_deg_df = pd.concat(all_deg_results_per_time).reset_index(drop=True)
        final_deg_df.to_csv(output_dir / f"top_degs_by_time_and_{args.clustering_method}_cluster.csv", index=False)
        print(f"Saved per-time DEG results to {output_dir / f'top_degs_by_time_and_{args.clustering_method}_cluster.csv'}")
    else: 
        print("No per-time DEG results generated.")
        
    adata_decoded.write_h5ad(output_dir / f"decoded_adata_with_{args.clustering_method}_clusters.h5ad")
    print(f"Saved AnnData to {output_dir / f'decoded_adata_with_{args.clustering_method}_clusters.h5ad'}")
    print(f"Analysis with {args.clustering_method} clustering finished.")

if __name__ == "__main__":
    args = parse_args()
    sc.settings.figdir = Path(args.output_dir) 
    sc.settings.autoshow = False
    sc.settings.autosave = False
    main(args)
