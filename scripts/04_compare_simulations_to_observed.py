#!/usr/bin/env python3
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import sys
import traceback
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr

# --- Setup Project Paths and PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# --- Import Custom Modules ---
from models.vae import VAEWrapper
from utils.plotting import compress_to_1d
from losses.custom_losses import sliced_wasserstein_distance # For quantitative comparison

def parse_args_compare():
    parser = argparse.ArgumentParser(description="Compare simulated trajectories (forward or backward) with observed data.")
    parser.add_argument('--simulation_dir', type=str, required=True, help="Path to the simulation results directory.")
    parser.add_argument('--simulation_type', type=str, required=True, choices=['backward', 'forward'], help="Type of simulation ('backward' from observed, 'forward' from progenitor).")
    parser.add_argument('--model_load_path', type=str, required=True, help="Path to the trained_system_model.pt file.")
    parser.add_argument('--config_train_load_path', type=str, required=True, help="Path to config_train.json from training.")
    parser.add_argument('--original_adata_path', type=str, required=True, help="Path to the original full AnnData object.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save comparison results.")
    
    parser.add_argument('--observed_time_points_str', nargs='+', required=True, help="List of observed biological time strings to compare against (e.g., E9.5 E10.5).")
    parser.add_argument('--spatial_cluster_resolution', type=float, default=0.5, help="Leiden resolution for spatial clustering of observed data.")
    parser.add_argument('--n_genes_for_de', type=int, default=20, help="Number of top marker genes per spatial cluster for DE analysis.")
    parser.add_argument('--latent_compress_method', type=str, default='umap', choices=['umap', 'pca'])
    parser.add_argument('--latent_compress_umap_neighbors', type=int, default=10)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default="", help="Device ('cpu', 'cuda', or empty for auto).")
    args = parser.parse_args()
    return args

def time_to_float(t_val):
    if isinstance(t_val, str): return float(t_val.replace('E',''))
    return float(t_val)

def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.device and args.device.lower() in ["cuda", "cpu", "mps"]:
        device = torch.device(args.device.lower())
    elif args.device and "cuda" in args.device.lower():
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda': torch.cuda.manual_seed_all(args.seed)

    # --- Create Output Directory ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_results = [] # To store quantitative metrics

    # --- Load Simulation Data ---
    sim_dir = Path(args.simulation_dir)
    print(f"Loading {args.simulation_type} simulation data from: {sim_dir}")
    try:
        if args.simulation_type == 'backward':
            biological_times_sim = np.load(sim_dir / "biological_times_backward.npy")
            # state_trajectory_bwd: (time, cells, features)
            state_trajectories_sim = np.load(sim_dir / "state_trajectories_backward.npy")
            sim_summary_file = sim_dir / "simulation_summary_backward.json"
        elif args.simulation_type == 'forward':
            biological_times_sim = np.load(sim_dir / "biological_times.npy") # Assumes single array of timepoints
            # state_trajectories_ensemble: (ensemble, time, cells, features) -> take first ensemble
            state_trajectories_ensemble = np.load(sim_dir / "state_trajectories_ensemble.npy")
            if state_trajectories_ensemble.ndim == 4 and state_trajectories_ensemble.shape[0] > 0:
                state_trajectories_sim = state_trajectories_ensemble[0] # Take first ensemble member
            elif state_trajectories_ensemble.ndim == 3: # Already (time, cells, features)
                 state_trajectories_sim = state_trajectories_ensemble
            else:
                raise ValueError(f"Unexpected shape for forward simulation trajectories: {state_trajectories_ensemble.shape}")
            sim_summary_file = sim_dir / "simulation_summary.json"
        else:
            raise ValueError(f"Unknown simulation_type: {args.simulation_type}")

        with open(sim_summary_file, 'r') as f: sim_summary = json.load(f)
        print(f"Simulated trajectory shape: {state_trajectories_sim.shape}, timepoints: {len(biological_times_sim)}")
    except Exception as e:
        print(f"Error loading simulation files: {e}"); traceback.print_exc(); sys.exit(1)

    # --- Load Original AnnData ---
    print(f"Loading original AnnData from: {args.original_adata_path}")
    adata_orig_full = sc.read_h5ad(args.original_adata_path)
    if hasattr(adata_orig_full.X, 'toarray'):
        adata_orig_full.X = adata_orig_full.X.toarray().astype(np.float32)
    else:
        adata_orig_full.X = adata_orig_full.X.astype(np.float32)
    
    # --- Load Model and Configs ---
    print("Loading VAE model and training configurations...")
    with open(args.config_train_load_path, 'r') as f: train_args = json.load(f)
    checkpoint = torch.load(args.model_load_path, map_location=device)

    vae_input_dim = train_args.get('input_dim', None)
    if vae_input_dim is None:
        checkpoint_train_args = checkpoint.get('train_args', checkpoint.get('train_args_snapshot', {}))
        vae_input_dim = checkpoint_train_args.get('input_dim', adata_orig_full.n_vars)
    
    vae_model = VAEWrapper(
        input_dim=vae_input_dim,
        hidden_dims=[int(d) for d in train_args['vae_hidden_dims'].split(',')],
        latent_dim=train_args['vae_latent_dim'],
        dropout_rate=train_args.get('vae_dropout_rate', 0.1)
    ).to(device).float()
    vae_model.load_state_dict(checkpoint['vae_state_dict']); vae_model.eval()

    spatial_dim = train_args['spatial_dim']
    time_key_orig = train_args['time_key']
    spatial_key_orig = train_args['spatial_key']

    # --- Iterate Over Observed Time Points for Comparison ---
    observed_bio_times_numeric = sorted([time_to_float(t) for t in args.observed_time_points_str])
    observed_bio_times_str_sorted = sorted(args.observed_time_points_str, key=time_to_float)

    for obs_time_num, obs_time_str in zip(observed_bio_times_numeric, observed_bio_times_str_sorted):
        print(f"\n--- Comparing with observed time point: {obs_time_str} (Numeric: {obs_time_num:.2f}) ---")
        current_metrics = {'observed_timepoint_str': obs_time_str, 'observed_timepoint_numeric': obs_time_num}

        # 1. Get Observed Data Slice
        adata_obs_t = adata_orig_full[adata_orig_full.obs[time_key_orig].astype(str) == obs_time_str].copy()
        if adata_obs_t.n_obs == 0:
            print(f"No observed cells for time {obs_time_str}. Skipping comparison for this timepoint.")
            comparison_results.append(current_metrics)
            continue
        
        # 2. Get Simulated Data at Closest Time
        closest_sim_idx = np.argmin(np.abs(biological_times_sim - obs_time_num))
        actual_sim_bio_time = biological_times_sim[closest_sim_idx]
        current_metrics['closest_simulated_time'] = actual_sim_bio_time
        print(f"Closest simulated time: {actual_sim_bio_time:.2f}")

        s_sim_t = state_trajectories_sim[closest_sim_idx, :, :spatial_dim]
        z_sim_t = state_trajectories_sim[closest_sim_idx, :, spatial_dim:]
        if s_sim_t.shape[0] == 0:
            print(f"No simulated cells at time {actual_sim_bio_time:.2f}. Skipping comparison.")
            comparison_results.append(current_metrics)
            continue
        
        s_sim_t_tensor = torch.tensor(s_sim_t, dtype=torch.float32).to(device)
        z_sim_t_tensor = torch.tensor(z_sim_t, dtype=torch.float32).to(device)

        # 3. Decode Simulated Latent States to Expression
        decoded_expr_list = []
        with torch.no_grad():
            bs = 512
            for i_batch in range(0, z_sim_t_tensor.shape[0], bs):
                batch_z = z_sim_t_tensor[i_batch:i_batch+bs]
                mean_recon_batch = vae_model.decode(batch_z)
                decoded_expr_list.append(mean_recon_batch.cpu().numpy())
        expr_sim_decoded_t = np.concatenate(decoded_expr_list, axis=0)

        # 4. Quantitative Spatial Comparison
        s_obs_t = torch.tensor(adata_obs_t.obsm[spatial_key_orig], dtype=torch.float32).to(device)
        swd_spatial = sliced_wasserstein_distance(s_sim_t_tensor, s_obs_t, n_projections=100).item()
        current_metrics['swd_spatial'] = swd_spatial
        print(f"  Spatial SWD (Sim vs Obs): {swd_spatial:.4f}")

        # 5. Quantitative Expression Comparison (Overall)
        expr_obs_t_tensor = torch.tensor(adata_obs_t.X, dtype=torch.float32).to(device)
        expr_sim_decoded_t_tensor = torch.tensor(expr_sim_decoded_t, dtype=torch.float32).to(device)
        swd_expression = sliced_wasserstein_distance(expr_sim_decoded_t_tensor, expr_obs_t_tensor, n_projections=100).item()
        current_metrics['swd_expression_overall'] = swd_expression
        print(f"  Overall Expression SWD (Sim Decoded vs Obs): {swd_expression:.4f}")
        
        mean_expr_obs = np.mean(adata_obs_t.X, axis=0)
        mean_expr_sim = np.mean(expr_sim_decoded_t, axis=0)
        corr_mean_expr, _ = pearsonr(mean_expr_obs, mean_expr_sim)
        current_metrics['pearson_mean_expression'] = corr_mean_expr
        print(f"  Pearson Correlation of Mean Gene Expression: {corr_mean_expr:.4f}")

        # 6. Spatial Clustering and Per-Cluster Comparison
        print("  Performing spatial clustering on observed data...")
        try:
            sc.pp.neighbors(adata_obs_t, use_rep='spatial', n_neighbors=15, key_added='spatial', random_state=args.seed) # use_rep expects key in obsm
            sc.tl.leiden(adata_obs_t, resolution=args.spatial_cluster_resolution, key_added='spatial_cluster', neighbors_key='spatial', random_state=args.seed)
            
            # Assign simulated cells to observed spatial clusters
            knn_classifier = KNeighborsClassifier(n_neighbors=1)
            knn_classifier.fit(adata_obs_t.obsm[spatial_key_orig], adata_obs_t.obs['spatial_cluster'])
            sim_spatial_clusters = knn_classifier.predict(s_sim_t)
            
            unique_obs_clusters = sorted(adata_obs_t.obs['spatial_cluster'].unique())
            cluster_comparison_metrics = []

            # DE for observed spatial clusters
            if len(unique_obs_clusters) > 1 :
                 sc.tl.rank_genes_groups(adata_obs_t, 'spatial_cluster', method='wilcoxon', key_added='rank_genes_spatial_clusters', n_genes=args.n_genes_for_de)
            
            for cluster_id in unique_obs_clusters:
                cluster_metrics = {'cluster_id': cluster_id}
                
                # Observed cells in this cluster
                obs_cluster_mask = adata_obs_t.obs['spatial_cluster'] == cluster_id
                adata_obs_cluster = adata_obs_t[obs_cluster_mask]
                
                # Simulated cells assigned to this cluster
                sim_cluster_mask = sim_spatial_clusters == cluster_id
                expr_sim_cluster = expr_sim_decoded_t[sim_cluster_mask]
                s_sim_cluster = s_sim_t[sim_cluster_mask]
                z_sim_cluster = z_sim_t[sim_cluster_mask]

                if adata_obs_cluster.n_obs > 0 and expr_sim_cluster.shape[0] > 0:
                    # Compare mean expression of all genes
                    mean_expr_obs_cl = np.mean(adata_obs_cluster.X, axis=0)
                    mean_expr_sim_cl = np.mean(expr_sim_cluster, axis=0)
                    corr_mean_expr_cl, _ = pearsonr(mean_expr_obs_cl, mean_expr_sim_cl)
                    cluster_metrics['pearson_mean_expr_cluster'] = corr_mean_expr_cl
                    
                    # Compare 1D VAE latent compression
                    # Observed
                    mu_obs_cl, _ = vae_model.vae.encoder(torch.tensor(adata_obs_cluster.X, dtype=torch.float32).to(device))
                    latent_1d_obs_cl = compress_to_1d(mu_obs_cl.detach().cpu().numpy(), method=args.latent_compress_method, n_neighbors=args.latent_compress_umap_neighbors, random_state=args.seed)
                    cluster_metrics['mean_latent_1d_obs_cluster'] = np.mean(latent_1d_obs_cl)
                    cluster_metrics['var_latent_1d_obs_cluster'] = np.var(latent_1d_obs_cl)
                    
                    # Simulated
                    latent_1d_sim_cl = compress_to_1d(z_sim_cluster, method=args.latent_compress_method, n_neighbors=args.latent_compress_umap_neighbors, random_state=args.seed)
                    cluster_metrics['mean_latent_1d_sim_cluster'] = np.mean(latent_1d_sim_cl)
                    cluster_metrics['var_latent_1d_sim_cluster'] = np.var(latent_1d_sim_cl)

                    # If DE was run, compare marker gene expression
                    if len(unique_obs_clusters) > 1 and 'rank_genes_spatial_clusters' in adata_obs_t.uns:
                        try:
                            marker_genes_df = pd.DataFrame(adata_obs_t.uns['rank_genes_spatial_clusters']['names'])[cluster_id]
                            marker_gene_indices = [adata_obs_t.var_names.get_loc(g) for g in marker_genes_df if g in adata_obs_t.var_names]
                            
                            if marker_gene_indices:
                                mean_marker_expr_obs_cl = np.mean(adata_obs_cluster.X[:, marker_gene_indices], axis=0)
                                mean_marker_expr_sim_cl = np.mean(expr_sim_cluster[:, marker_gene_indices], axis=0)
                                if len(mean_marker_expr_obs_cl) >1 : # Pearson needs at least 2 points
                                     corr_marker_expr_cl, _ = pearsonr(mean_marker_expr_obs_cl, mean_marker_expr_sim_cl)
                                     cluster_metrics['pearson_marker_expr_cluster'] = corr_marker_expr_cl
                        except KeyError: # Cluster_id might not be in DE results if it's too small etc.
                            print(f"    Warning: Cluster {cluster_id} not found in DE results or no marker genes.")
                        except Exception as e_de_access:
                            print(f"    Error accessing DE genes for cluster {cluster_id}: {e_de_access}")


                cluster_comparison_metrics.append(cluster_metrics)
                print(f"    Cluster {cluster_id}: Obs cells={adata_obs_cluster.n_obs}, Sim cells={expr_sim_cluster.shape[0]}. Mean Expr Corr: {cluster_metrics.get('pearson_mean_expr_cluster', 'N/A'):.3f}")

            current_metrics['cluster_comparisons'] = cluster_comparison_metrics
            
            # Save spatial cluster plot for observed data
            fig_spatial_cluster, ax_sc = plt.subplots()
            sc.pl.spatial(adata_obs_t, color='spatial_cluster', spot_size=30, ax=ax_sc, show=False, title=f"Observed Spatial Clusters - {obs_time_str}")
            fig_spatial_cluster.savefig(output_dir / f"observed_spatial_clusters_{obs_time_str.replace('.', 'p')}.png", dpi=150)
            plt.close(fig_spatial_cluster)

        except Exception as e_spatial_cluster:
            print(f"  Error during spatial clustering or per-cluster analysis: {e_spatial_cluster}")
            traceback.print_exc()
        
        comparison_results.append(current_metrics)

    # --- Save All Quantitative Results ---
    results_df = pd.DataFrame(comparison_results)
    results_df.to_csv(output_dir / "quantitative_comparison_summary.csv", index=False)
    with open(output_dir / "quantitative_comparison_summary.json", 'w') as f_json:
        json.dump(comparison_results, f_json, indent=2, cls=NpEncoder) # Use custom encoder for numpy types

    print(f"\nQuantitative comparison complete. Results saved to {output_dir.resolve()}")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__ == '__main__':
    args = parse_args_compare()
    main(args)