#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr #, wasserstein_distance # wasserstein_distance from scipy.stats is 1D only
from sklearn.cluster import KMeans # For basic clustering comparison
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import traceback

# --- Setup Project Paths and PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.vae import VAEWrapper # For loading VAE for potential decoding/latent analysis if needed
from utils.plotting import compress_to_1d # For UMAPs
from losses.custom_losses import sliced_wasserstein_distance # Assuming you have this


def parse_args_compare_slice():
    parser = argparse.ArgumentParser(description="Compare a simulated/predicted time slice with observed data.")
    parser.add_argument("--simulated_adata_path", type=str, required=True, help="Path to AnnData file of the simulated/predicted time slice (gene expression space).")
    parser.add_argument("--observed_adata_path", type=str, required=True, help="Path to AnnData file of the true observed time slice.")
    parser.add_argument("--config_train_load_path", type=str, help="Path to the training configuration JSON for the model used for simulation (for context/params).")
    parser.add_argument("--original_adata_for_genes_path", type=str, required=True, help="Path to an AnnData with original full gene list (e.g. training data before exclusion) to ensure consistent var_names.")

    parser.add_argument("--target_timepoint_str", type=str, required=True, help="String representation of the target timepoint (e.g., 'E10.5') for titling.")
    
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save comparison results and plots.")
    parser.add_argument("--cluster_resolution", type=float, default=0.5, help="Resolution for Leiden clustering.")
    parser.add_argument("--n_top_degs_plot", type=int, default=5, help="Number of top DEGs to show in comparison plots (if DEG analysis is implemented).")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="", help="Device to use ('cuda', 'cpu', or '').")
    return parser.parse_args()


def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.device and args.device.lower() in ["cuda", "cpu", "mps"]:
        device = torch.device(args.device.lower())
    elif args.device and "cuda" in args.device.lower():
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fix f-string syntax issue
    timepoint_safe = args.target_timepoint_str.replace('.', '_')
    
    # Save args
    with open(output_dir / f"compare_args_{timepoint_safe}.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"Loading simulated AnnData for {args.target_timepoint_str} from: {args.simulated_adata_path}")
    adata_sim = sc.read_h5ad(args.simulated_adata_path)
    print(f"Loading observed AnnData for {args.target_timepoint_str} from: {args.observed_adata_path}")
    adata_obs = sc.read_h5ad(args.observed_adata_path)
    
    adata_orig_genes = sc.read_h5ad(args.original_adata_for_genes_path)
    original_var_names = adata_orig_genes.var_names.copy()
    del adata_orig_genes

    # Ensure var_names are consistent and in the same order
    print("Aligning var_names between observed and simulated AnnDatas...")
    adata_obs = adata_obs[:, adata_obs.var_names.isin(original_var_names)].copy()
    adata_obs = adata_obs[:, original_var_names[original_var_names.isin(adata_obs.var_names)]].copy()
    
    adata_sim = adata_sim[:, adata_sim.var_names.isin(original_var_names)].copy()
    adata_sim = adata_sim[:, original_var_names[original_var_names.isin(adata_sim.var_names)]].copy()
    
    common_genes = adata_obs.var_names.intersection(adata_sim.var_names)
    if len(common_genes) < min(adata_obs.n_vars, adata_sim.n_vars):
        print(f"Warning: After alignment, common genes {len(common_genes)} is less than original. Check gene lists.")
    adata_obs = adata_obs[:, common_genes].copy()
    adata_sim = adata_sim[:, common_genes].copy()
    
    print(f"  Observed data shape after gene alignment: {adata_obs.shape}")
    print(f"  Simulated data shape after gene alignment: {adata_sim.shape}")
    if adata_obs.n_vars == 0:
        print("Error: No common genes found. Cannot proceed with comparison.")
        return

    results = {"timepoint": args.target_timepoint_str}

    # 0. Cell numbers
    results["n_cells_observed"] = adata_obs.n_obs
    results["n_cells_simulated"] = adata_sim.n_obs
    print(f"Observed cells: {adata_obs.n_obs}, Simulated cells: {adata_sim.n_obs}")

    # 1. Spatial Distribution Comparison (if spatial coords exist)
    if 'spatial' in adata_obs.obsm and 'spatial' in adata_sim.obsm:
        obs_spatial = adata_obs.obsm['spatial'].astype(np.float32)
        sim_spatial = adata_sim.obsm['spatial'].astype(np.float32)
        
        if obs_spatial.shape[0] > 0 and sim_spatial.shape[0] > 0:
            results["spatial_swd"] = sliced_wasserstein_distance(
                torch.from_numpy(obs_spatial).to(device), 
                torch.from_numpy(sim_spatial).to(device), 
                n_projections=100, p=1, device=device).item()
            print(f"  Spatial SWD (Sim vs Obs): {results['spatial_swd']:.4f}")

            fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
            axs[0].scatter(obs_spatial[:, 0], obs_spatial[:, 1], s=5, alpha=0.7, label="Observed")
            axs[0].set_title(f"Observed Spatial ({args.target_timepoint_str})")
            axs[0].set_xlabel("Spatial X"); axs[0].set_ylabel("Spatial Y"); axs[0].set_aspect('equal')
            axs[1].scatter(sim_spatial[:, 0], sim_spatial[:, 1], s=5, alpha=0.7, label="Simulated", color='orangered')
            axs[1].set_title(f"Simulated Spatial ({args.target_timepoint_str})")
            axs[1].set_xlabel("Spatial X"); axs[1].set_ylabel("Spatial Y"); axs[1].set_aspect('equal')
            plt.suptitle(f"Spatial Distribution Comparison - {args.target_timepoint_str}", fontsize=14)
            plt.tight_layout(rect=[0,0,1,0.96])
            plt.savefig(output_dir / f"spatial_comparison_{timepoint_safe}.png", dpi=150)
            plt.close(fig)
        else:
            print("  Skipping spatial comparison due to empty spatial coordinates in one of the datasets.")
            results["spatial_swd"] = np.nan
    else:
        print("  'spatial' key not found in obsm for one or both AnnDatas. Skipping spatial comparison.")
        results["spatial_swd"] = np.nan


    # 2. Overall Gene Expression Comparison
    obs_X = adata_obs.X.toarray() if hasattr(adata_obs.X, "toarray") else np.asarray(adata_obs.X)
    sim_X = adata_sim.X.toarray() if hasattr(adata_sim.X, "toarray") else np.asarray(adata_sim.X)
    
    if obs_X.shape[0] > 0 and sim_X.shape[0] > 0:
        results["expr_swd_overall"] = sliced_wasserstein_distance(
            torch.from_numpy(obs_X.astype(np.float32)).to(device),
            torch.from_numpy(sim_X.astype(np.float32)).to(device),
            n_projections=100, p=1, device=device).item()
        
        mean_obs_expr = np.mean(obs_X, axis=0)
        mean_sim_expr = np.mean(sim_X, axis=0)
        results["mean_expr_pearson_r"], results["mean_expr_pearson_p"] = pearsonr(mean_obs_expr, mean_sim_expr)
        print(f"  Overall Expression SWD (Sim vs Obs): {results['expr_swd_overall']:.4f}")
        print(f"  Pearson Correlation of Mean Gene Expression: {results['mean_expr_pearson_r']:.4f}")

        fig, ax = plt.subplots(figsize=(6,5))
        ax.scatter(mean_obs_expr, mean_sim_expr, alpha=0.5, s=10, edgecolors='k', linewidths=0.5)
        ax.set_xlabel("Mean Gene Expression (Observed)")
        ax.set_ylabel("Mean Gene Expression (Simulated)")
        ax.set_title(f"Mean Gene Expr Corr. ({args.target_timepoint_str})\nR={results['mean_expr_pearson_r']:.3f}")
        min_val = min(mean_obs_expr.min(), mean_sim_expr.min())
        max_val = max(mean_obs_expr.max(), mean_sim_expr.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1) # Diagonal line
        plt.tight_layout()
        plt.savefig(output_dir / f"mean_expr_correlation_{timepoint_safe}.png", dpi=150)
        plt.close(fig)
    else:
        print("  Skipping gene expression comparison due to empty expression data.")
        results["expr_swd_overall"] = np.nan
        results["mean_expr_pearson_r"], results["mean_expr_pearson_p"] = np.nan, np.nan


    # 3. Clustering and Visual Comparison
    # Basic preprocessing for visualization & clustering
    def preprocess_for_viz(adata_in, rep_name):
        adata_proc = adata_in.copy()
        
        print(f"  Preprocessing {rep_name}: {adata_proc.shape}")
        
        # Check for and handle NaN/infinite values
        if hasattr(adata_proc.X, 'toarray'):
            X_data = adata_proc.X.toarray()
        else:
            X_data = adata_proc.X
            
        nan_count = np.isnan(X_data).sum()
        inf_count = np.isinf(X_data).sum()
        
        if nan_count > 0 or inf_count > 0:
            print(f"    Found {nan_count} NaN and {inf_count} infinite values in {rep_name}, replacing with 0")
            X_data = np.nan_to_num(X_data, nan=0.0, posinf=0.0, neginf=0.0)
            adata_proc.X = X_data
        
        # Check for constant genes (all zeros or all same value)
        gene_var = np.var(X_data, axis=0)
        constant_genes = gene_var == 0
        n_constant = constant_genes.sum()
        
        if n_constant > 0:
            print(f"    Found {n_constant} constant genes in {rep_name}, removing them")
            adata_proc = adata_proc[:, ~constant_genes].copy()
            if adata_proc.n_vars == 0:
                print(f"    Error: All genes are constant in {rep_name}")
                return adata_proc
        
        sc.pp.normalize_total(adata_proc, target_sum=1e4)
        sc.pp.log1p(adata_proc)
        
        # Use cell_ranger flavor instead of seurat_v3 to avoid scikit-misc dependency
        try:
            if adata_proc.n_vars > 1:
                sc.pp.highly_variable_genes(adata_proc, n_top_genes=min(2000, adata_proc.n_vars -1), flavor="cell_ranger", subset=False)
            else:
                adata_proc.var['highly_variable'] = True
        except Exception as e:
            print(f"    Warning: HVG selection failed for {rep_name}: {e}")
            print(f"    Using all genes for {rep_name}")
            adata_proc.var['highly_variable'] = True
            
        if 'highly_variable' in adata_proc.var.columns:
             adata_proc = adata_proc[:, adata_proc.var.highly_variable].copy()
        
        if adata_proc.n_vars == 0: # Fallback if no HVGs found
            print(f"    Warning: No HVGs found for {rep_name}. Using all genes for PCA/UMAP.")
            adata_proc = adata_in.copy()
            sc.pp.normalize_total(adata_proc, target_sum=1e4); sc.pp.log1p(adata_proc)
        
        if adata_proc.n_obs > 1 and adata_proc.n_vars > 1:
            try:
                # Double-check for NaN after preprocessing
                if hasattr(adata_proc.X, 'toarray'):
                    X_check = adata_proc.X.toarray()
                else:
                    X_check = adata_proc.X
                    
                if np.isnan(X_check).any() or np.isinf(X_check).any():
                    print(f"    Cleaning NaN/inf values after preprocessing for {rep_name}")
                    X_check = np.nan_to_num(X_check, nan=0.0, posinf=0.0, neginf=0.0)
                    adata_proc.X = X_check
                
                sc.pp.pca(adata_proc, n_comps=min(50, adata_proc.n_obs-1, adata_proc.n_vars-1), svd_solver='arpack' if min(adata_proc.shape) > 50 else 'auto')
                sc.pp.neighbors(adata_proc, n_neighbors=min(15, adata_proc.n_obs-1), n_pcs=min(30, adata_proc.n_obs-1, adata_proc.obsm['X_pca'].shape[1]-1 if 'X_pca' in adata_proc.obsm else 0))
                sc.tl.leiden(adata_proc, resolution=args.cluster_resolution, key_added=f'cluster_{rep_name}', random_state=args.seed)
                if adata_proc.n_obs > 2: # UMAP requires n_obs > 2 generally
                     sc.tl.umap(adata_proc, random_state=args.seed, min_dist=0.3)
                else: 
                    print(f"    Skipping UMAP for {rep_name} due to too few cells ({adata_proc.n_obs}).")
            except Exception as e:
                print(f"    Error in dimensionality reduction for {rep_name}: {e}")
                print(f"    Skipping PCA/UMAP for {rep_name}")
        else:
            print(f"    Skipping PCA/Neighbors/Leiden/UMAP for {rep_name} due to insufficient cells/genes ({adata_proc.n_obs} cells, {adata_proc.n_vars} genes).")
        
        print(f"  Finished preprocessing {rep_name}: {adata_proc.shape}")
        return adata_proc

    if adata_obs.n_obs > 5: # Need some cells to do this meaningfully
        adata_obs_processed = preprocess_for_viz(adata_obs, "obs")
        if 'X_umap' in adata_obs_processed.obsm:
            sc.pl.umap(adata_obs_processed, color='cluster_obs', title=f'Observed Clusters ({args.target_timepoint_str})', 
                       show=False, save=f"_obs_clusters_{timepoint_safe}.png", legend_loc='on data',
                       frameon=False)
            plt.close()
    else:
        print("Too few observed cells to process for UMAP/clustering.")
        adata_obs_processed = adata_obs.copy()
        adata_obs_processed.obs['cluster_obs'] = '0'


    if adata_sim.n_obs > 5:
        adata_sim_processed = preprocess_for_viz(adata_sim, "sim")
        if 'X_umap' in adata_sim_processed.obsm:
            sc.pl.umap(adata_sim_processed, color='cluster_sim', title=f'Simulated Clusters ({args.target_timepoint_str})', 
                       show=False, save=f"_sim_clusters_{timepoint_safe}.png", legend_loc='on data',
                       frameon=False)
            plt.close()
    else:
        print("Too few simulated cells to process for UMAP/clustering.")
        adata_sim_processed = adata_sim.copy()
        adata_sim_processed.obs['cluster_sim'] = '0'

    # Compare cluster proportions
    obs_cluster_counts = adata_obs_processed.obs['cluster_obs'].value_counts(normalize=True).sort_index()
    sim_cluster_counts = adata_sim_processed.obs['cluster_sim'].value_counts(normalize=True).sort_index()
    
    df_proportions = pd.DataFrame({'Observed': obs_cluster_counts, 'Simulated': sim_cluster_counts}).fillna(0)
    fig_prop, ax_prop = plt.subplots(figsize=(max(6, 0.5 * len(df_proportions)), 5)) # Adjust width
    df_proportions.plot(kind='bar', ax=ax_prop)
    ax_prop.set_title(f'Cluster Proportions ({args.target_timepoint_str})')
    ax_prop.set_ylabel('Proportion of Cells')
    ax_prop.set_xlabel('Cluster ID')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / f"cluster_proportions_{timepoint_safe}.png", dpi=150)
    plt.close(fig_prop)
    results["cluster_proportions_comparison"] = df_proportions.to_dict()
    print(f"  Cluster proportions plot saved.")

    # Save final results
    results_df = pd.DataFrame([results])
    results_df.to_csv(output_dir / f"comparison_summary_{timepoint_safe}.csv", index=False)
    print(f"Comparison summary saved to {output_dir / f'comparison_summary_{timepoint_safe}.csv'}")
    
    adata_obs_processed.write_h5ad(output_dir / f"adata_observed_processed_{timepoint_safe}.h5ad")
    adata_sim_processed.write_h5ad(output_dir / f"adata_simulated_processed_{timepoint_safe}.h5ad")

    print(f"Comparison for {args.target_timepoint_str} finished. Plots and data saved in {output_dir.resolve()}")





    # --- START: New section for Presentation-Ready DEG Heatmap ---
    print("\nGenerating presentation-ready DEG heatmap for top clusters...")

    # Define the key for the clustering results
    cluster_key = 'cluster_obs'
    adata_to_plot = adata_obs_processed

    # Check if clustering has been performed
    if cluster_key in adata_to_plot.obs.columns:
        
        # 1. Identify the top 10 most frequent clusters (cell types)
        top_clusters = adata_to_plot.obs[cluster_key].value_counts().nlargest(10).index.tolist()
        print(f"  Identified top 10 clusters: {top_clusters}")

        # Filter the AnnData object to only include cells from these top clusters
        adata_top_clusters = adata_to_plot[adata_to_plot.obs[cluster_key].isin(top_clusters)].copy()
        
        # Ensure the cluster categories are updated
        adata_top_clusters.obs[cluster_key] = adata_top_clusters.obs[cluster_key].cat.remove_unused_categories()

        # 2. Find marker genes for these clusters.
        # Use the 'n_top_degs_plot' argument from your script's parser.
        # The default is 5, which matches your request.
        print(f"  Finding top {args.n_top_degs_plot} marker genes for each cluster...")
        sc.tl.rank_genes_groups(
            adata_top_clusters,
            groupby=cluster_key,
            method='wilcoxon', # Standard method for DEG
            use_raw=False, # Use the processed, normalized data
            groups=top_clusters, # Focus analysis on the top clusters
            n_genes=args.n_top_degs_plot # Number of genes to find per cluster
        )

        # 3. Create and save the focused matrix plot
        print("  Generating and saving the matrix plot...")
        fig_matrix, ax_matrix = plt.subplots(figsize=(12, 8)) # Adjust size as needed
        
        sc.pl.matrixplot(
            adata_top_clusters,
            var_names=sc.get.rank_genes_groups_df(adata_top_clusters, group=None, pval_cutoff=1)['names'].unique(),
            groupby=cluster_key,
            dendrogram=True,
            cmap='viridis', # A visually distinct colormap
            standard_scale='var',
            show=False,
            ax=ax_matrix
        )
        
        # Improve layout for presentation
        fig_matrix.tight_layout()

        # Define a clear filename for the new plot
        presentation_plot_path = output_dir / f"presentation_top10_clusters_matrixplot_{timepoint_safe}.png"
        
        plt.savefig(presentation_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig_matrix)
        
        print(f"  Presentation heatmap saved to: {presentation_plot_path}")

    else:
        print("  Skipping DEG heatmap generation because clustering key was not found.")

    # --- END: New section for Presentation-Ready DEG Heatmap ---


if __name__ == "__main__":
    args = parse_args_compare_slice()
    
    sc.settings.figdir = Path(args.output_dir) / "scanpy_plots" # Scanpy saves figures here
    sc.settings.figdir.mkdir(parents=True, exist_ok=True)
    sc.settings.autoshow = False
    sc.settings.autosave = True # Autosave scanpy plots

    main(args)