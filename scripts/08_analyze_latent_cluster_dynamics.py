#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import sys
from tqdm import tqdm

# --- Setup Project Paths and PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze latent space dynamics across clusters and timepoints.")
    parser.add_argument("--experiment1_results_dir", type=str, required=True, help="Directory containing Experiment 1 analysis results.")
    parser.add_argument("--simulation_dir", type=str, required=True, help="Directory containing original simulation outputs.")
    parser.add_argument("--simulation_type", type=str, default="backward", choices=["forward", "backward"])
    parser.add_argument("--config_train_load_path", type=str, required=True, help="Path to training configuration JSON.")
    
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for latent analysis results.")
    parser.add_argument("--figsize_width", type=float, default=12, help="Base figure width.")
    parser.add_argument("--figsize_height", type=float, default=8, help="Base figure height.")
    parser.add_argument("--save_format", type=str, default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def load_latent_cluster_data(experiment1_dir, simulation_dir, simulation_type, config_path):
    """Load and combine latent states with cluster assignments"""
    
    # Load cluster assignments from Experiment 1
    adata_clustered_path = experiment1_dir / "decoded_adata_with_clusters.h5ad"
    if not adata_clustered_path.exists():
        raise FileNotFoundError(f"Clustered data not found: {adata_clustered_path}")
    
    print(f"Loading clustered data from: {adata_clustered_path}")
    adata_clustered = sc.read_h5ad(adata_clustered_path)
    
    # Load original simulation trajectories to get latent states
    sim_dir = Path(simulation_dir)
    times_file = "biological_times_backward.npy" if simulation_type == "backward" else "biological_times.npy"
    traj_file_npz = "state_trajectories_backward.npz" if simulation_type == "backward" else "state_trajectories.npz"
    
    try:
        sim_bio_times = np.load(sim_dir / times_file)
        
        # Load the .npz file containing the dictionary of arrays
        trajectory_npz = np.load(sim_dir / traj_file_npz)
        # Sort keys to ensure correct time order (e.g., 'arr_0', 'arr_1', ..., 'arr_10', ...)
        sorted_keys = sorted(trajectory_npz.keys(), key=lambda k: int(k.split('_')[1]))
        sim_state_traj_list = [trajectory_npz[key] for key in sorted_keys]

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Simulation files not found in {sim_dir}: {e}")
    
    print(f"Loaded simulation trajectories: {len(sim_state_traj_list)} timepoints.")
    # print(f"Loaded simulation trajectories: {sim_state_traj.shape}")
    
    # Load config to get dimensions
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    spatial_dim = config.get('spatial_dim', 2)
    latent_dim = config.get('vae_latent_dim', 8)
    
    print(f"Configuration: spatial_dim={spatial_dim}, latent_dim={latent_dim}")
    
    return adata_clustered, sim_bio_times, sim_state_traj_list, spatial_dim, latent_dim


def extract_latent_cluster_stats(adata_clustered, sim_bio_times, sim_state_traj_list, spatial_dim, latent_dim):
    unique_sampled_times = sorted(adata_clustered.obs['sim_bio_time'].unique())
    print(f"Analyzing {len(unique_sampled_times)} sampled timepoints.")
    
    results_list = []
    
    for sample_idx, bio_time in enumerate(unique_sampled_times):
        adata_timepoint = adata_clustered[adata_clustered.obs['sim_bio_time'] == bio_time].copy()
        if adata_timepoint.n_obs == 0:
            continue

        original_sim_idx = np.argmin(np.abs(sim_bio_times - bio_time))
        state_array_t = sim_state_traj_list[original_sim_idx]
        
        # CORRECTED: Use the correct variable `state_array_t` to slice the latent states
        latent_states = state_array_t[:, spatial_dim:]
        
        if adata_timepoint.n_obs != latent_states.shape[0]:
            print(f"Warning: Cell count mismatch at time {bio_time}. AnnData: {adata_timepoint.n_obs}, Trajectory: {latent_states.shape[0]}. Skipping.")
            continue
            
        print(f"Processing timepoint {bio_time:.3f} (sim index {original_sim_idx}) with {adata_timepoint.n_obs} cells.")
            
        cluster_assignments = adata_timepoint.obs['kmeans_cluster'].values
        unique_clusters = sorted(adata_timepoint.obs['kmeans_cluster'].unique())
        
        for cluster_id in unique_clusters:
            cluster_mask = (cluster_assignments == cluster_id)
            cluster_latent = latent_states[cluster_mask]
            
            if cluster_latent.shape[0] == 0:
                continue
            
            n_cells = cluster_latent.shape[0]
            latent_means = np.mean(cluster_latent, axis=0)
            latent_stds = np.std(cluster_latent, axis=0)
            
            for latent_dim_idx in range(latent_dim):
                results_list.append({
                    'timepoint': bio_time,
                    'cluster': cluster_id,
                    'latent_dim': latent_dim_idx,
                    'latent_mean': latent_means[latent_dim_idx],
                    'latent_std': latent_stds[latent_dim_idx],
                    'n_cells': n_cells
                })
    
    results_df = pd.DataFrame(results_list)
    print(f"Generated {len(results_df)} records")
    return results_df

def create_latent_heatmaps(results_df, output_dir, args):
    """Create heatmap visualizations of latent dimensions by cluster"""
    
    unique_timepoints = sorted(results_df['timepoint'].unique())
    latent_dims = sorted(results_df['latent_dim'].unique())
    
    print(f"Creating heatmaps for {len(unique_timepoints)} timepoints and {len(latent_dims)} latent dimensions")
    
    # Create individual heatmaps for each timepoint
    for timepoint in unique_timepoints:
        timepoint_data = results_df[results_df['timepoint'] == timepoint].copy()
        
        if len(timepoint_data) == 0:
            continue
            
        # Pivot data for heatmap (clusters x latent_dims)
        heatmap_data = timepoint_data.pivot(index='cluster', columns='latent_dim', values='latent_mean')
        
        # IMPORTANT: Ensure cell_counts uses the same cluster order as heatmap_data
        cell_counts_all = timepoint_data.groupby('cluster')['n_cells'].first()
        # Reindex cell_counts to match heatmap_data index order
        cell_counts = cell_counts_all.reindex(heatmap_data.index)
        
        # Create figure with custom layout
        fig, (ax_heatmap, ax_counts) = plt.subplots(1, 2, figsize=(args.figsize_width, args.figsize_height), 
                                                   gridspec_kw={'width_ratios': [4, 1]})
        
        # Main heatmap - use origin='lower' to match y-axis label convention
        im = ax_heatmap.imshow(heatmap_data.values, cmap='RdBu_r', aspect='auto', origin='lower')
        
        # Set ticks and labels
        ax_heatmap.set_xticks(range(len(latent_dims)))
        ax_heatmap.set_xticklabels([f'Z{i}' for i in latent_dims])
        ax_heatmap.set_yticks(range(len(heatmap_data.index)))
        ax_heatmap.set_yticklabels([f'C{cluster}' for cluster in heatmap_data.index])
        
        ax_heatmap.set_xlabel('Latent Dimensions')
        ax_heatmap.set_ylabel('Clusters')
        ax_heatmap.set_title(f'Mean Latent Values at Time {timepoint:.3f}')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_heatmap)
        cbar.set_label('Mean Latent Value')
        
        # Add text annotations with values
        for i in range(len(heatmap_data.index)):
            for j in range(len(latent_dims)):
                value = heatmap_data.iloc[i, j]
                if not np.isnan(value):
                    ax_heatmap.text(j, i, f'{value:.2f}', ha='center', va='center', 
                                  color='white' if abs(value) > 0.5 else 'black', fontsize=8)
        
        # Cell count bars (proportional width visualization)
        # Use the same order as heatmap_data.index
        max_count = cell_counts.max()
        y_positions = range(len(cell_counts))
        
        # Normalize widths to 0-1 range
        normalized_widths = cell_counts.values / max_count
        
        # Create horizontal bars with proportional widths
        # Now both use the same cluster order (heatmap_data.index)
        for i, cluster in enumerate(heatmap_data.index):
            count = cell_counts.iloc[i]
            width = normalized_widths[i]
            ax_counts.barh(i, width, height=0.8, alpha=0.7)
            ax_counts.text(width/2, i, f'{count}', ha='center', va='center', fontweight='bold')
        
        ax_counts.set_ylim(-0.5, len(cell_counts) - 0.5)
        ax_counts.set_xlim(0, 1)
        ax_counts.set_yticks(y_positions)
        # Use the same cluster order as the heatmap
        ax_counts.set_yticklabels([f'C{cluster}' for cluster in heatmap_data.index])
        ax_counts.set_xlabel('Relative Cell Count')
        ax_counts.set_title('Cells per Cluster')
        
        # Add actual cell counts as text - use the same order
        for i, cluster in enumerate(heatmap_data.index):
            count = cell_counts.iloc[i]
            ax_counts.text(1.02, i, f'({count})', ha='left', va='center', transform=ax_counts.get_yaxis_transform())
        
        plt.tight_layout()
        
        # Save figure
        timepoint_safe = str(timepoint).replace('.', 'p')
        output_path = output_dir / f"latent_heatmap_time_{timepoint_safe}.{args.save_format}"
        plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved heatmap: {output_path}")

def create_temporal_dynamics_plot(results_df, output_dir, args):
    """Create plots showing how latent dimensions evolve over time for each cluster"""
    
    unique_clusters = sorted(results_df['cluster'].unique())
    unique_latent_dims = sorted(results_df['latent_dim'].unique())
    
    # Create subplot grid
    n_dims = len(unique_latent_dims)
    n_cols = min(4, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(args.figsize_width, args.figsize_height * n_rows / 2))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    
    for dim_idx, latent_dim in enumerate(unique_latent_dims):
        row = dim_idx // n_cols
        col = dim_idx % n_cols
        ax = axes[row, col]
        
        dim_data = results_df[results_df['latent_dim'] == latent_dim]
        
        for cluster_idx, cluster in enumerate(unique_clusters):
            cluster_data = dim_data[dim_data['cluster'] == cluster].sort_values('timepoint')
            
            if len(cluster_data) > 0:
                ax.plot(cluster_data['timepoint'], cluster_data['latent_mean'], 
                       marker='o', label=f'C{cluster}', color=colors[cluster_idx], linewidth=2)
                
                # Add error bars
                ax.fill_between(cluster_data['timepoint'], 
                              cluster_data['latent_mean'] - cluster_data['latent_std'],
                              cluster_data['latent_mean'] + cluster_data['latent_std'],
                              alpha=0.2, color=colors[cluster_idx])
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean Latent Value')
        ax.set_title(f'Latent Dimension Z{latent_dim}')
        ax.grid(True, alpha=0.3)
        
        if dim_idx == 0:  # Add legend to first subplot
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Hide unused subplots
    for dim_idx in range(n_dims, n_rows * n_cols):
        row = dim_idx // n_cols
        col = dim_idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f"latent_temporal_dynamics.{args.save_format}"
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved temporal dynamics plot: {output_path}")

def create_cluster_size_evolution(results_df, output_dir, args):
    """
    Create a 3-panel plot showing how cluster and total cell sizes evolve over time.
    1. Total cell count evolution (line plot).
    2. Absolute cell counts per cluster (stacked bar plot).
    3. Proportional cell counts per cluster (stacked area plot).
    """
    # Get cluster size data
    cluster_sizes = results_df.groupby(['timepoint', 'cluster'])['n_cells'].first().reset_index()
    pivot_sizes = cluster_sizes.pivot(index='timepoint', columns='cluster', values='n_cells').fillna(0)
    unique_clusters = sorted(cluster_sizes['cluster'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(args.figsize_width, args.figsize_height))
    
    # # Plot 1: Absolute numbers
    # for cluster_idx, cluster in enumerate(unique_clusters):
    #     cluster_data = cluster_sizes[cluster_sizes['cluster'] == cluster].sort_values('timepoint')
    #     ax1.plot(cluster_data['timepoint'], cluster_data['n_cells'], 
    #             marker='o', label=f'C{cluster}', color=colors[cluster_idx], linewidth=2)
    
    # ax1.set_xlabel('Time')
    # ax1.set_ylabel('Number of Cells')
    # ax1.set_title('Cluster Size Evolution (Absolute)')
    # ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # ax1.grid(True, alpha=0.3)
    
    # # Plot 2: Proportional
    # pivot_sizes = cluster_sizes.pivot(index='timepoint', columns='cluster', values='n_cells').fillna(0)
    # pivot_proportions = pivot_sizes.div(pivot_sizes.sum(axis=1), axis=0)
    
    # bottom = np.zeros(len(pivot_proportions))
    # for cluster_idx, cluster in enumerate(unique_clusters):
    #     if cluster in pivot_proportions.columns:
    #         ax2.bar(pivot_proportions.index, pivot_proportions[cluster], 
    #                bottom=bottom, label=f'C{cluster}', color=colors[cluster_idx])
    #         bottom += pivot_proportions[cluster].values
    
    # ax2.set_xlabel('Time')
    # ax2.set_ylabel('Proportion of Cells')
    # ax2.set_title('Cluster Size Evolution (Proportional)')
    # ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # plt.tight_layout()

    # Create a figure with 3 subplots, vertically aligned
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(args.figsize_width, args.figsize_height * 1.5), sharex=True)
    fig.suptitle("Cell Population Dynamics Over Simulated Time", fontsize=16)

    # --- Plot 1: Total Cell Count Evolution ---
    total_cells_per_time = pivot_sizes.sum(axis=1)
    ax1.plot(total_cells_per_time.index, total_cells_per_time.values, 
             marker='o', linestyle='-', color='black', label='Total Cells')
    ax1.set_ylabel("Total Number of Cells")
    ax1.set_title("Total Cell Count Evolution (Merging Effect)")
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.legend()
    
    # --- Plot 2: Absolute Cell Counts per Cluster (Stacked Bar) ---
    pivot_sizes.plot(kind='bar', stacked=True, ax=ax2, color=colors, width=0.8)
    ax2.set_ylabel("Number of Cells (Absolute)")
    ax2.set_title("Cluster Size Evolution (Absolute Counts)")
    ax2.legend().remove() # Remove legend here, will be on the last plot

    # --- Plot 3: Proportional Cell Counts per Cluster (Stacked Area) ---
    pivot_proportions = pivot_sizes.div(total_cells_per_time, axis=0).fillna(0)
    pivot_proportions.plot.area(ax=ax3, stacked=True, color=colors, linewidth=0)
    ax3.set_ylabel("Proportion of Cells (%)")
    ax3.set_title("Cluster Composition Evolution (Proportional)")
    ax3.set_ylim(0, 1) # Proportions must be between 0 and 1
    
    # Format the shared X-axis
    ax3.set_xlabel('Simulated Biological Time')
    # Set x-tick labels to be formatted time values
    tick_labels = [f"{t:.2f}" for t in pivot_proportions.index]
    ax3.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Place a single legend outside the plot area
    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, title="Cluster", loc='center left', bbox_to_anchor=(1.0, 0.5))
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96]) # Adjust layout to make space for legend and suptitle
    
    # Save figure
    output_path = output_dir / f"cluster_size_evolution.{args.save_format}"
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved cluster size evolution plot: {output_path}")

def main(args):
    np.random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting latent cluster dynamics analysis...")
    print(f"Output directory: {output_dir}")
    
    # Load data
    adata_clustered, sim_bio_times, sim_state_traj_list, spatial_dim, latent_dim =  load_latent_cluster_data(
        Path(args.experiment1_results_dir), 
        Path(args.simulation_dir), 
        args.simulation_type,
        args.config_train_load_path
    )
    
    # Extract latent statistics
    print("\nExtracting latent space statistics...")
    # results_df = extract_latent_cluster_stats(adata_clustered, sim_state_traj_list, spatial_dim, latent_dim)
    results_df = extract_latent_cluster_stats(adata_clustered, sim_bio_times, sim_state_traj_list, spatial_dim, latent_dim)
    

    # Save raw data as CSV
    csv_path = output_dir / "latent_cluster_statistics.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved raw data: {csv_path}")
    
    # Create summary statistics
    summary_df = results_df.groupby(['timepoint', 'cluster']).agg({
        'latent_mean': 'mean',
        'latent_std': 'mean', 
        'n_cells': 'first'
    }).reset_index()
    
    summary_csv_path = output_dir / "latent_cluster_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary data: {summary_csv_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Heatmaps for each timepoint
    create_latent_heatmaps(results_df, output_dir, args)
    
    # Temporal dynamics
    create_temporal_dynamics_plot(results_df, output_dir, args)
    
    # Cluster size evolution
    create_cluster_size_evolution(results_df, output_dir, args)
    
    print(f"\nLatent cluster dynamics analysis completed!")
    print(f"Results saved in: {output_dir}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Timepoints analyzed: {len(results_df['timepoint'].unique())}")
    print(f"  Clusters found: {len(results_df['cluster'].unique())}")
    print(f"  Latent dimensions: {len(results_df['latent_dim'].unique())}")
    print(f"  Total records: {len(results_df)}")

if __name__ == "__main__":
    args = parse_args()
    main(args)