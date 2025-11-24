#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from scipy.interpolate import griddata
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import cKDTree
import sys
import traceback
from tqdm import tqdm

# --- Setup Project Paths and PYTHONPATH ---
# Ensure the script can find custom modules in the 'src' directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.potential_field import SpatioTemporalPotential

def parse_args():
    """Parses command-line arguments for generating the animation."""
    parser = argparse.ArgumentParser(description="Generate potential field stream plot animations for spatiotemporal simulations.")
    # --- Input Data and Model Arguments ---
    parser.add_argument("--simulation_dir", type=str, required=True, help="Path to the simulation results directory (containing backward simulation).")
    parser.add_argument("--model_load_path", type=str, required=True, help="Path to the trained system_model_final.pt file.")
    parser.add_argument("--config_train_load_path", type=str, required=True, help="Path to the config_train.json from training.")
    parser.add_argument("--clustered_adata_path", type=str, required=True, help="Path to the AnnData object with cluster assignments.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the animation GIF.")
    
    # --- Animation Control Arguments ---
    parser.add_argument("--animation_fps", type=int, default=5, help="Frames per second for the output GIF.")
    parser.add_argument("--animation_time_step", type=int, default=1, help="Step between simulation timepoints to include in the animation (e.g., 1 for all, 2 for every other).")
    
    # --- Plotting and Visualization Arguments ---
    parser.add_argument("--background", type=str, default='potential', choices=['potential', 'cluster'], help="Feature to plot as the background for each frame.")
    parser.add_argument("--spot_size", type=float, default=1, help="Size of the spots in the background scatter plot.")
    parser.add_argument("--arrow_scale", type=float, default=1.0, help="Scaling factor for the stream plot arrows.")
    parser.add_argument("--grid_step", type=float, default=1.0, help="The size of each grid cell for the stream plot.")
    parser.add_argument("--mask_radius", type=float, default=2.0, help="Radius around cells to show streamlines. Set to 0 to disable masking.")
    
    # --- System and Reproducibility Arguments ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, nargs='?', const="", default="", help="Device to use ('cpu', 'cuda', or empty for auto-detection).")
    return parser.parse_args()

def main(args):
    """Main execution function to generate the potential stream animation."""
    # --- 1. Setup Environment and Paths ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Load Data, Models, and Configurations ---
    try:
        print("Loading data, models, and configurations...")
        with open(args.config_train_load_path, 'r') as f:
            train_cfg = json.load(f)
        checkpoint = torch.load(args.model_load_path, map_location=device)
        sim_dir = Path(args.simulation_dir)
        sim_times = np.load(sim_dir / "biological_times_backward.npy")
        traj_npz = np.load(sim_dir / "state_trajectories_backward.npz")
        traj_list = [traj_npz[k] for k in sorted(traj_npz.keys(), key=lambda k: int(k.split('_')[1]))]
        adata_clustered = sc.read_h5ad(args.clustered_adata_path)
        spatial_dim = train_cfg['spatial_dim']
        latent_dim = train_cfg['vae_latent_dim']
        potential_model = SpatioTemporalPotential(
            spatial_dim=spatial_dim, latent_dim=latent_dim,
            time_embedding_dim=train_cfg['potential_time_embedding_dim'],
            hidden_dims=[int(d) for d in train_cfg['potential_hidden_dims'].split(',')]
        ).to(device).float()
        potential_model.load_state_dict(checkpoint['potential_state_dict'])
        potential_model.train()
    except Exception as e:
        print(f"Error during data loading: {e}"); traceback.print_exc(); sys.exit(1)

    # --- 3. Prepare Cluster Information ---
    cluster_key = 'kmeans_cluster'
    if 'annotation_cluster' in adata_clustered.obs.columns and 'kmeans_cluster' not in adata_clustered.obs.columns:
        cluster_key = 'annotation_cluster'
    print(f"Using cluster key: '{cluster_key}' for visualization.")
    cluster_names = adata_clustered.obs[cluster_key].cat.categories.tolist()
    cluster_colors_list = adata_clustered.uns.get(f'{cluster_key}_colors')
    if cluster_colors_list is None:
        # Generate some default colors if not present
        cmap = plt.get_cmap('tab20', len(cluster_names))
        cluster_colors_list = [plt.matplotlib.colors.to_hex(cmap(i)) for i in range(len(cluster_names))]
    cluster_color_map = dict(zip(cluster_names, cluster_colors_list))

    # --- 4. Estimate Global Ranges for Consistent Plotting ---
    print("Estimating global ranges for consistent plotting...")
    all_cells_flat = np.vstack([arr for arr in traj_list if arr.shape[0] > 0])
    all_times_flat = np.repeat(sim_times, [arr.shape[0] for arr in traj_list])
    
    # Potential Range
    n_sample_potential = min(20000, all_cells_flat.shape[0])
    rng = np.random.default_rng(args.seed)
    sampled_indices = rng.choice(all_cells_flat.shape[0], n_sample_potential, replace=False)
    with torch.no_grad():
        sampled_states = torch.from_numpy(all_cells_flat[sampled_indices]).to(device).float()
        sampled_times = torch.from_numpy(all_times_flat[sampled_indices]).to(device).float()
        sampled_potentials = potential_model(sampled_states[:, :spatial_dim], sampled_states[:, spatial_dim:], sampled_times).cpu().numpy()
    potential_vmin = np.percentile(sampled_potentials, 5) if len(sampled_potentials) > 0 else 0
    potential_vmax = np.percentile(sampled_potentials, 95) if len(sampled_potentials) > 0 else 1
    print(f"  Estimated potential range: [{potential_vmin:.3f}, {potential_vmax:.3f}]")

    # Spatial Range
    all_spatial_coords = all_cells_flat[:, :spatial_dim]
    global_x_min, global_y_min = all_spatial_coords.min(axis=0)
    global_x_max, global_y_max = all_spatial_coords.max(axis=0)
    padding_x = (global_x_max - global_x_min) * 0.05
    padding_y = (global_y_max - global_y_min) * 0.05
    print(f"  Global spatial X-range: [{global_x_min:.2f}, {global_x_max:.2f}]")
    print(f"  Global spatial Y-range: [{global_y_min:.2f}, {global_y_max:.2f}]")

    # --- 5. Setup Animation Figure ---
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle("Potential Field Stream Animation", fontsize=16)
    
    # Setup colormap and colorbar for potential
    if args.background == 'potential':
        norm = Normalize(vmin=potential_vmin, vmax=potential_vmax)
        cmap = plt.get_cmap('viridis')
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(mappable, ax=ax, pad=0.02)
        cbar.set_label('Potential U')
    elif args.background == 'cluster':
        legend_elements = [Patch(facecolor=color, edgecolor=color, label=name) for name, color in cluster_color_map.items()]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Clusters")

    animation_indices = range(0, len(traj_list), args.animation_time_step)
    num_frames = len(animation_indices)
    
    # --- 6. Define Animation Update Function ---
    def update(frame_idx_step):
        sim_idx = animation_indices[frame_idx_step]
        ax.clear()
        ax.set_xlim(global_x_min - padding_x, global_x_max + padding_x)
        ax.set_ylim(global_y_min - padding_y, global_y_max + padding_y)
        ax.set_xlabel("Spatial Dim 1")
        ax.set_ylabel("Spatial Dim 2")
        ax.set_aspect('equal', adjustable='box')

        actual_sim_time = sim_times[sim_idx]
        state_t = traj_list[sim_idx]

        if state_t.shape[0] < 2:
            ax.text(0.5, 0.5, f"t={actual_sim_time:.2f}\nNo cells", ha='center', va='center', transform=ax.transAxes)
            return

        spatial_t = state_t[:, :spatial_dim]
        latent_t = state_t[:, spatial_dim:]
        
        # Plot Background
        if args.background == 'cluster':
            adata_clustered_t = adata_clustered[adata_clustered.obs['sim_bio_time'] == actual_sim_time]
            snapshot_clusters = ['Unknown'] * spatial_t.shape[0]
            if adata_clustered_t.n_obs > 0:
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(adata_clustered_t.obsm['spatial'], adata_clustered_t.obs[cluster_key])
                snapshot_clusters = knn.predict(spatial_t)
            
            colors = [cluster_color_map.get(c, 'grey') for c in snapshot_clusters]
            ax.scatter(spatial_t[:, 0], spatial_t[:, 1], c=colors, s=args.spot_size, alpha=0.8)

        elif args.background == 'potential':
            s_tensor = torch.from_numpy(spatial_t).to(device).float()
            z_tensor = torch.from_numpy(latent_t).to(device).float()
            t_tensor = torch.tensor([actual_sim_time], device=device, dtype=torch.float32)
            with torch.no_grad():
                potential_u = potential_model(s_tensor, z_tensor, t_tensor).cpu().numpy().flatten()
            ax.scatter(spatial_t[:, 0], spatial_t[:, 1], c=potential_u, cmap=cmap, norm=norm, s=args.spot_size, alpha=0.8)

        # Calculate and Plot Streamlines
        s_tensor.requires_grad_(True)
        grad_U_spatial, _ = potential_model.calculate_gradients(s_tensor, z_tensor, t_tensor, create_graph=False)
        potential_velocity = -grad_U_spatial.detach().cpu().numpy()

        x_min_f, y_min_f = spatial_t.min(axis=0)
        x_max_f, y_max_f = spatial_t.max(axis=0)
        nx = int(np.ceil((x_max_f - x_min_f) / args.grid_step))
        ny = int(np.ceil((y_max_f - y_min_f) / args.grid_step))
        grid_x, grid_y = np.linspace(x_min_f, x_max_f, nx), np.linspace(y_min_f, y_max_f, ny)
        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
        grid_points = np.vstack([grid_xx.ravel(), grid_yy.ravel()]).T

        grid_vx = griddata(spatial_t, potential_velocity[:, 0], grid_points, method='linear', fill_value=0).reshape(grid_xx.shape)
        grid_vy = griddata(spatial_t, potential_velocity[:, 1], grid_points, method='linear', fill_value=0).reshape(grid_yy.shape)

        if args.mask_radius > 0:
            tree = cKDTree(spatial_t)
            distances, _ = tree.query(grid_points, k=1)
            mask = distances.reshape(grid_xx.shape) > args.mask_radius
            grid_vx[mask] = np.nan
            grid_vy[mask] = np.nan

        ax.streamplot(grid_x, grid_y, grid_vx, grid_vy, density=1.5, color='black', linewidth=0.5, arrowsize=args.arrow_scale)
        
        # Add titles and text
        ax.set_title(f"Frame {frame_idx_step + 1}/{num_frames}", fontsize=10)
        ax.text(0.02, 0.95, f"Bio Time: {actual_sim_time:.2f}", transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))
        
        # Re-add legend for cluster background
        if args.background == 'cluster':
             ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Clusters")


    # --- 7. Create and Save the Animation ---
    print(f"Creating animation with {num_frames} frames...")
    ani = animation.FuncAnimation(fig, update, frames=tqdm(range(num_frames), desc="Animating frames"), blit=False)
    output_filename = f"potential_stream_animation_{args.background}.gif"
    output_path = output_dir / output_filename
    
    try:
        ani.save(output_path, writer='pillow', fps=args.animation_fps, dpi=120)
        print(f"\n✓ Animation successfully saved to: {output_path}")
    except Exception as e:
        print(f"\n❌ Error saving animation: {e}"); traceback.print_exc()
        
    plt.close(fig)
    print("\nAnalysis complete.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
