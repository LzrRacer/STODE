#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import matplotlib.pyplot as plt
import sys
import traceback
from scipy.interpolate import griddata
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import cKDTree

# --- Setup Project Paths and PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.potential_field import SpatioTemporalPotential

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate potential field stream plots for spatiotemporal simulations.")
    parser.add_argument("--simulation_dir", type=str, required=True, help="Path to the simulation results directory.")
    parser.add_argument("--model_load_path", type=str, required=True, help="Path to the trained system_model_final.pt file.")
    parser.add_argument("--config_train_load_path", type=str, required=True, help="Path to the config_train.json from training.")
    parser.add_argument("--clustered_adata_path", type=str, required=True, help="Path to the AnnData object with cluster assignments.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save analysis plots.")
    parser.add_argument("--timepoints_to_analyze", type=float, nargs='+', required=True, help="List of biological timepoints to plot.")
    parser.add_argument("--background", type=str, default='potential', choices=['potential', 'cluster'], help="Feature to plot as the background.")
    parser.add_argument("--spot_size", type=float, default=1, help="Size of the spots in the background scatter plot.")
    parser.add_argument("--arrow_scale", type=float, default=1.0, help="Scaling factor for the stream plot arrows.")
    parser.add_argument("--grid_step", type=float, default=1.0, help="The size of each grid cell for the stream plot (e.g., 1.0 for one pixel).")
    parser.add_argument("--mask_radius", type=float, default=2.0, help="Radius around cells to show streamlines. Set to 0 to disable masking.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="", help="Device ('cpu', 'cuda', or empty for auto).")
    return parser.parse_args()

def main(args):
    """Main execution function."""
    # --- Setup ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = output_dir

    # --- Load Data, Models, and Configs ---
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

    # --- Dynamically determine the cluster key ---
    cluster_config_path = Path(args.clustered_adata_path).parent / "config_exp1.json"
    if not cluster_config_path.exists():
        cluster_config_path = Path(args.clustered_adata_path).parent / "config.json"

    cluster_key = 'kmeans_cluster'
    if cluster_config_path.exists():
        with open(cluster_config_path, 'r') as f:
            cluster_cfg = json.load(f)
        if cluster_cfg.get("clustering_method") == "annotation":
            if 'kmeans_cluster' not in adata_clustered.obs.columns and 'annotation_cluster' in adata_clustered.obs.columns:
                cluster_key = 'annotation_cluster'
    else:
        if 'kmeans_cluster' not in adata_clustered.obs.columns and 'annotation_cluster' in adata_clustered.obs.columns:
            cluster_key = 'annotation_cluster'
    
    print(f"Using cluster key: '{cluster_key}' for visualization.")
    if cluster_key not in adata_clustered.obs.columns:
        raise KeyError(f"Final cluster key '{cluster_key}' not found in the clustered AnnData.")

    cluster_names = adata_clustered.obs[cluster_key].cat.categories.tolist()
    cluster_colors = adata_clustered.uns.get(f'{cluster_key}_colors', None)
    
    # --- Estimate Global Potential Range for Consistent Colormaps ---
    print("Estimating global potential range via subsampling...")
    all_cells_flat = []
    for i, arr in enumerate(traj_list):
        if arr.shape[0] > 0:
            for row_idx in range(arr.shape[0]):
                all_cells_flat.append({'time_idx': i, 'state': arr[row_idx, :]})
    
    n_sample_potential = min(20000, len(all_cells_flat))
    rng = np.random.default_rng(args.seed)
    sampled_indices = rng.choice(len(all_cells_flat), n_sample_potential, replace=False)
    
    sampled_potentials = []
    with torch.no_grad():
        for idx in sampled_indices:
            cell_info = all_cells_flat[idx]
            time_val = sim_times[cell_info['time_idx']]
            state = cell_info['state']
            s = torch.from_numpy(state[:spatial_dim]).unsqueeze(0).to(device).float()
            z = torch.from_numpy(state[spatial_dim:]).unsqueeze(0).to(device).float()
            t = torch.tensor([time_val], device=device, dtype=torch.float32)
            U = potential_model(s, z, t).cpu().item()
            sampled_potentials.append(U)
            
    potential_vmin = np.percentile(sampled_potentials, 5) if sampled_potentials else 0
    potential_vmax = np.percentile(sampled_potentials, 95) if sampled_potentials else 1
    print(f"  Estimated potential range (5th-95th percentile): [{potential_vmin:.3f}, {potential_vmax:.3f}]")

    # --- Process and Plot Each Timepoint ---
    print(f"\nAnalyzing {len(args.timepoints_to_analyze)} specified timepoints...")
    
    for target_time in args.timepoints_to_analyze:
        try:
            closest_sim_idx = np.argmin(np.abs(sim_times - target_time))
            actual_sim_time = sim_times[closest_sim_idx]
            print(f"\nProcessing target time {target_time:.2f} (closest sim time: {actual_sim_time:.2f})")

            state_t = traj_list[closest_sim_idx]
            if state_t.shape[0] < 2:
                print("  -> Not enough cells to plot for this timepoint.")
                continue
            
            spatial_t = state_t[:, :spatial_dim]
            latent_t = state_t[:, spatial_dim:]
            
            adata_snapshot = sc.AnnData(X=latent_t)
            adata_snapshot.obsm['spatial'] = spatial_t
            
            s_tensor = torch.from_numpy(spatial_t).to(device).float().requires_grad_(True)
            z_tensor = torch.from_numpy(latent_t).to(device).float()
            t_tensor = torch.tensor([actual_sim_time], device=device, dtype=torch.float32)
            
            grad_U_spatial, _ = potential_model.calculate_gradients(s_tensor, z_tensor, t_tensor, create_graph=False)
            grad_U_spatial_np = grad_U_spatial.detach().cpu().numpy()
            adata_snapshot.obsm['potential_velocity'] = -grad_U_spatial_np

            x_min, y_min = spatial_t.min(axis=0)
            x_max, y_max = spatial_t.max(axis=0)
            
            nx = int(np.ceil((x_max - x_min) / args.grid_step))
            ny = int(np.ceil((y_max - y_min) / args.grid_step))
            print(f"  Grid step {args.grid_step} -> Grid density: {nx}x{ny}")

            grid_x = np.linspace(x_min, x_max, nx)
            grid_y = np.linspace(y_min, y_max, ny)
            grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
            grid_points = np.vstack([grid_xx.ravel(), grid_yy.ravel()]).T

            grid_vx = griddata(spatial_t, adata_snapshot.obsm['potential_velocity'][:, 0], grid_points, method='linear', fill_value=0)
            grid_vy = griddata(spatial_t, adata_snapshot.obsm['potential_velocity'][:, 1], grid_points, method='linear', fill_value=0)
            grid_vx = grid_vx.reshape(grid_xx.shape)
            grid_vy = grid_vy.reshape(grid_yy.shape)

            if args.mask_radius > 0:
                print(f"  Masking stream plot to a radius of {args.mask_radius} around cells.")
                tree = cKDTree(spatial_t)
                distances, _ = tree.query(grid_points, k=1)
                distances = distances.reshape(grid_xx.shape)
                mask = distances > args.mask_radius
                grid_vx[mask] = np.nan
                grid_vy[mask] = np.nan

            adata_clustered_t = adata_clustered[adata_clustered.obs['sim_bio_time'] == actual_sim_time]
            if adata_clustered_t.n_obs > 0:
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(adata_clustered_t.obsm['spatial'], adata_clustered_t.obs[cluster_key])
                snapshot_clusters = knn.predict(spatial_t)
                adata_snapshot.obs[cluster_key] = pd.Categorical(snapshot_clusters, categories=cluster_names)
            else:
                adata_snapshot.obs[cluster_key] = 'Unknown'

            # --- Generate Plot ---
            fig, ax = plt.subplots(figsize=(8, 7))
            
            if args.background == 'cluster':
                sc.pl.spatial(adata_snapshot, color=cluster_key, palette=cluster_colors,
                              title=f"Potential Field (∇U) at t={actual_sim_time:.2f}",
                              show=False, spot_size=args.spot_size, ax=ax, legend_loc='right margin')
            elif args.background == 'potential':
                with torch.no_grad():
                    potential_u = potential_model(s_tensor, z_tensor, t_tensor).cpu().numpy().flatten()
                adata_snapshot.obs['potential_U'] = potential_u
                
                sc.pl.spatial(adata_snapshot, color="potential_U", cmap="viridis",
                              title=f"Potential Field (∇U) at t={actual_sim_time:.2f}",
                              vmin=potential_vmin, vmax=potential_vmax,
                              show=False, spot_size=args.spot_size, ax=ax)

            ax.streamplot(grid_x, grid_y, grid_vx, grid_vy, 
                          density=1.5, 
                          color='black', 
                          linewidth=0.5, 
                          arrowsize=args.arrow_scale)

            time_str_safe = f"{actual_sim_time:.2f}".replace('.', 'p')
            plt.savefig(output_dir / f"potential_stream_{args.background}_{time_str_safe}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Stream plot with '{args.background}' background saved to {output_dir}")

        except Exception as e:
            print(f"  ❌ Failed to process timepoint {target_time}: {e}")
            traceback.print_exc()

    print("\nAnalysis complete.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
