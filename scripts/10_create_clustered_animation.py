#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import sys
import traceback
from PIL import Image, ImageSequence

# --- Setup Project Paths and PYTHONPATH ---
# This block allows the script to be run from anywhere and still find the 'src' module.
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SRC_DIR = PROJECT_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from models.vae import VAEWrapper
except (IndexError, ImportError):
    print("Warning: Could not import VAEWrapper. Assuming script is not in project structure.")
    VAEWrapper = None


def parse_args():
    """Parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Create or reverse a GIF animation of simulated cell trajectories.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- Group for the primary function: Creating animations from data ---
    anim_group = parser.add_argument_group('Animation Creation Mode (Default)')
    anim_group.add_argument('--simulation_dir', type=str, help="Path to the directory with simulation results.")
    anim_group.add_argument('--model_load_path', type=str, help="Path to the trained VAE model file.")
    anim_group.add_argument('--config_train_load_path', type=str, help="Path to the config_train.json from training.")
    anim_group.add_argument('--original_adata_path', type=str, help="Path to original AnnData for annotations.")
    anim_group.add_argument('--output_dir', type=str, help="Directory to save the output animation.")
    anim_group.add_argument("--clustering_method", type=str, default="kmeans", choices=["kmeans", "annotation"], help="Method for clustering.")
    anim_group.add_argument("--annotation_key", type=str, default="annotation", help="Key in original_adata.obs for annotation-based clustering.")
    anim_group.add_argument("--num_clusters", type=int, default=8, help="Number of K-Means clusters.")
    anim_group.add_argument('--animation_frames', type=int, default=0, help="Max number of frames for animation (0 for all).")
    anim_group.add_argument('--animation_fps', type=int, default=15, help="Frames per second for the animation.")
    anim_group.add_argument('--animation_dot_size', type=int, default=15, help="Size of the dots in the animation.")
    anim_group.add_argument("--animation_end_time_bio", type=float, default=None, help="Biological time to end the animation at.")
    anim_group.add_argument('--create_reverse', action='store_true', help="After creating the animation, also save a reversed version.")

    # --- Group for the secondary function: Reversing an existing GIF ---
    reverse_group = parser.add_argument_group('Reverse-Only Mode')
    reverse_group.add_argument('--reverse_only_input_gif', type=str, metavar='PATH_TO_GIF',
                               help="Path to an existing GIF to reverse. This runs in a separate mode, ignoring all other flags.")

    # General script parameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default="", help="Device for PyTorch ('cpu', 'cuda').")
    
    return parser.parse_args()


def create_reverse_gif(input_path, output_path):
    """Reads an existing GIF and saves a reversed version."""
    print(f"\nReversing animation file: {input_path.name}...")
    try:
        with Image.open(input_path) as img:
            frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
            frames.reverse()
            duration = img.info.get('duration', 100)
            loop = img.info.get('loop', 0)
            frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration, loop=loop)
        print(f"✓ Reversed animation saved to {output_path}")
    except FileNotFoundError:
        print(f"✗ Error: Input GIF for reversal not found at {input_path}")
    except Exception as e:
        print(f"✗ Error reversing GIF: {e}")
        traceback.print_exc()

def get_cluster_centroids(latent_states, method, num_clusters, original_adata, annotation_key, vae_model, device):
    """Calculates cluster centroids based on the chosen method."""
    if method == 'kmeans':
        print(f"Calculating {num_clusters} K-Means centroids...")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto').fit(latent_states)
        return kmeans.cluster_centers_, [f'Cluster {i}' for i in range(num_clusters)]
    elif method == 'annotation':
        print(f"Calculating centroids from observed annotations using key '{annotation_key}'...")
        if annotation_key not in original_adata.obs: raise ValueError(f"Annotation key '{annotation_key}' not found.")
        with torch.no_grad():
            expr_tensor = torch.tensor(original_adata.X, dtype=torch.float32).to(device)
            original_latent_mu, _ = vae_model.vae.encoder(expr_tensor)
        original_adata.obsm['X_latent_mu'] = original_latent_mu.cpu().numpy()
        df = pd.DataFrame(original_adata.obsm['X_latent_mu'], index=original_adata.obs_names)
        df['annotation'] = original_adata.obs[annotation_key]
        centroids_series = df.groupby('annotation').mean()
        return centroids_series.values, centroids_series.index.tolist()

def assign_to_clusters(latent_states, centroids):
    """Assigns each latent state to the nearest centroid."""
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(centroids, np.arange(len(centroids)))
    return knn.predict(latent_states)

def create_clustered_animation(spatial_trajectory, cluster_labels_trajectory, biological_times, cluster_names, output_path, fig_title, fps, dot_size):
    """Creates and saves a GIF animation with cells colored by cluster labels."""
    print(f"Creating animation: {output_path.name}...")
    n_frames, _, _ = spatial_trajectory.shape
    n_clusters = len(cluster_names)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlabel("Spatial Dimension 1"); ax.set_ylabel("Spatial Dimension 2")
    ax.set_aspect('equal', adjustable='box')
    valid_data = spatial_trajectory[~np.isnan(spatial_trajectory[:, :, 0])]
    if valid_data.size == 0:
        print("Warning: No valid data to plot."); plt.close(fig); return
    min_x, min_y = np.min(valid_data, axis=0); max_x, max_y = np.max(valid_data, axis=0)
    padding = (max(max_x - min_x, max_y - min_y)) * 0.1
    ax.set_xlim(min_x - padding, max_x + padding); ax.set_ylim(min_y - padding, max_y + padding)
    cmap = plt.get_cmap('tab20' if n_clusters > 10 else 'tab10', n_clusters)
    colors = [cmap(i) for i in range(n_clusters)]
    legend_handles = [mpatches.Patch(color=colors[i], label=cluster_names[i]) for i in range(n_clusters)]
    ax.legend(handles=legend_handles, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    scatter = ax.scatter([], [], s=dot_size, alpha=0.8)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, ha='left', va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.7))
    fig.suptitle(fig_title, fontsize=16); plt.tight_layout(rect=[0, 0, 0.8, 0.95])
    def update(frame_idx):
        coords = spatial_trajectory[frame_idx, :, :]
        labels = cluster_labels_trajectory[frame_idx, :]
        valid_mask = ~np.isnan(coords[:, 0])
        scatter.set_offsets(coords[valid_mask])
        scatter.set_color([colors[int(l)] for l in labels[valid_mask]])
        time_text.set_text(f"Bio Time: {biological_times[frame_idx]:.2f}")
        return scatter, time_text
    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=True, interval=1000/fps)
    try:
        ani.save(output_path, writer='pillow', fps=fps, dpi=120)
        print(f"✓ Animation successfully saved to {output_path}")
    except Exception as e:
        print(f"✗ Error saving animation: {e}"); traceback.print_exc()
    plt.close(fig)


def main_animation_mode(args):
    """Main execution function for creating animations from simulation data."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # --- Argument Validation ---
    required_args = ['simulation_dir', 'model_load_path', 'config_train_load_path', 'original_adata_path', 'output_dir']
    if any(getattr(args, arg) is None for arg in required_args):
        print("Error: For animation creation, all of the following must be provided:")
        print(", ".join(f'--{arg}' for arg in required_args))
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Loading data and models...")
        sim_dir = Path(args.simulation_dir)
        traj_file = sim_dir / "state_trajectories_backward.npz"
        times_file = sim_dir / "biological_times_backward.npy"
        if not traj_file.exists() or not times_file.exists():
            print(f"Error: Simulation files not found in {sim_dir}. Please run the simulation step first.")
            sys.exit(1)
            
        trajectory_npz = np.load(traj_file)
        state_list = [trajectory_npz[k] for k in sorted(trajectory_npz.keys(), key=lambda k: int(k.split('_')[1]))]
        bio_times_full = np.load(times_file)

        with open(args.config_train_load_path, 'r') as f: train_args = json.load(f)
        adata_orig = sc.read_h5ad(args.original_adata_path)

        if VAEWrapper is None: raise ImportError("VAEWrapper could not be imported.")
        vae_model = VAEWrapper(adata_orig.n_vars, [int(d) for d in train_args['vae_hidden_dims'].split(',')], train_args['vae_latent_dim']).to(device).float()
        vae_model.load_state_dict(torch.load(args.model_load_path, map_location=device)['vae_state_dict'])
        vae_model.eval()

        spatial_dim, latent_dim = train_args['spatial_dim'], train_args['vae_latent_dim']

        print("Processing trajectory data...")
        n_frames = len(state_list)
        max_cells = max(arr.shape[0] for arr in state_list) if state_list else 0
        state_padded = np.full((n_frames, max_cells, spatial_dim + latent_dim), np.nan, dtype=np.float32)
        for i, arr in enumerate(state_list): state_padded[i, :arr.shape[0], :] = arr

        if args.animation_end_time_bio is not None:
            valid_indices = np.where(bio_times_full >= args.animation_end_time_bio)[0]
            if len(valid_indices) > 0:
                last_idx = valid_indices[-1]
                state_padded = state_padded[:last_idx + 1]
                bio_times_full = bio_times_full[:last_idx + 1]

        all_latent = state_padded[:, :, spatial_dim:].reshape(-1, latent_dim)
        valid_mask = ~np.isnan(all_latent[:, 0])
        centroids, cluster_names = get_cluster_centroids(all_latent[valid_mask], args.clustering_method, args.num_clusters, adata_orig, args.annotation_key, vae_model, device)
        all_labels = np.full(all_latent.shape[0], np.nan)
        all_labels[valid_mask] = assign_to_clusters(all_latent[valid_mask], centroids)
        cluster_labels_traj = all_labels.reshape(state_padded.shape[0], state_padded.shape[1])

        num_points = state_padded.shape[0]
        indices = np.linspace(0, num_points - 1, args.animation_frames, dtype=int) if args.animation_frames > 0 else np.arange(num_points)
        anim_spatial = state_padded[indices, :, :spatial_dim]
        anim_labels = cluster_labels_traj[indices, :]
        anim_times = bio_times_full[indices]

    except Exception as e:
        print(f"An error occurred during data loading or processing: {e}")
        traceback.print_exc()
        sys.exit(1)

    gif_filename = f"clustered_animation_{args.clustering_method}.gif"
    forward_output_path = output_dir / gif_filename
    create_clustered_animation(
        spatial_trajectory=anim_spatial,
        cluster_labels_trajectory=anim_labels,
        biological_times=anim_times,
        cluster_names=cluster_names,
        output_path=forward_output_path,
        fig_title=f"Backward Simulation with {args.clustering_method.title()} Clusters",
        fps=args.animation_fps,
        dot_size=args.animation_dot_size
    )

    if args.create_reverse:
        reverse_gif_filename = f"clustered_animation_{args.clustering_method}_reverse.gif"
        reverse_output_path = output_dir / reverse_gif_filename
        create_reverse_gif(forward_output_path, reverse_output_path)

    print("\nAnimation Mode complete.")


if __name__ == "__main__":
    args = parse_args()

    # --- Mode Dispatcher ---
    if args.reverse_only_input_gif:
        # --- Run in Reverse-Only Mode ---
        print("--- Running in Reverse-Only Mode ---")
        input_path = Path(args.reverse_only_input_gif)
        output_path = input_path.with_name(f"{input_path.stem}_reverse{input_path.suffix}")
        create_reverse_gif(input_path, output_path)
        print("\nReverse-Only Mode complete.")
    else:
        # --- Run in Animation Creation Mode (Default) ---
        print("--- Running in Animation Creation Mode ---")
        main_animation_mode(args)

