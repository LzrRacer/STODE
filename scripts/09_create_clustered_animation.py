#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import sys
import traceback
from tqdm import tqdm
from PIL import Image, ImageSequence

# --- Setup Project Paths and PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def parse_args():
    """Parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(description="Create a GIF animation of simulated cell trajectories with pre-computed clusters.")
    parser.add_argument('--simulation_dir', type=str, required=True, help="Path to the directory with simulation results.")
    parser.add_argument('--clustered_adata_path', type=str, required=True, help="Path to the AnnData object with pre-computed cluster assignments (from script 05).")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output animation.")
    parser.add_argument('--animation_frames', type=int, default=0, help="Max number of frames for animation (0 for all).")
    parser.add_argument('--animation_fps', type=int, default=15, help="Frames per second for the animation.")
    parser.add_argument('--animation_dot_size', type=int, default=15, help="Size of the dots in the animation.")
    parser.add_argument("--animation_end_time_bio", type=float, default=None, help="Biological time to end the animation at.")
    parser.add_argument('--create_reverse', action='store_true', help="After creating the animation, also save a reversed version.")
    parser.add_argument('--seed', type=int, default=42)
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
    except Exception as e:
        print(f"✗ Error reversing GIF: {e}")

def create_clustered_animation(spatial_trajectory, cluster_labels_trajectory, biological_times, cluster_names, cluster_colors, output_path, fig_title, fps, dot_size):
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
    
    # Use the provided color palette
    if cluster_colors is None or len(cluster_colors) < n_clusters:
        cmap = plt.get_cmap('tab20' if n_clusters > 10 else 'tab10', n_clusters)
        colors = [cmap(i) for i in range(n_clusters)]
    else:
        colors = cluster_colors

    cluster_name_to_color = {name: color for name, color in zip(cluster_names, colors)}
    
    legend_handles = [mpatches.Patch(color=color, label=name) for name, color in cluster_name_to_color.items()]
    ax.legend(handles=legend_handles, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    scatter = ax.scatter([], [], s=dot_size, alpha=0.8)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, ha='left', va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.7))
    fig.suptitle(fig_title, fontsize=16); plt.tight_layout(rect=[0, 0, 0.8, 0.95])

    def update(frame_idx):
        coords = spatial_trajectory[frame_idx, :, :]
        labels = cluster_labels_trajectory[frame_idx, :]
        valid_mask = ~np.isnan(coords[:, 0]) & ~pd.isna(labels)
        
        scatter.set_offsets(coords[valid_mask])
        frame_colors = [cluster_name_to_color[l] for l in labels[valid_mask]]
        scatter.set_color(frame_colors)
        time_text.set_text(f"Bio Time: {biological_times[frame_idx]:.2f}")
        return scatter, time_text

    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=True, interval=1000/fps)
    try:
        ani.save(output_path, writer='pillow', fps=fps, dpi=120)
        print(f"✓ Animation successfully saved to {output_path}")
    except Exception as e:
        print(f"✗ Error saving animation: {e}"); traceback.print_exc()
    plt.close(fig)

def main(args):
    """Main execution function for creating animations."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Loading data...")
        sim_dir = Path(args.simulation_dir)
        traj_file = sim_dir / "state_trajectories_backward.npz"
        times_file = sim_dir / "biological_times_backward.npy"
        
        trajectory_npz = np.load(traj_file)
        state_list = [trajectory_npz[k] for k in sorted(trajectory_npz.keys(), key=lambda k: int(k.split('_')[1]))]
        bio_times_full = np.load(times_file)

        adata_clustered = sc.read_h5ad(args.clustered_adata_path)
        
        # --- Determine the correct cluster key ---
        cluster_key = 'kmeans_cluster'
        if 'annotation_cluster' in adata_clustered.obs.columns:
             cluster_key = 'annotation_cluster'
        print(f"Using cluster key: '{cluster_key}'")

        cluster_names = adata_clustered.obs[cluster_key].cat.categories.tolist()
        cluster_colors = adata_clustered.uns.get(f'{cluster_key}_colors', None)
        
        print("Processing trajectory data and mapping clusters...")
        n_frames = len(state_list)
        max_cells = max(arr.shape[0] for arr in state_list) if state_list else 0
        
        if n_frames == 0 or max_cells == 0:
            print("Error: Trajectory data is empty."); sys.exit(1)

        # Pad spatial trajectory and prepare for cluster label mapping
        spatial_padded = np.full((n_frames, max_cells, 2), np.nan, dtype=np.float32)
        for i, arr in enumerate(state_list):
            spatial_padded[i, :arr.shape[0], :] = arr[:, :2]

        # Map cluster labels for each frame
        cluster_labels_traj = np.full((n_frames, max_cells), None, dtype=object)
        for i, t in enumerate(tqdm(bio_times_full, desc="Mapping clusters to frames")):
            adata_t = adata_clustered[adata_clustered.obs['sim_bio_time'] == t]
            if adata_t.n_obs > 0:
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(adata_t.obsm['spatial'], adata_t.obs[cluster_key])
                
                frame_spatial = spatial_padded[i, ~np.isnan(spatial_padded[i, :, 0])]
                if frame_spatial.shape[0] > 0:
                    frame_labels = knn.predict(frame_spatial)
                    cluster_labels_traj[i, :len(frame_labels)] = frame_labels

        if args.animation_end_time_bio is not None:
            valid_indices = np.where(bio_times_full >= args.animation_end_time_bio)[0]
            if len(valid_indices) > 0:
                last_idx = valid_indices[-1]
                spatial_padded = spatial_padded[:last_idx + 1]
                cluster_labels_traj = cluster_labels_traj[:last_idx + 1]
                bio_times_full = bio_times_full[:last_idx + 1]

        num_points = spatial_padded.shape[0]
        indices = np.linspace(0, num_points - 1, args.animation_frames, dtype=int) if args.animation_frames > 0 else np.arange(num_points)
        
        anim_spatial = spatial_padded[indices]
        anim_labels = cluster_labels_traj[indices]
        anim_times = bio_times_full[indices]

    except Exception as e:
        print(f"An error occurred during data loading or processing: {e}"); traceback.print_exc(); sys.exit(1)

    gif_filename = f"clustered_animation_{cluster_key}.gif"
    forward_output_path = output_dir / gif_filename
    create_clustered_animation(
        spatial_trajectory=anim_spatial,
        cluster_labels_trajectory=anim_labels,
        biological_times=anim_times,
        cluster_names=cluster_names,
        cluster_colors=cluster_colors,
        output_path=forward_output_path,
        fig_title=f"Backward Simulation with {cluster_key.replace('_', ' ').title()}s",
        fps=args.animation_fps,
        dot_size=args.animation_dot_size
    )

    if args.create_reverse:
        reverse_gif_filename = f"{Path(gif_filename).stem}_reverse.gif"
        reverse_output_path = output_dir / reverse_gif_filename
        create_reverse_gif(forward_output_path, reverse_output_path)

    print("\nAnimation Mode complete.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
