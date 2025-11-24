#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from matplotlib.colors import Normalize
import torch
import scanpy as sc 
from tqdm import tqdm
import sys
import traceback

# --- Setup Project Paths and PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.vae import VAEWrapper 
from utils.plotting import compress_to_1d 

def parse_args_analyze_backward():
    parser = argparse.ArgumentParser(description="Analyze and visualize BACKWARD simulated trajectories.")
    parser.add_argument('--backward_simulation_dir', type=str, required=True, help="Path to the directory with BACKWARD simulation results.")
    parser.add_argument('--model_load_path', type=str, required=True, help="Path to the trained_system_model.pt file.")
    parser.add_argument('--config_train_load_path', type=str, required=True, help="Path to the config_train.json from training.")
    parser.add_argument('--original_adata_path_for_vae_config', type=str, default=None, help="Path to original AnnData used in training (for VAE n_vars if not in config).")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save analysis and plots.")
    parser.add_argument('--animation_frames', type=int, default=0, help="Max number of frames for trajectory animation (0 for all available after truncation).")
    parser.add_argument('--animation_fps', type=int, default=10)
    parser.add_argument('--animation_dot_size', type=int, default=15)
    parser.add_argument("--animation_end_time_bio", type=float, default=None,
                        help="Biological time to end the animation at (e.g., 6.0 for E6.0). Default is to use full trajectory up to t=0.")
    parser.add_argument("--auto_adjust_fps_to_bio_time", action="store_true",
                        help="If set, automatically adjust FPS so 1 day of biological time is approx. 1 second of animation, overriding --animation_fps.")
    parser.add_argument('--latent_compress_method', type=str, default='umap', choices=['umap', 'pca', 'tsne'])
    parser.add_argument('--latent_compress_umap_neighbors', type=int, default=15)
    parser.add_argument('--latent_compress_subsample_fit', type=int, default=10000,
                        help="Number of cells to subsample for fitting UMAP. Set to 0 to disable. Default: 10000.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default="", help="Device ('cpu', 'cuda', or empty for auto).")
    parser.add_argument("--dynamic_axis_limits", action="store_true", help="Enable dynamic axis limits in animations to show tissue size changes.")
    args = parser.parse_args()
    return args

def create_animation(
    spatial_data_trajectory_np,
    color_values_trajectory_np,
    biological_times_trajectory_np,
    output_path,
    fig_title_base,
    cmap_name,
    cbar_label,
    sim_summary_info,
    animation_fps_val=10,
    dot_size_val=20,
    is_reversed_for_replay=False,
    dynamic_axis_limits=False
):
    print(f"Creating animation: {output_path.name}...")

    if not isinstance(spatial_data_trajectory_np, np.ndarray) or spatial_data_trajectory_np.ndim != 3 or spatial_data_trajectory_np.shape[0] == 0:
        print(f"Warning: Invalid or empty spatial_data_trajectory_np for animation '{fig_title_base}'. Skipping.")
        return
    
    n_anim_frames, n_anim_cells, spatial_dim_anim = spatial_data_trajectory_np.shape
    if spatial_dim_anim < 2 or n_anim_cells == 0:
        print(f"Warning: Invalid spatial dimensions or zero cells for animation '{fig_title_base}'. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlabel("Spatial Dim 1")
    ax.set_ylabel("Spatial Dim 2")
    ax.set_aspect('equal', adjustable='box')

    # --- Set initial axis limits correctly based on the mode ---
    if dynamic_axis_limits:
        # For dynamic axes, base initial limits on the first frame
        initial_spatial_coords = spatial_data_trajectory_np[0, :, :2]
        min_x, min_y = np.nanmin(initial_spatial_coords, axis=0) if initial_spatial_coords.size > 0 else (-1,-1)
        max_x, max_y = np.nanmax(initial_spatial_coords, axis=0) if initial_spatial_coords.size > 0 else (1,1)
    else:
        # For static axes, base limits on the entire trajectory to show size changes
        all_spatial_flat = spatial_data_trajectory_np[:, :, :2].reshape(-1, 2)
        min_x, min_y = np.nanmin(all_spatial_flat, axis=0) if all_spatial_flat.size > 0 else (-1, -1)
        max_x, max_y = np.nanmax(all_spatial_flat, axis=0) if all_spatial_flat.size > 0 else (1, 1)

    padding_x = (max_x - min_x) * 0.1 if (max_x - min_x) > 1e-6 else 0.1
    padding_y = (max_y - min_y) * 0.1 if (max_y - min_y) > 1e-6 else 0.1
    ax.set_xlim(min_x - padding_x, max_x + padding_x)
    ax.set_ylim(min_y - padding_y, max_y + padding_y)
    
    # Setup for colors, scatter plot, and text
    use_custom_colors = (
        color_values_trajectory_np is not None and
        isinstance(color_values_trajectory_np, np.ndarray) and
        color_values_trajectory_np.shape == (n_anim_frames, n_anim_cells) and
        not np.all(np.isnan(color_values_trajectory_np))
    )
    cbar_mappable = None
    norm = None
    if use_custom_colors:
        c_min_global, c_max_global = np.nanmin(color_values_trajectory_np), np.nanmax(color_values_trajectory_np)
        if not (np.isnan(c_min_global) or np.isnan(c_max_global)):
            norm = Normalize(vmin=c_min_global, vmax=c_max_global if c_max_global > c_min_global else c_min_global + 1e-6)
            cmap_obj = plt.get_cmap(cmap_name)
            cbar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        else:
            use_custom_colors = False
            
    initial_spatial_coords = spatial_data_trajectory_np[0, :, :2]
    scatter_kwargs = {'s': dot_size_val, 'alpha': 0.7}
    if use_custom_colors and cbar_mappable:
        initial_colors_frame0 = color_values_trajectory_np[0, :]
        scatter_kwargs['c'] = cbar_mappable.cmap(norm(initial_colors_frame0)) if not np.all(np.isnan(initial_colors_frame0)) else 'grey'
    else:
        scatter_kwargs['c'] = 'grey'
        
    scatter = ax.scatter(initial_spatial_coords[:, 0], initial_spatial_coords[:, 1], **scatter_kwargs)
    
    if use_custom_colors and cbar_mappable:
        fig.colorbar(cbar_mappable, ax=ax, label=cbar_label, pad=0.1)

    sim_start_time_display = sim_summary_info.get("started_from_biological_time", "N/A")
    sim_end_time_display = sim_summary_info.get("t0_bio", "N/A")
    
    fig_title_full = f"{fig_title_base}\n(Simulated {sim_start_time_display} -> {sim_end_time_display})"
    if is_reversed_for_replay:
        fig_title_full = f"{fig_title_base}\n(Replayed Sim: {sim_end_time_display} -> {sim_start_time_display})"
        
    fig.suptitle(fig_title_full, fontsize=14)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, ha='left', va='top', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])

    def update(frame_idx):
        """Function to update the animation frame."""
        spatial_coords = spatial_data_trajectory_np[frame_idx, :, :2]
        scatter.set_offsets(spatial_coords)

        if dynamic_axis_limits:
            valid_coords = spatial_coords[~np.isnan(spatial_coords).any(axis=1)]
            if valid_coords.shape[0] > 0:
                min_x, min_y = np.nanmin(valid_coords, axis=0)
                max_x, max_y = np.nanmax(valid_coords, axis=0)
                padding_x = (max_x - min_x) * 0.1 if (max_x - min_x) > 1e-6 else 0.1
                padding_y = (max_y - min_y) * 0.1 if (max_y - min_y) > 1e-6 else 0.1
                ax.set_xlim(min_x - padding_x, max_x + padding_x)
                ax.set_ylim(min_y - padding_y, max_y + padding_y)
        
        if use_custom_colors and norm is not None:
            current_color_vals = color_values_trajectory_np[frame_idx, :]
            scatter.set_color(cbar_mappable.cmap(norm(current_color_vals)) if not np.all(np.isnan(current_color_vals)) else 'grey')
            
        time_text.set_text(f"Bio Time: {biological_times_trajectory_np[frame_idx]:.2f}")
        ax.set_title(f"Frame {frame_idx+1}/{n_anim_frames}", fontsize=10)
        return scatter, time_text

    ani = animation.FuncAnimation(fig, update, frames=n_anim_frames, blit=True, interval=max(20, 1000 // animation_fps_val))
    try:
        ani.save(output_path, writer='pillow', fps=animation_fps_val, dpi=100)
    except Exception as e:
        print(f"Error saving animation with Pillow writer: {e}. Trying imagemagick...")
        try:
            ani.save(output_path, writer='imagemagick', fps=animation_fps_val, dpi=100)
        except Exception as e_im:
            print(f"Error saving animation with ImageMagick: {e_im}. Please ensure a writer is installed.")
            traceback.print_exc()
    plt.close(fig)
    print(f"Animation saved to {output_path}")

def main(args):
    """Main execution function for the analysis."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_dir = Path(args.backward_simulation_dir)
    print("Loading BACKWARD simulation data...")

    try:
        biological_times_sim_bwd_full = np.load(sim_dir / "biological_times_backward.npy")
        with open(sim_dir / "simulation_summary_backward.json", 'r') as f:
            sim_summary_info_global = json.load(f)
        trajectory_npz_path = sim_dir / "state_trajectories_backward.npz"
        if not trajectory_npz_path.exists():
             state_trajectory_bwd_full = np.load(sim_dir / "state_trajectories_backward.npy")
        else:
            trajectory_npz = np.load(trajectory_npz_path)
            sorted_keys = sorted(trajectory_npz.keys(), key=lambda k: int(k.split('_')[1]))
            state_trajectory_list = [trajectory_npz[key] for key in sorted_keys]
            print("Padding ragged trajectory data for animation...")
            n_frames = len(state_trajectory_list)
            max_cells = max(arr.shape[0] for arr in state_trajectory_list) if state_trajectory_list else 0
            state_dim = state_trajectory_list[0].shape[1] if state_trajectory_list and state_trajectory_list[0].ndim > 1 else 0
            if n_frames == 0 or max_cells == 0 or state_dim == 0:
                print("Error: Trajectory data is empty. Cannot create animation.")
                return
            state_trajectory_bwd_full = np.full((n_frames, max_cells, state_dim), np.nan, dtype=np.float32)
            for i, arr in enumerate(state_trajectory_list):
                state_trajectory_bwd_full[i, :arr.shape[0], :] = arr
            print(f"Padded trajectory to shape: {state_trajectory_bwd_full.shape}")
    except FileNotFoundError as e:
        print(f"Error: Simulation file not found in {sim_dir}. Please run backward simulation first.\nMissing file: {e.filename}")
        return

    with open(args.config_train_load_path, 'r') as f:
        train_args = json.load(f)
    spatial_dim = train_args['spatial_dim']
    latent_dim = train_args['vae_latent_dim']
    
    biological_times_anim_base = biological_times_sim_bwd_full.copy()
    state_trajectory_anim_base = state_trajectory_bwd_full.copy()
    if args.animation_end_time_bio is not None:
        valid_indices = np.where(biological_times_anim_base >= args.animation_end_time_bio)[0]
        if len(valid_indices) > 0:
            last_valid_sim_step_idx = valid_indices[-1] 
            state_trajectory_anim_base = state_trajectory_anim_base[:last_valid_sim_step_idx + 1, ...]
            biological_times_anim_base = biological_times_anim_base[:last_valid_sim_step_idx + 1]
            print(f"Trajectory truncated for animation. New end time: {biological_times_anim_base[-1]:.2f}, Num points: {len(biological_times_anim_base)}")
        else:
            print(f"Warning: animation_end_time_bio ({args.animation_end_time_bio}) is before the start of the trajectory. Using full original backward trajectory.")

    num_total_trajectory_points_base = state_trajectory_anim_base.shape[0]
    num_cells_base = state_trajectory_anim_base.shape[1]
    if num_total_trajectory_points_base == 0:
        print("Error: No trajectory points remaining after filtering. Cannot create animation.")
        return

    anim_spatial_full_base = state_trajectory_anim_base[:, :, :spatial_dim]
    anim_latent_full_base = state_trajectory_anim_base[:, :, spatial_dim:spatial_dim + latent_dim]
    flat_anim_latent_full_base = anim_latent_full_base.reshape(-1, latent_dim)
    valid_latent_mask = ~np.isnan(flat_anim_latent_full_base[:, 0])
    anim_colors_1d_latent_full_base = np.full((num_total_trajectory_points_base, num_cells_base), np.nan) 
    
    if np.any(valid_latent_mask):
        print(f"Compressing {valid_latent_mask.sum()} latent vectors to 1D using {args.latent_compress_method}...")
        try:
            compressed_1d_all_latent = compress_to_1d(
                flat_anim_latent_full_base[valid_latent_mask],
                method=args.latent_compress_method,
                n_neighbors=args.latent_compress_umap_neighbors,
                random_state=args.seed,
                subsample_fit=args.latent_compress_subsample_fit if args.latent_compress_subsample_fit > 0 else None
            )
            temp_full_1d = np.full(flat_anim_latent_full_base.shape[0], np.nan)
            temp_full_1d[valid_latent_mask] = compressed_1d_all_latent
            anim_colors_1d_latent_full_base = temp_full_1d.reshape(num_total_trajectory_points_base, num_cells_base)
            print(f"  1D compression complete. Min: {np.nanmin(anim_colors_1d_latent_full_base):.2f}, Max: {np.nanmax(anim_colors_1d_latent_full_base):.2f}")
        except Exception as e_comp:
            print(f"  Warning: compress_to_1d failed for animation coloring: {e_comp}")
            traceback.print_exc()
    else:
        print("  Warning: Not enough data to compress latent space for coloring.")

    indices_for_display_animation = np.linspace(0, num_total_trajectory_points_base - 1, args.animation_frames, dtype=int) if args.animation_frames > 0 and args.animation_frames < num_total_trajectory_points_base else np.arange(num_total_trajectory_points_base)
    
    anim_spatial_data_shrink_np = anim_spatial_full_base[indices_for_display_animation, :, :]
    anim_color_data_1d_shrink_np = anim_colors_1d_latent_full_base[indices_for_display_animation, :]
    anim_bio_times_shrink_np = biological_times_anim_base[indices_for_display_animation]

    fps_to_use = args.animation_fps
    if args.auto_adjust_fps_to_bio_time and len(anim_bio_times_shrink_np) > 1:
        bio_time_span_anim = abs(anim_bio_times_shrink_np[-1] - anim_bio_times_shrink_np[0])
        if bio_time_span_anim > 1e-6:
            fps_to_use = max(1, int(round(len(anim_bio_times_shrink_np) / bio_time_span_anim)))
            print(f"Auto-adjusting FPS to ~1 sec/day -> Using FPS = {fps_to_use}.")

    cbar_label_1d = f"Latent 1D ({args.latent_compress_method.upper()})"
    gif_path_shrink = output_dir / f"backward_contraction_to_t0_latent1D_{args.latent_compress_method}.gif"
    
    sim_summary_for_shrink_title = { 
        "started_from_biological_time": f"{biological_times_sim_bwd_full[0]:.2f}", 
        "t0_bio": f"{biological_times_anim_base[-1]:.2f}" 
    }
    create_animation(
        spatial_data_trajectory_np=anim_spatial_data_shrink_np,
        color_values_trajectory_np=anim_color_data_1d_shrink_np,
        biological_times_trajectory_np=anim_bio_times_shrink_np,
        output_path=gif_path_shrink,
        fig_title_base=f"Backward Contraction (Observed -> Earliest Animated)",
        cmap_name='viridis', 
        cbar_label=cbar_label_1d,
        sim_summary_info=sim_summary_for_shrink_title,
        animation_fps_val=fps_to_use,
        dot_size_val=args.animation_dot_size,
        is_reversed_for_replay=False,
        dynamic_axis_limits=args.dynamic_axis_limits
    )
    
    anim_spatial_data_replayed_np = anim_spatial_data_shrink_np[::-1, :, :].copy()
    anim_color_data_1d_replayed_np = anim_color_data_1d_shrink_np[::-1, :].copy()
    anim_bio_times_replayed_np = anim_bio_times_shrink_np[::-1].copy()
    gif_path_differentiate_replayed = output_dir / f"forward_differentiation_replayed_latent1D_{args.latent_compress_method}.gif"
    sim_summary_for_replay_title = {
        "started_from_biological_time": f"{anim_bio_times_replayed_np[0]:.2f}",
        "t0_bio": f"{anim_bio_times_replayed_np[-1]:.2f}" 
    }
    create_animation(
        spatial_data_trajectory_np=anim_spatial_data_replayed_np,
        color_values_trajectory_np=anim_color_data_1d_replayed_np,
        biological_times_trajectory_np=anim_bio_times_replayed_np,
        output_path=gif_path_differentiate_replayed,
        fig_title_base=f"Forward Differentiation (Replayed)",
        cmap_name='magma', 
        cbar_label=cbar_label_1d,
        sim_summary_info=sim_summary_for_replay_title,
        animation_fps_val=fps_to_use,
        dot_size_val=args.animation_dot_size,
        is_reversed_for_replay=True,
        dynamic_axis_limits=args.dynamic_axis_limits
    )

    print(f"Backward simulation analysis complete. GIFs saved to {output_dir.resolve()}")

if __name__ == "__main__":
    args = parse_args_analyze_backward()
    sc.settings.figdir = Path(args.output_dir) / "scanpy_plots" 
    sc.settings.figdir.mkdir(parents=True, exist_ok=True)
    sc.settings.autoshow = False 
    sc.settings.autosave = True
    main(args)