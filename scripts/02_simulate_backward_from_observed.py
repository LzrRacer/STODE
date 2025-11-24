#!/usr/bin/env python3
"""
scripts/02_simulate_backward_from_observed.py

Simulates backward trajectories starting from observed real data points (t_obs)
back to an earlier biological time (t_final_bio_target).

This script:
1. Loads the trained VAE, Potential, and TimeAwareODE models.
2. Encodes observed spatial transcriptomics data into the latent space.
3. Performs backward Euler integration using the learned vector field.
4. Outputs trajectories and metadata for downstream analysis.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import scanpy as sc
import torch

# --- Setup Project Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.time_ode import TimeAwareODE
from models.potential_field import SpatioTemporalPotential
from models.vae import VAEWrapper


def resolve_spatial_key(adata, requested_key: str) -> str:
    """
    Return a valid spatial key from adata.obsm.
    If requested_key is missing, tries common fallbacks.
    """
    obsm_keys = list(adata.obsm.keys())
    if requested_key in adata.obsm:
        return requested_key
    
    # Common fallbacks
    for cand in ["spatial", "spatial_aligned", "X_spatial", "X_umap", "X_pca"]:
        if cand in adata.obsm:
            print(
                f"[resolve_spatial_key] Requested '{requested_key}' not found; using '{cand}' instead. "
                f"(Available obsm keys: {obsm_keys})"
            )
            return cand
            
    raise KeyError(
        f"Requested spatial_key '{requested_key}' not found in adata.obsm. "
        f"Available keys: {obsm_keys}"
    )


def parse_args():
    ap = argparse.ArgumentParser(description="Simulate BACKWARD trajectories from an OBSERVED state.")
    
    # Paths
    ap.add_argument("--model_load_path", required=True, help="Path to the trained system_model.pt")
    ap.add_argument("--config_train_load_path", required=True, help="Path to the training config json")
    ap.add_argument("--observed_adata_path", required=True, help="Path to the .h5ad file containing observed data")
    ap.add_argument("--original_adata_path_for_vae_input_dim", default=None, 
                    help="Optional: Path to original data if observed data has different vars/features")
    ap.add_argument("--output_dir", required=True, help="Directory to save simulation results")

    # Simulation Parameters
    ap.add_argument("--observed_time_point_numeric", type=float, required=True, help="The numeric time t of the observed data")
    ap.add_argument("--t_final_bio_target", type=float, default=0.0, help="Target biological time to simulate backwards to")
    ap.add_argument("--simulation_n_steps", type=int, default=120, help="Number of integration steps")
    ap.add_argument("--num_cells_to_sample_from_observed", type=int, default=0, help="If > 0, subsample the observed data")
    ap.add_argument("--attraction_strength", type=float, default=0.0, 
                    help="Optional: Strength of explicit contractive drift on spatial coords (regularization)")
    ap.add_argument("--grid_size", type=int, default=0, help="Metadata field for compatibility with grid-based analyses")

    # System
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="")
    
    return ap.parse_args()


@torch.no_grad()
def main(args):
    # ------------------
    # 1. Setup & Config
    # ------------------
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load training configuration to ensure model architecture matches
    with open(args.config_train_load_path) as f:
        cfg = json.load(f)

    # ------------------
    # 2. Load Data
    # ------------------
    adata = sc.read_h5ad(args.observed_adata_path)
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray().astype(np.float32)
    else:
        adata.X = adata.X.astype(np.float32)

    spatial_key = resolve_spatial_key(adata, cfg.get("spatial_key", "spatial"))

    # Determine VAE input dimension
    input_dim = adata.n_vars
    if args.original_adata_path_for_vae_input_dim:
        ad0 = sc.read_h5ad(args.original_adata_path_for_vae_input_dim)
        input_dim = ad0.n_vars

    # ------------------
    # 3. Initialize Models
    # ------------------
    # VAE
    vae = VAEWrapper(
        input_dim=input_dim,
        hidden_dims=[int(d) for d in str(cfg.get("vae_hidden_dims", "128,64")).split(",")],
        latent_dim=cfg["vae_latent_dim"],
        dropout_rate=cfg.get("vae_dropout_rate", 0.1),
    ).to(device).float()

    # Potential Field
    potential = SpatioTemporalPotential(
        spatial_dim=cfg["spatial_dim"],
        latent_dim=cfg["vae_latent_dim"],
        time_embedding_dim=cfg.get("potential_time_embedding_dim", 4),
        hidden_dims=[int(d) for d in str(cfg.get("potential_hidden_dims", "32,16")).split(",")],
    ).to(device).float()

    # Time-Aware ODE (with optional time-affine drift support)
    time_affine_cfg = None
    if cfg.get("use_time_affine", False):
        time_affine_cfg = dict(
            basis=cfg.get("affine_basis", "fourier"),
            fourier_k=cfg.get("affine_fourier_k", 3),
            mlp_hidden=cfg.get("affine_mlp_hidden", 32),
            time_norm_mode="auto",
        )

    ode = TimeAwareODE(
        spatial_dim=cfg["spatial_dim"],
        latent_dim=cfg["vae_latent_dim"],
        time_embedding_dim=cfg.get("ode_time_embedding_dim", 4),
        grad_potential_dim=cfg["spatial_dim"] + cfg["vae_latent_dim"],
        hidden_dims=[int(d) for d in str(cfg.get("ode_hidden_dims", "64,32")).split(",")],
        spatial_speed_scale=cfg.get("ode_spatial_speed_scale", 1.0),
        latent_speed_scale=cfg.get("ode_latent_speed_scale", 0.25),
        time_affine_cfg=time_affine_cfg,
    ).to(device).float()

    # ------------------
    # 4. Load Weights
    # ------------------
    blob = torch.load(args.model_load_path, map_location=device)
    
    vae.load_state_dict(blob["vae_state_dict"], strict=False)
    vae.eval()

    potential.load_state_dict(blob["potential_state_dict"], strict=False)
    potential.eval()

    missing, unexpected = ode.load_state_dict(blob["ode_state_dict"], strict=False)
    if unexpected:
        print(f"[Warn] Unexpected ODE keys ignored: {sorted(unexpected)}")
    if missing:
        print(f"[Warn] Missing ODE keys: {sorted(missing)}")
    ode.eval()

    # ------------------
    # 5. Prepare Initial State
    # ------------------
    # Subsample if requested
    if args.num_cells_to_sample_from_observed > 0 and adata.n_obs > args.num_cells_to_sample_from_observed:
        sc.pp.subsample(adata, n_obs=args.num_cells_to_sample_from_observed, random_state=args.seed)

    # Encode Expression -> Latent z
    X_data = adata.X.astype(np.float32)
    x_t = torch.tensor(X_data, device=device, dtype=torch.float32)
    mu, _ = vae.vae.encoder(x_t)

    # Get Spatial Coordinates s
    s_obs = torch.tensor(adata.obsm[spatial_key], device=device, dtype=torch.float32)
    
    # Combined State y = [s, z]
    init_state = torch.cat([s_obs, mu], dim=1) 

    # ------------------
    # 6. Backward Simulation Loop
    # ------------------
    t_obs = float(args.observed_time_point_numeric)
    t_fin = float(args.t_final_bio_target)
    steps = int(args.simulation_n_steps)
    dt = abs(t_obs - t_fin) / max(1, steps)
    dt_t = torch.tensor(dt, device=device, dtype=torch.float32)

    def step_velocity(current_state, t_scalar):
        """Compute d(state)/dt using ODE with -∇U inputs."""
        t_b = torch.tensor([t_scalar], device=device, dtype=torch.float32).repeat(
            current_state.size(0), 1
        )
        s_rg = current_state[:, :cfg["spatial_dim"]].detach().requires_grad_(True)
        z_rg = current_state[:, cfg["spatial_dim"]:].detach().requires_grad_(True)
        
        # Calculate gradients (create_graph=False for inference)
        grad_U_s, grad_U_l = potential.calculate_gradients(s_rg, z_rg, t_b, create_graph=False)
        
        # ODE Forward pass
        dstate_dt = ode(t_b, (current_state, grad_U_s, grad_U_l))
        return dstate_dt

    states = [init_state.clone().detach()]
    current = init_state.clone().detach()

    print(f"Simulating backward: t={t_obs:.2f} -> t={t_fin:.2f} ({steps} steps)")
    
    for k in range(steps):
        tk = t_obs - k * dt  # Time flows backward
        vel = step_velocity(current, tk)

        # Optional: Explicit spatial contraction (regularization)
        if args.attraction_strength > 0:
            spatial_slice = slice(0, cfg["spatial_dim"])
            vel[:, spatial_slice] = vel[:, spatial_slice] - args.attraction_strength * current[:, spatial_slice]

        # Backward Euler Step: y(t-dt) = y(t) - dt * dy/dt
        current = current - dt_t * vel
        states.append(current.clone().detach())

    # ------------------
    # 7. Save Outputs
    # ------------------
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 7a. PyTorch dump (compact)
    torch.save(
        {
            "init_state": states[0].detach().cpu(),
            "final_state": states[-1].detach().cpu(),
            "t_obs": t_obs,
            "t_final": t_fin,
            "steps": steps,
            "cfg": cfg,
            "spatial_key_used": spatial_key,
        },
        outdir / "backward_simulation.pt",
    )

    # 7b. NumPy Arrays (for plotting/animation)
    traj = {f"arr_{i}": s.detach().cpu().numpy() for i, s in enumerate(states)}
    np.savez_compressed(outdir / "state_trajectories_backward.npz", **traj)
    np.save(outdir / "biological_times_backward.npy", np.linspace(t_obs, t_fin, steps + 1))

    # 7c. JSON Summary (for analysis pipeline)
    summary = {
        "num_initial_cells_simulated": int(states[0].shape[0]),
        "num_final_cells_simulated": int(states[-1].shape[0]),
        "t_obs": float(t_obs),
        "t_final": float(t_fin),
        "steps": int(steps),
        "grid_size": int(args.grid_size),
        "attraction_strength": float(args.attraction_strength),
        "spatial_key_used": str(spatial_key),
        "file_backward_simulation_pt": "backward_simulation.pt",
        "file_state_trajectories_npz": "state_trajectories_backward.npz",
        "file_bio_times_npy": "biological_times_backward.npy",
    }
    with open(outdir / "simulation_summary_backward.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Simulation Complete.")
    print(f"  Summary: {outdir / 'simulation_summary_backward.json'}")
    print(f"  Trajectories: {outdir / 'state_trajectories_backward.npz'}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
