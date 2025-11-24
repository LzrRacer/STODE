#!/usr/bin/env python3
# scripts/02_simulate_backward_from_observed.py

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import scanpy as sc
import torch

# --- Setup Project Paths and PYTHONPATH (must run before local imports) ---
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
    If requested_key is missing, try common fallbacks and print what's available.
    """
    obsm_keys = list(adata.obsm.keys())
    if requested_key in adata.obsm:
        return requested_key
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
    ap = argparse.ArgumentParser("Simulate BACKWARD trajectories from an OBSERVED state.")
    ap.add_argument("--model_load_path", required=True)
    ap.add_argument("--config_train_load_path", required=True)
    ap.add_argument("--original_adata_path_for_vae_input_dim", default=None)
    ap.add_argument("--observed_adata_path", required=True)
    ap.add_argument("--observed_time_point_numeric", type=float, required=True)
    ap.add_argument("--t_final_bio_target", type=float, default=0.0)
    ap.add_argument("--num_cells_to_sample_from_observed", type=int, default=0)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--simulation_n_steps", type=int, default=120)
    ap.add_argument("--grid_size", type=int, default=0)  # kept for CLI compatibility
    ap.add_argument(
        "--attraction_strength",
        type=float,
        default=0.0,
        help="Strength of an optional explicit contractive drift on spatial coords.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="")
    return ap.parse_args()


@torch.no_grad()
def main(args):
    # ------------------
    # Device & RNG
    # ------------------
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------------------
    # Load training config
    # ------------------
    with open(args.config_train_load_path) as f:
        cfg = json.load(f)

    # ------------------
    # Load observed data
    # ------------------
    adata = sc.read_h5ad(args.observed_adata_path)
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray().astype(np.float32)
    else:
        adata.X = adata.X.astype(np.float32)

    spatial_key = resolve_spatial_key(adata, cfg.get("spatial_key", "spatial"))

    # ------------------
    # Build models
    # ------------------
    # VAE input dim (can differ from observed adata if you trained on a different var set)
    input_dim = adata.n_vars
    if args.original_adata_path_for_vae_input_dim:
        ad0 = sc.read_h5ad(args.original_adata_path_for_vae_input_dim)
        input_dim = ad0.n_vars

    vae = VAEWrapper(
        input_dim=input_dim,
        hidden_dims=[int(d) for d in str(cfg.get("vae_hidden_dims", "128,64")).split(",")],
        latent_dim=cfg["vae_latent_dim"],
        dropout_rate=cfg.get("vae_dropout_rate", 0.1),
    ).to(device).float()

    potential = SpatioTemporalPotential(
        spatial_dim=cfg["spatial_dim"],
        latent_dim=cfg["vae_latent_dim"],
        time_embedding_dim=cfg.get("potential_time_embedding_dim", 4),
        hidden_dims=[int(d) for d in str(cfg.get("potential_hidden_dims", "32,16")).split(",")],
    ).to(device).float()

    # Time-aware ODE (match training, including optional time-affine drift)
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
    # Load weights (safe)
    # ------------------
    blob = torch.load(args.model_load_path, map_location=device)
    vae.load_state_dict(blob["vae_state_dict"], strict=False)
    vae.eval()

    potential.load_state_dict(blob["potential_state_dict"], strict=False)
    potential.eval()

    missing, unexpected = ode.load_state_dict(blob["ode_state_dict"], strict=False)
    if unexpected:
        print(f"[warn] Unexpected ODE keys ignored: {sorted(unexpected)}")
    if missing:
        print(f"[warn] Missing ODE keys: {sorted(missing)}")
    ode.eval()

    # ------------------
    # Build initial state at observed time
    # ------------------
    # For simplicity, we use all cells in adata as the observed set.
    if args.num_cells_to_sample_from_observed > 0 and adata.n_obs > args.num_cells_to_sample_from_observed:
        sc.pp.subsample(adata, n_obs=args.num_cells_to_sample_from_observed, random_state=args.seed)

    X = adata.X.astype(np.float32)
    x_t = torch.tensor(X, device=device, dtype=torch.float32)
    mu, _ = vae.vae.encoder(x_t)

    s_obs = torch.tensor(adata.obsm[spatial_key], device=device, dtype=torch.float32)
    init_state = torch.cat([s_obs, mu], dim=1)  # [N, spatial_dim + latent_dim]

    # ------------------
    # Backward Euler integration
    # ------------------
    t_obs = float(args.observed_time_point_numeric)
    t_fin = float(args.t_final_bio_target)
    steps = int(args.simulation_n_steps)
    dt = abs(t_obs - t_fin) / max(1, steps)
    dt_t = torch.tensor(dt, device=device, dtype=torch.float32)

    def step_velocity(current_state, t_scalar):
        """Compute d(state)/dt at time t_scalar using ODE with -∇U inputs."""
        t_b = torch.tensor([t_scalar], device=device, dtype=torch.float32).repeat(
            current_state.size(0), 1
        )
        s_rg = current_state[:, :cfg["spatial_dim"]].detach().requires_grad_(True)
        z_rg = current_state[:, cfg["spatial_dim"]:].detach().requires_grad_(True)
        grad_U_s, grad_U_l = potential.calculate_gradients(s_rg, z_rg, t_b, create_graph=False)
        dstate_dt = ode(t_b, (current_state, grad_U_s, grad_U_l))
        return dstate_dt

    states = [init_state.clone().detach()]
    current = init_state.clone().detach()

    for k in range(steps):
        tk = t_obs - k * dt  # going backward
        vel = step_velocity(current, tk)

        # Optional explicit contractive drift on spatial coords
        if args.attraction_strength > 0:
            spatial_slice = slice(0, cfg["spatial_dim"])
            vel[:, spatial_slice] = vel[:, spatial_slice] - args.attraction_strength * current[:, spatial_slice]

        current = current - dt_t * vel
        states.append(current.clone().detach())

    # ------------------
    # Save outputs
    # ------------------
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Compact tensor dump for downstream tools
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

    # Also save per-step arrays (useful for analysis/animations)
    traj = {f"arr_{i}": s.detach().cpu().numpy() for i, s in enumerate(states)}
    np.savez_compressed(outdir / "state_trajectories_backward.npz", **traj)
    np.save(outdir / "biological_times_backward.npy", np.linspace(t_obs, t_fin, steps + 1))

    print("Saved:", outdir / "backward_simulation.pt")
    print("Saved:", outdir / "state_trajectories_backward.npz")
    print("Saved:", outdir / "biological_times_backward.npy")

    # --- ADD THIS BLOCK to also emit the summary JSON expected by the analyzer ---
    summary = {
        "num_initial_cells_simulated": int(states[0].shape[0]),
        "num_final_cells_simulated": int(states[-1].shape[0]),
        "t_obs": float(t_obs),
        "t_final": float(t_fin),
        "steps": int(steps),
        "grid_size": int(args.grid_size),                # keep types consistent with analyzer
        "attraction_strength": float(args.attraction_strength),
        "spatial_key_used": str(spatial_key),
        "file_backward_simulation_pt": "backward_simulation.pt",
        "file_state_trajectories_npz": "state_trajectories_backward.npz",
        "file_bio_times_npy": "biological_times_backward.npy",
    }
    with open(outdir / "simulation_summary_backward.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", outdir / "simulation_summary_backward.json")

if __name__ == "__main__":
    args = parse_args()
    main(args)
