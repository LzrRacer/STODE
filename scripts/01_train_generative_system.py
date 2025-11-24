#!/usr/bin/env python3
"""
scripts/01_train_generative_system.py

Train the Generative Spatiotemporal System (STODE).
This script trains the VAE (optional), Potential Field, and Time-Aware ODE components.
Features included:
- Strong t0 convergence and anchoring.
- Time-affine drift regularization.
- Spatial shrinkage priors.
- Robust time label parsing and spatial key resolution.
"""

import re
import json
import sys
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pandas.api.types import CategoricalDtype

# --- Setup Project Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from geomloss import SamplesLoss  # noqa: F401
from losses.custom_losses import (
    sliced_wasserstein_distance,
    sinkhorn_divergence_loss,
    LatentTimeRegressor,
)
from utils.ot_utils import barycentric_map_entropic_ot
from models.vae import VAEWrapper
from models.potential_field import SpatioTemporalPotential
from models.time_ode import TimeAwareODE
from data_utils import PairedTimeDataset
from utils.training_helpers import calculate_t0_convergence_loss_smoothed
from losses.stode_loss import time_affine_regularizers, make_uniform_tau_grid


# ---------------------------
# Robust Time Parsing Helpers
# ---------------------------
_num_pattern = re.compile(r"\d+(?:\.\d+)?")

def _extract_first_number(s: str):
    m = _num_pattern.findall(str(s))
    if m:
        try:
            return float(m[0])
        except Exception:
            return None
    return None

def _time_label_sort_key(s: str):
    """Sort key: numeric part first if present, else lexicographic fallback."""
    v = _extract_first_number(s)
    if v is not None:
        return (0, v, str(s))
    return (1, str(s))

def get_ordered_time_labels_and_map(adata, time_key: str):
    """
    Returns ordered time labels and a mapping to numeric values.
    Prioritizes ordered Categorical dtype, then numeric substrings, then lexicographic.
    """
    ser = adata.obs[time_key]
    if isinstance(ser.dtype, CategoricalDtype) and ser.cat.ordered:
        labels = [str(x) for x in ser.cat.categories]
        time_float_map = {lab: float(i) for i, lab in enumerate(labels)}
        return labels, time_float_map

    labels = [str(x) for x in ser.unique()]
    labels_sorted = sorted(labels, key=_time_label_sort_key)

    time_float_map = {}
    for i, lab in enumerate(labels_sorted):
        v = _extract_first_number(lab)
        time_float_map[lab] = float(v) if v is not None else float(i)
    return labels_sorted, time_float_map


# ---------------------------
# Spatial Key Resolver
# ---------------------------
def resolve_spatial_key(adata, requested_key: str):
    """Return a valid spatial key from adata.obsm, with fallbacks."""
    obsm_keys = list(adata.obsm.keys())
    if requested_key in adata.obsm:
        return requested_key

    candidates = ["spatial", "spatial_aligned", "X_spatial", "X_umap", "X_pca"]
    for cand in candidates:
        if cand in adata.obsm:
            print(f"[resolve_spatial_key] Requested '{requested_key}' not found; using '{cand}'.")
            return cand

    raise KeyError(f"Requested spatial_key '{requested_key}' not found. Available: {obsm_keys}")


# ---------------------------
# Utilities
# ---------------------------
def parse_csv_floats(txt):
    return [float(x.strip()) for x in txt.split(",")]

def weight_schedule_t0(bio_time_scalar, tau=1.0, power=2.0):
    t = torch.as_tensor(bio_time_scalar, dtype=torch.float32, device="cpu").clone().detach()
    return float((tau / (t + tau + 1e-8)) ** power)

def euler_integrator_train_generative(
    ode_system_callable, potential_model_callable, initial_state_tensor,
    bio_time_at_initial_state, delta_bio_time_to_integrate, n_integration_steps,
    integrate_forward_in_biological_time=True
):
    initial_state_tensor = initial_state_tensor.float()
    bio_time_at_initial_state = bio_time_at_initial_state.float()
    dt_integration_tau = delta_bio_time_to_integrate / n_integration_steps if n_integration_steps > 0 else 0.0
    current_ode_state = initial_state_tensor
    
    if n_integration_steps == 0:
        return current_ode_state
        
    for step_idx in range(n_integration_steps):
        current_integration_tau = step_idx * dt_integration_tau
        if integrate_forward_in_biological_time:
            effective_bio_time_for_step = bio_time_at_initial_state + current_integration_tau
            dynamics_sign = 1.0
        else:
            effective_bio_time_for_step = bio_time_at_initial_state - current_integration_tau
            dynamics_sign = -1.0

        if effective_bio_time_for_step.ndim == 0:
            effective_bio_time_for_step_b = effective_bio_time_for_step.unsqueeze(0).repeat(current_ode_state.shape[0], 1)
        elif effective_bio_time_for_step.ndim == 1 and effective_bio_time_for_step.shape[0] == 1:
            effective_bio_time_for_step_b = effective_bio_time_for_step.repeat(current_ode_state.shape[0], 1)
        elif effective_bio_time_for_step.ndim == 1 and effective_bio_time_for_step.shape[0] == current_ode_state.shape[0]:
            effective_bio_time_for_step_b = effective_bio_time_for_step.unsqueeze(1)
        else:
            effective_bio_time_for_step_b = effective_bio_time_for_step

        spatial_dim = ode_system_callable.spatial_dim
        s_rg = current_ode_state[:, :spatial_dim].detach().requires_grad_(True)
        z_rg = current_ode_state[:, spatial_dim:].detach().requires_grad_(True)
        
        grad_U_s, grad_U_l = potential_model_callable.calculate_gradients(
            s_rg, z_rg, effective_bio_time_for_step_b, create_graph=True
        )
        
        ode_in = (current_ode_state, grad_U_s, grad_U_l)
        dstate_dt = ode_system_callable(effective_bio_time_for_step_b, ode_in)
        current_ode_state = current_ode_state + dynamics_sign * dt_integration_tau * dstate_dt
        
    return current_ode_state


@torch.no_grad()
def estimate_t0_center(adata, time_key, spatial_key, vae_model, t0_tp_str, keep_quantile=1.0, device='cpu'):
    uniq, _ = get_ordered_time_labels_and_map(adata, time_key)
    if t0_tp_str is None:
        if not uniq:
            raise RuntimeError("No timepoints found in AnnData.")
        t0_tp_str = uniq[0]
        
    ad_sub = adata[adata.obs[time_key].astype(str) == str(t0_tp_str)].copy()
    if ad_sub.n_obs == 0:
        raise RuntimeError(f"No cells at timepoint {t0_tp_str}")
        
    if 0 < keep_quantile < 1.0:
        n_keep = max(1, int(round(keep_quantile * ad_sub.n_obs)))
        sc.pp.subsample(ad_sub, n_obs=n_keep, random_state=123)
        
    X = ad_sub.X.toarray() if hasattr(ad_sub.X, "toarray") else np.asarray(ad_sub.X)
    X_t = torch.tensor(X.astype(np.float32), device=device)
    mu, _ = vae_model.vae.encoder(X_t)
    z0 = mu.mean(dim=0)
    s0 = torch.tensor(ad_sub.obsm[spatial_key], device=device, dtype=torch.float32).mean(dim=0)
    return s0, z0


def parse_args():
    parser = argparse.ArgumentParser(description="Train Generative Spatiotemporal System")

    # Core data/model args
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--time_key', type=str, default='timepoint')
    parser.add_argument('--spatial_key', type=str, default='spatial')
    parser.add_argument('--spatial_dim', type=int, default=2)
    parser.add_argument('--pretrained_vae_path', type=str, default=None)
    parser.add_argument('--vae_hidden_dims', type=str, default="128,64")
    parser.add_argument('--vae_latent_dim', type=int, default=8)
    parser.add_argument('--vae_recon_loss_type', type=str, default='gaussian')
    parser.add_argument('--vae_kl_weight', type=float, default=0.005)
    parser.add_argument('--vae_dropout_rate', type=float, default=0.1)
    parser.add_argument('--potential_time_embedding_dim', type=int, default=4)
    parser.add_argument('--potential_hidden_dims', type=str, default="32,16")
    parser.add_argument('--ode_time_embedding_dim', type=int, default=4)
    parser.add_argument('--ode_hidden_dims', type=str, default="64,32")
    parser.add_argument('--ode_damping_coeff', type=float, default=0.01)
    parser.add_argument('--ode_n_integration_steps', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--batch_size_transition', type=int, default=32)
    
    # Loss Weights
    parser.add_argument('--swd_n_projections', type=int, default=64)
    parser.add_argument('--swd_spatial_weight', type=float, default=1.0)
    parser.add_argument('--swd_latent_weight', type=float, default=1.0)
    parser.add_argument('--loss_weight_t0_distance', type=float, default=10.0)
    parser.add_argument('--loss_weight_t0_velocity_align', type=float, default=10.0)
    parser.add_argument('--loss_weight_vae_recon', type=float, default=1.0)
    parser.add_argument('--loss_weight_vae_kl', type=float, default=0.1)
    parser.add_argument('--loss_weight_ode_swd', type=float, default=1.0)
    parser.add_argument('--loss_weight_time_align', type=float, default=0.1)
    parser.add_argument('--loss_weight_force_consistency', type=float, default=0.1)
    parser.add_argument('--loss_weight_shrink', type=float, default=0.01, help="Weight for spatial radius shrinkage prior.")
    
    # T0 Anchoring
    parser.add_argument('--t0_target_source_timepoint_str', type=str, default=None)
    parser.add_argument('--t0_target_undiff_quantile', type=float, default=1.0)
    parser.add_argument('--target_t0_spatial_coord', type=str, default="auto")
    parser.add_argument('--target_t0_latent_coord', type=str, default="auto")
    parser.add_argument('--t0_time_scale', type=float, default=1.0)
    parser.add_argument('--t0_power', type=float, default=2.0)
    parser.add_argument('--t0_cutoff_time', type=float, default=1.0)

    # Dynamics scaling
    parser.add_argument('--ode_latent_speed_scale', type=float, default=0.25)
    parser.add_argument('--ode_spatial_speed_scale', type=float, default=1.0)

    # Environment
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default="")

    # Distribution loss mixing (SWD + Sinkhorn)
    parser.add_argument('--use_sinkhorn', action='store_true', help='Mix SWD with Sinkhorn divergence.')
    parser.add_argument('--alpha_swd', type=float, default=1.0, help='Mixture: Ldist = alpha*SWD + (1-alpha)*Sinkhorn.')
    parser.add_argument('--sinkhorn_p', type=int, default=2)
    parser.add_argument('--sinkhorn_blur', type=float, default=0.05)
    parser.add_argument('--sinkhorn_scaling', type=float, default=0.8)
    parser.add_argument('--loss_weight_dist', type=float, default=1.0)

    # OT alignment loss
    parser.add_argument('--use_ot_align', action='store_true', help='Enable OT barycentric alignment loss.')
    parser.add_argument('--ot_reg_epsilon', type=float, default=0.05)
    parser.add_argument('--loss_weight_ot_align', type=float, default=0.0)

    # Time-affine scaling
    parser.add_argument('--use_time_affine', action='store_true', help="Enable time-dependent affine drift x' = s(t)x + b(t).")
    parser.add_argument('--affine_basis', type=str, default='fourier', choices=['fourier', 'mlp'])
    parser.add_argument('--affine_fourier_k', type=int, default=2)
    parser.add_argument('--affine_mlp_hidden', type=int, default=32)
    parser.add_argument('--scale_prior_weight', type=float, default=5e-3)
    parser.add_argument('--scale_smooth_weight', type=float, default=5e-3)
    parser.add_argument('--bias_smooth_weight', type=float, default=5e-3)

    return parser.parse_args()


def main(args):
    # Setup Device & Seeds
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load Data
    adata = sc.read_h5ad(args.data_path)
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray().astype(np.float32)
    else:
        adata.X = adata.X.astype(np.float32)

    spatial_key = resolve_spatial_key(adata, args.spatial_key)

    # Initialize VAE
    vae = VAEWrapper(
        input_dim=adata.n_vars,
        hidden_dims=[int(d) for d in args.vae_hidden_dims.split(",")],
        latent_dim=args.vae_latent_dim,
        dropout_rate=args.vae_dropout_rate,
    ).to(device).float()

    if args.pretrained_vae_path and Path(args.pretrained_vae_path).exists():
        ckpt = torch.load(args.pretrained_vae_path, map_location=device)
        vae.load_state_dict(ckpt)
        for p in vae.parameters():
            p.requires_grad = False
        vae.eval()

    # Initialize Potential
    potential = SpatioTemporalPotential(
        spatial_dim=args.spatial_dim,
        latent_dim=args.vae_latent_dim,
        time_embedding_dim=args.potential_time_embedding_dim,
        hidden_dims=[int(d) for d in args.potential_hidden_dims.split(",")],
    ).to(device).float()

    # Initialize ODE with optional Time-Affine Drift
    time_affine_cfg = None
    if args.use_time_affine:
        time_affine_cfg = dict(
            basis=args.affine_basis,
            fourier_k=args.affine_fourier_k,
            mlp_hidden=args.affine_mlp_hidden,
            time_norm_mode="auto",
        )

    ode = TimeAwareODE(
        spatial_dim=args.spatial_dim,
        latent_dim=args.vae_latent_dim,
        time_embedding_dim=args.ode_time_embedding_dim,
        grad_potential_dim=args.spatial_dim + args.vae_latent_dim,
        hidden_dims=[int(d) for d in args.ode_hidden_dims.split(",")],
        spatial_speed_scale=args.ode_spatial_speed_scale,
        latent_speed_scale=args.ode_latent_speed_scale,
        time_affine_cfg=time_affine_cfg,
    ).to(device).float()

    time_reg = LatentTimeRegressor(latent_dim=args.vae_latent_dim).to(device).float()

    # Dataset & Loader
    unique_times_sorted, time_float_map = get_ordered_time_labels_and_map(adata, args.time_key)
    paired_ds = PairedTimeDataset(
        adata, args.time_key, spatial_key,
        unique_times_sorted, time_float_map
    )
    transition_loader = DataLoader(paired_ds, batch_size=1, shuffle=True)

    # t0 Anchors
    if args.target_t0_spatial_coord.lower() != "auto":
        s0 = torch.tensor(parse_csv_floats(args.target_t0_spatial_coord), dtype=torch.float32, device=device)
    else:
        s0, _ = estimate_t0_center(
            adata, args.time_key, spatial_key, vae,
            args.t0_target_source_timepoint_str, args.t0_target_undiff_quantile, device,
        )
        
    if args.target_t0_latent_coord.lower() != "auto":
        z0 = torch.tensor(parse_csv_floats(args.target_t0_latent_coord), dtype=torch.float32, device=device)
    else:
        _, z0 = estimate_t0_center(
            adata, args.time_key, spatial_key, vae,
            args.t0_target_source_timepoint_str, args.t0_target_undiff_quantile, device,
        )
        
    s0, z0 = s0.flatten(), z0.flatten()
    assert s0.numel() == args.spatial_dim and z0.numel() == args.vae_latent_dim

    # Optimizer
    params = list(potential.parameters()) + list(ode.parameters()) + list(time_reg.parameters())
    if any(p.requires_grad for p in vae.parameters()):
        params += list(vae.parameters())
    optimizer = optim.AdamW(params, lr=args.learning_rate)

    # Output Directory Setup
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "t0_targets.json").write_text(json.dumps({
        "t0_source_timepoint": args.t0_target_source_timepoint_str or unique_times_sorted[0],
        "s0": s0.detach().cpu().tolist(),
        "z0": z0.detach().cpu().tolist(),
    }, indent=2))

    # Shrinkage Config
    numeric_times_all = [time_float_map[t] for t in unique_times_sorted]
    LATEST_OBSERVED_TIME = max(numeric_times_all) if numeric_times_all else 1.0
    adata_latest = adata[adata.obs[args.time_key] == unique_times_sorted[-1]]
    spatial_latest = torch.tensor(adata_latest.obsm[spatial_key], device=device).float()
    initial_radius_at_latest_time = torch.sqrt(torch.sum(spatial_latest ** 2, dim=1)).mean().item()
    
    print(f"Using LATEST_OBSERVED_TIME={LATEST_OBSERVED_TIME:.2f} and initial_radius={initial_radius_at_latest_time:.2f}")

    # --- Training Loop ---
    loss_history = []
    print(f"Starting training for {args.epochs} epochs.")
    
    for epoch in range(args.epochs):
        epoch_losses = {k: 0.0 for k in [
            "total", "vae_recon", "vae_kl", "ode_swd", "time_align", "force_consistency",
            "t0_distance", "t0_velocity", "shrink", "affine"
        ]}
        n_batches = 0
        potential.train(); ode.train(); time_reg.train()
        if any(p.requires_grad for p in vae.parameters()):
            vae.train()
        else:
            vae.eval()

        pbar = tqdm(transition_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch_data in pbar:
            try:
                optimizer.zero_grad()

                # Unpack Data
                expr_t2 = batch_data["expr_k"].squeeze(0).to(device)
                spatial_t2 = batch_data["spatial_k"].squeeze(0).to(device)
                expr_t1_target = batch_data["expr_k_minus_1"].squeeze(0).to(device)
                spatial_t1_target = batch_data["spatial_k_minus_1"].squeeze(0).to(device)

                n_k, n_k_minus_1 = expr_t2.shape[0], expr_t1_target.shape[0]
                if n_k < 1 or n_k_minus_1 < 1:
                    continue
                    
                batch_size = args.batch_size_transition
                indices_k = torch.randperm(n_k)[:batch_size]
                indices_k_minus_1 = torch.randperm(n_k_minus_1)[:batch_size]

                expr_t2, spatial_t2 = expr_t2[indices_k], spatial_t2[indices_k]
                expr_t1_target, spatial_t1_target = expr_t1_target[indices_k_minus_1], spatial_t1_target[indices_k_minus_1]

                t2 = batch_data["time_k_numeric"].item()
                dt_bio = batch_data["delta_t_numeric"].item()
                t1 = t2 - dt_bio
                n = expr_t2.size(0)

                # VAE Forward
                with torch.set_grad_enabled(vae.training):
                    xhat_t2, mu_t2, logv_t2 = vae(expr_t2)
                    xhat_t1, mu_t1_target, logv_t1_target = vae(expr_t1_target)
                
                loss_recon = torch.tensor(0.0, device=device)
                loss_kl = torch.tensor(0.0, device=device)
                
                if vae.training:
                    loss_recon = F.mse_loss(xhat_t1, expr_t1_target) + F.mse_loss(xhat_t2, expr_t2)
                    kl_t1 = -0.5 * torch.mean(1 + logv_t1_target - mu_t1_target.pow(2) - logv_t1_target.exp())
                    kl_t2 = -0.5 * torch.mean(1 + logv_t2 - mu_t2.pow(2) - logv_t2.exp())
                    loss_kl = 0.5 * (kl_t1 + kl_t2)

                # Time Regression
                pred_t1 = time_reg(mu_t1_target.detach())
                pred_t2 = time_reg(mu_t2.detach())
                loss_time = F.smooth_l1_loss(pred_t1, torch.full_like(pred_t1, t1)) \
                          + F.smooth_l1_loss(pred_t2, torch.full_like(pred_t2, t2))

                # ODE Propagation (t2 -> t1)
                init_state_t2 = torch.cat([spatial_t2, mu_t2.detach()], dim=1)
                tgt_state_t1 = torch.cat([spatial_t1_target, mu_t1_target.detach()], dim=1)
                
                pred_state_t1 = euler_integrator_train_generative(
                    ode, potential, init_state_t2,
                    torch.tensor([t2], device=device), torch.tensor([dt_bio], device=device),
                    args.ode_n_integration_steps, integrate_forward_in_biological_time=False
                )

                # Distribution Loss (SWD / Sinkhorn)
                scale_vec = torch.cat([
                    torch.full((args.spatial_dim,), args.swd_spatial_weight, device=device),
                    torch.full((args.vae_latent_dim,), args.swd_latent_weight, device=device),
                ]).unsqueeze(0)
                
                loss_swd = sliced_wasserstein_distance(
                    pred_state_t1 * scale_vec, tgt_state_t1 * scale_vec, n_projections=args.swd_n_projections
                )
                
                if args.use_sinkhorn:
                    loss_sink = sinkhorn_divergence_loss(
                        pred_state_t1 * scale_vec, tgt_state_t1 * scale_vec,
                        p=args.sinkhorn_p, blur=args.sinkhorn_blur,
                        scaling=args.sinkhorn_scaling, debias=True
                    )
                    loss_dist = args.alpha_swd * loss_swd + (1.0 - args.alpha_swd) * loss_sink
                else:
                    loss_sink = torch.tensor(0.0, device=device)
                    loss_dist = loss_swd

                # OT Alignment (Optional)
                loss_ot_align = torch.tensor(0.0, device=device)
                if args.use_ot_align:
                    with torch.no_grad():
                        y_tilde = barycentric_map_entropic_ot(
                            init_state_t2.detach(), tgt_state_t1.detach(),
                            reg=args.ot_reg_epsilon, scale_vec=scale_vec
                        )
                    loss_ot_align = F.mse_loss(pred_state_t1, y_tilde)

                # Force Consistency (Velocity vs -Grad U)
                s_rg = pred_state_t1[:, :args.spatial_dim].detach().requires_grad_(True)
                z_rg = pred_state_t1[:, args.spatial_dim:].detach().requires_grad_(True)
                t1_b = torch.tensor([t1], device=device).repeat(n, 1)
                grad_U_s, grad_U_l = potential.calculate_gradients(s_rg, z_rg, t1_b, create_graph=True)
                
                ode_in_at_t1 = (pred_state_t1.detach(), grad_U_s, grad_U_l)
                vel_pred_at_t1 = ode(t1_b, ode_in_at_t1)
                vel_target_from_U = -torch.cat([grad_U_s, grad_U_l], dim=1).detach()
                loss_force = F.mse_loss(vel_pred_at_t1, vel_target_from_U)

                # t0 Convergence (Distance + Direction)
                loss_t0_distance = torch.tensor(0.0, device=device)
                if t1 <= args.t0_cutoff_time:
                    loss_t0_distance = calculate_t0_convergence_loss_smoothed(
                        pred_state_t1[:, :args.spatial_dim], pred_state_t1[:, args.spatial_dim:],
                        s0, z0, args.loss_weight_t0_distance, 0.5 * args.loss_weight_t0_distance,
                        args.loss_weight_t0_distance, 0.5 * args.loss_weight_t0_distance
                    )
                    
                with torch.no_grad():
                    unit_to_anchor = F.normalize(
                        torch.cat([s0.expand(n, -1), z0.expand(n, -1)], dim=1) - pred_state_t1, p=2, dim=1
                    )
                vel_unit = F.normalize(vel_pred_at_t1, p=2, dim=1)
                cos = (vel_unit * unit_to_anchor).sum(dim=1)
                loss_t0_vec = (1.0 - cos).mean()
                w_t = weight_schedule_t0(t1, tau=args.t0_time_scale, power=args.t0_power)
                loss_t0_velocity = args.loss_weight_t0_velocity_align * w_t * loss_t0_vec

                # Shrinkage Prior
                pred_spatial_t1 = pred_state_t1[:, :args.spatial_dim]
                radius_sq = torch.sum(pred_spatial_t1 ** 2, dim=1)
                R_target_t1 = (t1 / LATEST_OBSERVED_TIME) * initial_radius_at_latest_time
                shrink_loss_per_cell = torch.clamp(radius_sq - R_target_t1 ** 2, min=0)
                loss_shrink = torch.mean(shrink_loss_per_cell)

                # Time-Affine Regularizers
                loss_affine = torch.tensor(0.0, device=device)
                if args.use_time_affine and (
                    args.scale_prior_weight + args.scale_smooth_weight + args.bias_smooth_weight
                ) > 0:
                    tau_grid = make_uniform_tau_grid(n=9, device=device)
                    prior, smooth_log_s, smooth_b = time_affine_regularizers(ode.time_affine, tau_grid)
                    loss_affine = (
                        args.scale_prior_weight * prior
                        + args.scale_smooth_weight * smooth_log_s
                        + args.bias_smooth_weight * smooth_b
                    )

                # Total Loss
                total_loss = (
                    args.loss_weight_vae_recon * loss_recon
                    + args.loss_weight_vae_kl * loss_kl
                    + args.loss_weight_dist * loss_dist
                    + args.loss_weight_time_align * loss_time
                    + args.loss_weight_force_consistency * loss_force
                    + args.loss_weight_ot_align * loss_ot_align
                    + loss_t0_distance + loss_t0_velocity
                    + args.loss_weight_shrink * loss_shrink
                    + loss_affine
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                # Accumulate Loss Stats
                n_batches += 1
                for k, v in [
                    ("total", total_loss), ("vae_recon", loss_recon), ("vae_kl", loss_kl),
                    ("ode_swd", loss_swd), ("time_align", loss_time), ("force_consistency", loss_force),
                    ("t0_distance", loss_t0_distance), ("t0_velocity", loss_t0_velocity),
                    ("shrink", loss_shrink), ("affine", loss_affine),
                ]:
                    epoch_losses[k] += v.item()

                pbar.set_postfix({
                    "SWD": f"{loss_swd.item():.3f}",
                    "Shrink": f"{loss_shrink.item():.4f}",
                    "Affine": f"{loss_affine.item():.4f}",
                })

            except Exception as e:
                print(f"[Epoch {epoch+1}] Error in training loop: {e}")
                traceback.print_exc()

        # Log Epoch Stats
        if n_batches > 0:
            for k in epoch_losses:
                epoch_losses[k] /= n_batches
        loss_history.append({"epoch": epoch + 1, **epoch_losses})
        
        print("Epoch {}/{} | {}".format(
            epoch + 1, args.epochs,
            ", ".join([f"{k}:{v:.4f}" for k, v in epoch_losses.items()
                       if k in ["total", "ode_swd", "t0_velocity", "shrink", "affine"]])
        ))

        # Checkpoints
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            torch.save({
                "vae_state_dict": vae.state_dict(),
                "potential_state_dict": potential.state_dict(),
                "ode_state_dict": ode.state_dict(),
                "time_regressor_state_dict": time_reg.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "train_args_snapshot": vars(args),
            }, results_dir / f"system_model_epoch_{epoch+1}.pt")

    # Final Save
    final_blob = {
        "vae_state_dict": vae.state_dict(),
        "potential_state_dict": potential.state_dict(),
        "ode_state_dict": ode.state_dict(),
        "time_regressor_state_dict": time_reg.state_dict(),
        "train_args": vars(args),
    }
    torch.save(final_blob, results_dir / "system_model_final.pt")
    pd.DataFrame(loss_history).to_csv(results_dir / "loss_log.csv", index=False)
    print(f"Training complete. Saved final model to {results_dir / 'system_model_final.pt'}")


if __name__ == "__main__":
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.results_dir) / "config_train_gen_system.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    main(args)
