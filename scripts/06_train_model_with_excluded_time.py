#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import scanpy as sc
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os
import traceback

# --- Setup Project Paths and PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# --- Import Custom Modules ---
from models.vae import VAEWrapper
from models.potential_field import SpatioTemporalPotential
from models.time_ode import TimeAwareODE
from data.dataset import SingleTimeDataset
from losses.custom_losses import sliced_wasserstein_distance, LatentTimeRegressor
from utils.training_helpers import calculate_t0_convergence_loss_smoothed

# --- Helper Functions ---
def reparameterize(mu, logvar):
    """Reparameterization trick: z = mu + sigma * epsilon"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def time_to_float_local(t_val):
    """Convert time strings like 'E9.5' to numeric values like 9.5"""
    if isinstance(t_val, str) and t_val.startswith("E"):
        try:
            return float(t_val[1:])
        except ValueError:
            print(f"Warning: Could not convert time string '{t_val}' to float. Returning as is.")
            return t_val
    try:
        return float(t_val)
    except ValueError:
        print(f"Warning: Could not convert time value '{t_val}' to float. Returning as is.")
        return t_val

# --- Custom Dataset for Time Transitions ---
class TimeTransitionDataset(Dataset):
    """Dataset for pairs of consecutive timepoints"""
    def __init__(self, adata, time_key, spatial_key, time_str_to_numeric_map):
        self.adata = adata
        self.time_key = time_key
        self.spatial_key = spatial_key
        self.time_str_to_numeric_map = time_str_to_numeric_map
        
        # Create pairs of consecutive timepoints
        unique_times = sorted(self.time_str_to_numeric_map.keys(), 
                            key=lambda x: self.time_str_to_numeric_map[x])
        
        self.time_pairs = []
        for i in range(len(unique_times) - 1):
            t1, t2 = unique_times[i], unique_times[i + 1]
            self.time_pairs.append((t1, t2))
        
        print(f"Created {len(self.time_pairs)} time transition pairs: {self.time_pairs}")
    
    def __len__(self):
        return len(self.time_pairs)
    
    def __getitem__(self, idx):
        t1_str, t2_str = self.time_pairs[idx]
        
        # Get cells from each timepoint
        cells_t1 = self.adata[self.adata.obs[self.time_key].astype(str) == t1_str]
        cells_t2 = self.adata[self.adata.obs[self.time_key].astype(str) == t2_str]
        
        # Sample random cells from each timepoint
        n_samples = min(len(cells_t1), len(cells_t2), 100)  # Limit for memory
        
        if n_samples < 1:
            raise ValueError(f"Not enough cells in timepoints {t1_str} or {t2_str}")
        
        # Random sampling
        idx_t1 = np.random.choice(len(cells_t1), n_samples, replace=False)
        idx_t2 = np.random.choice(len(cells_t2), n_samples, replace=False)
        
        # Extract data
        expr_t1 = cells_t1.X[idx_t1].toarray() if hasattr(cells_t1.X, 'toarray') else cells_t1.X[idx_t1]
        expr_t2 = cells_t2.X[idx_t2].toarray() if hasattr(cells_t2.X, 'toarray') else cells_t2.X[idx_t2]
        
        spatial_t1 = cells_t1.obsm[self.spatial_key][idx_t1]
        spatial_t2 = cells_t2.obsm[self.spatial_key][idx_t2]
        
        return {
            'expr_t1': torch.tensor(expr_t1, dtype=torch.float32),
            'expr_t2': torch.tensor(expr_t2, dtype=torch.float32),
            'spatial_t1': torch.tensor(spatial_t1, dtype=torch.float32),
            'spatial_t2': torch.tensor(spatial_t2, dtype=torch.float32),
            'time_t1_numeric_scalar': self.time_str_to_numeric_map[t1_str],
            'time_t2_numeric_scalar': self.time_str_to_numeric_map[t2_str]
        }

def euler_integrator_train(ode_func, potential_func, initial_state, t_start_bio, t_end_bio, n_steps, is_forward):
    """Simplified Euler integrator for training"""
    current_state = initial_state.clone().detach().requires_grad_(True)
    dt = (t_end_bio - t_start_bio) / n_steps
    time_direction_sign = 1.0 if is_forward else -1.0

    for step in range(n_steps):
        current_t_bio = t_start_bio + step * dt * time_direction_sign
        
        # Get spatial and latent components
        s_coords_rg = current_state[:, :potential_func.spatial_dim].clone().detach().requires_grad_(True)
        l_coords_rg = current_state[:, potential_func.spatial_dim:].clone().detach().requires_grad_(True)

        # Calculate gradients
        spatial_grad, latent_grad = potential_func.calculate_gradients(
            s_coords_rg, l_coords_rg, 
            torch.tensor([float(current_t_bio)], device=current_state.device, dtype=torch.float32),
            create_graph=True
        )
        
        if spatial_grad is None or latent_grad is None:
            raise ValueError("Gradients from potential field are None.")

        # Prepare ODE input
        ode_input_state_grads_tuple = (
            current_state.clone().detach(),
            spatial_grad,
            latent_grad
        )
        
        # Get derivatives from ODE
        d_state_dt = ode_func(torch.tensor([float(current_t_bio)], device=current_state.device, dtype=torch.float32), ode_input_state_grads_tuple)

        # Euler step
        current_state = current_state + d_state_dt * dt * time_direction_sign
        current_state = current_state.clone().detach().requires_grad_(True)

    return current_state, None  # Return final state only for simplicity

def parse_args_train_excluded():
    parser = argparse.ArgumentParser(description="Train the generative system, optionally excluding a specific timepoint.")
    
    # Data Args
    parser.add_argument("--data_path", type=str, required=True, help="Path to the AnnData object (.h5ad).")
    parser.add_argument("--time_key", type=str, default="timepoint", help="Key in adata.obs for time information.")
    parser.add_argument("--spatial_key", type=str, default="spatial", help="Key in adata.obsm for spatial coordinates.")
    parser.add_argument("--exclude_timepoint_str", type=str, default=None, help="String representation of the timepoint to exclude from training (e.g., 'E10.5').")

    # Model Architecture Args
    parser.add_argument("--spatial_dim", type=int, default=2)
    parser.add_argument("--vae_hidden_dims", type=str, default="128,64")
    parser.add_argument("--vae_latent_dim", type=int, default=8)
    parser.add_argument("--vae_dropout_rate", type=float, default=0.1)
    parser.add_argument("--potential_time_embedding_dim", type=int, default=4)
    parser.add_argument("--potential_hidden_dims", type=str, default="32,16")
    parser.add_argument("--ode_time_embedding_dim", type=int, default=4)
    parser.add_argument("--ode_hidden_dims", type=str, default="64,32")
    parser.add_argument("--ode_damping_coeff", type=float, default=0.01)
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size_transition", type=int, default=32)
    parser.add_argument("--ode_n_integration_steps", type=int, default=3)
    
    # Loss Weights
    parser.add_argument("--vae_recon_loss_type", type=str, default="nb", choices=["mse", "nb"])
    parser.add_argument("--vae_kl_weight", type=float, default=0.005)
    parser.add_argument("--loss_weight_vae_recon", type=float, default=1.0)
    parser.add_argument("--loss_weight_vae_kl", type=float, default=0.1)
    parser.add_argument("--loss_weight_ode_swd", type=float, default=1.0)
    parser.add_argument("--loss_weight_time_align", type=float, default=0.1)
    parser.add_argument("--loss_weight_force_consistency", type=float, default=0.1)
    parser.add_argument("--loss_weight_t0_spatial_mean", type=float, default=0.5)
    parser.add_argument("--loss_weight_t0_spatial_variance", type=float, default=0.5)
    parser.add_argument("--loss_weight_t0_latent_mean", type=float, default=0.5)
    parser.add_argument("--loss_weight_t0_latent_variance", type=float, default=0.5)

    parser.add_argument("--target_t0_spatial_coord", type=str, default="0.0,0.0")
    parser.add_argument("--target_t0_latent_coord", type=str, default=None)

    parser.add_argument('--swd_spatial_weight', type=float, default=5.0)
    parser.add_argument('--swd_latent_weight', type=float, default=1.0)
    parser.add_argument('--swd_n_projections', type=int, default=50)


    # Output & Misc
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    
    return parser.parse_args()

def main(args):
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    if device.type == 'cuda': 
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training arguments
    with open(output_dir / "train_args_excluded_time_model.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load and Prepare Data
    print(f"Loading data from: {args.data_path}")
    adata = sc.read_h5ad(args.data_path)
    
    # Exclude timepoint if specified
    if args.exclude_timepoint_str:
        print(f"Attempting to exclude timepoint: {args.exclude_timepoint_str}")
        original_obs_count = adata.n_obs
        adata_obs_time_str = adata.obs[args.time_key].astype(str)
        adata = adata[adata_obs_time_str != str(args.exclude_timepoint_str)].copy()
        print(f"  Data filtered. Original N_obs: {original_obs_count}, After excluding '{args.exclude_timepoint_str}': {adata.n_obs}")
        if adata.n_obs == 0:
            print(f"Error: No data remaining after excluding timepoint {args.exclude_timepoint_str}. Exiting.")
            return

    print(f"Data shape for training: {adata.shape}")

    # Convert timepoints to numeric
    unique_times_str = sorted(adata.obs[args.time_key].astype(str).unique(), key=time_to_float_local)
    time_str_to_numeric_map = {t_str: time_to_float_local(t_str) for t_str in unique_times_str}
    unique_numeric_times = sorted(list(time_str_to_numeric_map.values()))
    
    if len(unique_numeric_times) < 2:
        print("Error: Need at least two distinct timepoints for training transitions. Exiting.")
        return

    # Create datasets
    time_transition_dataset = TimeTransitionDataset(adata, args.time_key, args.spatial_key, time_str_to_numeric_map)
    
    if len(time_transition_dataset) == 0:
        print("Error: TimeTransitionDataset is empty. Check timepoint pairing. Exiting.")
        return
    
    transition_loader = DataLoader(time_transition_dataset, batch_size=1, shuffle=True)  # Each item is a batch
    
    # VAE reconstruction dataset
    all_cells_dataset = SingleTimeDataset(
        X=adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
        spatial_coords=adata.obsm[args.spatial_key].astype(np.float32),
        transform=None
    )
    vae_recon_loader = DataLoader(all_cells_dataset, batch_size=args.batch_size_transition * 2, shuffle=True)

    # Initialize Models
    vae_input_dim = adata.n_vars
    vae = VAEWrapper(
        input_dim=vae_input_dim,
        hidden_dims=[int(d) for d in args.vae_hidden_dims.split(',')],
        latent_dim=args.vae_latent_dim,
        dropout_rate=args.vae_dropout_rate
    ).to(device).float()

    potential_field = SpatioTemporalPotential(
        spatial_dim=args.spatial_dim,
        latent_dim=args.vae_latent_dim,
        time_embedding_dim=args.potential_time_embedding_dim,
        hidden_dims=[int(d) for d in args.potential_hidden_dims.split(',')],
        output_scalar=True
    ).to(device).float()

    ode_system = TimeAwareODE(
        spatial_dim=args.spatial_dim,
        latent_dim=args.vae_latent_dim,
        time_embedding_dim=args.ode_time_embedding_dim,
        grad_potential_dim=args.spatial_dim + args.vae_latent_dim,
        hidden_dims=[int(d) for d in args.ode_hidden_dims.split(',')],
        damping_coeff=args.ode_damping_coeff
    ).to(device).float()

    time_regressor = LatentTimeRegressor(
        latent_dim=args.vae_latent_dim,
        hidden_dims=[32, 16],
        output_dim=1
    ).to(device).float()

    # Optimizer
    params = list(vae.parameters()) + list(potential_field.parameters()) + \
             list(ode_system.parameters()) + list(time_regressor.parameters())
    optimizer = optim.AdamW(params, lr=args.learning_rate)

    # Target t0 coordinates
    target_t0_spatial_mean_tensor = torch.tensor([float(x) for x in args.target_t0_spatial_coord.split(',')], device=device)
    if args.target_t0_latent_coord and args.target_t0_latent_coord.lower() != 'none':
        target_t0_latent_mean_tensor = torch.tensor([float(x) for x in args.target_t0_latent_coord.split(',')], device=device)
    else:
        target_t0_latent_mean_tensor = None

    # Training Loop
    loss_history = {
        "total": [], "vae_recon": [], "vae_kl": [], "ode_swd": [],
        "time_align": [], "force_consistency": [], "t0_convergence": []
    }
    print(f"Starting training loop for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        epoch_loss_sum = {k: 0.0 for k in loss_history}
        num_batches_processed = 0
        
        ode_system.train(); potential_field.train(); vae.train(); time_regressor.train()
        
        # Process transition pairs
        progress_bar_transitions = tqdm(transition_loader, desc=f"Epoch {epoch+1}/{args.epochs} Transitions", leave=False)
        for batch_idx, batch_data_transition in enumerate(progress_bar_transitions):
            optimizer.zero_grad()
            
            expr_t1 = batch_data_transition['expr_t1'].squeeze(0).to(device).float()
            expr_t2 = batch_data_transition['expr_t2'].squeeze(0).to(device).float()
            spatial_t1 = batch_data_transition['spatial_t1'].squeeze(0).to(device).float()
            spatial_t2 = batch_data_transition['spatial_t2'].squeeze(0).to(device).float()
            bio_time_t1_scalar = float(batch_data_transition['time_t1_numeric_scalar'])
            bio_time_t2_scalar = float(batch_data_transition['time_t2_numeric_scalar'])
            
            current_batch_size = expr_t1.size(0)
            num_batches_processed += 1

            # VAE Forward pass to get latent representations
            # Check VAE return format dynamically
            vae_result_t1 = vae(expr_t1)
            
            if len(vae_result_t1) == 3:
                # Format: (recon, mu, logvar)
                _, z_t1_mu, z_t1_logvar = vae_result_t1
            elif len(vae_result_t1) == 4:
                # Format: (recon, mu, logvar, _)
                _, z_t1_mu, z_t1_logvar, _ = vae_result_t1
            else:
                raise ValueError(f"Unexpected VAE return format: {len(vae_result_t1)} values")
            
            z_t1 = reparameterize(z_t1_mu, z_t1_logvar)
            
            vae_result_t2 = vae(expr_t2)
            if len(vae_result_t2) == 3:
                _, z_t2_mu_target, z_t2_logvar_target = vae_result_t2
            elif len(vae_result_t2) == 4:
                _, z_t2_mu_target, z_t2_logvar_target, _ = vae_result_t2
            else:
                raise ValueError(f"Unexpected VAE return format: {len(vae_result_t2)} values")
                
            z_t2_target = reparameterize(z_t2_mu_target, z_t2_logvar_target)

            # ODE Integration
            initial_ode_state_t1 = torch.cat([spatial_t1, z_t1], dim=1).detach().requires_grad_(True)
            
            predicted_final_state_t2, _ = euler_integrator_train(
                ode_system, potential_field, initial_ode_state_t1,
                float(bio_time_t1_scalar), float(bio_time_t2_scalar),
                args.ode_n_integration_steps,
                is_forward=(bio_time_t2_scalar > bio_time_t1_scalar)
            )
            
            predicted_spatial_t2 = predicted_final_state_t2[:, :args.spatial_dim]
            predicted_latent_t2 = predicted_final_state_t2[:, args.spatial_dim:]

            # Calculate Losses
            # loss_ode_swd_spatial = sliced_wasserstein_distance(predicted_spatial_t2, spatial_t2.detach(), 50, 1, device)
            # loss_ode_swd_latent = sliced_wasserstein_distance(predicted_latent_t2, z_t2_target.detach(), 50, 1, device)
            # loss_ode_swd = loss_ode_swd_spatial + loss_ode_swd_latent

            P = args.swd_n_projections
            loss_ode_swd_spatial = sliced_wasserstein_distance(predicted_spatial_t2, spatial_t2.detach(), P, 1, device)
            loss_ode_swd_latent  = sliced_wasserstein_distance(predicted_latent_t2,  z_t2_target.detach(), P, 1, device)
            loss_ode_swd = args.swd_spatial_weight * loss_ode_swd_spatial + \
                        args.swd_latent_weight  * loss_ode_swd_latent

            epoch_loss_sum["ode_swd"] += loss_ode_swd.item() * current_batch_size

            # Time alignment loss
            predicted_time_t1 = time_regressor(z_t1)
            predicted_time_t2 = time_regressor(z_t2_target)
            
            target_bio_time_t1_tensor = torch.full_like(predicted_time_t1, float(bio_time_t1_scalar), device=device, dtype=torch.float32)
            target_bio_time_t2_tensor = torch.full_like(predicted_time_t2, float(bio_time_t2_scalar), device=device, dtype=torch.float32)
            
            loss_time_align = torch.nn.functional.mse_loss(predicted_time_t1, target_bio_time_t1_tensor) + \
                               torch.nn.functional.mse_loss(predicted_time_t2, target_bio_time_t2_tensor)
            epoch_loss_sum["time_align"] += loss_time_align.item() * current_batch_size
            
            # Placeholder for force consistency
            loss_force_consistency = torch.tensor(0.0, device=device)
            epoch_loss_sum["force_consistency"] += loss_force_consistency.item() * current_batch_size

            # T0 convergence loss (simplified)
            loss_t0_conv = torch.tensor(0.0, device=device)
            epoch_loss_sum["t0_convergence"] += loss_t0_conv.item() * current_batch_size
            
            total_loss = (args.loss_weight_ode_swd * loss_ode_swd +
                          args.loss_weight_time_align * loss_time_align +
                          args.loss_weight_force_consistency * loss_force_consistency +
                          loss_t0_conv)
            
            total_loss.backward()
            optimizer.step()
            
            progress_bar_transitions.set_postfix({
                "ODE_SWD": f"{loss_ode_swd.item():.3f}",
                "TimeAlign": f"{loss_time_align.item():.3f}"
            })

        # VAE Reconstruction
        progress_bar_vae = tqdm(vae_recon_loader, desc=f"Epoch {epoch+1}/{args.epochs} VAE Recon", leave=False)
        for batch_data_vae in progress_bar_vae:
            optimizer.zero_grad()
            
            # Debug: Check what keys are available in the batch
            if epoch == 0:  # Only print on first epoch
                print(f"DEBUG: VAE batch keys: {list(batch_data_vae.keys()) if isinstance(batch_data_vae, dict) else 'Not a dict'}")
            
            # Handle different possible key names
            if isinstance(batch_data_vae, dict):
                if 'X' in batch_data_vae:
                    expr_batch = batch_data_vae['X'].to(device).float()
                elif 'expression' in batch_data_vae:
                    expr_batch = batch_data_vae['expression'].to(device).float()
                elif 'data' in batch_data_vae:
                    expr_batch = batch_data_vae['data'].to(device).float()
                else:
                    # Try the first tensor value
                    first_key = list(batch_data_vae.keys())[0]
                    expr_batch = batch_data_vae[first_key].to(device).float()
                    if epoch == 0:
                        print(f"DEBUG: Using key '{first_key}' for expression data")
            else:
                # If it's just a tensor (not a dict)
                expr_batch = batch_data_vae.to(device).float()
            
            # Handle VAE return format dynamically
            vae_result = vae(expr_batch)
            if len(vae_result) == 3:
                recon_expr, mu, logvar = vae_result
            elif len(vae_result) == 4:
                recon_expr, mu, logvar, _ = vae_result
            else:
                raise ValueError(f"Unexpected VAE return format: {len(vae_result)} values")
            
            # Debug VAE loss calculation on first epoch
            if epoch == 0:
                loss_result = vae.calculate_loss(
                    expr_batch, recon_expr, mu, logvar, 
                    loss_type=args.vae_recon_loss_type, 
                    kl_weight=args.vae_kl_weight
                )
                print(f"DEBUG: VAE calculate_loss returned {len(loss_result) if hasattr(loss_result, '__len__') else 'scalar'} values")
                print(f"DEBUG: Loss result type: {type(loss_result)}")
                if isinstance(loss_result, dict):
                    print(f"DEBUG: Loss result keys: {list(loss_result.keys())}")
                
            loss_result = vae.calculate_loss(
                expr_batch, recon_expr, mu, logvar, 
                loss_type=args.vae_recon_loss_type, 
                kl_weight=args.vae_kl_weight
            )
            
            # Handle dictionary return format
            if isinstance(loss_result, dict):
                # We now know the keys are: ['loss', 'recon_loss', 'kl_div']
                loss_vae_r = loss_result.get('recon_loss', loss_result.get('reconstruction_loss', loss_result.get('recon', list(loss_result.values())[0])))
                loss_vae_k = loss_result.get('kl_div', loss_result.get('kl_loss', loss_result.get('kl', torch.tensor(0.0, device=device))))
                loss_vae_recon_kl = loss_result.get('loss', loss_result.get('total_loss', loss_result.get('total', loss_vae_r + loss_vae_k)))
            elif isinstance(loss_result, (tuple, list)):
                if len(loss_result) == 3:
                    loss_vae_recon_kl, loss_vae_r, loss_vae_k = loss_result
                elif len(loss_result) == 2:
                    loss_vae_r, loss_vae_k = loss_result
                    loss_vae_recon_kl = loss_vae_r + loss_vae_k
                else:
                    # Fallback - use first value as reconstruction loss
                    loss_vae_r = loss_result[0]
                    loss_vae_k = torch.tensor(0.0, device=device)
                    loss_vae_recon_kl = loss_vae_r
            else:
                # Single value returned
                loss_vae_recon_kl = loss_result
                loss_vae_r = loss_result
                loss_vae_k = torch.tensor(0.0, device=device)
            
            # Ensure scalar tensors - improved check for tensor dimensions
            if hasattr(loss_vae_r, 'dim') and loss_vae_r.dim() > 0 and loss_vae_r.numel() > 1:
                loss_vae_r = loss_vae_r.mean()  # Take mean if it's a multi-element tensor
            if hasattr(loss_vae_k, 'dim') and loss_vae_k.dim() > 0 and loss_vae_k.numel() > 1:
                loss_vae_k = loss_vae_k.mean()
            
            final_vae_loss = args.loss_weight_vae_recon * loss_vae_r + \
                             args.loss_weight_vae_kl * loss_vae_k 
            
            final_vae_loss.backward()
            optimizer.step()

            epoch_loss_sum["vae_recon"] += loss_vae_r.item() * expr_batch.size(0)
            epoch_loss_sum["vae_kl"] += loss_vae_k.item() * expr_batch.size(0)
            progress_bar_vae.set_postfix({
                "VAE_Recon": f"{loss_vae_r.item():.3f}",
                "VAE_KL": f"{loss_vae_k.item():.3f}"
            })

        # Log epoch losses
        num_total_cells_for_vae = len(all_cells_dataset)
        num_total_transition_samples = len(time_transition_dataset)

        loss_history["vae_recon"].append(epoch_loss_sum["vae_recon"] / num_total_cells_for_vae if num_total_cells_for_vae > 0 else 0)
        loss_history["vae_kl"].append(epoch_loss_sum["vae_kl"] / num_total_cells_for_vae if num_total_cells_for_vae > 0 else 0)
        loss_history["ode_swd"].append(epoch_loss_sum["ode_swd"] / num_total_transition_samples if num_total_transition_samples > 0 else 0)
        loss_history["time_align"].append(epoch_loss_sum["time_align"] / num_total_transition_samples if num_total_transition_samples > 0 else 0)
        loss_history["force_consistency"].append(epoch_loss_sum["force_consistency"] / num_total_transition_samples if num_total_transition_samples > 0 else 0)
        loss_history["t0_convergence"].append(epoch_loss_sum["t0_convergence"] / num_total_transition_samples if num_total_transition_samples > 0 else 0)

        avg_total_loss_epoch = sum([
            loss_history["vae_recon"][-1], loss_history["vae_kl"][-1],
            loss_history["ode_swd"][-1], loss_history["time_align"][-1],
            loss_history["force_consistency"][-1], loss_history["t0_convergence"][-1]
        ]) / 6
        loss_history["total"].append(avg_total_loss_epoch)

        print(f"Epoch {epoch+1}/{args.epochs} Summary: "
              f"TOTAL: {loss_history['total'][-1]:.4f}, "
              f"VAE_RECON: {loss_history['vae_recon'][-1]:.4f}, VAE_KL: {loss_history['vae_kl'][-1]:.4f}, "
              f"ODE_SWD: {loss_history['ode_swd'][-1]:.4f}, TIME_ALIGN: {loss_history['time_align'][-1]:.4f}")

    # Save final model and loss history
    print("Training complete. Saving final model and history.")
    torch.save({
        'vae_state_dict': vae.state_dict(),
        'potential_state_dict': potential_field.state_dict(),
        'ode_state_dict': ode_system.state_dict(),
        'time_regressor_state_dict': time_regressor.state_dict(),
        'train_args': vars(args)
    }, output_dir / "system_model_final.pt")
    print(f"Final model saved to {output_dir / 'system_model_final.pt'}")

    # Plot loss curves
    fig, axs = plt.subplots(len(loss_history), 1, figsize=(10, 3 * len(loss_history)), sharex=True)
    if len(loss_history) == 1:
        axs = [axs]
    
    for i, (loss_name, values) in enumerate(loss_history.items()):
        axs[i].plot(values, label=loss_name)
        axs[i].set_ylabel(loss_name)
        axs[i].legend()
        axs[i].grid(True)
    axs[-1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png")
    plt.close(fig)
    print(f"Loss curves saved to {output_dir / 'loss_curves.png'}")

    # Save loss history as JSON
    with open(output_dir / "loss_history.json", 'w') as f:
        json.dump(loss_history, f, indent=2)

if __name__ == "__main__":
    args = parse_args_train_excluded()
    main(args)