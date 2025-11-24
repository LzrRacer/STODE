#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import scanpy as sc
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys # Keep sys import
import pandas # Import pandas directly
import traceback

# --- Setup Project Paths and PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# --- Import Custom Modules ---
from models.vae import VAEWrapper
from data.dataset import create_dataloaders

def parse_args_train_vae():
    parser = argparse.ArgumentParser(description="Train a Variational Autoencoder (VAE).")
    # ... (all args as before) ...
    parser.add_argument("--data_path", type=str, required=True, help="Path to AnnData object (.h5ad).")
    parser.add_argument("--time_key", type=str, default="timepoint")
    parser.add_argument("--spatial_key", type=str, default="spatial")
    parser.add_argument("--hidden_dims_str", type=str, default="256,128,64")
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--recon_loss_type", type=str, default="nb", choices=["mse", "nb", "gaussian"])
    parser.add_argument("--kl_weight", type=float, default=0.001)
    parser.add_argument("--output_model_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()
    return args

def main(args):
    # --- Setup ---
    # ... (device setup as before) ...
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device and args.device.lower() in ["cuda", "cpu", "mps"]:
        device = torch.device(args.device.lower())
    elif args.device and "cuda" in args.device.lower(): 
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "config_train_vae_only.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    # --- Load Data ---
    # ... (data loading as before) ...
    print(f"Loading AnnData from: {args.data_path}")
    try:
        adata = sc.read_h5ad(args.data_path)
        if hasattr(adata.X, 'toarray'): 
            adata.X = adata.X.toarray().astype(np.float32)
        else:
            adata.X = adata.X.astype(np.float32)
        print(f"Data loaded: {adata.shape[0]} cells, {adata.n_vars} genes.")
    except Exception as e:
        print(f"Error loading data: {e}"); traceback.print_exc(); sys.exit(1)

    # --- Create DataLoaders ---
    # ... (dataloader creation as before) ...
    if args.time_key not in adata.obs:
        print(f"Warning: Time key '{args.time_key}' not found in adata.obs. Adding a dummy timepoint column.")
        adata.obs[args.time_key] = "T0"
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            adata,
            time_key=args.time_key, 
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            shuffle=True,
            random_state=args.seed
        )
        print(f"Created DataLoaders: Train batches={len(train_loader)}, Val batches={len(val_loader)}")
    except Exception as e:
        print(f"Error creating DataLoaders: {e}"); traceback.print_exc(); sys.exit(1)

    # --- Initialize Model ---
    # ... (model init as before) ...
    input_dim = adata.n_vars
    hidden_dims = [int(d) for d in args.hidden_dims_str.split(',')]
    vae_model = VAEWrapper(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=args.latent_dim,
        dropout_rate=args.dropout_rate
    ).to(device).float()
    print("VAE model initialized.")
    print(vae_model)


    # --- Optimizer ---
    optimizer = optim.AdamW(vae_model.parameters(), lr=args.learning_rate)

    # --- Training Loop ---
    # ... (training loop as before) ...
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    print(f"Starting VAE training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        vae_model.train()
        epoch_train_loss = 0.0
        epoch_train_recon_loss = 0.0
        epoch_train_kl_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        for batch_idx, data_batch_from_loader in enumerate(progress_bar):
            if isinstance(data_batch_from_loader, dict):
                data_batch = data_batch_from_loader['x'].to(device)
            else: 
                data_batch = data_batch_from_loader.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae_model(data_batch)
            loss_dict = vae_model.calculate_loss(
                data_batch, recon_batch, mu, logvar,
                loss_type=args.recon_loss_type,
                kl_weight=args.kl_weight
            )
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * data_batch.size(0)
            epoch_train_recon_loss += loss_dict['recon_loss'].item() * data_batch.size(0)
            epoch_train_kl_loss += loss_dict['kl_div'].item() * data_batch.size(0)
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Recon": f"{loss_dict['recon_loss'].item():.4f}",
                "KL": f"{loss_dict['kl_div'].item():.4f}"
            })
        avg_epoch_train_loss = epoch_train_loss / len(train_loader.dataset)
        avg_epoch_train_recon_loss = epoch_train_recon_loss / len(train_loader.dataset)
        avg_epoch_train_kl_loss = epoch_train_kl_loss / len(train_loader.dataset)
        train_loss_history.append(avg_epoch_train_loss)
        vae_model.eval()
        epoch_val_loss = 0.0
        epoch_val_recon_loss = 0.0
        epoch_val_kl_loss = 0.0
        with torch.no_grad():
            for data_batch_from_loader_val in val_loader:
                if isinstance(data_batch_from_loader_val, dict):
                    data_batch_val = data_batch_from_loader_val['x'].to(device)
                else: 
                    data_batch_val = data_batch_from_loader_val.to(device)
                recon_batch_val, mu_val, logvar_val = vae_model(data_batch_val)
                val_loss_dict = vae_model.calculate_loss(
                    data_batch_val, recon_batch_val, mu_val, logvar_val,
                    loss_type=args.recon_loss_type,
                    kl_weight=args.kl_weight
                )
                val_loss = val_loss_dict['loss']
                epoch_val_loss += val_loss.item() * data_batch_val.size(0)
                epoch_val_recon_loss += val_loss_dict['recon_loss'].item() * data_batch_val.size(0)
                epoch_val_kl_loss += val_loss_dict['kl_div'].item() * data_batch_val.size(0)
        avg_epoch_val_loss = epoch_val_loss / len(val_loader.dataset)
        avg_epoch_val_recon_loss = epoch_val_recon_loss / len(val_loader.dataset)
        avg_epoch_val_kl_loss = epoch_val_kl_loss / len(val_loader.dataset)
        val_loss_history.append(avg_epoch_val_loss)
        print(f"Epoch {epoch+1}/{args.epochs} Summary: \n"
              f"  Train Loss: {avg_epoch_train_loss:.4f} (Recon: {avg_epoch_train_recon_loss:.4f}, KL: {avg_epoch_train_kl_loss:.4f})\n"
              f"  Val Loss:   {avg_epoch_val_loss:.4f} (Recon: {avg_epoch_val_recon_loss:.4f}, KL: {avg_epoch_val_kl_loss:.4f})")
        if avg_epoch_val_loss < best_val_loss:
            best_val_loss = avg_epoch_val_loss
            torch.save(vae_model.state_dict(), args.output_model_path)
            print(f"  ✨ New best model saved to {args.output_model_path} (Val Loss: {best_val_loss:.4f})")
        if (epoch + 1) % 20 == 0:
            chkpt_path = results_dir / f"vae_checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_epoch_train_loss,
                'val_loss': avg_epoch_val_loss,
            }, chkpt_path)
            print(f"  Checkpoint saved to {chkpt_path}")

    print("Training complete.")
    print(f"Best VAE model saved to: {args.output_model_path}")

    # --- Plot Loss Curves ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), train_loss_history, label='Train Loss')
    plt.plot(range(1, args.epochs + 1), val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('VAE Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / "vae_loss_curves.png", dpi=150)
    plt.close()
    print(f"Loss curves saved to {results_dir / 'vae_loss_curves.png'}")

    # --- Save Loss History Data ---
    # Explicitly use pandas here.
    loss_data_df = pandas.DataFrame({ # Use `pandas` directly instead of `pd`
        'epoch': range(1, args.epochs + 1),
        'train_loss': train_loss_history,
        'val_loss': val_loss_history
    })
    loss_data_df.to_csv(results_dir / "vae_loss_history.csv", index=False)
    print(f"Loss history data saved to {results_dir / 'vae_loss_history.csv'}")

if __name__ == "__main__":
    args = parse_args_train_vae()
    main(args)