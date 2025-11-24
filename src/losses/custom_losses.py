# src/losses/custom_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss  # pip install geomloss
import torch
import torch.nn.functional as F



class LatentTimeRegressor(nn.Module):
    """
    A simple MLP to regress biological time from a latent vector.
    """
    def __init__(self, latent_dim, hidden_dims=[64, 32], output_dim=1):
        super().__init__()
        layers = []
        current_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU()) # Using ReLU, could also be Tanh
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_dim))
        # No activation on final output for regression, or Sigmoid if time is normalized [0,1]
        # For now, assuming raw time prediction.
        self.regressor = nn.Sequential(*layers)

    def forward(self, latent_z):
        # latent_z is [batch_size, latent_dim]
        # Output should be [batch_size] or [batch_size, 1]
        time_pred = self.regressor(latent_z)
        return time_pred.squeeze(-1) # Ensure output is [batch_size] if output_dim=1

def sliced_wasserstein_distance(source_samples, target_samples, n_projections=50, p=2, device=None):
    """
    Sliced Wasserstein Distance between two sets of samples.
    Assumes source_samples and target_samples are [N, D] and [M, D] tensors.
    Args:
        source_samples (torch.Tensor): Samples from the source distribution (e.g., predicted states).
        target_samples (torch.Tensor): Samples from the target distribution (e.g., true states).
        n_projections (int): Number of random projections to use.
        p (int): Power for the L_p distance between sorted projected samples (typically 1 or 2).
        device (str, optional): Device to perform computations on. Defaults to source_samples.device.
    Returns:
        torch.Tensor: The Sliced Wasserstein Distance.
    """
    if source_samples.shape[0] == 0 or target_samples.shape[0] == 0:
        return torch.tensor(0.0, device=source_samples.device if source_samples.numel() > 0 else target_samples.device, requires_grad=True)

    if device is None:
        device = source_samples.device

    embedding_dim = source_samples.size(1)
    if embedding_dim != target_samples.size(1):
        raise ValueError(f"Embedding dimensions must match: got {embedding_dim} and {target_samples.size(1)}")

    # Generate random projections
    projections = torch.randn(embedding_dim, n_projections, device=device)
    projections = F.normalize(projections, dim=0) # Normalize columns

    # Project samples
    source_projections = source_samples @ projections
    target_projections = target_samples @ projections

    # Sort projected samples along each projection
    source_projections_sorted, _ = torch.sort(source_projections, dim=0)
    target_projections_sorted, _ = torch.sort(target_projections, dim=0)

    # Calculate L_p distance between sorted projections and average over projections
    # This part can be tricky if N != M. Standard SWD often assumes N=M or resamples.
    # For simplicity, if N!=M, we can only compare min(N,M) samples or use a method
    # that handles different numbers of samples (e.g. some forms of OT).
    # Here, let's use a simple approach that works if N=M, or truncates/pads if not.
    # A more robust way involves comparing empirical CDFs.
    # For now, assuming an approximate comparison is acceptable by matching sorted lists.

    # If batch sizes are different, this simple diff won't be ideal.
    # True SWD might involve quantile matching or other techniques.
    # Let's assume for now that we try to match distributions of similar sizes,
    # or this is an approximation.
    # If n_source != n_target, the mse/abs diff below will error or be misaligned.
    # For now, let's assume they are the same size (actual_batch_n is same for both)
    if source_projections_sorted.shape[0] != target_projections_sorted.shape[0]:
        # This can happen if source/target batches in training loop have different sizes
        # For SWD, it's common to resample to the same size or use a version tolerant to this.
        # A quick fix is to use the smaller number of samples for comparison for each projection.
        # This is a simplification.
        # print(f"Warning: Sliced Wasserstein different sample sizes: {source_projections_sorted.shape[0]} vs {target_projections_sorted.shape[0]}. Truncating.")
        min_samples = min(source_projections_sorted.shape[0], target_projections_sorted.shape[0])
        source_projections_sorted = source_projections_sorted[:min_samples, :]
        target_projections_sorted = target_projections_sorted[:min_samples, :]


    wasserstein_p_distances = torch.abs(source_projections_sorted - target_projections_sorted)
    
    if p == 1:
        loss = wasserstein_p_distances.mean()
    elif p == 2:
        loss = (wasserstein_p_distances.pow(2)).mean()
    else:
        loss = (wasserstein_p_distances.pow(p)).mean().pow(1./p) # L_p norm
        
    return loss


def sinkhorn_divergence_loss(x, y, p=2, blur=0.05, scaling=0.8, debias=True):
    """
    Sinkhorn divergence using GeomLoss. Returns a torch scalar.
    Args:
        x, y: [N, D] and [M, D] tensors (same D). Supports CUDA tensors.
        p: Wasserstein-p (usually 2).
        blur: Entropic regularization kernel width (epsilon ~ blur^2).
        scaling: Internal multi-scale factor (default 0.8).
        debias: If True, returns the Sinkhorn *divergence* (S(x,y) = OT(x,y) - 0.5*OT(x,x) - 0.5*OT(y,y)).
    """
    loss = SamplesLoss(loss="sinkhorn", p=p, blur=blur, scaling=scaling, debias=debias)
    return loss(x, y)
