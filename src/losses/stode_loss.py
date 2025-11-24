# src/losses/stode_loss.py
import torch
from typing import Tuple

def zero_center_prior_log_s(log_s_values: torch.Tensor) -> torch.Tensor:
    """
    log_s_values: [T,1] values sampled on a time grid.
    Penalize deviation of mean(log s) from 0 (so s(t) ~ 1 on average).
    """
    return (log_s_values.mean(dim=0) ** 2).sum()

def smoothness_fd2(x: torch.Tensor) -> torch.Tensor:
    """
    Squared finite-difference smoothness with 2nd-order differences.
    x: [T,D] sequence along time grid.
    Returns scalar penalty.
    """
    if x.shape[0] < 3:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    d1 = x[1:] - x[:-1]      # [T-1,D]
    d2 = d1[1:] - d1[:-1]    # [T-2,D]
    return (d2 ** 2).mean()

@torch.no_grad()
def make_uniform_tau_grid(n: int, device) -> torch.Tensor:
    """[T,1] τ grid in [0,1]."""
    T = max(3, int(n))
    return torch.linspace(0.0, 1.0, T, device=device).unsqueeze(1)

def time_affine_regularizers(affine_module, tau_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (prior, smooth_log_s, smooth_b)
    """
    log_s, b = affine_module.eval_grid(tau_grid)  # [T,1], [T,S]
    prior = zero_center_prior_log_s(log_s)
    smooth_log_s = smoothness_fd2(log_s)
    smooth_b = smoothness_fd2(b)
    return prior, smooth_log_s, smooth_b
