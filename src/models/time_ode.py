# src/models/time_ode.py
import torch
import torch.nn as nn
from typing import Optional, Tuple


class TimeAwareODE(nn.Module):
    def __init__(
        self,
        spatial_dim,
        latent_dim,
        time_embedding_dim,
        grad_potential_dim,  
        hidden_dims,
        damping_coeff: float = 0.0,
        spatial_speed_scale: float = 1.0,
        latent_speed_scale: float = 0.25,
        time_affine_cfg: Optional[dict] = None,  
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.latent_dim = latent_dim
        self.time_embedding_dim = time_embedding_dim
        self.damping_coeff = damping_coeff  
        self.spatial_speed_scale = float(spatial_speed_scale)  
        self.latent_speed_scale = float(latent_speed_scale)    

        # Optional time-dependent affine drift (s(t)*x + b(t)) in spatial dims
        self.time_affine = (
            TimeAffine1D(spatial_dim, **time_affine_cfg) if time_affine_cfg else None
        )

        if self.time_embedding_dim > 0:
            self.time_embedder_ode = nn.Linear(1, self.time_embedding_dim)
        else:
            self.time_embedder_ode = None

        # Input: current_spatial, current_latent_z, biological_time (embedded or raw),
        #        grad_U_spatial, grad_U_latent
        input_mlp_dim = (
            spatial_dim
            + latent_dim
            + (time_embedding_dim if time_embedding_dim > 0 else (1 if time_embedding_dim == 0 else 0))
            + grad_potential_dim
        )

        layers = []
        current_dim = input_mlp_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.Tanh())
            current_dim = h_dim

        self.output_net = nn.Linear(current_dim, spatial_dim + latent_dim)
        self.mlp = nn.Sequential(*layers)

    def forward(self, biological_time_t, state_and_grads_tuple):
        """
        Calculates d(state)/d(biological_time).
        biological_time_t: Current biological time. Tensor [1] or scalar.
        state_and_grads_tuple: (current_state, grad_U_s, grad_U_l)
            current_state: Tensor [batch, spatial_dim + latent_dim]
            grad_U_s: Tensor [batch, spatial_dim], gradient of potential U w.r.t. spatial coords
            grad_U_l: Tensor [batch, latent_dim], gradient of potential U w.r.t. latent z
        """
        current_state, grad_U_spatial, grad_U_latent = state_and_grads_tuple

        current_spatial = current_state[:, :self.spatial_dim]
        current_latent_z = current_state[:, self.spatial_dim:]

        inputs = [current_spatial, current_latent_z]

        # Handle biological_time_t input -> [B,1]
        if biological_time_t.ndim == 0:  # scalar
            biological_time_t_batch = biological_time_t.unsqueeze(0).repeat(current_state.shape[0], 1)
        elif biological_time_t.ndim == 1 and biological_time_t.shape[0] == 1:  # [1]
            biological_time_t_batch = biological_time_t.repeat(current_state.shape[0], 1)
        elif biological_time_t.ndim == 1:  # [batch_size]
            biological_time_t_batch = biological_time_t.unsqueeze(1)
        else:  # [batch_size, 1]
            biological_time_t_batch = biological_time_t

        if self.time_embedder_ode is not None:
            time_emb = self.time_embedder_ode(biological_time_t_batch)
            inputs.append(time_emb)
        elif self.time_embedding_dim == 0:  # Concatenate raw time
            inputs.append(biological_time_t_batch)
        # If time_embedding_dim < 0, time is ignored by ODE dynamics net directly (but influences grad_U)

        inputs.extend([grad_U_spatial, grad_U_latent])
        combined_input = torch.cat(inputs, dim=1)

        hidden_rep = self.mlp(combined_input)
        state_derivatives = self.output_net(hidden_rep)  # [d_spatial/dt, d_latent/dt]

        # Scale the two blocks differently
        d_spatial_dt = state_derivatives[:, :self.spatial_dim] * self.spatial_speed_scale
        d_latent_dt = state_derivatives[:, self.spatial_dim:] * self.latent_speed_scale


        if self.time_affine is not None:
            tB = biological_time_t_batch  # [B,1]
            s_t, b_t = self.time_affine(tB)  # s_t: [B,1], b_t: [B, S]
            # s(t) multiplies x element-wise (scalar shared across spatial dims)
            d_spatial_dt = d_spatial_dt + s_t * current_spatial + b_t

        return torch.cat([d_spatial_dt, d_latent_dt], dim=1)


import torch
import torch.nn as nn

class GradientFlowODE(nn.Module):
    """
    Implements the dynamics described in the paper: dy/dt = -∇U(y, t).
    
    Unlike previous versions that approximated the velocity field with a neural network,
    this module strictly enforces the gradient flow dynamics derived from the potential field.
    """
    def __init__(self, spatial_dim, latent_dim):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.latent_dim = latent_dim

    def forward(self, t, state_and_grads):
        """
        Returns the time derivative of the state y = [s, z].
        
        Args:
            t: Current time (scalar or tensor).
            state_and_grads: Tuple containing:
                - current_state: The state vector y.
                - grad_U_spatial: ∇_s U (Gradient w.r.t spatial coords).
                - grad_U_latent:  ∇_z U (Gradient w.r.t latent vars).
                
        Returns:
            dy/dt: The velocity vector [-∇_s U, -∇_z U].
        """
        _, grad_U_spatial, grad_U_latent = state_and_grads
        
        d_spatial_dt = -grad_U_spatial
        d_latent_dt = -grad_U_latent
        
        return torch.cat([d_spatial_dt, d_latent_dt], dim=1)


class TimeAffine1D(nn.Module):
    """
    s(t), b(t) for spatial dims.
    Options:
      - basis='fourier': φ(t)=[sin/cos] features, log s(t)=w^T φ(t), b(t)=B φ(t)
      - basis='mlp'    : small MLP on normalized τ∈[0,1]
    Shapes:
      s(t) : scalar shared across spatial dims (broadcast)  [B,1]
      b(t) : per-dim bias                                  [B,spatial_dim]
    """
    def __init__(
        self,
        spatial_dim: int,
        basis: str = "fourier",
        fourier_k: int = 3,
        mlp_hidden: int = 32,
        time_norm_mode: str = "auto",  # 'auto' (assume τ already), or 'minmax' with (t_min,t_max)
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
    ):
        super().__init__()
        self.spatial_dim = int(spatial_dim)
        self.basis = basis.lower()
        self.time_norm_mode = time_norm_mode
        self.register_buffer(
            "t_min",
            torch.tensor([t_min if t_min is not None else 0.0], dtype=torch.float32),
        )
        self.register_buffer(
            "t_max",
            torch.tensor([t_max if t_max is not None else 1.0], dtype=torch.float32),
        )

        if self.basis == "fourier":
            self.k = int(fourier_k)
            feat_dim = 2 * self.k
            # Parameters: log s(t)=w^T φ(t); b(t)=B φ(t)
            self.w_log_s = nn.Linear(feat_dim, 1, bias=False)
            self.B_bias = nn.Linear(feat_dim, self.spatial_dim, bias=True)
        elif self.basis == "mlp":
            self.net = nn.Sequential(
                nn.Linear(1, mlp_hidden),
                nn.Tanh(),
                nn.Linear(mlp_hidden, mlp_hidden),
                nn.Tanh(),
            )
            self.head_log_s = nn.Linear(mlp_hidden, 1)
            self.head_b = nn.Linear(mlp_hidden, self.spatial_dim)
        else:
            raise ValueError(f"Unknown basis '{basis}'")

    def _normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        # Map to τ∈[0,1] if needed
        if self.time_norm_mode == "auto":
            return t
        # min-max
        return (t - self.t_min) / (self.t_max - self.t_min + 1e-8)

    def _fourier_features(self, tau: torch.Tensor) -> torch.Tensor:
        # tau: [B,1] → φ: [B, 2k]
        freqs = torch.arange(1, self.k + 1, device=tau.device, dtype=tau.dtype).view(1, -1)
        x = 2.0 * torch.pi * tau @ freqs  # [B,k]
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

    @torch.no_grad()
    def eval_grid(self, tau_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Utility for regularizers/inspection.
        Returns log_s(tau) [T,1], b(tau) [T,spatial_dim]
        """
        log_s, b = self._forward_impl(tau_grid)
        return log_s, b

    def forward(self, t_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_s, b = self._forward_impl(t_in)
        s = torch.exp(log_s)  # positivity via log s
        return s, b

    def _forward_impl(self, t_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # t_in shape: [B,1] (works with other shapes as long as last dim=1)
        tau = self._normalize_time(t_in.float())
        if self.basis == "fourier":
            phi = self._fourier_features(tau)  # [B,2k]
            log_s = self.w_log_s(phi)          # [B,1]
            b = self.B_bias(phi)               # [B,S]
        else:
            h = self.net(tau)                  # [B,H]
            log_s = self.head_log_s(h)         # [B,1]
            b = self.head_b(h)                 # [B,S]
        return log_s, b
