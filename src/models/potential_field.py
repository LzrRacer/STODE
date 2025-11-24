# src/models/potential_field.py
import torch
import torch.nn as nn

class SpatioTemporalPotential(nn.Module):
    def __init__(self, spatial_dim, latent_dim, time_embedding_dim, hidden_dims, output_scalar=True):
        super().__init__()
        self.spatial_dim = int(spatial_dim)
        self.latent_dim = int(latent_dim)
        self.time_embedding_dim = int(time_embedding_dim)

        # Time embedding layer
        if self.time_embedding_dim > 0:
            self.time_embedder = nn.Linear(1, self.time_embedding_dim)
        else:
            self.time_embedder = None

        # Calculate MLP input dimension
        current_mlp_input_dim = self.spatial_dim + self.latent_dim
        if self.time_embedding_dim > 0:
            current_mlp_input_dim += self.time_embedding_dim
        elif self.time_embedding_dim == 0:  # direct time concatenation
            current_mlp_input_dim += 1
        
        # Build MLP layers
        layers = []
        if current_mlp_input_dim > 0:
            current_processing_dim = current_mlp_input_dim
            if hidden_dims:
                for h_dim_str in hidden_dims:
                    h_dim = int(h_dim_str)
                    if h_dim <= 0:
                        raise ValueError("Hidden dimensions must be positive.")
                    layers.append(nn.Linear(current_processing_dim, h_dim))
                    layers.append(nn.Tanh())
                    current_processing_dim = h_dim
            
            if output_scalar:
                layers.append(nn.Linear(current_processing_dim, 1))
            else:
                layers.append(nn.Linear(current_processing_dim, current_processing_dim))
            
            self.mlp = nn.Sequential(*layers)
        else:
            # Fallback for edge case
            if output_scalar:
                self.mlp = nn.Parameter(torch.randn(1))
            else:
                self.mlp = nn.Sequential()

    def forward(self, spatial_coords_rg, latent_z_rg, biological_time_rg):
        """
        Forward pass through the potential field.
        CRITICAL: This method must preserve gradients for spatial_coords_rg and latent_z_rg
        """
        # Ensure inputs are properly prepared for gradient computation
        batch_size = spatial_coords_rg.shape[0]
        device = spatial_coords_rg.device
        
        # Collect inputs for concatenation
        inputs_to_cat = []
        
        if self.spatial_dim > 0:
            inputs_to_cat.append(spatial_coords_rg)
        if self.latent_dim > 0:
            inputs_to_cat.append(latent_z_rg)

        # Process biological time
        if isinstance(biological_time_rg, (int, float)):
            bt_processed = torch.tensor([biological_time_rg], device=device, dtype=torch.float32)
        else:
            bt_processed = biological_time_rg.clone().float()
        
        # Ensure bt_processed has correct shape [batch_size, 1]
        if bt_processed.ndim == 0:  # scalar
            bt_processed = bt_processed.unsqueeze(0).repeat(batch_size, 1)
        elif bt_processed.ndim == 1:
            if bt_processed.shape[0] == 1 and batch_size > 1:
                bt_processed = bt_processed.repeat(batch_size, 1)
            elif bt_processed.shape[0] == batch_size:
                bt_processed = bt_processed.unsqueeze(1)
            else:
                # Handle mismatch
                bt_processed = bt_processed[0].unsqueeze(0).repeat(batch_size, 1)
        
        bt_processed = bt_processed.to(device)

        # Add time to inputs
        if self.time_embedder is not None:
            # Use time embedding
            time_emb = self.time_embedder(bt_processed)
            inputs_to_cat.append(time_emb)
        elif self.time_embedding_dim == 0:
            # Concatenate raw time
            inputs_to_cat.append(bt_processed)
        # If time_embedding_dim < 0, time is ignored

        # Handle case where no inputs to concatenate
        if not inputs_to_cat:
            if hasattr(self, 'mlp') and isinstance(self.mlp, nn.Parameter):
                return self.mlp.expand(batch_size, -1)
            else:
                # Return zeros that require grad if any parameters exist
                has_params_requiring_grad = any(p.requires_grad for p in self.parameters())
                return torch.zeros(batch_size, 1, device=device, requires_grad=has_params_requiring_grad)

        # Concatenate all inputs
        combined_input = torch.cat(inputs_to_cat, dim=1)
        
        # Forward pass through MLP
        potential = self.mlp(combined_input)
        
        return potential

    def calculate_gradients(self, spatial_coords, latent_z, biological_time, create_graph=False):
        """
        Calculate gradients of the potential field with respect to spatial and latent coordinates.
        
        FIXED VERSION: Ensures proper gradient computation during simulation.
        """
        # Ensure we're in a gradient-enabled context
        original_grad_mode = torch.is_grad_enabled()
        
        try:
            # Enable gradients for this computation
            torch.set_grad_enabled(True)
            
            # Create leaf tensors that require gradients
            # CRITICAL FIX: Ensure these are properly detached and require gradients
            s_coords_rg = spatial_coords.clone().detach().requires_grad_(True)
            l_coords_rg = latent_z.clone().detach().requires_grad_(True)
            
            # Biological time doesn't need gradients (it's a parameter, not a variable)
            t_bio_no_grad = biological_time.clone().detach().float()
            
            # CRITICAL FIX: Ensure the model is in the right state for gradient computation
            # Temporarily set model to train mode to ensure proper gradient flow
            was_training = self.training
            self.train()
            
            try:
                # Forward pass - this MUST produce a tensor that requires gradients
                potential_energy = self.forward(s_coords_rg, l_coords_rg, t_bio_no_grad)
                
                # Verify that we have gradients
                if not potential_energy.requires_grad:
                    print(f"EMERGENCY FIX: potential_energy doesn't require grad. Forcing gradient computation...")
                    # This should not happen with the fixes above, but as emergency fallback:
                    # Ensure all model parameters require gradients
                    for param in self.parameters():
                        if not param.requires_grad:
                            print(f"  Setting parameter to require_grad: {param.shape}")
                            param.requires_grad_(True)
                    
                    # Try forward pass again
                    potential_energy = self.forward(s_coords_rg, l_coords_rg, t_bio_no_grad)
                    
                    if not potential_energy.requires_grad:
                        print("  CRITICAL: Still no gradients. Returning zero gradients.")
                        return torch.zeros_like(s_coords_rg), torch.zeros_like(l_coords_rg)
                
                # Successful gradient computation
                if potential_energy.nelement() == 0 or potential_energy.shape[0] == 0:
                    return torch.zeros_like(s_coords_rg), torch.zeros_like(l_coords_rg)
                
                # Prepare for gradient computation
                grad_outputs_val = torch.ones_like(potential_energy, device=potential_energy.device)
                
                # Determine which inputs to compute gradients for
                inputs_for_grad_calc = []
                if self.spatial_dim > 0:
                    inputs_for_grad_calc.append(s_coords_rg)
                if self.latent_dim > 0:
                    inputs_for_grad_calc.append(l_coords_rg)
                
                if not inputs_for_grad_calc:
                    return torch.zeros_like(s_coords_rg), torch.zeros_like(l_coords_rg)
                
                # Compute gradients
                grads = torch.autograd.grad(
                    outputs=potential_energy,
                    inputs=inputs_for_grad_calc,
                    grad_outputs=grad_outputs_val,
                    create_graph=create_graph,
                    retain_graph=create_graph or torch.is_grad_enabled(),
                    allow_unused=True
                )
                
                # Extract gradients
                grad_U_spatial = torch.zeros_like(s_coords_rg)
                grad_U_latent = torch.zeros_like(l_coords_rg)
                
                current_grad_idx = 0
                if self.spatial_dim > 0:
                    if grads[current_grad_idx] is not None:
                        grad_U_spatial = grads[current_grad_idx]
                    current_grad_idx += 1
                
                if self.latent_dim > 0:
                    if current_grad_idx < len(grads) and grads[current_grad_idx] is not None:
                        grad_U_latent = grads[current_grad_idx]
                
                return grad_U_spatial, grad_U_latent
                
            finally:
                # Restore original training state
                self.train(was_training)
                
        finally:
            # Restore original gradient mode
            torch.set_grad_enabled(original_grad_mode)