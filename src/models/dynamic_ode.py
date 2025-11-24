import torch
import torch.nn as nn

class DynamicODE(nn.Module):
    def __init__(self, spatial_dim, latent_dim, potential_gradient_dim, ode_hidden_dims, damping_coeff=0.1):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.latent_dim = latent_dim
        self.damping_coeff = damping_coeff # Alpha from physics model

        # Input: current_spatial, current_latent_z, grad_potential_spatial, grad_potential_latent
        # grad_potential_dim = spatial_dim (for dU/dx, dU/dy) + latent_dim (for dU/dz_i)
        input_dim = spatial_dim + latent_dim + potential_gradient_dim # potential_gradient_dim is spatial_dim + latent_dim

        layers = []
        current_h_dim = input_dim
        for h_dim in ode_hidden_dims:
            layers.append(nn.Linear(current_h_dim, h_dim))
            layers.append(nn.Tanh()) # Using Tanh for bounded activations
            current_h_dim = h_dim
        
        # Output: d(spatial)/dt, d(latent_z)/dt
        self.output_layer_spatial = nn.Linear(current_h_dim, spatial_dim)
        self.output_layer_latent = nn.Linear(current_h_dim, latent_dim)
        
        self.net = nn.Sequential(*layers)

    def forward(self, t, state_and_grads):
        # state_and_grads is a tuple: (current_state, spatial_potential_grad, latent_potential_grad)
        # current_state: [batch, spatial_dim + latent_dim]
        # spatial_potential_grad: [batch, spatial_dim] (e.g., dU/dx, dU/dy)
        # latent_potential_grad: [batch, latent_dim] (e.g., dU/dz_i)
        
        current_state, grad_U_spatial, grad_U_latent = state_and_grads
        
        current_spatial = current_state[:, :self.spatial_dim]
        current_latent_z = current_state[:, self.spatial_dim:]

        # Concatenate all inputs for the network
        ode_input = torch.cat([current_spatial, current_latent_z, grad_U_spatial, grad_U_latent], dim=1)
        
        hidden_out = self.net(ode_input)

        # Predict velocities (which are the derivatives of position/latent state)
        # Basic model: velocity_spatial = -grad_U_spatial (movement down potential)
        #              velocity_latent  = -grad_U_latent
        # The network learns a more complex relationship, potentially incorporating damping implicitly
        # or we can add explicit damping.

        # For a physics-inspired model:
        # F = -grad(U)
        # dv/dt = F/m - damping*v  (if m=1, dv/dt = F - damping*v)
        # dx/dt = v
        # Here, the ODE directly outputs dx/dt and dz/dt.
        # So, these outputs are velocities.
        
        d_spatial_dt = self.output_layer_spatial(hidden_out) # This is velocity_spatial
        d_latent_dt = self.output_layer_latent(hidden_out)   # This is velocity_latent

        # Optional: Add explicit damping to the *outputted velocities*
        # This means the network learns the main driving force, and damping is applied post-hoc.
        # Or, the network could learn the damped velocity directly.
        # For simplicity, let's assume network learns F, and we compute v.
        # Or, let's assume network learns dx/dt directly.
        # If dx/dt = v_learned:
        # One way to interpret -grad(U) is as a force.
        # The ODE net learns to translate (state, grads) to (d_spatial/dt, d_latent/dt)
        # Let's assume the network output IS the velocity, and it has learned damping.
        # A simpler starting point is:
        # d_spatial_dt = -grad_U_spatial - self.damping_coeff * current_spatial_velocity (if we track velocity separately)
        # d_latent_dt  = -grad_U_latent  - self.damping_coeff * current_latent_velocity

        # If the ODE outputs dx/dt directly:
        # dx_dt = Network(state, grad_U_spatial, grad_U_latent)
        # This is common in Neural ODEs. The network implicitly learns the damping.
        
        # Let's make the network predict the "force-like" part, and apply damping here
        # Or simpler: network directly predicts velocity including damping effect
        # For now, let's assume network output d_spatial_dt includes damping.
        # This means grad_U_spatial and grad_U_latent are inputs that help it predict this.

        return torch.cat([d_spatial_dt, d_latent_dt], dim=1)