# src/utils/training_helpers.py
import torch
import torch.nn.functional as F # For F.mse_loss or F.smooth_l1_loss

def calculate_t0_convergence_loss_smoothed(
    predicted_spatial_t0, predicted_latent_t0, 
    target_t0_spatial_mean_tensor, target_t0_latent_mean_tensor, 
    spatial_mean_weight, spatial_variance_weight, 
    latent_mean_weight, latent_variance_weight,
    loss_type='smooth_l1' # 'mse' or 'smooth_l1'
):
    """
    Calculates a loss to encourage convergence at t0, penalizing deviation from
    target means and variance of the predicted points.
    Uses smooth_l1_loss for robustness or mse_loss.
    """
    loss = torch.tensor(0.0, device=predicted_spatial_t0.device, requires_grad=True) # Ensure it's a leaf tensor that requires grad
    n_points = predicted_spatial_t0.shape[0]

    if loss_type == 'mse':
        loss_fn_mean = F.mse_loss
        loss_fn_variance = F.mse_loss # Variance is mean squared error from empirical mean
    elif loss_type == 'smooth_l1':
        loss_fn_mean = F.smooth_l1_loss
        loss_fn_variance = F.smooth_l1_loss # Apply to deviations from empirical mean
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    # Spatial convergence
    if target_t0_spatial_mean_tensor is not None and spatial_mean_weight > 0:
        # Expand target mean to match batch size for loss calculation
        expanded_target_s_mean = target_t0_spatial_mean_tensor.unsqueeze(0).expand_as(predicted_spatial_t0)
        loss_s_mean = loss_fn_mean(predicted_spatial_t0, expanded_target_s_mean, reduction='mean')
        loss = loss + spatial_mean_weight * loss_s_mean
    
    if n_points > 1 and spatial_variance_weight > 0: # Need multiple points to calculate variance
        empirical_mean_spatial_t0 = predicted_spatial_t0.mean(dim=0, keepdim=True).detach() # Detach empirical mean
        expanded_empirical_s_mean = empirical_mean_spatial_t0.expand_as(predicted_spatial_t0)
        # Minimize variance: penalize deviation from the empirical mean of the batch
        loss_s_var = loss_fn_variance(predicted_spatial_t0, expanded_empirical_s_mean, reduction='mean')
        loss = loss + spatial_variance_weight * loss_s_var

    # Latent convergence
    if target_t0_latent_mean_tensor is not None and latent_mean_weight > 0:
        expanded_target_l_mean = target_t0_latent_mean_tensor.unsqueeze(0).expand_as(predicted_latent_t0)
        loss_l_mean = loss_fn_mean(predicted_latent_t0, expanded_target_l_mean, reduction='mean')
        loss = loss + latent_mean_weight * loss_l_mean

    if n_points > 1 and latent_variance_weight > 0:
        empirical_mean_latent_t0 = predicted_latent_t0.mean(dim=0, keepdim=True).detach()
        expanded_empirical_l_mean = empirical_mean_latent_t0.expand_as(predicted_latent_t0)
        loss_l_var = loss_fn_variance(predicted_latent_t0, expanded_empirical_l_mean, reduction='mean')
        loss = loss + latent_variance_weight * loss_l_var
            
    return loss