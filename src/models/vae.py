import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, NegativeBinomial, kl_divergence

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout_rate=0.1):
        super().__init__()

        # Build encoder layers
        modules = []

        # Input layer
        modules.append(nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        ))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate)
            ))

        self.encoder = nn.Sequential(*modules)

        # Mean and variance for latent space
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        result = self.encoder(x)
        mu = self.mu(result)
        log_var = self.log_var(result)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, dropout_rate=0.1):
        super().__init__()

        # Reverse the hidden dimensions
        hidden_dims = hidden_dims[::-1]

        # Build decoder layers
        modules = []

        # Input layer
        modules.append(nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        ))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate)
            ))

        self.decoder = nn.Sequential(*modules)

        # For Negative Binomial output
        self.theta = nn.Parameter(torch.ones(output_dim))
        self.mean_decoder = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, z):
        result = self.decoder(z)
        mean = F.softplus(self.mean_decoder(result))
        return mean, self.theta


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout_rate=0.1):
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder and decoder
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim, dropout_rate)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim, dropout_rate)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        mean, theta = self.decoder(z)
        return mean, mu, log_var

    def sample(self, num_samples, device):
        """
        Sample from the latent space and generate new data points
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        mean, _ = self.decoder(z)
        return mean


class VAEWrapper(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], latent_dim=64, dropout_rate=0.1):
        super().__init__()
        self.vae = VAE(input_dim, hidden_dims, latent_dim, dropout_rate)
        self.latent_dim = latent_dim # Store latent_dim

    def forward(self, x):
        return self.vae(x)

    def gaussian_likelihood(self, x_hat, x, scale=1.0):
        dist = Normal(x_hat, scale)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=1)

    def negative_binomial_loss(self, x, mean, theta, eps=1e-8):
        """
        Negative Binomial loss for count data

        For non-integer values, we round to the nearest integer
        to satisfy NegativeBinomial distribution requirements
        """
        # Round x to integers and ensure non-negative values
        x_int = torch.round(x).clamp(min=0).to(torch.int32)

        # Create NegativeBinomial distribution
        nb_dist = NegativeBinomial(mean + eps, logits=theta)

        # Calculate negative log likelihood
        return -nb_dist.log_prob(x_int).sum(dim=1)

    def mse_loss(self, x_hat, x):
        return F.mse_loss(x_hat, x, reduction='none').sum(dim=1)

    def calculate_loss(self, x, x_hat, mu, log_var, loss_type='nb', kl_weight=1.0):
        """
        Calculate VAE loss
        loss_type: 'gaussian', 'nb' (Negative Binomial), or 'mse'
        """
        # KL Divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

        # Reconstruction loss
        if loss_type == 'gaussian':
            recon_loss = -self.gaussian_likelihood(x_hat, x)
        elif loss_type == 'nb':
            # For Negative Binomial, x_hat is actually mean, and we need theta from the decoder
            mean, theta = self.vae.decoder(self.vae.reparameterize(mu, log_var))
            recon_loss = self.negative_binomial_loss(x, mean, theta)
        elif loss_type == 'mse':
            recon_loss = self.mse_loss(x_hat, x)
        elif loss_type == 'zinb':  # Zero-inflated Negative Binomial
            # This is a placeholder for future implementation
            # For now, fall back to standard NB
            mean, theta = self.vae.decoder(self.vae.reparameterize(mu, log_var))
            recon_loss = self.negative_binomial_loss(x, mean, theta)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Total loss
        loss = recon_loss + kl_weight * kl_div

        return {
            'loss': loss.mean(),
            'recon_loss': recon_loss.mean(),
            'kl_div': kl_div.mean()
        }

    def encode(self, x):
        """Get latent representation"""
        self.vae.eval()
        with torch.no_grad():
            mu, log_var = self.vae.encoder(x)
            z = self.vae.reparameterize(mu, log_var)
        return z, mu, log_var

    def decode(self, z):
        """Generate data from latent representation"""
        self.vae.eval()
        with torch.no_grad():
            mean, theta = self.vae.decoder(z)
        return mean

    def save(self, path):
        """Save model weights"""
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        """Load model weights"""
        self.load_state_dict(torch.load(path, map_location=device))