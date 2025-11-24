#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
cytotrace.py

Implementation of a CytoTRACE-like algorithm for calculating cell differentiation scores
and a Differentiation-Guided ODE model.

Author: Your Name
Date: 2025-XX-XX
"""

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
from scipy.sparse import issparse
import torch
from tqdm import tqdm

class CytoTRACE:
    """
    Implementation of CytoTRACE-like algorithm for assessing cell differentiation.

    Higher CytoTRACE scores indicate less differentiated (more stem-like) cells,
    while lower scores indicate more differentiated cells.
    """

    def __init__(self, n_top_genes=1000, n_neighbors=20, mode='vae_integration'):
        """
        Initialize the CytoTRACE calculator.

        Args:
            n_top_genes (int): Number of top correlated genes to use.
            n_neighbors (int): Number of neighbors for smoothing.
            mode (str): Mode for integration with VAE ('vae_integration' or 'standalone').
        """
        self.n_top_genes = n_top_genes
        self.n_neighbors = n_neighbors
        self.mode = mode

        # Store results
        self.gene_counts = None
        self.gene_correlations = None
        self.top_correlated_genes = None
        self.gcs_scores = None
        self.cytotrace_scores = None
        self.ranked_cells = None

    def calculate_gene_counts(self, adata):
        """
        Calculate the number of expressed genes for each cell.

        Args:
            adata: AnnData object with gene expression data.

        Returns:
            numpy.ndarray: Gene counts per cell.
        """
        # Get expression matrix
        if issparse(adata.X):
            X = adata.X.toarray()
        else:
            X = adata.X

        # Count non-zero expressions in each cell
        gene_counts = np.array((X > 0).sum(axis=1)).flatten()

        self.gene_counts = gene_counts
        return gene_counts

    def find_correlated_genes(self, adata):
        """
        Find genes that correlate with the gene count.

        Args:
            adata: AnnData object with gene expression data.

        Returns:
            pandas.DataFrame: Genes and their correlation with gene count.
        """
        if self.gene_counts is None:
            self.calculate_gene_counts(adata)

        # Get expression matrix
        if issparse(adata.X):
            X = adata.X.toarray()
        else:
            X = adata.X

        # Calculate correlation for each gene
        correlations = []
        for i in tqdm(range(X.shape[1]), desc="Finding correlated genes", leave=False):
            if np.var(X[:, i]) > 0:  # Skip genes with no variance
                corr, _ = pearsonr(X[:, i], self.gene_counts)
                correlations.append((i, corr))

        # Convert to DataFrame
        corr_df = pd.DataFrame(correlations, columns=['gene_idx', 'correlation'])
        corr_df = corr_df.sort_values('correlation', ascending=False)

        # Get gene names if available
        if adata.var_names is not None:
            gene_names = list(adata.var_names)
            corr_df['gene_name'] = corr_df['gene_idx'].apply(lambda x: gene_names[x])

        self.gene_correlations = corr_df

        # Select top correlated genes
        self.top_correlated_genes = corr_df.iloc[:self.n_top_genes]['gene_idx'].values

        return corr_df

    def compute_gcs(self, adata):
        """
        Compute the Gene Count Signature (GCS) for each cell.

        Args:
            adata: AnnData object with gene expression data.

        Returns:
            numpy.ndarray: GCS scores per cell.
        """
        if self.top_correlated_genes is None:
            self.find_correlated_genes(adata)

        # Get expression matrix
        if issparse(adata.X):
            X = adata.X.tocsc() # Use CSC for efficient column slicing
            gcs = np.array(X[:, self.top_correlated_genes].mean(axis=1)).flatten()
        else:
            X = adata.X
             # Average expression of top correlated genes
            gcs = np.mean(X[:, self.top_correlated_genes], axis=1)

        self.gcs_scores = gcs
        return gcs

    def smooth_scores(self, adata, use_latent=True):
        """
        Smooth the GCS scores using k-nearest neighbors in latent or PCA space.

        Args:
            adata: AnnData object with gene expression data.
            use_latent (bool): Whether to use VAE latent space for smoothing.
                               If False, PCA is used.

        Returns:
            numpy.ndarray: Smoothed CytoTRACE scores.
        """
        if self.gcs_scores is None:
            self.compute_gcs(adata)

        # Get embedding for similarity calculation
        if use_latent and 'X_latent' in adata.obsm:
            print("Using VAE latent space for KNN smoothing.")
            embedding = adata.obsm['X_latent']
        else:
            print("Using PCA space for KNN smoothing.")
            if 'X_pca' not in adata.obsm:
                print("PCA not found, computing...")
                sc.pp.pca(adata, n_comps=min(50, adata.shape[1]-1)) # Ensure n_comps < n_features
            embedding = adata.obsm['X_pca']

        # Find k-nearest neighbors
        print(f"Finding {self.n_neighbors} nearest neighbors...")
        nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')
        nn.fit(embedding)
        distances, indices = nn.kneighbors(embedding)

        # Apply Gaussian weighting (optional, simple averaging is also common)
        # sigma = np.mean(distances[:, 1:]) # Use mean distance to non-self neighbors
        # weights = np.exp(-distances**2 / (2 * sigma**2))
        # weights = weights / weights.sum(axis=1, keepdims=True)

        # Smooth scores (using simple averaging for efficiency)
        print("Smoothing GCS scores...")
        smoothed_scores = np.zeros_like(self.gcs_scores, dtype=float)
        for i in tqdm(range(len(self.gcs_scores)), desc="Smoothing scores", leave=False):
            neighbor_indices = indices[i]
            neighbor_scores = self.gcs_scores[neighbor_indices]
            # smoothed_scores[i] = np.sum(neighbor_scores * weights[i]) # Weighted average
            smoothed_scores[i] = np.mean(neighbor_scores) # Simple average

        # Scale to [0, 1] range for easier interpretation
        # Ensure denominator is not zero if all scores are the same
        min_score = np.min(smoothed_scores)
        max_score = np.max(smoothed_scores)
        if max_score == min_score:
             cytotrace_scores = np.ones_like(smoothed_scores) * 0.5 # Or np.zeros_like(...)
        else:
             cytotrace_scores = (smoothed_scores - min_score) / (max_score - min_score)


        self.cytotrace_scores = cytotrace_scores
        return cytotrace_scores

    def rank_cells(self):
        """
        Rank cells from least differentiated (highest score) to most differentiated.

        Returns:
            pandas.DataFrame: Ranked cells with indices and scores.
        """
        if self.cytotrace_scores is None:
            raise ValueError("You need to run smooth_scores() first.")

        indices = np.argsort(self.cytotrace_scores)[::-1]  # Sort in descending order
        ranked_df = pd.DataFrame({
            'cell_idx': indices,
            'cytotrace_score': self.cytotrace_scores[indices],
            'rank': np.arange(len(indices))
        })

        self.ranked_cells = ranked_df
        return ranked_df

    def run_pipeline(self, adata, use_latent=True):
        """
        Run the complete CytoTRACE pipeline.

        Args:
            adata: AnnData object with gene expression data.
            use_latent (bool): Whether to use VAE latent space for KNN.

        Returns:
            AnnData: Updated AnnData with CytoTRACE scores.
        """
        print("Running CytoTRACE pipeline...")

        # Calculate gene counts
        print("1. Calculating gene counts...")
        self.calculate_gene_counts(adata)

        # Find correlated genes
        print(f"2. Finding top {self.n_top_genes} genes correlated with gene count...")
        self.find_correlated_genes(adata)

        # Compute GCS
        print("3. Computing Gene Count Signature (GCS)...")
        self.compute_gcs(adata)

        # Smooth scores
        print(f"4. Smoothing scores using {self.n_neighbors} nearest neighbors...")
        self.smooth_scores(adata, use_latent=use_latent)

        # Rank cells
        print("5. Ranking cells...")
        self.rank_cells()

        # Store results in AnnData
        adata.obs['gene_count'] = self.gene_counts
        adata.obs['gcs'] = self.gcs_scores
        adata.obs['cytotrace_score'] = self.cytotrace_scores
        adata.obs['differentiation_rank'] = np.argsort(np.argsort(self.cytotrace_scores)[::-1])

        print("CytoTRACE pipeline completed.")
        return adata

    def calculate_spatial_differentiation_map(self, adata, spatial_key='spatial', resolution=100):
        """
        Calculate a spatial map of differentiation scores by averaging the scores of
        neighboring cells at each point in a grid.

        Args:
            adata: AnnData object with spatial coordinates and differentiation scores.
            spatial_key (str): Key in adata.obsm for spatial coordinates.
            resolution (int): Resolution of the grid.

        Returns:
            tuple: (grid_x, grid_y, differentiation_map) - Coordinates and values of the map.
        """
        if self.cytotrace_scores is None:
            raise ValueError("You need to run the pipeline first.")

        if spatial_key not in adata.obsm:
            raise ValueError(f"No spatial coordinates found with key '{spatial_key}'.")

        # Get spatial coordinates
        spatial_coords = adata.obsm[spatial_key]

        # Determine grid boundaries
        x_min, y_min = spatial_coords.min(axis=0)
        x_max, y_max = spatial_coords.max(axis=0)

        # Add some padding
        padding = 0.05
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= padding * x_range
        x_max += padding * x_range
        y_min -= padding * y_range
        y_max += padding * y_range

        # Create grid
        grid_x = np.linspace(x_min, x_max, resolution)
        grid_y = np.linspace(y_min, y_max, resolution)
        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

        # Create spatial points for grid
        grid_points = np.vstack([grid_xx.flatten(), grid_yy.flatten()]).T

        # Create KDTree for spatial coordinates for faster neighbor search
        from scipy.spatial import cKDTree
        print("Building KDTree for spatial coordinates...")
        kdtree = cKDTree(spatial_coords)

        # Search for k nearest cells for each grid point
        k = min(30, len(adata))  # Use up to 30 nearest neighbors or num cells if fewer
        print(f"Querying {k} nearest neighbors for each grid point...")
        distances, indices = kdtree.query(grid_points, k=k, workers=-1) # Use all available cores

        # Handle cases where k > number of cells or distances are infinite
        distances = np.nan_to_num(distances, nan=np.inf)
        valid_distances = distances[np.isfinite(distances)].flatten()
        if len(valid_distances) == 0:
             sigma = 1.0 # Default sigma if no valid distances
        else:
             sigma = np.mean(valid_distances) / 2 # Adjust sigma based on average valid distance

        if sigma == 0: sigma = 1e-6 # Prevent division by zero

        # Apply Gaussian weighting
        weights = np.exp(-distances**2 / (2 * sigma**2))
        # Normalize weights row-wise, handle rows with zero sum
        sum_weights = weights.sum(axis=1, keepdims=True)
        weights = np.divide(weights, sum_weights, out=np.zeros_like(weights), where=sum_weights!=0)


        # Compute weighted average of differentiation scores
        print("Computing weighted average differentiation scores for grid...")
        diff_map = np.zeros(len(grid_points))
        valid_indices_mask = indices < len(self.cytotrace_scores) # Ensure indices are valid

        for i in tqdm(range(len(grid_points)), desc="Averaging scores", leave=False):
             valid_row_indices = indices[i][valid_indices_mask[i]]
             valid_row_weights = weights[i][valid_indices_mask[i]]

             if len(valid_row_indices) > 0:
                 diff_map[i] = np.sum(self.cytotrace_scores[valid_row_indices] * valid_row_weights)
             else:
                 diff_map[i] = np.nan # Or some default value like 0 or mean score

        # Reshape back to grid
        differentiation_map = diff_map.reshape(grid_xx.shape)
        # Handle potential NaNs if kdtree query failed for some points
        if np.isnan(differentiation_map).any():
             print("Warning: NaNs found in differentiation map, filling with mean.")
             mean_score = np.nanmean(differentiation_map)
             differentiation_map = np.nan_to_num(differentiation_map, nan=mean_score)


        return grid_x, grid_y, differentiation_map

    def integrate_with_vae_latent(self, adata, latent_dim_idx=0, weight=0.5):
        """
        Integrate differentiation scores with VAE latent variables.

        Args:
            adata: AnnData object with VAE latent variables.
            latent_dim_idx (int): Index of latent dimension to modify.
            weight (float): Weight of differentiation score in the integration (0 to 1).

        Returns:
            AnnData: Updated AnnData with integrated latent variables.
        """
        if self.cytotrace_scores is None:
            raise ValueError("You need to run the pipeline first.")

        if 'X_latent_scaled' not in adata.obsm:
            raise ValueError("No scaled latent variables found in adata.obsm['X_latent_scaled'].")

        if not (0 <= weight <= 1):
             raise ValueError("Integration weight must be between 0 and 1.")

        # Get latent variables
        latent = adata.obsm['X_latent_scaled'].copy()

        if latent_dim_idx >= latent.shape[1] or latent_dim_idx < 0:
             raise ValueError(f"latent_dim_idx {latent_dim_idx} is out of bounds for latent space of dimension {latent.shape[1]}.")

        # Normalize differentiation scores to [-1, 1] - higher score = less differentiated (maps to positive end)
        # We map score 0 (most diff) to -1, score 1 (least diff) to +1
        diff_scaled = 2 * (self.cytotrace_scores - 0.5)

        # Integrate into selected latent dimension using weighted average
        latent[:, latent_dim_idx] = (1 - weight) * latent[:, latent_dim_idx] + weight * diff_scaled

        # Store integrated latent variables
        adata.obsm['X_latent_integrated'] = latent

        return adata


class DifferentiationGuidedODE(torch.nn.Module):
    """
    ODE model where dynamics can be influenced by cell state and differentiation scores.
    The training loop (external to this class) will define the target derivatives
    that this model learns to predict.
    """
    def __init__(self, latent_dim, hidden_dim=128, spatial_dim=2, 
                 time_scaling=1.0, dropout_rate=0.1): # Removed diff_attraction_strength
        super().__init__()
        self.latent_dim = latent_dim
        self.spatial_dim = spatial_dim
        self.time_scaling = time_scaling

        # Input to the ODE network: current state (spatial_coords, latent_states) 
        # AND current differentiation score.
        # Total input dim = spatial_dim + latent_dim + 1 (for diff_score)
        input_dim_ode = self.spatial_dim + self.latent_dim + 1 
        
        # Output: derivatives for spatial_coords and latent_states
        output_dim_ode = self.spatial_dim + self.latent_dim

        self.ode_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim_ode, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_dim, output_dim_ode) 
        )

    def forward(self, t, states, diff_scores):
        """
        Forward pass computes the derivative of the state.

        Args:
            t: Current time point (scalar, often unused if network is time-invariant).
            states: Current state tensor [batch_size, spatial_dim + latent_dim].
            diff_scores: Differentiation scores for each cell [batch_size].

        Returns:
            derivatives: Tensor [batch_size, spatial_dim + latent_dim], representing d(state)/dt.
        """
        # t = t * self.time_scaling # Time scaling can be applied if network is time-dependent

        if not torch.is_tensor(diff_scores):
            diff_scores = torch.tensor(diff_scores, dtype=torch.float32, device=states.device)
        if diff_scores.dim() == 1:
            diff_scores = diff_scores.unsqueeze(1) # Ensure [batch_size, 1]

        # Combine states and diff_scores for input to the network
        ode_input = torch.cat([states, diff_scores], dim=1)
        
        derivatives = self.ode_net(ode_input)
        
        return derivatives