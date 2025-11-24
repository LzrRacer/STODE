import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import torch
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d


# src/utils/plotting.py

import numpy as np
import matplotlib.pyplot as plt
# matplotlib.cm and Normalize are not strictly needed here if colorbar is handled outside
# import matplotlib.cm as cm 
# from matplotlib.colors import Normalize
from scipy.stats import binned_statistic_2d

# ... (compress_to_1d and other functions) ...

def plot_gridded_spatial_variable(
    adata_slice, 
    variable_key, 
    spatial_key,
    ax,  # Function now expects an Axes object to plot on
    grid_resolution=50, 
    cmap='viridis', 
    vmin=None, vmax=None, # Allow auto-scaling by default if not provided
    title=""
):
    """
    Creates a 2D gridded heatmap of a variable on spatial coordinates on a given ax.
    Returns the image artist for potential colorbar creation.
    """
    if adata_slice.n_obs == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis('off')
        return None 

    if variable_key not in adata_slice.obs:
        ax.text(0.5, 0.5, f"Key '{variable_key}' missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis('off')
        return None
        
    coords = adata_slice.obsm[spatial_key]
    values_to_plot = adata_slice.obs[variable_key].values

    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    valid_coord_mask = ~np.isnan(x_coords) & ~np.isnan(y_coords)
    valid_value_mask = ~np.isnan(values_to_plot)
    final_valid_mask = valid_coord_mask & valid_value_mask

    if not np.any(final_valid_mask):
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis('off')
        return None

    x_coords_valid = x_coords[final_valid_mask]
    y_coords_valid = y_coords[final_valid_mask]
    values_to_plot_valid = values_to_plot[final_valid_mask]

    if x_coords_valid.size == 0:
        ax.text(0.5, 0.5, "No valid coords", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis('off')
        return None

    x_min_data, x_max_data = np.min(x_coords_valid), np.max(x_coords_valid)
    y_min_data, y_max_data = np.min(y_coords_valid), np.max(y_coords_valid)
    
    padding_x = (x_max_data - x_min_data) * 0.01 if (x_max_data - x_min_data) > 1e-6 else 0.1 # Avoid zero padding for single point
    padding_y = (y_max_data - y_min_data) * 0.01 if (y_max_data - y_min_data) > 1e-6 else 0.1
    
    xbins = np.linspace(x_min_data - padding_x, x_max_data + padding_x, grid_resolution + 1)
    ybins = np.linspace(y_min_data - padding_y, y_max_data + padding_y, grid_resolution + 1)

    statistic, _, _, _ = binned_statistic_2d(
        x_coords_valid, y_coords_valid, values_to_plot_valid, 
        statistic='mean', bins=[xbins, ybins]
    )

    # Determine vmin and vmax for imshow if not provided
    # This ensures the colormap is scaled to the actual data being plotted in this specific ax
    # If a global scale is desired, vmin/vmax should be passed in.
    current_vmin = vmin if vmin is not None else np.nanmin(statistic)
    current_vmax = vmax if vmax is not None else np.nanmax(statistic)
    if np.isnan(current_vmin) or np.isnan(current_vmax) or current_vmin == current_vmax: # Handle all NaNs or single value
        current_vmin = 0
        current_vmax = 1
        if np.isnan(current_vmin) and np.isnan(current_vmax): # if statistic was all NaNs
             statistic = np.zeros_like(statistic) # Plot zeros to avoid imshow error

    img = ax.imshow(
        statistic.T, 
        origin='lower', 
        extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
        aspect='equal', # <<< SET ASPECT TO EQUAL
        cmap=cmap, 
        vmin=current_vmin, 
        vmax=current_vmax
    )
    
    ax.set_title(title)
    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y")
    # ax.axis('on') # Ensure axes are on if previously off for empty plots
    
    return img # Return the image artist (AxesImage)



def compress_to_1d(latent_codes, method='umap', n_neighbors=15, random_state=42):
    """
    Compress the latent codes to 1 dimension via UMAP or PCA.

    Args:
        latent_codes: (n_cells, latent_dim) numpy array
        method: 'umap' or 'pca'
        n_neighbors: Number of neighbors for UMAP (ignored if method=='pca')
        random_state: Random seed

    Returns:
        compressed_1d: (n_cells,) numpy array of 1D compression, scaled to [0, 1]
    """
    if method.lower() == 'umap':
        try:
            import umap
        except ImportError:
            print("UMAP is not installed. Falling back to PCA.")
            method = 'pca'

    if method.lower() == 'umap':
        import umap
        reducer = umap.UMAP(n_components=1, n_neighbors=n_neighbors, random_state=random_state)
        compressed = reducer.fit_transform(latent_codes)
        compressed_1d = compressed[:, 0]
    elif method.lower() == 'pca':
        pca = PCA(n_components=1, random_state=random_state)
        compressed = pca.fit_transform(latent_codes)
        compressed_1d = compressed[:, 0]
    else:
        raise ValueError(f"Invalid method '{method}'. Choose 'umap' or 'pca'.")

    # Scale to [0, 1] range
    scaler = MinMaxScaler()
    compressed_1d_scaled = scaler.fit_transform(compressed_1d.reshape(-1, 1)).flatten()

    return compressed_1d_scaled

def plot_latent_space_by_time(adata, latent_codes, time_key, spatial_key, latent_dim_to_color=0,
                             cmap='viridis', figsize=(18, 5)):
    """
    Plot latent space dimension values in spatial coordinates for each time point

    Args:
        adata: AnnData object with spatial coordinates
        latent_codes: Latent codes from VAE, numpy array (n_cells, latent_dim)
        time_key: Key in adata.obs for time information
        spatial_key: Key in adata.obsm for spatial coordinates
        latent_dim_to_color: Latent dimension to use for coloring
        cmap: Colormap to use
        figsize: Figure size

    Returns:
        fig: Matplotlib figure
    """
    # Get unique time points
    tp_unique = np.sort(adata.obs[time_key].unique())

    # Create figure
    fig, axes = plt.subplots(1, len(tp_unique), figsize=figsize, squeeze=False) # Ensure axes is always 2D
    axes = axes.flatten() # Flatten to 1D for easy iteration

    # Normalize colors across all time points
    vmin = np.min(latent_codes[:, latent_dim_to_color])
    vmax = np.max(latent_codes[:, latent_dim_to_color])
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Iterate over time points
    for i, t in enumerate(tp_unique):
        ax = axes[i]

        # Subset to time t
        mask = (adata.obs[time_key] == t)

        # Get spatial coordinates and latent codes
        coords_t = adata.obsm[spatial_key][mask]  # shape [num_cells_t, 2]
        z_t = latent_codes[mask]  # shape [num_cells_t, latent_dim]

        # Scatter plot
        sc = ax.scatter(
            coords_t[:, 0], coords_t[:, 1],
            c=z_t[:, latent_dim_to_color], cmap=cmap,
            s=5, alpha=0.7, norm=norm
        )

        ax.set_title(f"Time: {t}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal', adjustable='box') # Make axes square

    # Add colorbar
    cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.6, aspect=10) # Adjust aspect for thin colorbar
    cbar.set_label(f"Latent Dim {latent_dim_to_color} Value")

    plt.tight_layout()

    return fig


def plot_latent_space_by_annotation(adata, latent_codes, time_key, spatial_key, annotation_key,
                                   cmap='tab20', figsize=(18, 5)):
    """
    Plot latent space in spatial coordinates colored by annotation

    Args:
        adata: AnnData object with spatial coordinates
        latent_codes: Latent codes from VAE, numpy array (n_cells, latent_dim)
        time_key: Key in adata.obs for time information
        spatial_key: Key in adata.obsm for spatial coordinates
        annotation_key: Key in adata.obs for cell annotations
        cmap: Colormap to use
        figsize: Figure size

    Returns:
        fig: Matplotlib figure
    """
    # Get unique time points
    tp_unique = np.sort(adata.obs[time_key].unique())

    # Get unique annotations and sort them for consistent coloring
    annotations = sorted(adata.obs[annotation_key].astype(str).unique()) # Ensure string type

    # Create figure
    fig, axes = plt.subplots(1, len(tp_unique), figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Create categorical colormap
    cmap_obj = cm.get_cmap(cmap, len(annotations))

    # Create mapping from annotation to color index
    annotation_to_idx = {ann: i for i, ann in enumerate(annotations)}

    # Iterate over time points
    for i, t in enumerate(tp_unique):
        ax = axes[i]

        # Subset to time t
        mask = (adata.obs[time_key] == t)

        # Get spatial coordinates and annotations
        coords_t = adata.obsm[spatial_key][mask]
        ann_t = adata.obs[annotation_key][mask].astype(str) # Ensure string type

        # Convert annotations to color indices
        colors = [annotation_to_idx[ann] for ann in ann_t]

        # Scatter plot
        sc = ax.scatter(
            coords_t[:, 0], coords_t[:, 1],
            c=colors, cmap=cmap_obj,
            s=5, alpha=0.7, vmin=0, vmax=len(annotations)-1
        )

        ax.set_title(f"Time: {t}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal', adjustable='box')

    # Add legend (limited entries)
    n_ann = len(annotations)
    if n_ann > 20:
        legend_annotations = annotations[:20]
        truncated = True
    else:
        legend_annotations = annotations
        truncated = False

    legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=cmap_obj(annotation_to_idx[ann]),
                               markersize=8)
                     for ann in legend_annotations]

    legend_title = f"Cell Types{' (top 20)' if truncated else ''}"
    # Place legend outside the plot area
    fig.legend(legend_handles, legend_annotations,
              loc='lower center', bbox_to_anchor=(0.5, -0.1), # Adjust position
              ncol=min(5, len(legend_annotations)), title=legend_title)

    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to prevent overlap with legend
    # plt.subplots_adjust(bottom=0.25)  # make room for the legend

    return fig


def plot_latent_pca(latent_codes, adata, annotation_key, time_key, n_components=2, figsize=(12, 5)):
    """
    Plot PCA of latent space colored by annotation and time

    Args:
        latent_codes: Latent codes from VAE, numpy array (n_cells, latent_dim)
        adata: AnnData object with annotations
        annotation_key: Key in adata.obs for cell annotations (can be None)
        time_key: Key in adata.obs for time information
        n_components: Number of PCA components
        figsize: Figure size

    Returns:
        fig: Matplotlib figure containing both plots
    """
    # Compute PCA
    pca = PCA(n_components=n_components)
    latent_pca = pca.fit_transform(latent_codes)

    # Create DataFrame for easier plotting
    df = pd.DataFrame(latent_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    if annotation_key and annotation_key in adata.obs:
        df['annotation'] = adata.obs[annotation_key].astype(str).values
    else:
        df['annotation'] = 'N/A' # Placeholder if no annotation key
    df['time'] = adata.obs[time_key].astype(str).values # Ensure time is string for categorical plotting


    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- Plot by annotation ---
    if annotation_key and annotation_key in adata.obs:
        annotations = sorted(df['annotation'].unique())
        cmap = cm.get_cmap('tab20', len(annotations))
        annotation_to_idx = {ann: i for i, ann in enumerate(annotations)}

        for ann in annotations:
            mask = df['annotation'] == ann
            ax1.scatter(
                df.loc[mask, 'PC1'], df.loc[mask, 'PC2'],
                label=ann, alpha=0.7, s=5,
                color=cmap(annotation_to_idx[ann])
            )
        ax1.set_title('PCA by Cell Type')
        # Add annotation legend (limited entries)
        n_ann = len(annotations)
        if n_ann > 10:
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles[:10], labels[:10], loc='center left', bbox_to_anchor=(1.05, 0.5), title="Cell Types (top 10)", fontsize='small')
        elif n_ann > 0 and n_ann <= 10:
             ax1.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), title="Cell Types", fontsize='small')
        else:
             ax1.legend_ = None # No legend if no annotations

    else:
         # If no annotation key, plot all points in grey
         ax1.scatter(df['PC1'], df['PC2'], alpha=0.5, s=5, color='grey')
         ax1.set_title('PCA (No Annotation Provided)')


    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')


    # --- Plot by time ---
    times = np.sort(df['time'].unique())
    cmap_time = cm.get_cmap('viridis', len(times))
    time_to_idx = {t: i for i, t in enumerate(times)}

    for t in times:
        mask = df['time'] == t
        ax2.scatter(
            df.loc[mask, 'PC1'], df.loc[mask, 'PC2'],
            label=f'Time {t}', alpha=0.7, s=5,
            color=cmap_time(time_to_idx[t])
        )

    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax2.set_title('PCA by Time')
    if len(times) > 0:
         ax2.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), title="Time Points", fontsize='small')
    else:
         ax2.legend_ = None

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legends

    return fig


def plot_reconstruction_comparison(original_data, reconstructed_data, indices=None, n_samples=5, n_features=20, figsize=(15, 10)):
    """
    Plot comparison between original data and reconstructed data

    Args:
        original_data: Original data, numpy array (n_cells, n_features)
        reconstructed_data: Reconstructed data, numpy array (n_cells, n_features)
        indices: Indices of cells to plot, if None, random samples are selected
        n_samples: Number of cells to plot if indices is None
        n_features: Number of features to plot for each cell
        figsize: Figure size

    Returns:
        fig: Matplotlib figure
    """
    # Select random cells if indices not provided
    if indices is None:
        indices = np.random.choice(original_data.shape[0], n_samples, replace=False)
    else:
        n_samples = len(indices)

    # Create figure
    fig, axes = plt.subplots(n_samples, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Iterate over selected cells
    for i, idx in enumerate(indices):
        ax = axes[i]

        # Select top features for this cell based on original value
        orig = original_data[idx]
        recon = reconstructed_data[idx]

        if np.all(orig == 0): # Handle case where original data is all zero for a cell
             top_features = np.arange(min(n_features, len(orig)))
        else:
             # Sort features by original value, take top n_features
             sorted_indices = np.argsort(orig)
             top_features = sorted_indices[-n_features:]


        # Plot bar chart
        x = np.arange(len(top_features))
        width = 0.35

        ax.bar(x - width/2, orig[top_features], width, label='Original', color='blue', alpha=0.7)
        ax.bar(x + width/2, recon[top_features], width, label='Reconstructed', color='orange', alpha=0.7)

        ax.set_title(f'Cell {idx} - Top {len(top_features)} Expressed Features')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Feat {j}' for j in top_features], rotation=90, fontsize='small')
        ax.tick_params(axis='y', labelsize='small')
        ax.set_ylabel('Expression', fontsize='small')


        if i == 0:
            ax.legend()

    plt.tight_layout()

    return fig


def plot_latent_dimensions(adata, latent_codes, time_key, spatial_key, dims_to_plot=None,
                          n_dims=6, cmap='viridis', figsize=(15, 10)):
    """
    Plot multiple latent dimensions in spatial coordinates

    Args:
        adata: AnnData object with spatial coordinates
        latent_codes: Latent codes from VAE, numpy array (n_cells, latent_dim)
        time_key: Key in adata.obs for time information
        spatial_key: Key in adata.obsm for spatial coordinates
        dims_to_plot: List of dimensions to plot, if None, top n_dims with highest variance are used
        n_dims: Number of dimensions to plot if dims_to_plot is None
        cmap: Colormap to use
        figsize: Figure size

    Returns:
        fig: Matplotlib figure
    """
    # Get dimensions to plot
    if dims_to_plot is None:
        # Select dimensions with highest variance
        var_per_dim = np.var(latent_codes, axis=0)
        # Ensure n_dims doesn't exceed available dimensions
        n_dims = min(n_dims, latent_codes.shape[1])
        dims_to_plot = np.argsort(var_per_dim)[-n_dims:]
    n_dims = len(dims_to_plot) # Update n_dims based on dims_to_plot

    # Get unique time points
    tp_unique = np.sort(adata.obs[time_key].unique())
    n_tp = len(tp_unique)

    # Create figure
    fig, axes = plt.subplots(n_dims, n_tp, figsize=figsize, squeeze=False)

    # Iterate over dimensions and time points
    for i, dim in enumerate(dims_to_plot):
        # Normalize colors for this dimension across all time points
        vmin = np.min(latent_codes[:, dim])
        vmax = np.max(latent_codes[:, dim])
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap) # Create mappable for colorbar

        for j, t in enumerate(tp_unique):
            ax = axes[i, j]

            # Subset to time t
            mask = (adata.obs[time_key] == t)

            # Get spatial coordinates and latent codes
            coords_t = adata.obsm[spatial_key][mask]
            z_t = latent_codes[mask]

            if coords_t.shape[0] > 0: # Only plot if cells exist
                 # Scatter plot
                 sc = ax.scatter(
                     coords_t[:, 0], coords_t[:, 1],
                     c=z_t[:, dim], cmap=cmap,
                     s=3, alpha=0.7, norm=norm
                 )
            else:
                 ax.text(0.5, 0.5, 'No cells', ha='center', va='center', transform=ax.transAxes, fontsize='small')


            # Set title on top row only
            if i == 0:
                ax.set_title(f"Time: {t}")

            # Set dimension label on leftmost column
            if j == 0:
                ax.set_ylabel(f"Dim {dim}", fontsize='small')

            # Minimal ticks/labels for clarity
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal', adjustable='box')

        # Add colorbar for this dimension row (only if cells were plotted)
        if 'sc' in locals() and sc is not None:
            cbar_ax = fig.add_axes([0.92, axes[i,0].get_position().y0, 0.015, axes[i,0].get_position().height])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(f"Dim {dim} Val", fontsize='small')
            cbar.ax.tick_params(labelsize='small')
        del locals()['sc'] # Reset sc for next row


    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.9, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)

    return fig


def plot_spatial_clusters(adata, labels, time_key, spatial_key, cmap='tab20',
                         figsize=(18, 5), title="Spatial Clustering"):
    """
    Plot clusters in spatial coordinates

    Args:
        adata: AnnData object with spatial coordinates
        labels: Cluster labels, array-like (n_cells,)
        time_key: Key in adata.obs for time information
        spatial_key: Key in adata.obsm for spatial coordinates
        cmap: Colormap to use
        figsize: Figure size
        title: Plot title

    Returns:
        fig: Matplotlib figure
    """
    # Get unique time points
    tp_unique = np.sort(adata.obs[time_key].unique())
    n_tp = len(tp_unique)

    # Get unique clusters and sort them for consistent coloring
    # Convert labels to categorical if they are not already, for proper handling
    if not pd.api.types.is_categorical_dtype(labels):
         labels = pd.Categorical(labels)
    clusters = np.sort(labels.categories) # Use categories for sorted unique labels
    n_clusters = len(clusters)


    # Create figure
    fig, axes = plt.subplots(1, n_tp, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Create categorical colormap
    cmap_obj = cm.get_cmap(cmap, n_clusters)

    # Create mapping from cluster label to color index
    cluster_to_idx = {clust: i for i, clust in enumerate(clusters)}

    # Iterate over time points
    for i, t in enumerate(tp_unique):
        ax = axes[i]

        # Subset to time t
        mask = (adata.obs[time_key] == t)

        # Get spatial coordinates and labels
        coords_t = adata.obsm[spatial_key][mask]
        labels_t = labels[mask] # Get the subset of labels

        if coords_t.shape[0] > 0: # Only plot if cells exist
             # Convert labels to color indices
             colors = [cluster_to_idx[lbl] for lbl in labels_t]

             # Scatter plot
             sc = ax.scatter(
                 coords_t[:, 0], coords_t[:, 1],
                 c=colors, cmap=cmap_obj,
                 s=5, alpha=0.7, vmin=0, vmax=n_clusters-1
             )
        else:
             ax.text(0.5, 0.5, 'No cells', ha='center', va='center', transform=ax.transAxes, fontsize='small')


        ax.set_title(f"Time: {t}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal', adjustable='box')

    # Add legend (limited entries)
    if n_clusters > 0 and n_clusters <= 20:
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=cmap_obj(cluster_to_idx[clust]),
                                   markersize=8)
                         for clust in clusters]

        fig.legend(legend_handles, [f'Cluster {i}' for i in clusters],
                  loc='lower center', bbox_to_anchor=(0.5, -0.1 if n_clusters <= 10 else -0.15), # Adjust y offset based on number of clusters
                  ncol=min(10, n_clusters), title="Clusters")
        plt.subplots_adjust(bottom=0.25 if n_clusters > 5 else 0.15) # Adjust bottom margin for legend

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

    return fig


def plot_1d_compression_by_time(adata, compressed_1d, time_key, spatial_key, method='umap',
                                cmap='viridis', figsize=(18,5)):
    """
    Plot a 1D compression in spatial coordinates for each time point.
    Similar style to plot_latent_space_by_time.

    Args:
        adata: AnnData object with spatial coordinates
        compressed_1d: (n_cells,) 1D embedding from compress_to_1d
        time_key: Key in adata.obs for time information
        spatial_key: Key in adata.obsm for spatial coordinates
        method: 'umap' or 'pca', just used for labeling
        cmap: Colormap to use
        figsize: Figure size

    Returns:
        fig: Matplotlib figure
    """
    tp_unique = np.sort(adata.obs[time_key].unique())
    n_tp = len(tp_unique)

    fig, axes = plt.subplots(1, n_tp, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Normalize across entire 1D embedding (already scaled [0, 1])
    vmin = 0
    vmax = 1
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap) # Create mappable for colorbar

    for i, t in enumerate(tp_unique):
        ax = axes[i]

        # Subset to time t
        mask = (adata.obs[time_key] == t)
        coords_t = adata.obsm[spatial_key][mask]
        c_t = compressed_1d[mask]

        if coords_t.shape[0] > 0:
            sc = ax.scatter(
                coords_t[:, 0], coords_t[:, 1],
                c=c_t, cmap=cmap, s=5, alpha=0.7, norm=norm
            )
        else:
             ax.text(0.5, 0.5, 'No cells', ha='center', va='center', transform=ax.transAxes, fontsize='small')


        ax.set_title(f"Time: {t}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal', adjustable='box')

    # Add colorbar (only if cells were plotted)
    if 'sc' in locals() and sc is not None:
         cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.6, aspect=10)
         cbar.set_label(f"1D {method.upper()} Value")
    del locals()['sc']

    plt.tight_layout()

    return fig