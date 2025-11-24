
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import torch
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import binned_statistic_2d

def plot_gridded_spatial_variable(
    adata_slice, 
    variable_key, 
    spatial_key,
    ax,
    grid_resolution=50, 
    cmap='viridis', 
    vmin=None, vmax=None,
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
    
    padding_x = (x_max_data - x_min_data) * 0.01 if (x_max_data - x_min_data) > 1e-6 else 0.1
    padding_y = (y_max_data - y_min_data) * 0.01 if (y_max_data - y_min_data) > 1e-6 else 0.1
    
    xbins = np.linspace(x_min_data - padding_x, x_max_data + padding_x, grid_resolution + 1)
    ybins = np.linspace(y_min_data - padding_y, y_max_data + padding_y, grid_resolution + 1)

    statistic, _, _, _ = binned_statistic_2d(
        x_coords_valid, y_coords_valid, values_to_plot_valid, 
        statistic='mean', bins=[xbins, ybins]
    )

    current_vmin = vmin if vmin is not None else np.nanmin(statistic)
    current_vmax = vmax if vmax is not None else np.nanmax(statistic)
    if np.isnan(current_vmin) or np.isnan(current_vmax) or current_vmin == current_vmax:
        current_vmin = 0
        current_vmax = 1
        if np.isnan(current_vmin) and np.isnan(current_vmax):
             statistic = np.zeros_like(statistic)

    img = ax.imshow(
        statistic.T, 
        origin='lower', 
        extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
        aspect='equal',
        cmap=cmap, 
        vmin=current_vmin, 
        vmax=current_vmax
    )
    
    ax.set_title(title)
    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y")
    
    return img

def compress_to_1d(
    latent_codes,
    method='umap',
    n_neighbors=15,
    random_state=42,
    subsample_fit=None
):
    """
    Compress the latent codes to 1 dimension via UMAP or PCA.

    Args:
        latent_codes (np.ndarray): (n_cells, latent_dim) numpy array.
        method (str): 'umap' or 'pca'.
        n_neighbors (int): Number of neighbors for UMAP.
        random_state (int): Random seed.
        subsample_fit (int, optional): If set, fits the UMAP model on a random
            subsample of this size for speed, then transforms the full dataset.
            Defaults to None (fits on all data).

    Returns:
        np.ndarray: (n_cells,) numpy array of 1D compression, scaled to [0, 1].
    """
    if method.lower() == 'umap':
        try:
            import umap
        except ImportError:
            print("UMAP is not installed. Falling back to PCA.")
            method = 'pca'

    if method.lower() == 'umap':
        reducer = umap.UMAP(
            n_components=1,
            n_neighbors=n_neighbors,
            random_state=random_state,
            transform_seed=random_state
        )
        
        if subsample_fit and subsample_fit < latent_codes.shape[0]:
            print(f"Fitting UMAP on a random subsample of {subsample_fit} cells...")
            rng = np.random.default_rng(random_state)
            fit_indices = rng.choice(latent_codes.shape[0], size=subsample_fit, replace=False)
            fit_data = latent_codes[fit_indices]
            
            reducer.fit(fit_data)
            
            print("Transforming the full dataset using the fitted UMAP model...")
            compressed = reducer.transform(latent_codes)
        else:
            print("Fitting and transforming UMAP on the full dataset...")
            compressed = reducer.fit_transform(latent_codes)
        
        compressed_1d = compressed.flatten()

    elif method.lower() == 'pca':
        pca = PCA(n_components=1, random_state=random_state)
        compressed = pca.fit_transform(latent_codes)
        compressed_1d = compressed.flatten()
    else:
        raise ValueError(f"Invalid method '{method}'. Choose 'umap' or 'pca'.")

    scaler = MinMaxScaler()
    compressed_1d_scaled = scaler.fit_transform(compressed_1d.reshape(-1, 1)).flatten()

    return compressed_1d_scaled


def plot_latent_space_by_time(adata, latent_codes, time_key, spatial_key, latent_dim_to_color=0,
                             cmap='viridis', figsize=(18, 5)):
    """
    Plot latent space dimension values in spatial coordinates for each time point
    """
    tp_unique = np.sort(adata.obs[time_key].unique())
    fig, axes = plt.subplots(1, len(tp_unique), figsize=figsize, squeeze=False)
    axes = axes.flatten()
    vmin = np.min(latent_codes[:, latent_dim_to_color])
    vmax = np.max(latent_codes[:, latent_dim_to_color])
    norm = Normalize(vmin=vmin, vmax=vmax)

    for i, t in enumerate(tp_unique):
        ax = axes[i]
        mask = (adata.obs[time_key] == t)
        coords_t = adata.obsm[spatial_key][mask]
        z_t = latent_codes[mask]
        sc = ax.scatter(
            coords_t[:, 0], coords_t[:, 1],
            c=z_t[:, latent_dim_to_color], cmap=cmap,
            s=5, alpha=0.7, norm=norm
        )
        ax.set_title(f"Time: {t}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal', adjustable='box')

    cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.6, aspect=10)
    cbar.set_label(f"Latent Dim {latent_dim_to_color} Value")
    plt.tight_layout()
    return fig


def plot_latent_space_by_annotation(adata, latent_codes, time_key, spatial_key, annotation_key,
                                   cmap='tab20', figsize=(18, 5)):
    """
    Plot latent space in spatial coordinates colored by annotation
    """
    tp_unique = np.sort(adata.obs[time_key].unique())
    annotations = sorted(adata.obs[annotation_key].astype(str).unique())
    fig, axes = plt.subplots(1, len(tp_unique), figsize=figsize, squeeze=False)
    axes = axes.flatten()
    cmap_obj = cm.get_cmap(cmap, len(annotations))
    annotation_to_idx = {ann: i for i, ann in enumerate(annotations)}

    for i, t in enumerate(tp_unique):
        ax = axes[i]
        mask = (adata.obs[time_key] == t)
        coords_t = adata.obsm[spatial_key][mask]
        ann_t = adata.obs[annotation_key][mask].astype(str)
        colors = [annotation_to_idx[ann] for ann in ann_t]
        sc = ax.scatter(
            coords_t[:, 0], coords_t[:, 1],
            c=colors, cmap=cmap_obj,
            s=5, alpha=0.7, vmin=0, vmax=len(annotations)-1
        )
        ax.set_title(f"Time: {t}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal', adjustable='box')

    n_ann = len(annotations)
    legend_annotations = annotations[:20] if n_ann > 20 else annotations
    truncated = n_ann > 20
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=cmap_obj(annotation_to_idx[ann]),
                               markersize=8)
                     for ann in legend_annotations]
    legend_title = f"Cell Types{' (top 20)' if truncated else ''}"
    fig.legend(legend_handles, legend_annotations,
              loc='lower center', bbox_to_anchor=(0.5, -0.1),
              ncol=min(5, len(legend_annotations)), title=legend_title)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def plot_latent_pca(latent_codes, adata, annotation_key, time_key, n_components=2, figsize=(12, 5)):
    """
    Plot PCA of latent space colored by annotation and time
    """
    pca = PCA(n_components=n_components)
    latent_pca = pca.fit_transform(latent_codes)
    df = pd.DataFrame(latent_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    df['annotation'] = adata.obs[annotation_key].astype(str).values if annotation_key and annotation_key in adata.obs else 'N/A'
    df['time'] = adata.obs[time_key].astype(str).values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    if annotation_key and annotation_key in adata.obs:
        annotations = sorted(df['annotation'].unique())
        cmap = cm.get_cmap('tab20', len(annotations))
        annotation_to_idx = {ann: i for i, ann in enumerate(annotations)}
        for ann in annotations:
            mask = df['annotation'] == ann
            ax1.scatter(df.loc[mask, 'PC1'], df.loc[mask, 'PC2'], label=ann, alpha=0.7, s=5, color=cmap(annotation_to_idx[ann]))
        ax1.set_title('PCA by Cell Type')
        n_ann = len(annotations)
        if 0 < n_ann <= 10:
             ax1.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), title="Cell Types", fontsize='small')
        elif n_ann > 10:
             handles, labels = ax1.get_legend_handles_labels()
             ax1.legend(handles[:10], labels[:10], loc='center left', bbox_to_anchor=(1.05, 0.5), title="Cell Types (top 10)", fontsize='small')
    else:
         ax1.scatter(df['PC1'], df['PC2'], alpha=0.5, s=5, color='grey')
         ax1.set_title('PCA (No Annotation Provided)')

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')

    times = np.sort(df['time'].unique())
    cmap_time = cm.get_cmap('viridis', len(times))
    time_to_idx = {t: i for i, t in enumerate(times)}
    for t in times:
        mask = df['time'] == t
        ax2.scatter(df.loc[mask, 'PC1'], df.loc[mask, 'PC2'], label=f'Time {t}', alpha=0.7, s=5, color=cmap_time(time_to_idx[t]))
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax2.set_title('PCA by Time')
    if len(times) > 0:
         ax2.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), title="Time Points", fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    return fig


def plot_reconstruction_comparison(original_data, reconstructed_data, indices=None, n_samples=5, n_features=20, figsize=(15, 10)):
    """
    Plot comparison between original data and reconstructed data
    """
    if indices is None:
        indices = np.random.choice(original_data.shape[0], n_samples, replace=False)
    else:
        n_samples = len(indices)
    fig, axes = plt.subplots(n_samples, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]
        orig = original_data[idx]
        recon = reconstructed_data[idx]
        top_features = np.argsort(orig)[-n_features:] if not np.all(orig == 0) else np.arange(min(n_features, len(orig)))
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
    """
    if dims_to_plot is None:
        var_per_dim = np.var(latent_codes, axis=0)
        n_dims = min(n_dims, latent_codes.shape[1])
        dims_to_plot = np.argsort(var_per_dim)[-n_dims:]
    n_dims = len(dims_to_plot)
    tp_unique = np.sort(adata.obs[time_key].unique())
    n_tp = len(tp_unique)
    fig, axes = plt.subplots(n_dims, n_tp, figsize=figsize, squeeze=False)

    for i, dim in enumerate(dims_to_plot):
        vmin = np.min(latent_codes[:, dim])
        vmax = np.max(latent_codes[:, dim])
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sc_plot = None

        for j, t in enumerate(tp_unique):
            ax = axes[i, j]
            mask = (adata.obs[time_key] == t)
            coords_t = adata.obsm[spatial_key][mask]
            z_t = latent_codes[mask]
            if coords_t.shape[0] > 0:
                 sc_plot = ax.scatter(coords_t[:, 0], coords_t[:, 1], c=z_t[:, dim], cmap=cmap, s=3, alpha=0.7, norm=norm)
            else:
                 ax.text(0.5, 0.5, 'No cells', ha='center', va='center', transform=ax.transAxes, fontsize='small')
            if i == 0:
                ax.set_title(f"Time: {t}")
            if j == 0:
                ax.set_ylabel(f"Dim {dim}", fontsize='small')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal', adjustable='box')

        if sc_plot is not None:
            cbar_ax = fig.add_axes([0.92, axes[i,0].get_position().y0, 0.015, axes[i,0].get_position().height])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(f"Dim {dim} Val", fontsize='small')
            cbar.ax.tick_params(labelsize='small')

    plt.subplots_adjust(left=0.05, right=0.9, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)
    return fig


def plot_spatial_clusters(adata, labels, time_key, spatial_key, cmap='tab20',
                         figsize=(18, 5), title="Spatial Clustering"):
    """
    Plot clusters in spatial coordinates
    """
    tp_unique = np.sort(adata.obs[time_key].unique())
    n_tp = len(tp_unique)
    labels = pd.Categorical(labels)
    clusters = np.sort(labels.categories)
    n_clusters = len(clusters)
    fig, axes = plt.subplots(1, n_tp, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    cmap_obj = cm.get_cmap(cmap, n_clusters)
    cluster_to_idx = {clust: i for i, clust in enumerate(clusters)}

    for i, t in enumerate(tp_unique):
        ax = axes[i]
        mask = (adata.obs[time_key] == t)
        coords_t = adata.obsm[spatial_key][mask]
        labels_t = labels[mask]
        if coords_t.shape[0] > 0:
             colors = [cluster_to_idx[lbl] for lbl in labels_t]
             sc = ax.scatter(coords_t[:, 0], coords_t[:, 1], c=colors, cmap=cmap_obj, s=5, alpha=0.7, vmin=0, vmax=n_clusters-1)
        else:
             ax.text(0.5, 0.5, 'No cells', ha='center', va='center', transform=ax.transAxes, fontsize='small')
        ax.set_title(f"Time: {t}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal', adjustable='box')

    if 0 < n_clusters <= 20:
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=cmap_obj(cluster_to_idx[clust]),
                                   markersize=8)
                         for clust in clusters]
        fig.legend(legend_handles, [f'Cluster {i}' for i in clusters],
                  loc='lower center', bbox_to_anchor=(0.5, -0.1 if n_clusters <= 10 else -0.15),
                  ncol=min(10, n_clusters), title="Clusters")
        plt.subplots_adjust(bottom=0.25 if n_clusters > 5 else 0.15)
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_1d_compression_by_time(adata, compressed_1d, time_key, spatial_key, method='umap',
                                cmap='viridis', figsize=(18,5)):
    """
    Plot a 1D compression in spatial coordinates for each time point.
    """
    tp_unique = np.sort(adata.obs[time_key].unique())
    n_tp = len(tp_unique)
    fig, axes = plt.subplots(1, n_tp, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    norm = Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sc_plot = None

    for i, t in enumerate(tp_unique):
        ax = axes[i]
        mask = (adata.obs[time_key] == t)
        coords_t = adata.obsm[spatial_key][mask]
        c_t = compressed_1d[mask]
        if coords_t.shape[0] > 0:
            sc_plot = ax.scatter(coords_t[:, 0], coords_t[:, 1], c=c_t, cmap=cmap, s=5, alpha=0.7, norm=norm)
        else:
             ax.text(0.5, 0.5, 'No cells', ha='center', va='center', transform=ax.transAxes, fontsize='small')
        ax.set_title(f"Time: {t}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal', adjustable='box')

    if sc_plot is not None:
         cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.6, aspect=10)
         cbar.set_label(f"1D {method.upper()} Value")
    plt.tight_layout()
    return fig
