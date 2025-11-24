import torch
import numpy as np
import scanpy as sc
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp

class SingleCellDataset(Dataset):
    """
    Dataset for single-cell RNA-seq data
    """
    def __init__(self, adata, transform=None):
        """
        Args:
            adata: AnnData object containing expression data
            transform: Optional transform to apply to the data
        """
        # Make sure we're working with a matrix
        if sp.issparse(adata.X):
            self.X = adata.X.toarray()
        else:
            self.X = adata.X

        # Convert to float tensor
        self.X = torch.FloatTensor(self.X)

        # Store the observation metadata
        self.obs = adata.obs

        # Store spatial coordinates if available
        if 'spatial' in adata.obsm:
            self.spatial = torch.FloatTensor(adata.obsm['spatial'])
        else:
            self.spatial = None

        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]

        if self.transform:
            x = self.transform(x)

        return x


class TimeBasedDataset(Dataset):
    """
    Dataset that splits data by time points
    """
    def __init__(self, adata, time_key='timepoint', transform=None):
        """
        Args:
            adata: AnnData object containing expression data
            time_key: Key in adata.obs for time information
            transform: Optional transform to apply to the data
        """
        self.adata = adata
        self.time_key = time_key
        self.transform = transform

        # Get unique time points
        self.time_points = sorted(adata.obs[time_key].unique())

        # Create indices for each time point
        self.time_indices = {}
        for t in self.time_points:
            self.time_indices[t] = np.where(adata.obs[time_key] == t)[0]

    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, idx):
        # Get data
        if sp.issparse(self.adata.X):
            x = torch.FloatTensor(self.adata.X[idx].toarray().squeeze())
        else:
            x = torch.FloatTensor(self.adata.X[idx])

        # Get time point
        t = self.adata.obs[self.time_key].iloc[idx]

        # Get spatial coordinates if available
        if 'spatial' in self.adata.obsm:
            spatial = torch.FloatTensor(self.adata.obsm['spatial'][idx])
        else:
            spatial = None

        if self.transform:
            x = self.transform(x)

        return {'x': x, 'time': t, 'spatial': spatial, 'idx': idx}

    def get_time_loader(self, t, batch_size=128, shuffle=False):
        """
        Create a DataLoader for a specific time point
        """
        indices = self.time_indices[t]

        # Create subset dataset for this time point
        if sp.issparse(self.adata.X):
            X_t = self.adata.X[indices].toarray()
        else:
            X_t = self.adata.X[indices]

        # Create SingleCellDataset
        dataset = SingleCellDataset(self.adata[indices])

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class SingleTimeDataset(Dataset):
    """
    Dataset for a single time point in spatio-temporal data
    """
    def __init__(self, X, spatial_coords=None, transform=None):
        """
        Args:
            X: Expression data for a single time point
            spatial_coords: Spatial coordinates (optional)
            transform: Optional transform to apply to the data
        """
        # Convert to numpy array if sparse
        if sp.issparse(X):
            X = X.toarray()

        # Convert to tensor
        self.X = torch.FloatTensor(X)

        # Store spatial coordinates if available
        if spatial_coords is not None:
            self.spatial = torch.FloatTensor(spatial_coords)
        else:
            self.spatial = None

        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]

        if self.transform:
            x = self.transform(x)

        if self.spatial is not None:
            spatial = self.spatial[idx]
            return {'x': x, 'spatial': spatial, 'idx': idx}

        return x


def create_dataloaders(adata, time_key='timepoint', batch_size=128,
                       train_ratio=0.8, val_ratio=0.1, shuffle=True,
                       random_state=42):
    """
    Create train, validation, and test DataLoaders from an AnnData object
    """
    np.random.seed(random_state)

    # Total number of cells
    n_cells = adata.shape[0]

    # Create indices and shuffle them
    indices = np.arange(n_cells)
    if shuffle:
        np.random.shuffle(indices)

    # Split indices
    train_size = int(train_ratio * n_cells)
    val_size = int(val_ratio * n_cells)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    # Create datasets
    train_dataset = SingleCellDataset(adata[train_indices])
    val_dataset = SingleCellDataset(adata[val_indices])
    test_dataset = SingleCellDataset(adata[test_indices])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader