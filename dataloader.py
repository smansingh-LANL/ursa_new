import os
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
import netCDF4 as nc
import numpy as np


class NetCDFTrajectoryDataset(Dataset):
    def __init__(self, data_path, variable_name='data', transform=None):
        """
        Note: Recommended to explicitly call close() to close the NetCDF files.
        Args:
            data_path (str): Path to a .nc file (or directory with .nc files) containing the data.
                             Expected shape is either (N_traj, 21, 5, 128, 128) or (N_traj, 21, 128, 128, 5).
            variable_name (str): Name of the variable to extract.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_path = data_path
        self.variable_name = variable_name
        self.transform = transform

        with nc.Dataset(data_path, mode='r') as ds:
            self.N, self.T, self.C, self.H, self.W = ds[variable_name].shape

        self.n_samples_per_traj = self.T - 1   # we want (t, t+1) pairs, so there are T - 1 pairs in a trajectory
        self.n_samples = self.N * self.n_samples_per_traj
        self.ds = None

    def _ensure_open(self):
        """Ensures the NetCDF file is opened per worker."""
        if self.ds is None:
            self.ds = nc.Dataset(self.data_path, "r")

    def __del__(self):
        """Ensure the dataset is closed when the object is destroyed."""
        self.close()

    def close(self):
        """Explicitly close the dataset when done."""
        if self.ds is not None:
            self.ds.close()
            self.ds = None

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        self._ensure_open()

        traj_idx = idx // self.n_samples_per_traj
        time_idx = idx % self.n_samples_per_traj

        x = self.ds[self.variable_name][traj_idx, time_idx]     # snapshot at time t, shape: (C, H, W)
        y = self.ds[self.variable_name][traj_idx, time_idx + 1] # snapshot at time t+1, shape: (C, H, W)

        # replace tracer with energy for CE-RM problem
        if os.path.basename(self.data_path) == 'CE-RM.nc':  
            x[-1] = 0.5 * x[0] * np.sum(x[1:3] ** 2, axis=0) + x[3] / (1.4 - 1.0)
            y[-1] = 0.5 * y[0] * np.sum(y[1:3] ** 2, axis=0) + y[3] / (1.4 - 1.0)

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class NetCDFPairsInTrajectoryDataset(Dataset):
    def __init__(self, data_path, variable_name='data', transform=None):
        """
        Dataset that accesses all possible pairs in a trajectory for all trajectories in a NetCDF file.
        Intended to train rewards models. Recommended to explicitly call close() to close the NetCDF file.

        Args:
            data_path (str): Path to a .nc file (or directory with .nc files) containing the data.
                             Expected shape is either (N_traj, 21, 5, 128, 128) or (N_traj, 21, 128, 128, 5).
            variable_name (str): Name of the variable to extract.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_path = data_path
        self.variable_name = variable_name
        self.transform = transform

        with nc.Dataset(data_path, mode='r') as ds:
            self.N, self.T, self.C, self.H, self.W = ds[variable_name].shape
    
        self.n_samples_per_traj = self.T * self.T     # each trajectory can have T^2 possible pairs
        self.n_samples = self.N * self.n_samples_per_traj
        self.ds = None

    def _ensure_open(self):
        """Ensures the NetCDF file is opened per worker."""
        if self.ds is None:
            self.ds = nc.Dataset(self.data_path, "r")
    
    def __del__(self):
        """Ensure the dataset is closed when the object is destroyed."""
        self.close()

    def close(self):
        """Explicitly close the dataset when done."""
        if self.ds is not None:
            self.ds.close()
            self.ds = None
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        self._ensure_open()
        
        traj_idx = idx // self.n_samples_per_traj
        pair_idx = idx % self.n_samples_per_traj
        time_idx1 = pair_idx // self.T
        time_idx2 = pair_idx % self.T

        x = self.ds[self.variable_name][traj_idx, time_idx1]     # shape: (C, H, W)
        y = self.ds[self.variable_name][traj_idx, time_idx2]     # shape: (C, H, W)

        # replace tracer with energy for CE-RM problem
        if os.path.basename(self.data_path) == 'CE-RM.nc':
            x[-1] = 0.5 * x[0] * np.sum(x[1:3] ** 2, axis=0) + x[3] / (1.4 - 1.0)
            y[-1] = 0.5 * y[0] * np.sum(y[1:3] ** 2, axis=0) + y[3] / (1.4 - 1.0)

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), time_idx1, time_idx2
    

def get_train_val_test_split_from_dir(data_dir, ds_type=NetCDFTrajectoryDataset, 
                                      n_trajs_per_file=None, splits=[0.2, 0.7, 0.1],
                                      file_names = None):
    '''
    Copies what Arvind did for the train val test split, but for my lazy loaders
    '''
    if file_names == None:
        file_names = ["CE-CRP.nc", "CE-Gauss.nc", "CE-KH.nc", "CE-RP.nc"]
    train_ds, val_ds, test_ds = [], [], []

    for fname in file_names:
        file_path = os.path.join(data_dir, fname)
        if "CE-RM.nc" in fname:
            var_name = "solution"
        if "NS-PwC.nc" in fname:
            var_name = "velocity"
        else:
            var_name = "data"

        ds = ds_type(file_path, var_name)
        n_samples_per_traj = ds.n_samples_per_traj
        if n_trajs_per_file is not None:
            ds = Subset(ds, indices=range(n_trajs_per_file * n_samples_per_traj)) # get the first n_trajs
        else:
            n_trajs_per_file = ds.N

        n_train = int(splits[0] * n_trajs_per_file) * n_samples_per_traj
        n_val = int(splits[1] * n_trajs_per_file) * n_samples_per_traj

        train_set = Subset(ds, range(n_train))
        val_set = Subset(ds, range(n_train, n_train + n_val))
        test_set = Subset(ds, range(n_train + n_val, len(ds)))

        train_ds.append(train_set)
        val_ds.append(val_set)
        test_ds.append(test_set)

    train_ds = ConcatDataset(datasets=train_ds)
    val_ds = ConcatDataset(datasets=val_ds)
    test_ds = ConcatDataset(datasets=test_ds)

    return train_ds, val_ds, test_ds

import os
from torch.utils.data import ConcatDataset

def get_dataset_from_file(data_dir, file_name="NS-PwC.nc", ds_type=NetCDFTrajectoryDataset):
    """
    Load a single NetCDF dataset file and return it as a ConcatDataset.

    Args:
        data_dir (str): Directory containing the NetCDF file.
        file_name (str): Name of the NetCDF file.
        ds_type (Dataset): Dataset class to wrap the NetCDF file.

    Returns:
        dataset: ConcatDataset containing the loaded dataset.
    """
    file_path = os.path.join(data_dir, file_name)

    # Set the correct variable name based on file name
    if "CE-RM" in file_name:
        var_name = "solution"
    elif "FNS-KF" in file_name:
        var_name = "solution"
    elif "NS-" in file_name:
        var_name = "velocity"
    elif "Wave-Gauss" in file_name:
        var_name = "solution"
    else:
        var_name = "data"

    print(f"Loading file: {file_path}")
    print(f"Using variable: {var_name}")

    dataset = ds_type(file_path, var_name)

    # Wrap in ConcatDataset for consistency
    dataset = ConcatDataset([dataset])
    return dataset




def close_all_datasets(dataset):
    """Recursively finds and closes all datasets within ConcatDataset and Subset for ensured cleanup."""
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        for d in dataset.datasets:
            close_all_datasets(d)  # Recursively handle all sub-datasets
    elif isinstance(dataset, torch.utils.data.Subset):
        close_all_datasets(dataset.dataset)  # Go down to the base dataset
    elif hasattr(dataset, "close"):
        dataset.close()  # Close the dataset if it has a close() method


if __name__ == '__main__':

    from tqdm import tqdm
    from torch.utils.data import DataLoader

    data_path = '/projects/artimis/PDEgym/CompressibleEuler/downstream/fulldata/CE-RM.nc'
    ds = NetCDFTrajectoryDataset(data_path, variable_name='solution')
    
    print(ds[40][0].size(),len(ds))

    close_all_datasets(ds)



