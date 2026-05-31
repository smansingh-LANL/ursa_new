import os
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader
import netCDF4 as nc
import numpy as np
from typing import Optional, Tuple, List, Union


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
    

def _infer_variable_name(file_name: str) -> str:
    """Infer the NetCDF variable name based on common file-name patterns.

    This harmonizes the logic across helpers in this module.
    """
    base = os.path.basename(file_name)
    if "CE-RM" in base:
        return "solution"
    elif "FNS-KF" in base:
        return "solution"
    elif "NS-" in base:
        return "velocity"
    elif "Wave-Gauss" in base:
        return "solution"
    else:
        return "data"

def get_train_val_test_split_from_dir(data_dir,
                                      ds_type=NetCDFTrajectoryDataset,
                                      n_trajs_per_file: Optional[int] = None,
                                      splits: List[float] = [0.2, 0.7, 0.1],
                                      file_names: Optional[List[str]] = None):
    '''
    Copies what Arvind did for the train val test split, but for my lazy loaders
    '''
    if file_names == None:
        file_names = ["CE-CRP.nc", "CE-Gauss.nc", "CE-KH.nc", "CE-RP.nc"]
    train_ds, val_ds, test_ds = [], [], []

    for fname in file_names:
        file_path = os.path.join(data_dir, fname)
        var_name = _infer_variable_name(fname)

        ds = ds_type(file_path, var_name)
        n_samples_per_traj = ds.n_samples_per_traj
        if n_trajs_per_file is not None:
            ds = Subset(ds, indices=range(n_trajs_per_file * n_samples_per_traj))  # first n_trajs
            n_trajs_this_file = n_trajs_per_file
        else:
            n_trajs_this_file = ds.N

        # compute counts per file (by trajectory) and convert to sample ranges
        n_train_traj = int(splits[0] * n_trajs_this_file)
        n_val_traj = int(splits[1] * n_trajs_this_file)
        n_test_traj = max(0, n_trajs_this_file - n_train_traj - n_val_traj)

        n_train = n_train_traj * n_samples_per_traj
        n_val = n_val_traj * n_samples_per_traj

        train_set = Subset(ds, range(n_train))
        val_set = Subset(ds, range(n_train, n_train + n_val))
        test_set = Subset(ds, range(n_train + n_val, n_train + n_val + n_test_traj * n_samples_per_traj))

        train_ds.append(train_set)
        val_ds.append(val_set)
        test_ds.append(test_set)

    train_ds = ConcatDataset(datasets=train_ds)
    val_ds = ConcatDataset(datasets=val_ds)
    test_ds = ConcatDataset(datasets=test_ds)

    return train_ds, val_ds, test_ds

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
    var_name = _infer_variable_name(file_name)

    print(f"Loading file: {file_path}")
    print(f"Using variable: {var_name}")

    dataset = ds_type(file_path, var_name)

    # Wrap in ConcatDataset for consistency
    dataset = ConcatDataset([dataset])
    return dataset


def get_train_val_test_splits(
    data_path: str,
    ds_type=NetCDFTrajectoryDataset,
    splits: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    n_trajs_per_file: Optional[int] = None,
    return_loaders: bool = True,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> Union[Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset],
           Tuple[DataLoader, DataLoader, DataLoader]]:
    """
    Create train/val/test splits from a single NetCDF file path (or delegate to directory helper).

    Args:
        data_path: Path to a single .nc file or a directory. For a directory, delegates to
                   get_train_val_test_split_from_dir using the provided ds_type and splits.
        ds_type: Dataset class to wrap the NetCDF file (default: NetCDFTrajectoryDataset).
        splits: Fractions for (train, val, test). Default is (0.7, 0.2, 0.1).
        n_trajs_per_file: If provided, limit to the first N trajectories in the file.
        return_loaders: If True (default), return PyTorch DataLoaders. If False, return Datasets.
        batch_size: DataLoader batch size when return_loaders is True.
        shuffle: Whether to shuffle the training DataLoader.
        num_workers: DataLoader workers.
        pin_memory: DataLoader pin_memory flag.
        persistent_workers: DataLoader persistent_workers flag (only applies if num_workers > 0).

    Returns:
        - If return_loaders is False: (train_ds, val_ds, test_ds)
        - If return_loaders is True: (train_loader, val_loader, test_loader)
    """
    if os.path.isdir(data_path):
        # Delegate to the existing multi-file helper if a directory is passed.
        train_ds, val_ds, test_ds = get_train_val_test_split_from_dir(
            data_path, ds_type=ds_type, n_trajs_per_file=n_trajs_per_file, splits=list(splits)
        )
    else:
        # Single file case
        var_name = _infer_variable_name(data_path)
        ds = ds_type(data_path, var_name)

        n_samples_per_traj = ds.n_samples_per_traj
        total_trajs = ds.N if n_trajs_per_file is None else min(n_trajs_per_file, ds.N)

        n_train_traj = int(splits[0] * total_trajs)
        n_val_traj = int(splits[1] * total_trajs)
        n_test_traj = max(0, total_trajs - n_train_traj - n_val_traj)

        n_train = n_train_traj * n_samples_per_traj
        n_val = n_val_traj * n_samples_per_traj
        n_test = n_test_traj * n_samples_per_traj

        base_subset = ds if n_trajs_per_file is None else Subset(ds, range(total_trajs * n_samples_per_traj))

        train_ds = Subset(base_subset, range(n_train))
        val_ds = Subset(base_subset, range(n_train, n_train + n_val))
        test_ds = Subset(base_subset, range(n_train + n_val, n_train + n_val + n_test))

    if not return_loaders:
        return train_ds, val_ds, test_ds

    # Build DataLoaders (typical settings: shuffle for train only)
    pw = persistent_workers if num_workers > 0 else False
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=pw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory,
                            persistent_workers=pw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory,
                             persistent_workers=pw)

    return train_loader, val_loader, test_loader




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
    import argparse

    parser = argparse.ArgumentParser(description="Create train/val/test splits from a NetCDF dataset.")
    parser.add_argument('--path', required=True, help='Path to a single .nc file or a directory containing .nc files')
    parser.add_argument('--splits', type=float, nargs=3, default=(0.7, 0.2, 0.1),
                        metavar=('TRAIN', 'VAL', 'TEST'), help='Split fractions (default: 0.7 0.2 0.1)')
    parser.add_argument('--n-trajs', type=int, default=None, help='Limit to first N trajectories (optional)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for DataLoaders')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of DataLoader workers')
    parser.add_argument('--pin-memory', action='store_true', help='Enable pin_memory for DataLoaders')
    parser.add_argument('--persistent-workers', action='store_true', help='Enable persistent_workers (requires num_workers>0)')
    parser.add_argument('--datasets', action='store_true', help='Return Datasets instead of DataLoaders')

    args = parser.parse_args()

    outputs = get_train_val_test_splits(
        data_path=args.path,
        ds_type=NetCDFTrajectoryDataset,
        splits=tuple(args.splits),
        n_trajs_per_file=args.n_trajs,
        return_loaders=not args.datasets,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )

    if args.datasets:
        train_ds, val_ds, test_ds = outputs
        print(f"Train samples: {len(train_ds)}")
        print(f"Val samples:   {len(val_ds)}")
        print(f"Test samples:  {len(test_ds)}")
        # Ensure cleanup of open NetCDF files
        close_all_datasets(train_ds)
        close_all_datasets(val_ds)
        close_all_datasets(test_ds)
    else:
        train_loader, val_loader, test_loader = outputs
        print(f"Train samples: {len(train_loader.dataset)} | batches: {len(train_loader)}")
        print(f"Val samples:   {len(val_loader.dataset)} | batches: {len(val_loader)}")
        print(f"Test samples:  {len(test_loader.dataset)} | batches: {len(test_loader)}")
        # Ensure cleanup of open NetCDF files
        close_all_datasets(train_loader.dataset)
        close_all_datasets(val_loader.dataset)
        close_all_datasets(test_loader.dataset)




