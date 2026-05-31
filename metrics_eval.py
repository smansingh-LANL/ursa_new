"""
metrics_eval.py
----------------
Utilities to obtain train/val/test DataLoaders using ursa_new.dataloader.

Primary entry points:
 - get_loaders(data_path: str, **loader_kwargs) -> (train_loader, val_loader, test_loader)
 - CLI: python -m ursa_new.metrics_eval --path <file_or_dir> [--batch-size 16 ...]

This relies on get_train_val_test_splits from ursa_new.dataloader, which automatically
infers the correct NetCDF variable name from the file name patterns.
"""

from __future__ import annotations

import argparse
from typing import Tuple

from torch.utils.data import DataLoader

from dataloader import (
    get_train_val_test_splits,
    NetCDFTrajectoryDataset,
    close_all_datasets,
)


def get_loaders(
    data_path: str,
    *,
    splits: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    n_trajs: int | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders for a given NetCDF path.

    Args:
        data_path: Path to a single .nc file or a directory containing .nc files.
        splits: Fractions for (train, val, test). Default is (0.7, 0.2, 0.1).
        batch_size: DataLoader batch size.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for faster host-to-device transfer (use with CUDA).
        persistent_workers: Keep workers alive between epochs (requires num_workers > 0).
        n_trajs: Optional limit on number of trajectories to use from the source.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_loader, val_loader, test_loader = get_train_val_test_splits(
        data_path=data_path,
        ds_type=NetCDFTrajectoryDataset,
        splits=splits,
        n_trajs_per_file=n_trajs,
        return_loaders=True,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Obtain train/val/test DataLoaders for NetCDF datasets.")
    parser.add_argument('--path',default='/home/tpc-fkzzs/tpc26-2/data/PDEgym/CE-RP/CE-RP.nc', help='Path to a single .nc file or a directory containing .nc files')
    parser.add_argument('--splits', type=float, nargs=3, default=(0.7, 0.2, 0.1),
                        metavar=('TRAIN', 'VAL', 'TEST'), help='Split fractions (default: 0.7 0.2 0.1)')
    parser.add_argument('--n-trajs', type=int, default=None, help='Limit to first N trajectories (optional)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for DataLoaders')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of DataLoader workers')
    parser.add_argument('--pin-memory', action='store_true', help='Enable pin_memory for DataLoaders')
    parser.add_argument('--persistent-workers', action='store_true', help='Enable persistent_workers (requires num_workers>0)')

    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_loaders(
        data_path=args.path,
        splits=tuple(args.splits),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        n_trajs=args.n_trajs,
    )

    # Print a quick summary
    print(f"Train samples: {len(train_loader.dataset)} | batches: {len(train_loader)}")
    print(f"Val samples:   {len(val_loader.dataset)} | batches: {len(val_loader)}")
    print(f"Test samples:  {len(test_loader.dataset)} | batches: {len(test_loader)}")

    # Ensure cleanup of open NetCDF files
    close_all_datasets(train_loader.dataset)
    close_all_datasets(val_loader.dataset)
    close_all_datasets(test_loader.dataset)


if __name__ == "__main__":
    main()
