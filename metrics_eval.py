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
from typing import Tuple, Dict, List, Any

from torch.utils.data import DataLoader

from dataloader import (
    get_train_val_test_splits,
    NetCDFTrajectoryDataset,
    close_all_datasets,
)
from .dataloader import _infer_variable_name
from .physics_rewards import reward_massconservation


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
    parser.add_argument('--path', default='/home/tpc-fkzzs/tpc26-2/data/PDEgym/CE-RP/CE-RP.nc', help='Path to a single .nc file or a directory containing .nc files')
    parser.add_argument('--splits', type=float, nargs=3, default=(0.7, 0.2, 0.1),
                        metavar=('TRAIN', 'VAL', 'TEST'), help='Split fractions (default: 0.7 0.2 0.1)')
    parser.add_argument('--n-trajs', type=int, default=None, help='Limit to first N trajectories (optional)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for DataLoaders')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of DataLoader workers')
    parser.add_argument('--pin-memory', action='store_true', help='Enable pin_memory for DataLoaders')
    parser.add_argument('--persistent-workers', action='store_true', help='Enable persistent_workers (requires num_workers>0)')
    parser.add_argument('--eval-mass', action='store_true', help='Evaluate reward_massconservation per trajectory for (t, t+1) over selected split')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test', 'all'], help='Which split to evaluate for reward metrics')

    args = parser.parse_args()

    # Always show DataLoader summary first
    train_loader, val_loader, test_loader = get_loaders(
        data_path=args.path,
        splits=tuple(args.splits),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        n_trajs=args.n_trajs,
    )

    print(f"Train samples: {len(train_loader.dataset)} | batches: {len(train_loader)}")
    print(f"Val samples:   {len(val_loader.dataset)} | batches: {len(val_loader)}")
    print(f"Test samples:  {len(test_loader.dataset)} | batches: {len(test_loader)}")

    if args.eval_mass:
        results = evaluate_massconservation_per_trajectory(
            data_path=args.path,
            which_split=args.split,
            splits=tuple(args.splits),
            n_trajs=args.n_trajs,
        )
        # Print concise summary: mean per trajectory
        print(f"\nMass conservation reward per trajectory ({args.split} split):")
        for r in results['results']:
            print(f"traj {r['traj_index']}: mean={r['mean_reward']:.6f} (pairs={len(r['rewards'])})")

    # Ensure cleanup of open NetCDF files in loaders
    close_all_datasets(train_loader.dataset)
    close_all_datasets(val_loader.dataset)
    close_all_datasets(test_loader.dataset)


if __name__ == "__main__":
    main()

def evaluate_massconservation_per_trajectory(
    data_path: str,
    *,
    which_split: str = 'train',  # 'train' | 'val' | 'test' | 'all'
    splits: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    n_trajs: int | None = None,
) -> Dict[str, Any]:
    """
    For each trajectory, evaluate reward_massconservation for timesteps t and t+1
    across all available (t, t+1) pairs for that trajectory within the chosen split.

    This directly iterates the underlying NetCDFTrajectoryDataset to preserve
    trajectory ordering (independent of any DataLoader shuffling).

    Returns a dictionary with per-trajectory rewards and basic metadata.
    """
    # Build a base dataset from the single path (directory is also allowed, but this
    # function treats 'splits' as trajectory ranges in a single file). If a directory
    # is passed, we still open a single dataset using the inferred variable name; for
    # multi-file directories consider enhancing this to iterate files.
    var_name = _infer_variable_name(data_path)
    ds = NetCDFTrajectoryDataset(data_path, variable_name=var_name)

    try:
        N_traj = ds.N
        T_pairs = ds.n_samples_per_traj  # number of (t, t+1) pairs per traj

        n_train = int(splits[0] * N_traj)
        n_val = int(splits[1] * N_traj)
        # ensure exact partition
        n_test = max(0, N_traj - n_train - n_val)

        if which_split == 'train':
            start, end = 0, n_train
        elif which_split == 'val':
            start, end = n_train, n_train + n_val
        elif which_split == 'test':
            start, end = n_train + n_val, N_traj
        elif which_split == 'all':
            start, end = 0, N_traj
        else:
            raise ValueError(f"Invalid which_split: {which_split}")

        if n_trajs is not None:
            end = min(end, start + max(0, n_trajs))

        results: List[Dict[str, Any]] = []
        for traj_idx in range(start, end):
            rewards_this_traj: List[float] = []
            base = traj_idx * T_pairs
            for t in range(T_pairs):
                idx = base + t
                x, y = ds[idx]
                r = reward_massconservation(x, y)
                rewards_this_traj.append(float(r.item()))
            mean_r = float(sum(rewards_this_traj) / len(rewards_this_traj)) if rewards_this_traj else float('nan')
            results.append({
                'traj_index': traj_idx,
                'rewards': rewards_this_traj,
                'mean_reward': mean_r,
            })

        return {
            'split': which_split,
            'splits': splits,
            'n_traj_evaluated': end - start,
            'n_time_pairs_per_traj': T_pairs,
            'results': results,
        }
    finally:
        # Ensure the NetCDF file is closed
        ds.close()
