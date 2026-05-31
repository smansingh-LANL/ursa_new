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
import numpy as np

from torch.utils.data import DataLoader

from dataloader import (
    get_train_val_test_splits,
    NetCDFTrajectoryDataset,
    close_all_datasets,
)
from dataloader import _infer_variable_name
from physics_rewards import (
    reward_massconservation,
    reward_momentumconservation,
    reward_energyconservation,
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


def evaluate_conservation_rewards_numpy(
    data_path: str,
    *,
    which_split: str = 'train',
    splits: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    n_trajs: int | None = None,
    require_T: int = 21,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Evaluate mass, momentum, and energy conservation rewards for qualifying trajectories
    and return a numpy array with shape (3, Ntraj, T-1). Only include trajectories from
    datasets whose T equals 'require_T' (default 21), yielding T-1=20 time pairs.

    Returns:
        (arr, meta) where arr has shape (3, Ntraj, T-1) ordered as [mass, momentum, energy]
        and meta contains basic metadata such as counts and indices.
    """
    var_name = _infer_variable_name(data_path)
    ds = NetCDFTrajectoryDataset(data_path, variable_name=var_name)

    try:
        # Filter by required T
        if getattr(ds, 'T', None) != require_T:
            # No trajectories included if the file doesn't match requirement
            empty = np.zeros((3, 0, max(0, require_T - 1)), dtype=np.float32)
            return empty, {
                'split': which_split,
                'pairs_per_traj': require_T - 1,
                'n_traj_included': 0,
                'reason': f"Dataset T={ds.T} does not match require_T={require_T}",
            }

        N_traj = ds.N
        T_pairs = ds.n_samples_per_traj  # should be require_T - 1

        # Determine split ranges by trajectory index
        n_train = int(splits[0] * N_traj)
        n_val = int(splits[1] * N_traj)
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

        mass_all: List[List[float]] = []
        mom_all: List[List[float]] = []
        en_all: List[List[float]] = []
        included_indices: List[int] = []

        for traj_idx in range(start, end):
            base = traj_idx * T_pairs
            mass_vals: List[float] = []
            mom_vals: List[float] = []
            en_vals: List[float] = []
            for t in range(T_pairs):
                idx = base + t
                x, y = ds[idx]
                r_mass = reward_massconservation(x, y)
                r_mom = reward_momentumconservation(x, y)
                r_en = reward_energyconservation(y, x)  # note reversed args
                mass_vals.append(float(r_mass.item()))
                mom_vals.append(float(r_mom.item()))
                en_vals.append(float(r_en.item()))

            mass_all.append(mass_vals)
            mom_all.append(mom_vals)
            en_all.append(en_vals)
            included_indices.append(traj_idx)

        if len(included_indices) == 0:
            arr = np.zeros((3, 0, T_pairs), dtype=np.float32)
        else:
            arr = np.stack([
                np.array(mass_all, dtype=np.float32),
                np.array(mom_all, dtype=np.float32),
                np.array(en_all, dtype=np.float32),
            ], axis=0)  # (3, Ntraj, T_pairs)

        meta: Dict[str, Any] = {
            'split': which_split,
            'pairs_per_traj': T_pairs,
            'n_traj_included': len(included_indices),
            'traj_indices': included_indices,
        }
        return arr, meta
    finally:
        ds.close()


def plot_conservation_histograms(
    arr: np.ndarray,
    *,
    bins: int = 50,
    density: bool = True,
    figsize: Tuple[float, float] = (12, 4),
    save_path: str | None = None,
) -> None:
    """
    Plot three histograms (mass, momentum, energy) from an array shaped (3, Ntraj, T-1).

    Args:
        arr: numpy array with shape (3, Ntraj, T-1)
        bins: number of histogram bins
        density: whether to normalize the histogram to form a probability density
        figsize: figure size
        save_path: if provided, save figure to this path; otherwise show interactively
    """
    # Lazy import to avoid hard dependency at import time
    import matplotlib.pyplot as plt

    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"Expected arr shape (3, Ntraj, T-1); got {arr.shape}")

    labels = [
        ("Mass Conservation", "P(x)"),
        ("Momentum Conservation", "P(x)"),
        ("Energy Conservation", "P(x)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    for i, ax in enumerate(axes):
        data = arr[i].reshape(-1)  # flatten across trajectories and time pairs
        ax.hist(data, bins=bins, density=density, alpha=0.8, color='#1f77b4', edgecolor='black')
        ax.set_xlabel(labels[i][0])
        ax.set_ylabel(labels[i][1])
        ax.set_title(labels[i][0])

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


def plot_conservation_histograms_from_file(
    npy_path: str,
    *,
    bins: int = 50,
    density: bool = True,
    figsize: Tuple[float, float] = (12, 4),
    save_path: str | None = None,
) -> None:
    """
    Convenience wrapper to load a saved (3, Ntraj, T-1) .npy array and plot histograms.
    """
    arr = np.load(npy_path)
    plot_conservation_histograms(arr, bins=bins, density=density, figsize=figsize, save_path=save_path)


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
    parser.add_argument('--eval-all', action='store_true', help='Evaluate mass, momentum, energy for trajectories with full T and save as (3,Ntraj,20) numpy array')
    parser.add_argument('--require-T', type=int, default=21, help='Only include trajectories from files with exactly this many timesteps (default: 21)')
    parser.add_argument('--save-npy', type=str, default=None, help='Path to save the (3,Ntraj,20) numpy file (optional)')
    parser.add_argument('--plot-npy', type=str, default=None, help='Path to a saved (3,Ntraj,20) numpy array to plot as histograms')
    parser.add_argument('--save-fig', type=str, default=None, help='If provided, save the histogram figure to this path (e.g., .png)')

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

    if args.eval_all:
        arr, meta = evaluate_conservation_rewards_numpy(
            data_path=args.path,
            which_split=args.split,
            splits=tuple(args.splits),
            n_trajs=args.n_trajs,
            require_T=args.require_T,
        )
        print(f"\nAll conservation rewards array shape: {arr.shape} (expected (3, Ntraj, 20))")
        print(f"Included trajectories: {meta['n_traj_included']} | pairs per traj: {meta['pairs_per_traj']}")
        out_path = args.save_npy
        if out_path is None:
            # default file name based on split
            out_path = f"conservation_rewards_{args.split}.npy"
        np.save(out_path, arr)
        print(f"Saved numpy array to: {out_path}")

    if args.plot_npy is not None:
        plot_conservation_histograms_from_file(args.plot_npy, save_path=args.save_fig)

    # Ensure cleanup of open NetCDF files in loaders
    close_all_datasets(train_loader.dataset)
    close_all_datasets(val_loader.dataset)
    close_all_datasets(test_loader.dataset)


if __name__ == "__main__":
    main()
