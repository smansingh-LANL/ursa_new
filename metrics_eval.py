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
from typing import Tuple, Dict, List, Any, Optional
import numpy as np
import torch

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


# =====================
# Device/helper utilities
# =====================

def _resolve_device(device_str: Optional[str]) -> torch.device:
    if device_str is not None:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device('cuda')
    # Add MPS if desired; on Windows it's not typical. Fallback to CPU.
    return torch.device('cpu')


def _to_device_numpy(arr_np: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(arr_np).contiguous()
    if device.type == 'cuda':
        return t.pin_memory().to(device, non_blocking=True).to(torch.float32)
    return t.to(device=device, dtype=torch.float32)


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
    device: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Evaluate mass, momentum, and energy conservation rewards for qualifying trajectories
    and return a numpy array with shape (3, Ntraj, T-1). Only include trajectories from
    datasets whose T equals 'require_T' (default 21), yielding T-1=20 time pairs.

    Returns:
        (arr, meta) where arr has shape (3, Ntraj, T-1) ordered as [mass, momentum, energy]
        and meta contains basic metadata such as counts and indices.
    """
    dev = _resolve_device(device)
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

        included_indices: List[int] = []
        n_sel = max(0, end - start)
        rewards_t = torch.zeros((3, n_sel, T_pairs), dtype=torch.float32, device=dev)

        # Work from raw dataset to avoid Python-level __getitem__ and move directly to device
        ds._ensure_open()
        var = ds.ds[ds.variable_name]

        write_idx = 0
        for traj_idx in range(start, end):
            for t in range(T_pairs):
                # x_t and y_{t+1}
                snap_x = var[traj_idx, t]
                snap_y = var[traj_idx, t + 1]
                x = _to_device_numpy(snap_x, dev)  # shape (C, H, W)
                y = _to_device_numpy(snap_y, dev)

                r_mass = reward_massconservation(x, y)
                r_mom = reward_momentumconservation(x, y)
                r_en = reward_energyconservation(y, x)  # note reversed args

                rewards_t[0, write_idx, t] = r_mass
                rewards_t[1, write_idx, t] = r_mom
                rewards_t[2, write_idx, t] = r_en

            included_indices.append(traj_idx)
            write_idx += 1

        arr = rewards_t.detach().cpu().numpy()  # (3, Ntraj, T_pairs)

        meta: Dict[str, Any] = {
            'split': which_split,
            'pairs_per_traj': T_pairs,
            'n_traj_included': len(included_indices),
            'traj_indices': included_indices,
        }
        return arr, meta
    finally:
        ds.close()


# =============================
# Power spectrum (radial) utils
# =============================

_R_CACHE: Dict[Tuple[int, int, str, str], torch.Tensor] = {}


def _fft_freq_grids(H: int, W: int, device: torch.device, dtype: torch.dtype):
    fx = torch.fft.fftfreq(H, d=1.0, device=device, dtype=dtype)
    fy = torch.fft.fftfreq(W, d=1.0, device=device, dtype=dtype)
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')
    R = torch.sqrt(FX * FX + FY * FY)
    return FX, FY, R


def _radial_average(P: torch.Tensor, R: torch.Tensor, rmax: float = 0.5, nbins: int = 64) -> torch.Tensor:
    # P and R are HxW
    H, W = P.shape
    device = P.device
    dtype = P.dtype

    # Define bin edges [0, rmax] with nbins bins
    edges = torch.linspace(0.0, rmax, steps=nbins + 1, device=device, dtype=dtype)

    r_flat = R.reshape(-1)
    p_flat = P.reshape(-1)

    # Bin indices in 0..nbins-1
    bin_idx = torch.bucketize(r_flat, edges, right=False) - 1
    valid = (bin_idx >= 0) & (bin_idx < nbins)
    bin_idx = bin_idx[valid]
    p_valid = p_flat[valid]

    # Accumulate sums and counts per bin
    sums = torch.zeros(nbins, device=device, dtype=dtype)
    counts = torch.zeros(nbins, device=device, dtype=dtype)
    sums.scatter_add_(0, bin_idx, p_valid)
    ones = torch.ones_like(p_valid)
    counts.scatter_add_(0, bin_idx, ones)

    # Avoid div by zero
    counts = torch.clamp(counts, min=1.0)
    return sums / counts


def _log_radial_spectrum(x: torch.Tensor, nbins: int = 64, log_spectrum: bool = True) -> torch.Tensor:
    H, W = x.shape
    device = x.device
    dtype = x.dtype
    key = (H, W, str(device), str(dtype))
    R = _R_CACHE.get(key)
    if R is None:
        _, _, R = _fft_freq_grids(H, W, device, dtype)
        _R_CACHE[key] = R
    X = torch.fft.fft2(x)
    P = (X.real ** 2 + X.imag ** 2)
    kspec = _radial_average(P, R, rmax=0.5, nbins=nbins)
    return torch.log(kspec + 1e-20) if log_spectrum else kspec


def evaluate_radial_power_spectra(
    data_path: str,
    *,
    which_split: str = 'train',
    splits: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    n_trajs: int | None = None,
    require_T: int | None = 21,
    nbins: int = 64,
    log_spectrum: bool = True,
    channel: int = 0,
    all_channels: bool = False,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute radial power spectra for the specified channel across timesteps and trajectories.

    Returns:
        (arr, meta) where arr has shape (Ntraj, T, nbins)
    """
    dev = _resolve_device(device)
    var_name = _infer_variable_name(data_path)
    ds = NetCDFTrajectoryDataset(data_path, variable_name=var_name)

    try:
        N_traj = ds.N
        T = ds.T
        C = ds.C
        if (require_T is not None) and (T != require_T):
            empty = np.zeros(((C if all_channels else 0), 0, 0, nbins) if all_channels else (0, 0, nbins), dtype=np.float32)
            return empty, {
                'split': which_split,
                'n_traj_included': 0,
                'T': T,
                'reason': f"Dataset T={T} does not match require_T={require_T}",
            }

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

        n_sel = max(0, end - start)
        if all_channels:
            arr_t = torch.zeros((C, n_sel, T, nbins), dtype=torch.float32, device=dev)
        else:
            arr_t = torch.zeros((n_sel, T, nbins), dtype=torch.float32, device=dev)

        # Ensure underlying file is open for direct access
        ds._ensure_open()
        var = ds.ds[ds.variable_name]
        # Iterate selected trajectories and all timesteps
        write_idx = 0
        for traj_idx in range(start, end):
            for t in range(T):
                # snapshot shape: (C, H, W)
                snap = var[traj_idx, t]
                if all_channels:
                    for c in range(C):
                        xch_cpu = torch.from_numpy(snap[c]).contiguous()
                        if dev.type == 'cuda':
                            xch = xch_cpu.pin_memory().to(dev, non_blocking=True)
                        else:
                            xch = xch_cpu.to(dev)
                        xch = xch.to(torch.float32)
                        spec = _log_radial_spectrum(xch, nbins=nbins, log_spectrum=log_spectrum)
                        arr_t[c, write_idx, t, :] = spec
                else:
                    xch_cpu = torch.from_numpy(snap[channel]).contiguous()
                    if dev.type == 'cuda':
                        xch = xch_cpu.pin_memory().to(dev, non_blocking=True)
                    else:
                        xch = xch_cpu.to(dev)
                    xch = xch.to(torch.float32)
                    spec = _log_radial_spectrum(xch, nbins=nbins, log_spectrum=log_spectrum)
                    arr_t[write_idx, t, :] = spec
            write_idx += 1

        meta: Dict[str, Any] = {
            'split': which_split,
            'n_traj_included': n_sel,
            'T': T,
            'nbins': nbins,
            'C': C,
            'channel': (None if all_channels else channel),
        }
        arr_np = arr_t.detach().cpu().numpy()
        if all_channels:
            # Ensure output order (C, Ntraj, T, nbins)
            return arr_np, meta
        else:
            return arr_np, meta
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

    names = [
        "Mass Conservation",
        "Momentum Conservation",
        "Energy Conservation",
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    for i, ax in enumerate(axes):
        data = arr[i].reshape(-1)  # flatten across trajectories and time pairs
        data = np.abs(data)        # histogram over absolute values as requested
        ax.hist(data, bins=bins, density=density, alpha=0.8, color='#1f77b4', edgecolor='black')
        ax.set_xlabel(f"x = {names[i]}")
        ax.set_ylabel("P(x)")
        ax.set_title(names[i])

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
    parser.add_argument('--eval-ps', action='store_true', help='Evaluate radial power spectra for a channel over timesteps and trajectories')
    parser.add_argument('--nbins', type=int, default=64, help='Number of radial bins for power spectrum (default: 64)')
    parser.add_argument('--no-log-spectrum', action='store_true', help='Use linear spectrum instead of log(kspec+1e-20)')
    parser.add_argument('--channel', type=int, default=0, help='Channel index for power spectrum (default: 0)')
    parser.add_argument('--all-channels', action='store_true', help='Compute spectra for all channels and save (C,Ntraj,T,nbins)')
    parser.add_argument('--save-ps', type=str, default=None, help='Path to save power spectra array (Ntraj,T,nbins)')
    parser.add_argument('--device', type=str, default=None, help='Torch device to use (e.g., cuda, cuda:0, cpu). Defaults to CUDA if available.')
    parser.add_argument('--amp', action='store_true', help='Enable torch.cuda.amp.autocast for power spectra (experimental)')

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
            device=args.device,
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

    if args.eval_ps:
        arr_ps, meta_ps = evaluate_radial_power_spectra(
            data_path=args.path,
            which_split=args.split,
            splits=tuple(args.splits),
            n_trajs=args.n_trajs,
            require_T=args.require_T,
            nbins=args.nbins,
            log_spectrum=(not args.no_log_spectrum),
            channel=args.channel,
            all_channels=args.all_channels,
            device=args.device,
        )
        expected = "(C, Ntraj, T, nbins)" if args.all_channels else "(Ntraj, T, nbins)"
        print(f"\nPower spectra shape: {arr_ps.shape} (expected {expected})")
        print(f"Included trajectories: {meta_ps['n_traj_included']} | T: {meta_ps['T']} | nbins: {meta_ps['nbins']} | C: {meta_ps['C']} | channel: {meta_ps['channel']}")
        default_name = f"power_spectra_{args.split}_allch.npy" if args.all_channels else f"power_spectra_{args.split}.npy"
        out_ps = args.save_ps or default_name
        np.save(out_ps, arr_ps)
        print(f"Saved power spectra to: {out_ps}")

    # Ensure cleanup of open NetCDF files in loaders
    close_all_datasets(train_loader.dataset)
    close_all_datasets(val_loader.dataset)
    close_all_datasets(test_loader.dataset)


if __name__ == "__main__":
    main()
