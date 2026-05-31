"""
plot_power_spectra.py
----------------------
Plot radial power spectra saved to disk (from metrics_eval.evaluate_radial_power_spectra).

Input array format:
- Either a .npy file containing an array of shape (Ntraj, T, nbins)
- Or a .npz file with one or more arrays; we will attempt to load a key named
  'arr' or 'spectra' if present, otherwise the first array.

The script aggregates over trajectories (mean by default) to produce a (T, nbins)
profile, then plots all T curves on the same axes, colored from red (early time)
to blue (late time). No legend is drawn as requested.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np


def _load_power_spectra(path: str) -> np.ndarray:
    obj = np.load(path, allow_pickle=False)
    try:
        # .npz case -> pick sensible key, else first
        if isinstance(obj, np.lib.npyio.NpzFile):
            for key in ("arr", "spectra", "data"):
                if key in obj.files:
                    return obj[key]
            # fallback to first entry
            if len(obj.files) == 0:
                raise ValueError(f"Empty npz file: {path}")
            return obj[obj.files[0]]
        # .npy case -> ndarray directly
        if isinstance(obj, np.ndarray):
            return obj
        raise ValueError(f"Unsupported file contents in: {path}")
    finally:
        # NpzFile needs explicit close; ndarray does not
        try:
            obj.close()  # type: ignore[attr-defined]
        except Exception:
            pass


def aggregate_over_trajectories(arr: np.ndarray, method: str = "mean") -> np.ndarray:
    """
    Aggregate spectra over trajectories.

    Args:
        arr: (Ntraj, T, nbins)
        method: 'mean' or 'median'

    Returns:
        (T, nbins)
    """
    if arr.ndim == 3:
        # (Ntraj, T, nbins) -> (T, nbins)
        if method == "mean":
            return arr.mean(axis=0)
        if method == "median":
            return np.median(arr, axis=0)
    elif arr.ndim == 4:
        # (C, Ntraj, T, nbins) -> (C, T, nbins)
        if method == "mean":
            return arr.mean(axis=1)
        if method == "median":
            return np.median(arr, axis=1)
    else:
        raise ValueError(f"Expected input array with shape (Ntraj, T, nbins) or (C, Ntraj, T, nbins); got {arr.shape}")
    raise ValueError("method must be 'mean' or 'median'")


def plot_power_spectra_over_time(
    arr_T_nb: np.ndarray,
    *,
    rmax: float = 0.5,
    save_path: str | None = None,
    show: bool = True,
    title: str = "Radial Power Spectra over Time",
) -> None:
    """
    Plot all timesteps' spectra on the same axes, colored red->blue as time increases.

    Args:
        arr_T_nb: (T, nbins)
        rmax: radial max used during binning (default 0.5)
        save_path: file path to save the figure (e.g., .png)
        show: whether to display the figure interactively
        title: plot title
    """
    import matplotlib.pyplot as plt

    if arr_T_nb.ndim == 2:
        # Single channel case: (T, nbins)
        arr_list = [arr_T_nb]
        titles = [title]
    elif arr_T_nb.ndim == 3:
        # Multi-channel aggregated case: (C, T, nbins)
        arr_list = [arr_T_nb[c] for c in range(arr_T_nb.shape[0])]
        titles = [f"{title} (ch={c})" for c in range(arr_T_nb.shape[0])]
    else:
        raise ValueError(f"Expected (T, nbins) or (C, T, nbins); got {arr_T_nb.shape}")

    # Determine nbins from the first entry and build bin centers
    T0, nbins = arr_list[0].shape
    edges = np.linspace(0.0, rmax, nbins + 1, dtype=np.float32)
    centers = 0.5 * (edges[:-1] + edges[1:])

    cmap = plt.get_cmap("RdYlBu")  # 0->blue, 1->red, so invert by (1 - frac)
    eps = 1e-9

    ncols = min(3, len(arr_list))
    nrows = int(np.ceil(len(arr_list) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            if idx >= len(arr_list):
                axes[r, c].axis('off')
                continue
            cur = arr_list[idx]
            T = cur.shape[0]
            for t in range(T):
                frac = t / max(1, (T - 1))
                color = cmap(1.0 - frac)  # t=0 -> red, t=T-1 -> blue
                axes[r, c].plot(centers, cur[t], color=color, linewidth=1.25)
            axes[r, c].set_xlabel("k (radial frequency)")
            axes[r, c].set_ylabel("P(k)")
            axes[r, c].set_title(titles[idx])
            idx += 1

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot aggregated radial power spectra over time.")
    parser.add_argument("--path", required=True, help="Path to .npz or .npy file with (Ntraj, T, nbins) or (C, Ntraj, T, nbins)")
    parser.add_argument("--aggregate", choices=["mean", "median"], default="mean", help="Aggregation across trajectories (default: mean)")
    parser.add_argument("--rmax", type=float, default=0.5, help="Radial max used during binning (default: 0.5)")
    parser.add_argument("--save-fig", type=str, default=None, help="If provided, save figure to this path (e.g., .png)")
    parser.add_argument("--no-show", action="store_true", help="Do not display the figure interactively")
    parser.add_argument("--title", type=str, default="Radial Power Spectra over Time", help="Figure title")

    args = parser.parse_args()

    arr = _load_power_spectra(args.path)
    if arr.ndim not in (3, 4):
        raise ValueError(f"Expected (Ntraj, T, nbins) or (C, Ntraj, T, nbins); got {arr.shape}")

    agg = aggregate_over_trajectories(arr, method=args.aggregate)
    # agg: (T, nbins) if 3D input; (C, T, nbins) if 4D input
    plot_power_spectra_over_time(
        agg,
        rmax=args.rmax,
        save_path=args.save_fig,
        show=(not args.no_show),
        title=args.title,
    )


if __name__ == "__main__":
    main()
