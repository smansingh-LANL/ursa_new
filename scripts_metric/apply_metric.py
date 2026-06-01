"""
Apply euler_metrics to REAL CE-RP ground-truth trajectories (read-only).

We do NOT run any model inference. We load one (or a few) real trajectories
from the PDEgym CE-RP chunk files and evaluate the physics metrics on the
ground-truth time evolution t -> t+1.

Sanity expectations on ground truth:
  - Conservation rewards should be near 0 (real physics conserves them).
  - Density/pressure should stay positive (no violations).
  - Anisotropy A should be strongly non-zero (axis-aligned Riemann problem),
    UNLIKE isotropic random noise.

All reads are lazy/sliced; nothing is modified. Output is written ONLY to
our own folder.
"""

import os
import json
import numpy as np
import netCDF4 as nc

import euler_metrics as em

# Read-only source (a single chunk; owned by another user — we only read)
DATA_FILE = "/home/tpc-sjqyt/tpc26-2/data/PDEgym/CE-RP/data_0.nc"
VAR = "data"

# Our own output location
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(OUT_DIR, "metric_report_CE-RP.json")

N_TRAJ = 3          # how many trajectories to scan
TRAJ_IDS = [0, 100, 500]


def load_trajectory(ds, traj_idx):
    """Lazily read one trajectory: shape (T, 5, H, W)."""
    return np.asarray(ds.variables[VAR][traj_idx])   # (21, 5, 128, 128)


def main():
    ds = nc.Dataset(DATA_FILE, "r")
    N, T, C, H, W = ds.variables[VAR].shape
    print(f"Dataset: {DATA_FILE}")
    print(f"  shape = (N={N}, T={T}, C={C}, H={H}, W={W})  [trajectories, time, channels, y, x]")
    print(f"  channels = [rho, u, v, p, E]\n")

    report = {"data_file": DATA_FILE, "shape": [N, T, C, H, W], "trajectories": {}}

    for tid in TRAJ_IDS:
        traj = load_trajectory(ds, tid)            # (T, 5, H, W)
        print("=" * 64)
        print(f"Trajectory {tid}: stepping through {T-1} transitions (t -> t+1)")
        print("=" * 64)

        # Use ground-truth t+1 as both 'prediction' and 'reference' so the
        # reference-based spectral errors are ~0 (a self-consistency check),
        # while conservation/anisotropy reflect the real physics.
        step_records = []
        masses, energies, anisos = [], [], []
        for t in range(T - 1):
            x_t = traj[t]
            x_t1 = traj[t + 1]
            res = em.evaluate(pred=x_t1, initial=x_t, reference=x_t1)
            step_records.append({
                "t": t,
                "mass_reward": res.mass_reward,
                "momentum_reward": res.momentum_reward,
                "energy_reward": res.energy_reward,
                "density_violation_frac": res.density_violation_frac,
                "pressure_violation_frac": res.pressure_violation_frac,
                "entropy_decrease_frac": res.entropy_decrease_frac,
                "eos_relative_error": res.eos_relative_error,
                "density_anisotropy": res.density_anisotropy_pred,
                "ke_anisotropy": res.ke_anisotropy_pred,
            })
            masses.append(res.mass_reward)
            energies.append(res.energy_reward)
            anisos.append(res.density_anisotropy_pred)

        print(f"  mass reward      : mean {np.mean(masses):+.3e}  worst {np.min(masses):+.3e}")
        print(f"  energy reward    : mean {np.mean(energies):+.3e}  worst {np.min(energies):+.3e}")
        print(f"  density anisotropy A: range [{np.min(anisos):+.3f}, {np.max(anisos):+.3f}]"
              f"  mean {np.mean(anisos):+.3f}")
        print(f"  (a strongly non-zero A confirms the flow is anisotropic, as expected)")

        # Show the first transition in full detail
        x0, x1 = traj[0], traj[1]
        print("\n  --- Full metric report for transition t=0 -> t=1 ---")
        print(em.evaluate(pred=x1, initial=x0, reference=x1).summary())
        print()

        report["trajectories"][str(tid)] = step_records

    ds.close()

    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved full per-step report to: {OUT_JSON}")


if __name__ == "__main__":
    main()
