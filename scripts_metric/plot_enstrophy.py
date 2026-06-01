"""
Compute and plot enstrophy across all 715 CE-RP test trajectories.

Enstrophy = (1/2) * mean(omega^2), where omega = dv/dx - du/dy (vorticity).
Derivatives use central finite differences with periodic BCs.

Reads from the saved NetCDF on Lustre — no model inference required.
Saves to /home/tpc-bepxr/tpc26-2/plots/
"""
import os, sys, numpy as np, netCDF4 as nc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PRED_FILE = '/lustre/fs01/hackathons/teams/tpc26-2/tpc26_spus_predictions.nc'
OUT_DIR = '/home/tpc-bepxr/tpc26-2/plots'
os.makedirs(OUT_DIR, exist_ok=True)

ds = nc.Dataset(PRED_FILE, 'r')
N, T, C, H, W = ds.variables['gt'].shape
print(f'Loaded: {N} trajs, {T} steps, {C} ch, {H}x{W}')

dx = 1.0 / W
dy = 1.0 / H


def vorticity(u, v):
    """omega = dv/dx - du/dy, central differences with periodic BCs."""
    dvdx = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2.0 * dx)
    dudy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * dy)
    return dvdx - dudy


def enstrophy(u, v):
    """Enstrophy = (1/2) * mean(omega^2)."""
    omega = vorticity(u, v)
    return 0.5 * np.mean(omega ** 2)


# Collect enstrophy for all trajectories at every timestep
enst_gt = np.zeros((N, T))
enst_ar = np.zeros((N, T))
enst_os = np.zeros((N, T))

for idx in range(N):
    gt = np.asarray(ds.variables['gt'][idx]).astype(np.float32)
    ar = np.asarray(ds.variables['pred_ar'][idx]).astype(np.float32)
    os_ = np.asarray(ds.variables['pred_os'][idx]).astype(np.float32)

    for t in range(T):
        enst_gt[idx, t] = enstrophy(gt[t, 1], gt[t, 2])
        enst_ar[idx, t] = enstrophy(ar[t, 1], ar[t, 2])
        enst_os[idx, t] = enstrophy(os_[t, 1], os_[t, 2])

    if (idx + 1) % 100 == 0:
        print(f'  enstrophy: {idx+1}/{N}')

ds.close()

ts = np.arange(T)

# ================================================================
# Plot 1: Enstrophy over time (GT vs AR vs OS), mean only
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

ax = axes[0]
ax.plot(ts, enst_gt.mean(axis=0), 'k-o', ms=3, lw=1.8, label='Ground truth')
ax.plot(ts, enst_ar.mean(axis=0), 'r--s', ms=3, lw=1.5, label='Autoregressive')
ax.plot(ts, enst_os.mean(axis=0), 'b:^', ms=3, lw=1.5, label='One-step')
ax.set_xlabel('timestep t')
ax.set_ylabel('enstrophy  (1/2)·mean(ω²)')
ax.set_title('Enstrophy evolution', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: Relative enstrophy error
ax = axes[1]
rel_err_ar = np.abs(enst_ar - enst_gt) / (enst_gt + 1e-30)
rel_err_os = np.abs(enst_os - enst_gt) / (enst_gt + 1e-30)

ax.semilogy(ts[1:], rel_err_ar.mean(axis=0)[1:], 'r-o', ms=3, lw=1.5, label='Autoregressive')
ax.semilogy(ts[1:], rel_err_os.mean(axis=0)[1:], 'b-s', ms=3, lw=1.5, label='One-step')
ax.set_xlabel('timestep t')
ax.set_ylabel('|Ω_pred - Ω_GT| / Ω_GT')
ax.set_title('Relative enstrophy error', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

fig.suptitle(f'Enstrophy: mean over {N} CE-RP test trajectories\n'
             f'Ω = (1/2)·mean(ω²),  ω = ∂v/∂x - ∂u/∂y', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'enstrophy.png'), dpi=150)
plt.close(fig)
print(f'saved {OUT_DIR}/enstrophy.png')

# Print summary
print(f'\nEnstrophy summary (mean over {N} trajectories):')
print(f'{"t":>3}  {"GT":>12}  {"AR":>12}  {"OS":>12}  {"relErr_AR":>12}  {"relErr_OS":>12}')
print('-' * 68)
for t in range(T):
    print(f'{t:>3}  {enst_gt.mean(axis=0)[t]:12.4f}  '
          f'{enst_ar.mean(axis=0)[t]:12.4f}  '
          f'{enst_os.mean(axis=0)[t]:12.4f}  '
          f'{rel_err_ar.mean(axis=0)[t]:12.4e}  '
          f'{rel_err_os.mean(axis=0)[t]:12.4e}')
