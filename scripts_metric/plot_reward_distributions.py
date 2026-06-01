"""
Plot mass, momentum, energy conservation reward distributions
across all 715 test trajectories at t=1, t=10, t=20.
Each subplot shows 3 overlapping histograms (one per timestep).
Rewards are already normalized: -|sum(out)-sum(in)|/|sum(in)|.
"""
import os, sys, numpy as np, torch, netCDF4 as nc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/lustre/fs01/hackathons/teams/tpc26-2/models/SPUS')
from huggingface_hub import PyTorchModelHubMixin
from model import Unet2D
import euler_metrics as em

class SPUSUnet2D(Unet2D, PyTorchModelHubMixin):
    pass

DATA_FILE = '/home/tpc-sjqyt/tpc26-2/data/PDEgym/CE-RP/CE-RP_test.nc'
SPUS_DIR = '/lustre/fs01/hackathons/teams/tpc26-2/models/SPUS'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
VAR = 'data'
TIMESTEPS = [1, 10, 20]
NBINS = 50

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SPUSUnet2D.from_pretrained(SPUS_DIR).to(device).eval()

ds = nc.Dataset(DATA_FILE, 'r')
N_TRAJ = ds.variables[VAR].shape[0]
print(f'{N_TRAJ} trajectories, device={device}')

mass_rewards = {t: [] for t in TIMESTEPS}
momentum_rewards = {t: [] for t in TIMESTEPS}
energy_rewards = {t: [] for t in TIMESTEPS}

for idx in range(N_TRAJ):
    gt = np.asarray(ds.variables[VAR][idx]).astype(np.float32)

    ar = np.empty_like(gt)
    ar[0] = gt[0]
    cur = gt[0]
    for t in range(1, gt.shape[0]):
        x = torch.as_tensor(cur, dtype=torch.float32, device=device).reshape(1, 1, 5, 128, 128)
        with torch.no_grad():
            y = model(x)
        cur = y[0].cpu().numpy()
        ar[t] = cur

        if t in TIMESTEPS:
            mass_rewards[t].append(em.reward_mass_conservation(gt[t - 1], ar[t]))
            momentum_rewards[t].append(em.reward_momentum_conservation(gt[t - 1], ar[t]))
            energy_rewards[t].append(em.reward_energy_conservation(gt[t - 1], ar[t]))

    if (idx + 1) % 100 == 0 or idx == 0:
        print(f'  {idx+1}/{N_TRAJ}')

ds.close()

for t in TIMESTEPS:
    mass_rewards[t] = np.array(mass_rewards[t])
    momentum_rewards[t] = np.array(momentum_rewards[t])
    energy_rewards[t] = np.array(energy_rewards[t])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = {1: 'royalblue', 10: 'darkorange', 20: 'crimson'}

for ax, (title, data) in zip(axes, [
    ('Mass conservation reward', mass_rewards),
    ('Momentum conservation reward', momentum_rewards),
    ('Energy conservation reward', energy_rewards),
]):
    all_vals = np.concatenate([data[t] for t in TIMESTEPS])
    lo, hi = np.percentile(all_vals, 0.5), np.percentile(all_vals, 99.5)
    bins = np.linspace(lo, hi, NBINS + 1)

    for t in TIMESTEPS:
        ax.hist(data[t], bins=bins, alpha=0.5, color=colors[t],
                label=f't={t}  (mean={data[t].mean():.4f})', edgecolor='none')

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('reward  (-|delta|/|input|)')
    ax.set_ylabel('count')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='black', ls='--', lw=0.8, alpha=0.5)

fig.suptitle(f'Conservation reward distributions (autoregressive, {N_TRAJ} CE-RP test trajectories, nbins={NBINS})',
             fontsize=13)
fig.tight_layout()
out_path = os.path.join(OUT_DIR, 'spus_reward_distributions.png')
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f'saved {out_path}')
