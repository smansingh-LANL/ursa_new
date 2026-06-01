"""
Generate summary plots across all 715 CE-RP test trajectories.
Reads from the saved NetCDF on Lustre — no model inference required.
All curves show MEAN across trajectories, with std as a light shaded band.
"""
import os, sys, numpy as np, netCDF4 as nc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/lustre/fs01/hackathons/teams/tpc26-2/models/SPUS')
import euler_metrics as em

PRED_FILE = '/lustre/fs01/hackathons/teams/tpc26-2/tpc26_spus_predictions.nc'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

ds = nc.Dataset(PRED_FILE, 'r')
N, T, C, H, W = ds.variables['gt'].shape
print(f'Loaded: {N} trajs, {T} steps, {C} ch, {H}x{W}')

FIELDS = ['rho', 'u', 'v', 'p', 'E']

def ke_field(state_5hw):
    return 0.5 * state_5hw[0] * (state_5hw[1]**2 + state_5hw[2]**2)


# ================================================================
# 1. KE spectra:  GT vs autoregressive at t=1 and t=20
#    Top row: actual spectra, bottom row: relative error
#    mean + std across 715 trajs
# ================================================================
print('Computing KE spectra for all trajectories...')

spec_fns = [('E(kx)', em.spectrum_kx), ('E(ky)', em.spectrum_ky),
            ('E(k), k=sqrt(kx²+ky²)', em.spectrum_k)]
timesteps_spec = [1, 20]

spec_data = {}
for t in timesteps_spec:
    for tag in ['gt', 'pred_ar']:
        for sname, sfn in spec_fns:
            key = (t, tag, sname)
            spec_data[key] = []

for idx in range(N):
    gt_traj = np.asarray(ds.variables['gt'][idx]).astype(np.float32)
    ar_traj = np.asarray(ds.variables['pred_ar'][idx]).astype(np.float32)
    for t in timesteps_spec:
        ke_gt = ke_field(gt_traj[t])
        ke_ar = ke_field(ar_traj[t])
        for sname, sfn in spec_fns:
            _, E_gt = sfn(ke_gt)
            _, E_ar = sfn(ke_ar)
            spec_data[(t, 'gt', sname)].append(E_gt)
            spec_data[(t, 'pred_ar', sname)].append(E_ar)
    if (idx + 1) % 100 == 0:
        print(f'  spectra: {idx+1}/{N}')

for key in spec_data:
    spec_data[key] = np.array(spec_data[key])

# Get wavenumber axes (same for all trajs)
sample = np.asarray(ds.variables['gt'][0, 1]).astype(np.float32)
ke_sample = ke_field(np.asarray(ds.variables['gt'][0, 1]).astype(np.float32))
k_axes = {}
for sname, sfn in spec_fns:
    k_axes[sname], _ = sfn(ke_sample)

# Plot 1: KE spectra + error (2 rows x 3 cols)
fig, axes = plt.subplots(2, 3, figsize=(17, 10))

for col, (sname, sfn) in enumerate(spec_fns):
    ax_top = axes[0, col]
    ax_bot = axes[1, col]
    k = k_axes[sname]

    for t, cgt, cpred in [(1, 'royalblue', 'deepskyblue'), (20, 'darkred', 'salmon')]:
        E_gt_all = spec_data[(t, 'gt', sname)]
        E_ar_all = spec_data[(t, 'pred_ar', sname)]

        gt_mean = E_gt_all.mean(axis=0)
        gt_std = E_gt_all.std(axis=0)
        ar_mean = E_ar_all.mean(axis=0)
        ar_std = E_ar_all.std(axis=0)

        ax_top.loglog(k[1:], gt_mean[1:] + em.EPS, color=cgt, ls='-', lw=1.8,
                      label=f'GT t={t}')
        ax_top.fill_between(k[1:],
                            np.clip(gt_mean[1:] - gt_std[1:], 1e-40, None),
                            gt_mean[1:] + gt_std[1:],
                            color=cgt, alpha=0.15)
        ax_top.loglog(k[1:], ar_mean[1:] + em.EPS, color=cpred, ls='--', lw=1.5,
                      label=f'SPUS t={t}')
        ax_top.fill_between(k[1:],
                            np.clip(ar_mean[1:] - ar_std[1:], 1e-40, None),
                            ar_mean[1:] + ar_std[1:],
                            color=cpred, alpha=0.15)

        n = min(len(gt_mean), len(ar_mean))
        rel_err = np.abs(E_ar_all[:, :n] - E_gt_all[:, :n]) / (E_gt_all[:, :n] + em.EPS)
        err_mean = rel_err.mean(axis=0)
        err_std = rel_err.std(axis=0)

        color = 'blue' if t == 1 else 'red'
        ax_bot.semilogy(k[1:n], err_mean[1:n], color=color, ls='-', lw=1.5,
                        label=f't={t}', alpha=0.85)
        ax_bot.fill_between(k[1:n],
                            np.clip(err_mean[1:n] - err_std[1:n], 1e-8, None),
                            err_mean[1:n] + err_std[1:n],
                            color=color, alpha=0.15)

    ax_top.set_title(f'KE {sname}', fontsize=11)
    ax_top.grid(True, which='both', alpha=0.3)
    ax_top.legend(fontsize=8)
    ax_bot.set_title(f'KE {sname} (relative error)', fontsize=11)
    ax_bot.set_xlabel('wavenumber')
    ax_bot.set_ylabel('|E_pred − E_GT| / E_GT')
    ax_bot.grid(True, which='both', alpha=0.3)
    ax_bot.legend(fontsize=10)
    ax_bot.set_ylim(1e-4, 1e1)

axes[0, 0].set_ylabel('power (KE)')
fig.suptitle(f'KE spectra: GT vs SPUS (autoregressive) — mean ± std over {N} trajectories', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'ke_spectra_and_error.png'), dpi=150)
plt.close(fig)
print('  saved ke_spectra_and_error.png')


# ================================================================
# 2. KE spectra evolution at t=1, 5, 10, 15, 20 (mean + std)
# ================================================================
print('Computing KE spectra evolution...')

times_evo = [1, 5, 10, 15, 20]
times_evo = [t for t in times_evo if t < T]

evo_data = {}
for t in times_evo:
    for tag in ['gt', 'pred_ar', 'pred_os']:
        for sname, sfn in spec_fns:
            evo_data[(t, tag, sname)] = []

for idx in range(N):
    gt_traj = np.asarray(ds.variables['gt'][idx]).astype(np.float32)
    ar_traj = np.asarray(ds.variables['pred_ar'][idx]).astype(np.float32)
    os_traj = np.asarray(ds.variables['pred_os'][idx]).astype(np.float32)
    for t in times_evo:
        ke_gt = ke_field(gt_traj[t])
        ke_ar = ke_field(ar_traj[t])
        ke_os = ke_field(os_traj[t])
        for sname, sfn in spec_fns:
            _, E_gt = sfn(ke_gt)
            _, E_ar = sfn(ke_ar)
            _, E_os = sfn(ke_os)
            evo_data[(t, 'gt', sname)].append(E_gt)
            evo_data[(t, 'pred_ar', sname)].append(E_ar)
            evo_data[(t, 'pred_os', sname)].append(E_os)
    if (idx + 1) % 100 == 0:
        print(f'  spectra evo: {idx+1}/{N}')

for key in evo_data:
    evo_data[key] = np.array(evo_data[key])

fig, axes = plt.subplots(len(times_evo), 3, figsize=(16, 4*len(times_evo)))
for row, t in enumerate(times_evo):
    for col, (sname, _) in enumerate(spec_fns):
        ax = axes[row, col]
        k = k_axes[sname]
        for label, tag, color, ls in [('GT', 'gt', 'black', '-'),
                                       ('AR', 'pred_ar', 'red', '--'),
                                       ('OS', 'pred_os', 'blue', ':')]:
            arr = evo_data[(t, tag, sname)]
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            ax.loglog(k[1:], mean[1:] + em.EPS, color=color, ls=ls, lw=1.3, label=label)
            ax.fill_between(k[1:],
                            np.clip(mean[1:] - std[1:], 1e-40, None),
                            mean[1:] + std[1:],
                            color=color, alpha=0.1)
        ax.set_title(f't={t}  KE {sname}', fontsize=9)
        ax.grid(True, which='both', alpha=0.3)
        if row == 0:
            ax.legend(fontsize=7)
        if col == 0:
            ax.set_ylabel('power')
        if row == len(times_evo) - 1:
            ax.set_xlabel('wavenumber')

fig.suptitle(f'KE spectra evolution: mean ± std over {N} trajectories', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'ke_spectra_evolution.png'), dpi=130)
plt.close(fig)
print('  saved ke_spectra_evolution.png')


# ================================================================
# 3. Conservation rewards over time — mean + std
# ================================================================
print('Computing conservation rewards...')

mass_ar = np.zeros((N, T - 1))
mom_ar = np.zeros((N, T - 1))
ener_ar = np.zeros((N, T - 1))
mass_os = np.zeros((N, T - 1))
mom_os = np.zeros((N, T - 1))
ener_os = np.zeros((N, T - 1))

for idx in range(N):
    gt_traj = np.asarray(ds.variables['gt'][idx]).astype(np.float32)
    ar_traj = np.asarray(ds.variables['pred_ar'][idx]).astype(np.float32)
    os_traj = np.asarray(ds.variables['pred_os'][idx]).astype(np.float32)
    for t in range(1, T):
        mass_ar[idx, t-1] = em.reward_mass_conservation(gt_traj[t-1], ar_traj[t])
        mom_ar[idx, t-1] = em.reward_momentum_conservation(gt_traj[t-1], ar_traj[t])
        ener_ar[idx, t-1] = em.reward_energy_conservation(gt_traj[t-1], ar_traj[t])
        mass_os[idx, t-1] = em.reward_mass_conservation(gt_traj[t-1], os_traj[t])
        mom_os[idx, t-1] = em.reward_momentum_conservation(gt_traj[t-1], os_traj[t])
        ener_os[idx, t-1] = em.reward_energy_conservation(gt_traj[t-1], os_traj[t])
    if (idx + 1) % 100 == 0:
        print(f'  conservation: {idx+1}/{N}')

ts = np.arange(1, T)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (title, ar_data, os_data) in zip(axes, [
    ('Mass conservation reward', mass_ar, mass_os),
    ('Momentum conservation reward', mom_ar, mom_os),
    ('Energy conservation reward', ener_ar, ener_os),
]):
    ar_m, ar_s = ar_data.mean(axis=0), ar_data.std(axis=0)
    os_m, os_s = os_data.mean(axis=0), os_data.std(axis=0)

    ax.plot(ts, ar_m, 'o-', ms=3, color='crimson', label='autoregressive')
    ax.fill_between(ts, ar_m - ar_s, ar_m + ar_s, color='crimson', alpha=0.15)
    ax.plot(ts, os_m, 's-', ms=3, color='royalblue', label='one-step')
    ax.fill_between(ts, os_m - os_s, os_m + os_s, color='royalblue', alpha=0.15)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel('timestep t')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

axes[0].set_ylabel('reward  (-|Δ|/|input|)')
fig.suptitle(f'Conservation rewards: mean ± std over {N} trajectories', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'conservation_rewards.png'), dpi=150)
plt.close(fig)
print('  saved conservation_rewards.png')


# ================================================================
# 4. Per-field L2 error growth over time — mean + std
# ================================================================
print('Computing per-field L2 errors...')

l2_ar = np.zeros((N, T - 1, 5))
l2_os = np.zeros((N, T - 1, 5))

for idx in range(N):
    gt_traj = np.asarray(ds.variables['gt'][idx]).astype(np.float32)
    ar_traj = np.asarray(ds.variables['pred_ar'][idx]).astype(np.float32)
    os_traj = np.asarray(ds.variables['pred_os'][idx]).astype(np.float32)
    for t in range(1, T):
        for c in range(5):
            d_ar = ar_traj[t, c] - gt_traj[t, c]
            d_os = os_traj[t, c] - gt_traj[t, c]
            norm_gt = np.linalg.norm(gt_traj[t, c]) + em.EPS
            l2_ar[idx, t-1, c] = np.linalg.norm(d_ar) / norm_gt
            l2_os[idx, t-1, c] = np.linalg.norm(d_os) / norm_gt
    if (idx + 1) % 100 == 0:
        print(f'  L2 errors: {idx+1}/{N}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

for c, (name, clr) in enumerate(zip(FIELDS, colors)):
    m_ar = l2_ar[:, :, c].mean(axis=0)
    s_ar = l2_ar[:, :, c].std(axis=0)
    m_os = l2_os[:, :, c].mean(axis=0)
    s_os = l2_os[:, :, c].std(axis=0)

    axes[0].plot(ts, m_ar, marker='o', ms=3, color=clr, label=name)
    axes[0].fill_between(ts, np.clip(m_ar - s_ar, 1e-10, None), m_ar + s_ar,
                         color=clr, alpha=0.12)
    axes[1].plot(ts, m_os, marker='o', ms=3, color=clr, label=name)
    axes[1].fill_between(ts, np.clip(m_os - s_os, 1e-10, None), m_os + s_os,
                         color=clr, alpha=0.12)

axes[0].set_title('Autoregressive (errors compound)')
axes[1].set_title('One-step / teacher-forced')
for ax in axes:
    ax.set_xlabel('timestep t')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel('relative L2 error')
fig.suptitle(f'Per-field error growth: mean ± std over {N} trajectories', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'error_growth.png'), dpi=150)
plt.close(fig)
print('  saved error_growth.png')


# ================================================================
# 5. Reward distributions at t=1, 10, 20 (histograms, 50 bins)
# ================================================================
print('Plotting reward distributions...')

NBINS = 50
TIMESTEPS_DIST = [1, 10, 20]
TIMESTEPS_DIST = [t for t in TIMESTEPS_DIST if t < T]

mass_dist = {t: mass_ar[:, t-1] for t in TIMESTEPS_DIST}
mom_dist = {t: mom_ar[:, t-1] for t in TIMESTEPS_DIST}
ener_dist = {t: ener_ar[:, t-1] for t in TIMESTEPS_DIST}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
dist_colors = {1: 'royalblue', 10: 'darkorange', 20: 'crimson'}

for ax, (title, data) in zip(axes, [
    ('Mass conservation reward', mass_dist),
    ('Momentum conservation reward', mom_dist),
    ('Energy conservation reward', ener_dist),
]):
    all_vals = np.concatenate([data[t] for t in TIMESTEPS_DIST])
    lo, hi = np.percentile(all_vals, 0.5), np.percentile(all_vals, 99.5)
    bins = np.linspace(lo, hi, NBINS + 1)

    for t in TIMESTEPS_DIST:
        ax.hist(data[t], bins=bins, alpha=0.5, color=dist_colors[t],
                label=f't={t}  (mean={data[t].mean():.4f})', edgecolor='none')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('reward  (-|Δ|/|input|)')
    ax.set_ylabel('count')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='black', ls='--', lw=0.8, alpha=0.5)

fig.suptitle(f'Conservation reward distributions (autoregressive, {N} trajectories, {NBINS} bins)', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'reward_distributions.png'), dpi=150)
plt.close(fig)
print('  saved reward_distributions.png')


# ================================================================
# 6. Density power spectra at t=20 — mean + std (GT, AR, OS)
# ================================================================
print('Computing density spectra at t=20...')

rho_spec = {}
t_rho = min(20, T - 1)
for tag in ['gt', 'pred_ar', 'pred_os']:
    for sname, _ in spec_fns:
        rho_spec[(tag, sname)] = []

for idx in range(N):
    gt_traj = np.asarray(ds.variables['gt'][idx, t_rho]).astype(np.float32)
    ar_traj = np.asarray(ds.variables['pred_ar'][idx, t_rho]).astype(np.float32)
    os_traj = np.asarray(ds.variables['pred_os'][idx, t_rho]).astype(np.float32)

    for tag, arr in [('gt', gt_traj), ('pred_ar', ar_traj), ('pred_os', os_traj)]:
        rho = arr[0]
        for sname, sfn in spec_fns:
            _, E = sfn(rho)
            rho_spec[(tag, sname)].append(E)
    if (idx + 1) % 100 == 0:
        print(f'  density spectra: {idx+1}/{N}')

for key in rho_spec:
    rho_spec[key] = np.array(rho_spec[key])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for col, (sname, _) in enumerate(spec_fns):
    ax = axes[col]
    k = k_axes[sname]
    for label, tag, color, ls in [('GT', 'gt', 'black', '-'),
                                   ('AR', 'pred_ar', 'red', '--'),
                                   ('OS', 'pred_os', 'blue', ':')]:
        arr = rho_spec[(tag, sname)]
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.loglog(k[1:], mean[1:] + em.EPS, color=color, ls=ls, lw=1.5, label=label)
        ax.fill_between(k[1:],
                        np.clip(mean[1:] - std[1:], 1e-40, None),
                        mean[1:] + std[1:],
                        color=color, alpha=0.1)
    ax.set_title(f'Density {sname}  (t={t_rho})', fontsize=11)
    ax.set_xlabel('wavenumber')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9)
axes[0].set_ylabel('power')
fig.suptitle(f'Density power spectra at t={t_rho}: mean ± std over {N} trajectories', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'density_spectra.png'), dpi=150)
plt.close(fig)
print('  saved density_spectra.png')

ds.close()
print(f'\nAll plots saved to {OUT_DIR}')
