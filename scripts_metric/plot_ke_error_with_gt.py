import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import euler_metrics as em
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(OUT_DIR, 'spus_eval_traj0.npz'))
gt, ar = d['gt'], d['pred_ar']

def ke_field(state_5hw):
    rho, u, v = state_5hw[0], state_5hw[1], state_5hw[2]
    return 0.5 * rho * (u**2 + v**2)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
specs = [('E(kx)', em.spectrum_kx), ('E(ky)', em.spectrum_ky),
         ('E(k),  k=sqrt(kx^2+ky^2)', em.spectrum_k)]

# Row 0: actual spectra (GT and pred at t=1 and t=20)
for col, (name, fn) in enumerate(specs):
    ax = axes[0, col]
    for t, cgt, cpred in [(1, 'royalblue', 'deepskyblue'), (20, 'darkred', 'salmon')]:
        ke_gt   = ke_field(gt[t])
        ke_pred = ke_field(ar[t])
        k_r, E_r = fn(ke_gt)
        k_p, E_p = fn(ke_pred)
        ax.loglog(k_r[1:], E_r[1:]+em.EPS, color=cgt, ls='-', lw=1.8,
                  label=f'GT t={t}')
        ax.loglog(k_p[1:], E_p[1:]+em.EPS, color=cpred, ls='--', lw=1.5,
                  label=f'SPUS t={t}')
    ax.set_title(f'KE {name}', fontsize=11)
    ax.set_xlabel('wavenumber')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
axes[0, 0].set_ylabel('power (KE)')

# Row 1: relative error normalized by GT
for col, (name, fn) in enumerate(specs):
    ax = axes[1, col]
    for t, color in [(1, 'blue'), (20, 'red')]:
        ke_pred = ke_field(ar[t])
        ke_gt   = ke_field(gt[t])
        k_p, E_p = fn(ke_pred)
        k_r, E_r = fn(ke_gt)
        n = min(len(E_p), len(E_r))
        rel_err = np.abs(E_p[:n] - E_r[:n]) / (E_r[:n] + em.EPS)
        ax.semilogy(k_r[1:n], rel_err[1:n], color=color, ls='-', lw=1.5,
                    label=f't={t}', alpha=0.85)
    ax.set_title(f'KE {name}  (relative error)', fontsize=11)
    ax.set_xlabel('wavenumber')
    ax.set_ylabel('|E_pred - E_GT| / E_GT')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim(1e-4, 1e1)

fig.suptitle('KE spectra: ground truth vs SPUS (autoregressive)  —  t=1 and t=20', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'spus_ke_gt_and_error.png'), dpi=150)
plt.close(fig)
print('saved spus_ke_gt_and_error.png')
