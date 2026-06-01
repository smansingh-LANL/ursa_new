import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import euler_metrics as em
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(OUT_DIR, 'spus_eval_traj0.npz'))
gt, ar, os_ = d['gt'], d['pred_ar'], d['pred_os']

def ke_field(state_5hw):
    rho, u, v = state_5hw[0], state_5hw[1], state_5hw[2]
    return 0.5 * rho * (u**2 + v**2)

# --- Plot 1: KE spectra at final timestep ---
t = gt.shape[0] - 1
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
specs = [('E(kx)', em.spectrum_kx), ('E(ky)', em.spectrum_ky),
         ('E(k)  k=sqrt(kx^2+ky^2)', em.spectrum_k)]
for ax, (name, fn) in zip(axes, specs):
    for label, arr, style in [('ground truth', gt, 'k-'),
                               ('autoregressive', ar, 'r--'),
                               ('one-step', os_, 'b:')]:
        ke = ke_field(arr[t])
        k, E = fn(ke)
        ax.loglog(k[1:], E[1:]+em.EPS, style, label=label, linewidth=1.5)
    ax.set_title(f'KE {name}  (t={t})')
    ax.set_xlabel('wavenumber'); ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel('power')
fig.suptitle('Kinetic Energy power spectra at final timestep (t=20)')
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'spus_ke_spectra_final.png'), dpi=130)
plt.close(fig)

# --- Plot 2: KE spectra at multiple timesteps (t=1, 5, 10, 15, 20) ---
times = [1, 5, 10, 15, 20]
times = [t for t in times if t < gt.shape[0]]
fig, axes = plt.subplots(len(times), 3, figsize=(15, 4*len(times)))
for row, t in enumerate(times):
    for col, (name, fn) in enumerate(specs):
        ax = axes[row, col]
        for label, arr, style in [('ground truth', gt, 'k-'),
                                   ('autoregressive', ar, 'r--'),
                                   ('one-step', os_, 'b:')]:
            ke = ke_field(arr[t])
            k, E = fn(ke)
            ax.loglog(k[1:], E[1:]+em.EPS, style, label=label, linewidth=1.2)
        ax.set_title(f't={t}  KE {name}', fontsize=9)
        ax.grid(True, which='both', alpha=0.3)
        if row == 0: ax.legend(fontsize=7)
        if col == 0: ax.set_ylabel('power')
        if row == len(times)-1: ax.set_xlabel('wavenumber')
fig.suptitle('Kinetic Energy spectra: GT vs SPUS at multiple timesteps', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'spus_ke_spectra_evolution.png'), dpi=130)
plt.close(fig)

# --- Print KE spectral errors per step ---
print('KE spectral errors (relative L2) per timestep:')
print(f"{'t':>3} | {'KE E(kx)':>10} {'KE E(ky)':>10} {'KE E(k)':>10} | {'aniso_pred':>10} {'aniso_ref':>10}")
print('-'*68)
for t in range(1, gt.shape[0]):
    ke_p = ke_field(ar[t])
    ke_r = ke_field(gt[t])
    se = em.spectral_errors(ke_p, ke_r)
    print(f'{t:>3} | {se["err_kx"]:10.4e} {se["err_ky"]:10.4e} {se["err_k"]:10.4e} | {se["anisotropy_pred"]:+10.4f} {se["anisotropy_ref"]:+10.4f}')

print('\nSaved: spus_ke_spectra_final.png, spus_ke_spectra_evolution.png')
