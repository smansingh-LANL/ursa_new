import os, sys, argparse, numpy as np, torch, netCDF4 as nc
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import euler_metrics as em

SPUS_DIR = '/lustre/fs01/hackathons/teams/tpc26-2/models/SPUS'
DATA_FILE = '/home/tpc-sjqyt/tpc26-2/data/PDEgym/CE-RP/CE-RP_test.nc'
VAR = 'data'
N_TRAIN, N_VAL, N_TEST = 0, 0, 715
FIRST_TEST_IDX = N_TRAIN + N_VAL
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIELDS = ['rho','u','v','p','E']

def load_spus(device):
    sys.path.insert(0, SPUS_DIR)
    from huggingface_hub import PyTorchModelHubMixin
    from model import Unet2D
    class SPUSUnet2D(Unet2D, PyTorchModelHubMixin): pass
    return SPUSUnet2D.from_pretrained(SPUS_DIR).to(device).eval()

@torch.no_grad()
def step(model, state_chw, device):
    x = torch.as_tensor(state_chw, dtype=torch.float32, device=device)
    x = x.reshape(1, 1, *state_chw.shape)
    y = model(x)
    return y[0].detach().cpu().numpy()

def rollout(model, gt, device, mode):
    T = gt.shape[0]
    preds = np.empty_like(gt)
    preds[0] = gt[0]
    cur = gt[0]
    for t in range(1, T):
        feed = cur if mode == 'autoregressive' else gt[t-1]
        nxt = step(model, feed, device)
        preds[t] = nxt
        cur = nxt
    return preds

def score(preds, gt):
    T = gt.shape[0]
    rows = []
    for t in range(1, T):
        r = em.evaluate(pred=preds[t], initial=gt[t-1], reference=gt[t])
        rows.append(r)
    return rows

def metric_arrays(rows):
    g = lambda f: np.array([f(r) for r in rows])
    return {
        'l2': g(lambda r: r.l2_error),
        'linf': g(lambda r: r.linf_error),
        'mass': g(lambda r: r.mass_reward),
        'momentum': g(lambda r: r.momentum_reward),
        'energy': g(lambda r: r.energy_reward),
        'dens_viol': g(lambda r: r.density_violation_frac),
        'pres_viol': g(lambda r: r.pressure_violation_frac),
        'entropy_dec': g(lambda r: r.entropy_decrease_frac),
        'eos': g(lambda r: r.eos_relative_error),
        'rho_aniso': g(lambda r: r.density_anisotropy_pred),
        'rho_ek_err': g(lambda r: r.density_spectra['err_k']),
        'rho_ekx_err': g(lambda r: r.density_spectra['err_kx']),
        'rho_eky_err': g(lambda r: r.density_spectra['err_ky']),
        'ke_ek_err': g(lambda r: r.ke_spectra['err_k']),
    }

def per_field_l2(preds, gt):
    T = gt.shape[0]
    out = np.zeros((T-1, 5))
    for t in range(1, T):
        for c in range(5):
            d = preds[t,c] - gt[t,c]
            out[t-1,c] = np.linalg.norm(d) / (np.linalg.norm(gt[t,c]) + em.EPS)
    return out

def plot_error_growth(ar, os_, gt, path):
    t = np.arange(1, gt.shape[0])
    fl_ar = per_field_l2(ar, gt)
    fl_os = per_field_l2(os_, gt)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for c, name in enumerate(FIELDS):
        axes[0].plot(t, fl_ar[:,c], marker='o', ms=3, label=name)
        axes[1].plot(t, fl_os[:,c], marker='o', ms=3, label=name)
    axes[0].set_title('Autoregressive rollout (errors compound)')
    axes[1].set_title('One-step / teacher-forced (per-step error)')
    for ax in axes:
        ax.set_xlabel('timestep t'); ax.set_yscale('log')
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_ylabel('relative L2 error')
    fig.suptitle('SPUS prediction error vs ground truth (CE-RP test trajectory)')
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def plot_conservation(m_ar, m_os, path):
    t = np.arange(1, len(m_ar['mass'])+1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key, title in zip(axes, ['mass','momentum','energy'],
                               ['Mass reward','Momentum reward','Energy reward']):
        ax.plot(t, m_ar[key], 'o-', ms=3, label='autoregressive')
        ax.plot(t, m_os[key], 's-', ms=3, label='one-step')
        ax.set_title(f'{title}  (0 = perfectly conserved)')
        ax.set_xlabel('timestep t'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle('Conservation rewards: -|sum(out)-sum(in)|/|sum(in)|')
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def plot_fields(ar, os_, gt, path, times=(0,10,20)):
    ch = 0
    times = [t for t in times if t < gt.shape[0]]
    fig, axes = plt.subplots(3, len(times), figsize=(4*len(times), 11))
    rows = [('ground truth', gt), ('autoregressive', ar), ('one-step', os_)]
    for i, (label, arr) in enumerate(rows):
        for j, t in enumerate(times):
            vmin, vmax = gt[t,ch].min(), gt[t,ch].max()
            im = axes[i,j].imshow(arr[t,ch], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
            axes[i,j].set_title(f'{label}  rho  t={t}', fontsize=10)
            axes[i,j].axis('off')
            fig.colorbar(im, ax=axes[i,j], fraction=0.046)
    fig.suptitle('Density field: ground truth vs SPUS predictions')
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def plot_spectra(ar, os_, gt, path, t=None):
    if t is None: t = gt.shape[0]-1
    ch = 0
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    specs = [('E(kx)', em.spectrum_kx), ('E(ky)', em.spectrum_ky),
             ('E(k)  k=sqrt(kx^2+ky^2)', em.spectrum_k)]
    for ax, (name, fn) in zip(axes, specs):
        for label, arr, style in [('ground truth', gt, 'k-'),
                                   ('autoregressive', ar, 'r--'),
                                   ('one-step', os_, 'b:')]:
            k, E = fn(arr[t, ch])
            ax.loglog(k[1:], E[1:]+em.EPS, style, label=label)
        ax.set_title(f'density {name}  (t={t})')
        ax.set_xlabel('wavenumber'); ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('power')
    fig.suptitle('Density power spectra at final timestep')
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def print_table(m_ar, m_os):
    T = len(m_ar['l2'])
    print('\n' + '='*78)
    print('PER-STEP METRICS  (AR = autoregressive, OS = one-step / teacher-forced)')
    print('='*78)
    hdr = f"{'t':>3} | {'L2_AR':>9} {'L2_OS':>9} | {'mass_AR':>9} {'enrg_AR':>9} | {'rhoEk_AR':>9} {'dViol_AR':>8}"
    print(hdr); print('-'*len(hdr))
    for i in range(T):
        print(f'{i+1:>3} | {m_ar["l2"][i]:9.3e} {m_os["l2"][i]:9.3e} | '
              f'{m_ar["mass"][i]:9.2e} {m_ar["energy"][i]:9.2e} | '
              f'{m_ar["rho_ek_err"][i]:9.3e} {m_ar["dens_viol"][i]:8.4f}')
    print('-'*len(hdr))
    print(f'mean | {m_ar["l2"].mean():9.3e} {m_os["l2"].mean():9.3e} | '
          f'{m_ar["mass"].mean():9.2e} {m_ar["energy"].mean():9.2e} | '
          f'{m_ar["rho_ek_err"].mean():9.3e} {m_ar["dens_viol"].mean():8.4f}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traj', type=int, default=0)
    ap.add_argument('--data', type=str, default=DATA_FILE)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device = {device}, torch = {torch.__version__}')
    if not (N_TRAIN+N_VAL <= args.traj < N_TRAIN+N_VAL+N_TEST):
        print(f'WARNING: traj {args.traj} is NOT in test split [{FIRST_TEST_IDX}, {FIRST_TEST_IDX+N_TEST}).')
    print(f'loading model from {SPUS_DIR} ...')
    model = load_spus(device)
    print(f'reading trajectory {args.traj} from {os.path.basename(args.data)} ...')
    ds = nc.Dataset(args.data, 'r')
    gt = np.asarray(ds.variables[VAR][args.traj]).astype(np.float32)
    ds.close()
    print(f'  ground truth shape = {gt.shape}')
    print('running autoregressive rollout (0 -> 20) ...')
    ar = rollout(model, gt, device, 'autoregressive')
    print('running one-step (teacher-forced) predictions ...')
    os_ = rollout(model, gt, device, 'one_step')
    m_ar = metric_arrays(score(ar, gt))
    m_os = metric_arrays(score(os_, gt))
    print_table(m_ar, m_os)
    npz = os.path.join(OUT_DIR, f'spus_eval_traj{args.traj}.npz')
    np.savez_compressed(npz, gt=gt, pred_ar=ar, pred_os=os_,
                        **{f'ar_{k}':v for k,v in m_ar.items()},
                        **{f'os_{k}':v for k,v in m_os.items()})
    print(f'\nsaved arrays -> {npz}')
    plot_error_growth(ar, os_, gt, os.path.join(OUT_DIR, 'spus_error_growth.png'))
    plot_conservation(m_ar, m_os, os.path.join(OUT_DIR, 'spus_conservation.png'))
    plot_fields(ar, os_, gt, os.path.join(OUT_DIR, 'spus_fields.png'))
    plot_spectra(ar, os_, gt, os.path.join(OUT_DIR, 'spus_spectra.png'))
    print('saved plots -> spus_error_growth.png, spus_conservation.png, spus_fields.png, spus_spectra.png')

if __name__ == '__main__':
    main()
