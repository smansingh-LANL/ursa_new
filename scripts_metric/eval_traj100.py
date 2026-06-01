import time, os, sys, numpy as np, torch, netCDF4 as nc
sys.path.insert(0, '/lustre/fs01/hackathons/teams/tpc26-2/models/SPUS')
from huggingface_hub import PyTorchModelHubMixin
from model import Unet2D
import euler_metrics as em

class SPUSUnet2D(Unet2D, PyTorchModelHubMixin): pass
device = 'cuda'
model = SPUSUnet2D.from_pretrained('/lustre/fs01/hackathons/teams/tpc26-2/models/SPUS').to(device).eval()

TRAJ = 100
ds = nc.Dataset('/home/tpc-sjqyt/tpc26-2/data/PDEgym/CE-RP/CE-RP_test.nc', 'r')
gt = np.asarray(ds.variables['data'][TRAJ]).astype(np.float32)
ds.close()
print(f'traj={TRAJ}, gt shape={gt.shape}')

x = torch.randn(1, 1, 5, 128, 128, device=device)
with torch.no_grad():
    model(x)
torch.cuda.synchronize()

ar = np.empty_like(gt)
ar[0] = gt[0]
cur = gt[0]
step_times = []
for t in range(1, gt.shape[0]):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    x = torch.as_tensor(cur, dtype=torch.float32, device=device).reshape(1, 1, 5, 128, 128)
    with torch.no_grad():
        y = model(x)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    step_times.append(dt)
    nxt = y[0].cpu().numpy()
    ar[t] = nxt
    cur = nxt

os_ = np.empty_like(gt)
os_[0] = gt[0]
os_times = []
for t in range(1, gt.shape[0]):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    x = torch.as_tensor(gt[t - 1], dtype=torch.float32, device=device).reshape(1, 1, 5, 128, 128)
    with torch.no_grad():
        y = model(x)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    os_times.append(dt)
    os_[t] = y[0].cpu().numpy()

print('\nPER-STEP TIMING (ms):')
ar_mean = np.mean(step_times) * 1000
ar_std = np.std(step_times) * 1000
ar_total = np.sum(step_times) * 1000
os_mean = np.mean(os_times) * 1000
os_std = np.std(os_times) * 1000
os_total = np.sum(os_times) * 1000
print(f'  AR  mean={ar_mean:.2f}ms  std={ar_std:.2f}ms  total={ar_total:.1f}ms')
print(f'  OS  mean={os_mean:.2f}ms  std={os_std:.2f}ms  total={os_total:.1f}ms')
print(f'  Full 20-step rollout wall time: {np.sum(step_times):.3f}s')

def metric_arrays(preds, gt):
    rows = []
    for t in range(1, gt.shape[0]):
        r = em.evaluate(pred=preds[t], initial=gt[t - 1], reference=gt[t])
        rows.append(r)
    g = lambda f: np.array([f(r) for r in rows])
    return {
        'l2': g(lambda r: r.l2_error),
        'mass': g(lambda r: r.mass_reward),
        'energy': g(lambda r: r.energy_reward),
        'rho_ek_err': g(lambda r: r.density_spectra['err_k']),
        'dens_viol': g(lambda r: r.density_violation_frac),
        'ke_ek_err': g(lambda r: r.ke_spectra['err_k']),
    }

m_ar = metric_arrays(ar, gt)
m_os = metric_arrays(os_, gt)

sep = '=' * 78
print(f'\n{sep}')
print(f'PER-STEP METRICS  traj={TRAJ}  (AR = autoregressive, OS = one-step)')
print(sep)
hdr = '  t |     L2_AR     L2_OS |   mass_AR   enrg_AR |  rhoEk_AR   keEk_AR'
print(hdr)
print('-' * len(hdr))
T = len(m_ar['l2'])
for i in range(T):
    print(f'{i+1:>3} | {m_ar["l2"][i]:9.3e} {m_os["l2"][i]:9.3e} | '
          f'{m_ar["mass"][i]:9.2e} {m_ar["energy"][i]:9.2e} | '
          f'{m_ar["rho_ek_err"][i]:9.3e} {m_ar["ke_ek_err"][i]:9.3e}')
print('-' * len(hdr))
print(f'mean | {m_ar["l2"].mean():9.3e} {m_os["l2"].mean():9.3e} | '
      f'{m_ar["mass"].mean():9.2e} {m_ar["energy"].mean():9.2e} | '
      f'{m_ar["rho_ek_err"].mean():9.3e} {m_ar["ke_ek_err"].mean():9.3e}')

OUT = os.path.dirname(os.path.abspath(__file__))
np.savez_compressed(os.path.join(OUT, f'spus_eval_traj{TRAJ}.npz'),
                    gt=gt, pred_ar=ar, pred_os=os_,
                    step_times_ar=np.array(step_times),
                    step_times_os=np.array(os_times))
print(f'\nsaved spus_eval_traj{TRAJ}.npz')
