"""
Evaluate SPUS on ALL 715 CE-RP test trajectories.
Full 20-step autoregressive rollout + one-step for each.
Saves per-trajectory summary metrics to a single CSV + NPZ.
"""
import time, os, sys, numpy as np, torch, netCDF4 as nc, csv

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device = {device}, torch = {torch.__version__}')

model = SPUSUnet2D.from_pretrained(SPUS_DIR).to(device).eval()
print('model loaded')

ds = nc.Dataset(DATA_FILE, 'r')
N_TRAJ = ds.variables[VAR].shape[0]
T_STEPS = ds.variables[VAR].shape[1]
print(f'dataset: {N_TRAJ} trajectories, {T_STEPS} timesteps each')

x_warmup = torch.randn(1, 1, 5, 128, 128, device=device)
with torch.no_grad():
    model(x_warmup)
torch.cuda.synchronize()

METRIC_NAMES = [
    'l2_ar_mean', 'l2_ar_final', 'l2_os_mean',
    'mass_ar_mean', 'energy_ar_mean',
    'rho_ek_ar_mean', 'ke_ek_ar_mean',
    'dens_viol_max', 'pres_viol_max',
    'rollout_time_ms',
]

all_metrics = np.zeros((N_TRAJ, len(METRIC_NAMES)), dtype=np.float32)
all_l2_ar = np.zeros((N_TRAJ, T_STEPS - 1), dtype=np.float32)
all_l2_os = np.zeros((N_TRAJ, T_STEPS - 1), dtype=np.float32)

t_global = time.perf_counter()

for traj_idx in range(N_TRAJ):
    gt = np.asarray(ds.variables[VAR][traj_idx]).astype(np.float32)

    ar = np.empty_like(gt)
    ar[0] = gt[0]
    cur = gt[0]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t in range(1, T_STEPS):
        x = torch.as_tensor(cur, dtype=torch.float32, device=device).reshape(1, 1, 5, 128, 128)
        with torch.no_grad():
            y = model(x)
        nxt = y[0].cpu().numpy()
        ar[t] = nxt
        cur = nxt
    torch.cuda.synchronize()
    rollout_ms = (time.perf_counter() - t0) * 1000

    os_ = np.empty_like(gt)
    os_[0] = gt[0]
    for t in range(1, T_STEPS):
        x = torch.as_tensor(gt[t - 1], dtype=torch.float32, device=device).reshape(1, 1, 5, 128, 128)
        with torch.no_grad():
            y = model(x)
        os_[t] = y[0].cpu().numpy()

    l2_ar = np.zeros(T_STEPS - 1)
    l2_os = np.zeros(T_STEPS - 1)
    mass_ar = np.zeros(T_STEPS - 1)
    energy_ar = np.zeros(T_STEPS - 1)
    rho_ek_ar = np.zeros(T_STEPS - 1)
    ke_ek_ar = np.zeros(T_STEPS - 1)
    dv_max = 0.0
    pv_max = 0.0

    for t in range(1, T_STEPS):
        r_ar = em.evaluate(pred=ar[t], initial=gt[t - 1], reference=gt[t])
        r_os = em.evaluate(pred=os_[t], initial=gt[t - 1], reference=gt[t])
        i = t - 1
        l2_ar[i] = r_ar.l2_error
        l2_os[i] = r_os.l2_error
        mass_ar[i] = r_ar.mass_reward
        energy_ar[i] = r_ar.energy_reward
        rho_ek_ar[i] = r_ar.density_spectra['err_k']
        ke_ek_ar[i] = r_ar.ke_spectra['err_k']
        dv_max = max(dv_max, r_ar.density_violation_frac)
        pv_max = max(pv_max, r_ar.pressure_violation_frac)

    all_l2_ar[traj_idx] = l2_ar
    all_l2_os[traj_idx] = l2_os
    all_metrics[traj_idx] = [
        l2_ar.mean(), l2_ar[-1], l2_os.mean(),
        mass_ar.mean(), energy_ar.mean(),
        rho_ek_ar.mean(), ke_ek_ar.mean(),
        dv_max, pv_max,
        rollout_ms,
    ]

    if (traj_idx + 1) % 50 == 0 or traj_idx == 0:
        elapsed = time.perf_counter() - t_global
        eta = elapsed / (traj_idx + 1) * (N_TRAJ - traj_idx - 1)
        print(f'  traj {traj_idx+1:>4}/{N_TRAJ}  '
              f'L2_AR={l2_ar.mean():.3e}  L2_OS={l2_os.mean():.3e}  '
              f'rollout={rollout_ms:.1f}ms  '
              f'elapsed={elapsed:.0f}s  ETA={eta:.0f}s')

ds.close()
total_time = time.perf_counter() - t_global
print(f'\nDone: {N_TRAJ} trajectories in {total_time:.1f}s ({total_time/N_TRAJ:.2f}s/traj)')

csv_path = os.path.join(OUT_DIR, 'spus_eval_all_trajs.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['traj_idx'] + METRIC_NAMES)
    for i in range(N_TRAJ):
        w.writerow([i] + [f'{v:.6e}' for v in all_metrics[i]])

npz_path = os.path.join(OUT_DIR, 'spus_eval_all_trajs.npz')
np.savez_compressed(npz_path,
                    metrics=all_metrics,
                    metric_names=np.array(METRIC_NAMES),
                    l2_ar=all_l2_ar,
                    l2_os=all_l2_os)

print(f'saved {csv_path}')
print(f'saved {npz_path}')

print(f'\n{"="*70}')
print('SUMMARY ACROSS ALL {N_TRAJ} TEST TRAJECTORIES')
print(f'{"="*70}')
m = all_metrics
for i, name in enumerate(METRIC_NAMES):
    vals = m[:, i]
    print(f'  {name:<20s}  mean={vals.mean():10.4e}  std={vals.std():10.4e}  '
          f'min={vals.min():10.4e}  max={vals.max():10.4e}')
