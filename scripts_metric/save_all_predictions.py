"""
Run SPUS on all 715 CE-RP test trajectories and save full field predictions
to Lustre as a NetCDF file accessible to all team members.

Output: /lustre/fs01/hackathons/teams/tpc26-2/tpc26_spus_predictions.nc
  - gt:      (715, 21, 5, 128, 128) float32  — ground truth
  - pred_ar: (715, 21, 5, 128, 128) float32  — autoregressive predictions
  - pred_os: (715, 21, 5, 128, 128) float32  — one-step (teacher-forced)
"""
import time, os, sys, numpy as np, torch, netCDF4 as nc

sys.path.insert(0, '/lustre/fs01/hackathons/teams/tpc26-2/models/SPUS')
from huggingface_hub import PyTorchModelHubMixin
from model import Unet2D

class SPUSUnet2D(Unet2D, PyTorchModelHubMixin):
    pass

DATA_FILE = '/home/tpc-sjqyt/tpc26-2/data/PDEgym/CE-RP/CE-RP_test.nc'
SPUS_DIR = '/lustre/fs01/hackathons/teams/tpc26-2/models/SPUS'
OUT_FILE = '/lustre/fs01/hackathons/teams/tpc26-2/tpc26_spus_predictions.nc'
VAR = 'data'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device = {device}, torch = {torch.__version__}')

model = SPUSUnet2D.from_pretrained(SPUS_DIR).to(device).eval()
print('model loaded')

src = nc.Dataset(DATA_FILE, 'r')
N = src.variables[VAR].shape[0]
T = src.variables[VAR].shape[1]
C = src.variables[VAR].shape[2]
H = src.variables[VAR].shape[3]
W = src.variables[VAR].shape[4]
print(f'source: {N} trajectories, {T} timesteps, {C} channels, {H}x{W}')

out = nc.Dataset(OUT_FILE, 'w', format='NETCDF4')
out.description = 'SPUS (36M U-Net) predictions on CE-RP test set (715 trajectories)'
out.model = 'siddik-lanl/spus-pde-unet-36m'
out.source_data = DATA_FILE
out.channels = 'rho, u, v, p, E'
out.created = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())

out.createDimension('trajectory', N)
out.createDimension('time', T)
out.createDimension('channel', C)
out.createDimension('x', H)
out.createDimension('y', W)

dims = ('trajectory', 'time', 'channel', 'x', 'y')
chunking = (1, 1, C, H, W)

v_gt = out.createVariable('gt', 'f4', dims, chunksizes=chunking, zlib=False)
v_gt.description = 'Ground truth from CE-RP_test.nc'

v_ar = out.createVariable('pred_ar', 'f4', dims, chunksizes=chunking, zlib=False)
v_ar.description = 'Autoregressive predictions (feed own output forward from t=0)'

v_os = out.createVariable('pred_os', 'f4', dims, chunksizes=chunking, zlib=False)
v_os.description = 'One-step / teacher-forced predictions (feed GT at each step)'

x_warmup = torch.randn(1, 1, C, H, W, device=device)
with torch.no_grad():
    model(x_warmup)
torch.cuda.synchronize()

t_global = time.perf_counter()

for idx in range(N):
    gt = np.asarray(src.variables[VAR][idx]).astype(np.float32)

    ar = np.empty_like(gt)
    ar[0] = gt[0]
    cur = gt[0]
    for t in range(1, T):
        x = torch.as_tensor(cur, dtype=torch.float32, device=device).reshape(1, 1, C, H, W)
        with torch.no_grad():
            y = model(x)
        cur = y[0].cpu().numpy()
        ar[t] = cur

    os_ = np.empty_like(gt)
    os_[0] = gt[0]
    for t in range(1, T):
        x = torch.as_tensor(gt[t - 1], dtype=torch.float32, device=device).reshape(1, 1, C, H, W)
        with torch.no_grad():
            y = model(x)
        os_[t] = y[0].cpu().numpy()

    v_gt[idx] = gt
    v_ar[idx] = ar
    v_os[idx] = os_

    if (idx + 1) % 50 == 0 or idx == 0:
        elapsed = time.perf_counter() - t_global
        eta = elapsed / (idx + 1) * (N - idx - 1)
        fsize = os.path.getsize(OUT_FILE) / (1024**3)
        print(f'  traj {idx+1:>4}/{N}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s  file={fsize:.1f}GB')

src.close()
out.close()

total = time.perf_counter() - t_global
fsize = os.path.getsize(OUT_FILE) / (1024**3)
print(f'\nDone: {N} trajectories in {total:.1f}s')
print(f'saved {OUT_FILE}  ({fsize:.1f} GB)')

os.chmod(OUT_FILE, 0o664)
print('permissions set to 664 (rw-rw-r--) for team access')
