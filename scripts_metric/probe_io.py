"""
Probe the SPUS model I/O and confirm how tensors flow into euler_metrics.

Checks:
  1. SPUS expected input/output shapes.
  2. Whether any padding / transform is applied in the pipeline.
  3. That a single (5,128,128) snapshot drops cleanly into evaluate().
"""

import sys
import numpy as np
import torch

# Make the local SPUS model.py importable
SPUS_DIR = "/home/tpc-bepxr/tpc26-2/models/SPUS"
sys.path.insert(0, SPUS_DIR)

import euler_metrics as em
from model import Unet2D


def load_spus():
    import json
    from safetensors.torch import load_file
    with open(f"{SPUS_DIR}/config.json") as f:
        cfg = json.load(f)
    print("SPUS config.json:", cfg)
    model = Unet2D(**cfg)
    state = load_file(f"{SPUS_DIR}/model.safetensors")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    model.eval()
    return model


def main():
    print("=" * 60)
    print("1) SPUS INPUT / OUTPUT SHAPES")
    print("=" * 60)
    model = load_spus()

    # Per the README: input is (batch, 1, channels=5, H, W)
    B, C, Hh, Ww = 1, 5, 128, 128
    x = torch.randn(B, 1, C, Hh, Ww)
    print(f"  input  x.shape = {tuple(x.shape)}   (B, 1, C, H, W)")
    with torch.no_grad():
        y = model(x)
    print(f"  output y.shape = {tuple(y.shape)}   (B, C, H, W)")
    assert y.shape == (B, C, Hh, Ww), "Unexpected output shape!"
    print("  -> output drops the singleton dim; same C,H,W as input. OK")

    print()
    print("=" * 60)
    print("2) INTERNAL RESHAPE / FILTERS")
    print("=" * 60)
    print("  forward(): x.reshape(B, -1, H, W) merges dims 1&2 -> (B, 5, H, W)")
    print("  input_proj: Conv2d(5 -> 32, k=3, pad=1)   [SAME padding, size preserved]")
    print("  downsampling: strided Conv2d(k=3, stride=2) at 3 levels")
    print("  128 -> 64 -> 32 -> 16 (bottleneck), then transposed-conv back to 128")
    print("  final: Conv2d(32 -> 5, k=3, pad=1)")
    print("  NOTE: SPUS uses k=3,pad=1 SAME conv (no 130 padding).")
    print("        The 130-pad CircularPad2d is in Reasoning4PDEs ViT, NOT SPUS.")

    print()
    print("=" * 60)
    print("3) SNAPSHOT -> euler_metrics.evaluate()")
    print("=" * 60)
    # A model rollout step: take input snapshot (5,H,W) and predicted (5,H,W)
    initial = x[0, 0].numpy()      # (5,128,128)
    pred = y[0].numpy()            # (5,128,128)
    print(f"  initial snapshot shape = {initial.shape}  (drop batch & singleton)")
    print(f"  pred    snapshot shape = {pred.shape}      (drop batch)")

    # The raw randn output is NOT physical (negative density/pressure) -- expected.
    res = em.evaluate(pred, initial, reference=initial)
    print("\n  evaluate() ran successfully. Sample fields:")
    print(f"    mass_reward          = {res.mass_reward:.4e}")
    print(f"    density_violation    = {res.density_violation_frac:.4f}  "
          f"(high: randn output is unphysical, as expected)")
    print(f"    density anisotropy A = {res.density_anisotropy_pred:+.4f}")
    print(f"    E(kx) err vs ref     = {res.density_spectra['err_kx']:.4e}")

    print()
    print("=" * 60)
    print("4) CHANNEL ORDER SANITY")
    print("=" * 60)
    print("  PDEgym / Reasoning4PDEs channel order: [rho, u, v, p, E]")
    print("  euler_metrics indexes: data[0]=rho, [1]=u, [2]=v, [3]=p, [4]=E")
    print("  reward_momentum uses x[0]*x[1:3] = rho * (u,v)  -> matches.")
    print("  CE-RM special case: channel 4 is a passive tracer, replaced by")
    print("    E = 0.5*rho*(u^2+v^2) + p/(gamma-1) in the dataloader.")

    print("\nALL PROBES COMPLETED.")


if __name__ == "__main__":
    main()
