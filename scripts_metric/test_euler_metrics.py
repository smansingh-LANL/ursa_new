"""
Verification tests for euler_metrics.py using analytically-known cases.

Each test constructs a field whose spectral / conservation answer we can
work out by hand, then checks the code reproduces it.

Spectral definitions under test (k = sqrt(kx^2 + ky^2)):
  E(kx) = sum_ky |FFT2D(q)|^2 / N^2 , folded onto |kx|
  E(ky) = sum_kx |FFT2D(q)|^2 / N^2 , folded onto |ky|
  E(k)  = sum over shell |k| in [b, b+1) of |FFT2D(q)|^2 / N^2

Run:  uv run python test_euler_metrics.py
"""

import numpy as np
import euler_metrics as em

H = W = 128
PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, condition, detail=""):
    tag = PASS if condition else FAIL
    results.append((tag, name, detail))
    print(f"[{tag}] {name}" + (f"  -- {detail}" if detail else ""))


def make_state(rho, u, v, p):
    """Build a (5,H,W) PDEgym state with EOS-consistent energy."""
    E = 0.5 * rho * (u**2 + v**2) + p / (em.GAMMA - 1.0)
    return np.stack([rho, u, v, p, E], axis=0)


# Coordinate grids on the unit square
xs = np.arange(W) / W
ys = np.arange(H) / H
X, Y = np.meshgrid(xs, ys)


# ──────────────────────────────────────────────────────────
# 1. Conservation: identical states -> all rewards exactly 0
# ──────────────────────────────────────────────────────────
rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * X)
s = make_state(rho0, 0.2 + 0.0 * X, 0.1 + 0.0 * X, 1.0 + 0.2 * np.cos(2 * np.pi * Y))
r_mass = em.reward_mass_conservation(s, s)
r_mom = em.reward_momentum_conservation(s, s)
r_en = em.reward_energy_conservation(s, s)
check("identical -> mass reward == 0", abs(r_mass) < 1e-12, f"{r_mass:.2e}")
check("identical -> momentum reward == 0", abs(r_mom) < 1e-12, f"{r_mom:.2e}")
check("identical -> energy reward == 0", abs(r_en) < 1e-12, f"{r_en:.2e}")


# ──────────────────────────────────────────────────────────
# 2. Mass reward when density is doubled -> exactly -1
#    -|2S - S| / |S| = -1
# ──────────────────────────────────────────────────────────
s2 = s.copy()
s2[0] = 2.0 * s[0]
r = em.reward_mass_conservation(s, s2)
check("density x2 -> mass reward == -1", abs(r - (-1.0)) < 1e-12, f"{r:.6f}")


# ──────────────────────────────────────────────────────────
# 3. Mass reward when density scaled by 1.1 -> exactly -0.1
# ──────────────────────────────────────────────────────────
s3 = s.copy()
s3[0] = 1.1 * s[0]
r = em.reward_mass_conservation(s, s3)
check("density x1.1 -> mass reward == -0.1", abs(r - (-0.1)) < 1e-9, f"{r:.6f}")


# ──────────────────────────────────────────────────────────
# 4. Single-frequency field cos(2*pi*m*x):
#    power must sit at kx = m, ky = 0
# ──────────────────────────────────────────────────────────
m = 7
fx = np.cos(2 * np.pi * m * X)            # varies in x only
kx_vals, Ekx = em.spectrum_kx(fx)
ky_vals, Eky = em.spectrum_ky(fx)
peak_kx = int(kx_vals[np.argmax(Ekx)])
peak_ky = int(ky_vals[np.argmax(Eky)])
check("cos(2pi*7*x): E(kx) peaks at kx=7", peak_kx == m, f"peak at {peak_kx}")
check("cos(2pi*7*x): E(ky) peaks at ky=0", peak_ky == 0, f"peak at {peak_ky}")


# ──────────────────────────────────────────────────────────
# 5. Radial spectrum peaks at k = m for cos(2*pi*m*x)
#    since k = sqrt(m^2 + 0^2) = m
# ──────────────────────────────────────────────────────────
k_vals, Ek = em.spectrum_k(fx)
peak_k = int(k_vals[np.argmax(Ek)])
check("cos(2pi*7*x): E(k) peaks at k=7", peak_k == m, f"peak at {peak_k}")


# ──────────────────────────────────────────────────────────
# 6. Radial location for a diagonal mode cos(2*pi*(3x + 4y)):
#    kx=3, ky=4 -> k = sqrt(9+16) = 5
# ──────────────────────────────────────────────────────────
fdiag = np.cos(2 * np.pi * (3 * X + 4 * Y))
k_vals, Ek = em.spectrum_k(fdiag)
peak_k = int(k_vals[np.argmax(Ek)])
check("cos(2pi(3x+4y)): E(k) peaks at k=5", peak_k == 5, f"peak at {peak_k}  (sqrt(3^2+4^2))")


# ──────────────────────────────────────────────────────────
# 7. Parseval: sum(E(k)) == sum(field^2)  (energy conserved by FFT)
# ──────────────────────────────────────────────────────────
field = np.cos(2 * np.pi * (3 * X + 4 * Y)) + 0.5 * np.sin(2 * np.pi * 6 * Y)
total_real = float(np.sum(field**2))
_, Ek = em.spectrum_k(field, n_bins=W)     # full range so nothing is dropped
total_spec = float(np.sum(Ek))
rel = abs(total_spec - total_real) / total_real
check("Parseval: sum(E(k)) == sum(field^2)", rel < 1e-6, f"rel diff {rel:.2e}")


# ──────────────────────────────────────────────────────────
# 8. Anisotropy sign:
#    vertical stripes (vary in x)   -> A > 0  (toward +1)
#    horizontal stripes (vary in y) -> A < 0  (toward -1)
#    isotropic noise                -> A ~ 0
# ──────────────────────────────────────────────────────────
A_vert = em.anisotropy_from_axis_spectra(np.cos(2 * np.pi * m * X))
A_horz = em.anisotropy_from_axis_spectra(np.cos(2 * np.pi * m * Y))
rng = np.random.default_rng(0)
A_iso = em.anisotropy_from_axis_spectra(rng.standard_normal((H, W)))
check("vertical stripes -> A == +1", abs(A_vert - 1.0) < 1e-9, f"A={A_vert:+.4f}")
check("horizontal stripes -> A == -1", abs(A_horz - (-1.0)) < 1e-9, f"A={A_horz:+.4f}")
check("isotropic noise -> |A| < 0.05", abs(A_iso) < 0.05, f"A={A_iso:+.4f}")


# ──────────────────────────────────────────────────────────
# 9. EOS consistency: a state built from the EOS has ~0 error;
#    corrupting E raises it.
# ──────────────────────────────────────────────────────────
st = em.EulerState(s)
eos_err = em.eos_consistency(st)["eos_relative_error"]
check("EOS-consistent state -> eos error ~ 0", eos_err < 1e-12, f"{eos_err:.2e}")

s_bad = s.copy()
s_bad[4] = s[4] * 1.5     # corrupt energy
eos_err_bad = em.eos_consistency(em.EulerState(s_bad))["eos_relative_error"]
check("corrupted energy -> eos error > 0.1", eos_err_bad > 0.1, f"{eos_err_bad:.4f}")


# ──────────────────────────────────────────────────────────
# 10. Identical pred vs ref -> spectral errors == 0
# ──────────────────────────────────────────────────────────
serr = em.spectral_errors(rho0, rho0)
check("identical fields -> E(kx) err == 0", serr["err_kx"] < 1e-12, f"{serr['err_kx']:.2e}")
check("identical fields -> E(ky) err == 0", serr["err_ky"] < 1e-12, f"{serr['err_ky']:.2e}")
check("identical fields -> E(k)  err == 0", serr["err_k"] < 1e-12, f"{serr['err_k']:.2e}")


# ──────────────────────────────────────────────────────────
# 11. Positivity detection: inject negatives, count must match
# ──────────────────────────────────────────────────────────
s_neg = s.copy()
s_neg[0, :10, :10] = -1.0       # 100 negative density cells
viol = em.constraint_violations(em.EulerState(s_neg))
expected = 100 / (H * W)
check("density positivity: 100 negatives counted",
      abs(viol["density_violation_frac"] - expected) < 1e-12,
      f"{viol['density_violation_frac']:.6f} vs {expected:.6f}")


# ──────────────────────────────────────────────────────────
print("\n" + "=" * 50)
n_pass = sum(1 for t, _, _ in results if t == PASS)
n_total = len(results)
print(f"RESULT: {n_pass}/{n_total} checks passed")
if n_pass != n_total:
    print("FAILURES:")
    for t, name, detail in results:
        if t == FAIL:
            print(f"  - {name}: {detail}")
