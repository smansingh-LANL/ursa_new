"""
Physics-based validation metrics for Compressible Euler equation predictions.

PDEgym / Reasoning4PDEs channel convention (shape (5, H, W) per snapshot):
  Channel 0: density (rho)
  Channel 1: horizontal velocity (u)
  Channel 2: vertical velocity (v)
  Channel 3: pressure (p)
  Channel 4: total energy (E)

Domain: unit square [0,1]^2, 128x128 grid, gamma = 1.4.

Spectral analysis uses ONLY three quantities, all built from
  kx, ky, and  k = sqrt(kx^2 + ky^2):
    - E(kx) : power summed over ky, as a function of |kx|
    - E(ky) : power summed over kx, as a function of |ky|
    - E(k)  : power binned by the radial wavenumber k = sqrt(kx^2 + ky^2)

Compressible Euler is INVISCID (no diffusion term), so flows are not
isotropic: E(kx) and E(ky) generally differ, and that difference is the
physical signal we want to preserve.

References:
  https://github.com/lanl/Reasoning4PDEs/blob/main/physics_rewards.py
  https://github.com/lanl/SPUS-Small-PDE-U-net-Solver
  PDEgym: https://arxiv.org/abs/2405.19101
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union

GAMMA = 1.4
EPS = 1e-30


# ──────────────────────────────────────────────────────────
#  State accessor
# ──────────────────────────────────────────────────────────

@dataclass
class EulerState:
    """Wrapper around a (5, H, W) array/tensor in PDEgym primitive format."""
    data: Union[torch.Tensor, np.ndarray]

    def _to_np(self, x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    @property
    def rho(self):  return self.data[0]
    @property
    def u(self):    return self.data[1]
    @property
    def v(self):    return self.data[2]
    @property
    def p(self):    return self.data[3]
    @property
    def E(self):    return self.data[4]

    @property
    def rho_u(self):  return self.rho * self.u
    @property
    def rho_v(self):  return self.rho * self.v
    @property
    def kinetic_energy_density(self):
        return 0.5 * self.rho * (self.u**2 + self.v**2)
    @property
    def internal_energy(self):
        return self.E - self.kinetic_energy_density
    @property
    def entropy(self):
        rho = self._safe_positive(self.rho)
        p = self._safe_positive(self.p)
        return p / (rho ** GAMMA)
    @property
    def sound_speed(self):
        rho = self._safe_positive(self.rho)
        p = self._safe_positive(self.p)
        if isinstance(p, torch.Tensor):
            return torch.sqrt(GAMMA * p / rho)
        return np.sqrt(GAMMA * p / rho)

    def _safe_positive(self, x):
        if isinstance(x, torch.Tensor):
            return torch.clamp(x, min=EPS)
        return np.clip(x, EPS, None)


# ──────────────────────────────────────────────────────────
#  Conservation reward functions (match Reasoning4PDEs)
# ──────────────────────────────────────────────────────────

def reward_mass_conservation(x, y):
    """Mass: -|sum(rho_out) - sum(rho_in)| / |sum(rho_in)|."""
    rho_in, rho_out = x[0], y[0]
    if isinstance(rho_in, torch.Tensor):
        norm = torch.abs(torch.sum(rho_in))
        flux = torch.abs(torch.sum(rho_out - rho_in))
        return -torch.mean(flux / norm)
    norm = abs(np.sum(rho_in))
    flux = abs(np.sum(rho_out - rho_in))
    return -float(flux / (norm + EPS))


def reward_momentum_conservation(x, y):
    """Momentum: -|sum(rho*vel_out) - sum(rho*vel_in)| / |sum(rho*vel_in)|."""
    rhoV_in = x[0] * x[1:3]
    rhoV_out = y[0] * y[1:3]
    if isinstance(rhoV_in, torch.Tensor):
        norm = torch.abs(torch.sum(rhoV_in))
        flux = torch.abs(torch.sum(rhoV_out - rhoV_in))
        return -torch.mean(flux / norm)
    norm = abs(np.sum(rhoV_in))
    flux = abs(np.sum(rhoV_out - rhoV_in))
    return -float(flux / (norm + EPS))


def reward_energy_conservation(x, y):
    """Energy: -|sum(E_out) - sum(E_in)| / |sum(E_in)|."""
    E_in, E_out = x[4], y[4]
    if isinstance(E_in, torch.Tensor):
        norm = torch.abs(torch.sum(E_in))
        flux = torch.abs(torch.sum(E_out - E_in))
        return -torch.mean(flux / norm)
    norm = abs(np.sum(E_in))
    flux = abs(np.sum(E_out - E_in))
    return -float(flux / (norm + EPS))


# ──────────────────────────────────────────────────────────
#  Power spectra:  E(kx), E(ky), E(k)   with k = sqrt(kx^2 + ky^2)
# ──────────────────────────────────────────────────────────

def _psd2d(field_2d: np.ndarray) -> np.ndarray:
    """2D power spectral density S(kx, ky) = |FFT2D(q)|^2 / N^2 (not shifted)."""
    H, W = field_2d.shape
    fft2 = np.fft.fft2(field_2d)
    return (np.abs(fft2) ** 2) / (H * W)


def spectrum_kx(field_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    1D spectrum along x:  E(kx) = sum_ky S(kx, ky), folded onto |kx|.

    Returns (kx_vals, E_kx), kx_vals = 0..W//2 (integer cycles over the box).
    """
    H, W = field_2d.shape
    S = _psd2d(field_2d)
    E_over_kx = S.sum(axis=0)               # length W, indexed by kx (fft order)
    kx = np.fft.fftfreq(W) * W              # integer wavenumbers, fft order
    n = W // 2 + 1
    kx_vals = np.arange(n)
    E_kx = np.zeros(n)
    for i in range(W):
        E_kx[int(abs(kx[i]))] += E_over_kx[i]   # fold +/- kx together
    return kx_vals, E_kx


def spectrum_ky(field_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    1D spectrum along y:  E(ky) = sum_kx S(kx, ky), folded onto |ky|.
    """
    H, W = field_2d.shape
    S = _psd2d(field_2d)
    E_over_ky = S.sum(axis=1)               # length H, indexed by ky
    ky = np.fft.fftfreq(H) * H
    n = H // 2 + 1
    ky_vals = np.arange(n)
    E_ky = np.zeros(n)
    for i in range(H):
        E_ky[int(abs(ky[i]))] += E_over_ky[i]
    return ky_vals, E_ky


def spectrum_k(field_2d: np.ndarray, n_bins: Optional[int] = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """
    Radial spectrum:  E(k) = sum over the shell |k| in [k, k+1) of S(kx, ky),
    where  k = sqrt(kx^2 + ky^2).

    Returns (k_vals, E_k).
    """
    H, W = field_2d.shape
    if n_bins is None:
        n_bins = min(H, W) // 2
    S = _psd2d(field_2d)

    kx = np.fft.fftfreq(W) * W
    ky = np.fft.fftfreq(H) * H
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)               # <-- radial wavenumber, with sqrt

    k_vals = np.arange(n_bins)
    E_k = np.zeros(n_bins)
    k_int = np.round(K).astype(int)
    for b in range(n_bins):
        mask = k_int == b
        if mask.any():
            E_k[b] = S[mask].sum()
    return k_vals, E_k


def spectrum_relative_error(E_pred: np.ndarray, E_ref: np.ndarray) -> float:
    """Relative L2 error between two 1D spectra of equal binning."""
    n = min(len(E_pred), len(E_ref))
    denom = np.linalg.norm(E_ref[:n]) + EPS
    return float(np.linalg.norm(E_pred[:n] - E_ref[:n]) / denom)


def anisotropy_from_axis_spectra(field_2d: np.ndarray) -> float:
    """
    Directional imbalance derived ONLY from E(kx) and E(ky), via their
    mean wavenumbers:

        kx_mean = sum_kx kx * E(kx) / sum_kx E(kx)
        ky_mean = sum_ky ky * E(ky) / sum_ky E(ky)
        A       = (kx_mean - ky_mean) / (kx_mean + ky_mean)

    A = 0  -> same spread of scales in x and y (isotropic)
    A > 0  -> finer structure in x (e.g. vertical fronts)
    A < 0  -> finer structure in y (e.g. horizontal fronts)
    """
    kx_vals, E_kx = spectrum_kx(field_2d)
    ky_vals, E_ky = spectrum_ky(field_2d)
    # exclude DC (k=0) so the constant background doesn't dominate
    kx_mean = float((kx_vals[1:] * E_kx[1:]).sum() / (E_kx[1:].sum() + EPS))
    ky_mean = float((ky_vals[1:] * E_ky[1:]).sum() / (E_ky[1:].sum() + EPS))
    return (kx_mean - ky_mean) / (kx_mean + ky_mean + EPS)


def spectral_errors(pred_field: np.ndarray, ref_field: np.ndarray) -> dict:
    """
    Compare predicted vs reference scalar field across the three spectra.
    """
    _, Ekx_p = spectrum_kx(pred_field)
    _, Ekx_r = spectrum_kx(ref_field)
    _, Eky_p = spectrum_ky(pred_field)
    _, Eky_r = spectrum_ky(ref_field)
    _, Ek_p = spectrum_k(pred_field)
    _, Ek_r = spectrum_k(ref_field)

    A_p = anisotropy_from_axis_spectra(pred_field)
    A_r = anisotropy_from_axis_spectra(ref_field)

    return {
        "err_kx": spectrum_relative_error(Ekx_p, Ekx_r),
        "err_ky": spectrum_relative_error(Eky_p, Eky_r),
        "err_k": spectrum_relative_error(Ek_p, Ek_r),
        "anisotropy_pred": float(A_p),
        "anisotropy_ref": float(A_r),
        "anisotropy_abs_diff": float(abs(A_p - A_r)),
    }


# ──────────────────────────────────────────────────────────
#  Physical constraint checks
# ──────────────────────────────────────────────────────────

def constraint_violations(state: EulerState) -> dict:
    rho = state._to_np(state.rho)
    p = state._to_np(state.p)
    n = rho.size
    return {
        "density_violation_frac": float((rho <= 0).sum()) / n,
        "pressure_violation_frac": float((p <= 0).sum()) / n,
    }


def entropy_metrics(state0: EulerState, state1: EulerState) -> dict:
    s0 = state0._to_np(state0.entropy)
    s1 = state1._to_np(state1.entropy)
    decreased = s1 < s0 * (1.0 - 1e-8)
    ratio = np.mean(s1) / (np.mean(s0) + EPS)
    return {
        "entropy_decrease_frac": float(decreased.sum()) / s0.size,
        "mean_entropy_ratio": float(ratio),
    }


def eos_consistency(state: EulerState) -> dict:
    """E should equal 0.5*rho*(u^2+v^2) + p/(gamma-1)."""
    E_from_eos = state._to_np(state.kinetic_energy_density + state.p / (GAMMA - 1.0))
    E_actual = state._to_np(state.E)
    diff = E_actual - E_from_eos
    rel_err = np.linalg.norm(diff) / (np.linalg.norm(E_actual) + EPS)
    return {"eos_relative_error": float(rel_err),
            "eos_max_error": float(np.abs(diff).max())}


# ──────────────────────────────────────────────────────────
#  Pointwise errors vs reference
# ──────────────────────────────────────────────────────────

def pointwise_errors(pred: EulerState, ref: EulerState) -> dict:
    names = ["rho", "u", "v", "p", "E"]
    per_field = {}
    total_l2_sq = 0.0
    total_linf = 0.0
    for i, name in enumerate(names):
        p_f = pred._to_np(pred.data[i])
        r_f = ref._to_np(ref.data[i])
        diff = p_f - r_f
        denom = np.linalg.norm(r_f) + EPS
        rel_l2 = float(np.linalg.norm(diff) / denom)
        per_field[name] = rel_l2
        total_l2_sq += rel_l2 ** 2
        total_linf = max(total_linf, float(np.abs(diff).max()))
    return {"l2_error": float(np.sqrt(total_l2_sq)),
            "linf_error": total_linf,
            "field_errors": per_field}


# ──────────────────────────────────────────────────────────
#  Full evaluation
# ──────────────────────────────────────────────────────────

@dataclass
class MetricResult:
    mass_reward: float = 0.0
    momentum_reward: float = 0.0
    energy_reward: float = 0.0
    density_violation_frac: float = 0.0
    pressure_violation_frac: float = 0.0
    entropy_decrease_frac: float = 0.0
    mean_entropy_ratio: float = 1.0
    eos_relative_error: float = 0.0

    # Anisotropy of the prediction (reference-free)
    density_anisotropy_pred: Optional[float] = None
    ke_anisotropy_pred: Optional[float] = None

    # Spectral comparison vs reference
    density_spectra: dict = field(default_factory=dict)
    ke_spectra: dict = field(default_factory=dict)

    # Pointwise vs reference
    l2_error: Optional[float] = None
    linf_error: Optional[float] = None
    field_errors: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== Compressible Euler Physics Metrics ===",
            f"  Mass conservation reward:      {self.mass_reward:.6e}",
            f"  Momentum conservation reward:  {self.momentum_reward:.6e}",
            f"  Energy conservation reward:    {self.energy_reward:.6e}",
            f"  Density violation fraction:    {self.density_violation_frac:.4f}",
            f"  Pressure violation fraction:   {self.pressure_violation_frac:.4f}",
            f"  Entropy decrease fraction:     {self.entropy_decrease_frac:.4f}",
            f"  Mean entropy ratio:            {self.mean_entropy_ratio:.6f}",
            f"  EOS consistency error:         {self.eos_relative_error:.6e}",
        ]
        if self.density_anisotropy_pred is not None:
            lines.append(f"  Density anisotropy A (pred):   {self.density_anisotropy_pred:+.4f}")
        if self.ke_anisotropy_pred is not None:
            lines.append(f"  KE anisotropy A (pred):        {self.ke_anisotropy_pred:+.4f}")
        if self.density_spectra:
            d = self.density_spectra
            lines.append("  Density spectra err (vs ref):")
            lines.append(f"    E(kx) err: {d['err_kx']:.6e}")
            lines.append(f"    E(ky) err: {d['err_ky']:.6e}")
            lines.append(f"    E(k)  err: {d['err_k']:.6e}")
            lines.append(f"    aniso |dA|: {d['anisotropy_abs_diff']:.6e}"
                         f"  (pred {d['anisotropy_pred']:+.3f} vs ref {d['anisotropy_ref']:+.3f})")
        if self.ke_spectra:
            d = self.ke_spectra
            lines.append("  KE spectra err (vs ref):")
            lines.append(f"    E(kx) err: {d['err_kx']:.6e}")
            lines.append(f"    E(ky) err: {d['err_ky']:.6e}")
            lines.append(f"    E(k)  err: {d['err_k']:.6e}")
        if self.l2_error is not None:
            lines.append(f"  L2 error vs reference:         {self.l2_error:.6e}")
        if self.linf_error is not None:
            lines.append(f"  Linf error vs reference:       {self.linf_error:.6e}")
        if self.field_errors:
            lines.append("  Per-field relative L2 errors:")
            for k, v in self.field_errors.items():
                lines.append(f"    {k:>6s}: {v:.6e}")
        return "\n".join(lines)


def evaluate(pred: np.ndarray, initial: np.ndarray,
             reference: Optional[np.ndarray] = None) -> MetricResult:
    """
    Evaluate physics metrics for one prediction step.

    Args:
        pred:      model output at t+1, shape (5, H, W) = [rho, u, v, p, E]
        initial:   input state at t,    shape (5, H, W)
        reference: ground truth at t+1, shape (5, H, W), optional
    """
    s_init = EulerState(initial)
    s_pred = EulerState(pred)

    result = MetricResult(
        mass_reward=float(reward_mass_conservation(initial, pred)),
        momentum_reward=float(reward_momentum_conservation(initial, pred)),
        energy_reward=float(reward_energy_conservation(initial, pred)),
        **constraint_violations(s_pred),
        **entropy_metrics(s_init, s_pred),
        eos_relative_error=eos_consistency(s_pred)["eos_relative_error"],
    )

    rho_pred = s_pred._to_np(s_pred.rho)
    ke_pred = s_pred._to_np(s_pred.kinetic_energy_density)
    result.density_anisotropy_pred = anisotropy_from_axis_spectra(rho_pred)
    result.ke_anisotropy_pred = anisotropy_from_axis_spectra(ke_pred)

    if reference is not None:
        s_ref = EulerState(reference)
        pw = pointwise_errors(s_pred, s_ref)
        result.l2_error = pw["l2_error"]
        result.linf_error = pw["linf_error"]
        result.field_errors = pw["field_errors"]

        rho_ref = s_ref._to_np(s_ref.rho)
        ke_ref = s_ref._to_np(s_ref.kinetic_energy_density)
        result.density_spectra = spectral_errors(rho_pred, rho_ref)
        result.ke_spectra = spectral_errors(ke_pred, ke_ref)

    return result


def evaluate_rollout(predictions: list[np.ndarray], initial: np.ndarray,
                     references: Optional[list[np.ndarray]] = None) -> list[MetricResult]:
    results = []
    prev = initial
    for i, pred in enumerate(predictions):
        ref = references[i] if references is not None else None
        results.append(evaluate(pred, prev, ref))
        prev = pred
    return results


# ──────────────────────────────────────────────────────────
#  Self-test with synthetic data
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    H, W = 128, 128
    rng = np.random.default_rng(42)

    rho = np.clip(1.0 + 0.1 * rng.standard_normal((H, W)), 0.1, None)
    u = 0.5 * rng.standard_normal((H, W))
    v = 0.5 * rng.standard_normal((H, W))
    p = np.clip(1.0 + 0.1 * rng.standard_normal((H, W)), 0.1, None)
    E = 0.5 * rho * (u**2 + v**2) + p / (GAMMA - 1)
    initial = np.stack([rho, u, v, p, E], axis=0)

    pred = initial.copy()
    pred += 0.01 * rng.standard_normal(pred.shape)
    pred[0] = np.clip(pred[0], 0.01, None)
    pred[3] = np.clip(pred[3], 0.01, None)

    result = evaluate(pred, initial, reference=initial)
    print(result.summary())
