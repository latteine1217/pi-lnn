#!/usr/bin/env python3
"""analyze_gi_supplementary.py — GI test supplementary analysis (dt-conv + seed sensitivity).

What:
    Two opus-suggested supplementary checks not covered by main analyze_grid_independence.py:

    1. dt-convergence: compare N=256 dt=2.5e-4 vs dt=1.25e-4 at common time t=0.5
       → verify temporal truncation error << spatial truncation error
       → without this, "spatial convergence" claim is contaminated by dt error

    2. seed sensitivity: compare seed=42 vs seed=1 runs at N=256, N=512
       → verify single-realization GI test is representative of ensemble
       → guards against "lucky seed" critique on chaotic system

Why:
    Opus reviewer F1 + F3. Without these, paper §Methods has 2 attackable surfaces.

Usage:
    uv run python scripts/analyze_gi_supplementary.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DATA = Path("data/dns/gi_test_re10000")


def load(p: Path) -> dict:
    return np.load(p, allow_pickle=True).item()


def rel_L2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def spectral_interpolate(field: np.ndarray, target_N: int) -> np.ndarray:
    """Zero-pad in spectral space (same as analyze_grid_independence.spectral_interpolate)."""
    N = field.shape[-1]
    if target_N == N:
        return field
    hat = np.fft.fft2(field)
    hat_shift = np.fft.fftshift(hat)
    pad = (target_N - N) // 2
    padded_shift = np.zeros((target_N, target_N), dtype=complex)
    padded_shift[pad:pad + N, pad:pad + N] = hat_shift
    padded = np.fft.ifftshift(padded_shift)
    return np.real(np.fft.ifft2(padded) * (target_N / N) ** 2)


# ── 1. dt-convergence check ───────────────────────────────────────────
print("=" * 72)
print("1. dt-convergence check (N=256, t=0.5)")
print("=" * 72)

dt_full = load(DATA / "kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_seed42_icspectral.npy")
dt_half = load(DATA / "kolmogorov_dns_fp64_etdrk4_Re10000_N256_T0p5_dt1p25e4_si50_seed42_icspectral_dtconv.npy")

# Find t=0.5 frame in each
t_full = np.asarray(dt_full["time"], dtype=np.float64)
t_half = np.asarray(dt_half["time"], dtype=np.float64)
idx_full = int(np.argmin(np.abs(t_full - 0.5)))
idx_half = int(np.argmin(np.abs(t_half - 0.5)))
print(f"dt=2.5e-4 frame {idx_full} at t={t_full[idx_full]:.4f}")
print(f"dt=1.25e-4 frame {idx_half} at t={t_half[idx_half]:.4f}")

u_full = np.asarray(dt_full["u"], dtype=np.float64)[idx_full]
v_full = np.asarray(dt_full["v"], dtype=np.float64)[idx_full]
u_half = np.asarray(dt_half["u"], dtype=np.float64)[idx_half]
v_half = np.asarray(dt_half["v"], dtype=np.float64)[idx_half]

err_u_dt = rel_L2(u_full, u_half)
err_v_dt = rel_L2(v_full, v_half)
KE_full = float(0.5 * np.mean(u_full**2 + v_full**2))
KE_half = float(0.5 * np.mean(u_half**2 + v_half**2))

print(f"\nrel_L2(u, dt=2.5e-4 vs dt=1.25e-4) at t=0.5:  {err_u_dt:.3e}")
print(f"rel_L2(v, dt=2.5e-4 vs dt=1.25e-4) at t=0.5:  {err_v_dt:.3e}")
print(f"KE difference: {abs(KE_full - KE_half) / KE_half:.3e}")

# Compare to spatial error at t=0.5: N=256 vs N=1024 = 0.113%
SPATIAL_ERR_at_t05 = 1.13e-3  # from main analysis JSON
print(f"\nSpatial error rel_L2(u, N=256 vs N=1024) at t=0.5: {SPATIAL_ERR_at_t05:.3e}")
ratio = err_u_dt / SPATIAL_ERR_at_t05
print(f"\n→ dt error / spatial error ratio: {ratio:.3e}")
if ratio < 0.1:
    print(f"  ✅ PASS: dt error << spatial error (ratio {ratio:.3f} < 0.1, dt truncation negligible)")
else:
    print(f"  ⚠️  WARN: dt error not negligible vs spatial (ratio {ratio:.3f})")

# ── 2. seed sensitivity check ──────────────────────────────────────────
print()
print("=" * 72)
print("2. seed sensitivity check (compare seed=42 vs seed=1)")
print("=" * 72)

SPINUP_T = 2.0

for N in [256, 512]:
    print(f"\n--- N={N} ---")
    p42 = DATA / f"kolmogorov_dns_fp64_etdrk4_Re10000_N{N}_T5_dt2p5e4_si100_seed42_icspectral.npy"
    p01 = DATA / f"kolmogorov_dns_fp64_etdrk4_Re10000_N{N}_T5_dt2p5e4_si100_seed1_icspectral.npy"
    d42 = load(p42)
    d01 = load(p01)
    t42 = np.asarray(d42["time"], dtype=np.float64)
    t01 = np.asarray(d01["time"], dtype=np.float64)
    u42 = np.asarray(d42["u"], dtype=np.float64)
    v42 = np.asarray(d42["v"], dtype=np.float64)
    u01 = np.asarray(d01["u"], dtype=np.float64)
    v01 = np.asarray(d01["v"], dtype=np.float64)

    # Statistical: KE/Enstrophy time series
    KE42 = 0.5 * np.mean(u42**2 + v42**2, axis=(1, 2))
    KE01 = 0.5 * np.mean(u01**2 + v01**2, axis=(1, 2))
    om42 = np.asarray(d42["omega"], dtype=np.float64)
    om01 = np.asarray(d01["omega"], dtype=np.float64)
    Z42 = 0.5 * np.mean(om42**2, axis=(1, 2))
    Z01 = 0.5 * np.mean(om01**2, axis=(1, 2))

    mask = t42 >= SPINUP_T
    ke_diff_post = float(np.abs(KE42[mask] - KE01[mask]).max() / np.abs(KE42[mask]).max())
    z_diff_post = float(np.abs(Z42[mask] - Z01[mask]).max() / np.abs(Z42[mask]).max())
    # Pointwise rel_L2 at t=0.5 and t=5 (note: different IC → expect O(1) divergence)
    idx_t05 = int(np.argmin(np.abs(t42 - 0.5)))
    idx_t5 = int(np.argmin(np.abs(t42 - 5.0)))
    err_u_t05 = rel_L2(u42[idx_t05], u01[idx_t05])
    err_u_t5 = rel_L2(u42[idx_t5], u01[idx_t5])

    print(f"  Post-spinup (t>=2) KE max rel diff:        {ke_diff_post:.3e}")
    print(f"  Post-spinup (t>=2) Enstrophy max rel diff: {z_diff_post:.3e}")
    print(f"  Pointwise rel_L2(u, seed42 vs seed1) t=0.5: {err_u_t05:.3e}")
    print(f"  Pointwise rel_L2(u, seed42 vs seed1) t=5:   {err_u_t5:.3e}  (chaos-decoupled, expect O(1))")

    # Verdict: statistical agreement (chaos-immune)
    KE_GRID_DIFF_N256_vs_REF = 0.06e-2  # from main analysis (0.06%)
    if N == 256:
        spatial_err = KE_GRID_DIFF_N256_vs_REF
        if ke_diff_post < spatial_err * 50:  # seed变化能差至多 50x spatial
            print(f"  ✅ N={N} seed sensitivity OK: KE statistical {ke_diff_post*100:.3f}% < 50x spatial error")
        else:
            print(f"  ⚠️  N={N} seed sensitivity HIGH: KE statistical {ke_diff_post*100:.3f}% > 50x spatial error")

# ── 3. Save summary JSON ──────────────────────────────────────────────
out = {
    "dt_convergence": {
        "rel_L2_u_dt_full_vs_half_at_t0p5": err_u_dt,
        "rel_L2_v_dt_full_vs_half_at_t0p5": err_v_dt,
        "spatial_rel_L2_u_N256_vs_N1024_at_t0p5": SPATIAL_ERR_at_t05,
        "dt_error_over_spatial_error_ratio": ratio,
        "verdict": "PASS" if ratio < 0.1 else "WARN",
        "interpretation": (
            f"dt halving (2.5e-4 → 1.25e-4) at N=256 changes solution at t=0.5 by "
            f"{err_u_dt:.3e}, which is {ratio:.2f}× the spatial error (N=256 vs N=1024). "
            f"{'Temporal error << spatial error, dt=2.5e-4 temporally converged.' if ratio < 0.1 else 'dt error not negligible, may inflate spatial convergence claim.'}"
        ),
    },
    "seed_sensitivity": {},
}

print("\n" + "=" * 72)
print("Saved JSON: data/dns/gi_test_re10000/gi_supplementary_report.json")

out_path = DATA / "gi_supplementary_report.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
print("✅ Done.")
