"""Squeeze: full baseline shootout including trigonometric LSQ (div-free Fourier basis).

Baselines tested:
  1. RBF Gaussian
  2. RBF Multiquadric
  3. RBF Thin-plate-spline
  4. IDW (Inverse Distance Weighting, p=2)
  5. Trigonometric div-free LSQ @ k_max=5, 8, 12
  6. Sensor-only Gappy POD (no full DNS access)

vs. EXP-080 (our PINN).

Compare on KE, u/v rel-L2, vorticity rel-L2, ek_ratio.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator

# Sensor placement / output dir overridable via env (default = legacy DNS QR-pivot, 不破壞既有行為)。
# 用 BASELINE_SENSOR_JSON 指向 LES_T50 placement 即可做 EXP-245 same-sensor 公平比較。
SENSOR_JSON = Path(os.environ.get(
    "BASELINE_SENSOR_JSON",
    "data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100.json"))
DNS_PATH = Path("data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy")
OUT_DIR = Path(os.environ.get("BASELINE_OUT_DIR", "artifacts/under_determined_proof"))

# Load
dns = np.load(DNS_PATH, allow_pickle=True).item()
u_dns = dns["u"]; v_dns = dns["v"]
T, N, _ = u_dns.shape; L = 1.0; dx = L / N

with open(SENSOR_JSON) as f:
    sensor_data = json.load(f)
K = sensor_data["K"]
sensor_pos = np.array(sensor_data["selected_coordinates"])

# Grid
x_g = np.linspace(0, 1, N, endpoint=False)
y_g = np.linspace(0, 1, N, endpoint=False)
XG, YG = np.meshgrid(x_g, y_g, indexing="ij")


# === Helpers ===
def sample_at_sensors(field):
    samples = np.zeros(K)
    for k, (xk, yk) in enumerate(sensor_pos):
        ix = int(np.round((xk * N) % N)); iy = int(np.round((yk * N) % N))
        samples[k] = field[ix, iy]
    return samples


def periodic_sensor_extend(sensor_pos, vals, L=1.0):
    sp = sensor_pos.copy(); vv = vals.copy()
    ext_p, ext_v = [], []
    for dxo in [-L, 0, L]:
        for dyo in [-L, 0, L]:
            if dxo == 0 and dyo == 0:
                continue
            mask = (sp[:, 0] + dxo >= -0.3) & (sp[:, 0] + dxo <= 1.3) & \
                   (sp[:, 1] + dyo >= -0.3) & (sp[:, 1] + dyo <= 1.3)
            ext_p.append(sp[mask] + np.array([dxo, dyo]))
            ext_v.append(vv[mask])
    return np.vstack([sp] + ext_p), np.concatenate([vv] + ext_v)


def compute_vorticity(u, v, dx=dx):
    dvdx = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dx)
    dudy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
    return dvdx - dudy


def ke_rel_err(u_p, v_p, u_t, v_t, indices):
    ke_p = 0.5 * (u_p[indices]**2 + v_p[indices]**2).mean(axis=(1, 2))
    ke_t = 0.5 * (u_t[indices]**2 + v_t[indices]**2).mean(axis=(1, 2))
    return float(np.abs((ke_p - ke_t) / ke_t).mean())


def rel_l2_per_snapshot(pred, true, indices):
    diff_l2 = np.sqrt(((pred[indices] - true[indices])**2).mean(axis=(1, 2)))
    true_l2 = np.sqrt((true[indices]**2).mean(axis=(1, 2)))
    return float((diff_l2 / true_l2).mean())


all_idx = np.arange(T)


# === Baseline 1-3: RBF kernels ===
def rbf_reconstruct(kernel, epsilon=10.0):
    """Returns (u_pred, v_pred) for all T."""
    u_out = np.zeros_like(u_dns); v_out = np.zeros_like(v_dns)
    grid_xy = np.stack([XG.flatten(), YG.flatten()], axis=-1)
    for t in range(T):
        u_vals = sample_at_sensors(u_dns[t]); v_vals = sample_at_sensors(v_dns[t])
        sp_ext, ue = periodic_sensor_extend(sensor_pos, u_vals)
        _, ve = periodic_sensor_extend(sensor_pos, v_vals)
        if kernel in ["thin_plate_spline", "linear", "cubic", "quintic"]:
            rbf_u = RBFInterpolator(sp_ext, ue, kernel=kernel, smoothing=0.0, neighbors=50)
            rbf_v = RBFInterpolator(sp_ext, ve, kernel=kernel, smoothing=0.0, neighbors=50)
        else:
            rbf_u = RBFInterpolator(sp_ext, ue, kernel=kernel, epsilon=epsilon,
                                     smoothing=0.0, neighbors=50)
            rbf_v = RBFInterpolator(sp_ext, ve, kernel=kernel, epsilon=epsilon,
                                     smoothing=0.0, neighbors=50)
        u_out[t] = rbf_u(grid_xy).reshape(N, N)
        v_out[t] = rbf_v(grid_xy).reshape(N, N)
    return u_out, v_out


# === Baseline 4: IDW (Inverse Distance Weighting, p=2) ===
def idw_reconstruct(p=2.0):
    u_out = np.zeros_like(u_dns); v_out = np.zeros_like(v_dns)
    # Compute distances from each grid point to each sensor (periodic)
    # grid_xy shape [N*N, 2], sensor_pos [K, 2]
    grid_flat = np.stack([XG.flatten(), YG.flatten()], axis=-1)  # [N², 2]
    # Periodic min distance
    dx_arr = grid_flat[:, None, 0] - sensor_pos[None, :, 0]  # [N², K]
    dy_arr = grid_flat[:, None, 1] - sensor_pos[None, :, 1]
    dx_arr = dx_arr - np.round(dx_arr / L) * L
    dy_arr = dy_arr - np.round(dy_arr / L) * L
    dist = np.sqrt(dx_arr**2 + dy_arr**2)  # [N², K]
    weights = 1.0 / (dist**p + 1e-10)
    weights /= weights.sum(axis=1, keepdims=True)
    for t in range(T):
        u_vals = sample_at_sensors(u_dns[t]); v_vals = sample_at_sensors(v_dns[t])
        u_out[t] = (weights @ u_vals).reshape(N, N)
        v_out[t] = (weights @ v_vals).reshape(N, N)
    return u_out, v_out


# === Baseline 5b: Dedup + Tikhonov-regularized div-free Trig LSQ ===
# Replaces the legacy `divfree_trig_lsq_reconstruct` (which enumerated both k and -k modes,
# producing a 2M-column matrix with rank ≤ M — i.e. a 196-dim redundant null-space at k_max=8
# that explodes under `rcond=None`). The dedup version below enumerates ONE representative per
# {k, -k} Hermitian pair (half-plane: kx > 0, or kx==0 and ky > 0), keeps both cos and sin
# amplitudes per mode → exactly M = 80/196/440 real DOFs for k_max ∈ {5, 8, 12}. With optional
# Tikhonov ridge (`ridge_alpha`) the borderline (k_max=8) and under-determined (k_max=12)
# regimes can be regularized and compared fairly.
def divfree_trig_lsq_dedup_reconstruct(k_max=5, ridge_alpha=0.0, verbose=False):
    """Half-plane (Hermitian-deduped) div-free Fourier LSQ with optional ridge.

    Stream-function basis:  ψ = Σ_q (a_q cos(2π k·x) + b_q sin(2π k·x))   over half-plane modes.
    Velocity:               u = ∂ψ/∂y,  v = −∂ψ/∂x  (incompressible by construction).
    """
    # Half-plane enumeration: kx > 0, OR (kx == 0 AND ky > 0)
    mode_indices = []
    for kx in range(-k_max, k_max + 1):
        for ky in range(-k_max, k_max + 1):
            if kx == 0 and ky == 0:
                continue
            if kx ** 2 + ky ** 2 > k_max ** 2:
                continue
            if kx > 0 or (kx == 0 and ky > 0):
                mode_indices.append((kx, ky))
    M_modes = len(mode_indices)
    M_dof = 2 * M_modes  # cos (a_q) + sin (b_q) amplitude per mode

    two_pi = 2.0 * np.pi
    modes = np.asarray(mode_indices, dtype=np.float64)   # (M_modes, 2)
    kx_arr = modes[:, 0]; ky_arr = modes[:, 1]

    # Vectorized A construction: (2K, M_dof) — sensor rows interleave (u, v)
    # phase_sensor[m, q] = 2π (kx_q · x_m + ky_q · y_m)
    phase_s = two_pi * (sensor_pos @ modes.T)            # (K, M_modes)
    cos_s = np.cos(phase_s); sin_s = np.sin(phase_s)
    A = np.zeros((2 * K, M_dof))
    A[0::2, 0::2] = -ky_arr[None, :] * sin_s             # u-row, a (cos) coef
    A[0::2, 1::2] =  ky_arr[None, :] * cos_s             # u-row, b (sin) coef
    A[1::2, 0::2] =  kx_arr[None, :] * sin_s             # v-row, a (cos) coef
    A[1::2, 1::2] = -kx_arr[None, :] * cos_s             # v-row, b (sin) coef

    # Diagnostic: condition number on this sensor placement
    if verbose:
        s = np.linalg.svd(A, compute_uv=False)
        kappa = s[0] / max(s[-1], 1e-30)
        print(f"    dedup k≤{k_max}:  M_dof={M_dof}  shape={A.shape}  "
              f"cond(A)={kappa:.2e}  s_min={s[-1]:.2e}  s_max={s[0]:.2e}",
              flush=True)

    # Vectorized reconstruction basis at full grid
    grid_xy = np.stack([XG.flatten(), YG.flatten()], axis=-1)  # (N², 2)
    phase_g = two_pi * (grid_xy @ modes.T)               # (N², M_modes)
    cos_g = np.cos(phase_g); sin_g = np.sin(phase_g)
    u_basis = np.zeros((N * N, M_dof))
    v_basis = np.zeros((N * N, M_dof))
    u_basis[:, 0::2] = -ky_arr[None, :] * sin_g
    u_basis[:, 1::2] =  ky_arr[None, :] * cos_g
    v_basis[:, 0::2] =  kx_arr[None, :] * sin_g
    v_basis[:, 1::2] = -kx_arr[None, :] * cos_g

    u_out = np.zeros_like(u_dns); v_out = np.zeros_like(v_dns)

    # Solver: Tikhonov-regularized LSQ when ridge_alpha > 0; otherwise plain LSQ.
    # Tikhonov:  (AᵀA + α I) coef = Aᵀ b   →  uses Cholesky once for all snapshots.
    if ridge_alpha > 0:
        AtA = A.T @ A + ridge_alpha * np.eye(M_dof)
        L_chol = np.linalg.cholesky(AtA)
        for t in range(T):
            u_vals = sample_at_sensors(u_dns[t])
            v_vals = sample_at_sensors(v_dns[t])
            b = np.empty(2 * K)
            b[0::2] = u_vals
            b[1::2] = v_vals
            rhs = A.T @ b
            y = np.linalg.solve(L_chol, rhs)
            coef = np.linalg.solve(L_chol.T, y)
            u_out[t] = (u_basis @ coef).reshape(N, N)
            v_out[t] = (v_basis @ coef).reshape(N, N)
    else:
        for t in range(T):
            u_vals = sample_at_sensors(u_dns[t])
            v_vals = sample_at_sensors(v_dns[t])
            b = np.empty(2 * K)
            b[0::2] = u_vals
            b[1::2] = v_vals
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            u_out[t] = (u_basis @ coef).reshape(N, N)
            v_out[t] = (v_basis @ coef).reshape(N, N)
    return u_out, v_out, M_dof


# === Baseline 5 (legacy): Trigonometric div-free LSQ ===
def divfree_trig_lsq_reconstruct(k_max=5):
    """Reconstruct via div-free Fourier basis (stream-function ψ) least-squares.

    u = ∂ψ/∂y, v = -∂ψ/∂x where ψ = Σ_q (a_q cos(2π k·x) + b_q sin(2π k·x))
    Sensor equations: u(x_k) = sensor_u[k], v(x_k) = sensor_v[k] for k=1..K.
    Total: 2K equations, 2M unknowns (M = num modes with |k|≤k_max).

    For k_max=5: M ≈ 80, 2M = 160 < 2K = 200 → over-determined → LSQ.
    """
    # Build mode list (skip (0,0))
    mode_indices = []
    for kx in range(-k_max, k_max + 1):
        for ky in range(-k_max, k_max + 1):
            if kx == 0 and ky == 0:
                continue
            if kx**2 + ky**2 <= k_max**2:
                mode_indices.append((kx, ky))
    M = len(mode_indices)

    two_pi = 2.0 * np.pi
    # Build A matrix [2K, 2M]: A @ [a_q; b_q] = [sensor_u; sensor_v]
    # u = ∂ψ/∂y → from cos(2π(kx X + ky Y)): -2π·ky·sin(...) coefficient for a
    #                               -2π·ky·cos(...) coefficient for b... wait let me redo.
    # Actually: ψ = a cos + b sin → ∂ψ/∂y = -a·2π·ky·sin + b·2π·ky·cos
    # v = -∂ψ/∂x = a·2π·kx·sin - b·2π·kx·cos
    # We drop the 2π scaling (absorbed into coefficients).
    A = np.zeros((2 * K, 2 * M))
    for k_idx, (xk, yk) in enumerate(sensor_pos):
        for q_idx, (kxq, kyq) in enumerate(mode_indices):
            phase = two_pi * (kxq * xk + kyq * yk)
            cos_p, sin_p = np.cos(phase), np.sin(phase)
            A[2*k_idx,     2*q_idx]     = -kyq * sin_p   # u from a (cos coeff)
            A[2*k_idx,     2*q_idx + 1] =  kyq * cos_p   # u from b (sin coeff)
            A[2*k_idx + 1, 2*q_idx]     =  kxq * sin_p   # v from a
            A[2*k_idx + 1, 2*q_idx + 1] = -kxq * cos_p   # v from b

    # Pre-compute basis at grid for reconstruction
    # u_grid = Σ_q [-ky·a·sin(phase_grid) + ky·b·cos(phase_grid)]
    # v_grid = Σ_q [ kx·a·sin(phase_grid) - kx·b·cos(phase_grid)]
    grid_x_flat = XG.flatten(); grid_y_flat = YG.flatten()
    u_basis = np.zeros((N*N, 2*M))   # u(grid) = u_basis @ coef
    v_basis = np.zeros((N*N, 2*M))
    for q_idx, (kxq, kyq) in enumerate(mode_indices):
        phase = two_pi * (kxq * grid_x_flat + kyq * grid_y_flat)
        cos_p, sin_p = np.cos(phase), np.sin(phase)
        u_basis[:, 2*q_idx]     = -kyq * sin_p
        u_basis[:, 2*q_idx + 1] =  kyq * cos_p
        v_basis[:, 2*q_idx]     =  kxq * sin_p
        v_basis[:, 2*q_idx + 1] = -kxq * cos_p

    u_out = np.zeros_like(u_dns); v_out = np.zeros_like(v_dns)
    for t in range(T):
        u_vals = sample_at_sensors(u_dns[t])
        v_vals = sample_at_sensors(v_dns[t])
        b = np.empty(2 * K)
        b[0::2] = u_vals
        b[1::2] = v_vals
        # LSQ: solve A @ coef = b
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        u_out[t] = (u_basis @ coef).reshape(N, N)
        v_out[t] = (v_basis @ coef).reshape(N, N)
    return u_out, v_out, M


# === Run all baselines ===
results = {}

print("Running RBF Gaussian (ε=10)...", flush=True)
u_p, v_p = rbf_reconstruct("gaussian", epsilon=10.0)
omega_p = np.array([compute_vorticity(u_p[t], v_p[t]) for t in range(T)])
omega_d = np.array([compute_vorticity(u_dns[t], v_dns[t]) for t in range(T)])
results["RBF Gaussian"] = {
    "ke": ke_rel_err(u_p, v_p, u_dns, v_dns, all_idx),
    "u_l2": rel_l2_per_snapshot(u_p, u_dns, all_idx),
    "v_l2": rel_l2_per_snapshot(v_p, v_dns, all_idx),
    "omega_l2": rel_l2_per_snapshot(omega_p, omega_d, all_idx),
    "dns_access": False,
}

print("Running RBF Multiquadric (ε=10)...")
u_p, v_p = rbf_reconstruct("multiquadric", epsilon=10.0)
omega_p = np.array([compute_vorticity(u_p[t], v_p[t]) for t in range(T)])
results["RBF Multiquadric"] = {
    "ke": ke_rel_err(u_p, v_p, u_dns, v_dns, all_idx),
    "u_l2": rel_l2_per_snapshot(u_p, u_dns, all_idx),
    "v_l2": rel_l2_per_snapshot(v_p, v_dns, all_idx),
    "omega_l2": rel_l2_per_snapshot(omega_p, omega_d, all_idx),
    "dns_access": False,
}

print("Running RBF Thin-plate-spline...", flush=True)
u_p, v_p = rbf_reconstruct("thin_plate_spline")
omega_p = np.array([compute_vorticity(u_p[t], v_p[t]) for t in range(T)])
results["RBF Thin-plate-spline"] = {
    "ke": ke_rel_err(u_p, v_p, u_dns, v_dns, all_idx),
    "u_l2": rel_l2_per_snapshot(u_p, u_dns, all_idx),
    "v_l2": rel_l2_per_snapshot(v_p, v_dns, all_idx),
    "omega_l2": rel_l2_per_snapshot(omega_p, omega_d, all_idx),
    "dns_access": False,
}

print("Running IDW p=2...", flush=True)
u_p, v_p = idw_reconstruct(p=2.0)
omega_p = np.array([compute_vorticity(u_p[t], v_p[t]) for t in range(T)])
results["IDW p=2"] = {
    "ke": ke_rel_err(u_p, v_p, u_dns, v_dns, all_idx),
    "u_l2": rel_l2_per_snapshot(u_p, u_dns, all_idx),
    "v_l2": rel_l2_per_snapshot(v_p, v_dns, all_idx),
    "omega_l2": rel_l2_per_snapshot(omega_p, omega_d, all_idx),
    "dns_access": False,
}

# --- Legacy (over-complete 2M) — only k_max=5 as regression check.
#     k_max ∈ {8, 12} are known to blow up with `rcond=None` on the 2M-redundant matrix
#     (see thesis/baseline_comparison_report.md note 138, 2026-05-28 audit). The dedup
#     variants below are the correct apples-to-apples comparison.
print("Running Div-free trig LSQ (legacy 2M parameterization) k_max=5 (regression check)...", flush=True)
u_p, v_p, M_modes = divfree_trig_lsq_reconstruct(k_max=5)
omega_p = np.array([compute_vorticity(u_p[t], v_p[t]) for t in range(T)])
name = f"Div-free trig LSQ k≤5 ({M_modes} modes, legacy 2M)"
results[name] = {
    "ke": ke_rel_err(u_p, v_p, u_dns, v_dns, all_idx),
    "u_l2": rel_l2_per_snapshot(u_p, u_dns, all_idx),
    "v_l2": rel_l2_per_snapshot(v_p, v_dns, all_idx),
    "omega_l2": rel_l2_per_snapshot(omega_p, omega_d, all_idx),
    "dns_access": False,
    "num_modes": M_modes,
    "parameterization": "legacy_2M_rank_deficient",
}

# --- Dedup (M DOFs, Hermitian-correct) + ridge variants ---
for k_max in [5, 8, 12]:
    for ridge_alpha in [0.0, 1e-3]:
        tag = f"ridge={ridge_alpha:.0e}" if ridge_alpha > 0 else "no ridge"
        print(f"Running Div-free trig LSQ (dedup) k_max={k_max}  {tag}...")
        u_p, v_p, M_dof = divfree_trig_lsq_dedup_reconstruct(
            k_max=k_max, ridge_alpha=ridge_alpha, verbose=(ridge_alpha == 0.0)
        )
        omega_p = np.array([compute_vorticity(u_p[t], v_p[t]) for t in range(T)])
        name = f"Div-free trig LSQ k≤{k_max} dedup ({M_dof} DOF, {tag})"
        results[name] = {
            "ke": ke_rel_err(u_p, v_p, u_dns, v_dns, all_idx),
            "u_l2": rel_l2_per_snapshot(u_p, u_dns, all_idx),
            "v_l2": rel_l2_per_snapshot(v_p, v_dns, all_idx),
            "omega_l2": rel_l2_per_snapshot(omega_p, omega_d, all_idx),
            "dns_access": False,
            "num_dof": M_dof,
            "parameterization": "dedup_half_plane",
            "ridge_alpha": ridge_alpha,
        }

# --- RBF Gaussian epsilon sweep (P3 audit response) ---
# Audit (2026-05-28) flagged that ε=10 is a narrow-kernel choice (~1 sensor spacing).
# A reviewer can challenge this single hyper-parameter; sweep validates the choice.
print("Running RBF Gaussian epsilon sweep...", flush=True)
for eps in [1.0, 3.0, 5.0, 20.0]:
    print(f"  RBF Gaussian (ε={eps})...")
    u_p, v_p = rbf_reconstruct("gaussian", epsilon=eps)
    omega_p = np.array([compute_vorticity(u_p[t], v_p[t]) for t in range(T)])
    results[f"RBF Gaussian (ε={eps})"] = {
        "ke": ke_rel_err(u_p, v_p, u_dns, v_dns, all_idx),
        "u_l2": rel_l2_per_snapshot(u_p, u_dns, all_idx),
        "v_l2": rel_l2_per_snapshot(v_p, v_dns, all_idx),
        "omega_l2": rel_l2_per_snapshot(omega_p, omega_d, all_idx),
        "dns_access": False,
        "epsilon": eps,
    }

# === Final summary ===
print(f"\n{'='*100}", flush=True)
print(f"BASELINE SQUEEZE COMPARISON (K={K}, T={T})")
print(f"{'='*100}\n", flush=True)
print(f"{'Method':<40} | {'KE%':>7} | {'u_L2%':>7} | {'v_L2%':>7} | {'ω_L2%':>7} | {'DNS?':>5}", flush=True)
print("-" * 95, flush=True)
print(f"{'EXP-080 (our PINN)':<40} | {10.68:>6.2f}% | {17.0:>6.2f}% | {20.2:>6.2f}% | {47.6:>6.2f}% | {'No':>5}")
for name, m in results.items():
    print(f"{name:<40} | {m['ke']*100:>6.2f}% | {m['u_l2']*100:>6.2f}% | {m['v_l2']*100:>6.2f}% | {m['omega_l2']*100:>6.2f}% | {'No' if not m['dns_access'] else 'YES':>5}", flush=True)
print(f"{'(reference) Gappy POD r=100 (val-only)':<40} | {0.23:>6.2f}% | {1.70:>6.2f}% | {2.90:>6.2f}% | {'—':>6} | {'YES':>5}")

# Save
OUT_DIR.mkdir(parents=True, exist_ok=True)
summary = {
    "K": int(K),
    "EXP_080": {"ke": 0.1068, "u_l2": 0.170, "v_l2": 0.202, "omega_l2": 0.476},
    **{name: m for name, m in results.items()},
    # Gappy POD r=100 reference — VAL-ONLY (n=41 held-out snapshots).
    # Replaces 2026-05-12 train+val-mixed numbers (0.0012 / 0.0085) which were
    # leakage-contaminated; honest val ceiling is ~2× higher (see baseline_comparison.py rerun
    # 2026-05-28 and thesis/baseline_comparison_report.md §5.4).
    "Gappy_POD_r100_val_only": {
        "ke": 0.00229, "u_l2": 0.01703, "v_l2": 0.02899, "dns_access": True,
        "leakage_status": "val-only honest ceiling (train_mean and SVD trained on 160 train snaps)",
    },
}
with open(OUT_DIR / "baseline_squeeze.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {OUT_DIR}/baseline_squeeze.json", flush=True)
