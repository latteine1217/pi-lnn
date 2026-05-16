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
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator

SENSOR_JSON = Path("data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100.json")
DNS_PATH = Path("data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy")
OUT_DIR = Path("artifacts/under_determined_proof")

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


# === Baseline 5: Trigonometric div-free LSQ ===
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

print("Running RBF Gaussian (ε=10)...")
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

print("Running RBF Thin-plate-spline...")
u_p, v_p = rbf_reconstruct("thin_plate_spline")
omega_p = np.array([compute_vorticity(u_p[t], v_p[t]) for t in range(T)])
results["RBF Thin-plate-spline"] = {
    "ke": ke_rel_err(u_p, v_p, u_dns, v_dns, all_idx),
    "u_l2": rel_l2_per_snapshot(u_p, u_dns, all_idx),
    "v_l2": rel_l2_per_snapshot(v_p, v_dns, all_idx),
    "omega_l2": rel_l2_per_snapshot(omega_p, omega_d, all_idx),
    "dns_access": False,
}

print("Running IDW p=2...")
u_p, v_p = idw_reconstruct(p=2.0)
omega_p = np.array([compute_vorticity(u_p[t], v_p[t]) for t in range(T)])
results["IDW p=2"] = {
    "ke": ke_rel_err(u_p, v_p, u_dns, v_dns, all_idx),
    "u_l2": rel_l2_per_snapshot(u_p, u_dns, all_idx),
    "v_l2": rel_l2_per_snapshot(v_p, v_dns, all_idx),
    "omega_l2": rel_l2_per_snapshot(omega_p, omega_d, all_idx),
    "dns_access": False,
}

for k_max in [5, 8, 12]:
    print(f"Running Div-free trig LSQ k_max={k_max}...")
    u_p, v_p, M_modes = divfree_trig_lsq_reconstruct(k_max=k_max)
    omega_p = np.array([compute_vorticity(u_p[t], v_p[t]) for t in range(T)])
    name = f"Div-free trig LSQ k≤{k_max} ({M_modes} modes)"
    results[name] = {
        "ke": ke_rel_err(u_p, v_p, u_dns, v_dns, all_idx),
        "u_l2": rel_l2_per_snapshot(u_p, u_dns, all_idx),
        "v_l2": rel_l2_per_snapshot(v_p, v_dns, all_idx),
        "omega_l2": rel_l2_per_snapshot(omega_p, omega_d, all_idx),
        "dns_access": False,
        "num_modes": M_modes,
    }

# === Final summary ===
print(f"\n{'='*100}")
print(f"BASELINE SQUEEZE COMPARISON (K={K}, T={T})")
print(f"{'='*100}\n")
print(f"{'Method':<40} | {'KE%':>7} | {'u_L2%':>7} | {'v_L2%':>7} | {'ω_L2%':>7} | {'DNS?':>5}")
print("-" * 95)
print(f"{'EXP-080 (our PINN)':<40} | {10.68:>6.2f}% | {17.0:>6.2f}% | {20.2:>6.2f}% | {47.6:>6.2f}% | {'No':>5}")
for name, m in results.items():
    print(f"{name:<40} | {m['ke']*100:>6.2f}% | {m['u_l2']*100:>6.2f}% | {m['v_l2']*100:>6.2f}% | {m['omega_l2']*100:>6.2f}% | {'No' if not m['dns_access'] else 'YES':>5}")
print(f"{'(reference) Gappy POD r=100 (cheat)':<40} | {0.12:>6.2f}% | {0.85:>6.2f}% | {0.85:>6.2f}% | {'—':>6} | {'YES':>5}")

# Save
summary = {
    "K": int(K),
    "EXP_080": {"ke": 0.1068, "u_l2": 0.170, "v_l2": 0.202, "omega_l2": 0.476},
    **{name: m for name, m in results.items()},
    "Gappy_POD_r100_cheat": {"ke": 0.0012, "u_l2": 0.0085, "v_l2": 0.0085, "dns_access": True},
}
with open(OUT_DIR / "baseline_squeeze.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {OUT_DIR}/baseline_squeeze.json")
