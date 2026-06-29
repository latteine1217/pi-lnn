"""k≤8 dedup Trig LSQ ridge sweep.

Tests whether ANY Tikhonov ridge strength stabilizes the k≤8 case
(s_min = 0.022 → need λ ≳ s_min² ≈ 5e-4 to start damping, λ ≳ 0.022 for full damp).
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, "scripts")
# Import side-effects load DNS + sensors etc.  Skip the script's main loops by stopping import early
# — instead inline the loader.

SENSOR_JSON = Path("data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100.json")
DNS_PATH = Path("data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy")

dns = np.load(DNS_PATH, allow_pickle=True).item()
u_dns = dns["u"]; v_dns = dns["v"]
T, N, _ = u_dns.shape
with open(SENSOR_JSON) as f:
    sensor_data = json.load(f)
K = sensor_data["K"]
sensor_pos = np.array(sensor_data["selected_coordinates"])

x_g = np.linspace(0, 1, N, endpoint=False)
y_g = np.linspace(0, 1, N, endpoint=False)
XG, YG = np.meshgrid(x_g, y_g, indexing="ij")


def sample_at_sensors(field):
    samples = np.zeros(K)
    for k, (xk, yk) in enumerate(sensor_pos):
        ix = int(np.round((xk * N) % N)); iy = int(np.round((yk * N) % N))
        samples[k] = field[ix, iy]
    return samples


def ke_rel_err(u_p, v_p, u_t, v_t):
    ke_p = 0.5 * (u_p**2 + v_p**2).mean(axis=(1, 2))
    ke_t = 0.5 * (u_t**2 + v_t**2).mean(axis=(1, 2))
    return float(np.abs((ke_p - ke_t) / ke_t).mean())


def rel_l2_per_snapshot(pred, true):
    diff_l2 = np.sqrt(((pred - true) ** 2).mean(axis=(1, 2)))
    true_l2 = np.sqrt((true ** 2).mean(axis=(1, 2)))
    return float((diff_l2 / true_l2).mean())


def dedup_trig_lsq(k_max=8, ridge_alpha=0.0):
    mode_indices = []
    for kx in range(-k_max, k_max + 1):
        for ky in range(-k_max, k_max + 1):
            if kx == 0 and ky == 0:
                continue
            if kx ** 2 + ky ** 2 > k_max ** 2:
                continue
            if kx > 0 or (kx == 0 and ky > 0):
                mode_indices.append((kx, ky))
    M_dof = 2 * len(mode_indices)
    modes = np.asarray(mode_indices, dtype=np.float64)
    kx_arr = modes[:, 0]; ky_arr = modes[:, 1]
    two_pi = 2.0 * np.pi
    phase_s = two_pi * (sensor_pos @ modes.T)
    cos_s = np.cos(phase_s); sin_s = np.sin(phase_s)
    A = np.zeros((2 * K, M_dof))
    A[0::2, 0::2] = -ky_arr[None, :] * sin_s
    A[0::2, 1::2] =  ky_arr[None, :] * cos_s
    A[1::2, 0::2] =  kx_arr[None, :] * sin_s
    A[1::2, 1::2] = -kx_arr[None, :] * cos_s
    s = np.linalg.svd(A, compute_uv=False)
    cond = s[0] / max(s[-1], 1e-30)
    grid_xy = np.stack([XG.flatten(), YG.flatten()], axis=-1)
    phase_g = two_pi * (grid_xy @ modes.T)
    cos_g = np.cos(phase_g); sin_g = np.sin(phase_g)
    u_basis = np.zeros((N * N, M_dof)); v_basis = np.zeros((N * N, M_dof))
    u_basis[:, 0::2] = -ky_arr[None, :] * sin_g
    u_basis[:, 1::2] =  ky_arr[None, :] * cos_g
    v_basis[:, 0::2] =  kx_arr[None, :] * sin_g
    v_basis[:, 1::2] = -kx_arr[None, :] * cos_g
    u_out = np.zeros_like(u_dns); v_out = np.zeros_like(v_dns)
    AtA = A.T @ A + ridge_alpha * np.eye(M_dof)
    L_chol = np.linalg.cholesky(AtA)
    for t in range(T):
        u_vals = sample_at_sensors(u_dns[t])
        v_vals = sample_at_sensors(v_dns[t])
        b = np.empty(2 * K); b[0::2] = u_vals; b[1::2] = v_vals
        rhs = A.T @ b
        y = np.linalg.solve(L_chol, rhs)
        coef = np.linalg.solve(L_chol.T, y)
        u_out[t] = (u_basis @ coef).reshape(N, N)
        v_out[t] = (v_basis @ coef).reshape(N, N)
    return u_out, v_out, cond, s[-1], M_dof


# Sweep
print(f"k≤8 dedup ridge sweep  (K={K}, T={T})", flush=True)
print(f"{'ridge λ':>10} | {'KE%':>9} | {'u_L2%':>9} | {'v_L2%':>9} | {'cond':>10} | {'s_min':>10}", flush=True)
print("-" * 76, flush=True)
for ridge in [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
    u_p, v_p, cond, s_min, M_dof = dedup_trig_lsq(k_max=8, ridge_alpha=ridge)
    ke = ke_rel_err(u_p, v_p, u_dns, v_dns)
    ul = rel_l2_per_snapshot(u_p, u_dns)
    vl = rel_l2_per_snapshot(v_p, v_dns)
    label = f"{ridge:.0e}" if ridge > 0 else "0      "
    print(f"{label:>10} | {ke*100:>8.2f}% | {ul*100:>8.2f}% | {vl*100:>8.2f}% | {cond:>10.2e} | {s_min:>10.2e}", flush=True)
