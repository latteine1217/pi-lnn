"""Full baseline comparison including vorticity (where RBF should fail).

Adds vorticity & ek_ratio analysis to baseline_comparison.py results.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator

SENSOR_JSON = Path("data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100.json")
DNS_PATH = Path("data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy")
OUT_DIR = Path("artifacts/under_determined_proof")

# === Load DNS ===
dns = np.load(DNS_PATH, allow_pickle=True).item()
u_dns = dns["u"]
v_dns = dns["v"]
omega_dns = dns["omega"]  # already-computed DNS vorticity
T, N, _ = u_dns.shape
L = 1.0
dx = L / N

with open(SENSOR_JSON) as f:
    sensor_data = json.load(f)
K = sensor_data["K"]
sensor_pos = np.array(sensor_data["selected_coordinates"])

# Sample at sensors
def sample_at_sensors(field, sensor_pos):
    samples = np.zeros(K)
    for k, (xk, yk) in enumerate(sensor_pos):
        ix = int(np.round((xk * N) % N))
        iy = int(np.round((yk * N) % N))
        samples[k] = field[ix, iy]
    return samples

# Periodic extension
def periodic_sensor_extend(sensor_pos, vals, L=1.0):
    sp = sensor_pos.copy()
    vv = vals.copy()
    extends_pos, extends_v = [], []
    for dx_off in [-L, 0, L]:
        for dy_off in [-L, 0, L]:
            if dx_off == 0 and dy_off == 0:
                continue
            mask = (sp[:, 0] + dx_off >= -0.3) & (sp[:, 0] + dx_off <= 1.3) & \
                   (sp[:, 1] + dy_off >= -0.3) & (sp[:, 1] + dy_off <= 1.3)
            extends_pos.append(sp[mask] + np.array([dx_off, dy_off]))
            extends_v.append(vv[mask])
    return np.vstack([sp] + extends_pos), np.concatenate([vv] + extends_v)


# Grid for reconstruction
x = np.linspace(0, 1, N, endpoint=False)
y = np.linspace(0, 1, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")
grid_xy = np.stack([X.flatten(), Y.flatten()], axis=-1)


def compute_vorticity(u, v, dx=dx):
    """ω = ∂v/∂x - ∂u/∂y; periodic via np.gradient (or manual finite diff)."""
    # Use centered differences with periodic boundary (np.roll for periodic)
    dvdx = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dx)
    dudy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
    return dvdx - dudy


def energy_spectrum_1d(u, v, dx=dx):
    """1D radial spectrum from 2D field via FFT."""
    N = u.shape[0]
    u_hat = np.fft.fft2(u) / N**2
    v_hat = np.fft.fft2(v) / N**2
    e_hat = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)  # per-mode energy
    # Radial binning
    kx = np.fft.fftfreq(N, d=dx) * (2 * np.pi)
    ky = np.fft.fftfreq(N, d=dx) * (2 * np.pi)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    Kmag = np.sqrt(KX**2 + KY**2) / (2 * np.pi)  # in cycles/L unit
    Kmag_int = Kmag.astype(int)
    k_max = N // 2
    spectrum = np.zeros(k_max)
    for k in range(1, k_max + 1):
        mask = (Kmag_int == k)
        spectrum[k - 1] = e_hat[mask].sum()
    return np.arange(1, k_max + 1), spectrum


# Train/val
val_indices = np.linspace(0, T-1, 41, dtype=int)
val_mask = np.zeros(T, dtype=bool); val_mask[val_indices] = True


# === Compute RBF reconstruction ===
print("Computing RBF reconstruction...")
u_rbf = np.zeros_like(u_dns)
v_rbf = np.zeros_like(v_dns)
for t_idx in range(T):
    if t_idx % 50 == 0:
        print(f"  {t_idx}/{T}")
    u_vals = sample_at_sensors(u_dns[t_idx], sensor_pos)
    v_vals = sample_at_sensors(v_dns[t_idx], sensor_pos)
    sp_ext, u_ext = periodic_sensor_extend(sensor_pos, u_vals)
    _, v_ext = periodic_sensor_extend(sensor_pos, v_vals)
    rbf_u = RBFInterpolator(sp_ext, u_ext, kernel="gaussian", epsilon=10.0,
                             smoothing=0.0, neighbors=50)
    rbf_v = RBFInterpolator(sp_ext, v_ext, kernel="gaussian", epsilon=10.0,
                             smoothing=0.0, neighbors=50)
    u_rbf[t_idx] = rbf_u(grid_xy).reshape(N, N)
    v_rbf[t_idx] = rbf_v(grid_xy).reshape(N, N)

# Vorticity for RBF
omega_rbf = np.array([compute_vorticity(u_rbf[t], v_rbf[t]) for t in range(T)])
omega_dns_recomp = np.array([compute_vorticity(u_dns[t], v_dns[t]) for t in range(T)])  # use same FD scheme

# Compute metrics
def relative_l2_per_snapshot(pred, true, indices):
    diff_l2 = np.sqrt(((pred[indices] - true[indices])**2).mean(axis=(1, 2)))
    true_l2 = np.sqrt((true[indices]**2).mean(axis=(1, 2)))
    return float((diff_l2 / true_l2).mean())


def ke_rel_err(u_p, v_p, u_t, v_t, indices):
    ke_p = 0.5 * (u_p[indices]**2 + v_p[indices]**2).mean(axis=(1, 2))
    ke_t = 0.5 * (u_t[indices]**2 + v_t[indices]**2).mean(axis=(1, 2))
    return float(np.abs((ke_p - ke_t) / ke_t).mean())


# Energy spectrum at last snapshot for ek_ratio
k_dns, ek_dns_t5 = energy_spectrum_1d(u_dns[-1], v_dns[-1])
k_rbf, ek_rbf_t5 = energy_spectrum_1d(u_rbf[-1], v_rbf[-1])
ek_ratio_rbf = float(ek_rbf_t5[1] / ek_dns_t5[1])  # k_f = 2 (index 1)

# Print full comparison
all_idx = np.arange(T)
print(f"\n=== FULL BASELINE COMPARISON (K={K}) ===\n")
print(f"{'Metric':<25} | {'EXP-080':>10} | {'RBF':>10} | {'Δ (RBF-ours)':>15}")
print("-" * 75)

# Our model results (from EXP-080 evaluation summary)
our_ke = 0.1068
our_u_l2 = 0.170
our_v_l2 = 0.202
our_omega_l2 = 0.476
our_ek_ratio = 0.911

ke_rbf = ke_rel_err(u_rbf, v_rbf, u_dns, v_dns, all_idx)
u_l2_rbf = relative_l2_per_snapshot(u_rbf, u_dns, all_idx)
v_l2_rbf = relative_l2_per_snapshot(v_rbf, v_dns, all_idx)
omega_l2_rbf = relative_l2_per_snapshot(omega_rbf, omega_dns_recomp, all_idx)

print(f"{'KE rel-err':<25} | {our_ke*100:>9.2f}% | {ke_rbf*100:>9.2f}% | {(ke_rbf-our_ke)*100:>+14.2f}pp")
print(f"{'u rel-L2':<25} | {our_u_l2*100:>9.2f}% | {u_l2_rbf*100:>9.2f}% | {(u_l2_rbf-our_u_l2)*100:>+14.2f}pp")
print(f"{'v rel-L2':<25} | {our_v_l2*100:>9.2f}% | {v_l2_rbf*100:>9.2f}% | {(v_l2_rbf-our_v_l2)*100:>+14.2f}pp")
print(f"{'omega rel-L2':<25} | {our_omega_l2*100:>9.2f}% | {omega_l2_rbf*100:>9.2f}% | {(omega_l2_rbf-our_omega_l2)*100:>+14.2f}pp")
print(f"{'ek_ratio @ k_f=2':<25} | {our_ek_ratio:>10.4f} | {ek_ratio_rbf:>10.4f} | {(ek_ratio_rbf-our_ek_ratio):>+15.4f}")

print(f"\n→ Negative Δ means RBF is BETTER on that metric.")
print(f"→ Positive Δ means RBF is WORSE (we win).")

summary = {
    "K": int(K),
    "EXP_080": {
        "ke_rel_err": our_ke,
        "u_rel_l2": our_u_l2,
        "v_rel_l2": our_v_l2,
        "omega_rel_l2": our_omega_l2,
        "ek_ratio_kf": our_ek_ratio,
    },
    "RBF_gaussian": {
        "ke_rel_err": ke_rbf,
        "u_rel_l2": u_l2_rbf,
        "v_rel_l2": v_l2_rbf,
        "omega_rel_l2": omega_l2_rbf,
        "ek_ratio_kf": ek_ratio_rbf,
    },
}
with open(OUT_DIR / "baseline_comparison_full.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved: {OUT_DIR}/baseline_comparison_full.json")
