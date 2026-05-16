"""Baseline comparison: how does our PINN compare to classical sparse-sensor reconstruction?

Three baselines:
  1. RBF interpolation        — fair (no DNS access for training)
  2. Linear (Delaunay)        — fair (no DNS access)
  3. Gappy POD                — UPPER BOUND (uses DNS basis, "cheat")

Each method reconstructs (u, v) at full N×N grid from K=100 sensor values
per snapshot. Compute KE rel-err averaged over time.

Compare with EXP-080 (KE 10.68%).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator, LinearNDInterpolator

SENSOR_JSON = Path("data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100.json")
DNS_PATH = Path("data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy")
OUT_DIR = Path("artifacts/under_determined_proof")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === Load DNS ===
dns = np.load(DNS_PATH, allow_pickle=True).item()
u_dns = dns["u"]   # [T=201, N=256, N=256]
v_dns = dns["v"]
T, N, _ = u_dns.shape
print(f"DNS shape: {u_dns.shape}, T={T} snapshots")

# === Load sensor positions ===
with open(SENSOR_JSON) as f:
    sensor_data = json.load(f)
K = sensor_data["K"]
sensor_pos = np.array(sensor_data["selected_coordinates"])  # [K, 2] in [0,1)
print(f"K={K} sensors")

# Convert sensor pos to grid indices for sampling
def sample_field_at_sensors(field, sensor_pos, N=N, L=1.0):
    """Bilinear sample at sensor positions."""
    samples = np.zeros(K)
    for k, (xk, yk) in enumerate(sensor_pos):
        ix = int(np.round((xk / L * N) % N))
        iy = int(np.round((yk / L * N) % N))
        samples[k] = field[ix, iy]
    return samples

# Grid coordinates
x = np.linspace(0, 1, N, endpoint=False)
y = np.linspace(0, 1, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")
grid_xy = np.stack([X.flatten(), Y.flatten()], axis=-1)  # [N², 2]

# Train/val split (consistent with model evaluation: 160 train + 41 val)
val_mask = np.zeros(T, dtype=bool)
val_indices = np.linspace(0, T-1, 41, dtype=int)  # approximate val set
val_mask[val_indices] = True
train_indices = np.where(~val_mask)[0]
print(f"Train snapshots: {len(train_indices)}, Val: {val_mask.sum()}")


def compute_ke_rel_err(u_pred, v_pred, u_true, v_true, indices):
    """Per-snapshot KE rel-err averaged over indices."""
    ke_pred = 0.5 * (u_pred[indices]**2 + v_pred[indices]**2).mean(axis=(1, 2))
    ke_true = 0.5 * (u_true[indices]**2 + v_true[indices]**2).mean(axis=(1, 2))
    return float(np.abs((ke_pred - ke_true) / ke_true).mean())


def compute_pointwise_l2(pred, true, indices):
    """Per-snapshot L2 error / true L2 norm."""
    diff_l2 = np.sqrt(((pred[indices] - true[indices])**2).mean(axis=(1, 2)))
    true_l2 = np.sqrt((true[indices]**2).mean(axis=(1, 2)))
    return float((diff_l2 / true_l2).mean())


# === Baseline 1: RBF interpolation (Gaussian kernel) ===
print(f"\n=== Baseline 1: RBF (Gaussian) interpolation ===")
print(f"  No DNS access, single-snapshot reconstruction")

u_rbf = np.zeros_like(u_dns)
v_rbf = np.zeros_like(v_dns)

# Periodic domain: extend sensor positions with periodic copies for boundary
def periodic_sensor_extend(sensor_pos, vals, L=1.0):
    """Extend sensor positions to cover periodic boundaries."""
    sp = sensor_pos.copy()
    vv = vals.copy()
    extends = []
    extends_v = []
    for dx in [-L, 0, L]:
        for dy in [-L, 0, L]:
            if dx == 0 and dy == 0:
                continue
            mask = (sp[:, 0] + dx >= -0.3) & (sp[:, 0] + dx <= 1.3) & \
                   (sp[:, 1] + dy >= -0.3) & (sp[:, 1] + dy <= 1.3)
            extends.append(sp[mask] + np.array([dx, dy]))
            extends_v.append(vv[mask])
    return np.vstack([sp] + extends), np.concatenate([vv] + extends_v)


for t_idx in range(T):
    if t_idx % 50 == 0:
        print(f"  RBF: snapshot {t_idx}/{T}")
    u_vals = sample_field_at_sensors(u_dns[t_idx], sensor_pos)
    v_vals = sample_field_at_sensors(v_dns[t_idx], sensor_pos)

    sp_ext, u_ext = periodic_sensor_extend(sensor_pos, u_vals)
    _, v_ext = periodic_sensor_extend(sensor_pos, v_vals)

    rbf_u = RBFInterpolator(sp_ext, u_ext, kernel="gaussian", epsilon=10.0,
                             smoothing=0.0, neighbors=50)
    rbf_v = RBFInterpolator(sp_ext, v_ext, kernel="gaussian", epsilon=10.0,
                             smoothing=0.0, neighbors=50)
    u_rbf[t_idx] = rbf_u(grid_xy).reshape(N, N)
    v_rbf[t_idx] = rbf_v(grid_xy).reshape(N, N)

ke_rbf_all = compute_ke_rel_err(u_rbf, v_rbf, u_dns, v_dns, np.arange(T))
ke_rbf_train = compute_ke_rel_err(u_rbf, v_rbf, u_dns, v_dns, train_indices)
ke_rbf_val = compute_ke_rel_err(u_rbf, v_rbf, u_dns, v_dns, val_indices)
u_l2_rbf = compute_pointwise_l2(u_rbf, u_dns, np.arange(T))
v_l2_rbf = compute_pointwise_l2(v_rbf, v_dns, np.arange(T))
print(f"  RBF KE rel-err: all={ke_rbf_all:.4%}, train={ke_rbf_train:.4%}, val={ke_rbf_val:.4%}")
print(f"  RBF u rel-L2: {u_l2_rbf:.4%},  v rel-L2: {v_l2_rbf:.4%}")


# === Baseline 2: Gappy POD (UPPER BOUND: uses DNS basis) ===
print(f"\n=== Baseline 2: Gappy POD (DNS-trained basis, NOT engineering-transferable) ===")
# Combine u, v: stack as [T, 2*N²]
field_dns = np.concatenate([
    u_dns.reshape(T, N*N),
    v_dns.reshape(T, N*N)
], axis=1)  # [T, 2*N²]

# Subtract time-mean
field_mean = field_dns.mean(axis=0)
field_centered = field_dns - field_mean[None, :]

# SVD: train POD basis from TRAIN snapshots only (still cheating but slightly fair-er)
# Reduced SVD: only first r modes
for r in [50, 100, 150]:
    U_pod, S_pod, Vt_pod = np.linalg.svd(field_centered[train_indices], full_matrices=False)
    # POD basis: first r columns of Vt (right singular vectors)
    pod_basis = Vt_pod[:r]  # [r, 2*N²]

    # Reconstruct each snapshot from K sensor measurements (Gappy POD)
    # Sensor sampling: extracts u, v at K sensor positions = 2K measurements
    sensor_indices_u = []
    sensor_indices_v = []
    for (xk, yk) in sensor_pos:
        ix = int(np.round((xk * N) % N))
        iy = int(np.round((yk * N) % N))
        flat_idx = ix * N + iy
        sensor_indices_u.append(flat_idx)
        sensor_indices_v.append(N*N + flat_idx)
    sensor_indices = np.array(sensor_indices_u + sensor_indices_v)

    # Sensor sampling matrix S: [2K, 2N²]
    # For each snapshot, sensor reading = field_centered[t][sensor_indices]
    # Solve: pod_basis[:, sensor_indices] @ a_t = sensor_vals[t] - field_mean[sensor_indices]
    # → a_t = least-squares solution

    A_sample = pod_basis[:, sensor_indices].T  # [2K, r]

    u_pod = np.zeros_like(u_dns)
    v_pod = np.zeros_like(v_dns)
    for t_idx in range(T):
        sensor_vals = field_dns[t_idx, sensor_indices] - field_mean[sensor_indices]
        # Solve A_sample @ a = sensor_vals
        a, *_ = np.linalg.lstsq(A_sample, sensor_vals, rcond=None)
        # Reconstruct full field
        recon = pod_basis.T @ a + field_mean
        u_pod[t_idx] = recon[:N*N].reshape(N, N)
        v_pod[t_idx] = recon[N*N:].reshape(N, N)

    ke_pod = compute_ke_rel_err(u_pod, v_pod, u_dns, v_dns, np.arange(T))
    ke_pod_val = compute_ke_rel_err(u_pod, v_pod, u_dns, v_dns, val_indices)
    u_l2_pod = compute_pointwise_l2(u_pod, u_dns, np.arange(T))
    print(f"  Gappy POD r={r}: KE rel-err all={ke_pod:.4%} val={ke_pod_val:.4%}, u rel-L2={u_l2_pod:.4%}")


# === Final comparison ===
print(f"\n=== FINAL COMPARISON ===")
print(f"{'Method':<35} | {'KE rel-err':>12} | {'u rel-L2':>10} | {'Uses DNS?':>10}")
print("-" * 80)
print(f"{'EXP-080 (our PINN)':<35} | {'10.68%':>12} | {'17.0%':>10} | {'No':>10}")
print(f"{'Baseline 1: RBF interp':<35} | {ke_rbf_all*100:>11.2f}% | {u_l2_rbf*100:>9.2f}% | {'No':>10}")
print(f"{'Baseline 2: Gappy POD r=100 (cheat)':<35} | (see above) | (see above) | {'YES':>10}")
print(f"{'Spectral truncation k=4 floor':<35} | {'7.77%':>12} | {'-':>10} | {'-':>10}")
print(f"{'Spectral truncation k=6 floor':<35} | {'2.62%':>12} | {'-':>10} | {'-':>10}")

# Save summary
summary = {
    "K": int(K),
    "EXP_080": {"ke_rel_err": 0.1068, "u_rel_l2": 0.170, "v_rel_l2": 0.202},
    "rbf_gaussian": {
        "ke_rel_err_all": ke_rbf_all,
        "ke_rel_err_train": ke_rbf_train,
        "ke_rel_err_val": ke_rbf_val,
        "u_rel_l2": u_l2_rbf,
        "v_rel_l2": v_l2_rbf,
        "uses_dns": False,
    },
    "spectral_truncation_floor": {"k4": 0.0777, "k6": 0.0262, "k8": 0.0105},
}
with open(OUT_DIR / "baseline_comparison.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved summary: {OUT_DIR}/baseline_comparison.json")
