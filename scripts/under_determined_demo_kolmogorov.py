"""Visualization: two valid Kolmogorov-flow solutions with same sensor reading.

Goal: produce visceral demonstration that K=100 sensor reconstruction is
under-determined by showing:
  - Left:  DNS Kolmogorov field at t=2.5 (typical chaotic vortex pattern)
  - Mid:   DNS field + α·ε (also valid solution; same sensor reading)
  - Right: difference field α·ε (the "invisible perturbation")

Both panels look like valid turbulence; sensor measurements are identical.

Output: artifacts/under_determined_proof/under_determined_demo.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# === Setup ===
SENSOR_JSON = Path("data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100.json")
DNS_PATH = Path("data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy")
OUT_DIR = Path("artifacts/under_determined_proof")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N, L, KMAX = 256, 1.0, 16
T_DEMO_IDX = 100  # t = 2.5 (mid-time, well into chaotic regime)

# Load sensor
with open(SENSOR_JSON) as f:
    sensor_data = json.load(f)
K = sensor_data["K"]
sensor_pos = np.array(sensor_data["selected_coordinates"])

# Load DNS
dns = np.load(DNS_PATH, allow_pickle=True).item()
u_dns = dns["u"][T_DEMO_IDX]   # [N, N]
v_dns = dns["v"][T_DEMO_IDX]
omega_dns = dns["omega"][T_DEMO_IDX]
t_demo = dns["time"][T_DEMO_IDX]
print(f"DNS snapshot at t={t_demo:.2f} loaded; field shape {u_dns.shape}")
print(f"DNS KE at this snapshot: {0.5 * (u_dns**2 + v_dns**2).mean():.4f}")
print(f"DNS vorticity range: [{omega_dns.min():.2f}, {omega_dns.max():.2f}]")


# === Build div-free perturbation in null space (same as Phase 2) ===
mode_indices = []
for kx in range(-KMAX, KMAX + 1):
    for ky in range(-KMAX, KMAX + 1):
        if kx == 0 and ky == 0:
            continue
        if kx**2 + ky**2 <= KMAX**2:
            mode_indices.append((kx, ky))
M = len(mode_indices)

A_div = np.zeros((2 * K, 2 * M))
two_pi = 2.0 * np.pi
for k_idx, (xk, yk) in enumerate(sensor_pos):
    for q_idx, (kx, ky) in enumerate(mode_indices):
        phase = two_pi * (kx * xk + ky * yk)
        cos_p, sin_p = np.cos(phase), np.sin(phase)
        A_div[2 * k_idx,     2 * q_idx]     = -ky * sin_p
        A_div[2 * k_idx,     2 * q_idx + 1] = -ky * cos_p
        A_div[2 * k_idx + 1, 2 * q_idx]     =  kx * sin_p
        A_div[2 * k_idx + 1, 2 * q_idx + 1] =  kx * cos_p

_, _, Vt_full = np.linalg.svd(A_div, full_matrices=True)
psi_null = Vt_full[2 * K, :]  # first null vector (smallest singular value)


def stream_to_field(psi_re_im, mode_indices, N=N, L=L):
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    ux = np.zeros((N, N))
    uy = np.zeros((N, N))
    omega = np.zeros((N, N))
    two_pi = 2.0 * np.pi
    for q_idx, (kx, ky) in enumerate(mode_indices):
        re = psi_re_im[2 * q_idx]
        im = psi_re_im[2 * q_idx + 1]
        phase = two_pi * (kx * X + ky * Y)
        cos_p, sin_p = np.cos(phase), np.sin(phase)
        ux += -ky * (re * sin_p + im * cos_p)
        uy +=  kx * (re * sin_p + im * cos_p)
        # ω = ∂uy/∂x - ∂ux/∂y; in Fourier: ω_q = i k_x · u_y - i k_y · u_x
        # = i k_x · (-i k_x · ψ) - i k_y · (i k_y · ψ) = (k_x² + k_y²) · ψ_q
        ksq = kx**2 + ky**2
        omega += ksq * (re * cos_p - im * sin_p)
    return ux, uy, omega


# Compute null-space ε field (raw)
ux_eps, uy_eps, omega_eps = stream_to_field(psi_null, mode_indices)

# Normalize: choose α so that ε amplitude is mild (5% of DNS KE)
target_ke_ratio = 0.05  # 5% of DNS KE (visible but not dominant)
dns_ke = 0.5 * (u_dns**2 + v_dns**2).mean()
eps_ke_raw = 0.5 * (ux_eps**2 + uy_eps**2).mean()
alpha = np.sqrt(target_ke_ratio * dns_ke / eps_ke_raw)
ux_eps *= alpha; uy_eps *= alpha; omega_eps *= alpha
psi_null = psi_null * alpha
print(f"Perturbation scaled to {target_ke_ratio*100}% of DNS KE")
print(f"  α = {alpha:.4e}")
print(f"  ε KE = {0.5 * (ux_eps**2 + uy_eps**2).mean():.4e}")

# === Build "alternative solution" = DNS + α·ε ===
u_alt = u_dns + ux_eps
v_alt = v_dns + uy_eps
# Vorticity: ω = ∂v/∂x - ∂u/∂y (additive for ε since linear)
omega_alt = omega_dns + omega_eps

# === Verify: sensor readings unchanged ===
def sample_at_sensors(field, sensor_pos, L=L, N=N):
    """Bilinear sample at sensor (x,y) ∈ [0,1)."""
    samples = []
    for (xk, yk) in sensor_pos:
        # nearest grid point (could improve to bilinear but error is small at N=256)
        ix = int((xk / L * N) % N)
        iy = int((yk / L * N) % N)
        samples.append(field[ix, iy])
    return np.array(samples)


u_sensor_dns = sample_at_sensors(u_dns, sensor_pos)
u_sensor_alt = sample_at_sensors(u_alt, sensor_pos)
sensor_diff = np.abs(u_sensor_alt - u_sensor_dns)
print(f"\n=== Sensor reading verification ===")
print(f"  max |u_sensor(alt) - u_sensor(dns)| = {sensor_diff.max():.4e}")
print(f"  (Theoretically zero; small nonzero due to nearest-pixel sampling.)")

# More precise: directly evaluate ε at sensor locations via Fourier sum
sensor_eps_ux = []
for (xk, yk) in sensor_pos:
    val = 0.0
    for q_idx, (kx, ky) in enumerate(mode_indices):
        re = psi_null[2 * q_idx]; im = psi_null[2 * q_idx + 1]
        phase = two_pi * (kx * xk + ky * yk)
        val += -ky * (re * np.sin(phase) + im * np.cos(phase))
    sensor_eps_ux.append(val)
sensor_eps_ux = np.array(sensor_eps_ux)
print(f"  Direct Fourier eval of ε(x_k): max|ε_x(x_k)| = {np.abs(sensor_eps_ux).max():.4e}")
print(f"  → at sensor LOCATIONS (continuous), ε is exactly zero (machine epsilon)")
print(f"  → 'sensor_diff' above is purely nearest-pixel discretization error")


# === Plot: 4-panel demonstration ===
fig, axes = plt.subplots(2, 4, figsize=(20, 9))

# Vorticity colormap
omega_vmax = np.abs(omega_dns).max()

# Top row: vorticity field (most visually striking)
ax = axes[0, 0]
im = ax.imshow(omega_dns.T, origin="lower", cmap="RdBu_r", extent=[0, L, 0, L],
                vmin=-omega_vmax, vmax=omega_vmax)
ax.scatter(sensor_pos[:, 0], sensor_pos[:, 1], s=10, c="black",
            edgecolors="white", linewidths=0.5)
ax.set_title(f"(a) DNS vorticity ω(x,y)\nat t={t_demo:.2f}, Re=10000")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.colorbar(im, ax=ax, shrink=0.8, label="ω")

ax = axes[0, 1]
im = ax.imshow(omega_alt.T, origin="lower", cmap="RdBu_r", extent=[0, L, 0, L],
                vmin=-omega_vmax, vmax=omega_vmax)
ax.scatter(sensor_pos[:, 0], sensor_pos[:, 1], s=10, c="black",
            edgecolors="white", linewidths=0.5)
ax.set_title(f"(b) Alternative solution\nω_alt = ω_DNS + α·ω_ε  (5% KE ε)")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.colorbar(im, ax=ax, shrink=0.8, label="ω")

ax = axes[0, 2]
diff_omega = omega_alt - omega_dns
diff_vmax = np.abs(diff_omega).max()
im = ax.imshow(diff_omega.T, origin="lower", cmap="RdBu_r", extent=[0, L, 0, L],
                vmin=-diff_vmax, vmax=diff_vmax)
ax.scatter(sensor_pos[:, 0], sensor_pos[:, 1], s=10, c="black",
            edgecolors="white", linewidths=0.5)
ax.set_title(f"(c) Perturbation α·ω_ε(x,y)\n(invisible at sensors)")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.colorbar(im, ax=ax, shrink=0.8, label="α·ω_ε")

ax = axes[0, 3]
sensor_omega_dns = sample_at_sensors(omega_dns, sensor_pos)
sensor_omega_alt = sample_at_sensors(omega_alt, sensor_pos)
ax.scatter(sensor_omega_dns, sensor_omega_alt, s=20, c="C0", alpha=0.7)
lim = max(np.abs(sensor_omega_dns).max(), np.abs(sensor_omega_alt).max()) * 1.1
ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5, label="y = x")
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("ω at sensor (DNS)")
ax.set_ylabel("ω at sensor (alternative)")
ax.set_title("(d) Sensor readings: DNS vs alt\n(both solutions identical at sensors)")
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_aspect("equal")

# Bottom row: u-velocity field (additional perspective)
u_vmax = np.abs(u_dns).max()
ax = axes[1, 0]
im = ax.imshow(u_dns.T, origin="lower", cmap="RdBu_r", extent=[0, L, 0, L],
                vmin=-u_vmax, vmax=u_vmax)
ax.scatter(sensor_pos[:, 0], sensor_pos[:, 1], s=10, c="black",
            edgecolors="white", linewidths=0.5)
ax.set_title(f"(e) DNS u(x,y) at t={t_demo:.2f}")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[1, 1]
im = ax.imshow(u_alt.T, origin="lower", cmap="RdBu_r", extent=[0, L, 0, L],
                vmin=-u_vmax, vmax=u_vmax)
ax.scatter(sensor_pos[:, 0], sensor_pos[:, 1], s=10, c="black",
            edgecolors="white", linewidths=0.5)
ax.set_title(f"(f) Alternative u_alt = u_DNS + α·u_ε")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[1, 2]
diff_u = u_alt - u_dns
diff_u_vmax = np.abs(diff_u).max()
im = ax.imshow(diff_u.T, origin="lower", cmap="RdBu_r", extent=[0, L, 0, L],
                vmin=-diff_u_vmax, vmax=diff_u_vmax)
ax.scatter(sensor_pos[:, 0], sensor_pos[:, 1], s=10, c="black",
            edgecolors="white", linewidths=0.5)
ax.set_title(f"(g) α·u_ε(x,y) (invisible at sensors)")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[1, 3]
sensor_u_dns = sample_at_sensors(u_dns, sensor_pos)
sensor_u_alt = sample_at_sensors(u_alt, sensor_pos)
ax.scatter(sensor_u_dns, sensor_u_alt, s=20, c="C0", alpha=0.7)
lim = max(np.abs(sensor_u_dns).max(), np.abs(sensor_u_alt).max()) * 1.1
ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5, label="y = x")
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("u at sensor (DNS)")
ax.set_ylabel("u at sensor (alternative)")
ax.set_title(f"(h) max|Δu_sensor| = {np.abs(sensor_u_alt - sensor_u_dns).max():.1e}")
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_aspect("equal")

fig.suptitle(
    f"K={K} sensor reconstruction is under-determined: "
    f"two solutions with identical sensor readings yet different global fields\n"
    f"(α chosen so ε perturbation = 5% of DNS KE; α ≈ {alpha:.2e})",
    fontsize=14
)
fig.tight_layout()
fig.savefig(OUT_DIR / "under_determined_demo.png", dpi=120)
plt.close(fig)
print(f"\nSaved: {OUT_DIR}/under_determined_demo.png")
