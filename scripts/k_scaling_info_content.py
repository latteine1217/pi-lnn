"""K-scaling sensor information content（LES_N256 vs DNS-pivot, K = 100..750）。

What:
    對 K ∈ {100, 200, 300, 400, 500, 750}：
      - QR-pivot on LES_N256 feature matrix → K sensors
      - QR-pivot on DNS feature matrix → K sensors (control)
      - 計算 effective rank, POD recon error (r=50), pairwise redundancy, 8×8 coverage

Why:
    EXP-103 K=100 KE 52.0%（worst）。問題是否 K 不夠？K 增加能否讓 information 趕上 DNS-pivot K=100？

Output:
    docs/assets/sensor_k_scaling_info.png  — 4 panels, x-axis K, y-axis 4 metrics
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import qr  # type: ignore[import]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "axes.linewidth": 0.7, "axes.grid": True, "grid.linewidth": 0.4,
    "grid.alpha": 0.3, "grid.color": "#999999",
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

K_LIST = [100, 200, 300, 400, 500, 750]
DNS_PATH = "data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy"
LES_PATH = "../kolmogorov_generate/dns/data/dataset_les_re10000_n256_t5.npy"


def compute_features(u, v, L=1.0):
    """Build [5*T, N²] feature matrix matching既有 QR-pivot 流程."""
    T, N, _ = u.shape
    dx = dy = L / N
    dvdx = (np.roll(v, -1, axis=2) - np.roll(v, 1, axis=2)) / (2 * dx)
    dudy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dy)
    omega = dvdx - dudy
    dudx = (np.roll(u, -1, axis=2) - np.roll(u, 1, axis=2)) / (2 * dx)
    dudy2 = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dy)
    grad_u = np.sqrt(dudx ** 2 + dudy2 ** 2)
    dvdx2 = (np.roll(v, -1, axis=2) - np.roll(v, 1, axis=2)) / (2 * dx)
    dvdy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dy)
    grad_v = np.sqrt(dvdx2 ** 2 + dvdy ** 2)
    feats = [u, v, omega, grad_u, grad_v]
    stacks = []
    for f in feats:
        flat = f.reshape(T, N * N)
        std = flat.std() + 1e-8
        stacks.append(flat / std)
    return np.concatenate(stacks, axis=0).astype(np.float32)


def qr_pivot(M, K):
    _, _, P = qr(M, pivoting=True)
    return P[:K]


def coords_from_indices(indices, N, L=1.0):
    """flat index → (x, y) phys coords."""
    row, col = np.unravel_index(indices, (N, N))
    x = col * (L / N)
    y = row * (L / N)
    return np.stack([x, y], axis=1), row, col


def analyze(u, v, indices, N_grid, dns_pod_phi, dns_x_arr=None):
    """4 個 information metrics for given sensor indices on grid N_grid."""
    row, col = np.unravel_index(indices, (N_grid, N_grid))
    K = len(indices)
    T = u.shape[0]

    # B. Effective rank from sensor time-series matrix
    M = np.concatenate([u[:, row, col].T, v[:, row, col].T], axis=1)  # [K, 2T]
    M = M - M.mean(axis=1, keepdims=True)
    s = np.linalg.svd(M, compute_uv=False)
    p = s ** 2 / (s ** 2).sum()
    eff_rank = float(np.exp(-(p * np.log(p + 1e-30)).sum()))

    # D. Pairwise redundancy
    u_ts = u[:, row, col].T  # [K, T]
    v_ts = v[:, row, col].T
    u_corr = np.corrcoef(u_ts)
    v_corr = np.corrcoef(v_ts)
    triu = np.triu_indices(K, k=1)
    redundancy = 0.5 * (np.abs(u_corr[triu]).mean()
                         + np.abs(v_corr[triu]).mean())

    # C. POD recon error (top r=50 DNS modes); 要把 sensor index 對齊 N=256 DNS grid
    # 對 LES_N256: same grid, direct use
    # 對 DNS-pivot: also same grid
    # Get sub matrix Phi_sub
    N = N_grid
    Phi_u = dns_pod_phi["Phi_u"]  # [N, N, r]
    Phi_v = dns_pod_phi["Phi_v"]
    Phi = dns_pod_phi["Phi"]  # [2N², r]
    Phi_sub_u = Phi_u[row, col, :]  # [K, r]
    Phi_sub_v = Phi_v[row, col, :]
    Phi_sub = np.concatenate([Phi_sub_u, Phi_sub_v], axis=0)  # [2K, r]
    u_full = dns_pod_phi["u_full"]
    v_full = dns_pod_phi["v_full"]
    recon_errs = []
    for ti in [50, 100, 150, 199]:
        uf = u_full[ti]; vf = v_full[ti]
        target_full = np.concatenate([uf.reshape(-1), vf.reshape(-1)])
        target_full = target_full - target_full.mean()
        target_sensors = np.concatenate([
            uf[row, col] - uf.mean(),
            vf[row, col] - vf.mean(),
        ])
        try:
            a_hat, *_ = np.linalg.lstsq(Phi_sub, target_sensors, rcond=None)
            recon_full = Phi @ a_hat
            err = np.linalg.norm(recon_full - target_full) / max(
                np.linalg.norm(target_full), 1e-30)
            recon_errs.append(err)
        except Exception:
            recon_errs.append(float("nan"))
    recon_err_mean = float(np.mean(recon_errs))

    # E. 8x8 coverage
    coords = np.stack([col / N_grid, row / N_grid], axis=1)
    bin_idx = np.clip((coords * 8).astype(int), 0, 7)
    occ = len(set(map(tuple, bin_idx)))

    return {
        "K": K,
        "effective_rank": eff_rank,
        "redundancy": redundancy,
        "pod_recon_err": recon_err_mean,
        "cell_coverage_8x8": occ,
    }


def main():
    print("Loading DNS...")
    dns = np.load(DNS_PATH, allow_pickle=True).item()
    u_dns = np.asarray(dns["u"], dtype=np.float64)
    v_dns = np.asarray(dns["v"], dtype=np.float64)
    print(f"  DNS u shape={u_dns.shape}")

    print("Loading LES_N256...")
    les = np.load(LES_PATH, allow_pickle=True).item()
    u_les = np.asarray(les["u"], dtype=np.float64)
    v_les = np.asarray(les["v"], dtype=np.float64)
    print(f"  LES u shape={u_les.shape}")

    # Precompute DNS POD basis (r=50)
    print("Computing DNS POD basis (r=50)...")
    T, N, _ = u_dns.shape
    X = np.concatenate([u_dns.reshape(T, -1).T, v_dns.reshape(T, -1).T], axis=0)
    X = X - X.mean(axis=1, keepdims=True)
    U_pod, S_pod, _ = np.linalg.svd(X, full_matrices=False)
    r = 50
    Phi = U_pod[:, :r]
    Phi_u = Phi[:N * N].reshape(N, N, r)
    Phi_v = Phi[N * N:].reshape(N, N, r)
    dns_pod = {"Phi": Phi, "Phi_u": Phi_u, "Phi_v": Phi_v,
               "u_full": u_dns, "v_full": v_dns}
    cum = np.cumsum(S_pod ** 2) / (S_pod ** 2).sum()
    print(f"  Top-{r} captures {cum[r-1]*100:.2f}% variance")

    # Build feature matrices
    print("\nBuilding feature matrices...")
    feats_dns = compute_features(u_dns, v_dns)
    feats_les = compute_features(u_les, v_les)
    print(f"  DNS feature mat: {feats_dns.shape}")
    print(f"  LES feature mat: {feats_les.shape}")

    # Pre-run QR pivot once at K=max, take first K each scale
    K_max = max(K_LIST)
    print(f"\nQR-pivot once at K={K_max} (DNS + LES) — re-use prefix for smaller K...")
    indices_dns_max = qr_pivot(feats_dns, K_max)
    indices_les_max = qr_pivot(feats_les, K_max)
    print(f"  done: DNS={len(indices_dns_max)}, LES={len(indices_les_max)} indices")

    print("\n" + "=" * 78)
    print("K-scaling information content")
    print("=" * 78)
    print(f"  {'source':<20} {'K':>5} {'eff_rank':>10} {'redundancy':>12} "
          f"{'POD_recon':>11} {'8x8_cov':>9}")

    results = {"DNS-pivot": [], "LES_N256": []}
    for K in K_LIST:
        # DNS-pivot
        ind = indices_dns_max[:K]
        m = analyze(u_dns, v_dns, ind, N, dns_pod)
        results["DNS-pivot"].append(m)
        print(f"  {'DNS-pivot':<20} {K:>5} {m['effective_rank']:>10.2f} "
              f"{m['redundancy']:>12.4f} {m['pod_recon_err']:>11.4f} "
              f"{m['cell_coverage_8x8']:>7}/64")
        # LES_N256
        ind = indices_les_max[:K]
        m = analyze(u_dns, v_dns, ind, N, dns_pod)  # 重要：用 DNS u,v 抽 sensor value
        results["LES_N256"].append(m)
        print(f"  {'LES_N256':<20} {K:>5} {m['effective_rank']:>10.2f} "
              f"{m['redundancy']:>12.4f} {m['pod_recon_err']:>11.4f} "
              f"{m['cell_coverage_8x8']:>7}/64")
        print()

    # Plot
    print("[plot] Saving K-scaling curves...")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    K_arr = np.array(K_LIST)
    metrics_info = [
        ("effective_rank", "(A) Effective rank (info dim)", "higher better"),
        ("pod_recon_err", "(B) POD recon error (r=50)", "lower better"),
        ("redundancy", "(C) Pairwise redundancy", "lower better"),
        ("cell_coverage_8x8", "(D) 8x8 cell coverage", "higher better"),
    ]
    colors = {"DNS-pivot": "#1f77b4", "LES_N256": "#d62728"}
    markers = {"DNS-pivot": "o", "LES_N256": "s"}

    for (key, title, hint), ax in zip(metrics_info, axes.ravel()):
        for src in ("DNS-pivot", "LES_N256"):
            ys = [m[key] for m in results[src]]
            ax.plot(K_arr, ys, marker=markers[src], color=colors[src],
                    linewidth=1.6, markersize=6, label=src)
        ax.set_xlabel("K (number of sensors)")
        ax.set_ylabel(title.split(") ")[1])
        ax.set_title(f"{title} — {hint}")
        ax.legend(loc="best", fontsize=9)
        # 標出 K=100 處 reference
        ax.axvline(100, color="gray", linestyle=":", alpha=0.5)

    fig.suptitle("Sensor information content vs K (LES_N256 vs DNS-pivot on Re=10⁴ DNS)",
                 fontsize=11)
    out = Path("docs/assets/sensor_k_scaling_info.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  → {out}")

    # Cross-comparison summary
    print("\n=== Summary: LES_N256 K needed to match DNS-pivot K=100 ===")
    dns_100 = results["DNS-pivot"][0]
    print(f"  DNS-pivot K=100 reference:")
    for k in ("effective_rank", "pod_recon_err", "redundancy", "cell_coverage_8x8"):
        print(f"    {k}: {dns_100[k]}")
    print(f"\n  LES_N256 K* needed to match (or never matches):")
    for k, hib in [("effective_rank", True), ("pod_recon_err", False),
                   ("redundancy", False), ("cell_coverage_8x8", True)]:
        target = dns_100[k]
        # find smallest K where LES_N256[K][k] reaches target (according to hib)
        found = None
        for i, K in enumerate(K_LIST):
            v = results["LES_N256"][i][k]
            if (hib and v >= target) or ((not hib) and v <= target):
                found = K
                break
        if found is None:
            print(f"    {k}: NEVER MATCHES at K≤{K_LIST[-1]} (worst={results['LES_N256'][-1][k]:.3f} vs DNS K=100={target:.3f})")
        else:
            print(f"    {k}: K*={found} matches DNS_K=100")


if __name__ == "__main__":
    main()
