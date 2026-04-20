#!/usr/bin/env python3
"""generate_sensors_qrpivot.py — 以 QR Pivoting 從 DNS 場選取最優 sensor 位置，
並可診斷多組 sensor 的實際頻譜重建能力。

What: 從 DNS npy 載入全場時序，建構多特徵矩陣（u, v, omega, grad_u_mag, grad_v_mag），
      執行 column-pivoted QR 分解，選出 K 個空間最大線性獨立的 sensor 位置，
      輸出 JSON（位置）與 NPZ（u, v 時序）。
      可附加 --diagnose 模式：對多組 sensor 做 RBF 重建，
      計算各 wavenumber 的實際頻譜誤差，輸出對比圖。

Why: QR pivoting 在高維線性代數意義下選出的位置，能最大化空間表徵的覆蓋度；
     加入 omega/grad 特徵使 pivoting 偏向高頻渦旋與剪切層，覆蓋更多小尺度結構。
     Nyquist 估計（最近鄰距離法）對非均勻採樣不嚴謹，
     診斷模式使用實際重建誤差，給出各 k 的可觀測性直接量測。

Usage:
    # 生成 K=200 si100 sensor
    uv run python scripts/generate_sensors_qrpivot.py \\
        --dns   data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy \\
        --K     200 \\
        --out   data/kolmogorov_sensors/re10000 \\
        --tag   K200_N256_t0-5_si100

    # 診斷 K=100 vs K=200 的頻譜重建能力（不生成新 sensor）
    uv run python scripts/generate_sensors_qrpivot.py --diagnose \\
        --dns  data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy \\
        --compare \\
            data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100.json \\
            data/kolmogorov_sensors/re10000/sensors_qrpivot_K200_N256_t0-5_si100.json \\
        --out  docs/assets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ── 特徵計算 ─────────────────────────────────────────────────────────────────

def compute_omega(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """渦量 ω = ∂v/∂x − ∂u/∂y（週期邊界，二階中央差分）。

    Args:
        u, v: [T, N, N] 速度場
        dx, dy: 格距
    Returns:
        omega: [T, N, N]
    """
    dvdx = (np.roll(v, -1, axis=2) - np.roll(v, 1, axis=2)) / (2 * dx)
    dudy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dy)
    return dvdx - dudy


def compute_grad_mag(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """梯度幅值 |∇f| = sqrt((∂f/∂x)² + (∂f/∂y)²)。

    Args:
        field: [T, N, N]
    Returns:
        grad_mag: [T, N, N]
    """
    dfdx = (np.roll(field, -1, axis=2) - np.roll(field, 1, axis=2)) / (2 * dx)
    dfdy = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2 * dy)
    return np.sqrt(dfdx**2 + dfdy**2)


# ── QR Pivoting ───────────────────────────────────────────────────────────────

def qr_pivot_select(feature_matrix: np.ndarray, K: int) -> np.ndarray:
    """以 column-pivoted QR 從特徵矩陣中選出 K 個最線性獨立的欄（空間點）。

    Why QR pivoting: R 的對角元素降序排列，前 K 個 pivot 欄覆蓋最大的
    column space，即空間上最「有資訊量」的位置。

    Args:
        feature_matrix: [n_features, n_spatial] — 行是特徵，欄是空間點
        K: 選取的 sensor 數量
    Returns:
        indices: [K] int — 選取的空間欄索引（降序資訊量）
    """
    # scipy 的 qr 支援 pivoting=True，回傳排列向量 P
    from scipy.linalg import qr  # type: ignore[import]
    _, _, P = qr(feature_matrix, pivoting=True)
    return P[:K]


# ── 頻譜重建診斷 ──────────────────────────────────────────────────────────────

def _build_k_shells(N: int) -> np.ndarray:
    """預建 [N, N] 的徑向 wavenumber 整數矩陣，避免重複計算。"""
    kx = np.fft.fftfreq(N, d=1.0 / N).astype(int)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    return np.round(np.sqrt(KX**2 + KY**2)).astype(int)


def fourier_pseudoinverse_accuracy(
    sensor_coords: np.ndarray,
    true_fft: np.ndarray,
    x_arr: np.ndarray,
    k_max_eval: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """以 Fourier pseudo-inverse 評估 K 個 sensor 位置對各 wavenumber 的幾何覆蓋度。

    What: 針對每個整數 wavenumber k，建構測量矩陣 Φ[j,m] = exp(+i*2π*(kx_m*x_j + ky_m*y_j)/L)。
          先計算「理想孤立觀測」obs_ki = Φ @ u_hat_true（僅含 ki-shell 的場值），
          再用 least-squares 從 obs_ki 還原 u_hat，與真實比較。

    Why:  若直接用全場觀測 u(x_j) 當 RHS，全場包含所有 k 的貢獻，
          用單一 k-shell 的 M_ki 個未知數無法擬合全場 → accuracy 恆為 0（前版錯誤）。
          正確做法：隔離 ki-shell 後，測試 Φ 的行列式/秩能否從 K 個 sensor
          唯一識別 M_ki 個模態。K ≥ M_ki 時過定（高精度），K < M_ki 時欠定（精度下降）。

    Args:
        sensor_coords: [K, 2] sensor 位置（物理座標，x 在前）
        true_fft:      [N, N] 真實場的 2D FFT（已除以 N²）
        x_arr:         [N]    格點座標（x 和 y 共用，假設各向同性）
        k_max_eval:    評估的最大 wavenumber
    Returns:
        k_vals:   [k_max_eval] 整數 wavenumber 1..k_max_eval
        accuracy: [k_max_eval] accuracy(k) = 1 − ||û_recon − û_true||² / ||û_true||²
    """
    N = len(x_arr)
    L = float(x_arr[-1] - x_arr[0]) + float(x_arr[1] - x_arr[0])  # 週期長度
    kx_grid = np.fft.fftfreq(N, d=1.0 / N).astype(int)
    KX_all, KY_all = np.meshgrid(kx_grid, kx_grid, indexing="ij")
    K_mag_all = np.round(np.sqrt(KX_all**2 + KY_all**2)).astype(int)

    acc = np.zeros(k_max_eval)

    for ki in range(1, k_max_eval + 1):
        mask     = K_mag_all == ki
        kx_modes = KX_all[mask].ravel()
        ky_modes = KY_all[mask].ravel()
        M = len(kx_modes)
        if M == 0:
            continue

        # 真實 Fourier 係數 [M]
        u_hat_true = true_fft[mask].ravel()
        E_true = np.sum(np.abs(u_hat_true) ** 2)
        if E_true < 1e-30:
            acc[ki - 1] = 1.0
            continue

        # 測量矩陣 Φ [K, M]：IDFT 基底，exp(+i*2π*(kx*x + ky*y)/L)
        phase = (2 * np.pi / L) * (
            np.outer(sensor_coords[:, 0], kx_modes)
            + np.outer(sensor_coords[:, 1], ky_modes)
        )
        Phi = np.exp(1j * phase)  # [K, M]

        # 孤立觀測：只含 ki-shell 貢獻的場值 [K]
        obs_ki = Phi @ u_hat_true

        # 從孤立觀測重建 ki-shell 係數
        u_hat_recon, *_ = np.linalg.lstsq(Phi, obs_ki, rcond=None)

        E_res = np.sum(np.abs(u_hat_recon - u_hat_true) ** 2)
        acc[ki - 1] = max(0.0, 1.0 - float(E_res) / float(E_true))

    return np.arange(1, k_max_eval + 1), acc


def diagnose_spectral_coverage(
    dns_data: dict,
    sensor_jsons: list[Path],
    n_snapshots: int = 8,
    out_dir: Path | None = None,
) -> None:
    """對多組 sensor 計算各 wavenumber 的 RBF 重建誤差，輸出對比圖。

    What: 對 n_snapshots 個均勻分佈的 DNS 時間步，分別做 RBF 重建，
          計算重建場與真實場的能譜相對誤差 ε(k) = E_recon(k)/E_true(k)。
          ε(k)≈1 表示該 wavenumber 重建良好；ε(k)→0 表示模型無法從 sensor 重建。

    Why:  正確的頻譜覆蓋評估應以實際重建誤差為準，
          而非最近鄰距離（後者對非均勻採樣無效）。

    Args:
        dns_data:      已載入的 DNS dict（含 u, v, x, y, time）
        sensor_jsons:  各組 sensor 的 JSON 路徑清單
        n_snapshots:   用於平均的時間快照數
        out_dir:       圖片輸出目錄（None 則只顯示）
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import]

    u_full   = dns_data["u"].astype(np.float64)  # [T, N, N]
    x_arr    = dns_data["x"].astype(np.float64)
    y_arr    = dns_data["y"].astype(np.float64)
    time_arr = dns_data["time"]
    T, N, _  = u_full.shape

    # 全場查詢點 [N², 2]
    XX, YY = np.meshgrid(x_arr, y_arr, indexing="ij")
    query_coords = np.stack([XX.ravel(), YY.ravel()], axis=1)

    # 均勻選取快照索引（跳過 t=0，避免初始條件偏差）
    snap_idx = np.linspace(T // 10, T - 1, n_snapshots, dtype=int)

    K_EVAL = 50  # 評估 wavenumber 上限（覆蓋至慣性範圍）

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(sensor_jsons)))  # type: ignore[attr-defined]

    results: list[tuple[str, int, np.ndarray, np.ndarray]] = []

    for json_path, color in zip(sensor_jsons, colors):
        with open(json_path, encoding="utf-8") as f:
            meta = json.load(f)
        K_cfg  = meta["K"]
        coords = np.array(meta["selected_coordinates"])

        label    = f"K={K_cfg}"
        acc_acc  = np.zeros(K_EVAL)

        for t_idx in snap_idx:
            u_true   = u_full[t_idx].astype(np.float64)
            true_fft = np.fft.fft2(u_true) / (N * N)
            k_vals, acc = fourier_pseudoinverse_accuracy(
                coords, true_fft, x_arr, k_max_eval=K_EVAL
            )
            acc_acc += acc

        acc_mean = acc_acc / n_snapshots
        results.append((json_path.name, K_cfg, k_vals, acc_mean))

        axes[0].plot(k_vals, acc_mean, color=color, linewidth=2, label=label)
        axes[1].plot(k_vals, acc_mean, color=color, linewidth=2, label=label)

    # 能譜參考（右 y 軸）
    k_shells = _build_k_shells(N)
    F_ref    = np.fft.fft2(u_full[snap_idx[len(snap_idx)//2]].astype(np.float64))
    E_ref    = np.array([(np.abs(F_ref[k_shells == ki])**2).sum()
                         for ki in range(1, K_EVAL + 1)])
    ax2 = axes[1].twinx()
    ax2.semilogy(k_vals, E_ref, "k--", linewidth=1, alpha=0.4, label="DNS E(k)")
    ax2.set_ylabel("DNS Energy E(k)", fontsize=8)
    ax2.legend(loc="lower right", fontsize=7)

    # 參考線
    for ax in axes:
        ax.axhline(0.8, color="green",  linestyle="--", linewidth=1, label="acc=0.8")
        ax.axhline(0.5, color="orange", linestyle=":",  linewidth=1, label="acc=0.5")
        ax.set_xlabel("Wavenumber k")
        ax.set_ylabel("Reconstruction Accuracy  1 − ||û_recon−û_true||²/||û_true||²")
        ax.set_xlim(1, K_EVAL)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)

    axes[0].set_title("Fourier Pseudo-inverse Accuracy (full view)")
    axes[1].set_title(f"Zoom: k=1..{K_EVAL} + DNS Energy")

    plt.tight_layout()

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "sensor_spectral_coverage.png"
        plt.savefig(out_path, dpi=150)
        print(f"Saved diagnostic plot: {out_path}")
    else:
        plt.show()

    # 文字摘要
    print("\n=== 頻譜重建診斷摘要（Fourier pseudo-inverse, u-field, {} 快照平均）===".format(n_snapshots))
    print(f"  accuracy(k) = 1 − ||û_recon − û_true||² / ||û_true||²")
    print(f"  k_cutoff = 最後一個 accuracy > threshold 的 k")
    print()
    print(f"  {'K':<6}  {'k(acc>0.8)':<14}  {'k(acc>0.5)':<14}  {'k(acc>0.2)'}")
    for (name, K_tmp, k_v, acc) in results:
        def cutoff(thresh: float) -> int:
            above = np.where(acc > thresh)[0]
            return int(k_v[above[-1]]) if len(above) else 0
        print(f"  K={K_tmp:<4}  {cutoff(0.8):<14}  {cutoff(0.5):<14}  {cutoff(0.2)}")

    plt.close()


# ── 主程序 ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="QR-pivot sensor selection + spectral diagnosis")
    parser.add_argument("--dns",      required=True, help="DNS npy 檔路徑")
    parser.add_argument("--K",        type=int, default=200, help="sensor 數量（生成模式）")
    parser.add_argument("--out",      required=True, help="輸出目錄（sensor 或圖片）")
    parser.add_argument("--tag",      default=None,
                        help="輸出檔名標籤，預設依 K/N/time_range 自動生成")
    parser.add_argument("--diagnose", action="store_true",
                        help="診斷模式：只做頻譜重建對比，不生成新 sensor")
    parser.add_argument("--compare",  nargs="+", default=None,
                        help="診斷模式：指定要比較的 sensor JSON 路徑清單")
    parser.add_argument("--n-snapshots", type=int, default=8,
                        help="診斷用的時間快照數（預設 8）")
    args = parser.parse_args()

    dns_path = Path(args.dns)
    out_dir  = Path(args.out)

    # ── 載入 DNS ──────────────────────────────────────────────────────────────
    print(f"Loading DNS: {dns_path}")
    raw = np.load(dns_path, allow_pickle=True).item()

    u_full    = raw["u"].astype(np.float32)   # [T, N, N]
    v_full    = raw["v"].astype(np.float32)
    time_arr  = raw["time"].astype(np.float32) # [T]
    x_arr     = raw["x"].astype(np.float32)   # [N]
    y_arr     = raw["y"].astype(np.float32)

    T, N, _ = u_full.shape
    dx = float(x_arr[1] - x_arr[0])
    dy = float(y_arr[1] - y_arr[0])

    print(f"  DNS: T={T}, N={N}x{N}, t=[{time_arr[0]:.3f}, {time_arr[-1]:.3f}], dt={dx:.4f}")

    # ── 診斷模式（不生成 sensor）─────────────────────────────────────────────
    if args.diagnose:
        if not args.compare:
            parser.error("--diagnose 需要 --compare 指定至少一個 sensor JSON")
        diagnose_spectral_coverage(
            dns_data     = raw,
            sensor_jsons = [Path(p) for p in args.compare],
            n_snapshots  = args.n_snapshots,
            out_dir      = out_dir,
        )
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 計算衍生特徵 ──────────────────────────────────────────────────────────
    print("Computing derived features (omega, grad_u_mag, grad_v_mag)...")
    p_full    = raw["p"].astype(np.float32) if "p" in raw else None
    omega     = compute_omega(u_full, v_full, dx, dy)          # [T, N, N]
    grad_u    = compute_grad_mag(u_full, dx, dy)               # [T, N, N]
    grad_v    = compute_grad_mag(v_full, dx, dy)               # [T, N, N]

    # 若 DNS 含壓力場則納入；否則退回 5 特徵
    if p_full is not None:
        features_used = ["u", "v", "p", "omega", "grad_u_mag", "grad_v_mag"]
        feat_arrays   = [u_full, v_full, p_full, omega, grad_u, grad_v]
    else:
        features_used = ["u", "v", "omega", "grad_u_mag", "grad_v_mag"]
        feat_arrays   = [u_full, v_full, omega, grad_u, grad_v]
        print("  Warning: 'p' not found in DNS, using 5-feature set")

    # ── 建構特徵矩陣 [n_feat*T, N*N] ─────────────────────────────────────────
    # 每個空間點的欄向量 = 在所有時間步、所有特徵下的數值
    # normalize 每個特徵到 unit std 防止 omega/grad 量級主導
    print("Building feature matrix for QR pivoting...")
    stacks = []
    for feat_name, feat in zip(features_used, feat_arrays):
        flat = feat.reshape(T, N * N)           # [T, N²]
        std  = flat.std() + 1e-8
        stacks.append(flat / std)

    # feature_matrix: [n_feat*T, N²]
    feature_matrix = np.concatenate(stacks, axis=0).astype(np.float32)
    print(f"  Feature matrix: {feature_matrix.shape} "
          f"({len(features_used)} feats × {T} steps × {N}² spatial)")

    # ── QR Pivoting 選 sensor ────────────────────────────────────────────────
    K = args.K
    print(f"Running QR pivoting to select K={K} sensors...")
    indices = qr_pivot_select(feature_matrix, K)  # [K]
    print(f"  Selected {len(indices)} sensor indices")

    # 將 flat index 轉回 (i, j) → 物理座標 (x, y)
    row_idx, col_idx = np.unravel_index(indices, (N, N))
    coords = np.stack([x_arr[col_idx], y_arr[row_idx]], axis=1)  # [K, 2] (x, y)

    # ── 提取 sensor 時序 (u, v) ───────────────────────────────────────────────
    sensor_u = u_full[:, row_idx, col_idx].T.astype(np.float32)  # [K, T]
    sensor_v = v_full[:, row_idx, col_idx].T.astype(np.float32)

    # 最近鄰距離診斷
    from scipy.spatial import cKDTree  # type: ignore[import]
    tree = cKDTree(coords)
    nn_dists, _ = tree.query(coords, k=2)
    nn_mean = float(nn_dists[:, 1].mean())
    k_nyquist = 1.0 / (2.0 * nn_mean) if nn_mean > 0 else float("inf")
    print(f"  Nearest-neighbor mean dist: {nn_mean:.4f}, "
          f"effective Nyquist k_max ≈ {k_nyquist:.1f}")

    # ── 輸出檔名 ─────────────────────────────────────────────────────────────
    t0  = f"{time_arr[0]:.0f}".replace(".", "p")
    t1  = f"{time_arr[-1]:.0f}".replace(".", "p")
    tag = args.tag if args.tag else f"K{K}_N{N}_t{t0}-{t1}"

    json_path = out_dir / f"sensors_qrpivot_{tag}.json"
    npz_path  = out_dir / f"sensors_qrpivot_{tag}_dns_values.npz"

    # ── 寫出 JSON ─────────────────────────────────────────────────────────────
    meta = {
        "K": K,
        "resolution": f"{N}x{N}",
        "spatial_downsample_res": f"{N}x{N}",
        "spatial_downsample_stride": 1,
        "method": "qr_pivoting",
        "features": features_used,
        "time_stride": 1,
        "time_range": [float(time_arr[0]), float(time_arr[-1])],
        "time_steps": T,
        "selected_coordinates": coords.tolist(),
        "indices": [int(i) for i in indices],
        "source_file": str(dns_path),
        "dns_values_npz": str(npz_path),
        "sensor_dt": float(time_arr[1] - time_arr[0]),
        "sensor_time_points": T,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # ── 寫出 NPZ ──────────────────────────────────────────────────────────────
    np.savez(npz_path, time=time_arr, u=sensor_u, v=sensor_v)
    print(f"Saved NPZ:  {npz_path}")
    print(f"  u shape: {sensor_u.shape}, v shape: {sensor_v.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
