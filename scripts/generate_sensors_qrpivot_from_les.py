#!/usr/bin/env python3
"""generate_sensors_qrpivot_from_les.py — 「LES → QR-pivot → DNS sensor」工程 pipeline。

What:
    Step A. 從 LES .npy 載入 u, v 時序，計算 features (u, v, omega, grad_u, grad_v)
    Step B. 對 LES grid 上的 feature matrix 做 column-pivoted QR，選 K 個最大資訊量位置
    Step C. 把 LES grid index → physical coord (x, y) ∈ [0, L]²
    Step D. 投到 DNS grid：對每個 LES sensor coord，找最近 DNS grid 點
    Step E. 從 DNS u, v 抽出對應位置的 time series
    Step F. 輸出 JSON + NPZ，schema 與既有 sensors_qrpivot_K100_N256_*.{json,npz} 完全一致

Why:
    ENGINEERING_VISION + REAL_WORLD_PIPELINE 規定：工程現場無 DNS，sensor 位置只能由
    LES 大尺度估計來決定。既有 sensors_qrpivot_K100_N256_t0-5_si100.{json,npz} 是
    從 DNS 全場做 QR，與工程現場矛盾。本 script 修正此 anti-pattern。

Usage:
    uv run python scripts/generate_sensors_qrpivot_from_les.py \\
        --les data/les/kolmogorov_les_Re10000_N128_T15_bardina_standalone.npy \\
        --dns data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy \\
        --K 100 \\
        --t_spinup 5.0 \\
        --out data/kolmogorov_sensors/re10000 \\
        --tag K100_N256_t0-5_si100_lesinformed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ── 特徵計算 ─────────────────────────────────────────────────────────────────

def compute_omega(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """渦量 ω = ∂v/∂x − ∂u/∂y（週期邊界，二階中央差分）。"""
    dvdx = (np.roll(v, -1, axis=2) - np.roll(v, 1, axis=2)) / (2 * dx)
    dudy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dy)
    return dvdx - dudy


def compute_grad_mag(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """梯度幅值 |∇f| = sqrt((∂f/∂x)² + (∂f/∂y)²)。"""
    dfdx = (np.roll(field, -1, axis=2) - np.roll(field, 1, axis=2)) / (2 * dx)
    dfdy = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2 * dy)
    return np.sqrt(dfdx**2 + dfdy**2)


# ── QR Pivoting ───────────────────────────────────────────────────────────────

def qr_pivot_select(feature_matrix: np.ndarray, K: int) -> np.ndarray:
    """column-pivoted QR：選 K 個最線性獨立的空間點（feature_matrix.shape = [n_feat, N²]）。"""
    from scipy.linalg import qr  # type: ignore[import]
    _, _, P = qr(feature_matrix, pivoting=True)
    return P[:K]


# ── LES 場處理 ────────────────────────────────────────────────────────────────

def load_les(les_path: Path, t_spinup: float) -> dict:
    """載入 LES 場並排除 spin-up phase。"""
    p = np.load(les_path, allow_pickle=True).item()
    t = np.asarray(p["time"], dtype=np.float64)
    mask = t >= t_spinup
    if mask.sum() < 5:
        raise ValueError(
            f"LES 場排除 spin-up 後僅 {mask.sum()} 幀，需 ≥ 5；"
            f"請降低 --t_spinup（當前 {t_spinup}, T_end={t[-1]:.2f}）"
        )
    cfg = p.get("config", {})
    return {
        "u": np.asarray(p["u"], dtype=np.float32)[mask],
        "v": np.asarray(p["v"], dtype=np.float32)[mask],
        "time": t[mask],
        "L": float(cfg.get("L", 1.0)),
        "N": int(cfg.get("N", p["u"].shape[-1])),
        "config": cfg,
    }


def build_feature_matrix(les: dict) -> tuple[np.ndarray, list[str]]:
    """建 [n_feat*T, N²] 特徵矩陣（u, v, omega, grad_u_mag, grad_v_mag 各 normalize 到 unit std）。"""
    u, v = les["u"], les["v"]
    N = les["N"]
    L = les["L"]
    dx = dy = L / N
    omega = compute_omega(u, v, dx, dy)
    grad_u = compute_grad_mag(u, dx, dy)
    grad_v = compute_grad_mag(v, dx, dy)
    features_used = ["u", "v", "omega", "grad_u_mag", "grad_v_mag"]
    feat_arrays = [u, v, omega, grad_u, grad_v]
    T = u.shape[0]
    stacks = []
    for feat in feat_arrays:
        flat = feat.reshape(T, N * N)
        std = flat.std() + 1e-8
        stacks.append(flat / std)
    feature_matrix = np.concatenate(stacks, axis=0).astype(np.float32)
    return feature_matrix, features_used


# ── DNS 對應 sensor 抽取 ──────────────────────────────────────────────────────

def project_to_dns_grid(
    les_coords: np.ndarray,
    dns_x: np.ndarray,
    dns_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """每個 LES sensor coord 找最近 DNS grid point。

    Returns:
        coords_on_dns: [K, 2] (x, y) DNS grid 上的物理座標
        indices_on_dns: [K] flatten index in DNS grid (row-major row*N + col)
    """
    N_dns = len(dns_x)
    # 對每個 LES coord (x, y)，最近 DNS grid index
    x_idx = np.argmin(np.abs(les_coords[:, 0:1] - dns_x[None, :]), axis=1)
    y_idx = np.argmin(np.abs(les_coords[:, 1:2] - dns_y[None, :]), axis=1)
    coords_on_dns = np.stack([dns_x[x_idx], dns_y[y_idx]], axis=1)
    # DNS / LES array convention: u_full[t, axis_1=x, axis_2=y]
    # ⇒ flat index = x_idx * N + y_idx, row=x_idx, col=y_idx
    flat_indices = x_idx * N_dns + y_idx
    return coords_on_dns, flat_indices


def extract_dns_sensor_values(
    dns_path: Path,
    sensor_coords_dns: np.ndarray,
    dns_x: np.ndarray,
    dns_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """從 DNS 抽出 sensor 位置的 u, v time-series + time vector。"""
    raw = np.load(dns_path, allow_pickle=True).item()
    u_full = np.asarray(raw["u"], dtype=np.float32)
    v_full = np.asarray(raw["v"], dtype=np.float32)
    time_arr = np.asarray(raw["time"], dtype=np.float32)
    # 對每個 (x, y) sensor，回頭找 DNS index
    x_idx = np.argmin(np.abs(sensor_coords_dns[:, 0:1] - dns_x[None, :]), axis=1)
    y_idx = np.argmin(np.abs(sensor_coords_dns[:, 1:2] - dns_y[None, :]), axis=1)
    # DNS convention: u_full[t, axis_1=x, axis_2=y] — 用 (x_idx, y_idx) 順序索引
    sensor_u = u_full[:, x_idx, y_idx].T  # [K, T]
    sensor_v = v_full[:, x_idx, y_idx].T
    return sensor_u, sensor_v, time_arr


# ── 主程序 ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sensors via LES-based QR-pivot + DNS-derived sensor values"
    )
    parser.add_argument("--les", required=True, help="LES .npy（提供 large-scale mode 估計）")
    parser.add_argument("--dns", required=True, help="DNS .npy（提供 sensor 量測值）")
    parser.add_argument("--K", type=int, default=100, help="sensor 數量（預設 100）")
    parser.add_argument("--t_spinup", type=float, default=5.0,
                        help="LES spin-up 時間，排除前段（預設 5.0 秒）")
    parser.add_argument("--out", required=True, help="輸出目錄")
    parser.add_argument("--tag", default=None,
                        help="輸出檔名 tag，預設依 K/N/T 自動生成（並附 _lesinformed）")
    args = parser.parse_args()

    les_path = Path(args.les)
    dns_path = Path(args.dns)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Step A. 載入 LES（排除 spin-up）===
    print(f"[A] Loading LES (excluding t < {args.t_spinup})...")
    les = load_les(les_path, args.t_spinup)
    print(f"    LES grid N={les['N']}, L={les['L']}, "
          f"T_used=[{les['time'][0]:.3f}, {les['time'][-1]:.3f}], "
          f"frames_used={les['u'].shape[0]}")

    # === Step B. 建 feature matrix + QR-pivot ===
    print("[B] Building feature matrix + QR-pivot on LES grid...")
    feature_matrix, features_used = build_feature_matrix(les)
    print(f"    feature_matrix shape={feature_matrix.shape} "
          f"({len(features_used)} feats × T={les['u'].shape[0]} × N²={les['N']**2})")
    indices_les = qr_pivot_select(feature_matrix, args.K)
    print(f"    selected K={args.K} sensors on LES grid")

    # === Step C. LES grid index → physical coord ===
    N_les = les["N"]
    L = les["L"]
    les_x = np.arange(N_les) * (L / N_les)
    les_y = np.arange(N_les) * (L / N_les)
    # LES feature matrix 以 row-major 把 (axis_1=x, axis_2=y) flatten；
    # unravel 得 row=x_idx, col=y_idx → coord = (les_x[x_idx], les_y[y_idx])
    row_idx_les, col_idx_les = np.unravel_index(indices_les, (N_les, N_les))
    x_idx_les, y_idx_les = row_idx_les, col_idx_les
    coords_on_les = np.stack([les_x[x_idx_les], les_y[y_idx_les]], axis=1)  # [K, 2]
    print(f"[C] LES physical coords range: "
          f"x ∈ [{coords_on_les[:, 0].min():.3f}, {coords_on_les[:, 0].max():.3f}], "
          f"y ∈ [{coords_on_les[:, 1].min():.3f}, {coords_on_les[:, 1].max():.3f}]")

    # === Step D. Project to DNS grid ===
    print(f"[D] Projecting {args.K} LES coords to DNS grid (nearest neighbor)...")
    raw_dns = np.load(dns_path, allow_pickle=True).item()
    dns_x = np.asarray(raw_dns["x"], dtype=np.float64)
    dns_y = np.asarray(raw_dns["y"], dtype=np.float64)
    N_dns = len(dns_x)
    coords_on_dns, _ = project_to_dns_grid(coords_on_les, dns_x, dns_y)
    # 檢查唯一性（避免投到 DNS 後重疊）
    coords_tuples = [tuple(c) for c in coords_on_dns]
    n_unique = len(set(coords_tuples))
    if n_unique < args.K:
        print(f"    ⚠️  {args.K - n_unique} sensor 在投影後與其他重疊（DNS 網格 quantization）")
    print(f"    DNS grid N={N_dns}, dx={float(dns_x[1] - dns_x[0]):.5f}")

    # === Step E. Extract DNS sensor values ===
    print("[E] Extracting DNS sensor time-series...")
    sensor_u, sensor_v, time_arr = extract_dns_sensor_values(
        dns_path, coords_on_dns, dns_x, dns_y
    )
    print(f"    sensor_u shape={sensor_u.shape}, sensor_v shape={sensor_v.shape}")
    print(f"    DNS time range=[{time_arr[0]:.3f}, {time_arr[-1]:.3f}], "
          f"steps={len(time_arr)}")

    # === Step F. Write JSON + NPZ（schema 對齊既有 DNS-pivot）===
    t0 = f"{time_arr[0]:.0f}".replace(".", "p")
    t1 = f"{time_arr[-1]:.0f}".replace(".", "p")
    default_tag = f"K{args.K}_N{N_dns}_t{t0}-{t1}_si100_lesinformed"
    tag = args.tag if args.tag else default_tag

    json_path = out_dir / f"sensors_qrpivot_{tag}.json"
    npz_path = out_dir / f"sensors_qrpivot_{tag}_dns_values.npz"

    meta = {
        "K": args.K,
        "resolution": f"{N_dns}x{N_dns}",
        "spatial_downsample_res": f"{N_dns}x{N_dns}",
        "spatial_downsample_stride": 1,
        "method": "qr_pivoting_from_les",
        "features": features_used,
        "time_stride": 1,
        "time_range": [float(time_arr[0]), float(time_arr[-1])],
        "time_steps": int(len(time_arr)),
        "selected_coordinates": coords_on_dns.tolist(),
        "source_file": str(dns_path),
        "les_source_file": str(les_path),
        "les_N": N_les,
        "les_t_spinup": args.t_spinup,
        "dns_values_npz": str(npz_path),
        "sensor_dt": float(time_arr[1] - time_arr[0]),
        "sensor_time_points": int(len(time_arr)),
        "engineering_pipeline_note":
            "Sensor positions chosen via QR-pivot on LES (engineering proxy); "
            "sensor values extracted from DNS (real-world surrogate). "
            "Conforms to REAL_WORLD_PIPELINE in AGENTS.md.",
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[F] Saved JSON: {json_path}")

    np.savez(npz_path, time=time_arr, u=sensor_u, v=sensor_v)
    print(f"    Saved NPZ:  {npz_path}")
    print("Done.")


if __name__ == "__main__":
    main()
