#!/usr/bin/env python3
"""generate_sensors_pod_pivot_from_les.py — POD-pivot sensor selection on LES。

What:
    Manohar et al. 2018 / Brunton 2019 POD-pivot 演算法：
      1. SVD on LES snapshot matrix → leading r POD modes Φ ∈ ℝ^(N², r)
      2. Column-pivoted QR on Φᵀ → 選 K = r 個空間位置（each best aligned with 1 mode）
      3. 投到 DNS grid → 抽 sensor (u, v) time series

Why:
    QR-pivot on multi-feature stacked matrix （既有 generate_sensors_qrpivot_from_les.py）
    對 LES_N256 (dns-init) 選出 effective rank 11.33 / redundancy 0.342 的 sensors，
    EXP-103 KE 52%。本 script 改用 textbook POD-pivot，每 leading POD mode 對應 1 個 sensor，
    強制 sensor span leading-r mode space → 預期 lower redundancy + higher info dimensions。

Algorithm（與 QR-pivot 的差異）:
    QR-pivot:   M [n_feat × T, N²]  → choose K columns most linearly independent
    POD-pivot:  Φ [N², r]         → choose K = r positions, each "aligned with" 1 mode

Usage:
    uv run python scripts/generate_sensors_pod_pivot_from_les.py \\
        --les ../kolmogorov_generate/dns/data/dataset_les_re10000_n256_t5.npy \\
        --dns data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy \\
        --K 100 \\
        --channel u \\
        --out data/kolmogorov_sensors/re10000 \\
        --tag K100_N256_t0-5_si100_les_n256_podpivot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import qr  # type: ignore[import]


def pod_pivot_select(snapshot_matrix: np.ndarray, K: int) -> np.ndarray:
    """POD-pivot：先 SVD 得 leading K POD modes，再對 ΦSelf.T 做 column-pivoted QR。

    Args:
        snapshot_matrix: [n_spatial, T]，每列是一個時間 snapshot 的 flat 場
        K: sensor 數 = POD mode 數

    Returns:
        indices: [K] 選出的空間 flat indices
    """
    # Center
    X = snapshot_matrix - snapshot_matrix.mean(axis=1, keepdims=True)
    # SVD: X = U Σ Vᵀ; columns of U are POD modes
    print(f"  SVD on snapshot matrix shape={X.shape}...")
    U_pod, S_pod, _ = np.linalg.svd(X, full_matrices=False)
    cum = np.cumsum(S_pod ** 2) / (S_pod ** 2).sum()
    print(f"  Top-{K} POD modes capture {cum[K-1]*100:.3f}% variance")
    Phi = U_pod[:, :K]  # [n_spatial, K]
    # POD-pivot: QR with pivoting on Φᵀ → choose K columns (= K spatial positions)
    print(f"  Column-pivoted QR on Φᵀ shape={Phi.T.shape}...")
    _, _, P = qr(Phi.T, pivoting=True)
    return P[:K]


def load_les(path: Path, t_spinup: float) -> dict:
    p = np.load(path, allow_pickle=True).item()
    t = np.asarray(p["time"], dtype=np.float64)
    mask = t >= t_spinup
    if mask.sum() < 5:
        raise ValueError(f"LES 排除 spin-up 後僅 {mask.sum()} 幀，太少")
    cfg = p.get("config", {})
    return {
        "u": np.asarray(p["u"], dtype=np.float64)[mask],
        "v": np.asarray(p["v"], dtype=np.float64)[mask],
        "time": t[mask],
        "L": float(cfg.get("L", 1.0)),
        "N": int(cfg.get("N", p["u"].shape[-1])),
        "config": cfg,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--les", required=True)
    parser.add_argument("--dns", required=True)
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--t_spinup", type=float, default=0.0,
                        help="LES spin-up 排除（dns-init LES 用 0.0；stand-alone 用 5.0+）")
    parser.add_argument("--channel", choices=["u", "v", "uv"], default="uv",
                        help="POD 用哪個 channel 構建 snapshot matrix：u-only / v-only / joint uv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    les_path = Path(args.les)
    dns_path = Path(args.dns)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Step A. Load LES ===
    print(f"[A] Loading LES (excluding t < {args.t_spinup})...")
    les = load_les(les_path, args.t_spinup)
    u_les = les["u"]; v_les = les["v"]
    N_les = les["N"]; L = les["L"]
    T = u_les.shape[0]
    print(f"    LES N={N_les}, L={L}, frames={T}")

    # === Step B. Build snapshot matrix（per --channel）===
    print(f"[B] Building POD snapshot matrix (channel={args.channel})...")
    if args.channel == "u":
        X = u_les.reshape(T, -1).T  # [N², T]
    elif args.channel == "v":
        X = v_les.reshape(T, -1).T
    else:  # joint uv
        # Stacked channel: position i contributes 2 rows (u, v)
        # 為對齊「選 1 個 position」的 semantic，用 row-mean of (u_row, v_row) abs energy or 直接 concat as 2N²
        # 直接 [2N², T] 但選 K=K_total positions（u-row + v-row treated as separate candidate）
        # 之後 dedup position
        X = np.concatenate([u_les.reshape(T, -1).T, v_les.reshape(T, -1).T], axis=0)  # [2N², T]
    print(f"    snapshot matrix: {X.shape}")

    # === Step C. POD-pivot ===
    print(f"[C] POD-pivot select K={args.K} positions...")
    indices_raw = pod_pivot_select(X, args.K)

    if args.channel == "uv":
        # dedup: each raw_index ∈ [0, 2N²); first N² are u-channel, second N² are v-channel
        # 取 mod N² 得 spatial index，dedup 保前者
        seen = set()
        spatial_indices = []
        for ri in indices_raw:
            sp = int(ri % (N_les * N_les))
            if sp not in seen:
                seen.add(sp)
                spatial_indices.append(sp)
            if len(spatial_indices) >= args.K:
                break
        if len(spatial_indices) < args.K:
            # 若 dedup 後不足 K，用更多 raw 補
            print(f"    dedup 後僅 {len(spatial_indices)} positions, "
                  f"擴選 raw indices...")
            extended_raw = pod_pivot_select(X, args.K * 2)
            for ri in extended_raw:
                sp = int(ri % (N_les * N_les))
                if sp not in seen:
                    seen.add(sp)
                    spatial_indices.append(sp)
                if len(spatial_indices) >= args.K:
                    break
        indices = np.array(spatial_indices[:args.K])
    else:
        indices = indices_raw

    print(f"    selected {len(indices)} positions")

    # === Step D. LES indices → physical coords ===
    les_x = np.arange(N_les) * (L / N_les)
    les_y = np.arange(N_les) * (L / N_les)
    row_les, col_les = np.unravel_index(indices, (N_les, N_les))
    coords_on_les = np.stack([les_x[col_les], les_y[row_les]], axis=1)
    print(f"[D] LES coords range: x ∈ [{coords_on_les[:, 0].min():.3f}, "
          f"{coords_on_les[:, 0].max():.3f}]")

    # === Step E. Project to DNS grid, extract sensor values ===
    print("[E] Projecting to DNS + extracting sensor values...")
    raw_dns = np.load(dns_path, allow_pickle=True).item()
    dns_x = np.asarray(raw_dns["x"], dtype=np.float64)
    dns_y = np.asarray(raw_dns["y"], dtype=np.float64)
    u_dns = np.asarray(raw_dns["u"], dtype=np.float32)
    v_dns = np.asarray(raw_dns["v"], dtype=np.float32)
    time_arr = np.asarray(raw_dns["time"], dtype=np.float32)
    N_dns = len(dns_x)
    x_idx = np.argmin(np.abs(coords_on_les[:, 0:1] - dns_x[None, :]), axis=1)
    y_idx = np.argmin(np.abs(coords_on_les[:, 1:2] - dns_y[None, :]), axis=1)
    coords_on_dns = np.stack([dns_x[x_idx], dns_y[y_idx]], axis=1)
    sensor_u = u_dns[:, y_idx, x_idx].T
    sensor_v = v_dns[:, y_idx, x_idx].T
    print(f"    DNS N={N_dns}; sensor_u shape={sensor_u.shape}")

    # === Step F. Write JSON + NPZ ===
    t0 = f"{time_arr[0]:.0f}".replace(".", "p")
    t1 = f"{time_arr[-1]:.0f}".replace(".", "p")
    default_tag = f"K{args.K}_N{N_dns}_t{t0}-{t1}_si100_les_n256_podpivot"
    tag = args.tag if args.tag else default_tag

    json_path = out_dir / f"sensors_podpivot_{tag}.json"
    npz_path = out_dir / f"sensors_podpivot_{tag}_dns_values.npz"

    meta = {
        "K": args.K,
        "resolution": f"{N_dns}x{N_dns}",
        "spatial_downsample_res": f"{N_dns}x{N_dns}",
        "spatial_downsample_stride": 1,
        "method": f"pod_pivoting_from_les_{args.channel}",
        "time_stride": 1,
        "time_range": [float(time_arr[0]), float(time_arr[-1])],
        "time_steps": int(len(time_arr)),
        "selected_coordinates": coords_on_dns.tolist(),
        "source_file": str(dns_path),
        "les_source_file": str(les_path),
        "les_N": N_les,
        "les_t_spinup": args.t_spinup,
        "pod_channel": args.channel,
        "dns_values_npz": str(npz_path),
        "sensor_dt": float(time_arr[1] - time_arr[0]),
        "sensor_time_points": int(len(time_arr)),
        "engineering_pipeline_note":
            "POD-pivot sensor selection on LES (engineering proxy); sensor values from DNS. "
            "Per Manohar 2018 / Brunton 2019. Tests algorithm-vs-information bottleneck.",
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[F] Saved JSON: {json_path}")
    np.savez(npz_path, time=time_arr, u=sensor_u, v=sensor_v)
    print(f"    Saved NPZ:  {npz_path}")
    print("Done.")


if __name__ == "__main__":
    main()
