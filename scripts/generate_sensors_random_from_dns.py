#!/usr/bin/env python3
"""generate_sensors_random_from_dns.py — 從同一 DNS 場以 uniform random
placement 取 K 個感測器，輸出格式與 generate_sensors_qrpivot.py 相同。

What:
    給定 DNS .npy 與 placement seed，從 N×N grid 中均勻隨機選 K 個格點作為
    sensor，提取對應時序 (u, v)，輸出 JSON (位置) + NPZ (時序值)。
    用於 trained model 在「不同 sensor configuration、同一段 DNS」的
    generalization 測試。

Why:
    既有 evaluator 使用訓練見過的 sensor trajectory (QR-pivot K=100, seed
    固定)，那只能驗證「reconstruction on seen sensor」，無法回答「operator
    對 unseen sensor placement 是否 generalize」。本 script 從同一 DNS field
    取不同 placement 的 sensor，保留 underlying flow ground truth 一致，
    純粹變動 model input 來檢視 conditional operator 的 generalization。

How:
    1. load DNS dict (u, v, x, y, time)
    2. uniform random pick K indices ∈ [0, N²) with given seed
    3. extract sensor_u/v 時序 = DNS[:, row, col].T
    4. write JSON (selected_coordinates 等 meta) + NPZ (time, u, v)

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/generate_sensors_random_from_dns.py \\
        --dns  data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy \\
        --K    100 \\
        --seed 1 \\
        --out  data/kolmogorov_sensors/re10000_alt \\
        --tag  K100_N256_t0-5_si100_random_pseed1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dns", type=Path, required=True, help="DNS .npy path (same DNS as training)")
    p.add_argument("--K", type=int, default=100, help="number of sensors")
    p.add_argument("--seed", type=int, required=True, help="placement seed (for uniform random)")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--tag", type=str, required=True, help="filename tag (no extension)")
    p.add_argument("--avoid-duplicates", action="store_true", default=True,
                   help="ensure K sensors are at K distinct grid points")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # ── 1. load DNS ───────────────────────────────────────────────
    print(f"[load] DNS: {args.dns}")
    dns = np.load(args.dns, allow_pickle=True).item()
    u_full = dns["u"].astype(np.float64)       # [T, N, N]
    v_full = dns["v"].astype(np.float64)
    x_arr  = dns["x"].astype(np.float64)       # [N]
    y_arr  = dns["y"].astype(np.float64)
    time_arr = dns["time"]                     # [T]
    T, N, _ = u_full.shape
    print(f"  T={T}, N={N}, time ∈ [{time_arr[0]:.3f}, {time_arr[-1]:.3f}]")

    # ── 2. uniform random pick K distinct grid indices ──────────
    rng = np.random.default_rng(args.seed)
    n_total = N * N
    if args.K > n_total:
        raise ValueError(f"K={args.K} > N²={n_total}, 不能取出比格點還多的 sensor")
    indices = rng.choice(n_total, size=args.K, replace=not args.avoid_duplicates).astype(np.int64)
    if args.avoid_duplicates:
        assert len(set(indices.tolist())) == args.K, "duplicates leaked"
    indices.sort()  # 排序好讀
    # DNS u shape = (T, axis_1=x, axis_2=y); row-major flatten → row=x_idx, col=y_idx
    row_idx, col_idx = np.unravel_index(indices, (N, N))
    x_idx, y_idx = row_idx, col_idx  # 名稱與物理 axis 對齊
    coords = np.stack([x_arr[x_idx], y_arr[y_idx]], axis=1)  # [K, 2] (x, y)
    print(f"[sample] placement_seed={args.seed}, K={args.K}, distinct={args.avoid_duplicates}")
    print(f"  coord range: x ∈ [{coords[:,0].min():.3f}, {coords[:,0].max():.3f}], "
          f"y ∈ [{coords[:,1].min():.3f}, {coords[:,1].max():.3f}]")

    # ── 3. 提取 sensor 時序 ────────────────────────────────────
    # u_full[t, x_idx, y_idx]：axis=1=x, axis=2=y
    sensor_u = u_full[:, x_idx, y_idx].T.astype(np.float32)  # [K, T]
    sensor_v = v_full[:, x_idx, y_idx].T.astype(np.float32)
    print(f"[extract] sensor_u shape={sensor_u.shape}, sensor_v shape={sensor_v.shape}")

    # 最近鄰距離診斷（與 QR-pivot script 對齊）
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    nn_dists, _ = tree.query(coords, k=2)
    nn_mean = float(nn_dists[:, 1].mean())
    k_nyquist = 1.0 / (2.0 * nn_mean) if nn_mean > 0 else float("inf")
    print(f"  nearest-neighbor mean dist: {nn_mean:.4f}, "
          f"effective Nyquist k_max ≈ {k_nyquist:.1f}")

    # ── 4. 寫出 JSON + NPZ ─────────────────────────────────────
    json_path = args.out / f"sensors_random_{args.tag}.json"
    npz_path  = args.out / f"sensors_random_{args.tag}_dns_values.npz"

    meta = {
        "K": args.K,
        "resolution": f"{N}x{N}",
        "spatial_downsample_res": f"{N}x{N}",
        "spatial_downsample_stride": 1,
        "method": "uniform_random",
        "placement_seed": args.seed,
        "avoid_duplicates": args.avoid_duplicates,
        "features": [],  # random 不用 feature ranking
        "time_stride": 1,
        "time_range": [float(time_arr[0]), float(time_arr[-1])],
        "time_steps": int(T),
        "selected_coordinates": coords.tolist(),
        "indices": [int(i) for i in indices],
        "source_file": str(args.dns),
        "dns_values_npz": str(npz_path),
        "sensor_dt": float(time_arr[1] - time_arr[0]),
        "sensor_time_points": int(T),
        "diagnostics": {
            "nn_mean_dist": nn_mean,
            "effective_nyquist_kmax": k_nyquist,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[save] JSON: {json_path}")

    np.savez(npz_path, time=time_arr, u=sensor_u, v=sensor_v)
    print(f"[save] NPZ : {npz_path}")
    print("Done.")


if __name__ == "__main__":
    main()
