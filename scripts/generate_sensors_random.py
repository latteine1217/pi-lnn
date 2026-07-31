"""generate_sensors_random.py — Uniform random sensor placement baseline.

What:
    從 DNS 場的 N×N grid points 以 uniform random 選 K 個 sensor 位置，
    輸出與 `generate_sensors_qrpivot.py` **完全相同** 的 JSON + NPZ schema，
    讓 KolmogorovDataset 與下游 evaluator 無痛切換 sensor placement。

Why:
    Paper §5 sensor placement ablation：QR-pivot 之外的對照組必須與 QR
    在 K / N / time-stride / file schema 完全對齊，差異只在「placement 機制」。
    Random 是最樸素的 fair baseline (vs QR 的 spectral-aware pivot)。

Notes:
  - 同 grid points 採樣（不是連續座標）→ 與 QR 完全公平比較。
  - placement seed 由 --placement-seed 控制，與 training seed 解耦，
    便於做「N placements × M training seeds」 2-level ablation。
  - 輸出檔名 tag 包含 `_seed{N}` 區別 placement seed。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pi_con.sensors import coords_from_indices, flat_to_indices, sample_series  # noqa: E402


def random_select(N: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """從 [0, N²) 不重複隨機選 K 個 flat indices。

    Why uniform without replacement: K=100 << N²=65536，重複機率極小，
    但仍 enforce 不重複以避免 silent over-sampling 同一格點。
    """
    if K > N * N:
        raise ValueError(f"K ({K}) 不能大於 grid size N² ({N*N})")
    return rng.choice(N * N, size=K, replace=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Uniform random sensor placement")
    parser.add_argument("--dns", required=True, help="DNS npy 檔路徑")
    parser.add_argument("--K", type=int, default=100, help="sensor 數量")
    parser.add_argument("--placement-seed", type=int, required=True,
                        help="random placement seed（與 training seed 解耦）")
    parser.add_argument("--out", required=True, help="輸出目錄")
    parser.add_argument("--tag", default=None,
                        help="輸出檔名標籤，預設依 K/N/time_range/seed 自動生成")
    args = parser.parse_args()

    dns_path = Path(args.dns)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 載入 DNS ─────────────────────────────────────────
    print(f"Loading DNS: {dns_path}")
    raw = np.load(dns_path, allow_pickle=True).item()
    u_full = raw["u"].astype(np.float32)   # [T, N, N]
    v_full = raw["v"].astype(np.float32)
    time_arr = raw["time"].astype(np.float32)
    x_arr = raw["x"].astype(np.float32)
    y_arr = raw["y"].astype(np.float32)

    T, N, _ = u_full.shape
    print(f"  DNS: T={T}, N={N}×{N}, t=[{time_arr[0]:.3f}, {time_arr[-1]:.3f}]")

    # ── Uniform random 選 sensor ────────────────────────
    rng = np.random.default_rng(args.placement_seed)
    K = args.K
    indices = random_select(N, K, rng)
    print(f"  Uniform random selected K={K} sensors with placement_seed={args.placement_seed}")

    # flat index → (x_idx, y_idx) → 物理座標，全部經 pi_con.sensors（理由見 qrpivot 版）。
    x_idx, y_idx = flat_to_indices(indices, N)
    coords = coords_from_indices(x_idx, y_idx, x_arr, y_arr)  # [K, 2]

    sensor_u = sample_series(u_full, x_idx, y_idx).astype(np.float32)  # [K, T]
    sensor_v = sample_series(v_full, x_idx, y_idx).astype(np.float32)

    # 最近鄰距離診斷（與 QR 對照）
    from scipy.spatial import cKDTree  # type: ignore[import]
    tree = cKDTree(coords)
    nn_dists, _ = tree.query(coords, k=2)
    nn_mean = float(nn_dists[:, 1].mean())
    nn_min = float(nn_dists[:, 1].min())
    k_nyquist = 1.0 / (2.0 * nn_mean) if nn_mean > 0 else float("inf")
    print(f"  Nearest-neighbor: mean={nn_mean:.4f}, min={nn_min:.4f}, "
          f"effective Nyquist k_max≈{k_nyquist:.1f}")

    # ── 輸出檔名（與 QR naming 對齊）─────────────────────
    t0 = f"{time_arr[0]:.0f}".replace(".", "p")
    t1 = f"{time_arr[-1]:.0f}".replace(".", "p")
    if args.tag:
        tag = args.tag
    else:
        # 對齊 QR 用的 si100 後綴；此處無 stride 概念，但仍標記時間步來源
        tag = f"K{K}_N{N}_t{t0}-{t1}_si100_seed{args.placement_seed}"

    json_path = out_dir / f"sensors_random_{tag}.json"
    npz_path = out_dir / f"sensors_random_{tag}_dns_values.npz"

    # ── 寫出 JSON ─────────────────────────────────────────
    meta = {
        "K": K,
        "resolution": f"{N}x{N}",
        "spatial_downsample_res": f"{N}x{N}",
        "spatial_downsample_stride": 1,
        "method": "uniform_random",
        "placement_seed": int(args.placement_seed),
        "features": [],   # random 不用 feature matrix
        "time_stride": 1,
        "time_range": [float(time_arr[0]), float(time_arr[-1])],
        "time_steps": T,
        "selected_coordinates": coords.tolist(),
        "indices": [int(i) for i in indices],
        "source_file": str(dns_path),
        "dns_values_npz": str(npz_path),
        "sensor_dt": float(time_arr[1] - time_arr[0]) if T >= 2 else 0.0,
        "sensor_time_points": T,
        "nn_mean": nn_mean,
        "nn_min": nn_min,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # ── 寫出 NPZ ─────────────────────────────────────────
    np.savez(npz_path, time=time_arr, u=sensor_u, v=sensor_v)
    print(f"Saved NPZ:  {npz_path}")
    print(f"  u shape: {sensor_u.shape}, v shape: {sensor_v.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
