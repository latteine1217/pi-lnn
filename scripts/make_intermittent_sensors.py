"""Make an intermittent (randomly-dropped) sensor time series for the CfC-vs-vanilla test.

What:
    從 201-frame baseline sensor 檔隨機丟棄 fraction p 的時間 frame，產生「間斷」時序
    sensor 檔（時間戳非等距）。用於測 CfC（連續時間、per-step 真實 Δt）對 vanilla
    DeepONet（branch 只吃 co-temporal snapshot → gap 時 zero-order-hold）的差異。

Why this design:
    - 隨機 dropout 是 irregular-time-series 文獻（Latent-ODE / CfC missing-data）標準協議。
    - t=0 一律保留（CfC latent 起點 / t_start 定義）；其餘 index 依 seed 隨機丟。
    - dropped / retained 索引寫進 json，供 scheme-B 評估把誤差拆成 gap-time vs seen-time。
    - **mask 由 --mask-seed 決定**：B3 與 B0 必須用同一 seed 才是公平對照。

Contract（對齊 src/kolmogorov_dataset.py）:
    npz: time [T] (axis 0), u/v [K, T] (axis 1) — 對 retained 索引取子集。
    非等距時間戳由 CfC per-step dt（sensor_time 逐差）與 B0 searchsorted 原生支援。

Usage:
    uv run python scripts/make_intermittent_sensors.py \
        --sensor-json data/.../sensors_..._T50standalone.json \
        --dropout-frac 0.5 --mask-seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _subset_time_axis(arr: np.ndarray, t_len: int, keep: np.ndarray) -> np.ndarray:
    if arr.ndim == 1 and arr.shape[0] == t_len:
        return arr[keep]
    if arr.ndim == 2 and arr.shape[1] == t_len:
        return arr[:, keep]
    raise ValueError(f"無法辨識時間軸：shape={arr.shape}, 期望某軸長度=={t_len}。")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sensor-json", required=True, type=Path)
    ap.add_argument("--dropout-frac", required=True, type=float,
                    help="丟棄比例 p ∈ (0,1)；保留約 (1-p)·(T-1)+1 個 frame（t=0 一律留）")
    ap.add_argument("--mask-seed", type=int, default=0,
                    help="dropout mask 的隨機種子；B3 與 B0 必須用同一 seed")
    args = ap.parse_args()

    p = args.dropout_frac
    if not (0.0 < p < 1.0):
        raise ValueError(f"dropout-frac 必須 ∈ (0,1)，得到 {p}")

    src_json_path: Path = args.sensor_json.resolve()
    with open(src_json_path, encoding="utf-8") as f:
        meta = json.load(f)

    repo_root = Path(__file__).resolve().parent.parent
    npz_rel = meta["dns_values_npz"]
    src_npz_path = (repo_root / npz_rel).resolve()
    if not src_npz_path.exists():
        src_npz_path = (src_json_path.parent / Path(npz_rel).name).resolve()
    npz = np.load(src_npz_path, allow_pickle=True)
    t_len = int(npz["time"].shape[0])

    # ── 建 dropout mask：保留 index 0，其餘依 seed 隨機丟 p 比例 ──────────
    rng = np.random.default_rng(args.mask_seed)
    candidates = np.arange(1, t_len)                       # 可丟的 index（不含 0）
    n_drop = int(round(p * candidates.size))
    dropped = np.sort(rng.choice(candidates, size=n_drop, replace=False))
    keep = np.setdiff1d(np.arange(t_len), dropped)         # 已排序、含 0
    keep = np.sort(keep)

    new_arrays = {k: _subset_time_axis(npz[k], t_len, keep) for k in npz.files}
    new_time = new_arrays["time"].astype(np.float64)
    new_len = int(new_time.shape[0])

    # ── 輸出檔名：..._drop{pp}s{seed} ─────────────────────────────────
    tag = f"drop{int(round(p * 100)):02d}s{args.mask_seed}"
    stem = src_json_path.stem
    out_json_path = src_json_path.with_name(f"{stem}_{tag}.json")
    out_npz_path = src_npz_path.with_name(f"{stem}_{tag}_dns_values.npz")

    np.savez(out_npz_path, **new_arrays)

    # gap 長度統計（連續被丟的 run 長度，以 baseline Δt 為單位）
    if dropped.size:
        runs, cur = [], 1
        for a, b in zip(dropped[:-1], dropped[1:]):
            if b == a + 1:
                cur += 1
            else:
                runs.append(cur); cur = 1
        runs.append(cur)
        max_run, mean_run = int(max(runs)), float(np.mean(runs))
    else:
        max_run, mean_run = 0, 0.0

    new_meta = dict(meta)
    new_meta["time_steps"] = new_len
    new_meta["sensor_time_points"] = new_len
    new_meta["sensor_dt"] = None  # 非等距，dt 逐 step 由 sensor_time 決定
    new_meta["dns_values_npz"] = str(out_npz_path.relative_to(repo_root))
    new_meta["intermittent"] = {
        "mode": "random_dropout",
        "dropout_frac": p,
        "mask_seed": args.mask_seed,
        "retained_frame_idx": keep.tolist(),      # scheme-B: seen times
        "dropped_frame_idx": dropped.tolist(),    # scheme-B: gap times
        "baseline_frames": t_len,
        "max_gap_run": max_run,                   # 最長連續 gap（× baseline Δt=0.025）
        "mean_gap_run": mean_run,
        "note": ("軸: intermittent sensor availability. B3 與 B0 需用相同 mask_seed 對照。"
                 " 評估用 scheme-B：query 落完整 DNS，按 retained/dropped 拆誤差。"),
    }
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(new_meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] p={p:.2f} seed={args.mask_seed}: {t_len} → {new_len} retained "
          f"({dropped.size} dropped); max_gap={max_run}×Δt, mean_gap={mean_run:.2f}×Δt")
    print(f"     npz : {out_npz_path.relative_to(repo_root)}")
    print(f"     json: {out_json_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
