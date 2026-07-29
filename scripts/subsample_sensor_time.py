"""Subsample a Kolmogorov sensor time-series file along the TIME axis.

What:
    產生「軸 A（時間取樣密度）」ablation 用的稀疏時序 sensor 檔。以 stride k 對
    時間軸做等距子取樣，輸出新的 sensor npz + json。

Why:
    主線 sensor 檔為 201 frames（Δt_store = 0.025, T = 5）——此 201 是 DNS 儲存
    cadence（save_interval=100, dt=2.5e-4）的副產品，從未被實驗檢驗。本腳本產生
    T = (201-1)/k + 1 frames 的對照檔，餵給訓練做「data 監督時間密度」的單變數 ablation。

Design invariants（對齊 src/kolmogorov_dataset.py 的資料合約）:
    - 只動時間軸。npz: time [T] (axis 0), u/v [K, T] (axis 1)。
    - sensor 空間位置（json.selected_coordinates）與 DNS npy 完全不動 → sensor axis
      convention（x/y）不受影響；評估仍對完整 DNS 201 frames。
    - 訓練端 dt_phys = sensor_time[1] - sensor_time[0]（假設等距）。stride 子取樣
      保持等距，故安全；本腳本額外 assert 子取樣後 time 仍 uniform。

Usage:
    uv run python scripts/subsample_sensor_time.py \
        --sensor-json data/.../sensors_..._T50standalone.json \
        --stride 2

    輸出 <name>_tsub<k>.json 與 <name>_tsub<k>_dns_values.npz 到同目錄。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _subsample_array(arr: np.ndarray, t_len: int, stride: int) -> np.ndarray:
    """對長度 == t_len 的時間軸做 [::stride] 子取樣。

    1-D（time）→ 沿唯一軸切；2-D（u/v: [K, T]）→ 沿 shape[1]==t_len 的軸切。
    找不到對應時間軸則 raise（fail loud，不 silent 傳回原陣列）。
    """
    if arr.ndim == 1 and arr.shape[0] == t_len:
        return arr[::stride]
    if arr.ndim == 2 and arr.shape[1] == t_len:
        return arr[:, ::stride]
    # 罕見：K == T 造成 axis 歧義，或未預期形狀 → 明確報錯
    raise ValueError(
        f"無法辨識時間軸：shape={arr.shape}, 期望某軸長度=={t_len}。"
        " 請檢查 sensor npz 結構是否為 time[T] / u,v[K,T]。"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sensor-json", required=True, type=Path,
                    help="來源 sensor JSON（含 dns_values_npz 欄位指向對應 npz）")
    ap.add_argument("--stride", required=True, type=int,
                    help="時間軸 stride k（建議整除 (T-1) 以保留頭尾端點）")
    ap.add_argument("--out-suffix", default=None,
                    help="輸出檔名後綴，預設 tsub<k>")
    args = ap.parse_args()

    stride = args.stride
    if stride < 1:
        raise ValueError(f"stride 必須 >= 1，得到 {stride}")
    suffix = args.out_suffix or f"tsub{stride}"

    src_json_path: Path = args.sensor_json.resolve()
    with open(src_json_path, encoding="utf-8") as f:
        meta = json.load(f)

    # ── 解析對應 npz 路徑 ────────────────────────────────────────────
    npz_rel = meta.get("dns_values_npz")
    if npz_rel is None:
        raise KeyError(f"{src_json_path} 缺少 dns_values_npz 欄位，無法定位 sensor npz。")
    # dns_values_npz 存的是 repo-root 相對路徑
    repo_root = Path(__file__).resolve().parent.parent
    src_npz_path = (repo_root / npz_rel).resolve()
    if not src_npz_path.exists():
        # fallback: 與 json 同目錄
        src_npz_path = src_json_path.parent / Path(npz_rel).name
    if not src_npz_path.exists():
        raise FileNotFoundError(f"找不到 sensor npz：{npz_rel}")

    npz = np.load(src_npz_path, allow_pickle=True)
    t_len = int(npz["time"].shape[0])
    if (t_len - 1) % stride != 0:
        print(f"[WARN] (T-1)={t_len - 1} 不被 stride={stride} 整除 → "
              f"末端點 t={npz['time'][-1]:.4f} 會被丟棄，窗長縮短。")

    new_arrays: dict[str, np.ndarray] = {}
    for key in npz.files:
        new_arrays[key] = _subsample_array(npz[key], t_len, stride)

    new_time = new_arrays["time"].astype(np.float64)
    new_len = int(new_time.shape[0])
    # 驗證等距（訓練端 dt_phys 假設 uniform）
    diffs = np.diff(new_time)
    if new_len >= 2:
        rel = float(np.max(np.abs(diffs - diffs[0])) / max(abs(diffs[0]), 1e-12))
        if rel > 1e-4:
            raise ValueError(f"子取樣後 time 非 uniform（max rel-diff {rel:.2e}）。")
    new_dt = float(diffs[0]) if new_len >= 2 else float(meta.get("sensor_dt", 0.0))

    # ── 輸出檔名 ─────────────────────────────────────────────────────
    # <stem>.json → <stem>_<suffix>.json ；<stem>_dns_values.npz → <stem>_<suffix>_dns_values.npz
    json_stem = src_json_path.stem  # 例：..._T50standalone
    out_json_path = src_json_path.with_name(f"{json_stem}_{suffix}.json")
    out_npz_path = src_npz_path.with_name(f"{json_stem}_{suffix}_dns_values.npz")

    np.savez(out_npz_path, **new_arrays)

    # ── 更新 json metadata ───────────────────────────────────────────
    new_meta = dict(meta)
    new_meta["time_stride"] = int(meta.get("time_stride", 1)) * stride
    new_meta["time_steps"] = new_len
    new_meta["sensor_time_points"] = new_len
    new_meta["sensor_dt"] = new_dt
    new_meta["dns_values_npz"] = str(out_npz_path.relative_to(repo_root))
    new_meta["time_subsample_note"] = (
        f"軸 A ablation：由 {src_npz_path.name} 沿時間軸 stride={stride} 子取樣，"
        f"{t_len} → {new_len} frames（Δt {meta.get('sensor_dt')} → {new_dt}）。"
        " 空間位置與 DNS npy 未動；評估仍對完整 DNS。"
    )
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(new_meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] stride={stride}: {t_len} → {new_len} frames, Δt={new_dt:.4f}, "
          f"t∈[{new_time[0]:.3f},{new_time[-1]:.3f}]")
    print(f"     npz : {out_npz_path.relative_to(repo_root)}  "
          f"({', '.join(f'{k}{new_arrays[k].shape}' for k in npz.files)})")
    print(f"     json: {out_json_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
