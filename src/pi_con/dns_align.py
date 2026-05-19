"""DNS time-axis alignment helper — single source of truth for evaluators.

What: 給定 sensor t_val，找對應的 DNS time index。
Why:  evaluator 必須跟訓練端用同一份 dns_time mapping；之前兩支 evaluator
      （evaluate_cylinder.py、evaluate_deeponet_cfc.py）字面重複實作此函式，
      未來改一邊不改另一邊會 silent drift。集中於此 module 確保單一真相。

Two-tier 對齊邏輯：
  Tier 1 (nearest with ULP tolerance)：吸收 f32→f64 round-trip 損失。
  Tier 2 (floor)：物理上不偷看未來，且 sanity check 友善。
詳見 find_dns_time_idx 的 docstring。
"""
from __future__ import annotations

import numpy as np


def find_dns_time_idx(dns_time: np.ndarray, t_val: float) -> int:
    """Two-tier DNS time alignment for a single sensor t_val.

    Tier 1 (nearest with ULP tolerance):
      若 floor / floor+1 兩個鄰居中最近的距離 ≤ tolerance → 回傳該 nearest。
      tolerance = max(100·spacing(f32(magnitude)), 1e-6)，吸收 f32 round-trip 損失。
    Tier 2 (floor):
      否則回傳「最近 ≤ t_val」的 dns_time index，物理上不偷看未來。
    邊界：t_val < dns_time[0] clamp 到 0；t_val ≥ dns_time[-1] 回傳 last。

    Why ULP-tolerance（不是 fraction of dt）:
      production dataset 把 sensor_time cast 成 float32（kolmogorov_dataset.py:90、
      cylinder_dataset.py:185），f32 值常略小於對應 f64 dns_time（如 0.05 → 0.04999999...）。
      f32 round-trip 損失 ≤ ulp(value)/2，~1e-7 級。
      用 max(100·ulp(t), 1e-6) 足夠吸收 cast 損失，但不會誤吃 0.001 級的真實 misalign
      （該情況保留走 floor）。**不可用 0.1×dt：sparse DNS dt=0.5 時會吃 0.05，
      把 t=0.499 誤對齊到 0.5 (偷看未來)**。

    Args:
        dns_time: DNS 時間軸，1D float-like array
        t_val:    sensor 時間點

    Returns:
        對應的 dns_time index（int）

    Raises:
        ValueError: dns_time 為空陣列。
    """
    t_arr = np.asarray(dns_time, dtype=np.float64).ravel()
    if t_arr.size == 0:
        raise ValueError("dns_time 為空陣列，無法對齊")
    t = float(t_val)
    n = t_arr.size

    # 純 floor（clamp 到 [0, n-1]）
    floor_idx = int(np.searchsorted(t_arr, t, side="right")) - 1
    floor_idx = max(0, min(floor_idx, n - 1))

    if n > 1:
        # 候選：floor 與其右鄰居（左鄰居距 ≥ floor，數學上不會更近）
        candidates = [floor_idx]
        if floor_idx + 1 < n:
            candidates.append(floor_idx + 1)
        nearest_idx = min(candidates, key=lambda i: abs(t_arr[i] - t))
        ref_mag = max(abs(t_arr[nearest_idx]), abs(t), 1.0)
        tolerance = max(100.0 * float(np.spacing(np.float32(ref_mag))), 1e-6)
        if abs(t_arr[nearest_idx] - t) <= tolerance:
            return int(nearest_idx)

    return int(floor_idx)
