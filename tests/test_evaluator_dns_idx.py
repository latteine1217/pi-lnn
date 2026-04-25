"""tests/test_evaluator_dns_idx.py

TDD RED→GREEN for find_dns_time_idx (A7).

問題：evaluate_deeponet_cfc.py 每個時間步做 O(n) argmin，
      且每次迭代重新 .astype(float64) 建立臨時陣列。

語義：floor（回傳最近 ≤ t_val 的 dns_time index），而非 argmin（最近鄰）。
Why: floor 語義保證只取「已發生」的 DNS 幀，物理上更正確；
     searchsorted 實作 O(log n)，省去 O(n) argmin 的掃描。

驗收條件：
  1. 精確命中：t_val 恰好在 dns_time 中時，回傳正確 index
  2. 內插（floor）：t_val 在兩個 dns_time 之間時，回傳最近 ≤ t_val 的 index
  3. 下邊界：t_val < dns_time[0] 時，clamp 到 0
  4. 上邊界：t_val >= dns_time[-1] 時，回傳最後一個 index
  5. 對實際對齊的 sensor_time，結果與舊版 argmin 一致
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# A7：從 evaluator 匯入待實作的函式
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from evaluate_deeponet_cfc import find_dns_time_idx


# ── 共用 fixture ──────────────────────────────────────────────────────

DNS_TIME = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float64)


# ── 1. 精確命中 ───────────────────────────────────────────────────────

class TestFindDnsTimeIdxExact:
    def test_first_element(self):
        assert find_dns_time_idx(DNS_TIME, 0.0) == 0

    def test_middle_element(self):
        assert find_dns_time_idx(DNS_TIME, 1.5) == 3

    def test_last_element(self):
        assert find_dns_time_idx(DNS_TIME, 3.0) == 6


# ── 2. 內插（floor）──────────────────────────────────────────────────

class TestFindDnsTimeIdxFloor:
    def test_between_first_and_second(self):
        # t=0.3 → floor → index 0 (dns_time[0]=0.0)
        assert find_dns_time_idx(DNS_TIME, 0.3) == 0

    def test_between_middle(self):
        # t=1.2 → floor → index 2 (dns_time[2]=1.0)
        assert find_dns_time_idx(DNS_TIME, 1.2) == 2

    def test_just_below_boundary(self):
        # t=0.499 → floor → index 0
        assert find_dns_time_idx(DNS_TIME, 0.499) == 0


# ── 3. 邊界條件 ───────────────────────────────────────────────────────

class TestFindDnsTimeIdxBoundary:
    def test_below_all(self):
        assert find_dns_time_idx(DNS_TIME, -1.0) == 0

    def test_above_all(self):
        assert find_dns_time_idx(DNS_TIME, 10.0) == len(DNS_TIME) - 1

    def test_single_element_array(self):
        t_arr = np.array([1.0], dtype=np.float64)
        assert find_dns_time_idx(t_arr, 1.0) == 0
        assert find_dns_time_idx(t_arr, 0.5) == 0
        assert find_dns_time_idx(t_arr, 2.0) == 0


# ── 4. 與原始 argmin 方案一致 ────────────────────────────────────────

class TestFindDnsTimeIdxMatchesArgmin:
    def test_matches_argmin_for_all_sensor_times(self):
        """對 sensor_time 的每個值，結果應與舊版 argmin 完全一致。"""
        sensor_time = np.linspace(0.0, 3.0, 50, dtype=np.float32)
        dns_time_f64 = DNS_TIME

        for t_val in sensor_time:
            expected = int(np.argmin(np.abs(dns_time_f64 - float(t_val))))
            got = find_dns_time_idx(dns_time_f64, float(t_val))
            assert got == expected, (
                f"t_val={t_val:.4f}: argmin={expected}, find_dns_time_idx={got}"
            )
