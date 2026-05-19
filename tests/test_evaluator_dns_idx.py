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

# find_dns_time_idx 已抽到 src/pi_con/dns_align.py 共用 module（Round 4 B1 修補）。
# 兩 evaluator 都從這個 module import，避免 silent drift。
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pi_con import find_dns_time_idx


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


# ── 4. 對精確對齊的 sensor，與 argmin 一致 ────────────────────────────
# Why: floor 與 argmin 對任意 sensor_time 不必等價（floor 不偷看未來，argmin 取最近）。
#      但對 sensor_time ⊆ dns_time（精確對齊）兩者必須一致 — 此為實務常見 invariant
#      （sensor 從 DNS 取出，sensor_time 是 dns_time 的子集合）。

class TestFindDnsTimeIdxMatchesArgminOnAlignedSensors:
    def test_aligned_sensor_subset(self):
        """sensor_time = dns_time 子集合時，floor 與 argmin 必相同。"""
        sensor_time = DNS_TIME[::2]   # 對齊子集合：[0, 1, 2, 3]
        for t_val in sensor_time:
            expected = int(np.argmin(np.abs(DNS_TIME - float(t_val))))
            got = find_dns_time_idx(DNS_TIME, float(t_val))
            assert got == expected, (
                f"t_val={t_val:.4f}: argmin={expected}, find_dns_time_idx={got}"
            )

    def test_misaligned_sensor_uses_floor_not_argmin(self):
        """對 misaligned sensor，floor 必為 ≤ t_val（不偷看未來），即使 argmin 取最近。

        Tolerance: 10% × median(dt) = 0.05；t=0.306 距離 [0.0, 0.5] 鄰居 ≥ 0.194 → 走 floor。
        """
        assert find_dns_time_idx(DNS_TIME, 0.306) == 0
        assert int(np.argmin(np.abs(DNS_TIME - 0.306))) == 1


# ── 5. Production invariant: f32 sensor_time ─────────────────────────
# Why: dataset 把 npz["time"] cast 成 float32（src/kolmogorov_dataset.py:90,
#      src/cylinder_dataset.py:185），但 DNS time array 維持 float64。
#      f32 round-trip 後，許多 sensor_time 值略小於對應 f64 dns_time
#      (e.g. 0.05 in f64 → 0.04999999... in f32)。
#      純 floor 會把這些精確對齊的 sensor_time 誤 floor 到「前一幀」 →
#      evaluator sanity check (diff > 0.5·dns_dt) RuntimeError，無法跑 production 資料。
#      Two-tier (nearest with tolerance) 必須吸收這個 ulp 偏差。

class TestFindDnsTimeIdxF32SensorTime:
    def test_f32_aligned_sensor_must_not_floor_to_previous_frame(self):
        """sensor_time = dns_time.astype(f32)：每個 i 必對應 idx i（不能 off-by-one）。"""
        dns_f64 = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], dtype=np.float64)
        sensor_f32 = dns_f64.astype(np.float32)
        for i, t in enumerate(sensor_f32):
            got = find_dns_time_idx(dns_f64, float(t))
            assert got == i, (
                f"i={i}: t_f32={float(t):.10f} dns[{i}]={dns_f64[i]:.10f}, "
                f"got idx={got} → off-by-one due to f32 precision"
            )

    def test_f32_dense_grid_kolmogorov_like(self):
        """Kolmogorov-like：T=101 點均勻 [0, 5]，sensor f32 應對齊到自身 idx。"""
        dns_f64 = np.linspace(0.0, 5.0, 101, dtype=np.float64)
        sensor_f32 = dns_f64.astype(np.float32)
        for i, t in enumerate(sensor_f32):
            got = find_dns_time_idx(dns_f64, float(t))
            assert got == i, f"i={i}, t_f32={t:.10f}, got={got}"

    def test_f32_sanity_diff_bounded(self):
        """f32 ulp 偏差必須 < 0.5·dns_dt（evaluator sanity 不會誤 raise）。"""
        dns_f64 = np.linspace(0.0, 5.0, 101, dtype=np.float64)
        dns_dt = float(np.median(np.diff(dns_f64)))
        sensor_f32 = dns_f64.astype(np.float32)
        for t in sensor_f32:
            idx = find_dns_time_idx(dns_f64, float(t))
            assert abs(float(dns_f64[idx]) - float(t)) <= 0.5 * dns_dt, (
                f"t_f32={t:.10f} idx={idx} dns[idx]={dns_f64[idx]:.10f} "
                f"diff={abs(float(dns_f64[idx]) - float(t)):.2e} > 0.5*dt={0.5*dns_dt:.2e}"
            )


# ── 6. Cross-evaluator drift safety ──────────────────────────────────
# Why: 之前兩 evaluator 字面重複實作 find_dns_time_idx；現已抽到 pi_con.dns_align。
#      此 test 確保兩 evaluator 都從共用 module import（不再 inline 實作），
#      未來改一邊不改另一邊不會 silent drift。

class TestSharedModuleNoDrift:
    def test_evaluators_import_from_pi_con(self):
        """兩 evaluator 必須 import find_dns_time_idx from pi_con（不能 inline 實作）。"""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        for fname in ("evaluate_cylinder.py", "evaluate_deeponet_cfc.py"):
            text = (scripts_dir / fname).read_text(encoding="utf-8")
            # 檢查 import 存在（即從 pi_con 拿，符合 single source）
            assert "find_dns_time_idx" in text, f"{fname}: 找不到 find_dns_time_idx 引用"
            assert (
                "from pi_con import" in text and "find_dns_time_idx" in text
            ), f"{fname}: 必須 from pi_con import find_dns_time_idx"
            # 檢查沒有 inline 重新定義（會 silent drift）
            assert "def find_dns_time_idx" not in text, (
                f"{fname}: 不能 inline 重新定義 find_dns_time_idx — drift 風險，"
                f"應從 pi_con 共用 module import"
            )
