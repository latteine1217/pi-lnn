"""Regression guard：canonical 徑向能譜必須逐條滿足論文 §3 的譜定義。

每個測試對應 thesis/contents/chapter03.tex (subsec:spectrum_definition) 的一條
規定。`test_shell_uses_rounding_not_truncation` 專門釘住 2026-07-29 稽核在
`baseline_comparison_full.py` 發現的 bin 錯位，該錯誤能通過 Parseval 與 slope
檢查，只有逐點比對會顯形。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pi_con.spectral import parseval_residual, radial_energy_spectrum  # noqa: E402


def _unit_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """單位盒 [0,1)² 的座標網格，與 DNS 的 indexing='ij' 慣例一致。"""
    x = np.arange(n) / n
    return np.meshgrid(x, x, indexing="ij")


def test_single_mode_lands_in_its_own_shell():
    """u = cos(2πk₀x) 的能量必須全部落在 shell k₀（eq:radial_spectrum）。"""
    n, k0 = 64, 5
    xx, _ = _unit_grid(n)
    u = np.cos(2 * np.pi * k0 * xx)
    v = np.zeros_like(u)

    k, e = radial_energy_spectrum(u, v)

    assert e[k == k0].item() > 0
    leaked = e[k != k0].sum()
    assert leaked < 1e-20 * e.sum(), f"energy leaked outside shell k0: {leaked:.3e}"
    # cos 的能量分到 ±k0 兩個模，各 (1/2)²，總計 ½·2·¼ = ¼ = ½⟨u²⟩
    assert e[k == k0].item() == pytest.approx(0.5 * np.mean(u**2), rel=1e-12)


def test_mean_mode_is_excluded():
    """加一個均勻底流不得改變 E(k)：k=0 的 mean mode 被排除。"""
    n = 32
    rng = np.random.default_rng(0)
    u, v = rng.standard_normal((n, n)), rng.standard_normal((n, n))

    k_ref, e_ref = radial_energy_spectrum(u, v)
    k_off, e_off = radial_energy_spectrum(u + 3.7, v - 1.2)

    np.testing.assert_array_equal(k_ref, k_off)
    np.testing.assert_allclose(e_off, e_ref, rtol=0, atol=1e-15)


def test_shell_uses_rounding_not_truncation():
    """Shell 是 k-0.5 ≤ |k| < k+0.5，不是 k ≤ |k| < k+1。

    測試模必須落在兩種定義的差異區間 [k-0.5, k) 內，否則區分不了。
    取 (kx, ky) = (2, 3)：|k| = 3.606。論文的環帶歸 shell 4；截斷
    (`.astype(int)`，即 baseline_comparison_full.py 的 bug) 歸 shell 3。
    對照之下 (3, 3) 的 |k| = 4.243 落在兩者交集，兩種實作都歸 shell 4，
    無法作為 guard。
    """
    n = 64
    xx, yy = _unit_grid(n)
    u = np.cos(2 * np.pi * (2 * xx + 3 * yy))
    v = np.zeros_like(u)

    k, e = radial_energy_spectrum(u, v)

    kmag = np.sqrt(2.0**2 + 3.0**2)
    assert 3.5 <= kmag < 4.0, "測試前提：|k| 必須落在 round 與 truncate 的分歧區間"
    assert e[k == 4].item() > 0, "能量應落在 shell 4（半開環帶）"
    assert e[k == 3].item() == pytest.approx(0.0, abs=1e-20), (
        "能量落到 shell 3 表示 bin 索引用了截斷而非四捨五入"
    )


def test_corner_modes_are_discarded_not_folded():
    """|k| > n/2 的 corner mode 必須捨棄，不得折回末個 shell。"""
    n = 32
    xx, yy = _unit_grid(n)
    # (kx, ky) = (n/2, n/2) → |k| = 22.6 > n/2 = 16，屬 corner
    u = np.cos(2 * np.pi * ((n // 2) * xx + (n // 2) * yy))
    v = np.zeros_like(u)

    k, e = radial_energy_spectrum(u, v)

    assert k[-1] == n // 2
    assert e.sum() == pytest.approx(0.0, abs=1e-20), (
        "corner mode 的能量出現在譜中，表示被折回而非捨棄"
    )


def test_parseval_holds_for_band_limited_field():
    """能量集中的場上，ΣE(k)Δk = KE（eq:parseval）。"""
    n = 48
    xx, yy = _unit_grid(n)
    u = np.cos(2 * np.pi * 2 * yy) + 0.3 * np.sin(2 * np.pi * 3 * xx)
    v = 0.5 * np.cos(2 * np.pi * 1 * xx)

    assert parseval_residual(u, v) == pytest.approx(0.0, abs=1e-12)


def test_broadband_field_loses_only_corner_energy():
    """白噪音場的 Parseval 缺口為正且有界——即論文所述 corner 捨棄的代價。"""
    n = 64
    rng = np.random.default_rng(1)
    u, v = rng.standard_normal((n, n)), rng.standard_normal((n, n))

    residual = parseval_residual(u, v)
    # 方形網格內切圓外的面積佔比 1 - π/4 ≈ 21.5%，白噪音能量均勻分布故缺口趨近此值
    assert 0.15 < residual < 0.25, f"broadband Parseval 缺口異常: {residual:.4f}"


def test_wavenumbers_are_integer_cyclic():
    """回傳的波數是 1…n/2 的整數 cyclic 值，與 k_f 及 √(K/π) 同一個量。"""
    n = 40
    k, _ = radial_energy_spectrum(np.zeros((n, n)), np.zeros((n, n)))
    np.testing.assert_array_equal(k, np.arange(1, n // 2 + 1))


def test_inconsistent_dx_fails_loud():
    """傳錯 dx 會整體改變波數尺度，必須立刻中止而非默默算錯。"""
    n = 16
    z = np.zeros((n, n))

    radial_energy_spectrum(z, z, dx=1.0 / n)  # 正確值不得拋錯
    with pytest.raises(ValueError, match="unit-box convention"):
        radial_energy_spectrum(z, z, dx=1.0)


def test_rejects_non_square_input():
    with pytest.raises(ValueError, match="square 2D snapshot"):
        radial_energy_spectrum(np.zeros((8, 16)), np.zeros((8, 16)))
