"""tests/test_causal_weighting.py

#5 — PINN Causal weighting 單元測試。

驗收條件：
  1. eps=0 退化為均勻平均（向後相容兼回退路徑）
  2. eps>0 時權重隨時間單調遞減（因果性）
  3. 權重 detach（不通過 cumsum 反傳，避免梯度爆炸）
  4. 不合法輸入即時報錯
  5. 多殘差項共用同一 weight curve（time-only 因果）
  6. 空 bin 不污染 cumsum
  7. 退化情況：所有點同一時間 → 退回均勻平均
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pi_lnn.causal import causal_weighted_residual_loss


def _make_residuals(n: int, t_max: float = 1.0, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    times = torch.linspace(0.0, t_max, n)
    res_a = torch.randn(n, generator=g)
    res_b = torch.randn(n, generator=g)
    return [res_a, res_b], times


def test_eps_zero_recovers_uniform_mean():
    """eps=0 時所有 weights = 1，等同於均勻平均。"""
    residuals, times = _make_residuals(100)
    weighted, w_t = causal_weighted_residual_loss(
        residuals, times, num_bins=16, eps=0.0
    )
    # weights 全部 = 1（exp(-0 * 任何) = 1）
    assert torch.allclose(w_t, torch.ones_like(w_t))
    # 加權平均應接近 mean(r^2)（per-bin mean 後再均勻平均，與直接 mean 在同一量級）
    direct = torch.stack([r.pow(2).mean() for r in residuals])
    # bin 平均的近似誤差 ~10%
    assert torch.allclose(weighted, direct, rtol=0.15)


def test_weights_monotonic_decrease_with_eps():
    """eps>0 且早期殘差非零時，權重應單調遞減。"""
    times = torch.linspace(0.0, 1.0, 200)
    # 製造非零早期殘差
    res = torch.ones(200) * 0.5
    weighted, w_t = causal_weighted_residual_loss(
        [res], times, num_bins=16, eps=2.0
    )
    diffs = w_t[1:] - w_t[:-1]
    # 全部非正（容忍浮點誤差）
    assert (diffs <= 1e-7).all(), f"權重應遞減：{w_t.tolist()}"
    # 第一個權重應為 1.0（cumsum prev = 0）
    assert w_t[0].item() == pytest.approx(1.0, abs=1e-6)
    # 後段權重明顯小於前段
    assert w_t[-1].item() < w_t[0].item() * 0.5


def test_weights_are_detached():
    """權重不應將梯度反傳至殘差（防止 cumsum 鏈式爆炸）。"""
    times = torch.linspace(0.0, 1.0, 100)
    res = torch.randn(100, requires_grad=True)
    weighted, w_t = causal_weighted_residual_loss(
        [res], times, num_bins=8, eps=1.0
    )
    assert not w_t.requires_grad


def test_gradient_flows_through_residuals():
    """殘差自身應仍可接收梯度（驗證 detach 只影響 weights）。"""
    times = torch.linspace(0.0, 1.0, 100)
    res = torch.randn(100, requires_grad=True)
    weighted, _ = causal_weighted_residual_loss(
        [res], times, num_bins=8, eps=1.0
    )
    weighted.sum().backward()
    assert res.grad is not None
    assert res.grad.abs().sum().item() > 0


def test_invalid_num_bins_raises():
    times = torch.linspace(0, 1, 10)
    with pytest.raises(ValueError, match="num_bins 必須 >= 2"):
        causal_weighted_residual_loss([torch.randn(10)], times, num_bins=1, eps=1.0)


def test_invalid_eps_raises():
    times = torch.linspace(0, 1, 10)
    with pytest.raises(ValueError, match="eps 必須 >= 0"):
        causal_weighted_residual_loss([torch.randn(10)], times, num_bins=8, eps=-0.5)


def test_empty_residuals_raises():
    times = torch.linspace(0, 1, 10)
    with pytest.raises(ValueError, match="residuals 不可為空"):
        causal_weighted_residual_loss([], times, num_bins=8, eps=1.0)


def test_multiple_residuals_share_single_weight_curve():
    """多殘差項應共用同一 weight curve（因果是時間屬性）。"""
    times = torch.linspace(0, 1, 100)
    res_a = torch.randn(100) * 5.0  # 大殘差項
    res_b = torch.randn(100) * 0.1  # 小殘差項
    weighted, w_t = causal_weighted_residual_loss(
        [res_a, res_b], times, num_bins=16, eps=1.0
    )
    # 兩個殘差項都被同一 w_t 加權，意味 weighted[i] 與 res_i 比例關係相似
    # 不直接斷言這個比例，但驗證 weighted 不會把任一殘差變 0
    assert weighted[0].item() > 0
    assert weighted[1].item() > 0
    # weights 在邊界值正確
    assert w_t[0].item() == pytest.approx(1.0, abs=1e-6)


def test_degenerate_same_time_falls_back():
    """所有點同一時間時退回均勻 mean(r^2)。"""
    times = torch.full((50,), 0.5)
    res = torch.randn(50)
    weighted, w_t = causal_weighted_residual_loss(
        [res], times, num_bins=8, eps=1.0
    )
    direct = res.pow(2).mean()
    assert torch.allclose(weighted[0], direct, atol=1e-6)
    assert torch.allclose(w_t, torch.ones_like(w_t))


def test_high_eps_concentrates_on_early_bins():
    """eps 很大時權重幾乎全集中在 bin 0。"""
    times = torch.linspace(0, 1, 200)
    res = torch.ones(200) * 1.0  # 大早期殘差
    _, w_t = causal_weighted_residual_loss(
        [res], times, num_bins=16, eps=100.0
    )
    # bin 0 應主導
    assert w_t[0].item() > 0.99 * w_t.sum().item() / 16 * 16  # 量級檢查
    assert w_t[1].item() < w_t[0].item() * 0.01


def test_t_max_overrides_inferred_max():
    """顯式 t_max 應覆蓋 times.max() 推算（用於 time_marching 中限制 bin 範圍）。"""
    times = torch.linspace(0, 0.5, 100)  # 實際 max = 0.5
    res = torch.randn(100)
    # 強制 t_max=1.0 → bin 0..7 有點，bin 8..15 空
    weighted, w_t = causal_weighted_residual_loss(
        [res], times, num_bins=16, eps=0.0, t_max=1.0
    )
    assert w_t.shape == (16,)
    # eps=0 時權重應全為 1
    assert torch.allclose(w_t, torch.ones_like(w_t))


def test_empty_bins_do_not_inflate_cumsum():
    """空 bin 應貢獻 0 殘差，不污染 cumsum 權重。"""
    # 所有點集中在前半 (t in [0, 0.5])，t_max=1.0 → bin 0..8 有資料、bin 9..15 全空
    # （t=0.5 落在 bin 8 因 (0.5 * 16).long() = 8）
    times = torch.linspace(0, 0.5, 100)
    res = torch.ones(100) * 0.3
    _, w_t = causal_weighted_residual_loss(
        [res], times, num_bins=16, eps=2.0, t_max=1.0
    )
    # 前段（bin 0..8）有資料 → cumsum 遞增 → 權重單調遞減
    early_diffs = w_t[1:9] - w_t[:8]
    assert (early_diffs <= 1e-7).all()
    # 後段（bin 10..15）全空 → cumsum 不變 → 權重恆定
    late_diffs = (w_t[10:] - w_t[9:-1]).abs()
    assert (late_diffs < 1e-6).all(), f"後段權重應恆定：{w_t.tolist()}"
