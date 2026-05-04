"""tests/test_al_multiplier.py

AL multiplier 數值正確性測試。

涵蓋 spec v4 §3 修正紀錄的 BUG-1 / BUG-3 regression：
- BUG-1: clamp_() no-op → 改 out-of-place clamp
- BUG-3: step=0 fire 由 caller guard（class 自身仍可在 step=0 update，由 caller 決定）
- clip range (0, Λ) — 負 λ 物理上無意義
- EMA 行為（首次 vs 後續）
- step=0 caller guard 契約
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pi_lnn import AugmentedLagrangianMultiplier


def test_loss_term_formula() -> None:
    """λ·C + (ρ/2)·C² — gradient 只流過 C。"""
    al = AugmentedLagrangianMultiplier(init_lambda=2.0, rho=3.0)
    C = torch.tensor(0.5, requires_grad=True)
    loss = al.loss_term(C)
    # 2.0·0.5 + (3.0/2)·0.25 = 1.0 + 0.375 = 1.375
    assert abs(loss.item() - 1.375) < 1e-6
    loss.backward()
    # ∂loss/∂C = λ + ρ·C = 2.0 + 3.0·0.5 = 3.5
    assert abs(C.grad.item() - 3.5) < 1e-6


def test_clip_lower_bound_is_zero() -> None:
    """v4 修正：clip 範圍 (0, Λ)，不是 (-Λ, +Λ)。

    init_lambda=0, ρ=1, C=0 → λ 應為 0；負 λ 不可能（C ≥ 0 + clamp(0, ...)）。
    """
    al = AugmentedLagrangianMultiplier(init_lambda=0.0, rho=1.0, lambda_clip=10.0, ema_momentum=0.0)
    al.update(torch.tensor(0.0))
    assert al.lambda_.item() == 0.0


def test_clip_upper_bound_clamps() -> None:
    """v4 BUG-1 regression：clamp 必須真的 work（v1 in-place on temp 失效）。"""
    al = AugmentedLagrangianMultiplier(init_lambda=9.0, rho=1.0, lambda_clip=10.0, ema_momentum=0.0)
    # 9.0 + 1.0 * 5.0 = 14 → clamp to 10
    al.update(torch.tensor(5.0))
    assert al.lambda_.item() == 10.0
    # 即使再 update 大 C，λ 不超過 10
    al.update(torch.tensor(100.0))
    assert al.lambda_.item() == 10.0


def test_ema_first_call_no_smoothing() -> None:
    """首次 update：_initialized=False → ema_C 直接 copy C，不做 EMA 平滑。"""
    al = AugmentedLagrangianMultiplier(ema_momentum=0.9)
    assert not bool(al._initialized.item())
    al.update(torch.tensor(0.5))
    assert abs(al.ema_C.item() - 0.5) < 1e-6  # 不是 0.9*0 + 0.1*0.5 = 0.05
    assert bool(al._initialized.item())


def test_ema_subsequent_calls_smooth() -> None:
    """第二次起：_initialized=True → 標準 EMA new = m·old + (1-m)·c。"""
    al = AugmentedLagrangianMultiplier(ema_momentum=0.5)
    al.update(torch.tensor(1.0))   # ema_C = 1.0
    al.update(torch.tensor(0.5))   # ema_C = 0.5*1.0 + 0.5*0.5 = 0.75
    assert abs(al.ema_C.item() - 0.75) < 1e-6


def test_ema_zero_momentum_no_smoothing() -> None:
    """ema_momentum=0：每次都用最新 batch C，不平滑。"""
    al = AugmentedLagrangianMultiplier(ema_momentum=0.0)
    al.update(torch.tensor(1.0))
    al.update(torch.tensor(0.3))
    assert abs(al.ema_C.item() - 0.3) < 1e-6  # 直接覆寫，不是 EMA


def test_lambda_buffer_is_not_parameter() -> None:
    """λ 必須是 buffer 不是 Parameter（避免 optimizer state 污染）。"""
    al = AugmentedLagrangianMultiplier()
    param_names = [n for n, _ in al.named_parameters()]
    assert param_names == [], f"應該沒有 Parameter，得到 {param_names}"
    buffer_names = [n for n, _ in al.named_buffers()]
    assert "lambda_" in buffer_names
    assert "ema_C" in buffer_names
    assert "_initialized" in buffer_names


def test_loss_term_lambda_no_grad() -> None:
    """gradient 只流過 C，不流過 λ（λ 是 buffer）。"""
    al = AugmentedLagrangianMultiplier(init_lambda=1.0, rho=1.0)
    C = torch.tensor(0.5, requires_grad=True)
    loss = al.loss_term(C)
    loss.backward()
    # λ 不該有 grad attribute（buffer），確認 update() 不依賴 autograd
    assert not al.lambda_.requires_grad


def test_step_zero_caller_guard_pattern() -> None:
    """spec §4 規定：caller 必須 step > 0 and step % freq == 0 才 update。

    本測試模擬 caller 邏輯，驗證 al_cont 本身在被呼叫時行為正確。
    """
    al = AugmentedLagrangianMultiplier()
    # 模擬 caller：step=0 不該呼叫 update
    for step in range(5):
        if step > 0 and step % 2 == 0:
            al.update(torch.tensor(0.1 * step))
    # step=2 update(0.2), step=4 update(0.4) → 已 update 2 次
    assert bool(al._initialized.item())
    # 第一次 init (step=2): ema_C = 0.2，第二次 EMA (step=4): 0.5*0.2 + 0.5*0.4 = 0.3
    assert abs(al.ema_C.item() - 0.3) < 1e-6


def test_C_batch_shape_handling() -> None:
    """C_batch 接受 0-dim 或 [1] tensor；其他 shape 應 raise。"""
    al = AugmentedLagrangianMultiplier()
    # 0-dim
    al.update(torch.tensor(0.5))
    val_0d = al.ema_C.item()
    # [1] tensor — reshape(()) 應該 OK
    al2 = AugmentedLagrangianMultiplier()
    al2.update(torch.tensor([0.5]))
    assert abs(al2.ema_C.item() - val_0d) < 1e-6
    # [2] tensor — reshape(()) 自然 raise
    al3 = AugmentedLagrangianMultiplier()
    with pytest.raises(RuntimeError):
        al3.update(torch.tensor([0.5, 0.3]))
