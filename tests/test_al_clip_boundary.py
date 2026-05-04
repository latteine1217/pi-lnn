"""tests/test_al_clip_boundary.py

驗證 spec v4 §8 預期行為：
- λ 達 Λ 後不再增加（saturation）
- C → 0 時 λ 凍結（不會自我推高）
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pi_lnn import AugmentedLagrangianMultiplier


def test_lambda_saturates_at_clip() -> None:
    """λ 飽和 Λ 後，後續 update 不再增加。"""
    al = AugmentedLagrangianMultiplier(init_lambda=0.0, rho=2.0, lambda_clip=5.0, ema_momentum=0.0)
    # 重複大 C → λ 增到 clip
    for _ in range(20):
        al.update(torch.tensor(10.0))
    assert al.lambda_.item() == 5.0
    # 再 update → 仍 = 5.0
    al.update(torch.tensor(100.0))
    assert al.lambda_.item() == 5.0


def test_lambda_freezes_when_C_zero() -> None:
    """C → 0 時 λ 不再變動。"""
    al = AugmentedLagrangianMultiplier(init_lambda=2.0, rho=1.0, lambda_clip=10.0, ema_momentum=0.0)
    # 模擬 constraint 已收斂：C ≡ 0
    for _ in range(50):
        al.update(torch.tensor(0.0))
    # λ + ρ·0 = λ → 應維持 init=2.0
    assert abs(al.lambda_.item() - 2.0) < 1e-6


def test_lambda_monotone_non_decreasing_when_C_positive() -> None:
    """C > 0 → λ 單調非減（spec §2 framing 核心）。"""
    al = AugmentedLagrangianMultiplier(init_lambda=0.0, rho=1.0, lambda_clip=100.0, ema_momentum=0.0)
    history = [al.lambda_.item()]
    for c in [0.5, 0.3, 0.7, 0.2, 0.1, 0.4]:
        al.update(torch.tensor(c))
        history.append(al.lambda_.item())
    for i in range(1, len(history)):
        assert history[i] >= history[i - 1] - 1e-9, (
            f"λ 應該單調非減，第 {i} 步從 {history[i-1]} 變 {history[i]}"
        )


def test_negative_lambda_impossible() -> None:
    """init_lambda=0 + 任何非負 C → λ 永不為負（clip lower=0）。"""
    al = AugmentedLagrangianMultiplier(init_lambda=0.0, rho=1.0, lambda_clip=10.0, ema_momentum=0.0)
    for c in [0.0, 0.0, 0.5, 0.0, 0.3]:
        al.update(torch.tensor(c))
        assert al.lambda_.item() >= 0.0, f"λ 不該為負，得到 {al.lambda_.item()}"
