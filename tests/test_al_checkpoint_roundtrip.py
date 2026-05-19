"""tests/test_al_checkpoint_roundtrip.py

驗證 spec v4 §3 BUG-2 regression：`_initialized` 必須在 state_dict round-trip
後保留，否則 resume 會 EMA cold-start。
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pi_con import AugmentedLagrangianMultiplier


def test_state_dict_contains_all_buffers() -> None:
    al = AugmentedLagrangianMultiplier()
    sd = al.state_dict()
    assert "lambda_" in sd
    assert "ema_C" in sd
    assert "_initialized" in sd


def test_round_trip_preserves_state() -> None:
    al = AugmentedLagrangianMultiplier(init_lambda=0.5, rho=2.0, lambda_clip=10.0, ema_momentum=0.5)
    al.update(torch.tensor(0.7))
    al.update(torch.tensor(0.3))
    expected_lambda = al.lambda_.item()
    expected_ema = al.ema_C.item()
    assert bool(al._initialized.item())

    al2 = AugmentedLagrangianMultiplier(init_lambda=0.0, rho=2.0, lambda_clip=10.0, ema_momentum=0.5)
    al2.load_state_dict(al.state_dict())
    assert abs(al2.lambda_.item() - expected_lambda) < 1e-9
    assert abs(al2.ema_C.item() - expected_ema) < 1e-9
    assert bool(al2._initialized.item())


def test_round_trip_via_torch_save_load() -> None:
    """End-to-end：torch.save → torch.load 也保持狀態。"""
    al = AugmentedLagrangianMultiplier()
    for c in [0.5, 0.3, 0.4, 0.2]:
        al.update(torch.tensor(c))
    expected_lambda = al.lambda_.item()
    expected_ema = al.ema_C.item()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(al.state_dict(), f.name)
        fpath = f.name

    al2 = AugmentedLagrangianMultiplier()
    al2.load_state_dict(torch.load(fpath, weights_only=False))
    assert abs(al2.lambda_.item() - expected_lambda) < 1e-9
    assert abs(al2.ema_C.item() - expected_ema) < 1e-9
    assert bool(al2._initialized.item())
    Path(fpath).unlink()


def test_resume_does_not_cold_start_ema() -> None:
    """v3/v4 修正核心：resume 後再 update，EMA 應沿用既有值，不重置。"""
    al = AugmentedLagrangianMultiplier(ema_momentum=0.5)
    al.update(torch.tensor(0.8))   # ema_C = 0.8
    al.update(torch.tensor(0.4))   # ema_C = 0.5*0.8 + 0.5*0.4 = 0.6
    expected_after_resume = 0.5 * 0.6 + 0.5 * 0.2  # 下一步若 C=0.2

    # round-trip
    al2 = AugmentedLagrangianMultiplier(ema_momentum=0.5)
    al2.load_state_dict(al.state_dict())
    al2.update(torch.tensor(0.2))
    # 若 cold-start，ema_C 會是 0.2（直接覆寫），而不是 0.4
    assert abs(al2.ema_C.item() - expected_after_resume) < 1e-6, (
        f"Expected EMA continuation {expected_after_resume}, got {al2.ema_C.item()}"
    )


def test_initialized_flag_serializes_as_bool_tensor() -> None:
    al = AugmentedLagrangianMultiplier()
    al.update(torch.tensor(0.5))
    sd = al.state_dict()
    assert sd["_initialized"].dtype == torch.bool
    assert bool(sd["_initialized"].item()) is True
