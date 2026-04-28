"""tests/test_cfc_tau_init.py

#3 — CfC log_tau init range 配置化測試。

驗收條件：
  1. 預設 (-1, 1) 維持向後相容（log τ 邊界值 = ±1.0）
  2. 自訂範圍正確套用至所有 cells（包含 backward_cells）
  3. 不合法輸入（min >= max）即時報錯
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pi_lnn.blocks import CfCCell
from pi_lnn.encoders import TemporalCfCEncoder
from pi_lnn.operator import create_lnn_model


def test_cfc_cell_default_tau_backward_compat():
    """預設 (-1, 1) 維持與舊行為一致的 log τ 邊界。"""
    cell = CfCCell(input_size=8, hidden_size=16)
    assert cell.log_tau_a[0].item() == pytest.approx(-1.0, abs=1e-6)
    assert cell.log_tau_a[-1].item() == pytest.approx(1.0, abs=1e-6)


def test_cfc_cell_custom_tau_range():
    """自訂 (-3, 1.5) 應產出正確的 linspace 邊界。"""
    cell = CfCCell(input_size=8, hidden_size=32, log_tau_min=-3.0, log_tau_max=1.5)
    assert cell.log_tau_a[0].item() == pytest.approx(-3.0, abs=1e-6)
    assert cell.log_tau_a[-1].item() == pytest.approx(1.5, abs=1e-6)
    # linspace 應單調遞增
    diffs = cell.log_tau_a[1:] - cell.log_tau_a[:-1]
    assert (diffs > 0).all()


def test_cfc_cell_invalid_range_raises():
    """min >= max 必須立即報錯（fail fast）。"""
    with pytest.raises(ValueError, match="log_tau_min 必須 < log_tau_max"):
        CfCCell(input_size=4, hidden_size=8, log_tau_min=1.0, log_tau_max=1.0)
    with pytest.raises(ValueError, match="log_tau_min 必須 < log_tau_max"):
        CfCCell(input_size=4, hidden_size=8, log_tau_min=2.0, log_tau_max=1.0)


def test_temporal_encoder_propagates_tau_to_all_cells():
    """TemporalCfCEncoder 應將 tau 範圍傳遞至所有 forward + backward cells。"""
    enc = TemporalCfCEncoder(
        d_model=16,
        num_layers=2,
        num_token_attention_layers=0,
        use_bidirectional=True,
        cfc_log_tau_min=-3.0,
        cfc_log_tau_max=1.5,
    )
    for cell in list(enc.cells) + list(enc.backward_cells):
        assert cell.log_tau_a[0].item() == pytest.approx(-3.0, abs=1e-6)
        assert cell.log_tau_a[-1].item() == pytest.approx(1.5, abs=1e-6)


def test_create_lnn_model_propagates_tau_from_config():
    """create_lnn_model 應從 config 讀取 tau 範圍並套用。"""
    cfg = {
        "fourier_harmonics": 4,
        "d_model": 16,
        "d_time": 8,
        "num_spatial_cfc_layers": 1,
        "num_temporal_cfc_layers": 1,
        "operator_rank": 16,
        "cfc_log_tau_min": -2.5,
        "cfc_log_tau_max": 1.2,
    }
    model = create_lnn_model(cfg)
    for cell in model.temporal_encoder.cells:
        assert cell.log_tau_a[0].item() == pytest.approx(-2.5, abs=1e-6)
        assert cell.log_tau_a[-1].item() == pytest.approx(1.2, abs=1e-6)


def test_create_lnn_model_default_tau_unchanged():
    """未提供 tau 時應退回預設 (-1, 1)，保持既有實驗重現性。"""
    cfg = {
        "fourier_harmonics": 4,
        "d_model": 8,
        "d_time": 4,
        "num_spatial_cfc_layers": 1,
        "num_temporal_cfc_layers": 1,
        "operator_rank": 8,
    }
    model = create_lnn_model(cfg)
    cell = model.temporal_encoder.cells[0]
    assert cell.log_tau_a[0].item() == pytest.approx(-1.0, abs=1e-6)
    assert cell.log_tau_a[-1].item() == pytest.approx(1.0, abs=1e-6)


def test_cfc_cell_forward_unchanged_with_default_tau():
    """預設 tau 下 forward 數值應與不傳遞 tau 參數時完全一致。"""
    torch.manual_seed(42)
    cell_a = CfCCell(input_size=4, hidden_size=8)
    torch.manual_seed(42)
    cell_b = CfCCell(input_size=4, hidden_size=8, log_tau_min=-1.0, log_tau_max=1.0)
    x = torch.randn(2, 4)
    h = torch.randn(2, 8)
    out_a = cell_a(x, h, dt=0.1)
    out_b = cell_b(x, h, dt=0.1)
    assert torch.allclose(out_a, out_b, atol=1e-7)
