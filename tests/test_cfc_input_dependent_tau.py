"""tests/test_cfc_input_dependent_tau.py

input-dependent (liquid) time-constant 配置化測試。

對齊官方 CfC：原版 t_a = time_a(backbone) 為 input-dependent；本專案原實作把 τ
凍結成 static per-channel 參數。新增 cfc_input_dependent_tau flag 恢復液態 τ。

驗收條件：
  1. 預設 (False) 維持向後相容：無 time_a 屬性、forward 走 static 路徑。
  2. flag=True 但 time_a zero-init → forward 數值與 static 完全一致（平滑啟動）。
  3. time_a 非零時 input-dependent τ 真的改變輸出，且梯度可回傳。
  4. TemporalCfCEncoder / create_picon_model 正確傳遞 flag（含 backward_cells）。
  5. config DEFAULT 註冊新 key（否則 load_picon_config 會擋 unknown）。
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pi_con.blocks import CfCCell
from pi_con.config import DEFAULT_PICON_ARGS
from pi_con.encoders import TemporalCfCEncoder
from pi_con.operator import create_picon_model


def test_default_is_static_no_time_a():
    """預設 flag=False：不建 time_a，維持 static τ（向後相容）。"""
    cell = CfCCell(input_size=4, hidden_size=8)
    assert cell.input_dependent_tau is False
    assert not hasattr(cell, "time_a")


def test_zero_init_matches_static_forward():
    """flag=True 但 time_a zero-init → forward 與 static 路徑位元級一致。

    依賴 __init__ 中 ff1/ff2/log_tau_a/time_b 在 time_a 之前建構，故相同 seed 下
    這些權重一致；time_a zero-init 使 tanh(0)=0、log τ = log_tau_a，tau 完全相同。
    """
    torch.manual_seed(123)
    cell_static = CfCCell(input_size=4, hidden_size=8)
    torch.manual_seed(123)
    cell_liquid = CfCCell(input_size=4, hidden_size=8, input_dependent_tau=True)

    x = torch.randn(3, 4)
    h = torch.randn(3, 8)
    out_static = cell_static(x, h, dt=0.3)
    out_liquid = cell_liquid(x, h, dt=0.3)
    assert torch.allclose(out_static, out_liquid, atol=1e-7)


def test_input_dependent_tau_changes_output_when_active():
    """time_a 非零時，liquid 路徑輸出應 ≠ static 路徑（input-dependent τ 生效）。"""
    torch.manual_seed(0)
    cell = CfCCell(input_size=4, hidden_size=8, input_dependent_tau=True)
    nn.init.normal_(cell.time_a.weight, std=0.5)  # 打破 zero-init

    x = torch.randn(3, 4)
    h = torch.randn(3, 8)
    out_liquid = cell(x, h, dt=0.7)
    cell.input_dependent_tau = False  # 同權重改走 static
    out_static = cell(x, h, dt=0.7)
    assert not torch.allclose(out_liquid, out_static, atol=1e-5)


def test_input_dependent_tau_is_bounded():
    """tanh 有界性：即使 time_a 權重很大，τ 仍限制在 [τ0·e^-s, τ0·e^+s] 內，不爆。"""
    torch.manual_seed(1)
    scale = 2.0
    cell = CfCCell(input_size=4, hidden_size=8, input_dependent_tau=True, tau_mod_scale=scale)
    nn.init.normal_(cell.time_a.weight, std=50.0)  # 極端權重
    x = torch.randn(16, 4)
    h = torch.randn(16, 8)
    xh = torch.cat([x, h], dim=-1)
    log_tau = cell.log_tau_a + cell.tau_mod_scale * torch.tanh(cell.time_a(xh))
    lo = (cell.log_tau_a - scale).min().item()
    hi = (cell.log_tau_a + scale).max().item()
    assert log_tau.min().item() >= lo - 1e-5
    assert log_tau.max().item() <= hi + 1e-5
    assert torch.isfinite(log_tau).all()


def test_grad_flows_to_time_a():
    """梯度可回傳到 time_a（liquid τ 確實參與優化）。"""
    cell = CfCCell(input_size=4, hidden_size=8, input_dependent_tau=True)
    nn.init.normal_(cell.time_a.weight, std=0.1)
    x = torch.randn(3, 4)
    h = torch.randn(3, 8)
    loss = cell(x, h, dt=0.5).sum()
    loss.backward()
    assert cell.time_a.weight.grad is not None
    assert cell.time_a.weight.grad.abs().sum().item() > 0


def test_two_step_rollout_no_nan():
    """minimal repro：連續兩步 forward（不同 dt）數值有限、無 NaN。"""
    torch.manual_seed(0)
    cell = CfCCell(input_size=4, hidden_size=8, input_dependent_tau=True)
    nn.init.normal_(cell.time_a.weight, std=0.3)
    h = torch.zeros(2, 8)
    for dt in (0.1, 0.5):
        x = torch.randn(2, 4)
        h = cell(x, h, dt=dt)
        assert torch.isfinite(h).all()


def test_encoder_propagates_flag_to_all_cells():
    """TemporalCfCEncoder 應把 flag 傳給 forward + backward cells。"""
    enc = TemporalCfCEncoder(
        d_model=16,
        num_layers=2,
        num_token_attention_layers=0,
        use_bidirectional=True,
        cfc_input_dependent_tau=True,
    )
    for cell in list(enc.cells) + list(enc.backward_cells):
        assert cell.input_dependent_tau is True
        assert hasattr(cell, "time_a")


def test_encoder_default_static():
    """未指定時 encoder 維持 static τ（向後相容）。"""
    enc = TemporalCfCEncoder(d_model=16, num_layers=1, num_token_attention_layers=0)
    assert enc.cells[0].input_dependent_tau is False
    assert not hasattr(enc.cells[0], "time_a")


def test_create_model_reads_flag_from_config():
    """create_picon_model 應從 config 讀取 cfc_input_dependent_tau 並套用。"""
    cfg = {
        "fourier_harmonics": 4,
        "d_model": 16,
        "d_time": 8,
        "num_spatial_cfc_layers": 1,
        "num_temporal_cfc_layers": 1,
        "operator_rank": 16,
        "cfc_input_dependent_tau": True,
    }
    model = create_picon_model(cfg)
    for cell in model.temporal_encoder.cells:
        assert cell.input_dependent_tau is True
        assert hasattr(cell, "time_a")


def test_create_model_default_static_unchanged():
    """未提供 flag 時退回 static，保持既有實驗重現性。"""
    cfg = {
        "fourier_harmonics": 4,
        "d_model": 8,
        "d_time": 4,
        "num_spatial_cfc_layers": 1,
        "num_temporal_cfc_layers": 1,
        "operator_rank": 8,
    }
    model = create_picon_model(cfg)
    cell = model.temporal_encoder.cells[0]
    assert cell.input_dependent_tau is False
    assert not hasattr(cell, "time_a")


def test_config_default_registers_keys():
    """新 key 必須註冊在 DEFAULT_PICON_ARGS，否則 load_picon_config 擋 unknown。"""
    assert DEFAULT_PICON_ARGS["cfc_input_dependent_tau"] is False
    assert DEFAULT_PICON_ARGS["cfc_tau_mod_scale"] == pytest.approx(2.0)
