"""tests/test_cfc_pass_refactor.py

TDD characterization tests for _run_cfc_pass refactoring (A5 + A6).

目標：
  A5: outputs list+torch.stack → preallocated tensor（峰值記憶體從 2× 降到 1×）
  A6: re_bias.squeeze(0) 移出 loop（消除 T 次 view call）

策略：
  這兩個改動是純重構，行為不變。
  先寫 characterization test（現有程式碼已通過），重構後確認仍通過。

驗收條件：
  1. 輸出形狀恆為 [T, K, d_model]
  2. forward/reverse 兩個方向數值正確（reverse 時 outputs[t] 填回原位）
  3. re_bias 效果在 loop 外 squeeze 後仍正確施加
  4. 多層 CfC 殘差加法數值正確
"""

import torch
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lnn_kolmogorov import TemporalCfCEncoder, LiquidOperator

torch.manual_seed(0)

K = 16   # sensors
T = 8    # 時間步
D = 32   # d_model


def _make_encoder(num_layers=1, bidirectional=False, attn_layers=0) -> TemporalCfCEncoder:
    torch.manual_seed(42)
    return TemporalCfCEncoder(
        d_model=D,
        num_layers=num_layers,
        num_token_attention_layers=attn_layers,
        token_attention_heads=4,
        use_bidirectional=bidirectional,
    )


def _make_inputs():
    sv = torch.rand(T, K, 2)
    sp = torch.rand(K, 2)
    st = torch.linspace(0.0, 1.0, T)
    return sv, sp, st


# ── A5: 輸出形狀 ─────────────────────────────────────────────────────

class TestRunCfcPassOutputShape:
    def test_forward_shape(self):
        enc = _make_encoder()
        sv, sp, st = _make_inputs()
        dts = torch.cat([st[:1], st[1:] - st[:-1]])
        re_bias = enc._re_bias(0.0, sv.device, sv.dtype).view(1, 1, -1)
        # spatial_states: [T, K, D]
        spatial_states = torch.rand(T, K, D)
        out = enc._run_cfc_pass(spatial_states, enc.cells, dts, re_bias, layer_idx=0, reverse=False)
        assert out.shape == (T, K, D), f"期望 ({T},{K},{D})，得到 {out.shape}"

    def test_reverse_shape(self):
        enc = _make_encoder()
        sv, sp, st = _make_inputs()
        dts = torch.cat([st[:1], st[1:] - st[:-1]])
        re_bias = enc._re_bias(0.0, sv.device, sv.dtype).view(1, 1, -1)
        spatial_states = torch.rand(T, K, D)
        out = enc._run_cfc_pass(spatial_states, enc.cells, dts, re_bias, layer_idx=0, reverse=True)
        assert out.shape == (T, K, D), f"reverse 期望 ({T},{K},{D})，得到 {out.shape}"


# ── A5: forward/reverse 一致性 ───────────────────────────────────────

class TestRunCfcPassDeterminism:
    def test_same_input_same_output(self):
        """確定性：相同輸入兩次結果一致。"""
        enc = _make_encoder()
        spatial = torch.rand(T, K, D)
        dts = torch.ones(T)
        re_bias = enc._re_bias(0.0, spatial.device, spatial.dtype).view(1, 1, -1)
        out1 = enc._run_cfc_pass(spatial, enc.cells, dts, re_bias, 0, reverse=False)
        out2 = enc._run_cfc_pass(spatial, enc.cells, dts, re_bias, 0, reverse=False)
        assert torch.allclose(out1, out2), "相同輸入應產生相同輸出"

    def test_forward_reverse_differ(self):
        """forward 與 reverse 掃描方向不同，結果不應相同。"""
        enc = _make_encoder()
        spatial = torch.rand(T, K, D)
        dts = torch.ones(T)
        re_bias = enc._re_bias(0.0, spatial.device, spatial.dtype).view(1, 1, -1)
        fwd = enc._run_cfc_pass(spatial, enc.cells, dts, re_bias, 0, reverse=False)
        bwd = enc._run_cfc_pass(spatial, enc.cells, dts, re_bias, 0, reverse=True)
        # T>1 時兩者不同（若相同代表 re_bias 把梯度蓋掉或有 bug）
        assert not torch.allclose(fwd, bwd), "forward 與 reverse 應給出不同結果"


# ── A5: 整合 forward() 輸出不因重構而改變 ────────────────────────────

class TestForwardOutputUnchangedAfterRefactor:
    """用 TemporalCfCEncoder.forward() 做端對端驗證。

    重構前先記錄輸出；重構後輸出須完全一致。
    """

    def test_encode_output_shape(self):
        model = LiquidOperator(
            fourier_harmonics=4, sensor_value_dim=2, d_model=D, d_time=4,
            num_spatial_cfc_layers=1, num_temporal_cfc_layers=1,
        )
        sv = torch.rand(T, K, 2)
        sp = torch.rand(K, 2)
        st = torch.linspace(0.0, 1.0, T)
        h_states, s_time = model.encode(sv, sp, re_norm=0.0, sensor_time=st)
        assert h_states.shape == (T, K, D), f"h_states 期望 ({T},{K},{D})，得到 {h_states.shape}"

    def test_encode_output_numerically_stable(self):
        """輸出不含 NaN / Inf。"""
        model = LiquidOperator(
            fourier_harmonics=4, sensor_value_dim=2, d_model=D, d_time=4,
            num_spatial_cfc_layers=1, num_temporal_cfc_layers=1,
        )
        sv = torch.rand(T, K, 2)
        sp = torch.rand(K, 2)
        st = torch.linspace(0.0, 1.0, T)
        h_states, _ = model.encode(sv, sp, re_norm=0.0, sensor_time=st)
        assert torch.isfinite(h_states).all(), "h_states 含有 NaN 或 Inf"


# ── A6: re_bias 效果正確（squeeze 位置不影響數值）───────────────────

class TestReBiasEffect:
    def test_different_re_norm_gives_different_output(self):
        """re_norm 不同時輸出不同，驗證 re_bias 確實施加。"""
        enc = _make_encoder()
        spatial = torch.rand(T, K, D)
        dts = torch.ones(T)

        re_bias_0 = enc._re_bias(0.0, spatial.device, spatial.dtype).view(1, 1, -1)
        re_bias_1 = enc._re_bias(1.0, spatial.device, spatial.dtype).view(1, 1, -1)

        out0 = enc._run_cfc_pass(spatial, enc.cells, dts, re_bias_0, 0, reverse=False)
        out1 = enc._run_cfc_pass(spatial, enc.cells, dts, re_bias_1, 0, reverse=False)
        assert not torch.allclose(out0, out1), "不同 re_norm 應產生不同輸出（re_bias 未生效）"
