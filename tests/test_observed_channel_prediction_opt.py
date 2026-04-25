"""tests/test_observed_channel_prediction_opt.py

TDD for observed_channel_prediction single-pass optimization.

優化目標：
  原本：unique channel loop → 每個 channel 各呼叫一次 query_decoder（2 次）
  優化後：所有 N 個樣本（u+v 混合）一次呼叫 query_decoder，再向量化 normalize

驗收條件：
  1. 輸出形狀正確 [N]
  2. 與舊兩次呼叫路徑數值完全一致
  3. 單一 channel 情境下不崩潰
  4. per-channel normalize 正確（u 用 mean_u/std_u，v 用 mean_v/std_v）
"""

import torch
import pytest
from unittest.mock import patch, call
from src.lnn_kolmogorov import (
    LiquidOperator,
    observed_channel_prediction,
)

torch.manual_seed(0)

K  = 20   # sensors
T  = 5    # time steps
D  = 32   # d_model
N  = 40   # query samples


def _make_model() -> LiquidOperator:
    torch.manual_seed(42)
    return LiquidOperator(
        fourier_harmonics=4,
        sensor_value_dim=2,
        d_model=D,
        d_time=4,
        num_spatial_cfc_layers=1,
        num_temporal_cfc_layers=1,
    )


def _make_inputs(model, only_u=False):
    """共用 fixture：產生標準測試輸入。"""
    sv = torch.rand(T, K, 2)
    sp = torch.rand(K, 2)
    st = torch.linspace(0.0, 1.0, T)
    h_states, s_time = model.encode(sv, sp, re_norm=1.0, sensor_time=st)

    xy = torch.rand(N, 2)
    t_q = torch.rand(N)
    if only_u:
        c_obs = torch.zeros(N, dtype=torch.long)
    else:
        c_obs = torch.randint(0, 2, (N,))

    obs_names = ("u", "v")
    mean = torch.tensor([0.5, -0.3])
    std  = torch.tensor([1.2,  0.8])
    return xy, t_q, c_obs, obs_names, mean, std, h_states, s_time, sp


# ── 參考實作（舊兩次呼叫路徑）────────────────────────────────────────
def _reference_two_pass(net, xy, t_q, c_obs, obs_names, mean, std, h_states, s_time, sp):
    """舊邏輯的獨立複現，作為數值基準。"""
    preds = torch.empty_like(t_q)
    for obs_idx in torch.unique(c_obs).tolist():
        mask = c_obs == int(obs_idx)
        comp = torch.full((mask.sum(),), int(obs_idx), dtype=torch.long)
        raw = net.query_decoder(
            xy[mask], t_q[mask], comp, h_states, s_time, sp
        ).squeeze(1)
        preds[mask] = (raw - mean[int(obs_idx)]) / std[int(obs_idx)]
    return preds


# ── 1. 輸出形狀 ──────────────────────────────────────────────────────

class TestOutputShape:
    def test_shape_uv_mixed(self):
        model = _make_model()
        model.eval()
        xy, t_q, c_obs, names, mean, std, h, st, sp = _make_inputs(model)
        out = observed_channel_prediction(
            net=model, xy=xy, t_q=t_q, c_obs=c_obs,
            observed_channel_names=names,
            observed_channel_mean=mean, observed_channel_std=std,
            h_states=h, s_time=st, sensor_pos=sp,
        )
        assert out.shape == (N,), f"期望 ({N},)，得到 {out.shape}"

    def test_shape_only_u(self):
        model = _make_model()
        model.eval()
        xy, t_q, c_obs, names, mean, std, h, st, sp = _make_inputs(model, only_u=True)
        out = observed_channel_prediction(
            net=model, xy=xy, t_q=t_q, c_obs=c_obs,
            observed_channel_names=names,
            observed_channel_mean=mean, observed_channel_std=std,
            h_states=h, s_time=st, sensor_pos=sp,
        )
        assert out.shape == (N,)


# ── 2. 數值等同於舊兩次呼叫路徑 ─────────────────────────────────────

class TestNumericalEquivalence:
    def test_matches_two_pass_reference(self):
        model = _make_model()
        model.eval()
        xy, t_q, c_obs, names, mean, std, h, st, sp = _make_inputs(model)

        ref = _reference_two_pass(model, xy, t_q, c_obs, names, mean, std, h, st, sp)
        new = observed_channel_prediction(
            net=model, xy=xy, t_q=t_q, c_obs=c_obs,
            observed_channel_names=names,
            observed_channel_mean=mean, observed_channel_std=std,
            h_states=h, s_time=st, sensor_pos=sp,
        )
        assert torch.allclose(ref, new, atol=1e-5), \
            f"最大偏差: {(ref - new).abs().max().item():.2e}"

    def test_matches_two_pass_only_u(self):
        model = _make_model()
        model.eval()
        xy, t_q, c_obs, names, mean, std, h, st, sp = _make_inputs(model, only_u=True)

        ref = _reference_two_pass(model, xy, t_q, c_obs, names, mean, std, h, st, sp)
        new = observed_channel_prediction(
            net=model, xy=xy, t_q=t_q, c_obs=c_obs,
            observed_channel_names=names,
            observed_channel_mean=mean, observed_channel_std=std,
            h_states=h, s_time=st, sensor_pos=sp,
        )
        assert torch.allclose(ref, new, atol=1e-5)


# ── 3. query_decoder 只呼叫一次 ──────────────────────────────────────

class TestSingleDecoderCall:
    def test_query_decoder_called_once(self):
        """優化後 query_decoder 只應被呼叫一次，不論 channel 數。"""
        model = _make_model()
        model.eval()
        xy, t_q, c_obs, names, mean, std, h, st, sp = _make_inputs(model)

        with patch.object(
            model.query_decoder, "forward", wraps=model.query_decoder.forward
        ) as mock_fwd:
            observed_channel_prediction(
                net=model, xy=xy, t_q=t_q, c_obs=c_obs,
                observed_channel_names=names,
                observed_channel_mean=mean, observed_channel_std=std,
                h_states=h, s_time=st, sensor_pos=sp,
            )
            assert mock_fwd.call_count == 1, \
                f"query_decoder.forward 應只呼叫 1 次，實際 {mock_fwd.call_count} 次"


# ── 4. per-channel normalize 正確 ────────────────────────────────────

class TestPerChannelNormalization:
    def test_u_uses_u_mean_std(self):
        """所有 u 位置的輸出不應依賴 v 的 mean/std。"""
        model = _make_model()
        model.eval()
        xy, t_q, c_obs, names, mean, std, h, st, sp = _make_inputs(model)
        u_mask = c_obs == 0

        # 正常 normalize
        out_normal = observed_channel_prediction(
            net=model, xy=xy, t_q=t_q, c_obs=c_obs,
            observed_channel_names=names,
            observed_channel_mean=mean, observed_channel_std=std,
            h_states=h, s_time=st, sensor_pos=sp,
        )

        # 改變 v 的 mean/std，u 輸出不應改變
        mean_alt = mean.clone()
        std_alt  = std.clone()
        mean_alt[1] = 999.0
        std_alt[1]  = 999.0
        out_alt = observed_channel_prediction(
            net=model, xy=xy, t_q=t_q, c_obs=c_obs,
            observed_channel_names=names,
            observed_channel_mean=mean_alt, observed_channel_std=std_alt,
            h_states=h, s_time=st, sensor_pos=sp,
        )
        assert torch.allclose(out_normal[u_mask], out_alt[u_mask], atol=1e-5), \
            "改變 v 的 mean/std 不應影響 u 的預測"
