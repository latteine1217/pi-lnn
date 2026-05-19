"""tests/test_make_lnn_model_fn_cache.py

TDD for make_lnn_model_fn h_states caching (A4).

問題：LBFGS closure 中 data loss 路徑已呼叫 net.encode() 得到 _h/_st，
      但 physics 路徑的 make_lnn_model_fn 未傳入 h_states=_h，
      導致 make_lnn_model_fn 內部再呼叫一次 encode()。

驗收條件：
  1. 傳入 h_states 時，make_lnn_model_fn 不呼叫 net.encode()
  2. 不傳 h_states 時，make_lnn_model_fn 恰好呼叫一次 net.encode()
  3. 兩種路徑的模型輸出數值相同（h_states 正確被重用）
"""

import torch
import pytest
from unittest.mock import patch, call

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lnn_kolmogorov import LiquidOperator, make_lnn_model_fn

torch.manual_seed(0)

K = 16
T = 6
D = 32
N_PHYS = 8   # physics collocation points


def _make_model() -> LiquidOperator:
    torch.manual_seed(42)
    return LiquidOperator(
        fourier_harmonics=4, sensor_value_dim=2,
        d_model=D, d_time=4,
        num_spatial_cfc_layers=1, num_temporal_cfc_layers=1,
    )


def _standard_inputs():
    sv = torch.rand(T, K, 2)
    sp = torch.rand(K, 2)
    st = torch.linspace(0.0, 1.0, T)
    return sv, sp, st


# ── 1. h_states 提供時不重複 encode ──────────────────────────────────

class TestMakeLnnModelFnCachesHStates:
    def test_encode_not_called_when_h_states_provided(self):
        """當 h_states/s_time 由外部傳入時，內部不應再呼叫 net.encode()。"""
        model = _make_model()
        sv, sp, st = _standard_inputs()
        h_states, s_time = model.encode(sv, sp, re_norm=0.0, sensor_time=st)

        with patch.object(model, "encode", wraps=model.encode) as spy:
            make_lnn_model_fn(
                model, sv, sp, re_norm=0.0, sensor_time=st,
                device=torch.device("cpu"),
                h_states=h_states, s_time=s_time,
            )
            spy.assert_not_called()

    def test_encode_called_once_when_h_states_absent(self):
        """h_states 未提供時，內部應呼叫恰好一次 encode()。"""
        model = _make_model()
        sv, sp, st = _standard_inputs()

        with patch.object(model, "encode", wraps=model.encode) as spy:
            make_lnn_model_fn(
                model, sv, sp, re_norm=0.0, sensor_time=st,
                device=torch.device("cpu"),
            )
            assert spy.call_count == 1, f"期望 1 次 encode，實際 {spy.call_count} 次"


# ── 2. 兩種路徑輸出數值相同 ──────────────────────────────────────────

class TestMakeLnnModelFnOutputConsistency:
    def test_cached_vs_recomputed_h_states_give_same_output(self):
        """預先計算 h_states 傳入 vs 讓函式自行 encode，輸出應一致。"""
        model = _make_model()
        sv, sp, st = _standard_inputs()
        xyt = torch.rand(N_PHYS, 3)

        h_states, s_time = model.encode(sv, sp, re_norm=0.0, sensor_time=st)

        fn_cached = make_lnn_model_fn(
            model, sv, sp, re_norm=0.0, sensor_time=st,
            device=torch.device("cpu"),
            h_states=h_states, s_time=s_time,
        )
        fn_fresh = make_lnn_model_fn(
            model, sv, sp, re_norm=0.0, sensor_time=st,
            device=torch.device("cpu"),
        )

        out_cached = fn_cached(xyt, c=0)
        out_fresh  = fn_fresh(xyt, c=0)
        assert torch.allclose(out_cached, out_fresh, atol=1e-6), \
            "cached h_states 與重新 encode 應給出相同輸出"
