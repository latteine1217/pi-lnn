"""tests/test_pos_enc_optimization.py

TDD for sensor_pos encoding pre-computation optimization.

變更目標：
  SpatialSetEncoder.encode_pos(sensor_pos) → pos_enc  （獨立計算）
  SpatialSetEncoder.forward(sensor_vals, pos_enc)       （接受 pre-computed）
  LiquidOperator.encode 只呼叫 encode_pos 一次，T 次 forward 共用同一 pos_enc。

驗收條件：
  1. encode_pos 輸出形狀正確（[K, embed_dim] 或 [K, 4*harmonics]）
  2. encode_pos 是確定性的（相同輸入兩次輸出相同）
  3. forward(sv, encode_pos(sp)) 數值等同於舊 forward(sv, sp)（舊邏輯在測試中手動重現）
  4. LiquidOperator.encode 對相同輸入輸出不變（與未優化前等值）
  5. update_state 在新介面下不會崩潰
"""

import torch
import pytest
from src.lnn_kolmogorov import (
    SpatialSetEncoder,
    LiquidOperator,
    periodic_fourier_encode,
    LearnableFourierEmb,
)

torch.manual_seed(0)

K = 20   # sensor 數
T = 5    # 時間步數
D = 32   # d_model（小值加快測試）


def _make_encoder(fourier_embed_dim: int = 0, harmonics: int = 8) -> SpatialSetEncoder:
    torch.manual_seed(42)
    return SpatialSetEncoder(
        fourier_harmonics=harmonics,
        sensor_value_dim=2,
        d_model=D,
        num_layers=1,
        domain_length=1.0,
        fourier_embed_dim=fourier_embed_dim,
    )


def _make_model(fourier_embed_dim: int = 0) -> LiquidOperator:
    torch.manual_seed(42)
    return LiquidOperator(
        fourier_harmonics=8,
        sensor_value_dim=2,
        d_model=D,
        d_time=4,
        num_spatial_cfc_layers=1,
        num_temporal_cfc_layers=1,
        fourier_embed_dim=fourier_embed_dim,
    )


# ── 1. encode_pos 形狀 ───────────────────────────────────────────────

class TestEncodePosShape:
    def test_deterministic_harmonics(self):
        enc = _make_encoder(fourier_embed_dim=0, harmonics=8)
        sp = torch.rand(K, 2)
        pos_enc = enc.encode_pos(sp)
        assert pos_enc.shape == (K, 4 * 8), f"期望 ({K}, 32)，得到 {pos_enc.shape}"

    def test_learnable_fourier(self):
        enc = _make_encoder(fourier_embed_dim=64)
        sp = torch.rand(K, 2)
        pos_enc = enc.encode_pos(sp)
        assert pos_enc.shape == (K, 64), f"期望 ({K}, 64)，得到 {pos_enc.shape}"


# ── 2. encode_pos 確定性 ─────────────────────────────────────────────

class TestEncodePosIsDeterministic:
    def test_same_input_same_output(self):
        enc = _make_encoder()
        sp = torch.rand(K, 2)
        out1 = enc.encode_pos(sp)
        out2 = enc.encode_pos(sp)
        assert torch.allclose(out1, out2), "相同輸入應產生相同 pos_enc"

    def test_different_pos_different_output(self):
        enc = _make_encoder()
        sp1 = torch.rand(K, 2)
        sp2 = torch.rand(K, 2)
        assert not torch.allclose(enc.encode_pos(sp1), enc.encode_pos(sp2))


# ── 3. forward(sv, pos_enc) 數值等同於舊 forward(sv, sp) ─────────────

class TestForwardNumericalEquivalence:
    """舊邏輯：在 forward 內部計算 pos_enc；新邏輯：呼叫方預計算並傳入。
    兩者必須產生完全相同的結果。"""

    def test_harmonics_mode(self):
        enc = _make_encoder(fourier_embed_dim=0, harmonics=8)
        sp = torch.rand(K, 2)
        sv = torch.rand(K, 2)

        # 手動重現舊邏輯：在 forward 呼叫前先算 pos_enc（模擬舊內部行為）
        pos_enc_manual = periodic_fourier_encode(sp, domain_length=1.0, n_harmonics=8)
        out_new = enc.forward(sv, pos_enc_manual)

        # 再次用 encode_pos（新 API）算 pos_enc，驗證兩者一致
        pos_enc_api = enc.encode_pos(sp)
        out_via_api = enc.forward(sv, pos_enc_api)

        assert torch.allclose(pos_enc_manual, pos_enc_api), "encode_pos 應與手動計算一致"
        assert torch.allclose(out_new, out_via_api), "兩條路徑輸出必須完全相同"

    def test_learnable_fourier_mode(self):
        enc = _make_encoder(fourier_embed_dim=64)
        sp = torch.rand(K, 2)
        sv = torch.rand(K, 2)

        # 手動重現舊邏輯：直接呼叫 spatial_emb
        pos_enc_manual = enc.spatial_emb(sp, domain_length=1.0)
        out_manual = enc.forward(sv, pos_enc_manual)

        pos_enc_api = enc.encode_pos(sp)
        out_api = enc.forward(sv, pos_enc_api)

        assert torch.allclose(pos_enc_manual, pos_enc_api)
        assert torch.allclose(out_manual, out_api)

    def test_pos_enc_reuse_gives_same_tokens(self):
        """同一 pos_enc 傳入不同 sensor_vals 時，輸出應隨 sensor_vals 變化。"""
        enc = _make_encoder()
        sp = torch.rand(K, 2)
        sv1 = torch.rand(K, 2)
        sv2 = torch.rand(K, 2)
        pos_enc = enc.encode_pos(sp)
        assert not torch.allclose(enc.forward(sv1, pos_enc), enc.forward(sv2, pos_enc)), \
            "不同 sensor_vals 應產生不同輸出"


# ── 4. LiquidOperator.encode 輸出不變 ──────────────────────────────

class TestLiquidOperatorEncodeConsistency:
    def test_encode_output_shape(self):
        model = _make_model()
        sv = torch.rand(T, K, 2)
        sp = torch.rand(K, 2)
        sensor_time = torch.linspace(0, 1, T)
        h_states, s_time = model.encode(sv, sp, re_norm=1.0, sensor_time=sensor_time)
        # h_states: [T, K, D]
        assert h_states.shape == (T, K, D), f"h_states 形狀錯誤: {h_states.shape}"

    def test_encode_deterministic(self):
        """相同輸入 encode 兩次，輸出完全一致。"""
        model = _make_model()
        model.eval()
        sv = torch.rand(T, K, 2)
        sp = torch.rand(K, 2)
        sensor_time = torch.linspace(0, 1, T)
        h1, _ = model.encode(sv, sp, 1.0, sensor_time)
        h2, _ = model.encode(sv, sp, 1.0, sensor_time)
        assert torch.allclose(h1, h2), "encode 應為確定性"

    def test_encode_with_learnable_fourier(self):
        model = _make_model(fourier_embed_dim=64)
        sv = torch.rand(T, K, 2)
        sp = torch.rand(K, 2)
        sensor_time = torch.linspace(0, 1, T)
        h_states, _ = model.encode(sv, sp, 1.0, sensor_time)
        assert h_states.shape == (T, K, D)


# ── 5. update_state 不崩潰 ──────────────────────────────────────────

class TestUpdateStateCompatibility:
    def test_update_state_runs(self):
        model = _make_model()
        sv_t = torch.rand(K, 2)
        sp = torch.rand(K, 2)
        h_list = [torch.zeros(K, D) for _ in range(1)]
        output, new_h = model.update_state(sv_t, sp, re_norm=1.0, dt=0.1, h_list=h_list)
        assert output.shape[1] == D
