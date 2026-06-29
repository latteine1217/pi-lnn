"""tests/test_re_film_identity.py — Re FiLM identity-init + backward compat regression.

What:
    驗證 encoders.py use_re_film flag 的三個關鍵 invariant:
    1. use_re_film=False 路徑: state_dict 與舊版完全等價（保留 self.re_proj，無 re_gamma/re_beta key）
    2. use_re_film=True 路徑: state_dict 不再含 re_proj，改為 per-layer
       re_gamma_proj.{l}.weight/bias 與 re_beta_proj.{l}.weight/bias
    3. Identity init 數值等價: use_re_film=True forward/step 與 use_re_film=False 「移除 re_bias 加法」
       後在初始權重下數值相同（容忍 fp32 噪聲）。
       γ_init = 1, β_init = 0 → γ⊙x + β = x，等價於 no-modulation。

Why:
    Multi-Re 訓練要從 additive bias 升級為 multiplicative FiLM (γ_l, β_l)，但不能
    破壞既有 EXP-245 / EXP-094 baseline checkpoint。identity init 保證打開 flag 後
    初始 forward 行為與「移除舊 re_bias 加法」等價，給訓練從乾淨起點開始學 γ, β。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from picon_kolmogorov import SpatialSetEncoder, TemporalCfCEncoder  # noqa: E402


D = 32          # d_model
NUM_LAYERS = 3  # 多層才能驗 per-layer init 正確
K = 8           # sensors
T = 5           # time steps
RE_NORM = 1.2   # 任意 normalized Re


# ─────────────────────── TemporalCfCEncoder ───────────────────────

def _make_temporal(use_re_film: bool, seed: int = 42) -> TemporalCfCEncoder:
    torch.manual_seed(seed)
    return TemporalCfCEncoder(
        d_model=D,
        num_layers=NUM_LAYERS,
        num_token_attention_layers=0,   # 關掉 attention 隔離 FiLM 影響
        token_attention_heads=4,
        use_bidirectional=False,
        use_re_film=use_re_film,
    )


def test_temporal_no_film_state_dict_unchanged():
    """use_re_film=False 時 state_dict 必須與舊版完全一致：含 re_proj、不含 re_gamma/re_beta。"""
    enc = _make_temporal(use_re_film=False)
    keys = set(enc.state_dict().keys())
    assert any("re_proj" in k for k in keys), "舊路徑必須保留 re_proj"
    assert not any("re_gamma_proj" in k for k in keys), \
        "use_re_film=False 不應建立 re_gamma_proj"
    assert not any("re_beta_proj" in k for k in keys), \
        "use_re_film=False 不應建立 re_beta_proj"


def test_temporal_film_state_dict_has_per_layer_film():
    """use_re_film=True 時 state_dict 必須有 num_layers 份 (γ_l, β_l) projection。"""
    enc = _make_temporal(use_re_film=True)
    keys = set(enc.state_dict().keys())
    assert not any("re_proj" in k for k in keys), "FiLM 路徑不應建立舊 re_proj"
    for l in range(NUM_LAYERS):
        assert f"re_gamma_proj.{l}.weight" in keys, f"missing re_gamma_proj.{l}.weight"
        assert f"re_gamma_proj.{l}.bias"   in keys, f"missing re_gamma_proj.{l}.bias"
        assert f"re_beta_proj.{l}.weight"  in keys, f"missing re_beta_proj.{l}.weight"
        assert f"re_beta_proj.{l}.bias"    in keys, f"missing re_beta_proj.{l}.bias"


def test_temporal_film_identity_init_values():
    """γ_init: weight=0, bias=1 → γ(re)=1; β_init: weight=0, bias=0 → β(re)=0."""
    enc = _make_temporal(use_re_film=True)
    for layer_idx in range(NUM_LAYERS):
        gamma, beta = enc._re_film_pair(layer_idx, RE_NORM, torch.device("cpu"), torch.float32)
        assert torch.allclose(gamma, torch.ones(D)), f"layer {layer_idx} γ_init 應 ≡ 1"
        assert torch.allclose(beta,  torch.zeros(D)), f"layer {layer_idx} β_init 應 ≡ 0"


def _sync_common_weights(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    """把 src state_dict 中與 dst 共有的 key 複製到 dst。

    Why: 兩個不同 use_re_film 的 encoder 即使同 seed，建構過程多/少建立 FiLM modules
         會偷掉不同數量的 RNG，導致共同模組（CfC、attention、MLP）init 不同。
         同步共同 weights 後才能驗證 FiLM 本身的數值等價性。
    """
    src_sd, dst_sd = src.state_dict(), dst.state_dict()
    common = set(src_sd) & set(dst_sd)
    new_sd = {k: (src_sd[k] if k in common else v) for k, v in dst_sd.items()}
    dst.load_state_dict(new_sd)


def test_temporal_film_forward_equiv_to_no_bias_at_init():
    """Identity init 下：use_re_film=True forward 等價於 use_re_film=False 但 re_bias 強制歸零。

    驗證方式：兩個 encoder 同 seed 構造 → 同步共同 weights → 將 legacy 的 re_proj 歸零
    使 re_bias=0 → 兩者 forward 應在 fp32 容忍內相同。
    """
    enc_film  = _make_temporal(use_re_film=True,  seed=42)
    enc_legacy = _make_temporal(use_re_film=False, seed=42)
    _sync_common_weights(enc_film, enc_legacy)   # 共同 weights 對齊
    # 強制讓 legacy 的 re_bias = 0，使其行為等價於 no-modulation
    with torch.no_grad():
        enc_legacy.re_proj.weight.zero_()
        enc_legacy.re_proj.bias.zero_()

    torch.manual_seed(0)
    spatial_states = torch.randn(T, K, D)
    sensor_time = torch.linspace(0.0, 1.0, T)

    with torch.no_grad():
        out_film   = enc_film(spatial_states.clone(), RE_NORM, sensor_time.clone())
        out_legacy = enc_legacy(spatial_states.clone(), RE_NORM, sensor_time.clone())

    # 兩者除 attention（已關掉）與 re modulation 外完全相同 CfC weights
    # → identity FiLM 等價於 zero additive bias
    assert torch.allclose(out_film, out_legacy, atol=1e-6, rtol=1e-5), (
        f"identity init 下 FiLM forward 與 zero-bias legacy 不一致；"
        f"max abs diff = {(out_film - out_legacy).abs().max().item():.3e}"
    )


def test_temporal_film_step_equiv_to_no_bias_at_init():
    """同 forward 等價性，但驗 autoregressive step() 路徑。"""
    enc_film   = _make_temporal(use_re_film=True,  seed=42)
    enc_legacy = _make_temporal(use_re_film=False, seed=42)
    _sync_common_weights(enc_film, enc_legacy)
    with torch.no_grad():
        enc_legacy.re_proj.weight.zero_()
        enc_legacy.re_proj.bias.zero_()

    torch.manual_seed(0)
    spatial_state = torch.randn(K, D)
    h_list_film   = enc_film.init_hidden(K, torch.device("cpu"), torch.float32)
    h_list_legacy = enc_legacy.init_hidden(K, torch.device("cpu"), torch.float32)

    with torch.no_grad():
        h_film,   _ = enc_film.step(spatial_state.clone(),   h_list_film,   RE_NORM, dt=0.1)
        h_legacy, _ = enc_legacy.step(spatial_state.clone(), h_list_legacy, RE_NORM, dt=0.1)

    assert torch.allclose(h_film, h_legacy, atol=1e-6, rtol=1e-5), (
        f"step() identity-init 下不一致；max diff = {(h_film - h_legacy).abs().max().item():.3e}"
    )


# ─────────────────────── SpatialSetEncoder ───────────────────────

def _make_spatial(use_re_film: bool, seed: int = 42) -> SpatialSetEncoder:
    torch.manual_seed(seed)
    return SpatialSetEncoder(
        fourier_harmonics=4,
        sensor_value_dim=2,
        d_model=D,
        num_layers=NUM_LAYERS,
        domain_length=1.0,
        fourier_embed_dim=0,
        use_periodic_domain=True,
        use_re_film=use_re_film,
    )


def test_spatial_no_film_state_dict_unchanged():
    """SpatialSetEncoder use_re_film=False 不建立 FiLM 模組。"""
    enc = _make_spatial(use_re_film=False)
    keys = set(enc.state_dict().keys())
    assert not any("re_gamma_proj" in k for k in keys)
    assert not any("re_beta_proj"  in k for k in keys)


def test_spatial_film_state_dict_has_per_block_film():
    """SpatialSetEncoder use_re_film=True 必有 num_layers 份 per-block (γ_l, β_l)。"""
    enc = _make_spatial(use_re_film=True)
    keys = set(enc.state_dict().keys())
    for l in range(NUM_LAYERS):
        assert f"re_gamma_proj.{l}.weight" in keys
        assert f"re_beta_proj.{l}.weight"  in keys


def test_spatial_film_identity_forward():
    """Identity init: γ=1, β=0 下，FiLM 路徑 forward 應與 use_re_film=False 同 seed 完全相同。

    與 temporal 不同：spatial 沒有 additive bias，identity-init FiLM 直接 ≡ 不過 FiLM。
    所以兩個 encoder 在 identity init 下應 forward 完全一致（fp32 容忍內）。
    """
    enc_film   = _make_spatial(use_re_film=True,  seed=42)
    enc_legacy = _make_spatial(use_re_film=False, seed=42)
    _sync_common_weights(enc_film, enc_legacy)

    torch.manual_seed(0)
    sensor_vals = torch.randn(T, K, 2)
    sensor_pos  = torch.rand(K, 2)
    pos_enc_film   = enc_film.encode_pos(sensor_pos)
    pos_enc_legacy = enc_legacy.encode_pos(sensor_pos)

    with torch.no_grad():
        out_film   = enc_film(sensor_vals.clone(),   pos_enc_film.clone(),   re_norm=RE_NORM)
        out_legacy = enc_legacy(sensor_vals.clone(), pos_enc_legacy.clone(), re_norm=None)

    assert torch.allclose(out_film, out_legacy, atol=1e-6, rtol=1e-5), (
        f"spatial identity-init FiLM 應 ≡ no-FiLM forward；"
        f"max diff = {(out_film - out_legacy).abs().max().item():.3e}"
    )


def test_spatial_re_norm_none_safe_when_film_on():
    """use_re_film=True 但 re_norm=None 時不應 crash；行為 fallback 為 no-modulation。"""
    enc = _make_spatial(use_re_film=True)
    sensor_vals = torch.randn(K, 2)
    pos_enc = enc.encode_pos(torch.rand(K, 2))
    # 不應 raise
    out = enc(sensor_vals, pos_enc, re_norm=None)
    assert out.shape == (K, D)
