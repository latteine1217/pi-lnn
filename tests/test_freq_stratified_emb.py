"""tests/test_freq_stratified_emb.py

#6 — LearnableFourierEmb 頻率分層初始化測試。

驗收條件：
  1. 預設（無 bands）維持單一 σ 行為（向後相容）
  2. 多頻段 σ 套用到對應通道區段
  3. 各段 std 落在期望值 ±20% 容忍區間
  4. 不合法輸入即時報錯
  5. forward 行為與單頻段一致（不影響 sin/cos 結構）
  6. 通道分配總和正確（rounding edge case）
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pi_lnn.encodings import LearnableFourierEmb
from pi_lnn.operator import create_lnn_model


def _band_std(weight: torch.Tensor, start: int, end: int) -> float:
    return weight[start:end].std().item()


def test_default_single_sigma_backward_compat():
    """無 bands 參數時行為與舊版完全一致。"""
    torch.manual_seed(0)
    emb = LearnableFourierEmb(embed_dim=64, init_sigma=2.0)
    assert emb._band_layout is None
    # std 應接近 2.0（取樣誤差容忍）
    std = emb.proj.weight.std().item()
    assert 1.6 < std < 2.4


def test_three_band_layout_assigns_channels_correctly():
    """三段 bands 應正確分配通道並用對應 σ 初始化。"""
    torch.manual_seed(0)
    emb = LearnableFourierEmb(
        embed_dim=128,
        init_sigma_bands=(1.0, 4.0, 12.0),
        band_dim_ratios=(0.5, 0.375, 0.125),
    )
    half = 128 // 2  # = 64
    assert emb._band_layout is not None
    assert len(emb._band_layout) == 3
    assert sum(end - start for start, end, _ in emb._band_layout) == half
    # 通道分配：32, 24, 8
    starts_ends = [(s, e) for s, e, _ in emb._band_layout]
    assert starts_ends == [(0, 32), (32, 56), (56, 64)]
    # 各段 std 應接近對應 σ（24~32 樣本 std 估計誤差 ±20% 內）
    assert 0.7 < _band_std(emb.proj.weight, 0, 32) < 1.4
    assert 2.8 < _band_std(emb.proj.weight, 32, 56) < 5.6
    # 高頻段 8 樣本 std 估計誤差較大，放寬至 ±50%
    assert 6.0 < _band_std(emb.proj.weight, 56, 64) < 18.0


def test_partial_bands_raises():
    """只提供 sigma_bands 或 band_dim_ratios 其一應報錯。"""
    with pytest.raises(ValueError, match="必須同時提供或同時為 None"):
        LearnableFourierEmb(embed_dim=64, init_sigma_bands=(1.0, 4.0))
    with pytest.raises(ValueError, match="必須同時提供或同時為 None"):
        LearnableFourierEmb(embed_dim=64, band_dim_ratios=(0.5, 0.5))


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="長度需相同"):
        LearnableFourierEmb(
            embed_dim=64,
            init_sigma_bands=(1.0, 4.0),
            band_dim_ratios=(0.3, 0.3, 0.4),
        )


def test_ratio_sum_validation():
    with pytest.raises(ValueError, match="總和需 = 1.0"):
        LearnableFourierEmb(
            embed_dim=64,
            init_sigma_bands=(1.0, 4.0),
            band_dim_ratios=(0.3, 0.5),
        )


def test_invalid_sigma_or_ratio_raises():
    with pytest.raises(ValueError, match="所有 sigma 必須 > 0"):
        LearnableFourierEmb(
            embed_dim=64,
            init_sigma_bands=(1.0, -1.0),
            band_dim_ratios=(0.5, 0.5),
        )
    with pytest.raises(ValueError, match="所有 ratio 必須 > 0"):
        LearnableFourierEmb(
            embed_dim=64,
            init_sigma_bands=(1.0, 4.0),
            band_dim_ratios=(0.0, 1.0),
        )


def test_forward_shape_and_periodicity_preserved():
    """頻率分層不應破壞 sin/cos 結構：shape 正確且輸出範圍 [-1, 1]。"""
    emb = LearnableFourierEmb(
        embed_dim=64,
        init_sigma_bands=(1.0, 4.0, 12.0),
        band_dim_ratios=(0.5, 0.375, 0.125),
    )
    xy = torch.rand(10, 2)
    out = emb(xy, domain_length=1.0)
    assert out.shape == (10, 64)
    assert out.min() >= -1.0 - 1e-5 and out.max() <= 1.0 + 1e-5


def test_rounding_corner_case_dimension_sum():
    """通道分配在 rounding 後總和必須 = embed_dim/2。"""
    # embed_dim=10, half=5；ratios (0.4, 0.4, 0.2) → 2,2,1 = 5 ✓
    emb = LearnableFourierEmb(
        embed_dim=10,
        init_sigma_bands=(1.0, 2.0, 3.0),
        band_dim_ratios=(0.4, 0.4, 0.2),
    )
    half = 10 // 2
    assert sum(e - s for s, e, _ in emb._band_layout) == half


def test_create_lnn_model_propagates_bands():
    """create_lnn_model 應從 config 傳遞頻率分層參數至 spatial_emb。"""
    cfg = {
        "fourier_harmonics": 4,
        "d_model": 16,
        "d_time": 8,
        "num_spatial_cfc_layers": 1,
        "num_temporal_cfc_layers": 1,
        "operator_rank": 16,
        "fourier_embed_dim": 64,
        "fourier_sigma_bands": [1.0, 4.0, 12.0],
        "fourier_band_dim_ratios": [0.5, 0.375, 0.125],
    }
    model = create_lnn_model(cfg)
    spatial_emb = model.spatial_encoder.spatial_emb
    assert spatial_emb is not None
    assert spatial_emb._band_layout is not None
    assert len(spatial_emb._band_layout) == 3


def test_create_lnn_model_default_fourier_unchanged():
    """未提供 bands 時 spatial_emb 應走單頻段路徑。"""
    cfg = {
        "fourier_harmonics": 4,
        "d_model": 8,
        "d_time": 4,
        "num_spatial_cfc_layers": 1,
        "num_temporal_cfc_layers": 1,
        "operator_rank": 8,
        "fourier_embed_dim": 64,
    }
    model = create_lnn_model(cfg)
    spatial_emb = model.spatial_encoder.spatial_emb
    assert spatial_emb is not None
    assert spatial_emb._band_layout is None
