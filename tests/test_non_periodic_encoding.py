"""tests/test_non_periodic_encoding.py

驗證非週期域 (use_periodic_domain=False) 的位置編碼行為。

驗收條件：
  1. FourierEmbs 對 x=0 與 x=L 產生**不同**輸出（與真 RFF 行為一致）。
  2. LearnableFourierEmb 對 x=0 與 x=L 產生**相同**輸出（週期 BC，原行為保留）。
  3. SpatialSetEncoder(use_periodic_domain=False, fourier_embed_dim>0) 走 FourierEmbs 路徑。
  4. DeepONetCfCDecoder(use_periodic_domain=False, fourier_embed_dim>0) 走 FourierEmbs 路徑。
  5. use_periodic_domain=False 且 fourier_embed_dim==0 應 raise ValueError。
  6. 週期路徑 (use_periodic_domain=True) 行為與重構前完全一致（已存在 ckpt 可載入）。
"""
from __future__ import annotations

import pytest
import torch

from pi_lnn import (
    DeepONetCfCDecoder,
    FourierEmbs,
    LearnableFourierEmb,
    SpatialSetEncoder,
)


D_MODEL = 32
EMBED_DIM = 64
DOMAIN_L = 1.0


# ── 1. FourierEmbs 對域邊界可區分 ────────────────────────────────────


class TestFourierEmbsBoundary:
    def test_endpoints_distinguishable(self):
        torch.manual_seed(0)
        emb = FourierEmbs(EMBED_DIM, input_dim=2, init_sigma=2.0)
        emb.eval()
        xy_0 = torch.tensor([[0.0, 0.3]])
        xy_L = torch.tensor([[1.0, 0.3]])
        with torch.no_grad():
            e0 = emb(xy_0)
            eL = emb(xy_L)
        diff = (e0 - eL).abs().max().item()
        assert diff > 0.5, f"FourierEmbs 應對 x=0/x=L 產生顯著差異，實得 {diff:.3e}"

    def test_output_shape(self):
        emb = FourierEmbs(EMBED_DIM, input_dim=2)
        out = emb(torch.rand(10, 2))
        assert out.shape == (10, EMBED_DIM)

    def test_domain_length_arg_ignored(self):
        torch.manual_seed(0)
        emb = FourierEmbs(EMBED_DIM, input_dim=2)
        emb.eval()
        x = torch.rand(5, 2)
        with torch.no_grad():
            assert torch.allclose(emb(x), emb(x, domain_length=1.0))
            assert torch.allclose(emb(x), emb(x, domain_length=10.0))


# ── 2. LearnableFourierEmb 維持週期行為（回歸保護）─────────────────


class TestLearnableFourierEmbStillPeriodic:
    def test_endpoints_collapse(self):
        torch.manual_seed(0)
        emb = LearnableFourierEmb(EMBED_DIM)
        emb.eval()
        xy_0 = torch.tensor([[0.0, 0.3]])
        xy_L = torch.tensor([[1.0, 0.3]])
        with torch.no_grad():
            e0 = emb(xy_0, DOMAIN_L)
            eL = emb(xy_L, DOMAIN_L)
        diff = (e0 - eL).abs().max().item()
        assert diff < 1e-5, (
            f"LearnableFourierEmb 應對 x=0/x=L 恆等（週期 BC），實得 diff={diff:.3e}"
        )


# ── 3. SpatialSetEncoder dispatch ─────────────────────────────────


class TestSpatialSetEncoderDispatch:
    def test_periodic_uses_learnable(self):
        enc = SpatialSetEncoder(
            fourier_harmonics=8,
            sensor_value_dim=2,
            d_model=D_MODEL,
            num_layers=1,
            domain_length=DOMAIN_L,
            fourier_embed_dim=EMBED_DIM,
            use_periodic_domain=True,
        )
        assert isinstance(enc.spatial_emb, LearnableFourierEmb)

    def test_non_periodic_uses_fourier(self):
        enc = SpatialSetEncoder(
            fourier_harmonics=8,
            sensor_value_dim=2,
            d_model=D_MODEL,
            num_layers=1,
            domain_length=DOMAIN_L,
            fourier_embed_dim=EMBED_DIM,
            use_periodic_domain=False,
        )
        assert isinstance(enc.spatial_emb, FourierEmbs)

    def test_non_periodic_endpoint_encoding_distinct(self):
        torch.manual_seed(0)
        enc = SpatialSetEncoder(
            fourier_harmonics=8,
            sensor_value_dim=2,
            d_model=D_MODEL,
            num_layers=1,
            domain_length=DOMAIN_L,
            fourier_embed_dim=EMBED_DIM,
            use_periodic_domain=False,
        )
        enc.eval()
        sp = torch.tensor([[0.0, 0.5], [1.0, 0.5]])
        with torch.no_grad():
            pos_enc = enc.encode_pos(sp)
        assert (pos_enc[0] - pos_enc[1]).abs().max().item() > 0.5

    def test_non_periodic_requires_embed_dim(self):
        with pytest.raises(ValueError, match="fourier_embed_dim>0"):
            SpatialSetEncoder(
                fourier_harmonics=8,
                sensor_value_dim=2,
                d_model=D_MODEL,
                num_layers=1,
                fourier_embed_dim=0,
                use_periodic_domain=False,
            )


# ── 4. DeepONetCfCDecoder dispatch ────────────────────────────────


class TestDecoderDispatch:
    def _make(self, use_periodic_domain: bool, fourier_embed_dim: int = EMBED_DIM):
        return DeepONetCfCDecoder(
            fourier_harmonics=8,
            d_model=D_MODEL,
            d_time=4,
            domain_length=DOMAIN_L,
            query_mlp_hidden_dim=64,
            fourier_embed_dim=fourier_embed_dim,
            use_periodic_domain=use_periodic_domain,
        )

    def test_periodic_uses_learnable(self):
        dec = self._make(use_periodic_domain=True)
        assert isinstance(dec.spatial_emb, LearnableFourierEmb)

    def test_non_periodic_uses_fourier(self):
        dec = self._make(use_periodic_domain=False)
        assert isinstance(dec.spatial_emb, FourierEmbs)

    def test_non_periodic_zero_embed_raises(self):
        with pytest.raises(ValueError, match="fourier_embed_dim>0"):
            self._make(use_periodic_domain=False, fourier_embed_dim=0)


# ── 5. 週期回歸測試（確保 Kolmogorov 既有 ckpt 仍可載入）──────────


class TestPeriodicBackwardCompat:
    """週期路徑 (use_periodic_domain=True, fourier_embed_dim=128) 的
    state_dict 鍵名與 shape 必須與重構前一致。"""

    def test_state_dict_keys_unchanged(self):
        enc = SpatialSetEncoder(
            fourier_harmonics=8,
            sensor_value_dim=2,
            d_model=D_MODEL,
            num_layers=1,
            domain_length=DOMAIN_L,
            fourier_embed_dim=EMBED_DIM,
            use_periodic_domain=True,
        )
        sd = enc.state_dict()
        assert "spatial_emb.proj.weight" in sd, (
            "重構不應改變週期路徑的 state_dict 鍵名（會破壞既有 ckpt 載入）"
        )
        assert sd["spatial_emb.proj.weight"].shape == (EMBED_DIM // 2, 4)
