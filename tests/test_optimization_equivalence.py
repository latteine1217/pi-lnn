"""Equivalence tests for Batch A+B+C optimizations.

驗證新版（向量化 spatial encoder + attention 移出 CfC scan + decoder.forward_uvp）
與舊版（per-t loop + per-t attention + 3× decoder forward）數值等價。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pi_lnn import (  # noqa: E402
    DeepONetCfCDecoder,
    LiquidOperator,
    SpatialSetEncoder,
    TemporalCfCEncoder,
)
from pi_lnn.encodings import periodic_fourier_encode  # noqa: E402


def _build_decoder(rank: int = 32, hidden: int = 64) -> DeepONetCfCDecoder:
    torch.manual_seed(0)
    return DeepONetCfCDecoder(
        fourier_harmonics=8,
        d_model=hidden,
        d_time=8,
        domain_length=1.0,
        use_temporal_anchor=True,
        T_total=5.0,
        temporal_anchor_harmonics=2,
        num_query_mlp_layers=1,
        query_mlp_hidden_dim=hidden,
        output_head_gain=1.0,
        operator_rank=rank,
        fourier_embed_dim=0,
        use_periodic_domain=True,
    )


def test_decoder_forward_uvp_matches_per_component():
    """forward_uvp(xy, t_q) ≡ stack of forward(xy, t_q, c=k) for k=0,1,2."""
    torch.manual_seed(42)
    dec = _build_decoder().eval()
    K, T_steps, N = 12, 9, 7
    xy = torch.rand(N, 2)
    t_q = torch.rand(N) * 4.5
    sensor_pos = torch.rand(K, 2)
    sensor_time = torch.linspace(0.0, 5.0, T_steps)
    h_states = torch.randn(T_steps, K, 64)

    out_uvp = dec.forward_uvp(xy, t_q, h_states, sensor_time, sensor_pos)  # [N, 3]

    outs = []
    for k in range(3):
        c = torch.full((N,), k, dtype=torch.long)
        out_k = dec(xy, t_q, c, h_states, sensor_time, sensor_pos).squeeze(-1)
        outs.append(out_k)
    out_per_c = torch.stack(outs, dim=-1)  # [N, 3]

    max_abs = (out_uvp - out_per_c).abs().max().item()
    rel = max_abs / (out_per_c.abs().max().item() + 1e-12)
    print(f"  decoder forward_uvp vs per-c: max_abs={max_abs:.3e}, rel={rel:.3e}")
    assert max_abs < 1e-5, f"forward_uvp 偏離 per-c forward 太大：max_abs={max_abs}"


def test_spatial_encoder_T_axis_equivalence():
    """SpatialSetEncoder([T,K,C]) ≡ stack of SpatialSetEncoder([K,C])."""
    torch.manual_seed(7)
    K, C, T_steps = 12, 2, 5
    enc = SpatialSetEncoder(
        fourier_harmonics=8,
        sensor_value_dim=C,
        d_model=32,
        num_layers=1,
        domain_length=1.0,
        fourier_embed_dim=0,
    ).eval()
    sensor_vals = torch.randn(T_steps, K, C)
    sensor_pos = torch.rand(K, 2)
    pos_enc = enc.encode_pos(sensor_pos)

    out_batched = enc(sensor_vals, pos_enc)
    out_per_t = torch.stack([enc(sensor_vals[t], pos_enc) for t in range(T_steps)])

    max_abs = (out_batched - out_per_t).abs().max().item()
    print(f"  SpatialSetEncoder T-axis: max_abs={max_abs:.3e}")
    assert max_abs < 1e-5, f"SpatialSetEncoder T 軸向量化偏離：max_abs={max_abs}"


def test_temporal_encoder_attention_hoist_equivalence():
    """新版 forward (attn over [T,K,d]) ≡ 舊版 (per-t attn batch=1)。

    用同一隨機 seed 建模、同樣 input；差異純粹來自浮點 reduction 順序。
    """
    torch.manual_seed(11)
    d_model, K, T_steps = 32, 8, 4

    # 建一個 fresh module，然後手動模擬舊版逐 t 行為作為 reference。
    enc = TemporalCfCEncoder(
        d_model=d_model,
        num_layers=1,
        num_token_attention_layers=1,
        token_attention_heads=4,
        use_bidirectional=False,
    ).eval()
    spatial_states = torch.randn(T_steps, K, d_model)
    sensor_time = torch.linspace(0.0, 1.0, T_steps)
    re_norm = 0.5

    # 新版（已經是 forward 內部新邏輯）
    out_new = enc(spatial_states, re_norm, sensor_time)

    # 手動跑舊版邏輯
    dts = torch.cat([sensor_time[:1], sensor_time[1:] - sensor_time[:-1]])
    re_bias = enc._re_bias(re_norm, spatial_states.device, spatial_states.dtype).view(1, 1, -1)
    seq_old = spatial_states
    layer_idx = 0
    h = torch.zeros(K, d_model)
    outs = []
    for t in range(T_steps):
        x_t = seq_old[t]
        # 舊版 per-t attention：unsqueeze→squeeze
        x_t_attn = enc.token_blocks[0](x_t.unsqueeze(0)).squeeze(0)
        x_t_b = x_t_attn + re_bias.squeeze(0).squeeze(0)
        h = enc.cells[layer_idx](x_t_b, h, dt=dts[t])
        outs.append(h)
    out_old = torch.stack(outs)

    max_abs = (out_new - out_old).abs().max().item()
    rel = max_abs / (out_old.abs().max().item() + 1e-12)
    print(f"  TemporalCfC attention hoist: max_abs={max_abs:.3e}, rel={rel:.3e}")
    # MultiheadAttention 在 batch 大小變動時 BLAS 順序可能不同 → 稍寬鬆
    assert max_abs < 1e-4, f"TemporalCfC attention hoist 偏離過大：max_abs={max_abs}"


def test_full_operator_encode_then_decode_equivalence():
    """LiquidOperator.encode + query_decoder.forward_uvp 與
    LiquidOperator.encode + 3× forward 等價。"""
    torch.manual_seed(123)
    K, T_steps, N = 16, 6, 5
    net = LiquidOperator(
        fourier_harmonics=8,
        sensor_value_dim=2,
        d_model=64,
        d_time=8,
        num_spatial_cfc_layers=1,
        num_temporal_cfc_layers=1,
        domain_length=1.0,
        use_temporal_anchor=True,
        T_total=5.0,
        temporal_anchor_harmonics=2,
        num_token_attention_layers=1,
        token_attention_heads=4,
        num_query_mlp_layers=1,
        query_mlp_hidden_dim=64,
        operator_rank=64,
        fourier_embed_dim=0,
        use_periodic_domain=True,
    ).eval()

    sensor_vals = torch.randn(T_steps, K, 2)
    sensor_pos = torch.rand(K, 2)
    sensor_time = torch.linspace(0.0, 5.0, T_steps)
    re_norm = 0.5
    xy = torch.rand(N, 2)
    t_q = torch.rand(N) * 4.5

    h_states, s_time = net.encode(sensor_vals, sensor_pos, re_norm, sensor_time)
    out_uvp = net.query_decoder.forward_uvp(xy, t_q, h_states, s_time, sensor_pos)
    outs = []
    for k in range(3):
        c = torch.full((N,), k, dtype=torch.long)
        outs.append(net.query_decoder(xy, t_q, c, h_states, s_time, sensor_pos).squeeze(-1))
    out_per_c = torch.stack(outs, dim=-1)

    max_abs = (out_uvp - out_per_c).abs().max().item()
    print(f"  Full operator: max_abs={max_abs:.3e}")
    assert max_abs < 1e-4, f"Full operator forward_uvp 偏離 per-c：max_abs={max_abs}"


if __name__ == "__main__":
    print("=== Equivalence tests ===")
    test_decoder_forward_uvp_matches_per_component()
    test_spatial_encoder_T_axis_equivalence()
    test_temporal_encoder_attention_hoist_equivalence()
    test_full_operator_encode_then_decode_equivalence()
    print("=== ALL PASSED ===")
