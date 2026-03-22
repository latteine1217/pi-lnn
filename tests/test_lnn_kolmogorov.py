# tests/test_lnn_kolmogorov.py
import pytest
import torch
import torch.nn as nn
from pi_onet.lnn_kolmogorov import CfCCell, SpatialCfCEncoder, TemporalCfCEncoder, QueryCfCDecoder, LiquidOperator, create_lnn_model

D = 32

def test_cfccell_output_shape():
    """CfCCell([B, in], [B, hid]) → [B, hid]."""
    cell = CfCCell(input_size=16, hidden_size=D)
    x = torch.randn(8, 16)
    h = torch.zeros(8, D)
    h_new = cell(x, h, dt=1.0)
    assert h_new.shape == (8, D)

def test_cfccell_dt_sensitivity():
    """不同 dt 應產生不同輸出（gate 依賴 dt）。"""
    cell = CfCCell(input_size=16, hidden_size=D)
    x = torch.randn(4, 16)
    h = torch.zeros(4, D)
    with torch.no_grad():
        out1 = cell(x, h, dt=1.0)
        out2 = cell(x, h, dt=5.0)
    assert not torch.allclose(out1, out2)

def test_cfccell_backward():
    """loss.backward() 成功，所有參數梯度非 None。"""
    cell = CfCCell(input_size=16, hidden_size=D)
    x = torch.randn(4, 16)
    h = torch.zeros(4, D)
    h_new = cell(x, h, dt=1.0)
    h_new.sum().backward()
    for name, p in cell.named_parameters():
        assert p.grad is not None, f"{name} 無梯度"


RFF_F = 16
K = 12  # num sensors

def test_spatial_encoder_output_shape():
    """SpatialCfCEncoder: sensors[K,3] + pos[K,2] → [d_model]。"""
    B = torch.randn(2, RFF_F)
    enc = SpatialCfCEncoder(rff_features=RFF_F, d_model=D, num_layers=2)
    sensor_vals = torch.randn(K, 3)
    sensor_pos = torch.rand(K, 2) * 6.28
    s_t = enc(sensor_vals, sensor_pos, B)
    assert s_t.shape == (D,)

def test_spatial_encoder_backward():
    """梯度可從 s_t 流回 sensor_vals。"""
    B = torch.randn(2, RFF_F)
    enc = SpatialCfCEncoder(rff_features=RFF_F, d_model=D, num_layers=1)
    sensor_vals = torch.randn(K, 3, requires_grad=True)
    sensor_pos = torch.rand(K, 2)
    s_t = enc(sensor_vals, sensor_pos, B)
    s_t.sum().backward()
    assert sensor_vals.grad is not None


T = 21   # 感測器時間步數

def test_temporal_encoder_output_shape():
    """TemporalCfCEncoder: spatial_states[T, d_model] → [d_model]。"""
    enc = TemporalCfCEncoder(d_model=D, num_layers=2)
    states = torch.randn(T, D)
    h_enc = enc(states, re_norm=0.0, dt_phys=1.0)
    assert h_enc.shape == (D,)

def test_temporal_encoder_dt_effect():
    """不同 dt_phys 應產生不同 h_enc（CfC gate 依賴 Δt）。"""
    enc = TemporalCfCEncoder(d_model=D, num_layers=1)
    states = torch.randn(T, D)
    with torch.no_grad():
        h1 = enc(states, re_norm=0.0, dt_phys=1.0)
        h2 = enc(states, re_norm=0.0, dt_phys=0.1)
    assert not torch.allclose(h1, h2)

def test_temporal_encoder_re_effect():
    """不同 re_norm 應影響 h_enc（Re 以殘差方式加入每步輸入，貫穿序列演化）。"""
    enc = TemporalCfCEncoder(d_model=D, num_layers=1)
    states = torch.randn(T, D)
    with torch.no_grad():
        h1 = enc(states, re_norm=0.0, dt_phys=1.0)
        h2 = enc(states, re_norm=2.0, dt_phys=1.0)
    assert not torch.allclose(h1, h2)


N_Q = 16

def test_query_decoder_output_shape():
    """QueryCfCDecoder: (xy[N_q,2], t_q[N_q], c[N_q], h_enc[d]) → [N_q, 1]。"""
    B = torch.randn(2, RFF_F)
    dec = QueryCfCDecoder(rff_features=RFF_F, d_model=D, d_time=8)
    xy = torch.rand(N_Q, 2) * 6.28
    t_q = torch.rand(N_Q) * 20.0
    c = torch.randint(0, 3, (N_Q,))
    h_enc = torch.randn(D)
    out = dec(xy, t_q, c, h_enc, B)
    assert out.shape == (N_Q, 1)

def test_query_decoder_vectorized():
    """Decoder 不做 for-loop：N_q=1 與 N_q=512 的 forward 均可運行。"""
    B = torch.randn(2, RFF_F)
    dec = QueryCfCDecoder(rff_features=RFF_F, d_model=D, d_time=8)
    h_enc = torch.randn(D)
    for n in (1, 512):
        xy = torch.rand(n, 2)
        t_q = torch.rand(n)
        c = torch.randint(0, 3, (n,))
        assert dec(xy, t_q, c, h_enc, B).shape == (n, 1)

def test_liquid_operator_forward_shape():
    """LiquidOperator.forward → [N_q, 1]，無任何 Attention。"""
    cfg = {
        "rff_features": RFF_F, "rff_sigma": 1.0, "d_model": D, "d_time": 8,
        "num_spatial_cfc_layers": 1, "num_temporal_cfc_layers": 1,
    }
    net = create_lnn_model(cfg)
    # 確認無 Attention
    for mod in net.modules():
        assert not isinstance(mod, nn.MultiheadAttention), "發現 MultiheadAttention"
        assert not isinstance(mod, nn.TransformerEncoder), "發現 TransformerEncoder"
    sensor_vals = torch.randn(T, K, 3)
    sensor_pos = torch.rand(K, 2) * 6.28
    xy = torch.rand(N_Q, 2) * 6.28
    t_q = torch.rand(N_Q) * 20.0
    c = torch.randint(0, 3, (N_Q,))
    out = net(sensor_vals, sensor_pos, re_norm=0.0, dt_phys=1.0, xy=xy, t_q=t_q, c=c)
    assert out.shape == (N_Q, 1)

def test_liquid_operator_backward():
    """loss.backward() 成功，output_head 有梯度。"""
    cfg = {
        "rff_features": RFF_F, "rff_sigma": 1.0, "d_model": D, "d_time": 8,
        "num_spatial_cfc_layers": 1, "num_temporal_cfc_layers": 1,
    }
    net = create_lnn_model(cfg)
    sensor_vals = torch.randn(T, K, 3)
    sensor_pos = torch.rand(K, 2) * 6.28
    xy = torch.rand(N_Q, 2) * 6.28
    t_q = torch.rand(N_Q) * 20.0
    c = torch.randint(0, 3, (N_Q,))
    out = net(sensor_vals, sensor_pos, re_norm=0.0, dt_phys=1.0, xy=xy, t_q=t_q, c=c)
    out.sum().backward()
    assert net.query_decoder.output_head.weight.grad is not None
