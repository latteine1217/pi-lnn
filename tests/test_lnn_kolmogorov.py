# tests/test_lnn_kolmogorov.py
import pytest
import torch
import torch.nn as nn
from pi_onet.lnn_kolmogorov import CfCCell, SpatialCfCEncoder, TemporalCfCEncoder

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
