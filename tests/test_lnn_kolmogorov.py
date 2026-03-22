# tests/test_lnn_kolmogorov.py
import pytest
import torch
import torch.nn as nn
from pi_onet.lnn_kolmogorov import CfCCell, SpatialCfCEncoder

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
