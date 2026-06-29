"""Geometry-aware opt-in tests for PI-CON.

What: 驗證 graph spatial encoder 與 trunk geometry context 皆為顯式開關。
Why: Kolmogorov 舊主線必須在預設 false 時完全維持舊模型參數與行為；cylinder
     幾何功能啟用時則必須 fail fast，避免靜默退化成錯誤實驗。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pi_con import DEFAULT_PICON_ARGS, LiquidOperator, create_picon_model  # noqa: E402


def _make_model(**overrides) -> LiquidOperator:
    cfg = dict(
        fourier_harmonics=4,
        sensor_value_dim=2,
        d_model=32,
        d_time=4,
        num_spatial_cfc_layers=1,
        num_temporal_cfc_layers=1,
        fourier_embed_dim=0,
        query_mlp_hidden_dim=32,
        operator_rank=32,
    )
    cfg.update(overrides)
    return LiquidOperator(**cfg)


def test_default_model_has_no_geometry_parameters():
    """預設 false 時不建立任何 geometry-specific parameter，保持舊 checkpoint 相容。"""
    model = _make_model()
    names = {name for name, _ in model.named_parameters()}

    assert not any("graph_spatial" in name for name in names)
    assert not any("trunk_geo_context" in name for name in names)


def test_graph_spatial_encoder_requires_geometry_positions():
    """開啟 graph spatial encoder 後，未注入 geometry_pos 必須 fail fast。"""
    model = _make_model(use_graph_spatial_encoder=True)
    sensor_vals = torch.rand(3, 8, 2)
    sensor_pos = torch.rand(8, 2)
    sensor_time = torch.linspace(0.0, 1.0, 3)

    with pytest.raises(ValueError, match="use_graph_spatial_encoder=True"):
        model.encode(sensor_vals, sensor_pos, re_norm=0.0, sensor_time=sensor_time)


def test_graph_spatial_encoder_geometry_changes_tokens():
    """注入 geometry positions 後，B 路徑應實際改變 spatial token。"""
    torch.manual_seed(7)
    base = _make_model(use_graph_spatial_encoder=False)
    graph = _make_model(use_graph_spatial_encoder=True)
    graph.load_state_dict(base.state_dict(), strict=False)

    sensor_vals = torch.rand(2, 8, 2)
    sensor_pos = torch.rand(8, 2)
    geometry_pos = torch.rand(5, 2)

    pos_enc_base = base.spatial_encoder.encode_pos(sensor_pos)
    pos_enc_graph = graph.spatial_encoder.encode_pos(sensor_pos)
    graph.set_geometry_tokens(geometry_pos)

    out_base = base.spatial_encoder(sensor_vals, pos_enc_base, re_norm=0.0)
    out_graph = graph.spatial_encoder(
        sensor_vals,
        pos_enc_graph,
        re_norm=0.0,
        sensor_pos=sensor_pos,
        geometry_pos=graph.query_decoder.geometry_pos,
    )

    assert out_graph.shape == out_base.shape
    assert not torch.allclose(out_graph, out_base)


def test_graph_spatial_zero_gate_initially_preserves_tokens():
    """B-zero gate 開啟時，初始 geometry message 不應改變 sensor tokens。"""
    torch.manual_seed(7)
    base = _make_model(use_graph_spatial_encoder=False)
    graph = _make_model(use_graph_spatial_encoder=True, use_graph_spatial_gate=True)
    graph.load_state_dict(base.state_dict(), strict=False)

    sensor_vals = torch.rand(2, 8, 2)
    sensor_pos = torch.rand(8, 2)
    geometry_pos = torch.rand(5, 2)

    pos_enc_base = base.spatial_encoder.encode_pos(sensor_pos)
    pos_enc_graph = graph.spatial_encoder.encode_pos(sensor_pos)
    graph.set_geometry_tokens(geometry_pos)

    out_base = base.spatial_encoder(sensor_vals, pos_enc_base, re_norm=0.0)
    out_graph = graph.spatial_encoder(
        sensor_vals,
        pos_enc_graph,
        re_norm=0.0,
        sensor_pos=sensor_pos,
        geometry_pos=graph.query_decoder.geometry_pos,
    )

    assert torch.allclose(out_graph, out_base)


def test_geometry_preserve_base_rng_keeps_common_initialization():
    """geometry_preserve_base_rng=True 時，新增 geometry modules 不得改變舊參數初始化。"""
    torch.manual_seed(11)
    base = _make_model()
    torch.manual_seed(11)
    c_control = _make_model(
        use_trunk_geo_context=True,
        geometry_preserve_base_rng=True,
    )

    base_state = base.state_dict()
    c_state = c_control.state_dict()
    for name, expected in base_state.items():
        assert name in c_state
        assert torch.equal(c_state[name], expected), name


def test_trunk_geo_context_requires_geometry_positions():
    """開啟 C 後，decoder 沒 geometry memory 時不得靜默退回舊 trunk。"""
    model = _make_model(use_trunk_geo_context=True)
    xy = torch.rand(4, 2)
    t_q = torch.rand(4)
    c = torch.zeros(4, dtype=torch.long)
    h_states = torch.rand(3, 8, 32)
    sensor_time = torch.linspace(0.0, 1.0, 3)
    sensor_pos = torch.rand(8, 2)

    with pytest.raises(ValueError, match="use_trunk_geo_context=True"):
        model.query_decoder(xy, t_q, c, h_states, sensor_time, sensor_pos)


def test_trunk_geo_context_preserves_forward_uvp_shape():
    """注入 geometry positions 後，C 路徑仍維持 [N,3] uvp contract。"""
    model = _make_model(use_trunk_geo_context=True)
    model.set_geometry_tokens(torch.rand(6, 2))
    xy = torch.rand(4, 2)
    t_q = torch.rand(4)
    h_states = torch.rand(3, 8, 32)
    sensor_time = torch.linspace(0.0, 1.0, 3)
    sensor_pos = torch.rand(8, 2)

    out = model.query_decoder.forward_uvp(xy, t_q, h_states, sensor_time, sensor_pos)

    assert out.shape == (4, 3)
    assert torch.isfinite(out).all()


def test_create_model_allows_fixed_zero_forcing_for_cylinder_configs():
    """Cylinder 無 Kolmogorov forcing；固定 A=0 不應阻止 model 建立。"""
    cfg = dict(DEFAULT_PICON_ARGS)
    cfg.update(
        dataset_type="cylinder",
        use_periodic_domain=False,
        fourier_embed_dim=8,
        d_model=32,
        d_time=4,
        num_spatial_cfc_layers=1,
        num_temporal_cfc_layers=1,
        query_mlp_hidden_dim=32,
        operator_rank=32,
        kolmogorov_A=0.0,
        kolmogorov_k_f=0.0,
        learn_forcing_A=False,
        learn_forcing_k_f=False,
    )

    model = create_picon_model(cfg)

    assert not hasattr(model, "forcing")
