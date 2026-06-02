"""B3 Sensor-conditioned FiLM 的 identity-init 回歸測試。

驗證：use_sensor_film=True 但 γ identity-init (γ=1, β=0) 時，
_apply_sensor_film 是 identity transform（不破壞 trunk_feat）。
這保證啟用 FiLM 的初始狀態 == baseline，訓練從 baseline 附近啟動。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pi_con.decoder import DeepONetCfCDecoder  # noqa: E402


def _mk(use_film: bool, film_geo: bool) -> DeepONetCfCDecoder:
    torch.manual_seed(0)
    return DeepONetCfCDecoder(
        fourier_harmonics=16,
        d_model=64,
        d_time=16,
        query_mlp_hidden_dim=64,   # 必須 == d_model（FiLM 一致性檢查）
        operator_rank=64,
        fourier_embed_dim=32,
        use_periodic_domain=False,
        use_sensor_film=use_film,
        film_use_geometry=film_geo,
    )


def test_film_module_built_only_when_enabled():
    assert not hasattr(_mk(False, False), "film_mlp")
    assert hasattr(_mk(True, False), "film_mlp")


def test_film_identity_init_is_identity():
    d = _mk(True, False)
    feat = torch.randn(5, 64)
    h = torch.randn(5, 64)
    out = d._apply_sensor_film(feat, h, None, n_repeat=1)
    assert torch.allclose(out, feat, atol=1e-6), f"max diff {(out - feat).abs().max()}"


def test_film_3n_batch_repeats_correctly():
    d = _mk(True, False)
    feat = torch.randn(15, 64)   # 3N, N=5
    h = torch.randn(5, 64)
    out = d._apply_sensor_film(feat, h, None, n_repeat=3)
    assert torch.allclose(out, feat, atol=1e-6)


def test_film_dynamic_repeat_matches_explicit():
    """forward path 中 n_repeat 動態計算（trunk/h 比值）應與顯式傳入一致。"""
    d = _mk(True, False)
    feat = torch.randn(10, 64)  # 2N, N=5
    h = torch.randn(5, 64)
    out_explicit = d._apply_sensor_film(feat, h, None, n_repeat=2)
    # 動態版：trunk_feat.shape[0] // h.shape[0] = 2
    n_repeat_dyn = feat.shape[0] // h.shape[0]
    out_dyn = d._apply_sensor_film(feat, h, None, n_repeat=n_repeat_dyn)
    assert torch.allclose(out_explicit, out_dyn, atol=1e-7)


def test_film_geometry_requires_distance():
    d = _mk(True, True)
    feat = torch.randn(5, 64)
    h = torch.randn(5, 64)
    try:
        d._apply_sensor_film(feat, h, None, n_repeat=1)
        raise AssertionError("應 raise（film_use_geometry=True 但 body_distance=None）")
    except ValueError:
        pass


def test_film_dmodel_mismatch_raises():
    try:
        DeepONetCfCDecoder(
            fourier_harmonics=16, d_model=64, d_time=16,
            query_mlp_hidden_dim=128,  # != d_model → 應 raise
            operator_rank=64, fourier_embed_dim=32,
            use_periodic_domain=False, use_sensor_film=True,
        )
        raise AssertionError("應 raise（d_model != query_mlp_hidden_dim）")
    except ValueError:
        pass


def test_film_3d_h_branch_tokens_mean_pool():
    """h_branch_tokens 為 [N, K, d_model] 時（由 h_states[idx] 產生），
    mean pooling 後 FiLM 不應 crash，且 identity-init 仍是 identity。
    根因：h_states 形狀 [T, K, d_model]，h_states[idx] = [N, K, d_model]（3D）；
    舊實作的 gb[:, :hidden] 切 K 維而非 feature 維 → gamma 形狀錯誤 → crash。
    """
    d = _mk(True, False)
    N, K = 5, 10   # 5 queries, 10 sensor tokens
    feat = torch.randn(N, 64)
    h_3d = torch.randn(N, K, 64)   # 模擬 h_states[idx]（[N, K, d_model]）
    out = d._apply_sensor_film(feat, h_3d, None, n_repeat=1)
    assert out.shape == feat.shape, f"輸出形狀應等於 trunk_feat 形狀，收到 {out.shape}"
    assert torch.allclose(out, feat, atol=1e-6), f"identity-init 應保持 trunk_feat，max diff {(out - feat).abs().max()}"


def test_film_3d_h_with_3n_repeat():
    """forward_uvp path：trunk_feat=[3N, d_model]，h=[N, K, d_model]，n_repeat=3。"""
    d = _mk(True, False)
    N, K = 5, 10
    feat_3n = torch.randn(3 * N, 64)
    h_3d = torch.randn(N, K, 64)
    out = d._apply_sensor_film(feat_3n, h_3d, None, n_repeat=3)
    assert out.shape == feat_3n.shape, f"輸出形狀應等於 trunk_feat 形狀，收到 {out.shape}"
    assert torch.allclose(out, feat_3n, atol=1e-6), f"identity-init 應保持 trunk_feat，max diff {(out - feat_3n).abs().max()}"
