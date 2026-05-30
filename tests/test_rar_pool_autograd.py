"""Regression test：RAR pool 更新的 autograd graph 共享問題（EXP-272 job 3721 根因）。

事故：EXP-272 首次啟用 `physics_collocation_strategy="rar"`，在第一次 `_rar_update_pool`
即崩潰：`RuntimeError: Trying to backward through the graph a second time`。

根因：`_rar_update_pool` 內 u/v/p 是同一次 forward `uvp = uvp_fn(xyt_pool)` 的 slice，
共用同一張 autograd graph。`_g1(u)` 跑完 backward 後 saved tensors 被釋放，
`_g1(v)` 再對同圖 backward 即失敗。修法：前 N-1 次 grad 用 retain_graph=True。

本測試 monkeypatch `make_picon_model_fn_uvp`，用一個共享 graph 的假 uvp_fn，
讓真實的 `_rar_update_pool` / `_g1` 三次一階 grad 邏輯跑起來，
不需建完整 LiquidOperator。修正前 raise、修正後 PASS。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pi_con.physics as physics  # noqa: E402


class _StubDataset:
    """最小 dataset：只需 re_norm + sample_physics_points。"""

    re_norm = 0.5

    def sample_physics_points(self, rng, n, t_max, strategy="random"):
        xy = (rng.random((n, 2)) * 2.0 * np.pi).astype(np.float32)
        t = (rng.random(n) * float(t_max if t_max is not None else 1.0)).astype(np.float32)
        return xy, t


class _StubNet(torch.nn.Module):
    """_rar_update_pool 會呼叫 net.eval()/net.train()，給個空 module 即可。"""

    def forward(self, x):  # pragma: no cover - 不會被呼叫（make_... 已 patch）
        return x


def _shared_graph_uvp_fn_factory():
    """回傳一個 uvp closure：對 xyt 做單一 forward，輸出 [N,3] 共享同一張 graph。

    這精確重現真實 net 的行為——u/v/p 是同一次 forward 的 slice，
    若 _g1 不保留圖，第二次 grad 就會 double-backward。
    """
    W = torch.randn(3, 3)

    def uvp_fn(xyt: torch.Tensor) -> torch.Tensor:
        return torch.tanh(xyt @ W)  # [N, 3]，三欄共用 graph

    return uvp_fn


def test_rar_update_pool_no_double_backward(monkeypatch) -> None:
    """真實 _rar_update_pool 在共享 graph 的 uvp 上不應 double-backward 崩潰。"""

    def _fake_make(net, sensor_vals, sensor_pos, re_norm, sensor_time, device,
                   body_distance_fn=None, **kw):
        return _shared_graph_uvp_fn_factory()

    monkeypatch.setattr(physics, "make_picon_model_fn_uvp", _fake_make)

    rng = np.random.default_rng(0)
    n_select = 16
    datasets = [_StubDataset(), _StubDataset()]

    out = physics._rar_update_pool(
        net=_StubNet(),
        datasets=datasets,
        sensor_vals_list=[None, None],
        sensor_pos_list=[None, None],
        sensor_time_list=[None, None],
        rng=rng,
        n_select=n_select,
        pool_size=n_select * 10,
        t_max=5.0,
        k_f=2.0,
        A=1.0,
        domain_length=1.0,
        device=torch.device("cpu"),
        exploration_ratio=0.2,
    )

    assert len(out) == len(datasets), "每個 dataset 應回傳一組選點"
    for arr in out:
        assert arr.shape == (n_select, 3), f"預期 (n_select, 3)，實得 {arr.shape}"
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all(), "選點不應含 NaN/Inf"
