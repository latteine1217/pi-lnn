"""gradient_cosine_diagnostic 單元測試（PCGrad 前置診斷）。

驗證：
1. 回傳鍵齊全（cos_data_phys / cos_data_<task> / gnorm_<task>）。
2. cosine 落在 [-1, 1]；同向 loss → +1，反向 → -1（數值正確性）。
3. 在 backward 之前呼叫不破壞後續 backward（retain_graph 契約）。
"""
from __future__ import annotations

import torch

from pi_con.losses import gradient_cosine_diagnostic


def _shared_layer() -> torch.nn.Linear:
    torch.manual_seed(0)
    return torch.nn.Linear(4, 3)


def test_keys_and_range() -> None:
    layer = _shared_layer()
    x = torch.randn(8, 4)
    out = layer(x)
    losses = {
        "data": (out ** 2).mean(),
        "ns_u": (out - 1.0).pow(2).mean(),
        "ns_v": (out + 0.5).pow(2).mean(),
        "cont": out.abs().mean(),
    }
    ref = list(layer.parameters())
    res = gradient_cosine_diagnostic(losses, ref)

    # 鍵齊全
    assert "cos_data_phys" in res
    for t in ("ns_u", "ns_v", "cont"):
        assert f"cos_data_{t}" in res
        assert f"gnorm_{t}" in res
    assert "gnorm_data" in res
    # cosine 範圍
    for k, v in res.items():
        if k.startswith("cos_"):
            assert -1.0 - 1e-6 <= v <= 1.0 + 1e-6, f"{k}={v} 超出 [-1,1]"


def test_sign_correctness() -> None:
    """同一 loss 兩份 → cos=+1；負號 loss → cos=-1。"""
    layer = _shared_layer()
    x = torch.randn(8, 4)
    out = layer(x)
    base = (out ** 2).mean()
    ref = list(layer.parameters())
    res = gradient_cosine_diagnostic({"data": base, "ns_u": base, "ns_v": -base, "cont": base}, ref)
    assert abs(res["cos_data_ns_u"] - 1.0) < 1e-5
    assert abs(res["cos_data_ns_v"] + 1.0) < 1e-5


def test_does_not_break_backward() -> None:
    """診斷在 backward 之前呼叫，後續 l_total.backward() 仍須成功且梯度正確。"""
    layer = _shared_layer()
    x = torch.randn(8, 4)
    out = layer(x)
    l_data = (out ** 2).mean()
    l_ns_u = (out - 1.0).pow(2).mean()
    l_ns_v = (out + 0.5).pow(2).mean()
    l_cont = out.abs().mean()
    ref = list(layer.parameters())

    _ = gradient_cosine_diagnostic(
        {"data": l_data, "ns_u": l_ns_u, "ns_v": l_ns_v, "cont": l_cont}, ref
    )
    # 診斷未污染 .grad（autograd.grad 不累加）
    assert all(p.grad is None for p in ref)

    l_total = l_data + l_ns_u + l_ns_v + l_cont
    l_total.backward()  # 不應 raise（圖仍存活）
    assert all(p.grad is not None for p in ref)
