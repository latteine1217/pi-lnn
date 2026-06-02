"""3D channel NS residual — method of manufactured solutions 驗證。

每個 test 餵入解析已知場給 channel_ns_residuals，比對 residual 各項。
用 float64 提高二階 autograd 精度。
"""
import math

import torch

from pi_con.physics import channel_ns_residuals


def _make_xyzt(n: int, seed: int = 0) -> torch.Tensor:
    """產生 [n,4] = (x,y,z,t) normalized 隨機座標，requires_grad 供 autograd。"""
    g = torch.Generator().manual_seed(seed)
    xyzt = torch.rand(n, 4, generator=g, dtype=torch.float64)
    xyzt.requires_grad_(True)
    return xyzt


def test_constant_field_only_forcing():
    """常數場 → 所有導數 0 → mom_u = -body_force_x, 其餘 = 0, cont = 0。"""
    xyzt = _make_xyzt(16)
    bfx = 2.49e-3

    def fn(coords):
        base = torch.tensor([0.7, -0.3, 0.2, 1.5], dtype=coords.dtype)
        return base.expand(coords.shape[0], 4)

    mu, mv, mw, co = channel_ns_residuals(
        fn, xyzt, re=1000.0, body_force_x=bfx,
        Lx=8 * math.pi, Ly=2.0, Lz=3 * math.pi,
    )
    assert torch.allclose(mu, torch.full_like(mu, -bfx), atol=1e-9)
    assert torch.allclose(mv, torch.zeros_like(mv), atol=1e-9)
    assert torch.allclose(mw, torch.zeros_like(mw), atol=1e-9)
    assert torch.allclose(co, torch.zeros_like(co), atol=1e-9)


def test_linear_field_chain_rule_continuity_advection():
    """線性場 u=a·x, v=b·y, w=c·z → 驗證 chain rule、3D continuity、advection。"""
    xyzt = _make_xyzt(16, seed=1)
    a, b, c = 0.5, 0.3, -0.2
    Lx, Ly, Lz = 8 * math.pi, 2.0, 3 * math.pi

    def fn(coords):
        x, y, z = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
        u = a * x
        v = b * y
        w = c * z
        p = torch.zeros_like(x)
        return torch.cat([u, v, w, p], dim=1)

    mu, mv, mw, co = channel_ns_residuals(
        fn, xyzt, re=1000.0, body_force_x=0.0, Lx=Lx, Ly=Ly, Lz=Lz,
    )
    # continuity = a/Lx + b/Ly + c/Lz（處處常數，二階導 0 → viscous 0）
    expected_cont = a / Lx + b / Ly + c / Lz
    assert torch.allclose(co, torch.full_like(co, expected_cont), atol=1e-8)
    # mom_u/v/w = adv = (各分量·各自方向梯度)；對稱驗 u,v,w 三分量防 adv_v/adv_w 漏項
    # mom_u = u·du_dx = (a·x)·(a/Lx) = a²·x/Lx（du_dt, dp_dx, viscous 皆 0）
    x = xyzt[:, 0:1].detach()
    expected_mu = (a ** 2) * x / Lx
    assert torch.allclose(mu, expected_mu, atol=1e-7)
    # mom_v = v·dv_dy = (b·y)·(b/Ly) = b²·y/Ly
    y = xyzt[:, 1:2].detach()
    expected_mv = (b ** 2) * y / Ly
    assert torch.allclose(mv, expected_mv, atol=1e-7)
    # mom_w = w·dw_dz = (c·z)·(c/Lz) = c²·z/Lz
    z = xyzt[:, 2:3].detach()
    expected_mw = (c ** 2) * z / Lz
    assert torch.allclose(mw, expected_mw, atol=1e-7)


def test_taylor_green_xy_divergence_free():
    """Taylor-Green (x-y) div-free 場 → 3D continuity ≈ 0（驗證 dw_dz 正確併入）。"""
    xyzt = _make_xyzt(64, seed=2)
    k = 2 * math.pi

    def fn(coords):
        x, y = coords[:, 0:1], coords[:, 1:2]
        u = torch.sin(k * x) * torch.cos(k * y)
        v = -torch.cos(k * x) * torch.sin(k * y)
        w = torch.zeros_like(x)
        p = torch.zeros_like(x)
        return torch.cat([u, v, w, p], dim=1)

    _, _, _, co = channel_ns_residuals(
        fn, xyzt, re=1000.0, body_force_x=0.0, Lx=1.0, Ly=1.0, Lz=1.0,
    )
    assert torch.allclose(co, torch.zeros_like(co), atol=1e-6)


def test_viscous_laplacian_and_advection_z():
    """w=sin(k·z) 場 → 驗證 z 方向二階 viscous + advection w·dw_dz + continuity。"""
    xyzt = _make_xyzt(32, seed=3)
    Lz = 3 * math.pi
    k = 2 * math.pi
    re = 100.0
    nu = 1.0 / re

    def fn(coords):
        z = coords[:, 2:3]
        w = torch.sin(k * z)
        zero = torch.zeros_like(z)
        return torch.cat([zero, zero, w, zero], dim=1)

    _, _, mw, co = channel_ns_residuals(
        fn, xyzt, re=re, body_force_x=0.0, Lx=1.0, Ly=1.0, Lz=Lz,
    )
    z = xyzt[:, 2:3].detach()
    dw_dz = k * torch.cos(k * z) / Lz
    lap_w = -(k ** 2) * torch.sin(k * z) / (Lz ** 2)
    adv_w = torch.sin(k * z) * dw_dz          # w·dw_dz
    expected_mw = adv_w - nu * lap_w           # dw_dt=0, dp_dz=0
    assert torch.allclose(mw, expected_mw, atol=1e-6)
    assert torch.allclose(co, dw_dz, atol=1e-6)  # 只有 dw_dz 非零
