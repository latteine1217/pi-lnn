"""tests/test_ns_residual_chain_rule.py

驗證 unsteady_ns_residuals 與 pressure_poisson_residual 的 Lx, Ly chain-rule 修正：

物理 PDE 的梯度應在物理座標系算，但 autograd 給的是 normalized 座標下的梯度。
chain rule:  ∂u/∂x_phys = ∂u/∂x_norm / Lx,  ∂²u/∂x²_phys = ∂²u/∂x²_norm / Lx²

驗收條件：
  1. Lx=Ly=1.0 → 與「無 chain rule」舊版本數值等價（Kolmogorov 向後相容）
  2. Lx, Ly > 1 → 一階梯度被精確 scale by 1/Lx 與 1/Ly
  3. 二階梯度被精確 scale by 1/Lx² 與 1/Ly²
  4. Cylinder anisotropic case (Lx≠Ly) 的 x、y 方向梯度權重不同
  5. Pressure Poisson residual 同樣有 chain rule
"""
from __future__ import annotations

import torch

from pi_lnn.physics import pressure_poisson_residual, unsteady_ns_residuals


def _linear_uvp_fn(slope_u_x: float = 1.0, slope_v_y: float = 0.0):
    """建立 uvp_fn：u = slope_u_x · x, v = slope_v_y · y, p = 0。
    解析梯度：du/dx_norm=slope_u_x, dv/dy_norm=slope_v_y, 二階皆 0。
    """
    def uvp(xyt: torch.Tensor) -> torch.Tensor:
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        u = slope_u_x * x
        v = slope_v_y * y
        p = torch.zeros_like(x)
        return torch.cat([u, v, p], dim=1)
    return uvp


def _quadratic_uvp_fn():
    """u = x²/2, v = y²/2, p = 0。
    du/dx_norm = x, du/dx²_norm = 1
    dv/dy_norm = y, dv/dy²_norm = 1
    """
    def uvp(xyt: torch.Tensor) -> torch.Tensor:
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        u = 0.5 * x ** 2
        v = 0.5 * y ** 2
        p = torch.zeros_like(x)
        return torch.cat([u, v, p], dim=1)
    return uvp


# ── 1. Lx=Ly=1.0 backward compat ──────────────────────────────────


def test_chain_rule_lx_ly_one_matches_pre_chain_rule():
    """Kolmogorov backward compat：Lx=Ly=1.0 應與無 chain rule 舊版等價。
    取一個簡單線性 u=x：du/dx_norm=1, mom_u 的 advection 項 u·du/dx = x·1 = x。
    """
    uvp = _linear_uvp_fn(slope_u_x=1.0)
    xyt = torch.tensor([[0.5, 0.3, 0.1]], requires_grad=True)
    mom_u, _, cont = unsteady_ns_residuals(
        uvp, xyt, re=1.0, k_f=0.0, A=0.0, Lx=1.0, Ly=1.0
    )
    # u=x=0.5, du/dx=1（Lx=1 → physically same）
    # mom_u = du/dt(=0) + u·du/dx + v·du/dy + dp/dx - ν·∇²u - forcing(=0)
    #       = 0.5 · 1 + 0 + 0 - 1·0 - 0  = 0.5
    expected_mom_u = 0.5
    expected_cont = 1.0  # du/dx + dv/dy = 1 + 0 = 1
    assert torch.allclose(mom_u, torch.tensor([[expected_mom_u]]), atol=1e-6)
    assert torch.allclose(cont, torch.tensor([[expected_cont]]), atol=1e-6)


# ── 2. Lx > 1 → 一階梯度 scale by 1/Lx ────────────────────────────


def test_first_derivative_scales_by_one_over_lx():
    """u = x, Lx=4：du/dx_phys = 1/4, mom_u 的 u·du/dx = x · 0.25。
    cont = du/dx + dv/dy = 0.25 + 0 = 0.25。
    """
    uvp = _linear_uvp_fn(slope_u_x=1.0)
    xyt = torch.tensor([[0.5, 0.3, 0.0]], requires_grad=True)
    Lx = 4.0
    mom_u, _, cont = unsteady_ns_residuals(
        uvp, xyt, re=1.0, k_f=0.0, A=0.0, Lx=Lx, Ly=1.0
    )
    # mom_u = u(0.5) · du/dx_phys = 0.5 · (1/4) = 0.125
    assert torch.allclose(mom_u, torch.tensor([[0.5 / Lx]]), atol=1e-6)
    assert torch.allclose(cont, torch.tensor([[1.0 / Lx]]), atol=1e-6)


def test_first_derivative_y_scales_by_one_over_ly():
    """v = y, Ly=2：dv/dy_phys = 1/2, cont = 0 + 0.5 = 0.5。"""
    uvp = _linear_uvp_fn(slope_u_x=0.0, slope_v_y=1.0)
    xyt = torch.tensor([[0.3, 0.5, 0.0]], requires_grad=True)
    Ly = 2.0
    _, _, cont = unsteady_ns_residuals(
        uvp, xyt, re=1.0, k_f=0.0, A=0.0, Lx=1.0, Ly=Ly
    )
    assert torch.allclose(cont, torch.tensor([[1.0 / Ly]]), atol=1e-6)


# ── 3. Lx > 1 → 二階梯度 scale by 1/Lx² ───────────────────────────


def test_second_derivative_scales_by_one_over_lx_squared():
    """u = x²/2, Lx=3：
       du/dx²_phys = du/dx²_norm / Lx² = 1 / 9
       viscous term: -ν · du/dx²_phys = -1 · 1/9
       mom_u 包含 -ν·(du²+du²) 項
    """
    uvp = _quadratic_uvp_fn()
    xyt = torch.tensor([[0.5, 0.3, 0.0]], requires_grad=True)
    Lx, Ly = 3.0, 1.0
    re = 1.0  # nu = 1.0
    mom_u, _, _ = unsteady_ns_residuals(
        uvp, xyt, re=re, k_f=0.0, A=0.0, Lx=Lx, Ly=Ly
    )
    # u = x²/2 = 0.125; du/dx_norm = x = 0.5; du/dx_phys = 0.5/3
    # advection u·du/dx = 0.125 · (0.5/3) = 0.02083
    # du/dx²_norm = 1; du/dx²_phys = 1/9
    # viscous: -ν · du/dx²_phys = -1/9 ≈ -0.1111
    # du/dy_norm = 0, du/dy²_norm = 0 → 不貢獻
    # mom_u = 0 + 0.02083 + 0 + 0 - 1·(1/9 + 0) - 0
    expected = (0.125 * (0.5 / Lx)) - (1.0 / (Lx ** 2))
    assert torch.allclose(mom_u, torch.tensor([[expected]]), atol=1e-6)


# ── 4. Anisotropic (Lx ≠ Ly) → x、y 方向有不同 scale ──────────────


def test_anisotropic_lx_neq_ly():
    """Cylinder-like：Lx=0.322, Ly=0.172（real values）。
    u=x², v=y² → 二階皆 1（normalized）→ viscous 項應為 -ν·(1/Lx² + 1/Ly²)。
    """
    uvp = _quadratic_uvp_fn()
    xyt = torch.tensor([[0.5, 0.5, 0.0]], requires_grad=True)
    Lx, Ly = 0.322, 0.172
    re = 1.0
    mom_u, mom_v, _ = unsteady_ns_residuals(
        uvp, xyt, re=re, k_f=0.0, A=0.0, Lx=Lx, Ly=Ly
    )
    # u = x²/2 = 0.125, du/dx_phys = 0.5/Lx
    # adv = u·du/dx_phys = 0.125 · 0.5/Lx
    # viscous = -1·(1/Lx² + 1/Ly²)  (du/dy²=0)
    expected_mom_u = 0.125 * (0.5 / Lx) - 1.0 * (1.0 / (Lx ** 2) + 0.0)
    expected_mom_v = 0.125 * (0.5 / Ly) - 1.0 * (0.0 + 1.0 / (Ly ** 2))
    assert torch.allclose(mom_u, torch.tensor([[expected_mom_u]]), atol=1e-5)
    assert torch.allclose(mom_v, torch.tensor([[expected_mom_v]]), atol=1e-5)


# ── 5. Anisotropy 影響量化（cylinder 真實參數）──────────────────


def test_cylinder_anisotropy_quantification():
    """量化 cylinder 真實 Lx, Ly 下 viscous 項的 x、y 方向相對權重比。
    Lx=0.322, Ly=0.172 → 1/Lx² = 9.6, 1/Ly² = 33.8 → ratio ≈ 0.285。
    這正是審查報告中 anti-pattern A 量化的證據。
    """
    Lx, Ly = 0.322, 0.172
    weight_x = 1.0 / (Lx ** 2)
    weight_y = 1.0 / (Ly ** 2)
    ratio = weight_x / weight_y
    assert 0.25 < ratio < 0.35, (
        f"cylinder Lx/Ly anisotropy: viscous x:y 權重比應 ~0.285，實得 {ratio:.3f}"
    )


# ── 6. Pressure Poisson 同樣有 chain rule ────────────────────────


def test_poisson_residual_uses_chain_rule():
    """u = x, v = y, p = x²/2: ∇²p_norm = 1, ∇²p_phys = 1/Lx²；
       du/dx_norm = 1 → du/dx_phys = 1/Lx; dv/dy_phys = 1/Ly
       Poisson rhs = -(du/dx_phys² + dv/dy_phys² + 0) = -(1/Lx² + 1/Ly²)
       residual = ∇²p_phys - rhs = 1/Lx² - (-(1/Lx² + 1/Ly²)) = 2/Lx² + 1/Ly²
    """
    def uvp(xyt: torch.Tensor) -> torch.Tensor:
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        return torch.cat([x, y, 0.5 * x ** 2], dim=1)

    xyt = torch.tensor([[0.5, 0.5, 0.0]], requires_grad=True)
    Lx, Ly = 2.0, 3.0
    res = pressure_poisson_residual(uvp, xyt, Lx=Lx, Ly=Ly)
    expected = 2.0 / (Lx ** 2) + 1.0 / (Ly ** 2)
    assert torch.allclose(res, torch.tensor([[expected]]), atol=1e-6)


# ── 7. 預設 Lx=Ly=1.0 與不傳參數結果一致 ──────────────────────────


def test_default_lx_ly_one_when_omitted():
    """確保 Lx, Ly 預設 1.0 → 不傳與顯式傳 1.0 結果完全相同。
    這保證舊呼叫者（未傳 Lx/Ly）在升級後行為不變。
    """
    uvp = _quadratic_uvp_fn()
    xyt = torch.tensor([[0.5, 0.5, 0.0]], requires_grad=True)

    # 不傳 Lx/Ly
    xyt2 = xyt.detach().clone().requires_grad_(True)
    mom_u_default, _, _ = unsteady_ns_residuals(
        uvp, xyt2, re=1.0, k_f=0.0, A=0.0
    )
    # 顯式 Lx=Ly=1.0
    xyt3 = xyt.detach().clone().requires_grad_(True)
    mom_u_explicit, _, _ = unsteady_ns_residuals(
        uvp, xyt3, re=1.0, k_f=0.0, A=0.0, Lx=1.0, Ly=1.0
    )
    torch.testing.assert_close(mom_u_default, mom_u_explicit, rtol=1e-7, atol=1e-9)
