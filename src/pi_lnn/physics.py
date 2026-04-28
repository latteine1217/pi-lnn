"""Physics losses: NS/Poisson residuals + Residual-Adaptive Refinement + scheduling."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from pi_lnn.operator import make_lnn_model_fn_uvp
from pi_lnn.runtime import _grad


def unsteady_ns_residuals(
    uvp_fn: Callable,
    xyt: torch.Tensor,
    re: float,
    k_f: float = 4.0,
    A: float = 0.1,
    domain_length: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """What: 2D incompressible NS 的 primitive-variable momentum 與 continuity 殘差。

    Why: sparse-data 主線的實際觀測仍只有 u,v，但 momentum equation 需要壓力梯度。
         因此 p 回到模型的內部 physics 場，只參與 PDE 殘差，不作資料 supervision。
         uvp_fn 一次回傳 [N, 3]，相較舊版三個獨立 closure 共用 c-independent 計算
         並合併二階 autograd graph，數學上等價。
    """
    uvp = uvp_fn(xyt)                                      # [N, 3]
    u = uvp[:, 0:1]
    v = uvp[:, 1:2]
    p = uvp[:, 2:3]
    u_xyt = _grad(u, xyt)
    v_xyt = _grad(v, xyt)
    p_xyt = _grad(p, xyt)
    du_dx, du_dy, du_dt = u_xyt[:, 0:1], u_xyt[:, 1:2], u_xyt[:, 2:3]
    dv_dx, dv_dy, dv_dt = v_xyt[:, 0:1], v_xyt[:, 1:2], v_xyt[:, 2:3]
    dp_dx, dp_dy = p_xyt[:, 0:1], p_xyt[:, 1:2]
    du_dx2 = _grad(du_dx, xyt)[:, 0:1]
    du_dy2 = _grad(du_dy, xyt)[:, 1:2]
    dv_dx2 = _grad(dv_dx, xyt)[:, 0:1]
    dv_dy2 = _grad(dv_dy, xyt)[:, 1:2]
    nu = 1.0 / float(re)
    forcing_wavenumber = (2.0 * torch.pi * float(k_f)) / float(domain_length)
    forcing_x = A * torch.sin(forcing_wavenumber * xyt[:, 1:2])
    mom_u = du_dt + u * du_dx + v * du_dy + dp_dx - nu * (du_dx2 + du_dy2) - forcing_x
    mom_v = dv_dt + u * dv_dx + v * dv_dy + dp_dy - nu * (dv_dx2 + dv_dy2)
    cont = du_dx + dv_dy
    return mom_u, mom_v, cont


def pressure_poisson_residual(
    uvp_fn: Callable,
    xyt: torch.Tensor,
) -> torch.Tensor:
    """What: 2D incompressible 壓力 Poisson 方程殘差。

    Why: Primitive-variable NS 中 p 沒有資料監督，模型可藉由任意調整 p 來讓
         momentum residual 歸零，即使 u, v 是錯的（壓力自由度問題）。
         Poisson 方程從 NS + ∇·u=0 推導而來：
             ∇²p = -(∂u/∂x)² - (∂v/∂y)² - 2(∂u/∂y)(∂v/∂x)
         加入此約束後，p 必須與 u, v 的二階結構一致，壓力不再能自由漂移。

    數學推導：對動量方程取散度，使用 ∇·u=0 及 Kolmogorov forcing ∇·f=0 後得到。
    不需要任何額外觀測量，僅用模型輸出的 u, v, p via autograd。
    """
    uvp = uvp_fn(xyt)                                      # [N, 3]
    u = uvp[:, 0:1]
    v = uvp[:, 1:2]
    p = uvp[:, 2:3]
    u_xyt = _grad(u, xyt)
    v_xyt = _grad(v, xyt)
    p_xyt = _grad(p, xyt)
    du_dx = u_xyt[:, 0:1]
    du_dy = u_xyt[:, 1:2]
    dv_dx = v_xyt[:, 0:1]
    dv_dy = v_xyt[:, 1:2]
    dp_dx = p_xyt[:, 0:1]
    dp_dy = p_xyt[:, 1:2]
    dp_dx2 = _grad(dp_dx, xyt)[:, 0:1]   # ∂²p/∂x²
    dp_dy2 = _grad(dp_dy, xyt)[:, 1:2]   # ∂²p/∂y²
    laplacian_p = dp_dx2 + dp_dy2
    rhs = -(du_dx ** 2 + dv_dy ** 2 + 2.0 * du_dy * dv_dx)
    return laplacian_p - rhs


def _rar_update_pool(
    net,
    datasets,
    sensor_vals_list: list,
    sensor_pos_list: list,
    sensor_time_list: list,
    rng: np.random.Generator,
    n_select: int,
    pool_size: int,
    t_max: float | None,
    k_f: float,
    A: float,
    domain_length: float,
    device,
    exploration_ratio: float = 0.2,
) -> list[np.ndarray]:
    """What: RAR（Residual Adaptive Refinement）pool 更新。

    Why: 均勻隨機採樣可能長期錯過 t≈0 等高殘差區域；
         每隔 rar_update_freq 步，從大候選集中選 top 殘差點，
         讓 physics loss 集中在模型最難收斂的區域。
         保留 exploration_ratio 比例的隨機點，防止采样退化到固定幾個點。

    近似殘差：略去黏性項（Re=10000 時 ν=1e-4，黏性貢獻約 0.2%）；
    使用 create_graph=False，僅計算一階導數，完全避免二階 autograd 建圖。
    此近似只影響 pool 中的點排序，不影響訓練 loss 本身的精確性。

    Returns:
        list of (n_select, 3) float32 numpy arrays, one per dataset.
    """
    n_top = max(1, round(n_select * (1.0 - exploration_ratio)))
    n_rand = n_select - n_top
    kw = 2.0 * torch.pi * float(k_f) / float(domain_length)
    result = []
    net.eval()
    for i, ds in enumerate(datasets):
        xy_np, t_np = ds.sample_physics_points(rng, n=pool_size, t_max=t_max, strategy="random")
        xyt_pool = torch.tensor(
            np.concatenate([xy_np, t_np[:, None]], axis=1),
            dtype=torch.float32, device=device, requires_grad=True,
        )
        uvp_fn = make_lnn_model_fn_uvp(
            net, sensor_vals_list[i], sensor_pos_list[i],
            re_norm=ds.re_norm, sensor_time=sensor_time_list[i], device=device,
        )
        uvp = uvp_fn(xyt_pool)
        u = uvp[:, 0:1]
        v = uvp[:, 1:2]
        p = uvp[:, 2:3]

        def _g1(y: torch.Tensor) -> torch.Tensor:
            g = torch.autograd.grad(
                y, xyt_pool, torch.ones_like(y),
                create_graph=False, allow_unused=True,
            )[0]
            return g if g is not None else torch.zeros_like(xyt_pool)

        u_xyt = _g1(u)
        v_xyt = _g1(v)
        p_xyt = _g1(p)

        du_dx = u_xyt[:, 0:1]; du_dy = u_xyt[:, 1:2]; du_dt = u_xyt[:, 2:3]
        dv_dx = v_xyt[:, 0:1]; dv_dy = v_xyt[:, 1:2]; dv_dt = v_xyt[:, 2:3]
        dp_dx = p_xyt[:, 0:1]; dp_dy = p_xyt[:, 1:2]

        forcing = float(A) * torch.sin(kw * xyt_pool[:, 1:2]).detach()
        mom_u = du_dt + u.detach() * du_dx + v.detach() * du_dy + dp_dx - forcing
        mom_v = dv_dt + u.detach() * dv_dx + v.detach() * dv_dy + dp_dy
        cont  = du_dx + dv_dy

        res_mag = (mom_u.detach() ** 2 + mom_v.detach() ** 2 + cont.detach() ** 2).squeeze(-1)
        _, top_idx = torch.topk(res_mag, n_top)
        selected = xyt_pool[top_idx].detach().cpu().numpy()
        if n_rand > 0:
            rxy, rt = ds.sample_physics_points(rng, n=n_rand, t_max=t_max, strategy="random")
            selected = np.concatenate([selected, np.concatenate([rxy, rt[:, None]], axis=1)], axis=0)
        result.append(selected.astype(np.float32))
    net.train()
    return result


def physics_points_at_step(
    step: int,
    start: int,
    end: int,
    ramp_steps: int,
    warmup_steps: int = 0,
) -> int:
    """What: 依訓練步數線性增加 physics collocation 點數（curriculum）。

    Why: 訓練初期 data loss 未收斂，大量 physics 點造成梯度衝突；
         先等 warmup_steps 讓模型有基本擬合，再花 ramp_steps 步逐步增加點數。

    Args:
        step:          當前步數（從 1 開始）
        start:         初始點數（warmup 期間及 ramp 起始值）
        end:           最終點數
        ramp_steps:    從 start 線性增長至 end 所需步數；0 = warmup 後立即用 end
        warmup_steps:  開始 ramp 前的等待步數（此期間固定用 start）
    Returns:
        當前步數對應的整數點數
    """
    if step <= warmup_steps:
        return start
    if ramp_steps <= 0:
        return end
    progress = min((step - warmup_steps) / ramp_steps, 1.0)
    return int(round(start + (end - start) * progress))


def physics_weight_at_step(
    step: int,
    final_weight: float,
    warmup_steps: int,
    ramp_steps: int,
) -> float:
    """What: 線性 physics warmup/ramp。"""
    if step < 1:
        raise ValueError(f"step 必須從 1 開始，收到 {step}")
    if final_weight < 0.0:
        raise ValueError(f"final_weight 不可為負，收到 {final_weight}")
    if warmup_steps < 0 or ramp_steps < 0:
        raise ValueError(
            f"warmup_steps / ramp_steps 不可為負，收到 {warmup_steps}, {ramp_steps}"
        )
    if final_weight == 0.0:
        return 0.0
    if step <= warmup_steps:
        return 0.0
    if ramp_steps == 0:
        return final_weight
    progress = min((step - warmup_steps) / ramp_steps, 1.0)
    return float(final_weight * progress)
