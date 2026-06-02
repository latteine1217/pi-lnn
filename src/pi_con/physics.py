"""Physics losses: NS/Poisson residuals + Residual-Adaptive Refinement + scheduling."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from pi_con.operator import make_picon_model_fn_uvp
from pi_con.runtime import _grad


def unsteady_ns_residuals(
    uvp_fn: Callable,
    xyt: torch.Tensor,
    re: float,
    k_f: float | torch.Tensor = 4.0,
    A: float | torch.Tensor = 0.1,
    domain_length: float = 1.0,
    Lx: float = 1.0,
    Ly: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """What: 2D incompressible NS 的 primitive-variable momentum 與 continuity 殘差。

    Why: sparse-data 主線的實際觀測仍只有 u,v，但 momentum equation 需要壓力梯度。
         因此 p 回到模型的內部 physics 場，只參與 PDE 殘差，不作資料 supervision。
         uvp_fn 一次回傳 [N, 3]，相較舊版三個獨立 closure 共用 c-independent 計算
         並合併二階 autograd graph，數學上等價。

    Coordinate chain rule (Lx, Ly):
        Datasets 將座標正規化到 [0, 1]² 但 model output u, v 仍是物理 m/s。
        autograd 給的 du/dx_norm 是 per unit normalized x，需 chain rule 轉物理梯度：
            du/dx_phys  = du/dx_norm  / Lx
            d²u/dx²_phys = d²u/dx²_norm / Lx²
        Cylinder Lx=0.322, Ly=0.172（anisotropic）；Kolmogorov Lx=Ly=1.0（座標即物理）。
        若不修正，cylinder 黏性項 x、y 方向相對權重差 (Ly/Lx)² ≈ 0.29，
        模型實際在解一個被等比拉伸的偽 NS 方程。
    """
    uvp = uvp_fn(xyt)                                      # [N, 3]
    u = uvp[:, 0:1]
    v = uvp[:, 1:2]
    p = uvp[:, 2:3]
    u_xyt = _grad(u, xyt)
    v_xyt = _grad(v, xyt)
    p_xyt = _grad(p, xyt)
    # normalized 座標下的梯度（autograd 直接給）
    du_dx_n, du_dy_n, du_dt = u_xyt[:, 0:1], u_xyt[:, 1:2], u_xyt[:, 2:3]
    dv_dx_n, dv_dy_n, dv_dt = v_xyt[:, 0:1], v_xyt[:, 1:2], v_xyt[:, 2:3]
    dp_dx_n, dp_dy_n = p_xyt[:, 0:1], p_xyt[:, 1:2]
    du_dx2_n = _grad(du_dx_n, xyt)[:, 0:1]
    du_dy2_n = _grad(du_dy_n, xyt)[:, 1:2]
    dv_dx2_n = _grad(dv_dx_n, xyt)[:, 0:1]
    dv_dy2_n = _grad(dv_dy_n, xyt)[:, 1:2]
    # chain rule: 把 normalized 梯度轉物理梯度。Kolmogorov Lx=Ly=1 時無變化。
    sx, sy = float(Lx), float(Ly)
    du_dx, du_dy = du_dx_n / sx, du_dy_n / sy
    dv_dx, dv_dy = dv_dx_n / sx, dv_dy_n / sy
    dp_dx, dp_dy = dp_dx_n / sx, dp_dy_n / sy
    du_dx2, du_dy2 = du_dx2_n / (sx ** 2), du_dy2_n / (sy ** 2)
    dv_dx2, dv_dy2 = dv_dx2_n / (sx ** 2), dv_dy2_n / (sy ** 2)
    nu = 1.0 / float(re)
    # Forcing 用 normalized y（k_f cycles per [0,1] domain），與 Lx/Ly 無關。
    # 量級協定：A 為 physical 加速度單位 (m/s²)。修 C2 後 uvp_fn 已 denormalize，
    # advection u·∇u、∇p、ν·∇²u、forcing 都在 physical 量級，互相 commensurate。
    # k_f / A 可為 float 或 0-dim tensor（ForcingPrior 學習時）；不再 float(.) 強轉，
    # 否則會切斷 forcing 參數的 autograd path。
    forcing_wavenumber = (2.0 * torch.pi * k_f) / float(domain_length)
    forcing_x = A * torch.sin(forcing_wavenumber * xyt[:, 1:2])
    mom_u = du_dt + u * du_dx + v * du_dy + dp_dx - nu * (du_dx2 + du_dy2) - forcing_x
    mom_v = dv_dt + u * dv_dx + v * dv_dy + dp_dy - nu * (dv_dx2 + dv_dy2)
    cont = du_dx + dv_dy
    return mom_u, mom_v, cont


def channel_ns_residuals(
    uvwp_fn: Callable,
    xyzt: torch.Tensor,
    re: float,
    body_force_x: float = 0.0,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """What: 3D incompressible NS（channel flow）的 momentum(u,v,w) 與 continuity 殘差。

    Why: channel 是首個 3D + wall-bounded case。相較 2D `unsteady_ns_residuals`，
         多 w 分量、z 方向梯度、z 方向二階黏性，continuity 加 dw_dz。
         Forcing 改為 constant streamwise body force（取代 mean dP/dx，對應 Lethe
         flow-control 維持 U_b），物理值 body_force_x = u_τ²/h。
         不改既有 2D 函數 → Kolmogorov/Cylinder zero regression。

    Coordinate chain rule (Lx, Ly, Lz):
        xyzt 正規化到 [0,1]，output 為物理 m/s。
        d/dx_phys = d/dx_norm / Lx；d²/dx²_phys = d²/dx²_norm / Lx²（y,z 同理）。
        channel full domain: Lx=8π, Ly=2, Lz=3π。

    Args:
        uvwp_fn: callable，xyzt[N,4] -> [N,4]（u,v,w,p，物理量級）。
        xyzt:    [N,4] = (x,y,z,t) normalized，requires_grad=True。
        re:      Reynolds number，ν = 1/re。
        body_force_x: constant streamwise driving force（= u_τ²/h）。
        Lx,Ly,Lz: 物理域長度，chain-rule 尺度因子。

    Returns:
        (mom_u, mom_v, mom_w, cont)，各 [N,1]，物理量級。
    """
    uvwp = uvwp_fn(xyzt)                                   # [N, 4]
    u = uvwp[:, 0:1]
    v = uvwp[:, 1:2]
    w = uvwp[:, 2:3]
    p = uvwp[:, 3:4]
    u_g = _grad(u, xyzt)
    v_g = _grad(v, xyzt)
    w_g = _grad(w, xyzt)
    p_g = _grad(p, xyzt)
    # 一階 normalized（index 0=x, 1=y, 2=z, 3=t）
    du_dx_n, du_dy_n, du_dz_n, du_dt = u_g[:, 0:1], u_g[:, 1:2], u_g[:, 2:3], u_g[:, 3:4]
    dv_dx_n, dv_dy_n, dv_dz_n, dv_dt = v_g[:, 0:1], v_g[:, 1:2], v_g[:, 2:3], v_g[:, 3:4]
    dw_dx_n, dw_dy_n, dw_dz_n, dw_dt = w_g[:, 0:1], w_g[:, 1:2], w_g[:, 2:3], w_g[:, 3:4]
    dp_dx_n, dp_dy_n, dp_dz_n = p_g[:, 0:1], p_g[:, 1:2], p_g[:, 2:3]
    # 二階 normalized（沿各自方向）
    du_dx2_n = _grad(du_dx_n, xyzt)[:, 0:1]
    du_dy2_n = _grad(du_dy_n, xyzt)[:, 1:2]
    du_dz2_n = _grad(du_dz_n, xyzt)[:, 2:3]
    dv_dx2_n = _grad(dv_dx_n, xyzt)[:, 0:1]
    dv_dy2_n = _grad(dv_dy_n, xyzt)[:, 1:2]
    dv_dz2_n = _grad(dv_dz_n, xyzt)[:, 2:3]
    dw_dx2_n = _grad(dw_dx_n, xyzt)[:, 0:1]
    dw_dy2_n = _grad(dw_dy_n, xyzt)[:, 1:2]
    dw_dz2_n = _grad(dw_dz_n, xyzt)[:, 2:3]
    sx, sy, sz = float(Lx), float(Ly), float(Lz)
    # chain rule → 物理梯度
    du_dx, du_dy, du_dz = du_dx_n / sx, du_dy_n / sy, du_dz_n / sz
    dv_dx, dv_dy, dv_dz = dv_dx_n / sx, dv_dy_n / sy, dv_dz_n / sz
    dw_dx, dw_dy, dw_dz = dw_dx_n / sx, dw_dy_n / sy, dw_dz_n / sz
    dp_dx, dp_dy, dp_dz = dp_dx_n / sx, dp_dy_n / sy, dp_dz_n / sz
    du_dx2, du_dy2, du_dz2 = du_dx2_n / sx ** 2, du_dy2_n / sy ** 2, du_dz2_n / sz ** 2
    dv_dx2, dv_dy2, dv_dz2 = dv_dx2_n / sx ** 2, dv_dy2_n / sy ** 2, dv_dz2_n / sz ** 2
    dw_dx2, dw_dy2, dw_dz2 = dw_dx2_n / sx ** 2, dw_dy2_n / sy ** 2, dw_dz2_n / sz ** 2
    nu = 1.0 / float(re)
    lap_u = du_dx2 + du_dy2 + du_dz2
    lap_v = dv_dx2 + dv_dy2 + dv_dz2
    lap_w = dw_dx2 + dw_dy2 + dw_dz2
    adv_u = u * du_dx + v * du_dy + w * du_dz
    adv_v = u * dv_dx + v * dv_dy + w * dv_dz
    adv_w = u * dw_dx + v * dw_dy + w * dw_dz
    mom_u = du_dt + adv_u + dp_dx - nu * lap_u - body_force_x
    mom_v = dv_dt + adv_v + dp_dy - nu * lap_v
    mom_w = dw_dt + adv_w + dp_dz - nu * lap_w
    cont = du_dx + dv_dy + dw_dz
    return mom_u, mom_v, mom_w, cont


def pressure_poisson_residual(
    uvp_fn: Callable,
    xyt: torch.Tensor,
    Lx: float = 1.0,
    Ly: float = 1.0,
) -> torch.Tensor:
    """What: 2D incompressible 壓力 Poisson 方程殘差。

    Why: Primitive-variable NS 中 p 沒有資料監督，模型可藉由任意調整 p 來讓
         momentum residual 歸零，即使 u, v 是錯的（壓力自由度問題）。
         Poisson 方程從 NS + ∇·u=0 推導而來：
             ∇²p = -(∂u/∂x)² - (∂v/∂y)² - 2(∂u/∂y)(∂v/∂x)
         加入此約束後，p 必須與 u, v 的二階結構一致，壓力不再能自由漂移。

    數學推導：對動量方程取散度，使用 ∇·u=0 及 Kolmogorov forcing ∇·f=0 後得到。
    不需要任何額外觀測量，僅用模型輸出的 u, v, p via autograd。

    Coordinate chain rule: 與 unsteady_ns_residuals 相同，把 normalized 梯度轉物理。
    """
    uvp = uvp_fn(xyt)                                      # [N, 3]
    u = uvp[:, 0:1]
    v = uvp[:, 1:2]
    p = uvp[:, 2:3]
    u_xyt = _grad(u, xyt)
    v_xyt = _grad(v, xyt)
    p_xyt = _grad(p, xyt)
    du_dx_n = u_xyt[:, 0:1]
    du_dy_n = u_xyt[:, 1:2]
    dv_dx_n = v_xyt[:, 0:1]
    dv_dy_n = v_xyt[:, 1:2]
    dp_dx_n = p_xyt[:, 0:1]
    dp_dy_n = p_xyt[:, 1:2]
    dp_dx2_n = _grad(dp_dx_n, xyt)[:, 0:1]   # ∂²p/∂x² (normalized)
    dp_dy2_n = _grad(dp_dy_n, xyt)[:, 1:2]   # ∂²p/∂y² (normalized)
    sx, sy = float(Lx), float(Ly)
    du_dx, du_dy = du_dx_n / sx, du_dy_n / sy
    dv_dx, dv_dy = dv_dx_n / sx, dv_dy_n / sy
    dp_dx2, dp_dy2 = dp_dx2_n / (sx ** 2), dp_dy2_n / (sy ** 2)
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
    body_distance_fns: list | None = None,
) -> list[np.ndarray]:
    """What: RAR（Residual Adaptive Refinement）pool 更新。

    Why: 均勻隨機採樣可能長期錯過 t≈0 等高殘差區域；
         每隔 rar_update_freq 步，從大候選集中選 top 殘差點，
         讓 physics loss 集中在模型最難收斂的區域。
         保留 exploration_ratio 比例的隨機點，防止采样退化到固定幾個點。

    近似殘差：略去黏性項（Re=10000 時 ν=1e-4，黏性貢獻約 0.2%）；
    使用 create_graph=False，僅計算一階導數，完全避免二階 autograd 建圖。
    此近似只影響 pool 中的點排序，不影響訓練 loss 本身的精確性。

    Note: uvp_fn 透過 make_picon_model_fn_uvp 取得，內部已套
    physics_output_mean/std 反 normalization → mom_u, mom_v, cont 為物理單位，
    與訓練 path 的 unsteady_ns_residuals 一致。

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
        # Hard body BC：必須傳 body_distance_fn（differentiable SDF callable）。
        # operator.py 在 use_hard_body_bc=True 但 fn=None 時會 raise ValueError。
        # RAR pool 評估只用 first-order grads，body_distance_fn 不需 create_graph。
        uvp_fn = make_picon_model_fn_uvp(
            net, sensor_vals_list[i], sensor_pos_list[i],
            re_norm=ds.re_norm, sensor_time=sensor_time_list[i], device=device,
            body_distance_fn=body_distance_fns[i] if body_distance_fns is not None else None,
        )
        uvp = uvp_fn(xyt_pool)
        u = uvp[:, 0:1]
        v = uvp[:, 1:2]
        p = uvp[:, 2:3]

        # u/v/p 是同一次 forward `uvp` 的 slice，共用同一張 autograd graph。
        # 三次一階 grad 必須讓前兩次 retain_graph=True，否則第一次 backward 就釋放
        # saved tensors，第二次即 double-backward 崩潰（EXP-272 job 3721 根因）。
        def _g1(y: torch.Tensor, retain: bool) -> torch.Tensor:
            g = torch.autograd.grad(
                y, xyt_pool, torch.ones_like(y),
                create_graph=False, retain_graph=retain, allow_unused=True,
            )[0]
            return g if g is not None else torch.zeros_like(xyt_pool)

        u_xyt = _g1(u, retain=True)
        v_xyt = _g1(v, retain=True)
        p_xyt = _g1(p, retain=False)  # 最後一次釋放圖

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
