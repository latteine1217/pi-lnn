"""Loss-side machinery: GradNorm dynamic loss weighting + AL-continuity + sparse-channel prediction."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from pi_con.operator import LiquidOperator  # noqa: F401  (used in annotation only)


_DEFAULT_TASK_LAYOUTS: dict[int, list[str]] = {
    4: ["data", "ns_u", "ns_v", "cont"],
    5: ["data", "ns_u", "ns_v", "cont", "bc"],
}


class GradNormWeights(nn.Module):
    """What: GradNorm（Chen et al., 2018）的可學習 task 權重。

    Why: 直接管理 [data, ns_u, ns_v, cont(, bc)] 等 task 的權重比例。
         以 w_data = 1 為基準，physics weights 表達相對 data 的比例，
         讓 GradNorm 能真正動態調整 physics/data 平衡，而非受限於 sum=const 的固定比例。
         初始值 [1.0, 0.01, 0.01, 0.01] → physics 從 1% data 出發，由 GradNorm 自行決定是否加強。

    AL-continuity（spec v4）相容：當 use_augmented_lagrangian=true 時 AL term 完全在
    GradNorm 之外處理，不應出現在 task_names 中（"al" 不允許）。
    """

    def __init__(
        self,
        init_weights: list[float],
        task_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        w = torch.tensor(init_weights, dtype=torch.float32)
        self.log_weights = nn.Parameter(torch.log(w))
        # task_names: 顯式指定 → 驗長度；未指定 → 由 init_weights 長度推斷 4/5-task 預設 layout
        if task_names is None:
            n = len(init_weights)
            if n not in _DEFAULT_TASK_LAYOUTS:
                raise ValueError(
                    f"未指定 task_names 時 init_weights 長度必須 ∈ {sorted(_DEFAULT_TASK_LAYOUTS)}，收到 {n}"
                )
            task_names = list(_DEFAULT_TASK_LAYOUTS[n])
        else:
            if len(task_names) != len(init_weights):
                raise ValueError(
                    f"task_names 長度 ({len(task_names)}) 與 init_weights ({len(init_weights)}) 不符"
                )
        self.task_names: tuple[str, ...] = tuple(task_names)

    @property
    def weights(self) -> torch.Tensor:
        return torch.exp(self.log_weights)

    def index_of(self, name: str) -> int:
        """What: 回傳 task 名稱對應 index；找不到 raise KeyError。"""
        try:
            return self.task_names.index(name)
        except ValueError as e:
            raise KeyError(f"task name {name!r} 不在 {self.task_names}") from e

    def __contains__(self, name: str) -> bool:
        return name in self.task_names

    def normalize_to_data_(self) -> None:
        """固定 w_data = 1，其餘 task 表示相對 data 的比例。

        Why: 取代 sum=const 的歸一化，避免 data weight 膨脹導致 physics weights
             被壓縮到微小量級，使 GradNorm 真正能改變 physics/data 的比例關係。
        """
        with torch.no_grad():
            self.log_weights -= self.log_weights[0].clone()


class AugmentedLagrangianMultiplier(nn.Module):
    """What: 單一 scalar penalty multiplier (λ, ρ) for one scalar constraint C ≥ 0.

    Why: continuity 是純 scalar、無 gauge 自由度。因為 C = mean((∇·u)²) ≥ 0，
         本實作其實是 accumulated-multiplier penalty schedule，不是 textbook 對
         signed g(x)=0 的 AL；λ 單調非減直到 hit clip。命名沿用 AL 以對齊文獻。
         不繼承 nn.Parameter（λ 不靠 gradient 更新，靠 dual ascent）。

    參考 spec: docs/superpowers/specs/2026-05-04-al-continuity-design.md §3
    """

    def __init__(
        self,
        init_lambda: float = 0.0,
        rho: float = 1.0,
        lambda_clip: float = 10.0,
        ema_momentum: float = 0.5,
    ) -> None:
        super().__init__()
        # 全部用 buffer（含 _initialized）→ state_dict 完整保存，resume 不會 EMA cold-start
        self.register_buffer("lambda_", torch.tensor(float(init_lambda)))
        self.register_buffer("ema_C", torch.tensor(0.0))
        self.register_buffer("_initialized", torch.tensor(False))
        # rho / lambda_clip / ema_momentum 為靜態 hyperparameter（v1 不 schedule）→ Python float
        self.rho = float(rho)
        self.lambda_clip = float(lambda_clip)
        self.ema_momentum = float(ema_momentum)

    def loss_term(self, C: torch.Tensor) -> torch.Tensor:
        """λ·C + (ρ/2)·C² — primal-side differentiable term.

        Gradient 只流過 C（lambda_ 是 buffer 不接 autograd，rho 是 Python float）。
        """
        return self.lambda_ * C + 0.5 * self.rho * C ** 2

    @torch.no_grad()
    def update(self, C_batch: torch.Tensor) -> None:
        """Dual update：λ ← clip(λ + ρ·C̃, 0, Λ)。

        - C ≥ 0 always（squared divergence），故 lower clip = 0（負 λ 物理上無意義）
        - 用 out-of-place clamp 後寫回 lambda_，避免 in-place on temp 失效
        - C_batch 必須 scalar 或 numel==1 tensor；其他 shape 由 reshape 自然 raise
        """
        c_val = C_batch.detach().reshape(()).to(self.ema_C.device, self.ema_C.dtype)
        if self.ema_momentum > 0.0 and bool(self._initialized.item()):
            self.ema_C.mul_(self.ema_momentum).add_(c_val * (1.0 - self.ema_momentum))
        else:
            self.ema_C.copy_(c_val)
            self._initialized.fill_(True)
        new_lambda = (self.lambda_ + self.rho * self.ema_C).clamp(0.0, self.lambda_clip)
        self.lambda_.copy_(new_lambda)


def _gradnorm_step(
    gn_weights: GradNormWeights,
    losses: list[torch.Tensor],
    ref_params: list[torch.Tensor],
    ema_momentum: float = 0.5,
    min_weight: float = 0.0,
    max_weight: float = 0.0,
) -> None:
    """What: 一次 GradNorm 權重更新（直接公式 + EMA，無 optimizer）。

    Why: 各 task 的目標權重直接由梯度範數反比公式算出，再以 EMA 平滑寫回。
         不需要 create_graph=True 或獨立 optimizer，計算成本低且無 lr 調參問題。
         有效步長 = (1 - ema_momentum)，momentum=0.5 → 每次更新走 50%。

    公式：
        G_i      = ||∇_W L_i||_2          （各 task 對 ref_params 的梯度範數）
        mean_G   = mean(G_i)
        w_i_raw  = mean_G / (G_i + 1e-5 * mean_G)   （梯度範數小 → 權重大）
        w_i_norm = w_i_raw / w_i_raw[0]              （data 為基準，w_data = 1）
        w_new    = momentum * w_old + (1 - momentum) * w_i_norm
        w_new    = clamp(w_new, min_weight, max_weight)（sanity bounds，data weight 不受限）

    Args:
        losses:        [l_data, l_ns_u, l_ns_v, l_cont(, l_bc)]（retain_graph=True 保留計算圖）
        ref_params:    reference layer 參數（trunk_out.weight + bias）
        ema_momentum:  EMA 動量；有效步長 = 1 - ema_momentum
        min_weight:    所有 task weight 下限（除 data 外，data 永遠 = 1）。
                       Why: 防 GradNorm 自我催化 pathology —— 某 task gradient 結構性弱
                            （如 PINN cont 在 distance feature 加強 NS 後相對更小）會被
                            反比公式 deprioritize 到趨近 0，該約束於 evaluate 場上崩。
                            cylinder_007/008 驗證：w_cont 跌到 0.047 → div_L2=2.95（DNS 0.03）。
                            Floor 0.05~0.1 是合理 sanity guardrail（不取代動態調）。
                            0.0 = 不加 floor（向後相容預設）。
        max_weight:    physics task weight 上限（data 永遠 = 1，不受限）。
                       Why: 對偶 min_weight，防 GradNorm 把 ns 權重升太高壓抑 data 學習。
                            EXP-071 驗證：3-task GradNorm + AL 把 ns_u/ns_v 推到 0.30 級別 →
                            data 權重相對被壓 → KE rel-err 從 7.80% → 14.57%。
                            EXP-075 設計用 max_weight=0.20 作 cap，期望兼顧 div 與 KE。
                            0.0 = 不加 cap（向後相容預設）。
                            **僅 cap index ≥ 1（physics tasks），data (index 0) 永保 = 1**。
    """
    ws_old = gn_weights.weights.detach().clone()

    # 計算各 task 對 ref_params 的梯度範數（不需要 create_graph）
    G = []
    for l_i in losses:
        grads = torch.autograd.grad(
            l_i, ref_params,
            retain_graph=True, create_graph=False, allow_unused=True,
        )
        g_norms = [g.reshape(-1).norm() for g in grads if g is not None]
        G.append(torch.stack(g_norms).norm() if g_norms else torch.zeros(1, device=ws_old.device).squeeze())

    G_stack = torch.stack(G).detach()
    mean_G = G_stack.mean()

    # 目標權重：梯度範數越小 → 權重越大（讓各 task 梯度貢獻拉齊）
    w_raw = mean_G / (G_stack + 1e-5 * mean_G)
    # 以 data 為基準歸一化：w_data = 1，physics 表達相對比例
    w_computed = w_raw / w_raw[0].clamp(min=1e-8)

    # EMA：new = momentum * old + (1 - momentum) * computed
    w_new = ema_momentum * ws_old + (1.0 - ema_momentum) * w_computed
    # Sanity floor / cap：防止 GradNorm pathology
    # 對 data weight (index 0) 不套 max_weight cap（data 永遠 = 1，cap 它無意義且會破壞歸一化）
    if min_weight > 0.0:
        w_new = torch.clamp(w_new, min=min_weight)
    if max_weight > 0.0:
        w_phys = w_new[1:].clamp(max=max_weight)  # physics tasks (index ≥ 1)
        w_new = torch.cat([w_new[:1], w_phys])
    with torch.no_grad():
        gn_weights.log_weights.copy_(torch.log(w_new.clamp(min=1e-8)))


def gradient_cosine_diagnostic(
    losses_by_name: dict[str, torch.Tensor],
    ref_params: list[torch.Tensor],
) -> dict[str, float]:
    """What: 量化各 task gradient（對 ref_params）之間的 cosine similarity。

    Why: PCGrad gradient surgery 只在 task 梯度方向衝突（cosine < 0）時才有作用。
         導入 PCGrad 前必須先用數據證明衝突真實存在，否則屬過度設計（CLAUDE.md
         Simplicity）。本函式沿用 GradNorm 的 trunk_out 參考層（autograd.grad，
         成本與 _gradnorm_step 同級），純觀測：不寫任何 .grad、不改訓練行為。

    呼叫契約：須在 backward 之前呼叫，且各 loss 的計算圖仍存活
             （內部用 retain_graph=True，呼叫端後續 backward 不受影響）。

    回傳（全為 python float，已 detach）：
        cos_data_phys    : cos(g_data, g_ns_u + g_ns_v + g_cont)  ← 2-group PCGrad 主指標
        cos_data_<task>  : cos(g_data, g_<task>)，每個非 data task 各一
        gnorm_<task>     : ‖g_<task>‖₂（量級失衡 context，與 GradNorm 對照用）

    若 ref_params 對某 loss 無貢獻（grad=None）以零向量代入，cosine 退化為 0。
    """
    flat: dict[str, torch.Tensor] = {}
    for name, l_i in losses_by_name.items():
        grads = torch.autograd.grad(
            l_i, ref_params,
            retain_graph=True, create_graph=False, allow_unused=True,
        )
        flat[name] = torch.cat([
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for g, p in zip(grads, ref_params)
        ]).detach()

    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        denom = a.norm() * b.norm()
        return float((a @ b) / denom) if denom > 1e-12 else 0.0

    out: dict[str, float] = {f"gnorm_{n}": float(v.norm()) for n, v in flat.items()}
    if "data" in flat:
        phys = [flat[n] for n in ("ns_u", "ns_v", "cont") if n in flat]
        if phys:
            phys_agg = phys[0]
            for p in phys[1:]:
                phys_agg = phys_agg + p
            out["cos_data_phys"] = _cos(flat["data"], phys_agg)
        for n, v in flat.items():
            if n != "data":
                out[f"cos_data_{n}"] = _cos(flat["data"], v)
    return out


def pcgrad_two_group_backward(
    data_loss: torch.Tensor,
    phys_loss: torch.Tensor,
    params: list[torch.Tensor],
    extra_loss: torch.Tensor | None = None,
) -> float:
    """What: 2-group 對稱 PCGrad（Yu et al., 2020）— 把 data 與 physics 兩組梯度
            的方向衝突分量投影掉，結果寫回 params 的 .grad，取代 l_total.backward()。

    Why: cosine 診斷（EXP-cosdiag）顯示 data/physics 梯度在 trunk_out 上大致正交、
         mean≈0、僅偶發 cos<0。PCGrad 只在 cos<0 的 step 生效（削掉互相抵銷分量），
         其餘 step 為 identity。採 2-group（physics task 已合併為一組）對應 cosine
         診斷主軸，且只需 2 次 full-graph backward（per-task 4-group 需 4 次）。

    對稱投影（2-task 無 order 依賴，用原始梯度同時投影）：
        若 g_d · g_p < 0：
            g_p ← g_p − (g_d·g_p / ‖g_d‖²) g_d
            g_d ← g_d − (g_d·g_p / ‖g_p‖²) g_p   （用原始 g_d·g_p）
        否則不變。
    最終 grad = g_d_pc + g_p_pc，覆寫 p.grad。

    AL / BC 等不進投影的項由 extra_loss 帶入（比照 AL-GradNorm 解耦原則，spec v4 §5）：
    在投影寫回 .grad 後再 backward accumulate。

    Args:
        data_loss: 已加權 data group scalar（ws[data]·l_data），帶計算圖。
        phys_loss: 已加權 physics group scalar（Σ ws[i]·l_i over ns_u/ns_v/cont(/bc)）。
        params:    net.parameters() list；投影在整個共享參數向量上定義。
        extra_loss: optional 非投影項（AL term + 固定 weight BC）；None = 無。

    Returns:
        cos(g_data, g_phys)（float，供 log）。

    [RISK: 記憶體] 兩次 full-graph autograd.grad（phys 含二階物理圖）期間二階圖駐留 ~×2；
        flatten 梯度向量 2 份（P×4B 各一），P~3M 時約 25MB，相對 activation 圖可忽略。
    """
    need_retain = extra_loss is not None
    g_d = torch.autograd.grad(
        data_loss, params, retain_graph=True, create_graph=False, allow_unused=True,
    )
    g_p = torch.autograd.grad(
        phys_loss, params, retain_graph=need_retain, create_graph=False, allow_unused=True,
    )
    fd = torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for g, p in zip(g_d, params)
    ])
    fp = torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for g, p in zip(g_p, params)
    ])

    dot = fd @ fp
    if dot < 0:
        fd_pc = fd - (dot / (fp @ fp).clamp(min=1e-12)) * fp
        fp_pc = fp - (dot / (fd @ fd).clamp(min=1e-12)) * fd
    else:
        fd_pc, fp_pc = fd, fp
    g_total = fd_pc + fp_pc

    # 覆寫 p.grad（取代 backward；optimizer.zero_grad 已清空，此處直接賦值）
    offset = 0
    for p in params:
        n = p.numel()
        p.grad = g_total[offset:offset + n].view_as(p).clone()
        offset += n

    # 非投影項：在圖仍存活時 backward，accumulate 進 p.grad（+= 行為正確）
    if extra_loss is not None:
        extra_loss.backward()

    denom = fd.norm() * fp.norm()
    return float(dot / denom) if denom > 1e-12 else 0.0


def observed_channel_prediction(
    net: "LiquidOperator",
    xy: torch.Tensor,
    t_q: torch.Tensor,
    c_obs: torch.Tensor,
    observed_channel_names: tuple[str, ...],
    observed_channel_mean: torch.Tensor,
    observed_channel_std: torch.Tensor,
    h_states: torch.Tensor,
    s_time: torch.Tensor,
    sensor_pos: torch.Tensor,
    body_distance: torch.Tensor | None = None,
) -> torch.Tensor:
    """What: 依實際觀測通道名稱產生對應預測值。

    Why: sparse-data 主線目前只監督真實可量測的 u,v。p 僅保留在 physics residual
         內部使用，避免在資料項中引入不可量測通道。
         單次 query_decoder 呼叫處理所有 N 個樣本（u+v 混合），
         再以向量化 normalize 取代 per-channel loop，消除一次重複 trunk forward。

    body_distance: optional [N] tensor，model use_hard_body_bc=True 時必須提供
                   （differentiable，給 forward 出口 gate 用）。
    """
    raw_pred = net.query_decoder(
        xy, t_q, c_obs, h_states, s_time, sensor_pos, body_distance=body_distance,
    ).squeeze(1)
    mean_vec = observed_channel_mean[c_obs]
    std_vec = observed_channel_std[c_obs]
    return (raw_pred - mean_vec) / std_vec


class WakeAmplitudePrior:
    """Wake-amplitude envelope prior — sensor-derived 單側能量上界（工程可遷移）。

    What:
        從 K 個 wake sensor 的 (u, v) 時序計算每點能量包絡 E_k = p_quantile_t(u²+v²)，
        在 wake-local 區域對重建場施加單側上界懲罰：
            L = mean_{x∈Ω_wake, t}[ max(0, e_θ(x,t) − γ·Ê_obs(x))² ]
        其中 e_θ = u_θ²+v_θ²（物理單位），Ê_obs(x) 為 sensor envelope 的高斯加權內插。

    Why（codex 辯論 Round 2 共識 + CEXP-045 pre-check）:
        cylinder 所有失敗的共同 failure mode 是 over-energy（ke_pred/ref 2–7×），
        且與 divergence 正交（CEXP-043 div 低 KE 仍爆）。唯一直擊 over-energy 的
        no-new-sensor lever 是「直接約束能量尺度」。本 prior 用單側上界：
          - 不獎勵低能量（不會 trivial collapse → mean-flow；data loss 仍逼真實振幅）
          - 不新增 pointwise BC、不碰 body、不強化 NS、不用 DNS full field
          - envelope 只用 K=100 sensor 觀測值 → 工程可遷移

    工程可遷移性:
        所有量（per-sensor envelope、sensor 位置、bandwidth）只來自 sensor u,v 時序，
        無 DNS 全場。符合 ENGINEERING_VISION（現場只有稀疏 sensor + PDE）。

    單位協定:
        ds.sensor_vals 已正規化（zero-mean/unit-std）；uvp_fn（use_physics_denormalization=true）
        回傳物理 (u,v)。故 envelope 必須先反正規化 sensor_vals 回物理：
        u_phys = u_norm·std + mean，才能與 e_θ 同尺度比較。
    """

    def __init__(
        self,
        sensor_vals: torch.Tensor,   # [K, T, C] normalized
        sensor_pos: torch.Tensor,    # [K, 2] normalized domain coords
        u_idx: int,
        v_idx: int,
        u_mean: float, u_std: float,
        v_mean: float, v_std: float,
        percentile: float = 0.95,
        gamma: float = 1.5,
        radius_scale: float = 2.0,
        sigma_scale: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        dev = device if device is not None else sensor_pos.device
        self.pos = sensor_pos.to(dev)            # [K, 2]
        self.gamma = float(gamma)

        # 反正規化回物理單位後計算 per-sensor 能量包絡
        u_phys = sensor_vals[:, :, u_idx] * u_std + u_mean   # [K, T]
        v_phys = sensor_vals[:, :, v_idx] * v_std + v_mean   # [K, T]
        e = (u_phys ** 2 + v_phys ** 2).to(dev)              # [K, T]
        # p_quantile over time：保留 shedding peak，不退化成 mean-flow envelope
        self.E = torch.quantile(e, float(percentile), dim=1)  # [K]

        # sensor 間 median nearest-neighbor distance → bandwidth / region radius
        D = torch.cdist(self.pos, self.pos)       # [K, K]
        D.fill_diagonal_(float("inf"))
        nn_dist = D.min(dim=1).values             # [K]
        nn_med = nn_dist.median().clamp(min=1e-6)
        self.sigma = float(sigma_scale) * nn_med
        self.radius = float(radius_scale) * nn_med

    def sample_points(self, rng, n: int, t_lo: float, t_hi: float,
                      device: torch.device) -> torch.Tensor:
        """在 sensor 周圍 σ-jitter 取 n 個 wake-local collocation 點 [n, 3]=(x,y,t)。

        Why jitter-around-sensor 而非 uniform+mask：保證點落在 Ω_wake（sensor 覆蓋區），
        避免外推到 inlet/body/outlet（重演 BC/sensor 衝突）。
        """
        K = self.pos.shape[0]
        idx = rng.integers(0, K, size=n)
        base = self.pos[idx]                                          # [n, 2]
        jitter = torch.tensor(
            rng.normal(0.0, float(self.sigma), size=(n, 2)),
            dtype=torch.float32, device=device,
        )
        xy = (base.to(device) + jitter).clamp(0.0, 1.0)
        t = torch.tensor(
            rng.uniform(t_lo, t_hi, size=(n,)), dtype=torch.float32, device=device
        )
        return torch.cat([xy, t[:, None]], dim=1)

    def _env_obs(self, xy: torch.Tensor) -> torch.Tensor:
        """高斯加權內插 sensor envelope → Ê_obs(x) [N]。"""
        d2 = torch.cdist(xy, self.pos) ** 2                          # [N, K]
        w = torch.softmax(-d2 / (2.0 * self.sigma ** 2), dim=1)      # [N, K]
        return (w * self.E[None, :]).sum(dim=1)                      # [N]

    def cap_loss(self, uvp_fn, xyt: torch.Tensor) -> torch.Tensor:
        """單側上界懲罰 mean(relu(e_θ − γ·Ê_obs)²)。uvp_fn 回傳物理 [N,3]。"""
        uvp = uvp_fn(xyt)
        e_theta = uvp[:, 0] ** 2 + uvp[:, 1] ** 2                    # [N] physical
        cap = self.gamma * self._env_obs(xyt[:, :2])                 # [N]
        excess = torch.relu(e_theta - cap)
        return torch.mean(excess ** 2)
