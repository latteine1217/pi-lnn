"""Loss-side machinery: GradNorm dynamic loss weighting + AL-continuity + sparse-channel prediction."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from pi_lnn.operator import LiquidOperator  # noqa: F401  (used in annotation only)


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
