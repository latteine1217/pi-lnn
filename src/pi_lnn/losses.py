"""Loss-side machinery: GradNorm dynamic loss weighting + sparse-channel prediction."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from pi_lnn.operator import LiquidOperator  # noqa: F401  (used in annotation only)


class GradNormWeights(nn.Module):
    """What: GradNorm（Chen et al., 2018）的可學習 task 權重。

    Why: 直接管理 [data, ns_u, ns_v, cont] 四個 task 的權重比例。
         以 w_data = 1 為基準，physics weights 表達相對 data 的比例，
         讓 GradNorm 能真正動態調整 physics/data 平衡，而非受限於 sum=const 的固定比例。
         初始值 [1.0, 0.01, 0.01, 0.01] → physics 從 1% data 出發，由 GradNorm 自行決定是否加強。
    """

    def __init__(self, init_weights: list[float]) -> None:
        super().__init__()
        w = torch.tensor(init_weights, dtype=torch.float32)
        self.log_weights = nn.Parameter(torch.log(w))

    @property
    def weights(self) -> torch.Tensor:
        return torch.exp(self.log_weights)

    def normalize_to_data_(self) -> None:
        """固定 w_data = 1，其餘 task 表示相對 data 的比例。

        Why: 取代 sum=const 的歸一化，避免 data weight 膨脹導致 physics weights
             被壓縮到微小量級，使 GradNorm 真正能改變 physics/data 的比例關係。
        """
        with torch.no_grad():
            self.log_weights -= self.log_weights[0].clone()


def _gradnorm_step(
    gn_weights: GradNormWeights,
    losses: list[torch.Tensor],
    ref_params: list[torch.Tensor],
    ema_momentum: float = 0.5,
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

    Args:
        losses:        [l_data, l_ns_u, l_ns_v, l_cont]（retain_graph=True 保留計算圖）
        ref_params:    reference layer 參數（trunk_out.weight + bias）
        ema_momentum:  EMA 動量；有效步長 = 1 - ema_momentum
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
) -> torch.Tensor:
    """What: 依實際觀測通道名稱產生對應預測值。

    Why: sparse-data 主線目前只監督真實可量測的 u,v。p 僅保留在 physics residual
         內部使用，避免在資料項中引入不可量測通道。
         單次 query_decoder 呼叫處理所有 N 個樣本（u+v 混合），
         再以向量化 normalize 取代 per-channel loop，消除一次重複 trunk forward。
    """
    raw_pred = net.query_decoder(xy, t_q, c_obs, h_states, s_time, sensor_pos).squeeze(1)
    mean_vec = observed_channel_mean[c_obs]
    std_vec = observed_channel_std[c_obs]
    return (raw_pred - mean_vec) / std_vec
