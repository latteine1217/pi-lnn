"""PINN causal weighting (Wang et al. 2022, "Respecting causality is all you need").

What:
    將 collocation 點按時間切 N 個 bin，計算 per-bin 殘差後施加因果權重
        w_t = exp(-eps * cumsum(prev L_t')).detach()
    讓 t=0 收斂前 t>0 的 physics loss 不主導梯度。

Why:
    chaotic flow（Re=10000 Kolmogorov）的 Lyapunov 不穩定使早期誤差指數放大；
    若 t=0 與 t=5 的 physics loss 對等加權，模型可能犧牲 t=0 的精度去
    平均誤差。Causal 加權強制時間先後順序，呼應「先有 IC，才有後續演化」。

    與既有 t_early_weight 相比：
    - t_early_weight 是硬閾值（t<=threshold 給定固定 multiplier）
    - causal weight 是連續且自適應（依當前殘差動態決定權重曲線）
    - 兩者同時開啟會雙重加權，建議用 causal 時將 t_early_weight=1.0
"""
from __future__ import annotations

import torch


def _per_bin_mean(
    values: torch.Tensor,
    bin_idx: torch.Tensor,
    num_bins: int,
) -> torch.Tensor:
    """What: scatter-mean，把 [N] 殘差按 bin index 平均成 [num_bins]。

    Why: torch.scatter_reduce 'mean' 在 MPS 不支援；改用 scatter_add 計算
         分子分母再相除，並對空 bin 退化為 0（不貢獻 loss、不污染 cumsum）。
    """
    sums = torch.zeros(num_bins, device=values.device, dtype=values.dtype)
    counts = torch.zeros(num_bins, device=values.device, dtype=values.dtype)
    sums = sums.scatter_add(0, bin_idx, values)
    counts = counts.scatter_add(0, bin_idx, torch.ones_like(values))
    safe_counts = counts.clamp(min=1.0)
    means = sums / safe_counts
    # 空 bin 應為 0，避免被 cumsum 累積
    return torch.where(counts > 0, means, torch.zeros_like(means))


def causal_weighted_residual_loss(
    residuals: list[torch.Tensor],
    times: torch.Tensor,
    num_bins: int,
    eps: float,
    t_min: float = 0.0,
    t_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """What: 對多個物理殘差項共同施加因果權重，回傳加權後總 loss + 權重向量。

    Why:
        因果性是時間屬性，與物理量（mom_u/mom_v/cont）無關；同一個 w_t 同時
        加權所有殘差項，保持 momentum/continuity 之間的相對權重不變
        （讓 GradNorm 仍能正確平衡 task）。

    Args:
        residuals: list of [N, 1] or [N] 殘差張量（同 collocation 點集）
        times: [N] collocation 點的時間座標（已從 xyt[:, 2] 取出）
        num_bins: 時間 bin 數；建議 16~32（太大 cumsum 不穩，太小喪失因果分辨）
        eps: 因果嚴格度；0 退化為均勻平均，∞ 只訓練 t=t_min 的 bin
        t_min, t_max: bin 邊界；t_max=None 則用 times.max()

    Returns:
        weighted_losses: [num_residuals] tensor，每個殘差項的 causal-weighted mean
        weights: [num_bins] detached weight vector（供 logging）

    Notes:
        weights 必須 detach，避免 cumsum 鏈式反傳造成梯度爆炸（Wang 2022 §3.2）。
    """
    if num_bins < 2:
        raise ValueError(f"num_bins 必須 >= 2，收到 {num_bins}")
    if eps < 0:
        raise ValueError(f"eps 必須 >= 0，收到 {eps}")
    if not residuals:
        raise ValueError("residuals 不可為空")

    times = times.reshape(-1)
    t_hi = float(times.max().item()) if t_max is None else float(t_max)
    t_lo = float(t_min)
    # 退化情況：(a) bin 範圍非正、(b) 所有點同一時間 → 均勻平均，無因果結構
    same_time = float(times.max().item() - times.min().item()) < 1e-12
    if t_hi <= t_lo or same_time:
        means = [r.reshape(-1).pow(2).mean() for r in residuals]
        return (
            torch.stack(means),
            torch.ones(num_bins, device=times.device, dtype=times.dtype),
        )

    # 線性 binning；clamp 防止浮點誤差使 idx == num_bins
    norm = (times - t_lo) / (t_hi - t_lo)
    bin_idx = (norm * num_bins).long().clamp(0, num_bins - 1)

    # 計算 per-bin 殘差平方均值（與 GradNorm 介面相容：仍是 mean(r^2)）
    per_bin_sq: list[torch.Tensor] = []
    for r in residuals:
        sq = r.reshape(-1) ** 2
        per_bin_sq.append(_per_bin_mean(sq, bin_idx, num_bins))

    # 因果權重：w_t = exp(-eps * sum_{t' < t} mean residual at t')
    # 用「所有殘差項的和」做 cumsum（單一權重曲線，不偏好任何一個 task）
    total_per_bin = torch.stack(per_bin_sq, dim=0).sum(dim=0)  # [num_bins]
    cum_prev = torch.cat([
        torch.zeros(1, device=total_per_bin.device, dtype=total_per_bin.dtype),
        total_per_bin[:-1].cumsum(dim=0),
    ])
    weights = torch.exp(-eps * cum_prev).detach()  # 重點：detach 防梯度爆炸

    # 加權平均：sum(w_t * L_t) / sum(w_t)
    weight_sum = weights.sum().clamp(min=1e-12)
    weighted = torch.stack([
        (weights * pb).sum() / weight_sum
        for pb in per_bin_sq
    ])
    return weighted, weights
