"""Positional and temporal encodings for sparse-sensor operator learning."""
from __future__ import annotations

import torch
import torch.nn as nn


def periodic_fourier_encode(z: torch.Tensor, domain_length: float, n_harmonics: int) -> torch.Tensor:
    """What: 對 2D 座標 (x, y) 做確定性多諧波週期 Fourier 編碼。

    Why: 取代 RFF 的隨機頻率抽樣，改用整數倍 2π/L 頻率。
         1) 嚴格滿足 [0,L]^2 週期邊界條件（與 jaxpi PeriodEmbs 相同設計）。
         2) x/y 軸獨立編碼，消除 RFF 隨機角度偏差所造成的 x-stripe 偽影。
         3) 無隨機性，結果與 seed 無關。

    Returns:
        [N, 4 * n_harmonics]
        排列：[sin(2πk x/L), cos(2πk x/L), sin(2πk y/L), cos(2πk y/L)] for k=1..n_harmonics
    """
    x = z[:, 0:1]  # [N, 1]
    y = z[:, 1:2]  # [N, 1]
    ks = torch.arange(1, n_harmonics + 1, device=z.device, dtype=z.dtype)  # [H]
    cx = (2.0 * torch.pi / domain_length) * ks * x  # [N, H]
    cy = (2.0 * torch.pi / domain_length) * ks * y  # [N, H]
    # stack → [N, H, 4]，reshape 後順序與原版一致：[sin_x_k1, cos_x_k1, sin_y_k1, cos_y_k1, ...]
    return torch.stack([cx.sin(), cx.cos(), cy.sin(), cy.cos()], dim=2).reshape(z.shape[0], -1)


class LearnableFourierEmb(nn.Module):
    """What: PeriodEmbs(k=1) → 可學習 Fourier 投影 的級聯，等價於 jaxpi
    Embedding(periodicity=set, fourier_emb=set)。

    Why: 為 [0,L]² 週期 BC 而設。`period_enc` 強制把 (x,y) 映射到週期循環，
         使 x=0 與 x=L 編碼恆等（驗證見 tests/test_pos_enc_optimization.py）。
         僅適用週期問題（如 Kolmogorov）；非週期問題（如 cylinder）請改用 FourierEmbs。

    Args:
        embed_dim: 輸出維度（需為偶數）。
        init_sigma: 投影矩陣初始化標準差（對應 jaxpi embed_scale=2.0）。
    """

    def __init__(self, embed_dim: int, init_sigma: float = 2.0) -> None:
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim 必須為偶數，收到 {embed_dim}")
        self.proj = nn.Linear(4, embed_dim // 2, bias=False)
        nn.init.normal_(self.proj.weight, std=init_sigma)

    def forward(self, xy: torch.Tensor, domain_length: float) -> torch.Tensor:
        c = 2.0 * torch.pi / domain_length
        x, y = xy[:, 0:1], xy[:, 1:2]
        period_enc = torch.cat(
            [torch.sin(c * x), torch.cos(c * x), torch.sin(c * y), torch.cos(c * y)],
            dim=-1,
        )  # [N, 4]
        proj = self.proj(period_enc)  # [N, embed_dim // 2]
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # [N, embed_dim]


class FourierEmbs(nn.Module):
    """What: 真 RFF——對原始座標做高斯隨機投影 + sin/cos，對應 jaxpi
    archs.py:83-97 FourierEmbs。

    Why: 適用非週期域（如 cylinder wake，x=0 與 x=L 物理意義截然不同）。
         與 LearnableFourierEmb 的關鍵差別在於不預先 sin/cos 週期化，
         因此能區分 x=0 與 x=L。

    Args:
        embed_dim: 輸出維度（需為偶數）。
        input_dim: 輸入座標維度（預設 2D）。
        init_sigma: 投影矩陣初始化標準差，控制有效頻率帶寬。
    """

    def __init__(self, embed_dim: int, input_dim: int = 2, init_sigma: float = 2.0) -> None:
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim 必須為偶數，收到 {embed_dim}")
        self.proj = nn.Linear(input_dim, embed_dim // 2, bias=False)
        nn.init.normal_(self.proj.weight, std=init_sigma)

    def forward(self, xy: torch.Tensor, domain_length: float | None = None) -> torch.Tensor:
        # domain_length 簽名與 LearnableFourierEmb 對齊但不使用——
        # RFF 的有效頻率由 init_sigma 決定，與 domain 大小無關。
        del domain_length
        proj = self.proj(xy)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


def temporal_phase_anchor(t: torch.Tensor, T_total: float, n_harmonics: int = 2) -> torch.Tensor:
    """What: 產生絕對時間的確定性 temporal-phase-anchor 特徵。

    Why: trunk 目前只有 dt_to_query（相對時間），在 chaotic 流場中，
         同樣的 dt 但不同絕對時間 t 的動態完全不同。
         注入 sin/cos(2π n t / T_total) 提供絕對時間定位，
         讓模型可以區分「t=0.1 的流場狀態」與「t=4.1 的流場狀態」。
         與空間的 forcing_phase_anchor 對稱設計。

    Args:
        t: [N, 1] 絕對時間
        T_total: 模擬總時長，用於正規化
        n_harmonics: 諧波數；每個諧波貢獻 sin/cos 共 2 維

    Returns:
        [N, 2 * n_harmonics]
    """
    ns = torch.arange(1, n_harmonics + 1, device=t.device, dtype=t.dtype)  # [H]
    angles = (2.0 * torch.pi / T_total) * ns * t  # [N, H]，t 為 [N, 1] 廣播
    # stack → [N, H, 2]，reshape 後順序與原版一致：[sin_n1, cos_n1, sin_n2, cos_n2, ...]
    return torch.stack([angles.sin(), angles.cos()], dim=2).reshape(t.shape[0], -1)
