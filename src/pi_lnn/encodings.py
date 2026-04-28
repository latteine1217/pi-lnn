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
    """What: PeriodEmbs(k=1) + 可學習 Fourier 投影，對齊 jaxpi FourierEmbs 設計。

    Why: periodic_fourier_encode 使用確定性整數諧波（k=1..n），硬性上限 k>n 完全無法表達。
         此模組先以固定 k=1 週期編碼保證週期 BC，再用可學習投影矩陣（init N(0,σ)）
         讓網路在訓練中自適應頻率分佈，隱性覆蓋高頻。

    頻率分層（turbulence 多尺度）：
        若 init_sigma_bands 與 band_dim_ratios 均提供，則將 embed_dim/2 個投影通道
        切成 N 個頻段，分別用對應 σ 初始化。對齊 Kolmogorov 能量分佈（低頻能量多但
        頻率窄、高頻能量少但頻率廣），讓單一 embedding 同時涵蓋多尺度。
        例：bands=(1.0, 4.0, 12.0), ratios=(0.5, 0.375, 0.125)
            → 50% 通道做低頻精度，37.5% 做中頻，12.5% 做高頻探索。

    Args:
        embed_dim: 輸出維度（需為偶數）。
        init_sigma: 單頻段時的投影矩陣初始化標準差（對應 jaxpi embed_scale=2.0）。
        init_sigma_bands: 多頻段時各段的 σ；None 走單頻段路徑（向後相容）。
        band_dim_ratios: 多頻段時各段佔 embed_dim/2 的比例；長度需與 sigma_bands 相同，
                          總和需 = 1.0（容忍 ±1e-6）。各段至少分配 1 個通道。
    """

    def __init__(
        self,
        embed_dim: int,
        init_sigma: float = 2.0,
        init_sigma_bands: tuple[float, ...] | list[float] | None = None,
        band_dim_ratios: tuple[float, ...] | list[float] | None = None,
    ) -> None:
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim 必須為偶數，收到 {embed_dim}")
        half = embed_dim // 2
        self.proj = nn.Linear(4, half, bias=False)

        if init_sigma_bands is None and band_dim_ratios is None:
            nn.init.normal_(self.proj.weight, std=init_sigma)
            self._band_layout: list[tuple[int, int, float]] | None = None
            return

        if init_sigma_bands is None or band_dim_ratios is None:
            raise ValueError(
                "init_sigma_bands 與 band_dim_ratios 必須同時提供或同時為 None"
            )
        sigmas = list(init_sigma_bands)
        ratios = list(band_dim_ratios)
        if len(sigmas) != len(ratios):
            raise ValueError(
                f"init_sigma_bands ({len(sigmas)}) 與 band_dim_ratios ({len(ratios)}) "
                f"長度需相同"
            )
        if len(sigmas) < 1:
            raise ValueError("至少需要 1 個 band")
        if any(s <= 0 for s in sigmas):
            raise ValueError(f"所有 sigma 必須 > 0，收到 {sigmas}")
        if any(r <= 0 for r in ratios):
            raise ValueError(f"所有 ratio 必須 > 0，收到 {ratios}")
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"band_dim_ratios 總和需 = 1.0，收到 {sum(ratios)}")

        # 分配通道數：先按 ratio 取整，再把剩餘通道分給最大 ratio 的段
        counts = [max(1, int(round(r * half))) for r in ratios]
        diff = half - sum(counts)
        if diff != 0:
            # 按 ratio 大小排序的索引，依序加 / 減 1 直到加總正確
            order = sorted(range(len(ratios)), key=lambda i: -ratios[i])
            i = 0
            while diff > 0:
                counts[order[i % len(order)]] += 1
                diff -= 1
                i += 1
            while diff < 0:
                if counts[order[i % len(order)]] > 1:
                    counts[order[i % len(order)]] -= 1
                    diff += 1
                i += 1
        if sum(counts) != half:
            raise RuntimeError(f"band 分配出錯：counts={counts}, half={half}")

        layout: list[tuple[int, int, float]] = []
        cursor = 0
        with torch.no_grad():
            for n_dim, sigma in zip(counts, sigmas):
                end = cursor + n_dim
                self.proj.weight[cursor:end].normal_(0.0, sigma)
                layout.append((cursor, end, sigma))
                cursor = end
        self._band_layout = layout

    def forward(self, xy: torch.Tensor, domain_length: float) -> torch.Tensor:
        c = 2.0 * torch.pi / domain_length
        x, y = xy[:, 0:1], xy[:, 1:2]
        period_enc = torch.cat(
            [torch.sin(c * x), torch.cos(c * x), torch.sin(c * y), torch.cos(c * y)],
            dim=-1,
        )  # [N, 4]
        proj = self.proj(period_enc)  # [N, embed_dim // 2]
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # [N, embed_dim]


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
