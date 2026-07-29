"""Radial energy spectrum figures: per-K panels and the K-scaling triptych.

What:
    畫 DNS vs PI-CON 的 E(k)，附 k^-3 enstrophy-cascade 參考線、forcing k_f，
    以及該 sensor budget 的 Nyquist scale √(K/π)。提供論文用的單張版與投影片用
    的三連版，兩者共用同一個 panel 繪製函式。

Why:
    論文版與投影片版原本是兩支腳本，各自帶一份 panel 邏輯與 spectrum_at_t5，
    差別只在線寬、字級與是否在 Nyquist 線旁標值。把這些差異收成 `PanelStyle`
    資料而非 if 分支後，兩版共用一條繪製路徑，樣式漂移無從發生。

Note:
    `k_nyq = √(K/π)` 是 sensor-count **scale**，不是硬上限；圖上不得標成
    ceiling / bound（見 thesis/CLAUDE.md 主訊息）。
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pi_con.plot_style import DNS, OKABE_ITO, PICON

__all__ = [
    "PanelStyle",
    "PAPER_PANEL",
    "SLIDE_PANEL",
    "draw_spectrum_panel",
    "draw_k_scaling_single",
    "draw_k_scaling_triptych",
]

K_F = 2.0                      # forcing wavenumber (1/m), cyclic convention
NYQUIST_COLOUR = OKABE_ITO["green"]
REFERENCE_COLOUR = "0.5"       # k^-3 reference line
_K3_ANCHOR_K = 3.0             # k^-3 參考線錨定的波數


@dataclass(frozen=True)
class PanelStyle:
    """Per-venue line weights and annotation switches for one spectrum panel."""

    dns_lw: float
    picon_lw: float
    nyquist_lw: float
    annotate_nyquist: bool
    nyquist_in_legend: bool
    legend_fontsize: float


# 論文：單張小圖，Nyquist 值寫在 panel 標題，故不佔圖例位、線不必加粗
PAPER_PANEL = PanelStyle(
    dns_lw=1.3, picon_lw=1.3, nyquist_lw=1.1,
    annotate_nyquist=False, nyquist_in_legend=False, legend_fontsize=6.5,
)
# 投影片：三張並排時每張僅約 180 px 寬，細綠線在投影下不可辨，故加粗、就地標值，
# 並在單一共用圖例中列出
SLIDE_PANEL = PanelStyle(
    dns_lw=1.6, picon_lw=1.8, nyquist_lw=2.0,
    annotate_nyquist=True, nyquist_in_legend=True, legend_fontsize=8.5,
)


def draw_spectrum_panel(
    ax: Axes,
    k_dns: np.ndarray,
    e_dns: np.ndarray,
    k_pred: np.ndarray,
    e_pred: np.ndarray,
    k_nyq: float,
    style: PanelStyle = PAPER_PANEL,
) -> None:
    """Draw one DNS-vs-PI-CON spectrum panel onto `ax`."""
    md, mp = e_dns > 0, e_pred > 0
    ax.loglog(k_dns[md], e_dns[md], color=DNS, linestyle="-",
              linewidth=style.dns_lw, label="DNS")
    ax.loglog(k_pred[mp], e_pred[mp], color=PICON, linestyle="--",
              linewidth=style.picon_lw, label="PI-CON")

    anchor = np.interp(_K3_ANCHOR_K, k_dns[md], e_dns[md])
    kk = k_dns[(k_dns >= K_F) & (k_dns <= k_dns.max())]
    ax.loglog(kk, anchor * (kk / _K3_ANCHOR_K) ** (-3.0), color=REFERENCE_COLOUR,
              linestyle=":", linewidth=1.0, label=r"$k^{-3}$")

    ax.axvline(K_F, color=DNS, linestyle="-.", linewidth=0.7)
    ax.axvline(k_nyq, color=NYQUIST_COLOUR, linestyle="--",
               linewidth=style.nyquist_lw,
               label=r"$k_{\max}^{\rm sensor}$" if style.nyquist_in_legend else None)
    if style.annotate_nyquist:
        ax.annotate(rf"$k_{{\max}}\!\approx\!{k_nyq:.2f}$", xy=(k_nyq, 4e-2),
                    xytext=(3, 0), textcoords="offset points",
                    color=NYQUIST_COLOUR, fontsize=10, fontweight="bold", ha="left")

    ax.set_xlabel(r"wavenumber $k$ (1/m)")


def draw_k_scaling_single(
    k_dns: np.ndarray,
    e_dns: np.ndarray,
    k_pred: np.ndarray,
    e_pred: np.ndarray,
    sensor_count: int,
) -> Figure:
    """Return the single-panel figure for one sensor budget (thesis §4.4)."""
    k_nyq = float(np.sqrt(sensor_count / np.pi))
    fig, ax = plt.subplots(figsize=(3.4, 3.0), constrained_layout=True)
    draw_spectrum_panel(ax, k_dns, e_dns, k_pred, e_pred, k_nyq, PAPER_PANEL)
    ax.set_ylabel(r"$E(k)$ (m$^3$/s$^2$)")
    ax.set_title(
        rf"$K={sensor_count}$, $k_{{\max}}^{{\rm sensor}}\approx {k_nyq:.2f}$",
        fontsize=9,
    )
    ax.legend(loc="lower left", fontsize=PAPER_PANEL.legend_fontsize)
    return fig


def draw_k_scaling_triptych(
    panels: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ke_mape_pct: dict[int, float],
) -> Figure:
    """Return the shared-y triptych for slides.

    Args:
        panels: sensor count -> (k_dns, e_dns, k_pred, e_pred).
        ke_mape_pct: sensor count -> KE MAPE in percent, shown in each title so
            the deck needs no separate numbers slide.
    """
    missing = set(panels) - set(ke_mape_pct)
    if missing:
        raise KeyError(f"ke_mape_pct is missing entries for K={sorted(missing)}")

    fig, axes = plt.subplots(1, len(panels), figsize=(10.2, 3.1), sharey=True,
                             constrained_layout=True)
    for ax, (sensor_count, (k_d, e_d, k_p, e_p)) in zip(axes, panels.items()):
        k_nyq = float(np.sqrt(sensor_count / np.pi))
        draw_spectrum_panel(ax, k_d, e_d, k_p, e_p, k_nyq, SLIDE_PANEL)
        # 非 usetex（見 pi_con.plot_style），故 % 直接寫，不可用 LaTeX 的 \%
        ax.set_title(rf"$K={sensor_count}$   ·   KE {ke_mape_pct[sensor_count]:.2f}%",
                     fontsize=12.5, fontweight="bold", pad=4)
        ax.tick_params(labelsize=9)
        ax.set_ylim(1e-11, 5e-1)

    axes[0].set_ylabel(r"$E(k)$ (m$^3$/s$^2$)", fontsize=10)
    axes[0].legend(loc="lower left", fontsize=SLIDE_PANEL.legend_fontsize,
                   framealpha=0.9)
    return fig
