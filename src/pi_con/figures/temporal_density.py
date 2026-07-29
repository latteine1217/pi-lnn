"""Temporal sampling-density ablation (軸 A)：重建誤差 vs snapshot 數。

What:
    固定窗長 [0, 5]、只改 sensor 時間監督密度的 sweep，畫 KE rel-err 與
    low-band rel-err 對 snapshot 數（log-x），上方第二軸標對應的 Δt。

Why:
    主線的 201 個 snapshot 是 DNS 儲存 cadence 的副產品而非調校結果。此圖回答
    重建品質對時間取樣密度是否敏感、201 是否落在過取樣平台區。
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from pi_con.plot_style import STYLE_CYCLE, figwidth

__all__ = ["draw_temporal_density_ablation"]

# 論文低頻準則（已定義的門檻，屬事實非解讀）
LOW_BAND_CRITERION_PCT = 10.0


def draw_temporal_density_ablation(
    frames: list[int],
    ke_rel_err_pct: list[float],
    low_band_rel_err_pct: list[float],
    total_time: float = 5.0,
) -> Figure:
    """Return the ablation figure. Caller applies rcParams and saves.

    Args:
        frames: snapshot 數，由多到少或少到多皆可（x 軸自行 log-scale）。
        ke_rel_err_pct, low_band_rel_err_pct: 對應的相對誤差，單位為百分比。
        total_time: 監督窗長 T，用於上方第二軸的 Δt = T/(N_t-1)。
    """
    if not (len(frames) == len(ke_rel_err_pct) == len(low_band_rel_err_pct)):
        raise ValueError(
            f"length mismatch: frames={len(frames)}, ke={len(ke_rel_err_pct)}, "
            f"low={len(low_band_rel_err_pct)}"
        )

    fig, ax = plt.subplots(figsize=(figwidth("thesis", "single"), 3.0))

    # 底色標示 baseline (201f) 所在的取樣區間 Δt <= 0.1 s；語意留給 caption
    ax.axvspan(48, 215, color="0.85", alpha=0.35, zorder=0)
    ax.axhline(LOW_BAND_CRITERION_PCT, color="0.5", ls=":", lw=0.9, zorder=1)
    ax.text(205, LOW_BAND_CRITERION_PCT + 0.5, r"$10\%$ low-band criterion",
            fontsize=7, color="0.4", ha="right", va="bottom")

    for i, (series, label) in enumerate((
        (ke_rel_err_pct, "KE rel-err"),
        (low_band_rel_err_pct, "low-band rel-err"),
    )):
        colour, marker, linestyle = STYLE_CYCLE[i]
        ax.plot(frames, series, color=colour, marker=marker,
                linestyle=linestyle, label=label)

    ax.set_xscale("log")
    ax.set_xticks(frames)
    ax.set_xticklabels([str(f) for f in frames])
    ax.set_xlim(9.5, 230)
    ax.set_ylim(0, 25)
    ax.set_xlabel(rf"Number of temporal snapshots $N_t$ over $T={total_time:g}$ s (–)")
    # 純 mathtext，非 usetex：百分號不得跳脫，否則軸標會字面印出反斜線
    ax.set_ylabel("Relative error (%)")
    ax.legend(loc="upper right", fontsize=8)

    secax = ax.secondary_xaxis("top")
    secax.set_xscale("log")
    secax.set_xticks(frames)
    secax.set_xticklabels([f"{total_time / (n - 1):.3g}" for n in frames])
    secax.set_xlabel(r"Sensor sampling interval $\Delta t$ (s)")

    return fig
