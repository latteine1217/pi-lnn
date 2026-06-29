"""Generate the PI-CON deployment-pipeline figure (capability-contrast narrative).

What: 一條左到右的工程部署流程圖：便宜 LES 佈點 → K=100 感測器量測 →
      PI-CON 重建全場；強調 capability contrast（LES 只能佈點/給統計，
      無法重建單一 realization；PI-CON 從同一佈點 + 感測器歷史重建實際流場）。
Why: §Methodology 的 REAL_WORLD_PIPELINE 需要視覺化「現場可得 = LES + sensor + PDE，
     DNS 不存在」，並把 advisor 想要的賣點（model 比 LES 更能還原真實流場）以
     誠實的 capability 框架呈現（非誤差賽跑數字）。固定座標避免箭頭拓撲錯誤。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures"
OUT_STEM = OUT_DIR / "picon_deployment_pipeline"

COLORS = {
    "ink": "#1f2933",
    "muted": "#59636f",
    "novel": "#1f6fb2",
    "novel_fill": "#e7f1fb",
    "backbone": "#566173",
    "backbone_fill": "#eef2f7",
    "line": "#253142",
    "light": "#cbd5df",
    "warn": "#b43d15",
}


def add_box(ax, x, y, w, h, title, subtitle="", *, edge, face, title_color=None,
            title_size=9.0, subtitle_size=7.0, lw=1.4):
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=lw, edgecolor=edge, facecolor=face, zorder=3,
        )
    )
    ax.text(x + w / 2, y + h * (0.62 if subtitle else 0.5), title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=title_color or edge, zorder=4)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center",
                fontsize=subtitle_size, color=COLORS["ink"], linespacing=1.25, zorder=4)


def add_badge(ax, cx, cy, num, *, color=COLORS["novel"], r=0.19):
    ax.add_patch(patches.Circle((cx, cy), r, facecolor=color, edgecolor="white", linewidth=1.1, zorder=7))
    ax.text(cx, cy, str(num), ha="center", va="center", fontsize=9.0, fontweight="bold", color="white", zorder=8)


def add_arrow(ax, start, end, *, color=COLORS["line"], label=None, label_xy=None,
              lw=1.6, linestyle="-", connectionstyle="arc3,rad=0.0", mutation_scale=15.0):
    ax.add_patch(
        patches.FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=mutation_scale, linewidth=lw,
            color=color, linestyle=linestyle, connectionstyle=connectionstyle,
            shrinkA=5, shrinkB=5, zorder=2,
        )
    )
    if label:
        if label_xy is None:
            label_xy = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(label_xy[0], label_xy[1], label, ha="center", va="center", fontsize=6.6,
                color=color, bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92), zorder=6)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})

    fig, ax = plt.subplots(figsize=(12.2, 3.5), dpi=240)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    # 四個主流程節點：灰 = 現場可得的便宜素材；藍 = 本方法與其產出（hero）
    les = (0.35, 2.05, 3.0, 1.25)
    sens = (4.20, 2.05, 3.0, 1.25)
    picon = (8.05, 2.05, 3.0, 1.25)
    field = (11.90, 2.05, 3.0, 1.25)

    add_box(ax, *les, "Low-cost LES", "no DNS\nstatistical surrogate", edge=COLORS["backbone"], face=COLORS["backbone_fill"])
    add_box(ax, *sens, "K = 100 sensors", "QR-pivot placement\nmeasure u, v in time", edge=COLORS["backbone"], face=COLORS["backbone_fill"])
    add_box(ax, *picon, "PI-CON operator", "sensor MSE + PDE residual\n(no full-field target)", edge=COLORS["novel"], face=COLORS["novel_fill"], title_color=COLORS["novel"], lw=2.4)
    add_box(ax, *field, "Reconstructed field", "u, v, p over full domain", edge=COLORS["novel"], face=COLORS["novel_fill"], title_color=COLORS["novel"], lw=2.4)

    add_arrow(ax, (les[0] + les[2], 2.68), (sens[0], 2.68), label="POD + QR-pivot", label_xy=(3.78, 2.95))
    add_arrow(ax, (sens[0] + sens[2], 2.68), (picon[0], 2.68), label="sensor time series", label_xy=(7.63, 2.95))
    add_arrow(ax, (picon[0] + picon[2], 2.68), (field[0], 2.68), label="query (x, y, t)", label_xy=(11.48, 2.95))

    # 階段下方的 capability 對比註解（誠實：LES 只佈點、無 phase；PI-CON 還原實際 realization）
    ax.text(les[0] + les[2] / 2, 1.78, "phase-decorrelated;\nplaces sensors only", ha="center", va="top",
            fontsize=6.6, color=COLORS["warn"], linespacing=1.2)
    ax.text(field[0] + field[2] / 2, 1.78, "actual realisation,\nenergy-dominant scales", ha="center", va="top",
            fontsize=6.6, color=COLORS["novel"], linespacing=1.2)

    # capability hook：從 LES 到 reconstructed field 的對比 bracket
    add_arrow(ax, (les[0] + les[2] / 2, 1.18), (field[0] + field[2] / 2, 1.18),
              color=COLORS["muted"], lw=1.2, linestyle=(0, (5, 3)), mutation_scale=12.0,
              connectionstyle="arc3,rad=-0.10")
    ax.text((les[0] + field[0] + field[2]) / 2, 0.70,
            "The LES proxy places the sensors but cannot reconstruct the realisation; PI-CON does, from the placement and sensor histories alone.",
            ha="center", va="center", fontsize=7.2, fontweight="bold", color=COLORS["ink"],
            bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.05", fc="white", ec=COLORS["light"], lw=0.9), zorder=5)

    # DNS 僅離線評估、不進訓練
    dns = (11.90, 3.66, 3.0, 0.66)
    add_box(ax, *dns, "DNS full field", "offline evaluation only", edge=COLORS["muted"], face="white",
            title_color=COLORS["muted"], title_size=7.6, subtitle_size=6.2, lw=1.1)
    add_arrow(ax, (dns[0] + dns[2] / 2, dns[1]), (field[0] + field[2] / 2, field[1] + field[3]),
              color=COLORS["muted"], lw=1.0, linestyle=(0, (3, 3)), mutation_scale=10.0, label="compare", label_xy=(13.95, 3.50))
    ax.text(8.1, 4.30, "Field-deployable inputs: low-cost LES, K sparse sensors, and the PDE -- no DNS full field in training.",
            ha="center", va="center", fontsize=7.4, fontweight="bold", color=COLORS["muted"], zorder=5)

    fig.savefig(f"{OUT_STEM}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{OUT_STEM}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{OUT_STEM}.png", bbox_inches="tight", pad_inches=0.04, dpi=360)
    plt.close(fig)


if __name__ == "__main__":
    main()
