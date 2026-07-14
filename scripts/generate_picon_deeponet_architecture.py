"""Generate a deterministic PI-CON DeepONet architecture figure.

What: 產生論文用兩行 DeepONet branch/trunk 架構圖。
Why: Image generation 容易新增錯誤箭頭；固定座標與箭頭拓撲可避免
     Branch Basis / Trunk Basis 被誤接到 Field。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures"
OUT_STEM = OUT_DIR / "picon_deeponet_architecture"

COLORS = {
    "ink": "#1a1a1a",
    "muted": "#5a6572",
    "line": "#33414f",
    "light": "#c9d3dd",
    "gray": "#3a4652",
    "gray_fill": "#eef2f6",
    # 統一論文配色（Okabe-Ito）：backbone = 中性灰、novelty = PI-CON 藍 (#0072B2)。
    # 灰底骨幹 + 藍色 accent + 粗框 + 編號 badge（非顏色通道），確保灰階列印仍可區分。
    "novel": "#0072B2",
    "novel_fill": "#e2eef6",
    "backbone": "#5b6670",
    "backbone_fill": "#f2f5f8",
    # branch / trunk 泳道背景（極淡，僅作結構分群，不喧賓奪主）。
    "lane_branch": "#f6f8fa",
    "lane_trunk": "#f6f8fa",
    "lane_edge": "#e3e9ef",
}


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str = "",
    *,
    edge: str,
    face: str,
    title_color: str | None = None,
    title_size: float = 9.2,
    subtitle_size: float = 7.4,
    lw: float = 1.05,
) -> None:
    """固定尺寸節點，文字置中，避免自動 layout 改變拓撲。"""
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.075",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h * (0.62 if subtitle else 0.5),
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color or edge,
        zorder=4,
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.33,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=COLORS["ink"],
            linespacing=1.25,
            zorder=4,
        )


def add_chip(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    edge: str,
    width: float,
    height: float = 0.36,
    fontsize: float = 7.0,
) -> None:
    chip = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.06",
        linewidth=1.0,
        edgecolor=edge,
        facecolor="white",
        zorder=5,
    )
    ax.add_patch(chip)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=edge,
        zorder=6,
    )


def add_badge(ax, cx: float, cy: float, num: int, *, color: str = COLORS["novel"], r: float = 0.17) -> None:
    """貢獻編號 badge：自繪實心圓 + 白色數字，避免依賴字型的 circled-digit glyph。"""
    ax.add_patch(patches.Circle((cx, cy), r, facecolor=color, edgecolor="white", linewidth=1.1, zorder=7))
    ax.text(cx, cy, str(num), ha="center", va="center", fontsize=8.0, fontweight="bold", color="white", zorder=8)


def add_legend(ax, x: float, y: float) -> None:
    """灰底骨幹 vs accent 貢獻的雙列圖例，呼應 tab:deeponet_gaps 的 1/2/3 編號。"""
    sw = 0.40
    h = 0.26
    # 第一列：inherited backbone
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y), sw, h, boxstyle="round,pad=0.01,rounding_size=0.04",
            facecolor=COLORS["backbone_fill"], edgecolor=COLORS["backbone"], linewidth=1.0, zorder=5,
        )
    )
    ax.text(x + sw + 0.16, y + h / 2, "Inherited DeepONet backbone", fontsize=6.8, va="center", color=COLORS["ink"], zorder=6)
    # 第二列：this work 的貢獻
    y2 = y - 0.46
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y2), sw, h, boxstyle="round,pad=0.01,rounding_size=0.04",
            facecolor=COLORS["novel_fill"], edgecolor=COLORS["novel"], linewidth=1.8, zorder=5,
        )
    )
    add_badge(ax, x + sw + 0.16, y2 + h / 2, 1, r=0.135)
    ax.text(
        x + sw + 0.40, y2 + h / 2,
        "Added by PI-CON (numbered 1-3)",
        fontsize=6.8, va="center", color=COLORS["novel"], zorder=6, fontweight="bold",
    )


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["line"],
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
    lw: float = 1.0,
    linestyle: str = "-",
    connectionstyle: str = "arc3,rad=0.0",
    mutation_scale: float = 10.0,
    zorder: int = 2,
) -> None:
    arrow = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=5,
        shrinkB=5,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    if label:
        if label_xy is None:
            label_xy = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            ha="center",
            va="center",
            fontsize=6.6,
            color=color,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.92),
            zorder=6,
        )


def add_bracket(ax, x0: float, x1: float, y: float, label: str) -> None:
    """畫 trainable modules bracket，避免 feedback 指到 observations。"""
    path = MplPath(
        [(x0, y), (x0, y + 0.16), (x1, y + 0.16), (x1, y)],
        [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO],
    )
    ax.add_patch(
        patches.PathPatch(
            path,
            fill=False,
            edgecolor=COLORS["gray"],
            linewidth=0.9,
            linestyle=(0, (4, 3)),
            zorder=1,
        )
    )
    ax.text(
        (x0 + x1) / 2,
        y + 0.28,
        label,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=COLORS["gray"],
        zorder=5,
    )


def add_lane(ax, x: float, y: float, w: float, h: float, *, fill: str, edge: str) -> None:
    """極淡的圓角泳道背景，將 branch / trunk 兩條 encoder path 視覺分群。"""
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=0.8, edgecolor=edge, facecolor=fill, zorder=0,
        )
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "cm",  # Computer Modern：公式呈現 LaTeX 編譯樣式，與內文數學一致
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(12.4, 5.8), dpi=240)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 6.7)
    ax.axis("off")

    # branch / trunk encoder-path 泳道背景（zorder 0，最底層）：淡色圓角分群。
    add_lane(ax, 0.12, 4.34, 6.85, 1.74, fill=COLORS["lane_branch"], edge=COLORS["lane_edge"])
    add_lane(ax, 2.55, 1.60, 4.42, 1.22, fill=COLORS["lane_trunk"], edge=COLORS["lane_edge"])

    # 分區標題（zone label）：統一中性灰，讓 accent 顏色專門表達「貢獻」
    ax.text(0.30, 6.24, "BRANCH PATH: SPARSE SENSOR MEMORY", fontsize=8.0, fontweight="bold", color=COLORS["muted"])
    ax.text(2.60, 2.94, "TRUNK PATH: QUERY FEATURE", fontsize=8.0, fontweight="bold", color=COLORS["muted"])
    ax.text(7.05, 6.24, "INTERACTION", fontsize=8.0, fontweight="bold", color=COLORS["muted"])
    ax.text(11.35, 6.24, "OUTPUT AND LOSS", fontsize=8.0, fontweight="bold", color=COLORS["muted"])

    # Branch row
    obs = (0.35, 4.55, 1.72, 1.08)
    spatial = (2.62, 4.55, 1.82, 1.08)
    cfc = (4.95, 4.55, 1.76, 1.08)
    add_box(ax, *obs, "Observations", "sensor values\npositions + time", edge=COLORS["backbone"], face=COLORS["backbone_fill"])
    add_chip(ax, obs[0] + 0.10, obs[1] + obs[3] + 0.09, "K = 100 sensors", edge=COLORS["backbone"], width=1.52)
    add_box(ax, *spatial, "Spatial Tokens", "Fourier/RFF encoding\nresidual MLP", edge=COLORS["backbone"], face=COLORS["backbone_fill"])
    add_chip(ax, spatial[0] + 0.24, spatial[1] + spatial[3] + 0.09, "tokens: [T,K,d]", edge=COLORS["backbone"], width=1.34)
    add_box(ax, *cfc, "CfC Memory", "token attention\ncausal scan dt", edge=COLORS["novel"], face=COLORS["novel_fill"], title_size=8.6, lw=1.9)
    add_chip(ax, cfc[0] + 0.33, cfc[1] + cfc[3] + 0.09, "h: [T,K,d]", edge=COLORS["novel"], width=1.10)
    add_badge(ax, cfc[0] + 0.16, cfc[1] + cfc[3] - 0.16, 1)
    add_arrow(ax, (obs[0] + obs[2], 5.09), (spatial[0], 5.09))
    add_arrow(ax, (spatial[0] + spatial[2], 5.09), (cfc[0], 5.09))

    # Trunk row
    query = (2.75, 1.92, 1.45, 0.76)
    trunk_feature = (4.95, 1.82, 1.86, 0.94)
    add_box(ax, *query, "Query", "(x, y, t_q, c)", edge=COLORS["backbone"], face=COLORS["backbone_fill"], title_size=8.8)
    add_box(
        ax,
        *trunk_feature,
        "Trunk Feature",
        "Fourier + temporal\nanchor, dt_to_query",
        edge=COLORS["backbone"],
        face=COLORS["backbone_fill"],
        title_size=8.4,
        subtitle_size=6.8,
    )
    add_arrow(ax, (query[0] + query[2], 2.29), (trunk_feature[0], 2.29))

    # Cross-attention block
    attn = (7.15, 3.76, 2.90, 1.18)
    add_box(
        ax,
        *attn,
        "Cross-Attention Readout",
        "Q from trunk feature\nK,V from CfC memory\n+ isotropic distance bias",
        edge=COLORS["novel"],
        face=COLORS["novel_fill"],
        title_color=COLORS["novel"],
        title_size=7.7,
        subtitle_size=6.6,
        lw=1.9,
    )
    add_badge(ax, attn[0] + 0.16, attn[1] + attn[3] - 0.16, 2)
    add_arrow(
        ax,
        (cfc[0] + cfc[2], 5.12),
        (attn[0], 4.55),
        label="K,V tokens\nlatest h(t_k <= t_q)",
        label_xy=(7.62, 5.42),
        connectionstyle="arc3,rad=-0.08",
    )
    add_arrow(
        ax,
        (trunk_feature[0] + trunk_feature[2], 2.28),
        (attn[0], 4.05),
        label="Q",
        label_xy=(7.05, 3.08),
        connectionstyle="angle3,angleA=0,angleB=-90",
    )

    # Basis + fusion: 嚴格 Y-shaped merge，basis 只進 Fusion。
    branch_basis = (10.30, 4.70, 1.45, 0.64)
    trunk_basis = (10.30, 2.48, 1.45, 0.64)
    fusion = (12.20, 3.48, 1.80, 0.88)
    field = (14.35, 3.52, 0.95, 0.78)
    loss = (13.95, 1.62, 2.05, 1.30)
    add_box(ax, *branch_basis, "Branch Basis", "query-conditioned\ncontext", edge=COLORS["backbone"], face="white", title_size=7.8, subtitle_size=6.2)
    add_box(ax, *trunk_basis, "Trunk Basis", "component-wise\nbasis", edge=COLORS["backbone"], face="white", title_size=7.8, subtitle_size=6.2)
    add_box(ax, *fusion, "DeepONet Fusion", "branch × trunk basis\ncomponent-wise output", edge=COLORS["backbone"], face=COLORS["backbone_fill"], title_size=7.8, subtitle_size=6.2)
    add_box(ax, *field, "Field", "u, v, p\n(p physics-only)", edge=COLORS["backbone"], face="white", title_size=7.8, subtitle_size=6.2)
    add_box(
        ax,
        *loss,
        "Loss",
        "AL constraint: div u = 0\nsensor MSE + NS residual\nGradNorm balancing",
        edge=COLORS["novel"],
        face=COLORS["novel_fill"],
        title_color=COLORS["novel"],
        title_size=8.2,
        subtitle_size=6.2,
        lw=1.9,
    )
    add_badge(ax, loss[0] + 0.16, loss[1] + loss[3] - 0.16, 3)
    add_arrow(ax, (attn[0] + attn[2], 4.58), (branch_basis[0], 5.00))
    add_arrow(ax, (trunk_feature[0] + trunk_feature[2], 2.28), (trunk_basis[0], 2.72), connectionstyle="arc3,rad=0.02")
    add_arrow(ax, (branch_basis[0] + branch_basis[2], 5.00), (fusion[0], 4.12), connectionstyle="arc3,rad=-0.06")
    add_arrow(ax, (trunk_basis[0] + trunk_basis[2], 2.72), (fusion[0], 3.72), connectionstyle="arc3,rad=0.06")
    add_arrow(ax, (fusion[0] + fusion[2], 3.92), (field[0], 3.92))
    add_arrow(ax, (field[0] + field[2], 3.92), (loss[0], 2.55), label="autograd", label_xy=(14.35, 3.24), connectionstyle="angle3,angleA=0,angleB=90")

    # Optimizer feedback
    opt = (10.05, 0.68, 2.10, 0.68)
    params = (7.25, 0.68, 2.02, 0.68)
    add_box(ax, *opt, "SOAP + Schedule-Free", "preconditioned updates", edge=COLORS["gray"], face="white", title_color=COLORS["ink"], title_size=6.9, subtitle_size=5.8, lw=1.15)
    add_box(ax, *params, "Model Parameters", "theta: encoder + CfC + decoder", edge=COLORS["gray"], face="white", title_color=COLORS["ink"], title_size=7.2, subtitle_size=5.8, lw=1.15)
    add_arrow(ax, (loss[0] + 0.55, loss[1]), (opt[0] + opt[2], opt[1] + 0.36), label="optimize", label_xy=(13.0, 1.05), linestyle=(0, (4, 3)), lw=1.2)
    add_arrow(ax, (opt[0], opt[1] + 0.36), (params[0] + params[2], params[1] + 0.36), label="update theta", label_xy=(9.82, 1.32), linestyle=(0, (4, 3)), lw=1.2)
    add_bracket(ax, spatial[0] - 0.05, fusion[0] + fusion[2] + 0.05, 1.52, "trainable modules")
    add_arrow(ax, (params[0] + params[2] / 2, params[1] + params[3]), (8.40, 1.68), linestyle=(0, (4, 3)), lw=1.2)

    # Loss formula and supervision note
    ax.text(
        4.05,
        0.23,
        r"$\mathcal{L} = \mathrm{GradNorm}\left(\mathcal{L}_{\mathrm{data}},\, "
        r"\mathcal{L}_{\mathrm{NS},u},\, \mathcal{L}_{\mathrm{NS},v},\, "
        r"\mathcal{L}_{\mathrm{cont}}\right) + \mathcal{L}_{\mathrm{AL}}$",
        ha="center",
        va="center",
        fontsize=9.5,
        color=COLORS["ink"],
        bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.05", fc="white", ec=COLORS["light"], lw=0.9),
        zorder=5,
    )
    ax.text(
        11.2,
        0.23,
        "Full-field DNS is not used as supervision.",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=COLORS["gray"],
        zorder=5,
    )

    # 圖例：灰底骨幹 vs accent 貢獻（編號 1-3 對應 tab:deeponet_gaps）
    add_legend(ax, 0.40, 1.18)

    fig.savefig(f"{OUT_STEM}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{OUT_STEM}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{OUT_STEM}.png", bbox_inches="tight", pad_inches=0.04, dpi=360)
    plt.close(fig)


if __name__ == "__main__":
    main()
