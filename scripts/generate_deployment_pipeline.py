"""Generate the PI-CON deployment-pipeline figure (WHERE-vs-WHAT narrative).

What: 兩軌關係圖，明確區分「模型真正吃什麼」與「離線鷹架」：
      - 上排（綠）：Low-cost LES --POD+QR-pivot--> Sensor positions（只給位置，不給資料）。
      - 中排（可部署資料路徑，solid）：DNS field（real-world stand-in，橘）--read u,v at K pts-->
        Sensor time series（藍）--sensor stream--> PI-CON（藍）--query--> Reconstructed field（紫）。
      - 右下（紫，dashed）：Offline evaluation（DNS full field，只做離線 benchmark）。
      - 左下註解框：LES 決定 WHERE、DNS stand-in 供 WHAT。
Why: 舊版單一直線把 LES 畫成資料上游，讀者誤以為模型需要 LES 資料，且 DNS 角色不清。
     本版用 solid=可部署資料/訓練路徑、dashed=位置指標與離線評估，讓
     「訓練只吃 sparse sensor + PDE residual、LES 只佈點、DNS 全場僅離線評估」一眼可辨。
     固定座標避免箭頭拓撲錯誤與文字重疊（舊版頂部標題壓到 DNS box 的 bug 已移除）。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures"
OUT_STEM = OUT_DIR / "picon_deployment_pipeline"

# 語意分組 + 色盲友善（關鍵區分另以 solid/dashed 線型冗餘編碼，不單靠顏色）
COLORS = {
    "ink": "#1f2933",
    "line": "#253142",
    # 綠：placement 鷹架（LES 決定 WHERE）
    "place_edge": "#2e7d5b", "place_fill": "#e7f3ec",
    # 橘：DNS real-world stand-in（供 WHAT）
    "dns_edge": "#6f7a85", "dns_fill": "#eef1f3",
    # 藍：可部署輸入 + 方法（hero）
    "novel_edge": "#1f6fb2", "novel_fill": "#e7f1fb",
    # 紫：輸出 + 離線評估
    "out_edge": "#7b4fa0", "out_fill": "#f1e9f7",
    "muted": "#59636f",
}

DASH = (0, (4, 3))  # 離線 / 位置指標關係的統一虛線樣式


def add_box(ax, x, y, w, h, title, subtitle="", *, edge, face,
            title_size=8.1, subtitle_size=7.0, lw=1.5, dashed=False):
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=lw, edgecolor=edge, facecolor=face, zorder=3,
            linestyle=DASH if dashed else "solid",
        )
    )
    ax.text(x + w / 2, y + h * (0.64 if subtitle else 0.5), title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=edge, zorder=4)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center",
                fontsize=subtitle_size, color=COLORS["ink"], linespacing=1.22, zorder=4)


def add_arrow(ax, start, end, *, color=COLORS["line"], label=None, label_xy=None,
              lw=1.7, dashed=False, mutation_scale=15.0, connectionstyle="arc3,rad=0.0",
              label_size=6.7):
    ax.add_patch(
        patches.FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=mutation_scale, linewidth=lw,
            color=color, linestyle=DASH if dashed else "solid",
            connectionstyle=connectionstyle, shrinkA=4, shrinkB=4, zorder=2,
        )
    )
    if label:
        if label_xy is None:
            label_xy = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(label_xy[0], label_xy[1], label, ha="center", va="center",
                fontsize=label_size, color=color, linespacing=1.15,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.95), zorder=6)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})

    fig, ax = plt.subplots(figsize=(12.8, 6.1), dpi=240)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 17)
    ax.set_ylim(0.2, 8.2)
    ax.axis("off")

    # ── boxes: (x, y, w, h) 以左下角為基準 ──
    les    = (0.35, 6.65, 3.60, 1.15)
    pos    = (5.25, 6.65, 3.70, 1.15)
    dns    = (0.35, 3.55, 3.60, 1.45)
    series = (5.25, 3.55, 3.70, 1.45)
    picon  = (10.30, 3.55, 3.05, 1.45)
    field  = (14.45, 3.55, 2.35, 1.45)
    ev     = (10.30, 0.75, 3.60, 1.15)

    # 上排：placement 鷹架（綠）
    add_box(ax, *les, "Low-cost LES", "DNS-free statistical surrogate",
            edge=COLORS["place_edge"], face=COLORS["place_fill"])
    add_box(ax, *pos, "Sensor positions", "locations only — LES gives\npositions, not data",
            edge=COLORS["place_edge"], face=COLORS["place_fill"])

    # 中排：可部署資料路徑（橘 stand-in → 藍輸入 → 藍方法 → 紫輸出）
    add_box(ax, *dns, "DNS field (real-world stand-in)", "experimentally inaccessible\ntrue flow",
            edge=COLORS["dns_edge"], face=COLORS["dns_fill"], title_size=8.0)
    add_box(ax, *series, "Sensor time series", "the input — K = 100, read\nfrom DNS at the positions",
            edge=COLORS["novel_edge"], face=COLORS["novel_fill"])
    add_box(ax, *picon, "PI-CON operator", "sensor MSE + PDE residual\n(no full-field target)",
            edge=COLORS["novel_edge"], face=COLORS["novel_fill"], lw=2.4)
    add_box(ax, *field, "Reconstructed field", "$u,\\ v,\\ p$",
            edge=COLORS["out_edge"], face=COLORS["out_fill"], lw=2.4, title_size=8.0)

    # 右下：離線評估（紫、虛線框）
    add_box(ax, *ev, "Offline evaluation", "DNS full field — offline only",
            edge=COLORS["out_edge"], face=COLORS["out_fill"], title_size=8.0, subtitle_size=6.6,
            lw=1.3, dashed=True)

    def cx(b):  # box center x
        return b[0] + b[2] / 2

    # ── arrows ──
    # 上排 placement 流：LES --> positions（solid, 綠）
    add_arrow(ax, (les[0] + les[2], 7.225), (pos[0], 7.225), color=COLORS["place_edge"],
              label="POD + QR-pivot", label_xy=(4.60, 7.56))
    # positions --> series：dashed「where to read」（位置指標，非資料流）
    add_arrow(ax, (cx(pos), pos[1]), (cx(series), series[1] + series[3]),
              color=COLORS["place_edge"], dashed=True, mutation_scale=13.0,
              label="where to read", label_xy=(8.00, 5.80))

    # 中排 可部署資料路徑（solid）
    add_arrow(ax, (dns[0] + dns[2], 4.275), (series[0], 4.275), color=COLORS["line"],
              label="read $u, v$ at\nthe K positions", label_xy=(4.60, 4.80))
    add_arrow(ax, (series[0] + series[2], 4.275), (picon[0], 4.275), color=COLORS["line"],
              label="sensor stream", label_xy=(9.62, 4.62))
    add_arrow(ax, (picon[0] + picon[2], 4.275), (field[0], 4.275), color=COLORS["line"],
              label="query $(x, t)$", label_xy=(13.90, 4.62), label_size=6.4)

    # 離線評估（dashed）：DNS full field 與 reconstruction 都只在此比對
    add_arrow(ax, (cx(dns), dns[1]), (ev[0] + 0.20, ev[1] + ev[3]),
              color=COLORS["muted"], dashed=True, mutation_scale=12.0,
              label="full field,\noffline only", label_xy=(5.55, 2.55))
    add_arrow(ax, (cx(field), field[1]), (ev[0] + ev[2] - 0.20, ev[1] + ev[3]),
              color=COLORS["muted"], dashed=True, mutation_scale=12.0,
              label="compare offline", label_xy=(15.05, 2.62))

    # 左下 WHERE/WHAT 註解框（綠、虛線）
    cap = (0.35, 0.55, 4.65, 1.35)
    ax.add_patch(patches.FancyBboxPatch(
        (cap[0], cap[1]), cap[2], cap[3], boxstyle="round,pad=0.03,rounding_size=0.06",
        linewidth=1.1, edgecolor=COLORS["place_edge"], facecolor="white",
        linestyle=DASH, zorder=3))
    ax.text(cap[0] + cap[2] / 2, cap[1] + cap[3] / 2,
            "LES fixes WHERE the sensors sit; the DNS field\n"
            "(real-world stand-in) supplies WHAT they read\nat those K points.",
            ha="center", va="center", fontsize=7.2, style="italic",
            color=COLORS["place_edge"], linespacing=1.28, zorder=4)

    fig.savefig(f"{OUT_STEM}.svg", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(f"{OUT_STEM}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(f"{OUT_STEM}.png", bbox_inches="tight", pad_inches=0.05, dpi=360)
    plt.close(fig)
    print(f"saved {OUT_STEM}.pdf / .png / .svg")


if __name__ == "__main__":
    main()
