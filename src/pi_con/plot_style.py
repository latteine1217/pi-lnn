"""Journal-style matplotlib rcParams shared across all evaluator scripts.

What:
    Centralized NeurIPS/ICLR-standard rcParams (DPI≥300, sans-serif Helvetica,
    4-side spines, inner ticks, subtle grid, square legend frame, etc.). All
    evaluator entry-point scripts must `apply_journal_rcparams()` before any
    plt.subplots() call.

Why:
    Multiple evaluators (evaluate_deeponet_cfc, evaluate_cylinder, aim_diagnostic)
    previously each had their own (or no) rcParams. Style drift between figures
    in the same paper looks unprofessional. Centralizing here forces consistency
    and makes future style tweaks one-line.
"""
from __future__ import annotations

import matplotlib.pyplot as plt


_PREFERRED_FONTS = ["Helvetica", "Arial", "DejaVu Sans"]

JOURNAL_RCPARAMS: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": _PREFERRED_FONTS,
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "axes.linewidth": 0.7,
    # 保留 4 邊 spines（NeurIPS/ICLR 多數論文圖標準）；
    # 場圖另以 _style_field_axes 套用更深的邊框。
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "grid.color": "#999999",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.fontsize": 7,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#666666",
    "legend.fancybox": False,
    "legend.borderpad": 0.4,
    "legend.borderaxespad": 0.4,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.0,
    "lines.linewidth": 1.4,
    "lines.markersize": 3.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.dpi": 100,
    # 嵌入 TrueType 字型,避免 matplotlib 預設輸出 Type-3(部分期刊/印刷拒收)
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_journal_rcparams() -> None:
    """Apply journal-standard rcParams to global matplotlib state. Idempotent."""
    plt.rcParams.update(JOURNAL_RCPARAMS)


# ══════════════════════════════════════════════════════════════════════
#  Thesis-wide colour palette (Okabe--Ito, colour-blind safe)
#
#  Single source of truth for figure colours. Import these names from every
#  plotting script so DNS / PI-CON / LES / baselines and the multi-series
#  palettes stay identical across the whole thesis. Never hardcode a figure
#  colour elsewhere; reuse or add a semantic entry here.
#
#  Convention (colour + line style are redundant, so figures survive B/W print):
#      DNS / ground truth  -> black, solid
#      PI-CON / prediction -> blue,  dashed
#      LES surrogate       -> green, dotted
# ══════════════════════════════════════════════════════════════════════

OKABE_ITO = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "sky":        "#56B4E9",
    "green":      "#009E73",  # bluish-green
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",  # reddish-purple
    "grey":       "#999999",
}

# Fixed semantic roles (use these names in scripts, never raw hex)
DNS       = OKABE_ITO["black"]       # ground-truth / reference           (solid)
PICON     = OKABE_ITO["blue"]        # proposed method / prediction       (dashed)
LES       = OKABE_ITO["green"]       # LES statistical surrogate          (dotted)
BASELINE  = OKABE_ITO["vermillion"]  # forward-CFD / interpolation baselines
ORACLE    = OKABE_ITO["orange"]      # DNS-oracle placement / secondary reference
OUTPUT    = OKABE_ITO["purple"]      # reconstructed-field / output entity (schematics)
ACCENT    = OKABE_ITO["sky"]
MUTED     = OKABE_ITO["grey"]

DNS_LS, PICON_LS, LES_LS = "-", "--", ":"

# Ordered qualitative sequence for arbitrary multi-series (yellow omitted: low contrast)
SERIES = [OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["green"],
          OKABE_ITO["orange"], OKABE_ITO["purple"], OKABE_ITO["sky"], OKABE_ITO["black"]]

# Grid-independence N-sweep (fixed; N=1024 reference = black)
N_COLORS = {128: OKABE_ITO["blue"], 256: OKABE_ITO["vermillion"],
            512: OKABE_ITO["green"], 1024: OKABE_ITO["black"]}
N_MARKERS = {128: "o", 256: "s", 512: "^", 1024: "D"}
N_LINESTYLES = {128: "--", 256: "-", 512: "-.", 1024: ":"}

# Sensor-count K-sweep
K_COLORS = {100: OKABE_ITO["blue"], 200: OKABE_ITO["vermillion"], 400: OKABE_ITO["green"]}

# Placement strategies
PLACEMENT_COLORS = {"dns": OKABE_ITO["orange"], "les": OKABE_ITO["blue"],
                    "random": OKABE_ITO["vermillion"]}

# Schematic entity fills/edges (method-overview boxes); shares PI-CON blue & LES green
SCHEMATIC = {
    "place_edge": OKABE_ITO["green"],  "place_fill": "#e7f3ec",   # LES / placement
    "dns_edge":   OKABE_ITO["grey"],   "dns_fill":   "#eef1f3",   # DNS reference (neutral)
    "method_edge": OKABE_ITO["blue"],  "method_fill": "#e2eff8",  # PI-CON / inputs
    "out_edge":   OKABE_ITO["purple"], "out_fill":   "#f6ecf2",   # reconstructed field / eval
    "ink":  "#1f2933",
    "line": "#253142",
    "muted": OKABE_ITO["grey"],
}
