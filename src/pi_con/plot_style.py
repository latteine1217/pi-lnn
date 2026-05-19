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
}


def apply_journal_rcparams() -> None:
    """Apply journal-standard rcParams to global matplotlib state. Idempotent."""
    plt.rcParams.update(JOURNAL_RCPARAMS)
