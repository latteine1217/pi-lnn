#!/usr/bin/env python3
"""journal_style.py — one-call matplotlib styling for journal figures.

What:
    `setup_style(venue)` applies an rcParams profile tuned to a target venue
    (ICLR/NeurIPS, IEEE, Nature, JFM) so plots come out camera-ready instead of
    matplotlib-default. Also exposes figure-width helpers, a colorblind-safe
    palette, a marker/linestyle cycle, and a `save_figure` that writes vector.

Why:
    Two things sink figures in review: matplotlib defaults (saturated Tab10,
    square markers, top/right spines, 100 DPI) and venue-spec violations (wrong
    width, raster-where-vector-required, sub-minimum fonts). Encoding both here
    means every plot script starts correct rather than re-deriving the numbers.

Usage:
    from journal_style import setup_style, figwidth, PALETTE, STYLE_CYCLE, save_figure

    setup_style("jfm")
    fig, ax = plt.subplots(figsize=(figwidth("jfm", "single"), 2.6))
    for i, (label, y) in enumerate(series.items()):
        c, m, ls = STYLE_CYCLE[i]
        ax.plot(x, y, color=c, marker=m, linestyle=ls, markevery=20, label=label)
    ax.set_xlabel("k"); ax.set_ylabel("E(k)")
    ax.legend()
    save_figure(fig, "thesis/figures/spectrum")   # writes .pdf (+ .png preview)

Design notes:
    - Serif venues (ICLR/IEEE/JFM) get a serif/Computer-Modern label font to
      match the body text; Nature gets sans-serif (Helvetica/Arial).
    - The palette is the Wong colorblind-safe set (Nature Methods 2011). Every
      series also gets a distinct marker + linestyle so it survives grayscale.
    - This module has no project dependencies — copy it next to any plot script.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Figure widths in inches (see references/venues.md) ──────────────────
WIDTHS_IN: dict[str, dict[str, float]] = {
    "iclr":    {"single": 5.50, "half": 2.70},
    "neurips": {"single": 5.50, "half": 2.70},
    "ieee":    {"single": 3.50, "double": 7.16},
    "nature":  {"single": 3.46, "double": 7.20},  # 89 mm / 183 mm
    "jfm":     {"single": 5.00, "half": 2.40},
    "tmlr":    {"single": 6.50, "half": 3.20},  # \textwidth 6.5in (JMLR-style 1-col)
}

# ── Colorblind-safe palette (Wong 2011, Nature Methods) ─────────────────
# Ordered so the first few are maximally distinct. Black anchors references.
PALETTE: list[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow (use sparingly — low contrast on white)
    "#000000",  # black (reference / ground truth)
]

# Marker + linestyle cycle so series differ without relying on color.
# Deliberately avoids the square marker 's' as the primary (user preference);
# 'o'/'^'/'D' read cleaner at small sizes.
_MARKERS = ["o", "^", "D", "v", "s", "P", "X", "*"]
_LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
STYLE_CYCLE: list[tuple[str, str, str]] = list(
    zip(PALETTE, _MARKERS, _LINESTYLES)
)


def figwidth(venue: str, kind: str = "single") -> float:
    """Return the physical figure width in inches for venue/kind.

    kind is 'single'/'half' (single-column venues) or 'single'/'double'
    (two-column venues). Raises on an unknown combination so a typo fails
    loud rather than silently producing a wrong-width figure.
    """
    venue = venue.lower()
    if venue not in WIDTHS_IN:
        raise KeyError(f"unknown venue {venue!r}; known: {sorted(WIDTHS_IN)}")
    table = WIDTHS_IN[venue]
    if kind not in table:
        raise KeyError(
            f"venue {venue!r} has no width {kind!r}; available: {sorted(table)}"
        )
    return table[kind]


def setup_style(venue: str = "iclr") -> None:
    """Apply a venue-tuned rcParams profile. Call once before plotting."""
    venue = venue.lower()
    serif = venue in {"iclr", "neurips", "ieee", "jfm", "tmlr"}

    # Per-venue minimum label font at final size (see venues.md).
    base_font = {
        "iclr": 9, "neurips": 9, "ieee": 8, "nature": 7, "jfm": 9, "tmlr": 8,
    }.get(venue, 9)

    if serif:
        family = "serif"
        font_list = ["Times New Roman", "Times", "DejaVu Serif"]
        mathtext = "cm"  # Computer Modern math — matches LaTeX body
    else:  # Nature
        family = "sans-serif"
        font_list = ["Helvetica", "Arial", "DejaVu Sans"]
        mathtext = "dejavusans"

    mpl.rcParams.update({
        "font.family": family,
        f"font.{family}": font_list,
        "mathtext.fontset": mathtext,
        "font.size": base_font,
        "axes.labelsize": base_font,
        "axes.titlesize": base_font + 1,
        "xtick.labelsize": base_font - 1,
        "ytick.labelsize": base_font - 1,
        "legend.fontsize": base_font - 1,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        # Vector-first: high savefig DPI is a fallback; prefer .pdf output.
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # embed TrueType so fonts are editable/searchable
        "ps.fonttype": 42,
        # Chartjunk removal.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.25,
        # Use the colorblind-safe palette as the default color cycle.
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def save_figure(fig, stem: str, formats=("pdf", "png")) -> list[Path]:
    """Save `fig` to <stem>.<ext> for each format. Defaults to vector PDF
    plus a PNG preview. Returns the written paths.

    Writing PDF first enforces the vector-first rule; the PNG is a convenience
    preview, not the submission asset.
    """
    out = Path(stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in formats:
        p = out.with_suffix(f".{ext}")
        fig.savefig(p)
        written.append(p)
    return written


if __name__ == "__main__":
    # Smoke test: render a tiny multi-series figure in each venue style.
    import numpy as np

    x = np.linspace(0, 2 * np.pi, 200)
    for v in ("iclr", "ieee", "nature", "jfm"):
        setup_style(v)
        kind = "double" if v in {"ieee", "nature"} else "single"
        fig, ax = plt.subplots(figsize=(figwidth(v, kind), 2.4))
        for i in range(3):
            c, m, ls = STYLE_CYCLE[i]
            ax.plot(x, np.sin(x + i), color=c, marker=m, linestyle=ls,
                    markevery=25, label=f"series {i}")
        ax.set_xlabel("x"); ax.set_ylabel("amplitude")
        ax.legend()
        paths = save_figure(fig, f"/tmp/journal_style_smoke_{v}")
        print(f"{v}: {[str(p) for p in paths]}")
        plt.close(fig)
