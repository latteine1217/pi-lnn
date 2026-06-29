"""Placement strategy vs KE rel-err comparison (EXP-271/245/296/297 + EXP-266 random).

What:
    Horizontal bar chart of KE rel-err for K=100 QR-pivot placements that differ
    only in the source field used for placement (all seed=42, 1024 collo, 20k,
    identical training config). Random-uniform (EXP-266, n=5 placement seeds)
    drawn as a reference band.

Why:
    Diagnostic for the DNS-init LES placement study: shows that placement quality
    is governed by the statistical convergence of the LES POD window, not by
    seeding the LES from the DNS. NOT for the thesis yet (research-only:
    DNS-init placements are engineering non-transferable).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pi_con.plot_style import apply_journal_rcparams

OUT = ROOT / "diagnostics" / "placement_vs_ke_comparison"

# (label, KE %, category)  seed=42 single-seed unless noted
GREY, GREEN, VERM, BLUE = "#808080", "#009E73", "#D55E00", "#0072B2"
BARS = [
    ("FPS space-fill\n(EXP-298, DNS-free, no LES)", 4.54, BLUE),
    ("DNS oracle\n(EXP-271, needs DNS)", 4.69, GREY),
    ("LES $T{=}50$ random-IC\n(EXP-245, DNS-free)", 5.90, GREEN),
    ("LES $t{=}30$ DNS-init\n(EXP-296, research-only)", 7.79, VERM),
    ("LES $t{=}5$ DNS-init\n(EXP-297, research-only)", 9.12, VERM),
]
RANDOM_MEAN, RANDOM_STD = 7.95, 0.68  # EXP-266 n=5 placement seeds


def main() -> None:
    apply_journal_rcparams()
    labels = [b[0] for b in BARS]
    vals = [b[1] for b in BARS]
    colors = [b[2] for b in BARS]
    y = np.arange(len(BARS))[::-1]  # best (DNS oracle) on top

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    # random-uniform reference band
    ax.axvspan(RANDOM_MEAN - RANDOM_STD, RANDOM_MEAN + RANDOM_STD,
               color="#0072B2", alpha=0.12, zorder=0)
    ax.axvline(RANDOM_MEAN, color="#0072B2", lw=1.0, ls=(0, (4, 3)), zorder=1,
               label=f"random uniform (EXP-266): {RANDOM_MEAN:.2f} $\\pm$ {RANDOM_STD:.2f}%")

    ax.barh(y, vals, height=0.62, color=colors, zorder=2, edgecolor="white", lw=0.5)
    for yi, v in zip(y, vals):
        ax.text(v + 0.1, yi, f"{v:.2f}%", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel(r"Kinetic-energy relative error (%)")
    ax.set_xlim(0, 10.5)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Placement source vs reconstruction error "
                 "($K{=}100$, seed 42, 20k)", fontsize=9)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.pdf")
    fig.savefig(f"{OUT}.png", dpi=300)
    print(f"saved {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
