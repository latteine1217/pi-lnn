"""Frame-by-frame LES-vs-DNS same-IC predictability figure (appendix).

What:
    Three-panel figure quantifying how a reference-solver LES seeded from the
    DNS initial condition tracks the DNS over t in [0,5]s: (a) velocity error,
    (b) field correlation, (c) statistical-quantity ratios. Supports the
    independent-realisation argument in Sec. LES placement: even from an
    identical IC the LES tracks the DNS only ~1 eddy turnover before chaos.

Why:
    The placement LES is an independent realisation that never coincides
    pointwise with the DNS; this figure measures the predictability horizon
    that makes statistical (not trajectory) matching the only viable basis.

Data:
    diagnostics/les_sameic_frame_err.npz produced by the FIXED solver run
    (psi=+omega/K^2) from the DNS IC; see scripts/frame_err.py provenance.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pi_con.plot_style import apply_journal_rcparams

T_EDDY = 1.99  # s, DNS value (Table dns_verification): L/U_rms

# colorblind-safe (Wong) palette
BLUE, ORANGE, GREEN, VERM, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"

DATA = Path(__file__).resolve().parent.parent / "diagnostics" / "les_sameic_frame_err.npz"
OUT = Path(__file__).resolve().parent.parent / "thesis" / "figures" / "results" / "les_sameic_predictability"


def main() -> None:
    apply_journal_rcparams()
    d = np.load(DATA)
    T = d["T"]

    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.45))

    # (a) velocity error
    ax[0].plot(T, d["relL2"] * 100, color=BLUE, lw=1.4, label=r"rel-$L^2$ (full)")
    ax[0].plot(T, d["relLow"] * 100, color=GREEN, lw=1.4, ls="--",
               label=r"rel-$L^2$ ($k\leq5$)")
    ax[0].plot(T, d["relLinf"] * 100, color=VERM, lw=1.0, ls=":",
               label=r"rel-$L^\infty$")
    ax[0].axhline(141, color="0.5", lw=0.7, ls=(0, (1, 1)))
    ax[0].text(0.15, 145, r"$\sqrt{2}$ (decorrelated)", fontsize=6, color="0.4")
    ax[0].set_ylabel("Velocity error (%)")
    ax[0].set_ylim(0, 160)
    ax[0].legend(loc="center left", fontsize=6.5)
    ax[0].text(-0.18, 1.04, "(a)", transform=ax[0].transAxes, fontsize=10, fontweight="bold")

    # (b) correlation
    ax[1].plot(T, d["cu"], color=BLUE, lw=1.4, label=r"$\mathrm{corr}(u)$")
    ax[1].plot(T, d["cw"], color=VERM, lw=1.4, ls="--", label=r"$\mathrm{corr}(\omega)$")
    ax[1].axhline(0, color="0.6", lw=0.6)
    ax[1].set_ylabel("Pearson correlation (dimensionless)")
    ax[1].set_ylim(-0.15, 1.05)
    ax[1].legend(loc="lower left", fontsize=6.5)
    ax[1].text(-0.20, 1.04, "(b)", transform=ax[1].transAxes, fontsize=10, fontweight="bold")

    # (c) statistical ratios
    ax[2].plot(T, d["ker"], color=BLUE, lw=1.4, label=r"$\mathrm{KE}_{\rm LES}/\mathrm{KE}_{\rm DNS}$")
    ax[2].plot(T, d["ensr"], color=ORANGE, lw=1.4, ls="--",
               label=r"$\mathcal{Z}_{\rm LES}/\mathcal{Z}_{\rm DNS}$")
    ax[2].axhline(1.0, color="0.6", lw=0.6)
    ax[2].set_ylabel("LES/DNS ratio (dimensionless)")
    ax[2].set_ylim(0.6, 1.05)
    ax[2].legend(loc="lower left", fontsize=6.5)
    ax[2].text(-0.22, 1.04, "(c)", transform=ax[2].transAxes, fontsize=10, fontweight="bold")

    for a in ax:
        a.set_xlabel(r"Time $t$ (s)")
        a.set_xlim(0, 5)
        for k in (1, 2):
            a.axvline(k * T_EDDY, color="0.55", lw=0.7, ls=(0, (4, 3)))

    fig.tight_layout(w_pad=1.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.pdf")
    fig.savefig(f"{OUT}.png", dpi=300)
    print(f"saved {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
