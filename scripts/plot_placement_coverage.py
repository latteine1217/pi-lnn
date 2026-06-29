"""Why DNS-init placement is worse: coverage visualisation + mechanism panels.

Top row: each placement's sensors on the domain over a 'distance-to-nearest-sensor'
heatmap (bright = uncovered gap). Bottom: (left) gappy-POD with the true DNS basis
is equal for all placements (rules out linear conditioning); (right) enstrophy
decay shows the DNS-init placement window is a violent transient (root cause of
the clustered placement).  Diagnostic only -- not for the thesis.
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pi_con.plot_style import apply_journal_rcparams

SDIR = ROOT / "data/kolmogorov_sensors/re10000"
OUT = ROOT / "diagnostics" / "placement_coverage_mechanism"
PLAC = [
    ("DNS oracle", "sensors_qrpivot_K100_N256_t0-5_si100.json", 4.69),
    ("LES $T{=}50$ random-IC", "sensors_qrpivot_K100_N256_t0-5_si100_les_n256_T50standalone.json", 5.90),
    ("LES $t{=}30$ DNS-init", "sensors_qrpivot_les_n256_T30dnsinit.json", 7.79),
    ("LES $t{=}5$ DNS-init", "sensors_qrpivot_les_n256_T5dnsinit.json", 9.12),
]


def coords(f):
    return np.asarray(json.load(open(SDIR / f))["selected_coordinates"])


def main() -> None:
    apply_journal_rcparams()
    g = np.linspace(0, 1, 192, endpoint=False)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    grid = np.stack([GX.ravel(), GY.ravel()], 1)

    fig = plt.figure(figsize=(7.4, 4.8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.0], hspace=0.42, wspace=0.12)

    for j, (name, f, ke) in enumerate(PLAC):
        c = coords(f)
        d = np.sqrt(((grid[:, None] - c[None]) ** 2).sum(-1)).min(1).reshape(192, 192)
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(d.T, origin="lower", extent=[0, 1, 0, 1], cmap="magma",
                       vmin=0, vmax=0.22, aspect="equal")
        ax.scatter(c[:, 0], c[:, 1], s=4, c="cyan", edgecolors="none")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}\nKE {ke:.2f}%  (max gap {d.max():.3f})", fontsize=6.8)
    cb = fig.colorbar(im, ax=fig.axes[:4], location="right", fraction=0.012, pad=0.01)
    cb.set_label("dist. to nearest sensor", fontsize=6.5); cb.ax.tick_params(labelsize=6)

    # bottom-left: gappy-POD (oracle basis) vs trained NN
    gp = np.load(ROOT / "diagnostics/gappy_pod_results.npz")
    names = ["DNS-oracle", "T50", "t30", "t5"]
    gappy = [float(gp[f"keerr_{n}"][1]) for n in names]  # r=20
    trained = [4.69, 5.90, 7.79, 9.12]
    axg = fig.add_subplot(gs[1, :2])
    xx = np.arange(4)
    axg.bar(xx - 0.2, gappy, 0.38, color="#999999", label="gappy-POD (true DNS basis)")
    axg.bar(xx + 0.2, trained, 0.38, color="#D55E00", label="trained PI-CON")
    axg.set_xticks(xx); axg.set_xticklabels(["DNS\norc.", "T50", "t30", "t5"], fontsize=7)
    axg.set_ylabel("KE rel-err (%)", fontsize=8)
    axg.legend(fontsize=6.5, loc="upper left")
    axg.set_title("Linear conditioning equal; NN differs", fontsize=8)

    # bottom-right: enstrophy decay (root cause)
    ew = np.load(ROOT / "diagnostics/enstrophy_windows.npz")
    axe = fig.add_subplot(gs[1, 2:])
    axe.semilogy(ew["tdns"], ew["edns"], "k-", lw=1.5, label="DNS target")
    axe.semilogy(ew["t50"][ew["t50"] <= 8], ew["e50"][ew["t50"] <= 8], color="#009E73", lw=1.2, label="T50 (steady)")
    axe.semilogy(ew["t5"], ew["e5"], color="#CC79A7", lw=1.5, ls=":", label="t5 DNS-init")
    axe.set_xlim(0, 8); axe.set_xlabel("$t$ (s)", fontsize=8)
    axe.set_ylabel(r"enstrophy (1/s$^2$)", fontsize=8)
    axe.legend(fontsize=6.5, loc="upper right")
    axe.set_title("DNS-init window = violent decay", fontsize=8)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.pdf"); fig.savefig(f"{OUT}.png", dpi=300)
    print(f"saved {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
