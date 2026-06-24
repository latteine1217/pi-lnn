"""DNS vs LES comparison: same-IC vorticity fields over time + energy spectrum.

What:
    Combined figure, entirely from the same-initial-condition LES (FIXED solver
    from the DNS field at t=0, integrated to t=5s). Top: two-row (DNS / LES) by
    six-column (time) grid of vorticity snapshots over t in [0,5]s, each time
    column normalised by the DNS RMS vorticity. Bottom: radial energy spectrum
    of the LES against the DNS at the matched final time t=5s, with [3,30]
    inertial-range slope fits.

Why:
    Replaces the prior single-snapshot LES-vs-DNS field/spectrum figure with a
    coherent time-resolved comparison from one LES run.

Data:
    diagnostics/vort_compare_arrays.npz (same-IC vorticity grid),
    diagnostics/les_sameic_t5_uv.npz (same-IC LES t=5 snapshot),
    data/dns/...npy (DNS, t=5 snapshot for the spectrum).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pi_con.plot_style import apply_journal_rcparams

T_EDDY = 1.99  # s
K_SENSOR = np.sqrt(100 / np.pi)  # ~5.64 1/m, K=100 Nyquist
BLUE, VERM = "#0072B2", "#D55E00"
VLIM = 3.0
DNS = ROOT / "data" / "dns" / "kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy"
OUT = ROOT / "thesis" / "figures" / "results" / "dns_les_vorticity_compare"


def radial_spectrum(u, v):
    N = u.shape[-1]
    uh, vh = np.fft.fft2(u), np.fft.fft2(v)
    e2d = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2) / (N ** 4)
    km = np.fft.fftfreq(N, d=1.0 / N)
    kx, ky = np.meshgrid(km, km, indexing="ij")
    krad = np.sqrt(kx ** 2 + ky ** 2)
    edges = np.arange(0.5, N // 2 + 1.5, 1.0)
    kb = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(krad.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < len(kb))
    spec = np.zeros(len(kb))
    np.add.at(spec, idx[ok], e2d.ravel()[ok])
    return kb, spec


def slope(k, E, lo=3, hi=30):
    m = (k >= lo) & (k <= hi) & (E > 1e-30)
    s, b = np.polyfit(np.log10(k[m]), np.log10(E[m]), 1)
    return float(s), float(b)


def main() -> None:
    apply_journal_rcparams()
    d = np.load(ROOT / "diagnostics" / "vort_compare_arrays.npz")
    odns, oles = d["omega_dns"], d["omega_les"]
    times, rms, L = d["times"], d["rms_dns"], float(d["L"])
    nt = len(times)

    les5 = np.load(ROOT / "diagnostics" / "les_sameic_t5_uv.npz")
    dns = np.load(DNS, allow_pickle=True).item()
    du, dv = np.asarray(dns["u"][-1], float), np.asarray(dns["v"][-1], float)
    k_les, E_les = radial_spectrum(np.asarray(les5["u"], float), np.asarray(les5["v"], float))
    k_dns, E_dns = radial_spectrum(du, dv)
    s_les, b_les = slope(k_les, E_les)
    s_dns, b_dns = slope(k_dns, E_dns)
    print(f"spectrum slope [3,30] @ t=5: LES(same-IC)={s_les:.2f}  DNS={s_dns:.2f}")

    fig = plt.figure(figsize=(7.2, 4.7))
    gs = fig.add_gridspec(3, nt, height_ratios=[1, 1, 1.35], hspace=0.16, wspace=0.06)
    ext = [0, L, 0, L]
    im = None
    vax = []
    for i, fields in enumerate((odns, oles)):
        for j in range(nt):
            a = fig.add_subplot(gs[i, j]); vax.append(a)
            im = a.imshow((fields[j] / rms[j]).T, origin="lower", extent=ext,
                          cmap="RdBu_r", vmin=-VLIM, vmax=VLIM, aspect="equal",
                          interpolation="bilinear")
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_linewidth(0.6)
            if i == 0:
                a.set_title(rf"$t={times[j]:.0f}$ s" +
                            (rf" (${times[j]/T_EDDY:.1f}\,t_{{\rm e}}$)" if times[j] > 0 else ""),
                            fontsize=6.5)
    # row labels (vax: DNS row first [0..nt-1], LES row next [nt..])
    vax[0].set_ylabel("DNS", fontsize=9)
    vax[nt].set_ylabel("LES", fontsize=9)

    cbar = fig.colorbar(im, ax=vax, location="right", fraction=0.012, pad=0.01,
                        extend="both")
    cbar.set_label(r"$\omega / \omega_{\rm rms}^{\rm DNS}$ (--)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)

    # spectrum panel (centered)
    axs = fig.add_subplot(gs[2, 1:nt - 1])
    axs.loglog(k_dns[E_dns > 0], E_dns[E_dns > 0], color=BLUE, lw=1.5,
               label=f"DNS ($t=5$ s, slope {s_dns:.2f})")
    axs.loglog(k_les[E_les > 0], E_les[E_les > 0], color=VERM, lw=1.5, ls="--",
               label=f"LES ($t=5$ s, slope {s_les:.2f})")
    kf = np.array([3.0, 30.0])
    axs.loglog(kf, 10 ** (s_dns * np.log10(kf) + b_dns), color=BLUE, lw=0.8, ls=":")
    axs.loglog(kf, 10 ** (s_les * np.log10(kf) + b_les), color=VERM, lw=0.8, ls=":")
    kref = np.array([3.0, 60.0])
    Eanc = float(E_dns[np.argmin(np.abs(k_dns - 3))])
    axs.loglog(kref, Eanc * (kref / 3) ** (-3), color="0.5", lw=0.7, ls=(0, (5, 2)))
    axs.text(40, Eanc * (40 / 3) ** (-3) * 1.4, r"$k^{-3}$", fontsize=6.5, color="0.45")
    axs.axvline(K_SENSOR, color="0.6", lw=0.7, ls=(0, (1, 1)))
    axs.text(K_SENSOR * 1.1, axs.get_ylim()[1] * 0.25, r"$k_{\rm sensor}$",
             fontsize=6, color="0.4", va="top")
    axs.set_xlabel(r"Wavenumber $k$ (1/m)", fontsize=8)
    axs.set_ylabel(r"$E(k)$ (m$^3$/s$^2$)", fontsize=8)
    axs.legend(loc="upper right", fontsize=6.3)
    axs.tick_params(labelsize=7)
    axs.text(-0.16, 1.02, "(b)", transform=axs.transAxes, fontsize=10, fontweight="bold")
    vax[0].text(-0.32, 1.10, "(a)", transform=vax[0].transAxes,
                fontsize=10, fontweight="bold")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.pdf")
    fig.savefig(f"{OUT}.png", dpi=300)
    print(f"saved {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
