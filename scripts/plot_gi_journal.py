#!/usr/bin/env python3
"""plot_gi_journal.py — journal-style GI figures for thesis §3.1.

What:
    Produces 3 figures from data/dns/gi_test_re10000/:
      - Main: 2-panel side-by-side
          (a) KE(t) overlay with shaded post-spin-up band
          (b) E(k) at t=5 with k_sensor and k_eta vertical guides
      - Supp 1: log-log convergence (rel_L2 u/v/ω vs N) with slope ref lines
      - Supp 2: 2-panel Enstrophy(t) + max|div|(t) (with round-off floor annotation)

Why:
    CFD reviewer fix — current 6 figures have presentation issues. This script
    produces publication-grade figures with axis units, journal style, no redundancy.

Usage:
    uv run python scripts/plot_gi_journal.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path("data/dns/gi_test_re10000")
OUT_DIR = Path("thesis/figures/results")


# ── Journal style (NeurIPS/ICLR + CLAUDE.md memory preferences) ──────
def setup_style() -> None:
    mpl.rcParams.update({
        "pdf.fonttype": 42,  # 嵌入 TrueType,避免 Type-3
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "legend.handlelength": 2.5,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
    })


# Color & marker palette (B&W-print safe — distinct via marker + linestyle)
N_COLORS = {  # Okabe--Ito (matches pi_con.plot_style.N_COLORS)
    128:  "#0072B2",   # blue
    256:  "#D55E00",   # vermillion (main result, emphasised)
    512:  "#009E73",   # bluish-green
    1024: "#000000",   # black (reference)
}
N_MARKERS = {128: "o", 256: "s", 512: "^", 1024: "D"}
N_LINESTYLES = {128: "--", 256: "-", 512: "-.", 1024: ":"}


def load(name: str) -> dict:
    return np.load(DATA_DIR / name, allow_pickle=True).item()


def compute_spectrum(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N = u.shape[-1]
    uh = np.fft.fft2(u)
    vh = np.fft.fft2(v)
    e2d = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2) / (N ** 4)
    k_mode = np.fft.fftfreq(N, d=1.0 / N)
    kx, ky = np.meshgrid(k_mode, k_mode, indexing="ij")
    k_rad = np.sqrt(kx ** 2 + ky ** 2)
    bin_edges = np.arange(0.5, N // 2 + 1.5, 1.0)
    k_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.digitize(k_rad.ravel(), bin_edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < len(k_bins))
    spec = np.zeros(len(k_bins))
    np.add.at(spec, bin_idx[valid], e2d.ravel()[valid])
    return k_bins, spec


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load 4 N runs (seed=42 only)
    runs = {}
    for N in [128, 256, 512, 1024]:
        runs[N] = load(
            f"kolmogorov_dns_fp64_etdrk4_Re10000_N{N}_T5_dt2p5e4_si100_seed42_icspectral.npy"
        )

    # Precompute time series + spectra
    ts = {}
    spec_t5 = {}
    for N, run in runs.items():
        t = np.asarray(run["time"], dtype=np.float64)
        u = np.asarray(run["u"], dtype=np.float64)
        v = np.asarray(run["v"], dtype=np.float64)
        om = np.asarray(run["omega"], dtype=np.float64)
        ke = 0.5 * np.mean(u ** 2 + v ** 2, axis=(1, 2))
        z = 0.5 * np.mean(om ** 2, axis=(1, 2))
        # div from spectral
        L = float(run["config"].get("L", 1.0))
        k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
        kx, ky = np.meshgrid(k, k, indexing="ij")
        div_max = np.zeros(len(t))
        for i in range(len(t)):
            di = np.real(np.fft.ifft2(1j * kx * np.fft.fft2(u[i]) + 1j * ky * np.fft.fft2(v[i])))
            div_max[i] = float(np.abs(di).max())
        ts[N] = {"t": t, "KE": ke, "Z": z, "div": div_max}
        idx_t5 = int(np.argmin(np.abs(t - 5.0)))
        spec_t5[N] = compute_spectrum(u[idx_t5], v[idx_t5])

    # ── MAIN figure: 2-panel KE + E(k) ──
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # Panel (a) KE(t)
    for N in [128, 256, 512, 1024]:
        s = ts[N]
        emphasize = (N == 256)
        axL.plot(
            s["t"], s["KE"],
            color=N_COLORS[N], linestyle=N_LINESTYLES[N],
            linewidth=1.6 if emphasize else 1.0,
            label=f"$N={N}$",
            marker=N_MARKERS[N], markevery=20, markersize=4,
            alpha=1.0 if emphasize else 0.85,
        )
    axL.axvspan(2.0, 5.0, alpha=0.08, color="gray", zorder=-1)
    axL.text(3.5, axL.get_ylim()[1] * 0.95 if axL.get_ylim()[1] > 0 else 0.155,
             "post-spin-up", fontsize=7, color="gray", ha="center", va="top")
    axL.set_xlabel(r"Time $t$ [s]")
    axL.set_ylabel(r"Kinetic energy $\langle \frac{1}{2}|\mathbf{u}|^2 \rangle$ [m$^2$/s$^2$]")
    axL.set_title("(a) Kinetic energy time series")
    # Fix y-axis range and place a 2x2 legend in the lower-right empty region
    # (t in [3, 5], KE < 0.130 contains no data lines).
    axL.set_ylim(0.115, 0.165)
    axL.legend(loc="lower right", ncol=2, fontsize=8, framealpha=0.95,
               handlelength=2.0, columnspacing=1.2, borderaxespad=0.5)
    axL.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

    # Panel (b) E(k) at t=5
    k_sensor = np.sqrt(100 / np.pi)  # 5.64
    k_eta = 41.47
    for N in [128, 256, 512, 1024]:
        kb, sp = spec_t5[N]
        mask = sp > 1e-16
        emphasize = (N == 256)
        axR.loglog(
            kb[mask], sp[mask],
            color=N_COLORS[N], linestyle=N_LINESTYLES[N],
            linewidth=1.6 if emphasize else 1.0,
            label=f"$N={N}$",
            marker=N_MARKERS[N], markevery=8, markersize=3.5,
            alpha=1.0 if emphasize else 0.85,
        )
    # Vertical guides
    axR.axvline(k_sensor, color="purple", linestyle="--", linewidth=0.8, alpha=0.7)
    axR.axvline(k_eta, color="orange", linestyle="--", linewidth=0.8, alpha=0.7)
    axR.text(k_sensor * 1.05, 3e-14, r"$k_{\rm sensor}=\sqrt{K/\pi}\approx 5.64$",
             fontsize=7, color="purple", rotation=90, va="bottom", ha="left")
    axR.text(k_eta * 1.05, 3e-14, r"$k_\eta \approx 41.5$",
             fontsize=7, color="orange", rotation=90, va="bottom", ha="left")
    axR.set_xlabel(r"Wavenumber $k$ [1/m]")
    axR.set_ylabel(r"Energy spectrum $E(k)$ [m$^3$/s$^2$]")
    axR.set_title("(b) Energy spectrum at $t = 5$ s")
    axR.set_ylim(1e-15, 1e0)
    axR.legend(loc="lower left")
    axR.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    out_main = OUT_DIR / "grid_indep_main.png"
    fig.savefig(out_main)
    fig.savefig(out_main.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved {out_main}")

    # ── SUPP figure 1: log-log convergence ──
    # Reuse partial JSON metrics for rel_L2 at multiple t (need to recompute since
    # main report is vs ref=1024)
    import json
    with open(DATA_DIR / "gi_analysis_report.json") as f:
        j = json.load(f)
    m_u = j["metrics"]["rel_L2_u"]
    m_v = j["metrics"]["rel_L2_v"]
    m_om = j["metrics"]["rel_L2_omega"]
    times = ["t=0.50", "t=1.00", "t=2.00", "t=5.00"]
    time_labels = {"t=0.50": r"$t=0.5$ s", "t=1.00": r"$t=1$ s", "t=2.00": r"$t=2$ s", "t=5.00": r"$t=5$ s"}
    time_colors = {"t=0.50": "#1f77b4", "t=1.00": "#2ca02c", "t=2.00": "#ff7f0e", "t=5.00": "#d62728"}
    time_markers = {"t=0.50": "o", "t=1.00": "s", "t=2.00": "^", "t=5.00": "D"}

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0))
    field_data = [("u", m_u), ("v", m_v), (r"\omega", m_om)]
    for ax, (sym, data) in zip(axes, field_data):
        for t_key in times:
            if t_key not in data:
                continue
            ns_dict = data[t_key]
            Ns = sorted([int(k.split("=")[1]) for k in ns_dict.keys()])
            errs = [ns_dict[f"N={n}"] for n in Ns]
            ax.loglog(
                Ns, errs,
                color=time_colors[t_key], marker=time_markers[t_key],
                linewidth=1.0, markersize=4, label=time_labels[t_key],
            )
        # Reference slope guides
        N_ref = np.array([128, 512])
        for p, label in [(4, "$N^{-4}$"), (8, "$N^{-8}$")]:
            err_ref = 1e-1 * (N_ref[0] / N_ref) ** p
            ax.loglog(N_ref, err_ref, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
            ax.text(N_ref[1] * 1.05, err_ref[1], label, fontsize=7, color="gray", va="center")
        ax.set_xlabel(r"Grid resolution $N$")
        ax.set_ylabel(rf"$\|\Delta {sym}\|_2 / \|{sym}_{{\rm ref}}\|_2$ [-]")
        ax.set_title(f"(${sym}$)")
        ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)
        if ax is axes[0]:
            ax.legend(loc="lower left")
    fig.suptitle("Log-log convergence of pointwise $L^2$ error (ref $N=1024$)", y=1.02)
    fig.tight_layout()
    out_supp1 = OUT_DIR / "grid_indep_supp_loglog.png"
    fig.savefig(out_supp1)
    fig.savefig(out_supp1.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved {out_supp1}")

    # ── SUPP figure 2: Enstrophy + Div ──
    fig, (axE, axD) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for N in [128, 256, 512, 1024]:
        s = ts[N]
        emphasize = (N == 256)
        axE.plot(
            s["t"], s["Z"],
            color=N_COLORS[N], linestyle=N_LINESTYLES[N],
            linewidth=1.6 if emphasize else 1.0,
            label=f"$N={N}$",
            marker=N_MARKERS[N], markevery=20, markersize=3.5,
            alpha=1.0 if emphasize else 0.85,
        )
    axE.axvspan(2.0, 5.0, alpha=0.08, color="gray", zorder=-1)
    axE.set_xlabel(r"Time $t$ [s]")
    axE.set_ylabel(r"Enstrophy $\langle \frac{1}{2}\omega^2 \rangle$ [1/s$^2$]")
    axE.set_title("(a) Enstrophy time series")
    axE.legend(loc="upper right")
    axE.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

    for N in [128, 256, 512, 1024]:
        s = ts[N]
        axD.semilogy(
            s["t"], s["div"],
            color=N_COLORS[N], linestyle=N_LINESTYLES[N],
            linewidth=1.0,
            label=f"$N={N}$",
            alpha=0.85,
        )
    axD.set_xlabel(r"Time $t$ [s]")
    axD.set_ylabel(r"$\max_{\mathbf{x}} |\nabla \cdot \mathbf{u}|$ [1/s]")
    axD.set_title("(b) Incompressibility (round-off floor)")
    axD.legend(loc="upper right")
    axD.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)
    axD.text(0.5, 0.05,
             r"Stratification reflects fp64 round-off accumulation scaling with $N^2$,"
             "\nnot physical divergence — all $N$ are at machine $\\varepsilon$ level.",
             transform=axD.transAxes, fontsize=7, color="gray", ha="center", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))
    fig.tight_layout()
    out_supp2 = OUT_DIR / "grid_indep_supp_enstrophy_div.png"
    fig.savefig(out_supp2)
    fig.savefig(out_supp2.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved {out_supp2}")

    # ── MAIN figure 2: error vs N convergence summary (linear y) ──
    # Y reported in percent for natural reading.
    import json
    with open(DATA_DIR / "gi_analysis_report.json") as f:
        j = json.load(f)
    verdict = j["verdict"]
    m_u = j["metrics"]["rel_L2_u"]
    m_om = j["metrics"]["rel_L2_omega"]

    Ns = [128, 256, 512]
    metrics_data = {
        r"$\Delta\,$KE (post-spin-up)": [
            100 * verdict["KE_max_rel_diff_post_spinup_all_N"][f"N={N}"] for N in Ns
        ],
        r"$\Delta\,$Enstrophy (post-spin-up)": [
            100 * verdict["Enstrophy_max_rel_diff_post_spinup_all_N"][f"N={N}"] for N in Ns
        ],
        r"$\|\Delta u\|_2 / \|u\|_2$ at $t=0.5$ s": [
            100 * m_u["t=0.50"][f"N={N}"] for N in Ns
        ],
        r"$\|\Delta \omega\|_2 / \|\omega\|_2$ at $t=0.5$ s": [
            100 * m_om["t=0.50"][f"N={N}"] for N in Ns
        ],
    }
    metric_styles = {
        r"$\Delta\,$KE (post-spin-up)":             ("#d62728", "o", "-"),
        r"$\Delta\,$Enstrophy (post-spin-up)":      ("#1f77b4", "s", "-"),
        r"$\|\Delta u\|_2 / \|u\|_2$ at $t=0.5$ s":     ("#2ca02c", "^", "--"),
        r"$\|\Delta \omega\|_2 / \|\omega\|_2$ at $t=0.5$ s": ("#ff7f0e", "D", "--"),
    }

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for name, vals in metrics_data.items():
        col, mk, ls = metric_styles[name]
        ax.plot(Ns, vals, color=col, marker=mk, linestyle=ls, linewidth=1.3,
                markersize=6.5, markeredgewidth=0.6, label=name)

    # Engineering tolerance horizontal line at 2 %
    ax.axhline(2.0, color="#2a8c2a", linestyle="--", linewidth=0.7, alpha=0.65)
    ax.text(515, 2.25, r"$2\%$ engineering tolerance", fontsize=7.5, color="#2a8c2a",
            ha="right", va="bottom", alpha=0.85)

    # Production grid vertical reference
    ax.axvline(256, color="black", linestyle=":", linewidth=0.5, alpha=0.35)
    ax.text(259, 13.8, "production $N=256$", fontsize=7.5, color="black",
            rotation=90, va="top", ha="left", alpha=0.65)

    ax.set_xticks([128, 256, 512])
    ax.set_xticklabels(["128", "256", "512"])
    ax.set_xlim(110, 530)
    ax.set_ylim(0, 14.5)
    ax.set_xlabel(r"Grid resolution $N$")
    ax.set_ylabel(r"Relative error vs $N{=}1024$ reference [\%]")
    ax.set_title(r"Grid convergence ($Re=10^4$, $T=5$ s)")
    ax.legend(loc="upper right", fontsize=7.5, handlelength=2.2)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.4)

    fig.tight_layout()
    out_conv = OUT_DIR / "grid_indep_convergence.png"
    fig.savefig(out_conv)
    fig.savefig(out_conv.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved {out_conv}")

    print("Done.")


if __name__ == "__main__":
    main()
