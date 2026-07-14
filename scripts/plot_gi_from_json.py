#!/usr/bin/env python
"""從 gi_analysis_report.json 重繪 grid-independence 的 3 張 vector PDF。

What/Why:
- 原 plot_gi_journal.py 需載入 4 個原始 DNS 場 npy (N=128/256/512/1024) 才能畫
  main figure 的 energy-spectrum panel；該場數據已不在本地 (N1024 全缺，N128/256/512
  僅在他人 home)，重跑 1024^2 DNS 成本過高。
- 但 convergence / supp_loglog / supp_enstrophy_div 三張所需數據已全在
  gi_analysis_report.json：
    * rel_L2_{u,v,omega}: vs N=1024 ref，4 個評估時間點
    * Enstrophy_t_series / max_div_t_series: per N，密集 201 點 (t=0..5)
- 故本 script 純由 json 重繪這 3 張 vector PDF；main figure (spectrum) 維持既有 png。
- 繪圖樣式常數 import 自 plot_gi_journal，確保與其餘圖一致。

只負責輸出 PDF（論文用 vector）；既有 png 由原 script 維護，不在此覆寫。
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_gi_journal import (  # noqa: E402
    setup_style, N_COLORS, N_LINESTYLES, N_MARKERS, DATA_DIR, OUT_DIR,
)

N_LIST = [128, 256, 512, 1024]


def plot_supp_loglog(j) -> None:
    """SUPP 1: 逐場 rel-L2 收斂 (loglog vs N)，數據純 json。"""
    m_u = j["metrics"]["rel_L2_u"]
    m_v = j["metrics"]["rel_L2_v"]
    m_om = j["metrics"]["rel_L2_omega"]
    times = ["t=0.50", "t=1.00", "t=2.00", "t=5.00"]
    time_labels = {"t=0.50": r"$t=0.5$ s", "t=1.00": r"$t=1$ s",
                   "t=2.00": r"$t=2$ s", "t=5.00": r"$t=5$ s"}
    time_colors = {"t=0.50": "#0072B2", "t=1.00": "#009E73",
                   "t=2.00": "#E69F00", "t=5.00": "#D55E00"}  # Okabe--Ito
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
    out = OUT_DIR / "grid_indep_supp_loglog.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def plot_supp_enstrophy_div(j) -> None:
    """SUPP 2: Enstrophy(t) + max|div|(t)，數據改自 json 密集時序 (原用場算 ts)。"""
    ens = j["metrics"]["Enstrophy_t_series"]
    dvg = j["metrics"]["max_div_t_series"]

    fig, (axE, axD) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for N in N_LIST:
        es = ens[f"N={N}"]
        emphasize = (N == 256)
        axE.plot(
            np.asarray(es["t"]), np.asarray(es["Enstrophy"]),
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

    for N in N_LIST:
        ds = dvg[f"N={N}"]
        axD.semilogy(
            np.asarray(ds["t"]), np.asarray(ds["max_div"]),
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
    out = OUT_DIR / "grid_indep_supp_enstrophy_div.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def plot_convergence(j) -> None:
    """MAIN-2: error vs N 收斂彙整 (linear y, percent)，數據純 json。"""
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
    metric_styles = {  # Okabe--Ito
        r"$\Delta\,$KE (post-spin-up)":             ("#D55E00", "o", "-"),
        r"$\Delta\,$Enstrophy (post-spin-up)":      ("#0072B2", "s", "-"),
        r"$\|\Delta u\|_2 / \|u\|_2$ at $t=0.5$ s":     ("#009E73", "^", "--"),
        r"$\|\Delta \omega\|_2 / \|\omega\|_2$ at $t=0.5$ s": ("#E69F00", "D", "--"),
    }

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for name, vals in metrics_data.items():
        col, mk, ls = metric_styles[name]
        ax.plot(Ns, vals, color=col, marker=mk, linestyle=ls, linewidth=1.3,
                markersize=6.5, markeredgewidth=0.6, label=name)
    ax.axhline(2.0, color="#2a8c2a", linestyle="--", linewidth=0.7, alpha=0.65)
    ax.text(515, 2.25, r"$2\%$ engineering tolerance", fontsize=7.5, color="#2a8c2a",
            ha="right", va="bottom", alpha=0.85)
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
    out = OUT_DIR / "grid_indep_convergence.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "gi_analysis_report.json") as f:
        j = json.load(f)
    plot_supp_loglog(j)
    plot_supp_enstrophy_div(j)
    plot_convergence(j)
    print("Done (3 vector PDFs from json; main spectrum figure remains png).")


if __name__ == "__main__":
    main()
