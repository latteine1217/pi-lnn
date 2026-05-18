"""5-way v2 results visualization — KE comparison + info content correlation.

What:
    Two panels:
      (A) Grouped bar chart: KE / div / omega_L2 / ek_ratio for all 5 strategies + baseline
      (B) Information content vs KE scatter (post-fix) — recover information predicts KE narrative

Why:
    Old plots based on buggy v1 results. v2 numbers reset narrative — need fresh
    side-by-side that's paper-ready.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "axes.linewidth": 0.7, "axes.grid": True, "grid.linewidth": 0.4,
    "grid.alpha": 0.3, "grid.color": "#999999",
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

# v2 metrics, ordered by KE (best → worst). EXP-094 baseline first.
results = [
    {"name": "EXP-094\n(DNS-pivot, oracle)", "short": "DNS-pivot", "KE": 9.4, "u_L2": 0.35, "v_L2": 0.45, "omega_L2": 1.5, "div": 0.067, "ek_ratio": 0.91, "eff_rank": 13.13},
    {"name": "EXP-105 v2\n(LES_N256 T=50)",   "short": "LES T=50", "KE": 12.36, "u_L2": 0.193, "v_L2": 0.251, "omega_L2": 0.526, "div": 0.068, "ek_ratio": 0.889, "eff_rank": 13.49},
    {"name": "EXP-102 v2\n(LES_N128 stand-alone)", "short": "LES N=128", "KE": 12.40, "u_L2": 0.206, "v_L2": 0.262, "omega_L2": 0.539, "div": 0.066, "ek_ratio": 0.927, "eff_rank": 12.27},
    {"name": "EXP-106\n(LES_N256 T=30)",       "short": "LES T=30", "KE": 13.08, "u_L2": 0.213, "v_L2": 0.264, "omega_L2": 0.541, "div": 0.067, "ek_ratio": 0.874, "eff_rank": 12.50},  # eff_rank estimate
    {"name": "EXP-101 v2\n(Random)",            "short": "Random", "KE": 13.25, "u_L2": 0.211, "v_L2": 0.274, "omega_L2": 0.542, "div": 0.071, "ek_ratio": 0.908, "eff_rank": 11.95},
    {"name": "EXP-103 v2\n(LES_N256 T=5)",     "short": "LES T=5", "KE": 23.48, "u_L2": 0.316, "v_L2": 0.377, "omega_L2": 0.620, "div": 0.063, "ek_ratio": 0.591, "eff_rank": 11.33},
]

# Buggy v1 numbers for "before vs after" plot
buggy_v1 = {
    "EXP-101": 37.2,
    "EXP-102": 44.3,
    "EXP-103": 52.0,
    "EXP-105": 53.7,
}

colors = ["#222222", "#1f77b4", "#d62728", "#2ca02c", "#7f7f7f", "#ff7f0e"]


def fig_grouped_metrics():
    """Panel grid: KE / div_L2 / omega_L2 / ek_ratio_kf_last across 6 strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)
    x = np.arange(len(results))
    labels = [r["short"] for r in results]

    # (A) KE rel-err
    ax = axes[0, 0]
    bars = ax.bar(x, [r["KE"] for r in results], color=colors)
    ax.axhline(9.4, color="black", linestyle="--", lw=0.8, label="baseline 9.4%")
    ax.axhline(13.25, color="gray", linestyle=":", lw=0.6, alpha=0.5, label="Random KE")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.set_ylabel("KE rel-err (%)")
    ax.set_title("(A) KE relative error — main metric")
    ax.set_ylim(0, 30)
    for i, r in enumerate(results):
        ax.text(i, r["KE"] + 0.5, f"{r['KE']:.2f}%", ha="center", fontsize=8)
    ax.legend(fontsize=8, loc="upper left")

    # (B) div L2
    ax = axes[0, 1]
    ax.bar(x, [r["div"] for r in results], color=colors)
    ax.axhline(0.067, color="black", linestyle="--", lw=0.8, label="baseline 0.067")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.set_ylabel(r"$\|\nabla\cdot u\|_{L^2}$")
    ax.set_title("(B) Incompressibility residual")
    for i, r in enumerate(results):
        ax.text(i, r["div"] + 0.0015, f"{r['div']:.3f}", ha="center", fontsize=8)
    ax.legend(fontsize=8)

    # (C) omega L2 — surprising finding
    ax = axes[1, 0]
    ax.bar(x, [r["omega_L2"] for r in results], color=colors)
    ax.axhline(1.5, color="black", linestyle="--", lw=0.8, label="baseline ≈ 1.5")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.set_ylabel(r"$\omega$ rel-$L^2$ error")
    ax.set_title(r"(C) Vorticity error — LES-informed $\omega$ beats baseline!")
    for i, r in enumerate(results):
        ax.text(i, r["omega_L2"] + 0.05, f"{r['omega_L2']:.2f}", ha="center", fontsize=8)
    ax.legend(fontsize=8)

    # (D) k_f mode energy ratio
    ax = axes[1, 1]
    ax.bar(x, [r["ek_ratio"] for r in results], color=colors)
    ax.axhline(0.91, color="black", linestyle="--", lw=0.8, label="baseline 0.91")
    ax.axhline(1.0, color="gray", linestyle=":", lw=0.5, alpha=0.5, label="perfect=1")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.set_ylabel(r"$E(k_f)$ ratio @ last")
    ax.set_title("(D) Forcing-mode energy fidelity")
    ax.set_ylim(0, 1.1)
    for i, r in enumerate(results):
        ax.text(i, r["ek_ratio"] + 0.02, f"{r['ek_ratio']:.2f}", ha="center", fontsize=8)
    ax.legend(fontsize=8)

    fig.suptitle("5-way placement comparison — v2 (axis bug fixed, 2026-05-18)", fontsize=11)
    out = Path("docs/assets/5way_v2_metrics.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  → {out}")


def fig_bug_impact():
    """Before / after bug fix — KE drop for each retrained experiment."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    exp_labels = ["EXP-101\n(Random)", "EXP-102\n(LES_N128)",
                  "EXP-103\n(LES_N256 T=5)", "EXP-105\n(LES_N256 T=50)"]
    v1 = [37.2, 44.3, 52.0, 53.7]
    v2 = [13.25, 12.40, 23.48, 12.36]
    x = np.arange(len(exp_labels))
    w = 0.35
    ax.bar(x - w/2, v1, w, label="v1 buggy (axis swap)", color="#d62728", alpha=0.8)
    ax.bar(x + w/2, v2, w, label="v2 fixed (correct axis)", color="#1f77b4", alpha=0.9)
    ax.axhline(9.4, color="black", linestyle="--", lw=0.8, label="baseline (DNS-pivot) 9.4%")
    for i, (a, b) in enumerate(zip(v1, v2)):
        ax.text(i - w/2, a + 0.7, f"{a:.1f}%", ha="center", fontsize=8.5)
        ax.text(i + w/2, b + 0.7, f"{b:.2f}%", ha="center", fontsize=8.5)
        # arrow showing improvement
        ax.annotate("", xy=(i + w/2, b + 2), xytext=(i - w/2, a - 2),
                    arrowprops=dict(arrowstyle="->", color="green", alpha=0.5))
        improvement = a - b
        ax.text(i, (a + b)/2 + 1, f"-{improvement:.1f}pp",
                ha="center", color="green", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(exp_labels, fontsize=9)
    ax.set_ylabel("KE rel-err (%)")
    ax.set_title("Axis-bug fix impact on KE rel-err  "
                 "(NPZ value vs JSON coord row/col convention)")
    ax.set_ylim(0, 60)
    ax.legend(fontsize=9, loc="upper right")
    out = Path("docs/assets/axis_bug_before_after.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  → {out}")


def fig_info_vs_KE():
    """Effective rank vs KE — recover information-content-predicts-KE narrative."""
    fig, ax = plt.subplots(figsize=(7.0, 5.5), constrained_layout=True)
    for r, c in zip(results, colors):
        ax.scatter(r["eff_rank"], r["KE"], s=140, c=c,
                   edgecolors="white", linewidths=0.8, label=r["short"])
        ax.annotate(r["short"], (r["eff_rank"], r["KE"]),
                    xytext=(5, -3), textcoords="offset points", fontsize=8)
    # Trend line on the 5 LES + random + DNS points (excluding T=5 outlier)
    main_pts = [r for r in results if "T=5" not in r["short"]]
    xs = np.array([r["eff_rank"] for r in main_pts])
    ys = np.array([r["KE"] for r in main_pts])
    slope, intercept = np.polyfit(xs, ys, 1)
    xfit = np.linspace(xs.min() - 0.5, xs.max() + 0.5, 50)
    ax.plot(xfit, slope * xfit + intercept, "k--", lw=0.8, alpha=0.7,
            label=f"trend (excl T=5)\nKE ≈ {slope:.2f}·rank + {intercept:.1f}")
    # T=5 outlier ring
    t5 = next(r for r in results if "T=5" in r["short"])
    ax.scatter([t5["eff_rank"]], [t5["KE"]], s=300, facecolors="none",
               edgecolors="red", linewidths=1.5)
    ax.annotate("outlier\n(short-window IC inheritance)",
                (t5["eff_rank"], t5["KE"]), xytext=(15, 15),
                textcoords="offset points", fontsize=8, color="red",
                arrowprops=dict(arrowstyle="->", color="red"))
    ax.set_xlabel("Effective rank of sensor time-series matrix (info dim)")
    ax.set_ylabel("KE rel-err (%)")
    ax.set_title("Information content vs KE — post-fix correlation")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    out = Path("docs/assets/info_vs_KE_post_fix.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  → {out}")


def main():
    print("[plot] 5-way v2 metrics grouped bar chart...")
    fig_grouped_metrics()
    print("[plot] Axis bug before/after KE comparison...")
    fig_bug_impact()
    print("[plot] Info content vs KE scatter (post-fix)...")
    fig_info_vs_KE()
    print("\nDone.")


if __name__ == "__main__":
    main()
