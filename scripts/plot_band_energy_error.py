"""Replot band-energy relative error vs time (journal style), dropping the t=0 IC spike.

What: 從 eval 的 series.npz(band_low/mid/high + time)畫期刊風格 band-error-vs-time。
Why: 原圖 t=0 high-k 相對誤差爆到 ~1e5(IC high-k 能量近零的假象),把 y 軸撐爆、
     壓垮其餘可看性。此腳本「移除 t=0 那一幀」並套 journal_style,輸出向量 PDF。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pi_con.plot_style import apply_journal_rcparams  # noqa: E402

SERIES = ROOT / "artifacts/eval_245_seed42_export/series.npz"   # scp 回來放這
OUT_STEM = ROOT / "thesis/figures/results/band_energy_rel_error_vs_time"

# 三個 band：label, npz-key, (color, marker, linestyle) —— colorblind-safe + 灰階可分
BANDS = [
    (r"Low-$k$  ($k \leq 5$)",       "band_low",  ("#0072B2", "o", "-")),
    (r"Mid-$k$  ($5 < k \leq 16$)", "band_mid",  ("#D55E00", "s", "--")),
    (r"High-$k$ ($k > 16$)",        "band_high", ("#009E73", "^", "-.")),
]


def main() -> None:
    d = np.load(SERIES)
    t = d["time"]
    print(f"[data] n={len(t)}  t={t[0]:.3f}..{t[-1]:.3f}")
    for _, key, _ in BANDS:
        v = d[key]
        print(f"[data] {key}: t0={v[0]:.3e}  min={np.nanmin(v):.3e}  max={np.nanmax(v):.3e}")

    # 移除 t=0 那一幀(IC high-k 假象)；保險起見也擋掉 non-finite
    sl = slice(1, None)
    t_p = t[sl]

    apply_journal_rcparams()  # sans-serif，與其他 result 圖一致
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    mev = max(1, len(t_p) // 12)
    for label, key, (c, m, ls) in BANDS:
        y = d[key][sl].astype(float)
        y[~np.isfinite(y)] = np.nan
        ax.semilogy(t_p, y, color=c, marker=m, linestyle=ls, markevery=mev,
                    markersize=4, linewidth=1.4, label=label)

    ax.axhline(1.0, color="0.6", linestyle=":", linewidth=0.9)  # 100% 飽和參考
    ax.text(t_p[-1] * 1.01, 1.0, "100%", color="0.5", fontsize=7, va="center", ha="left")

    ax.set_xlabel(r"Time $t$ (s)")
    ax.set_ylabel("Band relative error")
    ax.set_xlim(t_p[0], t_p[-1])
    # ylim 由 matplotlib 自動決定
    ax.legend(frameon=False, loc="lower left")
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_STEM}.{ext}")
    plt.close(fig)
    print(f"[saved] {OUT_STEM}.pdf / .png")


if __name__ == "__main__":
    main()
