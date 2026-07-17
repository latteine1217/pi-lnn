"""Replot band-energy relative error vs time (journal style), dropping the t=0 IC spike.

What: 從 EXP-245 五顆 seed 的 series.npz(band_low/mid/high + time)畫期刊風格
      band-error-vs-time，含 mean ± 1σ 包絡與個別 seed 線。
Why: 原圖 t=0 high-k 相對誤差爆到 ~1e5(IC high-k 能量近零的假象),把 y 軸撐爆、
     壓垮其餘可看性。此腳本「移除 t=0 那一幀」並套 journal_style,輸出向量 PDF。

Why multi-seed (2026-07-17)
===========================
原本硬編 `artifacts/eval_245_seed42_export/series.npz` 單一 seed，但論文與投影片都把
本圖標成 EXP-245 n=5 —— 圖上卻沒有任何離散度，與同頁 n=5 的 ± 數字不一致。改為讀滿
五顆 seed 並畫 mean ± 1σ。

量的定義（追到 evaluate_deeponet_cfc.py）
=========================================
逐個時間快照 t：
  1. energy_spectrum_1d(u, v, dx) — 2D FFT 後依 k=√(kx²+ky²) 徑向平均得 E(k)，
     波數單位 cycles/domain（故 k_f=2 對得上）；超過 Nyquist 的角落 bin 被 mask。
  2. compute_band_energies — 以固定切點把 E(k) 加總成三段（k=0 的 DC 被排除）：
     low k≤5 · mid 5<k≤16 · high k>16。
  3. band 相對誤差 = |E_band(pred) − E_band(ref)| / E_band(ref)。

判讀要點：這是 band 積分**能量**的誤差，不是場誤差 —— 能量對了不代表相位對了。
且此指標不對稱：低估最多到 100%（E_pred=0 ⇒ 誤差恰為 1），高估則無上界。
high band 貼在 1.0 即「模型在該帶放了近乎零能量」，不是「錯到最大」。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pi_con.plot_style import apply_journal_rcparams  # noqa: E402

# EXP-245 multi-seed group (n=5)。用 glob 而非硬編路徑，但**斷言數量**：
# 2026-07-17 這批 eval 目錄曾由 `_mac` 改名為 `_final`，硬編路徑會安靜地讀到
# 被取代的舊資料（實測差 0.05 pp，與論文表對不上）。寧可大聲失敗。
SEED_GLOB = "artifacts/exp245_seeds/eval_245_seed?_final"
N_SEEDS_EXPECTED = 5
OUT_STEM = ROOT / "thesis/figures/results/band_energy_rel_error_vs_time"

# 三個 band：label, npz-key, (color, marker, linestyle) —— colorblind-safe + 灰階可分
BANDS = [
    (r"Low-$k$  ($k \leq 5$)",       "band_low",  ("#0072B2", "o", "-")),
    (r"Mid-$k$  ($5 < k \leq 16$)", "band_mid",  ("#D55E00", "s", "--")),
    (r"High-$k$ ($k > 16$)",        "band_high", ("#009E73", "^", "-.")),
]


def load_seeds() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load the n=5 EXP-245 series, stacked as (n_seeds, n_time) per band."""
    dirs = sorted(ROOT.glob(SEED_GLOB))
    if len(dirs) != N_SEEDS_EXPECTED:
        raise SystemExit(
            f"[abort] expected {N_SEEDS_EXPECTED} seed dirs matching {SEED_GLOB}, "
            f"found {len(dirs)}: {[d.name for d in dirs]}\n"
            f"        本圖標示為 n=5；seed 數不符時不畫，以免圖與標籤不一致。"
        )
    series = [np.load(d / "series.npz") for d in dirs]
    t = series[0]["time"]
    for d, s in zip(dirs, series):
        if not np.allclose(s["time"], t):
            raise SystemExit(f"[abort] time grid of {d.name} differs from {dirs[0].name}")
    stack = {key: np.stack([s[key] for s in series]) for _, key, _ in BANDS}
    print(f"[data] {len(dirs)} seeds: {', '.join(d.name for d in dirs)}")
    return t, stack


def main() -> None:
    t, stack = load_seeds()
    print(f"[data] n_t={len(t)}  t={t[0]:.3f}..{t[-1]:.3f}")

    # 移除 t=0 那一幀(IC high-k 能量近零 -> 分母趨零, rel-err ~1e5)
    sl = slice(1, None)
    t_p = t[sl]

    apply_journal_rcparams()  # sans-serif，與其他 result 圖一致
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    mev = max(1, len(t_p) // 12)

    for label, key, (c, m, ls) in BANDS:
        a = stack[key][:, sl].astype(float)
        a[~np.isfinite(a)] = np.nan
        mean = np.nanmean(a, axis=0)
        std = np.nanstd(a, axis=0, ddof=1)

        # 個別 seed（點線、半透明）
        for s in range(a.shape[0]):
            ax.semilogy(t_p, a[s], color=c, linewidth=0.6, alpha=0.35, linestyle=":",
                        zorder=2)
        # ±1σ 包絡。此量恆正而 y 軸為對數：mean-σ 若 ≤ 0（實測 band_low 200 點中
        # 有 1 點，位於誤差過零的深谷，σ≈mean 表分布高度偏斜）則夾到正的地板，
        # 讓包絡在該處延伸至軸底而非留下破洞。
        lo = np.maximum(mean - std, np.nanmin(a[a > 0]) * 0.5)
        ax.fill_between(t_p, lo, mean + std, color=c, alpha=0.18, linewidth=0, zorder=3)
        # mean（原本的線型 + marker 維持不變）
        ax.semilogy(t_p, mean, color=c, marker=m, linestyle=ls, markevery=mev,
                    markersize=4, linewidth=1.4, label=label, zorder=4)

    ax.axhline(1.0, color="0.6", linestyle=":", linewidth=0.9)  # 100% 飽和參考
    ax.text(t_p[-1] * 1.01, 1.0, "100%", color="0.5", fontsize=7, va="center", ha="left")

    ax.set_xlabel(r"Time $t$ (s)")
    ax.set_ylabel("Band relative error")
    ax.set_xlim(t_p[0], t_p[-1])
    # ylim 由 matplotlib 自動決定
    leg = ax.legend(frameon=False, loc="lower left", title=r"mean $\pm 1\sigma$, $n=5$ seeds")
    leg.get_title().set_fontsize(7)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_STEM}.{ext}")
    plt.close(fig)
    print(f"[saved] {OUT_STEM}.pdf / .png")


if __name__ == "__main__":
    main()
