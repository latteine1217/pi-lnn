"""Open-loop forward-CFD divergence vs sensor-conditioned reconstruction.

What: 畫 velocity rel-L2(t) —— PI-CON 全窗曲線(n=5 mean ± 1σ)對照 open-loop
      forward-CFD 的兩個實測端點,顯示「發散 vs 收斂」。
Why : 原本的 P25 用表格比較,但兩欄取了不同時間窗(forward-CFD 的 t=5 snapshot
      對 PI-CON 的 t≳3.3 late-window mean),看起來像挑對自己有利的窗。改畫
      軌跡後不需要挑窗:兩端點都報,方向自明。

      這也修正了原表格的誤導:forward-CFD 在 t=0 其實**很準**(u rel-L2 5.2%,
      它用 200 張 offline DNS snapshot 建 POD-rank-40 基底),失敗不在起點而在
      open-loop 積分——chaotic amplification 把 u 誤差放大 29.3×。PI-CON 相反,
      起點差(warm-up,u 26.9%)但全程 re-condition on the sensor stream 而收斂
      到 7.28%。兩條軌跡交叉,這才是 open-loop 與 sensor-conditioned 的本質差別。

資料誠實性
==========
forward-CFD 只存了 t=0 與 t=5 兩張場(reports/forward_cfd_baseline_T5_rank40.npz;
產生它的 solver 腳本不在 repo 也不在 git 歷史)。因此它只畫兩個實測 marker,
中間以虛線連接並在圖例標明無中間快照——不以插值冒充實測軌跡。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pi_con.plot_style import apply_journal_rcparams, PICON  # noqa: E402

# 與 band-energy / enstrophy 圖同一組防呆:glob 但斷言數量。
SEED_GLOB = "artifacts/exp245_seeds/eval_245_seed?_final"
N_SEEDS_EXPECTED = 5
FCFD_JSON = ROOT / "reports/forward_cfd_baseline_T5_rank40.json"
OUT_STEM = ROOT / "thesis/figures/results/forward_cfd_divergence"

FCFD_C = "#E97132"  # forward-CFD:橘(與 PI-CON 紫/藍區隔,colourblind-safe)


def load_picon() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    dirs = sorted(ROOT.glob(SEED_GLOB))
    if len(dirs) != N_SEEDS_EXPECTED:
        raise SystemExit(
            f"[abort] expected {N_SEEDS_EXPECTED} seed dirs matching {SEED_GLOB}, "
            f"found {len(dirs)}: {[d.name for d in dirs]}"
        )
    series = [np.load(d / "series.npz") for d in dirs]
    t = series[0]["time"]
    for d, s in zip(dirs, series):
        if not np.allclose(s["time"], t):
            raise SystemExit(f"[abort] time grid of {d.name} differs from {dirs[0].name}")
    print(f"[data] PI-CON {len(dirs)} seeds: {', '.join(d.name for d in dirs)}")
    return t, {c: np.stack([s[f"{c}_rel_L2"] for s in series]) * 100.0 for c in ("u", "v")}


def load_forward_cfd() -> dict[str, tuple[float, float]]:
    """Return {component: (rel-L2 at t=0, rel-L2 at t=5)} in percent."""
    d = json.loads(FCFD_JSON.read_text())
    t0, tT = d["metrics_at_t0"], d["metrics_at_T"]
    print(f"[data] forward-CFD: {d['method']}, POD rank {d['pod_rank']} "
          f"from {d['pod_snapshots_used']} DNS snapshots")
    return {c: (t0[f"{c}_rel_L2"] * 100.0, tT[f"{c}_rel_L2"] * 100.0) for c in ("u", "v")}


def main() -> None:
    t, picon = load_picon()
    fcfd = load_forward_cfd()

    apply_journal_rcparams()
    fig, ax = plt.subplots(figsize=(5.6, 3.2))

    # 100% 參考:誤差達到訊號量級,場已與參考去相關。
    ax.axhline(100.0, color="0.65", linestyle=":", linewidth=0.9, zorder=1)
    ax.text(5.02, 100.0, "100%", color="0.5", fontsize=7, va="center", ha="left")

    # 只畫 u:訊息是「發散 vs 收斂」,一個分量就足夠;四條線反而讓圖失焦。
    # v 的端點數字改由投影片右欄承載(6.1% -> 203.9%,同樣的方向)。
    a = picon["u"]
    mean, std = a.mean(axis=0), a.std(axis=0, ddof=1)
    ax.fill_between(t, mean - std, mean + std, color=PICON, alpha=0.18,
                    linewidth=0, zorder=3)
    ax.semilogy(t, mean, color=PICON, linewidth=1.7, zorder=4,
                label=r"PI-CON, sensor-conditioned (mean $\pm 1\sigma$, $n=5$)")

    e0, e5 = fcfd["u"]
    # forward-CFD 只有首尾兩張場。用「標註箭頭」而非資料線:箭頭是示意的視覺語言,
    # 不會被誤讀成量到的軌跡;直線在對數軸上則會被讀成實測的指數成長。
    ax.annotate("", xy=(4.86, e5 * 0.82), xytext=(0.14, e0 * 1.18),
                arrowprops=dict(arrowstyle="-|>", color=FCFD_C, linewidth=1.3,
                                linestyle=(0, (5, 3)), alpha=0.75,
                                shrinkA=2, shrinkB=2), zorder=2)
    ax.semilogy([0.0, 5.0], [e0, e5], color=FCFD_C, linestyle="none",
                marker="o", markersize=7.5, markerfacecolor="white",
                markeredgewidth=1.9, zorder=5,
                label=r"Forward-CFD, open-loop (only these 2 are stored)")
    # 只標放大倍率;「中間沒有快照」已由圖例的 "only these 2 are stored" 說明,不重複。
    ax.text(2.5, 33, r"$\times 29$", color=FCFD_C, fontsize=9.5,
            ha="center", va="bottom", rotation=20, fontweight="bold")

    ax.set_xlabel(r"time $t$ (s)")
    ax.set_ylabel(r"$u$ rel-$L_2$ (%)")
    ax.set_xlim(0, 5)
    ax.set_ylim(3.5, 400)
    ax.legend(frameon=False, fontsize=7.2, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_STEM}.{ext}")
    plt.close(fig)

    print(f"[saved] {OUT_STEM}.pdf / .png")
    for comp in ("u", "v"):
        a = picon[comp]
        e0, e5 = fcfd[comp]
        print(f"  {comp}: forward-CFD {e0:5.1f}% -> {e5:6.1f}%  ({e5/e0:.1f}x, diverges) | "
              f"PI-CON {a[:, 0].mean():5.1f}% -> {a[:, -1].mean():5.2f}%  "
              f"({a[:, -1].mean()/a[:, 0].mean():.2f}x, converges)")


if __name__ == "__main__":
    main()
