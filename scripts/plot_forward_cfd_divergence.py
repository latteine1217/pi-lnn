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
開放迴路軌跡取自 `reports/forward_cfd_rerun_T5_rank40.npz`——由 repo 內的
`scripts/forward_cfd_baseline.py --integrate` 重跑產生,每 0.025 s 存一幀(201 幀),
故為**逐時實測**軌跡,不再是首尾兩點的示意連線。

⚠️ 這是重跑,不是原始 .npz。原始產物只存首尾兩張場,且其 solver 腳本不在 git 歷史;
本腳本的配方經指紋比對(5 項中 4 項吻合、IC 殘差約 1%)重建。混沌放大使兩者端點不同
(u 160.3% vs 152.8%、v 172.9% vs 203.9%),因此**兩批資料不可混用**——圖與 appendix07
的數字必須同時出自重跑。
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
FCFD_NPZ = ROOT / "reports/forward_cfd_rerun_T5_rank40.npz"
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


def load_forward_cfd() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (time, {component: rel-L2 series in percent}) for the open-loop free run."""
    d = np.load(FCFD_NPZ)
    print(f"[data] open-loop free run: {len(d['time'])} frames from {FCFD_NPZ.name} "
          f"(rerun of scripts/forward_cfd_baseline.py --integrate)")
    return d["time"], {c: d[f"{c}_rel_L2"] * 100.0 for c in ("u", "v")}


def main() -> None:
    t, picon = load_picon()
    t_f, fcfd = load_forward_cfd()

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

    f = fcfd["u"]
    ax.semilogy(t_f, f, color=FCFD_C, linewidth=1.7, zorder=4,
                label=r"Open-loop free run (no data assimilated)")
    ax.text(2.55, 26, rf"$\times {f[-1] / f[0]:.1f}$", color=FCFD_C, fontsize=9.5,
            ha="center", va="bottom", rotation=27, fontweight="bold")

    ax.set_xlabel(r"time $t$")
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
        e0, e5 = fcfd[comp][0], fcfd[comp][-1]
        print(f"  {comp}: open-loop {e0:5.2f}% -> {e5:6.1f}%  ({e5/e0:.1f}x, diverges) | "
              f"PI-CON {a[:, 0].mean():5.1f}% -> {a[:, -1].mean():5.2f}%  "
              f"({a[:, -1].mean()/a[:, 0].mean():.2f}x, converges)")


if __name__ == "__main__":
    main()
