"""合併 forcing-mode diagnostic：amp ratio + phase error 2-panel（取代原 Fig 4.8/4.9 兩張）。

What: 讀 EXP-245 B3 五個 seed 的 series.npz，畫 (a) forcing-mode |û_{k_f}| 的 pred/DNS
      振幅比、(b) forcing-mode 相位誤差 arg(û_pred)−arg(û_DNS)，皆 n=5 mean±σ envelope。
Why : k_f=2 在 sensor-resolvable low band，其振幅恢復是低頻能量恢復的特例（已被 spectrum/
      band-error 覆蓋）；相位是 spectrum/band 看不到的補充 sanity check。合併成一張 compact
      diagnostic，避免單一受迫低模態獨佔兩張主文圖。
資料: artifacts/exp245_seeds/eval_245_seed{a,b,c,d,e}_mac/series.npz（seed 42/1/2/3/4）。
"""
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pi_con.plot_style import apply_journal_rcparams  # noqa: E402

SEED_DIRS = [ROOT / f"artifacts/exp245_seeds/eval_245_seed{s}_mac" for s in "abcde"]
OUTDIR = ROOT / "thesis/figures/results"
OUTDIR.mkdir(parents=True, exist_ok=True)
PRED = "#D55E00"


def wrap(x: np.ndarray) -> np.ndarray:
    """相位差 wrap 到 (-π, π]，避免 ±π 跳變污染。"""
    return np.angle(np.exp(1j * x))


def main() -> None:
    series = [np.load(d / "series.npz") for d in SEED_DIRS]
    t = series[0]["time"]
    amp_ratio = np.stack([s["kf_amp"] / np.maximum(s["kf_amp_dns"], 1e-12) for s in series])
    phase_err = np.stack([wrap(s["kf_phase"] - s["kf_phase_dns"]) for s in series])

    apply_journal_rcparams()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # (a) amplitude ratio
    m, sd = amp_ratio.mean(0), amp_ratio.std(0, ddof=1)
    for i in range(amp_ratio.shape[0]):
        ax1.plot(t, amp_ratio[i], color=PRED, lw=0.5, alpha=0.3)
    ax1.plot(t, m, color=PRED, ls="--", lw=1.4, label=r"PI-CON / DNS ($n=5$)")
    ax1.fill_between(t, m - sd, m + sd, color=PRED, alpha=0.18)
    ax1.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax1.set_xlabel(r"$t$ [s]")
    ax1.set_ylabel(r"$|\hat{u}_{k_f}|_{\rm pred}\,/\,|\hat{u}_{k_f}|_{\rm DNS}$ (dimensionless)")
    ax1.set_title("(a) Forcing-mode amplitude ratio", fontsize=9)
    ax1.set_xlim(0, 5)
    ax1.legend(frameon=True, fontsize=8)

    # (b) phase error
    mp, sdp = phase_err.mean(0), phase_err.std(0, ddof=1)
    for i in range(phase_err.shape[0]):
        ax2.plot(t, phase_err[i], color=PRED, lw=0.5, alpha=0.3)
    ax2.plot(t, mp, color=PRED, ls="--", lw=1.4)
    ax2.fill_between(t, mp - sdp, mp + sdp, color=PRED, alpha=0.18)
    ax2.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax2.set_xlabel(r"$t$ [s]")
    ax2.set_ylabel(r"$\arg\hat{u}_{k_f}^{\rm pred}-\arg\hat{u}_{k_f}^{\rm DNS}$ [rad]")
    ax2.set_title("(b) Forcing-mode phase error", fontsize=9)
    ax2.set_xlim(0, 5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"kf_mode_diagnostic.{ext}")
    plt.close(fig)

    late = t >= 1.0
    print(f"[kf] wrote {OUTDIR/'kf_mode_diagnostic.pdf'} (+png)")
    print(f"[kf] amp ratio (t>=1) mean={m[late].mean():.3f}, range [{m[late].min():.3f}, {m[late].max():.3f}]")
    print(f"[kf] phase err (t>=1) mean={mp[late].mean():+.4f} rad, |max|={np.abs(mp[late]).max():.4f} rad, end={mp[-1]:+.4f}")


if __name__ == "__main__":
    main()
