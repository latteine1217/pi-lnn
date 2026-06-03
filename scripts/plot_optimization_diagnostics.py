"""#6 優化收斂診斷圖：loss curves + GradNorm 權重 + Augmented-Lagrangian dual λ。

What: 從 B3 (EXP-245) metrics.jsonl(20000 步) 畫三聯圖，直接證明訓練收斂、GradNorm
      動態平衡四個 task、AL dual λ 隨 continuity 違反成長（constraint active）。
Why : 方法強調 stiff multi-task + AL + GradNorm + SOAP，但論文無任何收斂圖；AL λ 軌跡是
      「continuity constraint 主動」的直接證據（取代僅以最終 div ratio 0.39% 間接佐證）。
資料: artifacts/_lab_rsync/exp245_b3_seed42/metrics.jsonl。
"""
import json
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pi_con.plot_style import apply_journal_rcparams  # noqa: E402

METRICS = ROOT / "artifacts/_lab_rsync/exp245_b3_seed42/metrics.jsonl"
OUTDIR = ROOT / "thesis/figures/results"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Wong 色盲友善 palette
C = {
    "data": "#0072B2", "ns_u": "#D55E00", "ns_v": "#E69F00",
    "cont": "#009E73", "total": "#000000", "lam": "#CC79A7",
}


def load_metrics() -> dict:
    rows = [json.loads(line) for line in METRICS.read_text().splitlines() if line.strip()]
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows], dtype=float) for k in keys}


def main() -> None:
    m = load_metrics()
    step = m["step"]

    apply_journal_rcparams()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 2.8))

    # (a) loss curves（log y）
    ax1.semilogy(step, m["l_data"], color=C["data"], lw=1.1, label=r"$\mathcal{L}_{\rm data}$")
    ax1.semilogy(step, m["l_ns"], color=C["ns_u"], lw=1.1, label=r"$\mathcal{L}_{\rm NS}$")
    ax1.semilogy(step, m["l_cont"], color=C["cont"], lw=1.1, label=r"$\mathcal{L}_{\rm cont}$")
    ax1.semilogy(step, m["l_total"], color=C["total"], lw=1.1, label=r"$\mathcal{L}_{\rm total}$")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("loss (dimensionless)")
    ax1.set_title("(a) Loss components", fontsize=9)
    ax1.legend(frameon=True, fontsize=7, ncol=2)

    # (b) GradNorm 權重
    ax2.plot(step, m["gn_w_data"], color=C["data"], lw=1.2, label=r"$w_{\rm data}$")
    ax2.plot(step, m["gn_w_ns_u"], color=C["ns_u"], lw=1.2, label=r"$w_{{\rm NS},u}$")
    ax2.plot(step, m["gn_w_ns_v"], color=C["ns_v"], lw=1.2, ls="--", label=r"$w_{{\rm NS},v}$")
    ax2.plot(step, m["gn_w_cont"], color=C["cont"], lw=1.2, label=r"$w_{\rm cont}$")
    ax2.set_xlabel("training step")
    ax2.set_ylabel("GradNorm weight (dimensionless)")
    ax2.set_title("(b) Task weights", fontsize=9)
    ax2.legend(frameon=True, fontsize=7, ncol=2)

    # (c) AL dual λ + signed continuity residual EMA
    ax3.plot(step, m["al_lambda_cont"], color=C["lam"], lw=1.4, label=r"$\lambda_{\rm cont}$")
    ax3.set_xlabel("training step")
    ax3.set_ylabel(r"AL dual $\lambda_{\rm cont}$ (dimensionless)", color=C["lam"])
    ax3.tick_params(axis="y", labelcolor=C["lam"])
    ax3.set_title("(c) Augmented-Lagrangian dual", fontsize=9)
    axr = ax3.twinx()
    axr.plot(step, m["al_C_ema_cont"], color="#56B4E9", lw=1.0, alpha=0.8,
             label=r"$\tilde{C}_{\rm ema}$")
    axr.set_ylabel(r"signed continuity $\tilde{C}_{\rm ema}$", color="#56B4E9")
    axr.tick_params(axis="y", labelcolor="#56B4E9")
    axr.axhline(0.0, color="grey", lw=0.6, ls=":")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"optimization_diagnostics.{ext}")
    plt.close(fig)

    print(f"[#6] wrote {OUTDIR/'optimization_diagnostics.pdf'} (+png)")
    print(f"[#6] final λ_cont={m['al_lambda_cont'][-1]:.4f}, "
          f"max λ={m['al_lambda_cont'].max():.4f}")
    print(f"[#6] final weights: data={m['gn_w_data'][-1]:.3f} ns_u={m['gn_w_ns_u'][-1]:.3f} "
          f"ns_v={m['gn_w_ns_v'][-1]:.3f} cont={m['gn_w_cont'][-1]:.3f}")
    print(f"[#6] final losses: data={m['l_data'][-1]:.4e} ns={m['l_ns'][-1]:.4e} "
          f"cont={m['l_cont'][-1]:.4e}")


if __name__ == "__main__":
    main()
