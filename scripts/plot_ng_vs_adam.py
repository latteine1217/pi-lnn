"""Plot NG vs Adam loss curves on Kolmogorov Re=1000.

期刊風格 (NeurIPS/ICLR)：無方塊 marker、合適字型、DPI=200、tight spines。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NG_LOG = ROOT / "artifacts/exp_ng_001_re1000_smoke/loss_log.json"
ADAM_LOG = ROOT / "artifacts/exp_ng_001_baseline_adam/loss_log.json"
OUT_DIR = ROOT / "artifacts"


def _load(path: Path) -> dict:
    with open(path) as f:
        logs = json.load(f)
    return {
        "step": np.array([m["step"] for m in logs]),
        "wall": np.array([m["wall"] for m in logs]),
        "l_data": np.array([m["l_data"] for m in logs]),
        "l_phys": np.array([m["l_physics"] for m in logs]),
        "l_total": np.array([m["l_total"] for m in logs]),
    }


def main() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    })

    ng = _load(NG_LOG)
    adam = _load(ADAM_LOG)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)

    # Panel (a): loss vs step
    ax = axes[0]
    ax.semilogy(ng["step"], ng["l_data"], color="#d62728", label="NG (l_data)")
    ax.semilogy(ng["step"], ng["l_total"], color="#d62728", ls="--", alpha=0.6, label="NG (l_total)")
    ax.semilogy(adam["step"], adam["l_data"], color="#1f77b4", label="Adam (l_data)")
    ax.semilogy(adam["step"], adam["l_total"], color="#1f77b4", ls="--", alpha=0.6, label="Adam (l_total)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("(a) Loss vs iteration", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(True, which="both", alpha=0.3, lw=0.4)

    # Panel (b): loss vs wall-time
    ax = axes[1]
    ax.semilogy(ng["wall"], ng["l_data"], color="#d62728", label="NG")
    ax.semilogy(adam["wall"], adam["l_data"], color="#1f77b4", label="Adam")
    ax.set_xlabel("Wall-time (s)")
    ax.set_ylabel("l_data")
    ax.set_title("(b) Loss vs wall-time", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(True, which="both", alpha=0.3, lw=0.4)

    out = OUT_DIR / "ng_vs_adam_kolmogorov_re1000.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")

    # Numeric summary
    print(f"\nNG  : steps={len(ng['step'])}, wall_total={ng['wall'][-1]:.1f}s, "
          f"l_data: init={ng['l_data'][0]:.3e} final={ng['l_data'][-1]:.3e} min={ng['l_data'].min():.3e}")
    print(f"Adam: steps={len(adam['step'])}, wall_total={adam['wall'][-1]:.1f}s, "
          f"l_data: init={adam['l_data'][0]:.3e} final={adam['l_data'][-1]:.3e} min={adam['l_data'].min():.3e}")


if __name__ == "__main__":
    main()
