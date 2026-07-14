"""energy_spectrum.pdf — DNS vs PI-CON radial energy spectrum at t=5 (unified style).

What: 從既有 eval fields export（artifacts/eval_245_seed42_fields/fields.npz，含 u_pred/
      v_pred/u_ref/v_ref 全場）在 t=5 直接算 radial E(k)，畫 DNS vs PI-CON，免 checkpoint。
Why : 舊圖由 evaluator 產生（PI-CON 橘）；本腳本用 pi_con.plot_style 統一語意色
      （DNS 黑 solid、PI-CON 藍 dashed），與其他 result 圖一致，且不需重跑模型。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pi_con.plot_style import apply_journal_rcparams, DNS, PICON  # noqa: E402
from evaluate_deeponet_cfc import energy_spectrum_1d  # noqa: E402

FIELDS = ROOT / "artifacts/eval_245_seed42_fields/fields.npz"
OUT = ROOT / "thesis/figures/results/energy_spectrum"
K_F = 2.0
K_SENSOR = np.sqrt(100.0 / np.pi)  # ≈ 5.64


def main() -> None:
    apply_journal_rcparams()
    d = np.load(FIELDS)
    t = np.asarray(d["time"], dtype=np.float64)
    it = int(np.argmin(np.abs(t - 5.0)))
    L = 1.0
    dx = L / d["u_ref"].shape[-1]

    k_dns, e_dns = energy_spectrum_1d(d["u_ref"][it], d["v_ref"][it], dx)
    k_pred, e_pred = energy_spectrum_1d(d["u_pred"][it], d["v_pred"][it], dx)

    fig, ax = plt.subplots(figsize=(5.5, 3.0), constrained_layout=True)
    md, mp = e_dns > 0, e_pred > 0
    ax.loglog(k_dns[md], e_dns[md], color=DNS, linestyle="-", linewidth=1.4, label="DNS")
    ax.loglog(k_pred[mp], e_pred[mp], color=PICON, linestyle="--", linewidth=1.4, label="PI-CON")
    ax.axvline(K_F, color="#000000", linestyle="-.", linewidth=0.8)
    ax.text(K_F * 1.05, ax.get_ylim()[0] * 3, r"$k_f=2$", fontsize=7, rotation=90, va="bottom")
    ax.axvline(K_SENSOR, color="#009E73", linestyle="--", linewidth=1.0)
    ax.text(K_SENSOR * 1.05, ax.get_ylim()[0] * 3,
            r"$k_{\max}^{\rm sensor}\approx 5.64$", fontsize=7, color="#009E73",
            rotation=90, va="bottom")
    ax.set_xlabel(r"wavenumber $k$ (1/m)")
    ax.set_ylabel(r"$E(k)$ (m$^3$/s$^2$)")
    ax.legend(loc="lower left")
    for ext in ("pdf", "png"):
        fig.savefig(OUT.with_suffix(f".{ext}"))
    plt.close(fig)
    print(f"[energy_spectrum] wrote {OUT}.pdf (+png); t={t[it]:.2f}")


if __name__ == "__main__":
    main()
