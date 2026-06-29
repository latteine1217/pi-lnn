"""plot_forward_cfd_spectrum.py — Forward-CFD baseline vs DNS 能譜（統一風格）。

What:
    從既有的 forward-CFD baseline artifact（home-gpu 跑的正確 ETDRK4 結果，
    reports/forward_cfd_baseline_T5_rank40.npz）讀取 t=5 的 (u,v) 場，
    用 evaluator 的 energy_spectrum_1d 算能譜，畫出 DNS vs Forward-CFD 比較。

Why:
    先前在 lab-server 重寫的 ETDRK4 forward solver 發散成 NaN（重複造輪子 +
    數值不穩定）。正確結果早已由 forward_cfd_baseline.py 算出並存成 npz；
    本腳本只負責「讀 npz → 算譜 → 統一風格出圖」，不重跑 solver。

    artifact 內容（KE_pred=0.1200, KE_ref=0.1248）與 §Results Table 一致。

Usage:
    uv run python scripts/plot_forward_cfd_spectrum.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pi_con.plot_style import apply_journal_rcparams  # noqa: E402
from evaluate_deeponet_cfc import energy_spectrum_1d  # noqa: E402

NPZ = ROOT / "reports/forward_cfd_baseline_T5_rank40.npz"
OUT = ROOT / "thesis/figures/results/forward_cfd_spectrum_t5"
K_FORCING = 2.0
DOMAIN_L = 1.0


def main() -> None:
    d = dict(np.load(NPZ))
    N = d["u_pred"].shape[0]
    dx = DOMAIN_L / N

    if not (np.isfinite(d["u_pred"]).all() and np.isfinite(d["v_pred"]).all()):
        raise ValueError("forward-CFD field contains non-finite values — bad artifact")

    k_dns, e_dns = energy_spectrum_1d(d["u_ref"], d["v_ref"], dx)
    k_cfd, e_cfd = energy_spectrum_1d(d["u_pred"], d["v_pred"], dx)

    apply_journal_rcparams()
    fig, ax = plt.subplots(figsize=(5.5, 2.8), constrained_layout=True)
    ax.loglog(k_dns, e_dns, color="#000000", linestyle="-", linewidth=1.4, label="DNS")
    ax.loglog(k_cfd, e_cfd, color="#0072B2", linestyle="--", linewidth=1.4,
              label="Forward-CFD baseline")

    idx_f = np.argmin(np.abs(k_dns - K_FORCING))
    e0 = e_dns[idx_f]
    k_ref = np.logspace(np.log10(K_FORCING), np.log10(k_dns[-1] * 0.8), 80)
    ax.loglog(k_ref, e0 * (k_ref / K_FORCING) ** (-5 / 3),
              color="0.5", linestyle=":", linewidth=0.9, label=r"$k^{-5/3}$")
    ax.axvline(K_FORCING, color="black", linestyle="-.", linewidth=0.8,
               label=f"$k_f={K_FORCING:.0f}$")

    ax.set_xlabel(r"Wavenumber $k$ (1/m)")
    ax.set_ylabel(r"Energy $E(k)$ (m$^3$/s$^2$)")
    ax.legend(loc="lower left")
    ax.grid(True, which="both", alpha=0.25)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT.with_suffix(f".{ext}"))
    plt.close(fig)
    print(f"[saved] {OUT}.pdf / .png "
          f"(KE_pred={0.5*np.mean(d['u_pred']**2+d['v_pred']**2):.4f}, "
          f"KE_ref={0.5*np.mean(d['u_ref']**2+d['v_ref']**2):.4f})")


if __name__ == "__main__":
    main()
