"""#5 重建場 enstrophy / dissipation budget 圖（誠實揭露 high-k 截斷後果）。

What: 用既有 eval series.npz 的 enstrophy(t) 對照重建場 vs DNS，並換算 dissipation
      ε(t)=2ν·Z(t)（ν=1/Re）。
Why : K=100 sensor Nyquist 截斷 → 重建場缺中高頻渦量 → enstrophy/dissipation 系統性低估。
      這是 sensor-information ceiling 的物理後果，誠實量化反而強化「中高頻 bounded 非架構失敗」，
      並界定 out-of-scope（TKE/dissipation budget 任務需更多 sensor）。
資料: artifacts/eval_245_seed42_export/series.npz（B3 LES-T50 seed42, 與 Table 4.7 KE 5.90% 同 run）。
"""
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pi_con.plot_style import apply_journal_rcparams  # noqa: E402

NU = 1e-4  # Re = 10000 → ν = 1/Re
SERIES = ROOT / "artifacts/eval_245_seed42_export/series.npz"
OUTDIR = ROOT / "thesis/figures/results"
OUTDIR.mkdir(parents=True, exist_ok=True)

DNS_C = "#000000"
PRED_C = "#D55E00"


def main() -> None:
    d = np.load(SERIES)
    t = d["time"]
    Z_pred, Z_dns = d["enstrophy"], d["enstrophy_dns"]
    eps_pred, eps_dns = 2 * NU * Z_pred, 2 * NU * Z_dns

    # 後 spin-up 窗 t>=1 的 time-mean deficit（與 thesis「t>=1 KE~5%」一致的穩態窗）
    m = t >= 1.0
    z_deficit = (1.0 - Z_pred[m].mean() / Z_dns[m].mean()) * 100.0
    z_deficit_all = (1.0 - Z_pred.mean() / Z_dns.mean()) * 100.0

    apply_journal_rcparams()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ax1.plot(t, Z_dns, color=DNS_C, lw=1.4, label="DNS")
    ax1.plot(t, Z_pred, color=PRED_C, ls="--", lw=1.4, label="PI-CON")
    ax1.set_xlabel(r"$t$ [s]")
    ax1.set_ylabel(r"enstrophy $\mathcal{Z}(t)$ [1/s$^2$]")
    ax1.set_xlim(0, 5)
    ax1.legend(frameon=True, fontsize=8)

    ax2.plot(t, eps_dns, color=DNS_C, lw=1.4, label="DNS")
    ax2.plot(t, eps_pred, color=PRED_C, ls="--", lw=1.4, label="PI-CON")
    ax2.set_xlabel(r"$t$ [s]")
    ax2.set_ylabel(r"dissipation $\varepsilon(t)=2\nu\mathcal{Z}$ [m$^2$/s$^3$]")
    ax2.set_xlim(0, 5)
    ax2.legend(frameon=True, fontsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"dissipation_budget.{ext}")
    plt.close(fig)

    print(f"[#5] wrote {OUTDIR/'dissipation_budget.pdf'} (+png)")
    print(f"[#5] enstrophy deficit (t>=1 mean): {z_deficit:.1f}%   (full window: {z_deficit_all:.1f}%)")
    print(f"[#5] Z_dns(t>=1) mean={Z_dns[m].mean():.2f}, Z_pred mean={Z_pred[m].mean():.2f} [1/s^2]")
    print(f"[#5] eps_dns(t>=1) mean={eps_dns[m].mean():.4e}, eps_pred mean={eps_pred[m].mean():.4e} [m^2/s^3]")


if __name__ == "__main__":
    main()
