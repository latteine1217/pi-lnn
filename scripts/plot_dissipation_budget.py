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
from pi_con.plot_style import apply_journal_rcparams, DNS as DNS_C, PICON as PRED_C  # noqa: E402

NU = 1e-4  # Re = 10000 → ν = 1/Re
SERIES = ROOT / "artifacts/eval_245_seed42_export/series.npz"
OUTDIR = ROOT / "thesis/figures/results"
OUTDIR.mkdir(parents=True, exist_ok=True)

# DNS_C / PRED_C imported from pi_con.plot_style (Okabe--Ito semantic palette)


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
    # 單 panel：enstrophy Z(t)（左軸）；dissipation ε=2νZ 僅為常數倍率，
    # 用右側 twin 軸提供換算刻度，不重畫同形曲線（避免左右兩 panel 冗餘）。
    fig, ax1 = plt.subplots(figsize=(5.2, 3.0))

    ax1.plot(t, Z_dns, color=DNS_C, lw=1.4, label="DNS")
    ax1.plot(t, Z_pred, color=PRED_C, ls="--", lw=1.4, label="PI-CON")
    ax1.set_xlabel(r"$t$ [s]")
    ax1.set_ylabel(r"enstrophy $\mathcal{Z}(t)$ [1/s$^2$]")
    ax1.set_xlim(0, 5)
    ax1.legend(frameon=True, fontsize=8)

    # twin 右軸：dissipation ε = 2νZ（純刻度換算）
    axr = ax1.twinx()
    lo, hi = ax1.get_ylim()
    axr.set_ylim(2 * NU * lo, 2 * NU * hi)
    axr.set_ylabel(r"dissipation $\varepsilon=2\nu\mathcal{Z}$ [m$^2$/s$^3$]")

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
