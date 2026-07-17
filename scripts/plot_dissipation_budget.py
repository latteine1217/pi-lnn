"""#5 重建場 enstrophy / dissipation budget 圖（誠實揭露 high-k 截斷後果）。

What: 用既有 eval series.npz 的 enstrophy(t) 對照重建場 vs DNS，並換算 dissipation
      ε(t)=2ν·Z(t)（ν=1/Re）。
Why : K=100 sensor Nyquist 截斷 → 重建場缺中高頻渦量 → enstrophy/dissipation 系統性低估。
      這是 sensor-information ceiling 的物理後果，誠實量化反而強化「中高頻 bounded 非架構失敗」，
      並界定 out-of-scope（TKE/dissipation budget 任務需更多 sensor）。
資料: EXP-245 multi-seed group (n=5)，見 SEED_GLOB。

Why multi-seed (2026-07-17)
===========================
原本硬編 `artifacts/eval_245_seed42_export/series.npz` 單一 seed。本圖與 band-energy 圖
同為純量時間序列（非場的視覺化），與主表、fig:main_trajectories 一樣可跨 seed 平均；
單 seed 版無法分辨「系統性 deficit」與「單次實現的巧合」。改為 n=5 mean ± 1σ。
（欄位視覺化類的圖仍維持 seed 42，那類圖無法平均。）

DNS 參考線跨 seed 位元相同（DNS 為固定 ground truth），故只畫一條，不做包絡。
"""
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pi_con.plot_style import apply_journal_rcparams, DNS as DNS_C, PICON as PRED_C  # noqa: E402

NU = 1e-4  # Re = 10000 → ν = 1/Re
# 用 glob 但**斷言數量**：2026-07-17 這批 eval 目錄曾由 `_mac` 改名為 `_final`，
# 硬編路徑會安靜地讀到被取代的舊資料（`_mac` 評自 ScheduleFree train-mode
# checkpoint，與主表差 0.05 pp）。寧可大聲失敗。
SEED_GLOB = "artifacts/exp245_seeds/eval_245_seed?_final"
N_SEEDS_EXPECTED = 5
OUTDIR = ROOT / "thesis/figures/results"
OUTDIR.mkdir(parents=True, exist_ok=True)

# DNS_C / PRED_C imported from pi_con.plot_style (Okabe--Ito semantic palette)


def load_seeds() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (time, Z_pred stacked (n_seeds, n_time), Z_dns)."""
    dirs = sorted(ROOT.glob(SEED_GLOB))
    if len(dirs) != N_SEEDS_EXPECTED:
        raise SystemExit(
            f"[abort] expected {N_SEEDS_EXPECTED} seed dirs matching {SEED_GLOB}, "
            f"found {len(dirs)}: {[d.name for d in dirs]}\n"
            f"        本圖標示為 n=5；seed 數不符時不畫，以免圖與 caption 不一致。"
        )
    series = [np.load(d / "series.npz") for d in dirs]
    t = series[0]["time"]
    Z_dns = series[0]["enstrophy_dns"]
    for d, s in zip(dirs, series):
        if not np.allclose(s["time"], t):
            raise SystemExit(f"[abort] time grid of {d.name} differs from {dirs[0].name}")
        if not np.allclose(s["enstrophy_dns"], Z_dns):
            raise SystemExit(
                f"[abort] DNS enstrophy of {d.name} differs from {dirs[0].name}; "
                f"the DNS reference must be identical across seeds."
            )
    print(f"[#5] {len(dirs)} seeds: {', '.join(d.name for d in dirs)}")
    return t, np.stack([s["enstrophy"] for s in series]), Z_dns


def main() -> None:
    t, Z_pred_stack, Z_dns = load_seeds()

    # 後 spin-up 窗 t>=1 的 time-mean deficit（與 thesis「t>=1 KE~5%」一致的穩態窗）。
    # 逐 seed 算 deficit 再取 mean±std —— 不是先平均曲線再算一個 deficit。
    m = t >= 1.0
    per_seed = np.array([(1.0 - zp[m].mean() / Z_dns[m].mean()) * 100.0
                         for zp in Z_pred_stack])
    z_deficit, z_deficit_sd = per_seed.mean(), per_seed.std(ddof=1)

    Z_mean = Z_pred_stack.mean(axis=0)
    Z_std = Z_pred_stack.std(axis=0, ddof=1)

    apply_journal_rcparams()
    # 單 panel：enstrophy Z(t)（左軸）；dissipation ε=2νZ 僅為常數倍率，
    # 用右側 twin 軸提供換算刻度，不重畫同形曲線（避免左右兩 panel 冗餘）。
    fig, ax1 = plt.subplots(figsize=(5.2, 3.0))

    ax1.plot(t, Z_dns, color=DNS_C, lw=1.4, label="DNS", zorder=5)
    for zp in Z_pred_stack:
        ax1.plot(t, zp, color=PRED_C, ls=":", lw=0.6, alpha=0.35, zorder=2)
    ax1.fill_between(t, Z_mean - Z_std, Z_mean + Z_std, color=PRED_C, alpha=0.18,
                     linewidth=0, zorder=3)
    ax1.plot(t, Z_mean, color=PRED_C, ls="--", lw=1.4,
             label=r"PI-CON (mean $\pm 1\sigma$, $n=5$)", zorder=4)
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
    print(f"[#5] enstrophy deficit (t>=1): {z_deficit:.1f} +- {z_deficit_sd:.1f} %  "
          f"per-seed {np.array2string(per_seed, precision=1)}")
    print(f"[#5] Z_dns(t>=1) mean={Z_dns[m].mean():.2f}, "
          f"Z_pred(t>=1) mean={Z_mean[m].mean():.2f} [1/s^2]")
    print(f"[#5] eps_dns(t>=1) mean={2*NU*Z_dns[m].mean():.4e}, "
          f"eps_pred mean={2*NU*Z_mean[m].mean():.4e} [m^2/s^3]")


if __name__ == "__main__":
    main()
