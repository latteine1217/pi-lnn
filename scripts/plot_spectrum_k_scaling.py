"""spectrum_K{100,200,400}_nyquist.pdf — per-K radial energy spectrum at t=5.

What: 對 K=100/200/400 三個 sensor budget，各畫一張 DNS vs PI-CON 能譜（t=5），
      標 k^-3 enstrophy-cascade 參考、forcing k_f、sensor Nyquist √(K/π)。
Why : K-scaling 論述（§4.4）需展示重建頻寬隨 sensor 數右移。統一色（DNS 黑 solid、
      PI-CON 藍 dashed、Nyquist 綠 dashed）與其他 result 圖一致。
資料: 各 K 的 eval fields.npz（含 u_pred/v_pred/u_ref/v_ref 全場），免 checkpoint 重畫。
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

OUTDIR = ROOT / "thesis/figures/results"
K_F = 2.0
NYQ_C = "#009E73"   # sensor Nyquist (green, matches thesis-wide LES/green semantics)
REF_C = "0.5"       # k^-3 reference (grey dotted)

# K -> fields.npz
FIELDS = {
    100: ROOT / "artifacts/eval_245_seed42_fields/fields.npz",
    200: ROOT / "artifacts/eval_K200_local/fields.npz",
    400: ROOT / "artifacts/eval_K400_local/fields.npz",
}


def spectrum_at_t5(npz_path: Path):
    d = np.load(npz_path)
    t = np.asarray(d["time"], dtype=np.float64)
    it = int(np.argmin(np.abs(t - 5.0)))
    dx = 1.0 / d["u_ref"].shape[-1]
    k_d, e_d = energy_spectrum_1d(d["u_ref"][it], d["v_ref"][it], dx)
    k_p, e_p = energy_spectrum_1d(d["u_pred"][it], d["v_pred"][it], dx)
    return k_d, e_d, k_p, e_p


def main() -> None:
    apply_journal_rcparams()
    for K, path in FIELDS.items():
        if not path.exists():
            print(f"[spectrum_K] SKIP K={K}: missing {path}", file=sys.stderr)
            continue
        k_d, e_d, k_p, e_p = spectrum_at_t5(path)
        k_nyq = np.sqrt(K / np.pi)

        fig, ax = plt.subplots(figsize=(3.4, 3.0), constrained_layout=True)
        md, mp = e_d > 0, e_p > 0
        ax.loglog(k_d[md], e_d[md], color=DNS, linestyle="-", linewidth=1.3, label="DNS")
        ax.loglog(k_p[mp], e_p[mp], color=PICON, linestyle="--", linewidth=1.3, label="PI-CON")

        # k^-3 enstrophy-cascade reference (anchored to DNS near k=3)
        anchor = np.interp(3.0, k_d[md], e_d[md])
        kk = k_d[(k_d >= K_F) & (k_d <= k_d.max())]
        ax.loglog(kk, anchor * (kk / 3.0) ** (-3.0), color=REF_C, linestyle=":",
                  linewidth=1.0, label=r"$k^{-3}$")

        ax.axvline(K_F, color="#000000", linestyle="-.", linewidth=0.7)
        ax.axvline(k_nyq, color=NYQ_C, linestyle="--", linewidth=1.1)
        ax.set_xlabel(r"wavenumber $k$ (1/m)")
        ax.set_ylabel(r"$E(k)$ (m$^3$/s$^2$)")
        ax.set_title(rf"$K={K}$, $k_{{\max}}^{{\rm sensor}}\approx {k_nyq:.2f}$", fontsize=9)
        ax.legend(loc="lower left", fontsize=6.5)
        out = OUTDIR / f"spectrum_K{K}_nyquist"
        for ext in ("pdf", "png"):
            fig.savefig(out.with_suffix(f".{ext}"))
        plt.close(fig)
        print(f"[spectrum_K] wrote {out}.pdf  (k_nyq={k_nyq:.2f})")


if __name__ == "__main__":
    main()
