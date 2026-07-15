"""spectrum_k_scaling_triptych.png — 三個 sensor budget 的能譜並排（投影片用）。

What: K=100/200/400 的 DNS vs PI-CON 能譜（t=5）畫成共用 y 軸的三連圖，單一圖例，
      每格標各自的 sensor Nyquist √(K/π)。
Why : 論文 §4.4 的 per-K 單張圖（plot_spectrum_k_scaling.py）各自帶軸標與圖例，
      三張並排到投影片寬度時每張只剩約 180 px，綠色 Nyquist 線不可辨。共用軸 + 單一
      圖例把重複裝飾拿掉，同樣寬度下資料區約增為三倍，投影時才看得到 cutoff 右移。
      論文用原本的 subfigure 版本，此檔僅供 slide。
資料: 與 plot_spectrum_k_scaling.py 相同的 eval fields.npz，免 checkpoint 重畫。
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

OUT = ROOT / "thesis/slide/public/images/spectrum_k_scaling_triptych.png"
K_F = 2.0
NYQ_C = "#009E73"
REF_C = "0.5"

FIELDS = {
    100: ROOT / "artifacts/eval_245_seed42_fields/fields.npz",
    200: ROOT / "artifacts/eval_K200_local/fields.npz",
    400: ROOT / "artifacts/eval_K400_local/fields.npz",
}

# KE MAPE 標進 panel 標題，投影片才不必再擺一張數字卡。
# 出處：thesis/contents/chapter04.tex tab:k_scaling_nyquist (chapter04.tex:285)。
# K=100 為 seed-42 單跑（n=5 平均為 5.71 %）。
KE_MAPE = {100: 5.90, 200: 2.47, 400: 1.76}


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
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), sharey=True,
                             constrained_layout=True)

    for ax, (K, path) in zip(axes, FIELDS.items()):
        if not path.exists():
            raise FileNotFoundError(path)
        k_d, e_d, k_p, e_p = spectrum_at_t5(path)
        k_nyq = np.sqrt(K / np.pi)
        md, mp = e_d > 0, e_p > 0

        ax.loglog(k_d[md], e_d[md], color=DNS, linestyle="-", linewidth=1.6, label="DNS")
        ax.loglog(k_p[mp], e_p[mp], color=PICON, linestyle="--", linewidth=1.8, label="PI-CON")

        anchor = np.interp(3.0, k_d[md], e_d[md])
        kk = k_d[(k_d >= K_F) & (k_d <= k_d.max())]
        ax.loglog(kk, anchor * (kk / 3.0) ** (-3.0), color=REF_C, linestyle=":",
                  linewidth=1.0, label=r"$k^{-3}$")

        ax.axvline(K_F, color="#000000", linestyle="-.", linewidth=0.7)
        ax.axvline(k_nyq, color=NYQ_C, linestyle="--", linewidth=2.0,
                   label=r"$k_{\max}^{\rm sensor}$")
        # 在 Nyquist 線旁直接標值，投影時不必回頭看圖例
        ax.annotate(rf"$k_{{\max}}\!\approx\!{k_nyq:.2f}$", xy=(k_nyq, 4e-2),
                    xytext=(3, 0), textcoords="offset points",
                    color=NYQ_C, fontsize=10, fontweight="bold", ha="left")
        # 非 usetex（見 pi_con.plot_style），故 % 直接寫，不可用 LaTeX 的 \%。
        ax.set_title(rf"$K={K}$   ·   KE {KE_MAPE[K]:.2f}%",
                     fontsize=12.5, fontweight="bold", pad=4)
        ax.set_xlabel(r"wavenumber $k$ (1/m)", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.set_ylim(1e-11, 5e-1)

    axes[0].set_ylabel(r"$E(k)$ (m$^3$/s$^2$)", fontsize=10)
    axes[0].legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f"[triptych] wrote {OUT}")


if __name__ == "__main__":
    main()
