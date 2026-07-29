"""Per-K radial energy spectrum at t=5 — thesis panels or slide triptych.

What: 對 K=100/200/400 三個 sensor budget，畫 DNS vs PI-CON 能譜（t=5），標
      k^-3 enstrophy-cascade 參考、forcing k_f、sensor Nyquist scale √(K/π)。
Why : K-scaling 論述（§4.4）需展示重建頻寬隨 sensor 數右移。

兩種版面：
    --layout paper  每個 K 一張獨立圖 → thesis/figures/results/spectrum_K*_nyquist
    --layout slide  三連共用 y 軸、單一圖例 → thesis/slide/public/images/…triptych.png
                    論文用 subfigure 版本，三連版僅供投影片：三張並排到投影片寬度
                    時每張只剩約 180 px，共用軸與單一圖例把重複裝飾拿掉，同樣寬度
                    下資料區約增為三倍，投影時才看得到 cutoff 右移。

繪圖本身在 pi_con.figures.spectrum；此處只負責讀 npz 與輸出。
資料: 各 K 的 eval fields.npz（含 u_pred/v_pred/u_ref/v_ref 全場），免 checkpoint 重畫。

Usage:
    uv run python scripts/plot_spectrum_k_scaling.py --layout paper
    uv run python scripts/plot_spectrum_k_scaling.py --layout slide
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_deeponet_cfc import energy_spectrum_1d  # noqa: E402
from pi_con.figures.spectrum import (  # noqa: E402
    draw_k_scaling_single,
    draw_k_scaling_triptych,
)
from pi_con.plot_style import apply_journal_rcparams, save_figure  # noqa: E402

# K -> fields.npz
FIELDS = {
    100: ROOT / "artifacts/eval_245_seed42_fields/fields.npz",
    200: ROOT / "artifacts/eval_K200_local/fields.npz",
    400: ROOT / "artifacts/eval_K400_local/fields.npz",
}

# KE MAPE 標進 slide panel 標題，投影片才不必再擺一張數字卡。
# 出處：thesis/contents/chapter04.tex tab:k_scaling_nyquist (chapter04.tex:285)。
# K=100 為 seed-42 單跑（n=5 平均為 5.71 %）。
KE_MAPE = {100: 5.90, 200: 2.47, 400: 1.76}


def spectrum_at_t5(npz_path: Path):
    """Return (k_dns, E_dns, k_pred, E_pred) at the frame nearest t=5."""
    d = np.load(npz_path)
    t = np.asarray(d["time"], dtype=np.float64)
    it = int(np.argmin(np.abs(t - 5.0)))
    dx = 1.0 / d["u_ref"].shape[-1]
    k_d, e_d = energy_spectrum_1d(d["u_ref"][it], d["v_ref"][it], dx)
    k_p, e_p = energy_spectrum_1d(d["u_pred"][it], d["v_pred"][it], dx)
    return k_d, e_d, k_p, e_p


def render_paper() -> None:
    """One standalone figure per sensor budget; missing runs are skipped."""
    outdir = ROOT / "thesis/figures/results"
    for sensor_count, path in FIELDS.items():
        if not path.exists():
            print(f"[spectrum_K] SKIP K={sensor_count}: missing {path}", file=sys.stderr)
            continue
        fig = draw_k_scaling_single(*spectrum_at_t5(path), sensor_count)
        written = save_figure(fig, str(outdir / f"spectrum_K{sensor_count}_nyquist"))
        plt.close(fig)
        print(f"[spectrum_K] wrote {written[0]}")


def render_slide() -> None:
    """Single triptych; every run must be present so panels stay comparable."""
    for sensor_count, path in FIELDS.items():
        if not path.exists():
            raise FileNotFoundError(f"K={sensor_count}: {path}")
    panels = {k: spectrum_at_t5(p) for k, p in FIELDS.items()}
    fig = draw_k_scaling_triptych(panels, KE_MAPE)
    out = ROOT / "thesis/slide/public/images/spectrum_k_scaling_triptych.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[triptych] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layout", choices=("paper", "slide"), default="paper",
                    help="paper: one figure per K (thesis); slide: shared-y triptych")
    args = ap.parse_args()

    apply_journal_rcparams()
    (render_paper if args.layout == "paper" else render_slide)()


if __name__ == "__main__":
    main()
