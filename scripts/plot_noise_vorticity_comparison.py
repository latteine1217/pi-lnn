"""Vorticity reconstruction under increasing sensor noise.

What
====
兩列 × 六欄：上列是 t=5 的重建渦度場（DNS 參考 + 0/1/3/5/10 % 噪音），
下列是各自對 DNS 的誤差場。回答「噪音把重建弄成什麼樣子」這個視覺問題，
補上噪音那頁只有數字、看不出偏差長相的缺口。

Why this layout
===============
上列共用同一組色階（取自 DNS 的 99 百分位），所以欄與欄之間可以直接比對；
下列共用另一組對稱色階。兩列各自共用色階是這張圖唯一能成立的前提——
若每格自動縮放，視覺差異會是色階的假象而非場的差異。

資料一致性
==========
所有欄位同為 **seed 42**（EXP-245 的 seed 42 與 EXP-290 的 `_a` 同 seed，
已對照 configs/stable/exp_{245,290_noise01_a}.toml 的 seed 欄）。混 seed 會把
訓練隨機性混進噪音效應，故此處不接受跨 seed 拼圖。

用法
====
    uv run python scripts/plot_noise_vorticity_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pi_con.plot_style import apply_journal_rcparams  # noqa: E402

CLEAN_NPZ = ROOT / "artifacts/eval_245_seed42_fields/fields.npz"
NOISE_DIR = ROOT / "artifacts/eval_noise_fields"
NOISE_LEVELS = [("01", "1 %"), ("03", "3 %"), ("05", "5 %"), ("10", "10 %")]
OUT_STEM = ROOT / "thesis/figures/results/noise_vorticity_comparison"


def load_omega(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (omega_pred, omega_ref) at the final frame."""
    if not npz_path.exists():
        raise SystemExit(f"[abort] missing {npz_path}")
    d = np.load(npz_path)
    return d["omega_pred"][-1].astype(np.float64), d["omega_ref"][-1].astype(np.float64)


def main() -> None:
    clean_pred, ref = load_omega(CLEAN_NPZ)
    print(f"[data] clean baseline: {CLEAN_NPZ.relative_to(ROOT)}")

    preds = [("0 %", clean_pred)]
    for tag, label in NOISE_LEVELS:
        p, r = load_omega(NOISE_DIR / f"noise{tag}" / "fields.npz")
        # 參考場必須逐格相同，否則不是同一條 DNS 軌跡
        if not np.allclose(r, ref, rtol=0, atol=1e-6):
            raise SystemExit(f"[abort] DNS reference of noise{tag} differs from the clean run")
        preds.append((label, p))
        print(f"[data] noise {label}: {(NOISE_DIR / f'noise{tag}' / 'fields.npz').relative_to(ROOT)}")

    # 共用色階：上列取 DNS 的 99 百分位，下列取所有誤差場的 99 百分位
    vlim = float(np.percentile(np.abs(ref), 99))
    elim = float(np.percentile(np.abs(np.stack([p - ref for _, p in preds])), 99))

    apply_journal_rcparams()
    ncol = len(preds) + 1
    fig = plt.figure(figsize=(2.05 * ncol, 4.5))
    gs = GridSpec(2, ncol, figure=fig, wspace=0.06, hspace=0.14,
                  left=0.045, right=0.90, top=0.90, bottom=0.04)
    imshow_kw = dict(origin="lower", extent=(0, 1, 0, 1), aspect="equal", cmap="RdBu_r")

    ax = fig.add_subplot(gs[0, 0])
    im_f = ax.imshow(ref, vmin=-vlim, vmax=vlim, **imshow_kw)
    ax.set_title("DNS", fontsize=10, pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.1)

    for j, (label, pred) in enumerate(preds, start=1):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(pred, vmin=-vlim, vmax=vlim, **imshow_kw)
        ax.set_title(f"noise {label}", fontsize=10, pad=4)
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(gs[1, j])
        im_e = ax.imshow(pred - ref, vmin=-elim, vmax=elim, **imshow_kw)
        ax.set_xticks([]); ax.set_yticks([])

    # 下列第一格不放圖：那裡沒有「DNS 減 DNS」這種東西，改標示該列是什麼
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    ax.text(0.5, 0.5, "error\nvs DNS", ha="center", va="center",
            fontsize=10, color="0.35", transform=ax.transAxes)

    cax_f = fig.add_axes([0.915, 0.525, 0.013, 0.345])
    fig.colorbar(im_f, cax=cax_f).set_label(r"$\omega$  (1/s)", fontsize=9)
    cax_e = fig.add_axes([0.915, 0.075, 0.013, 0.345])
    fig.colorbar(im_e, cax=cax_e).set_label(r"$\omega_{\rm pred}-\omega_{\rm DNS}$  (1/s)", fontsize=9)

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_STEM}.{ext}")
    plt.close(fig)
    print(f"[saved] {OUT_STEM}.pdf / .png")

    print(f"  shared scales: field ±{vlim:.2f}, error ±{elim:.2f} (99th percentile)")
    for label, pred in preds:
        rel = 100 * np.linalg.norm(pred - ref) / np.linalg.norm(ref)
        print(f"  noise {label:>4}: omega rel-L2 = {rel:5.2f} %")


if __name__ == "__main__":
    main()
