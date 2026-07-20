"""Vorticity reconstruction as the sensor budget K grows.

What
====
兩列 × 四欄：上列是 t=5 的重建渦度場（DNS 參考 + K=100/200/400），下列是各自
對 DNS 的誤差場。回答「加 sensor 把重建變成什麼樣子」這個視覺問題，與噪音那頁
（plot_noise_vorticity_comparison.py）同一設計語言。

Why this layout
===============
上列共用一組色階（DNS 的 99 百分位），下列共用另一組對稱色階，欄與欄才可比。
每格自動縮放會讓視覺差異變成色階假象。

資料一致性
==========
三個 K 皆 single-seed（K=100 seed 42、K=200 EXP-269、K=400 EXP-270），與 §K-scaling
的三連能譜同源，故 caption 標 single-seed。三者的 DNS 參考逐格相同（同一條軌跡），
腳本以 assert 檢查。K=400 用 512 collocation（非 1024），與能譜頁揭露一致。

用法
====
    uv run python scripts/plot_kscaling_vorticity_comparison.py
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

COLUMNS = [
    ("K = 100", ROOT / "artifacts/eval_245_seed42_fields/fields.npz"),
    ("K = 200", ROOT / "artifacts/eval_K200_local/fields.npz"),
    ("K = 400", ROOT / "artifacts/eval_K400_local/fields.npz"),
]
OUT_STEM = ROOT / "thesis/figures/results/kscaling_vorticity_comparison"


def load_omega(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise SystemExit(f"[abort] missing {npz_path}")
    d = np.load(npz_path)
    return d["omega_pred"][-1].astype(np.float64), d["omega_ref"][-1].astype(np.float64)


def main() -> None:
    ref = None
    preds = []
    for label, path in COLUMNS:
        p, r = load_omega(path)
        if ref is None:
            ref = r
        elif not np.allclose(r, ref, rtol=0, atol=1e-4):
            raise SystemExit(f"[abort] DNS reference of {label} differs from the first column")
        preds.append((label, p))
        print(f"[data] {label}: {path.relative_to(ROOT)}")

    vlim = float(np.percentile(np.abs(ref), 99))
    elim = float(np.percentile(np.abs(np.stack([p - ref for _, p in preds])), 99))

    apply_journal_rcparams()
    ncol = len(preds) + 1
    fig = plt.figure(figsize=(2.05 * ncol, 4.5))
    gs = GridSpec(2, ncol, figure=fig, wspace=0.06, hspace=0.14,
                  left=0.055, right=0.885, top=0.90, bottom=0.04)
    imshow_kw = dict(origin="lower", extent=(0, 1, 0, 1), aspect="equal", cmap="RdBu_r")

    ax = fig.add_subplot(gs[0, 0])
    im_f = ax.imshow(ref, vmin=-vlim, vmax=vlim, **imshow_kw)
    ax.set_title("DNS", fontsize=10, pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    for j, (label, pred) in enumerate(preds, start=1):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(pred, vmin=-vlim, vmax=vlim, **imshow_kw)
        ax.set_title(label, fontsize=10, pad=4)
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(gs[1, j])
        im_e = ax.imshow(pred - ref, vmin=-elim, vmax=elim, **imshow_kw)
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    ax.text(0.5, 0.5, "error\nvs DNS", ha="center", va="center",
            fontsize=10, color="0.35", transform=ax.transAxes)

    cax_f = fig.add_axes([0.90, 0.525, 0.014, 0.345])
    fig.colorbar(im_f, cax=cax_f).set_label(r"$\omega$  (1/s)", fontsize=9)
    cax_e = fig.add_axes([0.90, 0.075, 0.014, 0.345])
    fig.colorbar(im_e, cax=cax_e).set_label(r"$\omega_{\rm pred}-\omega_{\rm DNS}$  (1/s)", fontsize=9)

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_STEM}.{ext}")
    plt.close(fig)
    print(f"[saved] {OUT_STEM}.pdf / .png")
    print(f"  shared scales: field +/-{vlim:.2f}, error +/-{elim:.2f} (99th percentile)")
    for label, pred in preds:
        rel = 100 * np.linalg.norm(pred - ref) / np.linalg.norm(ref)
        print(f"  {label}: omega rel-L2 = {rel:5.2f} %")


if __name__ == "__main__":
    main()
