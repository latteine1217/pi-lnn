#!/usr/bin/env python3
"""為投影片「Why the field has to be reconstructed」產生左右兩張對照圖。

What
----
`why_probes_only.png` —— 工程現場真正拿得到的東西:100 個探針位置上的 ω 讀值,
                          其餘位置一片空白。
`why_full_field.png`  —— 判斷流場結構 / 梯度 / 受力所需要的連續場。

Why
---
先前版本把左圖畫成「白底橘點」、右圖畫成「彩色流場」,兩張圖沒有共同的
色標,讀者無法把左圖的點理解成「右圖在那 100 個位置的取樣」,只會看成
兩張不相干的圖。此處讓兩張圖共用同一組 colormap 與 clim、同一個定義域,
左圖的每個點就是右圖對應位置的實際值 —— 稀疏性才會被看出來。

資料來源皆為實測,非示意:
- 場:DNS Re=10^4, N=256, t = 5.0
- 位置:EXP-245 active baseline 使用的 LES_T50 QR-pivot 佈點 (K=100)

Axis convention 依 CLAUDE.md:omega[t, x_idx, y_idx];繪圖時轉置使 x 為水平軸。
"""

from pathlib import Path

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DNS = ROOT / "data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy"
SENSORS = (
    ROOT
    / "data/kolmogorov_sensors/re10000"
    / "sensors_qrpivot_K100_N256_t0-5_si100_les_n256_T50standalone.json"
)
OUTDIR = ROOT / "thesis/slide/public/images"

T_INDEX = -1  # t = 5.0
CMAP = "RdBu_r"
CLIP_PCT = 99.0  # 對稱 clim 取百分位,避免少數極值把整張圖洗白


def main() -> None:
    data = np.load(DNS, allow_pickle=True).item()
    omega = np.asarray(data["omega"][T_INDEX])  # [x, y]
    n = omega.shape[0]

    vmax = float(np.percentile(np.abs(omega), CLIP_PCT))
    vmin = -vmax

    coords = np.asarray(json.load(SENSORS.open())["selected_coordinates"])  # (K, 2) = (x, y)
    xi = np.clip((coords[:, 0] * n).round().astype(int), 0, n - 1)
    yi = np.clip((coords[:, 1] * n).round().astype(int), 0, n - 1)
    vals = omega[xi, yi]

    common = dict(figsize=(3.2, 3.2), dpi=300)

    # --- 右圖:連續場 ---------------------------------------------------
    fig, ax = plt.subplots(**common)
    ax.imshow(
        omega.T,
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    _finish(ax)
    fig.savefig(OUTDIR / "why_full_field.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # --- 左圖:只有探針讀值 ---------------------------------------------
    fig, ax = plt.subplots(**common)
    ax.set_facecolor("#FFFFFF")
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=vals,
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        s=54,
        linewidths=0.6,
        edgecolors="#4B5563",
        zorder=3,
    )
    _finish(ax)
    fig.savefig(OUTDIR / "why_probes_only.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"[ok] K = {len(coords)} probes, omega clim = +/- {vmax:.2f} 1/s, t = {data['time'][T_INDEX]:.2f} s")
    print(f"[ok] wrote {OUTDIR/'why_probes_only.png'}")
    print(f"[ok] wrote {OUTDIR/'why_full_field.png'}")


def _finish(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#D8D2E0")
        s.set_linewidth(0.9)


if __name__ == "__main__":
    main()
