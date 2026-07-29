#!/usr/bin/env python3
"""Temporal sampling-density ablation (軸 A) — reconstruction error vs snapshot count.

What this answers
-----------------
主線用 201 個時間 snapshot（Δt=0.025, T=5）——此 201 是 DNS 儲存 cadence
（save_interval=100, dt=2.5e-4）的副產品，非調校結果。本圖以固定窗長 [0,5]、
只改 sensor 時間監督密度的 sweep（EXP-303~306 + EXP-245 baseline）回答：
重建品質對時間取樣密度是否敏感、201 是否落在過取樣區。

Reads the 5 offline-eval summaries under data/eval_tdensity_summaries/ and plots
KE rel-err 與 low-frequency band rel-err vs snapshot count（log-x）。

Caveats baked into the caption (see thesis):
    - Single seed (=42) → 這是 positive-finding sweep，201/101/51 之間的小差
      落在 seed 雜訊帶內，不宣稱可分辨；圖只主張「平台 → 崖」的大尺度趨勢。
    - Scheme A：每個 run 在自己的時間格上評估（common endpoint t=5）。

Usage:
    uv run python scripts/plot_temporal_density_ablation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from journal_style import setup_style, figwidth, STYLE_CYCLE, save_figure

REPO = Path(__file__).resolve().parent.parent
SUMDIR = REPO / "data" / "eval_tdensity_summaries"
T_TOTAL = 5.0
T_EDDY = 1.99  # 協議 T/t_eddy = 2.51 @ T=5 → t_eddy ≈ 1.99（用於 top-axis 註記）

# (exp_id, frames, summary stem)
RUNS = [
    (245, 201, "exp245-b3-les-T50-20k"),
    (303, 101, "exp303-b3-tdensity-101f-20k"),
    (304,  51, "exp304-b3-tdensity-51f-20k"),
    (305,  26, "exp305-b3-tdensity-26f-20k"),
    (306,  11, "exp306-b3-tdensity-11f-20k"),
]


def load() -> tuple[list[int], list[float], list[float]]:
    frames, ke, low = [], [], []
    for _eid, nfr, stem in RUNS:
        s = json.loads((SUMDIR / f"{stem}.json").read_text())
        frames.append(nfr)
        ke.append(s["ke_rel_err_mean"] * 100.0)
        low.append(s["band_energy_rel_err_mean"]["low"] * 100.0)
    return frames, ke, low


def main() -> None:
    frames, ke, low = load()

    setup_style("jfm")
    fig, ax = plt.subplots(figsize=(figwidth("jfm", "single"), 3.0))

    # 底色標示 baseline (201f) 所在的取樣區間 Δt <= 0.1 s（無解讀詞，語意留給 caption）
    ax.axvspan(48, 215, color="0.85", alpha=0.35, zorder=0)

    # 論文低頻準則 10%（已定義之門檻，屬事實非解讀）；標籤放右端線上方避開曲線
    ax.axhline(10.0, color="0.5", ls=":", lw=0.9, zorder=1)
    ax.text(205, 10.5, r"$10\%$ low-band criterion", fontsize=7, color="0.4",
            ha="right", va="bottom")

    c0, m0, l0 = STYLE_CYCLE[0]
    c1, m1, l1 = STYLE_CYCLE[1]
    ax.plot(frames, ke, color=c0, marker=m0, linestyle=l0,
            label=r"KE rel-err")
    ax.plot(frames, low, color=c1, marker=m1, linestyle=l1,
            label=r"low-band rel-err")

    ax.set_xscale("log")
    ax.set_xticks(frames)
    ax.set_xticklabels([str(f) for f in frames])
    ax.set_xlim(9.5, 230)
    ax.set_ylim(0, 25)
    ax.set_xlabel(r"Number of temporal snapshots $N_t$ over $T=5$ s (–)")
    ax.set_ylabel(r"Relative error (\%)")
    ax.legend(loc="upper right", fontsize=8)

    # 上方第二軸：對應 Δt = T/(N_t-1)
    def n_to_dt(n):
        return T_TOTAL / (n - 1)
    secax = ax.secondary_xaxis("top")
    secax.set_xscale("log")
    secax.set_xticks(frames)
    secax.set_xticklabels([f"{n_to_dt(f):.3g}" for f in frames])
    secax.set_xlabel(r"Sensor sampling interval $\Delta t$ (s)")

    out = save_figure(fig, "thesis/figures/results/temporal_density_ablation")
    print("wrote:", *[str(p) for p in out])


if __name__ == "__main__":
    main()
