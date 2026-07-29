#!/usr/bin/env python3
"""Temporal sampling-density ablation (軸 A) — reconstruction error vs snapshot count.

Reads the 5 offline-eval summaries under data/eval_tdensity_summaries/
(EXP-303~306 + EXP-245 baseline) and renders the ablation figure.

Caveats baked into the thesis caption:
    - Single seed (=42) → positive-finding sweep；201/101/51 之間的小差落在 seed
      雜訊帶內，不宣稱可分辨。圖只主張「平台 → 崖」的大尺度趨勢。
    - Scheme A：每個 run 在自己的時間格上評估（common endpoint t=5）。

繪圖本身在 pi_con.figures.temporal_density；此處只負責讀資料與輸出。

Usage:
    uv run python scripts/plot_temporal_density_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pi_con.figures.temporal_density import draw_temporal_density_ablation  # noqa: E402
from pi_con.plot_style import apply_journal_rcparams, save_figure  # noqa: E402

SUMDIR = ROOT / "data" / "eval_tdensity_summaries"
T_TOTAL = 5.0

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
    apply_journal_rcparams()
    fig = draw_temporal_density_ablation(frames, ke, low, total_time=T_TOTAL)
    out = save_figure(fig, str(ROOT / "thesis/figures/results/temporal_density_ablation"))
    print("wrote:", *[str(p) for p in out])


if __name__ == "__main__":
    main()
