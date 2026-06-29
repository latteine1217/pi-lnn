#!/usr/bin/env python3
"""plot_les_T50_vorticity_with_sensors.py — LES T=50 渦度場與 K=100 sensor 位置疊圖。

What:
    讀 LES T=50 standalone 模擬資料的 t=50 渦度場，疊上現行主用 sensor 配置
    (sensors_qrpivot_K100_N256_t0-5_si100_les_n256_T50standalone.json) 的 100 個位置。
    輸出單張 thesis-quality 圖。

Why:
    EXP-245 (current baseline) 用的 sensor 是從 LES T=50 跑出來的，需要在 thesis 中
    展示「LES 大尺度估計 → QR-pivot 選點 → DNS 量測位置」的整體 sensor 佈點脈絡。
    既有 docs/assets/les_n256_T50_vs_dns_*.png 只比較 LES vs DNS，未顯示 sensor 點。

Usage:
    uv run python scripts/plot_les_T50_vorticity_with_sensors.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── 路徑 ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
LES_NPY = ROOT / "data/les/kolmogorov_les_Re10000_N256_T50_standalone.npy"
SENSOR_JSON = (
    ROOT
    / "data/kolmogorov_sensors/re10000"
    / "sensors_qrpivot_K100_N256_t0-5_si100_les_n256_T50standalone.json"
)
OUT_DOCS = ROOT / "docs/assets/les_T50_vorticity_with_sensors.png"
OUT_THESIS_PNG = ROOT / "thesis/figures/architecture/les_T50_vorticity_with_sensors.png"
OUT_THESIS_PDF = ROOT / "thesis/figures/architecture/les_T50_vorticity_with_sensors.pdf"

# ── 載入資料 ─────────────────────────────────────────────────────────────────

print(f"[load] LES NPY: {LES_NPY}", flush=True)
data = np.load(LES_NPY, allow_pickle=True).item()
x = data["x"]                       # (256,) ∈ [0, 1)
y = data["y"]                       # (256,) ∈ [0, 1)
time = data["time"]                 # (2501,) ∈ [0, 50]
omega = data["omega"][-1]           # (256, 256), axis_0=x, axis_1=y  (依 KNOWN_PITFALLS sensor axis convention)
t_final = float(time[-1])

print(f"[load] omega shape={omega.shape}, t_final={t_final:.4f}", flush=True)
print(f"[load] omega range=[{omega.min():.3f}, {omega.max():.3f}]", flush=True)

print(f"[load] sensor JSON: {SENSOR_JSON}", flush=True)
with open(SENSOR_JSON) as f:
    sjson = json.load(f)
coords = np.asarray(sjson["selected_coordinates"])  # (K, 2) = (x, y)
K = len(coords)
print(f"[load] K={K} sensors, x∈[{coords[:,0].min():.3f},{coords[:,0].max():.3f}], "
      f"y∈[{coords[:,1].min():.3f},{coords[:,1].max():.3f}]", flush=True)

# ── 繪圖 ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

fig, ax = plt.subplots(figsize=(6.0, 5.6), constrained_layout=True)

# pcolormesh: x 為水平軸、y 為垂直軸；omega.T 是因為 omega[x_idx, y_idx] → 顯示時 row=y, col=x
vmax = float(np.abs(omega).max())
mesh = ax.pcolormesh(
    x, y, omega.T,
    cmap="RdBu_r",
    vmin=-vmax, vmax=vmax,
    shading="auto",
    rasterized=True,
)

# Sensor 位置：白色填充 + 黑邊，小圓點，避免方塊 marker
ax.scatter(
    coords[:, 0], coords[:, 1],
    s=18, c="white", edgecolors="black", linewidths=0.6,
    marker="o", zorder=3,
    label=f"K={K} sensors (QR-pivot on LES)",
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_title(f"LES vorticity $\\omega$ at $t={t_final:.0f}$ with K={K} sensor placement")

# Colorbar
cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
cbar.set_label(r"$\omega$")
cbar.outline.set_linewidth(0.6)

# 圖例（白底黑框）
leg = ax.legend(loc="upper right", framealpha=0.85, fontsize=9)
leg.get_frame().set_linewidth(0.6)

# 去掉冗餘的 spine
for spine in ax.spines.values():
    spine.set_linewidth(0.8)

# ── 輸出 ─────────────────────────────────────────────────────────────────────

OUT_DOCS.parent.mkdir(parents=True, exist_ok=True)
OUT_THESIS_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DOCS, dpi=300, bbox_inches="tight")
fig.savefig(OUT_THESIS_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_THESIS_PDF, bbox_inches="tight")  # vector, for thesis
print(f"[save] {OUT_DOCS.relative_to(ROOT)}", flush=True)
print(f"[save] {OUT_THESIS_PNG.relative_to(ROOT)}", flush=True)
print(f"[save] {OUT_THESIS_PDF.relative_to(ROOT)}", flush=True)
