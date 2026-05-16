#!/usr/bin/env python3
"""plot_kolmogorov_sensor_distribution.py — 期刊風格 Kolmogorov K=100 sensor 分布。

What:
    從 Kolmogorov sensor JSON（generate_*_qrpivot_*.py 產出）+ DNS .npy
    讀 sensor (x, y) + ω field at t_target，畫期刊風格 placement plot。

Why:
    Slide / 論文需要展示 K = 100 sensors 在 2-D Kolmogorov 上的實際分布
    （與 cylinder JHTDB 範例圖不同 — Kolmogorov 是 periodic domain，no body）。

Usage:
    uv run python scripts/plot_kolmogorov_sensor_distribution.py \\
        --json data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100.json \\
        --dns  data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy \\
        --t-target 5.0 \\
        --out  thesis/slide/public/images/sensor_distribution_kolmogorov_K100.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── 期刊樣式（與 evaluate_deeponet_cfc / plot_sensor_distribution 對齊）────────
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "axes.linewidth": 0.7,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.grid": False,
    }
)


def main() -> None:
    p = argparse.ArgumentParser(description="Kolmogorov sensor placement plot")
    p.add_argument("--json", type=Path, required=True, help="sensor JSON path")
    p.add_argument("--dns", type=Path, required=True, help="DNS .npy (dict)")
    p.add_argument("--t-target", type=float, default=5.0, help="time slice to render")
    p.add_argument("--out", type=Path, required=True, help="output PNG path")
    p.add_argument(
        "--field-mode",
        choices=["omega", "speed"],
        default="omega",
        help="background field (default: vorticity)",
    )
    args = p.parse_args()

    meta = json.loads(args.json.read_text())
    K = int(meta["K"])
    method = meta.get("method", "qr_pivoting")
    indices = np.asarray(meta["indices"], dtype=np.int64)
    n_grid = int(meta["resolution"].split("x")[0])

    dns = np.load(args.dns, allow_pickle=True).item()
    t_arr = np.asarray(dns["time"])
    x_axis = np.asarray(dns["x"])
    y_axis = np.asarray(dns["y"])

    # sensor (x, y) — JSON 的 indices 是 row-major flat index 在 256x256 grid 上
    rows = indices // n_grid  # → y axis
    cols = indices % n_grid  # → x axis
    sensor_x = x_axis[cols]
    sensor_y = y_axis[rows]

    # 背景場
    t_idx = int(np.argmin(np.abs(t_arr - args.t_target)))
    t_actual = float(t_arr[t_idx])

    if args.field_mode == "omega":
        field = dns["omega"][t_idx]
        cb_label = r"$\omega$"
        cmap = "RdBu_r"
        v_lim = float(np.percentile(np.abs(field), 99.0))
        norm_kwargs = dict(vmin=-v_lim, vmax=v_lim)
    else:
        u = dns["u"][t_idx]
        v = dns["v"][t_idx]
        field = np.sqrt(u**2 + v**2)
        cb_label = r"$|U|$"
        cmap = "viridis"
        norm_kwargs = dict(vmin=0.0, vmax=float(field.max()))

    fig, ax = plt.subplots(figsize=(4.4, 4.0), constrained_layout=True)
    extent = (
        float(x_axis[0]),
        float(x_axis[-1]),
        float(y_axis[0]),
        float(y_axis[-1]),
    )
    im = ax.imshow(
        field,
        origin="lower",
        cmap=cmap,
        extent=extent,
        aspect="equal",
        interpolation="bilinear",
        **norm_kwargs,
    )
    ax.scatter(
        sensor_x,
        sensor_y,
        s=24,
        marker="o",
        facecolors="white",
        edgecolors="#0F2D52",
        linewidths=0.9,
        zorder=3,
        label=f"K = {K}",
    )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    method_pretty = {"qr_pivoting": "QR-pivot", "qr": "QR-pivot"}.get(method, method)
    ax.set_title(
        f"K = {K} sensors · {method_pretty} · {cb_label} at $t={t_actual:.2f}$",
        pad=6,
    )

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.6)
    cb.set_label(cb_label, fontsize=9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out}  (K={K}, t={t_actual:.2f}, method={method_pretty})")


if __name__ == "__main__":
    main()
