# scripts/plot_cylinder_sensor_K100_K200.py
# 比較 cylinder K=100 vs K=200 QR-pivot sensor 分佈（疊在 vorticity 背景 + body 輪廓）
#
# What: 讀 arrow 取一個 snapshot 算 vorticity → 疊 body + sensor 散點 → 上下對照圖。
# Why : 升 K=200 前先目視檢查 sensor 是否仍集中 wake、是否誤入 body、覆蓋是否合理。

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import matplotlib.pyplot as plt

ARROW = ("/Users/latteine/Documents/coding/RealPDEBench/data/realpdebench/"
         "cylinder/hf_dataset/numerical/data-00000-of-00092.arrow")
T_SNAP = 600          # 已脫離 spin-up 的 snapshot
LX, LY = 0.3223, 0.1721


def load_field(arrow_path):
    # 與 generate_sensors_qrpivot_cylinder.load_shard 同：IPC stream，single-row table
    with open(arrow_path, "rb") as f:
        batch = pa.ipc.open_stream(f).read_next_batch()
    row = {name: batch.column(name)[0].as_py() for name in batch.schema.names}
    T, H, W = row["shape_t"], row["shape_h"], row["shape_w"]
    u = np.frombuffer(row["u"], dtype=np.float32).reshape(T, H, W)[T_SNAP]
    v = np.frombuffer(row["v"], dtype=np.float32).reshape(T, H, W)[T_SNAP]
    return u, v, H, W


def main():
    u, v, H, W = load_field(ARROW)
    body = (np.abs(u) < 1e-9) & (np.abs(v) < 1e-9)
    dx, dy = LX / W, LY / H
    omega = np.gradient(v, dx, axis=1) - np.gradient(u, dy, axis=0)
    omega_m = np.ma.array(omega, mask=body)
    extent = [0, LX, 0, LY]
    vmax = np.percentile(np.abs(omega_m.compressed()), 99)
    xx = np.linspace(0, LX, W)
    yy = np.linspace(0, LY, H)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.0), constrained_layout=True)
    for ax, K in zip(axes, (100, 200)):
        d = json.load(open(f"data/cylinder_sensors/sensors_qrpivot_K{K}_cylinder_Re10031.json"))
        xy = np.array(d["selected_coordinates"])
        ax.imshow(omega_m, origin="lower", extent=extent, cmap="RdBu_r",
                  vmin=-vmax, vmax=vmax, aspect="auto")
        ax.contour(xx, yy, body.astype(float), levels=[0.5], colors="k", linewidths=1.0)
        ax.scatter(xy[:, 0], xy[:, 1], s=10, c="lime", edgecolors="k",
                   linewidths=0.3, zorder=5)
        ax.set_title(f"QR-pivot K={K}  (Re=10031, {len(xy)} sensors)", fontsize=11)
        ax.set_ylabel("y")
        ax.set_xlim(0, LX)
        ax.set_ylim(0, LY)
    axes[-1].set_xlabel("x")

    out = Path("docs/figures/cylinder_sensor_K100_vs_K200.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
