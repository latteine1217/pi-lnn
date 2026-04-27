#!/usr/bin/env python3
"""visualize_cylinder_data.py — Cylinder wake 資料可視化。

輸出：
  docs/assets/cylinder/
    overview_vorticity.png  — 4 個時刻的渦度場 + sensor 位置
    overview_velocity.png   — u / v 速度場
    temporal_signals.png    — KE(t)、probe v(t)、sensor 位置分布
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import pyarrow as pa
import json

ARROW_PATH = "/Users/latteine/Documents/coding/RealPDEBench/data/realpdebench/cylinder/hf_dataset/numerical/data-00000-of-00092.arrow"
SENSOR_JSON = "data/cylinder_sensors/sensors_qrpivot_K100_cylinder_Re10031.json"
OUT_DIR = Path("docs/assets/cylinder")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 資料載入 ─────────────────────────────────────────────────────────────────

def load_shard(path: str) -> dict:
    with open(path, "rb") as f:
        reader = pa.ipc.open_stream(f)
        batch = reader.read_next_batch()
    row = {n: batch.column(n)[0].as_py() for n in batch.schema.names}
    T, H, W = row["shape_t"], row["shape_h"], row["shape_w"]
    return dict(
        u=np.frombuffer(row["u"], dtype=np.float32).reshape(T, H, W),
        v=np.frombuffer(row["v"], dtype=np.float32).reshape(T, H, W),
        p=np.frombuffer(row["p"], dtype=np.float32).reshape(T, H, W),
        x=np.frombuffer(row["x"], dtype=np.float64).reshape(H, W),
        y=np.frombuffer(row["y"], dtype=np.float64).reshape(H, W),
        t=np.frombuffer(row["t"], dtype=np.float64)[:T],
        T=T, H=H, W=W,
        Re=float(row["sim_id"].replace(".h5", "")),
    )


def vorticity(u: np.ndarray, v: np.ndarray,
              x2d: np.ndarray, y2d: np.ndarray) -> np.ndarray:
    """FD 渦度 ω = ∂v/∂x - ∂u/∂y（非均勻格）"""
    x_1d = x2d[0, :]
    y_1d = y2d[:, 0]
    dvdx = np.gradient(v, x_1d, axis=1)
    dudy = np.gradient(u, y_1d, axis=0)
    return dvdx - dudy


def cylinder_patch(cx: float, cy: float, r: float) -> mpatches.Circle:
    return mpatches.Circle((cx, cy), r, color="white", zorder=5,
                            linewidth=1.2, edgecolor="#888")


# ── 繪圖工具 ─────────────────────────────────────────────────────────────────

def add_cylinder(ax, cx, cy, r):
    ax.add_patch(cylinder_patch(cx, cy, r))


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    print("Loading cylinder wake data …")
    d = load_shard(ARROW_PATH)
    u, v, p = d["u"], d["v"], d["p"]
    x, y, t = d["x"], d["y"], d["t"]
    T, H, W, Re = d["T"], d["H"], d["W"], d["Re"]

    # 偵測 cylinder body
    body = (np.abs(u[200]) < 1e-4) & (np.abs(v[200]) < 1e-4)
    cx = float(x[body].mean())
    cy = float(y[body].mean())
    r  = float(max(x[body].max() - x[body].min(),
                   y[body].max() - y[body].min()) / 2)
    print(f"  Re={Re:.0f}  grid={H}×{W}  T={T}")
    print(f"  Cylinder: center=({cx:.4f},{cy:.4f}), r≈{r:.4f}")

    # Sensor 位置
    with open(SENSOR_JSON) as f:
        meta = json.load(f)
    sx = np.array([c[0] for c in meta["selected_coordinates"]])
    sy = np.array([c[1] for c in meta["selected_coordinates"]])

    # 選取 4 個代表時刻
    t_targets = [1.0, 5.0, 10.0, 18.0]
    t_idx = [np.argmin(np.abs(t - tt)) for tt in t_targets]

    # ── 圖 1：渦度場 + sensor ──────────────────────────────────────────────────
    print("Plotting vorticity overview …")
    fig, axes = plt.subplots(4, 1, figsize=(14, 14),
                             gridspec_kw={"hspace": 0.35})
    fig.suptitle(f"Cylinder Wake — Vorticity Field (Re={Re:.0f})",
                 fontsize=14, fontweight="bold", y=0.98)

    for ax, ti, tt in zip(axes, t_idx, t_targets):
        omega = vorticity(u[ti], v[ti], x, y)
        omega[body] = np.nan

        vmax = np.nanpercentile(np.abs(omega), 98)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax.pcolormesh(x, y, omega, cmap="RdBu_r", norm=norm,
                           shading="auto", rasterized=True)
        add_cylinder(ax, cx, cy, r)

        # Sensor 位置
        ax.scatter(sx, sy, s=15, c="lime", marker="o", zorder=6,
                   linewidths=0.4, edgecolors="black", label="K=100 sensors")

        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_aspect("equal")
        ax.set_title(f"t = {t[ti]:.2f} s", fontsize=11)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

        cb = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
        cb.set_label("ω (1/s)", fontsize=9)

    # 只在第一個 panel 顯示 legend
    axes[0].legend(loc="upper right", fontsize=9, markerscale=1.5)

    out = OUT_DIR / "overview_vorticity.png"
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")

    # ── 圖 2：u / v 速度場（t=10s） ────────────────────────────────────────────
    print("Plotting velocity fields …")
    ti_mid = np.argmin(np.abs(t - 10.0))
    fig, axes = plt.subplots(1, 2, figsize=(16, 4),
                             gridspec_kw={"wspace": 0.25})
    fig.suptitle(f"Velocity Field at t = {t[ti_mid]:.1f} s  (Re={Re:.0f})",
                 fontsize=13, fontweight="bold")

    for ax, field, label, cmap in zip(
            axes, [u[ti_mid], v[ti_mid]], ["u velocity (m/s)", "v velocity (m/s)"],
            ["RdBu_r", "RdBu_r"]):
        data = field.copy().astype(float)
        data[body] = np.nan
        vmax = np.nanpercentile(np.abs(data), 99)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.pcolormesh(x, y, data, cmap=cmap, norm=norm,
                           shading="auto", rasterized=True)
        add_cylinder(ax, cx, cy, r)
        ax.scatter(sx, sy, s=18, c="lime", marker="o", zorder=6,
                   linewidths=0.5, edgecolors="black")
        ax.set_xlim(x.min(), x.max()); ax.set_ylim(y.min(), y.max())
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    out = OUT_DIR / "overview_velocity.png"
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")

    # ── 圖 3：時間序列 ────────────────────────────────────────────────────────
    print("Plotting temporal signals …")
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)

    # KE(t)
    ax_ke = fig.add_subplot(gs[0, :])
    ke = 0.5 * (u.mean(axis=(1, 2))**2 + v.mean(axis=(1, 2))**2)
    # 每 10 幀取樣
    s = 10
    ax_ke.plot(t[::s], ke[::s], linewidth=1.2, color="#2563eb")
    ax_ke.set_xlabel("t (s)"); ax_ke.set_ylabel("Mean KE (m²/s²)")
    ax_ke.set_title("Mean Kinetic Energy over Time", fontsize=11)
    ax_ke.grid(alpha=0.3)

    # probe v(t) — 下游 cylinder 附近偵測 vortex shedding
    probe_i = H // 2 + 5
    probe_j = int(W * 0.35)
    ax_v = fig.add_subplot(gs[1, 0])
    ax_v.plot(t[::s], v[::s, probe_i, probe_j], linewidth=1.0, color="#dc2626")
    ax_v.set_xlabel("t (s)"); ax_v.set_ylabel("v (m/s)")
    ax_v.set_title(f"Probe v at ({x[probe_i,probe_j]:.3f}, {y[probe_i,probe_j]:.3f})",
                   fontsize=11)
    ax_v.grid(alpha=0.3)

    # sensor 位置圖
    ax_s = fig.add_subplot(gs[1, 1])
    omega_snap = vorticity(u[ti_mid], v[ti_mid], x, y)
    omega_snap[body] = np.nan
    vmax = np.nanpercentile(np.abs(omega_snap), 98)
    ax_s.pcolormesh(x, y, omega_snap, cmap="RdBu_r",
                    norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
                    shading="auto", alpha=0.6, rasterized=True)
    add_cylinder(ax_s, cx, cy, r)
    ax_s.scatter(sx, sy, s=30, c="lime", marker="o", zorder=6,
                 linewidths=0.6, edgecolors="black", label=f"K={len(sx)} sensors")
    ax_s.set_xlim(x.min(), x.max()); ax_s.set_ylim(y.min(), y.max())
    ax_s.set_aspect("equal")
    ax_s.set_title("QR-Pivot Sensor Positions (K=100)", fontsize=11)
    ax_s.set_xlabel("x (m)"); ax_s.set_ylabel("y (m)")
    ax_s.legend(loc="upper right", fontsize=9)

    out = OUT_DIR / "temporal_signals.png"
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")
    print("Done.")


if __name__ == "__main__":
    main()
