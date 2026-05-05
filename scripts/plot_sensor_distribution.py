#!/usr/bin/env python3
"""plot_sensor_distribution.py — 期刊風格 sensor 分布視覺化（cylinder wake 背景）

What:
    從 sensor JSON（generate_sensors_qrpivot_cylinder.py 產出）+ Arrow shard
    讀取 sensor 位置與參考流場（vorticity at t_target），畫期刊格式分布圖。

Why:
    - 純 QR sensor 集中在 wake 高資訊區，但其他區域空缺
    - Hybrid (uniform + QR) 用兩階段 sampling，需直觀比對覆蓋差異
    - 論文 / 報告需要乾淨的 sensor placement plot 作為 method illustration

期刊格式（NeurIPS/ICLR）：
    DPI 300, sans-serif (Helvetica/Arial/DejaVu Sans), 字型 9-10 pt,
    軸線細 (0.7-0.8), 4-edge spines, inner ticks, 細灰 grid (alpha=0.3),
    避免方塊 marker（用 'o' / '^' / '*' 等開放性）。

Usage:
    uv run python scripts/plot_sensor_distribution.py \\
        --json data/cylinder_sensors/sensors_hybrid20qr80_K100_cylinder_Re10031.json \\
        --shard /path/to/data-00000-of-00092.arrow \\
        --t-target 10.0 \\
        --out artifacts/sensor_distribution.png

    --field-mode 可選 vorticity / u / v / speed（預設 vorticity）。
    --compare-json2 可加第二個 sensor JSON 做併排對照（左右 subplot）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
from matplotlib.colors import TwoSlopeNorm


# ── 期刊圖樣式（與 evaluate_deeponet_cfc 一致）─────────────────────────────────
_PREFERRED_FONTS = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": _PREFERRED_FONTS,
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "axes.linewidth": 0.7,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.grid": False,        # 流場 plot 不要 grid 干擾
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#666666",
    "legend.fancybox": False,
    "legend.borderpad": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 4.0,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.dpi": 100,
})


# ── 資料讀取 ─────────────────────────────────────────────────────────────────
def load_shard(path: Path) -> dict:
    """讀 Arrow shard（與 generate_sensors_qrpivot_cylinder.py 一致）."""
    with open(path, "rb") as f:
        reader = pa.ipc.open_stream(f)
        batch = reader.read_next_batch()
    row = {name: batch.column(name)[0].as_py() for name in batch.schema.names}
    T, H, W = row["shape_t"], row["shape_h"], row["shape_w"]
    t_len = row["t_shape"]
    x_H, x_W = row["x_shape_h"], row["x_shape_w"]
    return {
        "u": np.frombuffer(row["u"], dtype=np.float32).reshape(T, H, W),
        "v": np.frombuffer(row["v"], dtype=np.float32).reshape(T, H, W),
        "vo": (np.frombuffer(row["vo"], dtype=np.float32).reshape(T, H, W)
               if row["vo"] is not None else None),
        "x": np.frombuffer(row["x"], dtype=np.float64).reshape(x_H, x_W),
        "y": np.frombuffer(row["y"], dtype=np.float64).reshape(x_H, x_W),
        "t": np.frombuffer(row["t"], dtype=np.float64)[:t_len],
    }


def detect_body(u: np.ndarray, v: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """偵測 cylinder body（與 generate_sensors 同邏輯）."""
    idx = np.arange(0, u.shape[0], max(1, u.shape[0] // 40))
    mag = np.median(np.abs(u[idx]) + np.abs(v[idx]), axis=0)
    return mag < threshold


def vorticity_fd(u: np.ndarray, v: np.ndarray,
                 x_1d: np.ndarray, y_1d: np.ndarray) -> np.ndarray:
    """非週期 FD 渦度 ω = ∂v/∂x − ∂u/∂y（用物理座標，1/s 單位）."""
    dvdx = np.gradient(v, x_1d, axis=1)
    dudy = np.gradient(u, y_1d, axis=0)
    return dvdx - dudy


def pick_field(shard: dict, mode: str, t_idx: int,
               x_1d: np.ndarray, y_1d: np.ndarray) -> tuple[np.ndarray, str]:
    """依 mode 取背景場 + 對應 colorbar label."""
    if mode == "vorticity":
        if shard.get("vo") is not None:
            f = shard["vo"][t_idx]
        else:
            f = vorticity_fd(shard["u"][t_idx], shard["v"][t_idx], x_1d, y_1d)
        return f, "vorticity ω (1/s)"
    if mode == "u":
        return shard["u"][t_idx], "u (m/s)"
    if mode == "v":
        return shard["v"][t_idx], "v (m/s)"
    if mode == "speed":
        return np.sqrt(shard["u"][t_idx]**2 + shard["v"][t_idx]**2), "|u| (m/s)"
    raise ValueError(f"未知 field mode: {mode}")


# ── 繪圖 ─────────────────────────────────────────────────────────────────────
def plot_sensor_panel(
    ax,
    field: np.ndarray,
    x2d: np.ndarray,
    y2d: np.ndarray,
    body_mask: np.ndarray,
    sensors: dict,
    title: str,
    field_label: str,
) -> None:
    """畫單一 panel：流場 colormap + cylinder body mask + 兩類 sensor overlay.

    sensors: dict with optional keys 'uniform_xy' / 'qr_xy' / 'all_xy'
             (preferred order: uniform + qr if hybrid; fallback all_xy if pure QR)
    """
    # Mask body 為 NaN，避免在 body 內部繪 vorticity
    field_masked = field.astype(float).copy()
    field_masked[body_mask] = np.nan

    vmax = np.nanpercentile(np.abs(field_masked), 99)
    if not np.isfinite(vmax) or vmax < 1e-8:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    pcm = ax.pcolormesh(x2d, y2d, field_masked, cmap="RdBu_r", norm=norm,
                         shading="auto", rasterized=True)

    # cylinder body 用實心灰
    body_field = np.where(body_mask, 1.0, np.nan)
    ax.pcolormesh(x2d, y2d, body_field, cmap="Greys", vmin=0, vmax=1,
                  shading="auto", rasterized=True, alpha=0.85)

    # Sensors overlay
    if "uniform_xy" in sensors and len(sensors["uniform_xy"]) > 0:
        ux, uy = sensors["uniform_xy"][:, 0], sensors["uniform_xy"][:, 1]
        ax.scatter(ux, uy, marker="o", s=22, facecolor="#fde725",
                   edgecolor="black", linewidth=0.6,
                   label=f"Uniform ({len(ux)})", zorder=5)
    if "qr_xy" in sensors and len(sensors["qr_xy"]) > 0:
        qx, qy = sensors["qr_xy"][:, 0], sensors["qr_xy"][:, 1]
        ax.scatter(qx, qy, marker="^", s=18, facecolor="#26c6da",
                   edgecolor="black", linewidth=0.5,
                   label=f"QR pivot ({len(qx)})", zorder=4)
    if ("uniform_xy" not in sensors and "qr_xy" not in sensors
            and "all_xy" in sensors):
        ax_xy = sensors["all_xy"]
        ax.scatter(ax_xy[:, 0], ax_xy[:, 1], marker="o", s=18,
                   facecolor="#fb8500", edgecolor="black", linewidth=0.5,
                   label=f"Sensors ({len(ax_xy)})", zorder=4)

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", framealpha=0.95)

    # Colorbar
    cbar = plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(field_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def extract_sensors_xy(payload: dict, x2d: np.ndarray, y2d: np.ndarray) -> dict:
    """從 sensor JSON 抽出 (x, y) 物理座標，分 uniform/QR 或 all。"""
    H, W = x2d.shape
    out = {}
    if payload.get("uniform_sensor_flat") and payload.get("qr_sensor_flat"):
        for tag, key in (("uniform_xy", "uniform_sensor_flat"),
                         ("qr_xy", "qr_sensor_flat")):
            flat = np.asarray(payload[key], dtype=np.int64)
            if flat.size == 0:
                continue
            i = flat // W
            j = flat % W
            out[tag] = np.stack([x2d[i, j], y2d[i, j]], axis=1)
    else:
        # 純 QR / 舊 schema：用 sensor_flat
        flat = np.asarray(payload["sensor_flat"], dtype=np.int64)
        i = flat // W
        j = flat % W
        out["all_xy"] = np.stack([x2d[i, j], y2d[i, j]], axis=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="期刊風格 sensor 分布視覺化")
    parser.add_argument("--json", type=Path, required=True,
                        help="sensor JSON path（generate_sensors_qrpivot_cylinder.py 產出）")
    parser.add_argument("--shard", type=Path, required=True,
                        help="Arrow shard 路徑（取背景流場）")
    parser.add_argument("--t-target", type=float, default=10.0,
                        help="背景流場目標時刻（秒，會找最接近的 frame）")
    parser.add_argument("--field-mode", choices=["vorticity", "u", "v", "speed"],
                        default="vorticity", help="背景場類型（預設 vorticity）")
    parser.add_argument("--compare-json2", type=Path, default=None,
                        help="第二個 sensor JSON（用於併排對照，左 vs 右 subplot）")
    parser.add_argument("--out", type=Path, required=True, help="輸出 PNG 路徑")
    parser.add_argument("--body-threshold", type=float, default=1e-4,
                        help="cylinder body 偵測閾值（與 generate_sensors 一致）")
    args = parser.parse_args()

    print(f"Loading shard: {args.shard}")
    shard = load_shard(args.shard)
    x2d = shard["x"]
    y2d = shard["y"]
    H, W = x2d.shape
    x_1d = x2d[0, :]
    y_1d = y2d[:, 0]
    body_mask = detect_body(shard["u"], shard["v"], args.body_threshold)

    # 取最接近 t_target 的 frame
    t_idx = int(np.argmin(np.abs(shard["t"] - args.t_target)))
    t_actual = float(shard["t"][t_idx])
    print(f"Using t = {t_actual:.3f}s (frame {t_idx})")

    field, field_label = pick_field(shard, args.field_mode, t_idx, x_1d, y_1d)

    # Sensor JSON 解析
    print(f"Loading sensor JSON: {args.json}")
    payload = json.loads(args.json.read_text())
    sensors_a = extract_sensors_xy(payload, x2d, y2d)
    title_a = f"{payload.get('method', 'sensors')}: K={payload['K']}"
    if payload.get("n_uniform", 0) > 0:
        title_a += f" ({payload['n_uniform']} uniform + {payload['n_qr']} QR)"

    if args.compare_json2 is None:
        fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
        plot_sensor_panel(ax, field, x2d, y2d, body_mask, sensors_a,
                           title_a, field_label)
        fig.suptitle(f"Cylinder wake @ t={t_actual:.1f}s",
                     fontsize=11, y=1.02)
    else:
        # 左右對照
        print(f"Loading 2nd sensor JSON: {args.compare_json2}")
        payload_b = json.loads(args.compare_json2.read_text())
        sensors_b = extract_sensors_xy(payload_b, x2d, y2d)
        title_b = f"{payload_b.get('method', 'sensors')}: K={payload_b['K']}"
        if payload_b.get("n_uniform", 0) > 0:
            title_b += f" ({payload_b['n_uniform']} uniform + {payload_b['n_qr']} QR)"
        fig, axes = plt.subplots(1, 2, figsize=(15.0, 4.8), constrained_layout=True)
        plot_sensor_panel(axes[0], field, x2d, y2d, body_mask, sensors_a,
                           title_a, field_label)
        plot_sensor_panel(axes[1], field, x2d, y2d, body_mask, sensors_b,
                           title_b, field_label)
        fig.suptitle(f"Cylinder wake @ t={t_actual:.1f}s — sensor distribution comparison",
                     fontsize=11, y=1.03)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
