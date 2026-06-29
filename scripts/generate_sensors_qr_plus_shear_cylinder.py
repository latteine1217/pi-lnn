#!/usr/bin/env python3
"""generate_sensors_qr_plus_shear_cylinder.py — 95 QR wake + 5 body-ring shear sensor placement。

What:
    Cylinder wake sensor placement，結合兩類 sensor：
      - n_shear 個「body 環最大剪切」sensor：在 cylinder body 相鄰的流體格點中，
        選時間平均 |∇u|+|∇v| 最大的點。物理上對應上下分離剪切層（渦街生成源）。
      - K - n_shear 個 QR-pivot wake sensor：在「排除 body + 排除已選 shear 點」的
        剩餘流體域上做 column-pivoted QR，抓 wake dominant modes。

Why:
    state（docs/cylinder_log_v2.md）顯示所有 geometry-awareness 嘗試都 over-energy。
    Codex 辯論共識：根因不是缺 encoder，是「wake 的生成源（separation shear layer）
    從未被觀測」。純 QR wake sensor（CEXP-002）集中 x>0.10 尾跡，看不到分離點動態。
    本 placement 用 5 個 shear sensor 直接觀測分離剪切層，補上缺失的動態資訊。

    Axis convention（KNOWN_PITFALLS EXP-101/102/103/105 災難根因）：
    完全沿用 generate_sensors_qrpivot_cylinder.py 的索引——sensor_i = flat // W（行/H），
    sensor_j = flat % W（列/W），coords 與 values 同源（u[:, i, j].T），結構性一致。

Usage:
    uv run python scripts/generate_sensors_qr_plus_shear_cylinder.py \\
        --shards /path/to/data-00000-of-00092.arrow \\
        --K 100 --n-shear 5 \\
        --time-stride 20 \\
        --out data/cylinder_sensors

Outputs:
    sensors_qr95shear5_K100_cylinder_Re{RE}.json
    sensors_qr95shear5_K100_cylinder_Re{RE}_values.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
from scipy.linalg import qr
from scipy.ndimage import binary_dilation

N_FEATURES = 4  # u, v, |∇u|_fd, |∇v|_fd


# ── 資料讀取（沿用 generate_sensors_qrpivot_cylinder.py）─────────────────────────

def load_shard(path: Path) -> dict:
    """從 Arrow IPC stream 讀取一個 cylinder shard。u/v/p/vo: [T,H,W]；x/y: [H,W]；t: [T]。"""
    with open(path, "rb") as f:
        reader = pa.ipc.open_stream(f)
        batch = reader.read_next_batch()

    row = {name: batch.column(name)[0].as_py() for name in batch.schema.names}
    T, H, W = row["shape_t"], row["shape_h"], row["shape_w"]
    t_len = row["t_shape"]
    x_H, x_W = row["x_shape_h"], row["x_shape_w"]

    return {
        "sim_id": row["sim_id"],
        "Re": float(row["sim_id"].replace(".h5", "")),
        "u": np.frombuffer(row["u"], dtype=np.float32).reshape(T, H, W),
        "v": np.frombuffer(row["v"], dtype=np.float32).reshape(T, H, W),
        "p": np.frombuffer(row["p"], dtype=np.float32).reshape(T, H, W),
        "x": np.frombuffer(row["x"], dtype=np.float64).reshape(x_H, x_W),
        "y": np.frombuffer(row["y"], dtype=np.float64).reshape(x_H, x_W),
        "t": np.frombuffer(row["t"], dtype=np.float64)[:t_len],
    }


def detect_cylinder_mask(u: np.ndarray, v: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """偵測 cylinder body：各時間步速度量級中位數 < threshold 的格點。True = body interior。"""
    idx = np.arange(0, u.shape[0], max(1, u.shape[0] // 40))
    mag = np.median(np.abs(u[idx]) + np.abs(v[idx]), axis=0)
    return mag < threshold


def fd_gradient_magnitude(field: np.ndarray, x2d: np.ndarray, y2d: np.ndarray) -> np.ndarray:
    """有限差分梯度量級 |∇f|，tensor-product 非均勻格（x 沿 axis=1，y 沿 axis=0）。"""
    x_1d = x2d[0, :]
    y_1d = y2d[:, 0]
    dfdx = np.gradient(field, x_1d, axis=1)
    dfdy = np.gradient(field, y_1d, axis=0)
    return np.sqrt(dfdx ** 2 + dfdy ** 2).astype(np.float32)


# ── Shear-ring 選點 ──────────────────────────────────────────────────────────

def mean_shear_field(shard: dict, time_stride: int) -> np.ndarray:
    """時間平均剪切量級 mean_t(|∇u| + |∇v|)，[H, W]。分離剪切層處最大。"""
    u_all, v_all = shard["u"], shard["v"]
    x2d, y2d = shard["x"], shard["y"]
    T = u_all.shape[0]
    t_idx = np.arange(0, T, time_stride)
    acc = np.zeros(u_all.shape[1:], dtype=np.float64)
    for ti in t_idx:
        acc += fd_gradient_magnitude(u_all[ti], x2d, y2d)
        acc += fd_gradient_magnitude(v_all[ti], x2d, y2d)
    return (acc / len(t_idx)).astype(np.float32)


def shear_ring_sensors(
    body_mask: np.ndarray,
    fluid_mask: np.ndarray,
    shear: np.ndarray,
    n_shear: int,
) -> np.ndarray:
    """選 body 相鄰流體環中時間平均剪切最大的 n_shear 個格點。

    Returns: [n_shear] flat indices（在 H×W），sorted。
    """
    H, W = body_mask.shape
    # body 膨脹一圈與流體域交集 = body-adjacent ring（8-連通）
    ring = binary_dilation(body_mask, iterations=1) & fluid_mask
    ring_flat = np.argwhere(ring.reshape(-1)).ravel()
    if len(ring_flat) < n_shear:
        raise ValueError(f"body ring 只有 {len(ring_flat)} 格 < n_shear={n_shear}")

    ring_shear = shear.reshape(-1)[ring_flat]
    top = ring_flat[np.argsort(ring_shear)[::-1][:n_shear]]
    return np.sort(top)


# ── Snapshot matrix + QR（沿用 generate_sensors_qrpivot_cylinder.py）──────────────

def normalize_rows(A: np.ndarray) -> None:
    A -= A.mean(axis=1, keepdims=True)
    std = A.std(axis=1, keepdims=True)
    std[std < 1e-10] = 1.0
    A /= std


def build_snapshot_matrix(shard: dict, time_stride: int, fluid_mask: np.ndarray) -> np.ndarray:
    """Snapshot matrix A ∈ ℝ^{(N_feat × T_sub) × N_fluid}（單 shard）。"""
    n_fluid = fluid_mask.sum()
    flat_fluid = fluid_mask.reshape(-1)
    x2d, y2d = shard["x"], shard["y"]

    u_all, v_all = shard["u"], shard["v"]
    T = u_all.shape[0]
    t_idx = np.arange(0, T, time_stride)
    T_sub = len(t_idx)
    print(f"  Snapshot: T={T} → {T_sub} frames (stride={time_stride})")

    block = np.empty((N_FEATURES * T_sub, n_fluid), dtype=np.float32)
    for bi, ti in enumerate(t_idx):
        u, v = u_all[ti], v_all[ti]
        grad_u = fd_gradient_magnitude(u, x2d, y2d)
        grad_v = fd_gradient_magnitude(v, x2d, y2d)
        features = np.stack([
            u.reshape(-1)[flat_fluid],
            v.reshape(-1)[flat_fluid],
            grad_u.reshape(-1)[flat_fluid],
            grad_v.reshape(-1)[flat_fluid],
        ], axis=0)
        block[bi * N_FEATURES:(bi + 1) * N_FEATURES] = features

    print(f"Snapshot matrix: {block.shape}  [{block.nbytes / 1e6:.0f} MB]")
    normalize_rows(block)
    return block


def qr_pivot_sensors(A: np.ndarray, K: int) -> np.ndarray:
    """Gram → truncated SVD → QR column pivoting → top-K column indices（sorted）。"""
    print(f"Gram matrix G = A A^T ({A.shape[0]}×{A.shape[0]}) ...")
    G = (A @ A.T).astype(np.float64)
    eigenvalues, V = np.linalg.eigh(G)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, V = eigenvalues[order], V[:, order]
    pos_ev = eigenvalues[eigenvalues > 0]
    explained = eigenvalues[:K].sum() / pos_ev.sum() if pos_ev.size > 0 else 0.0
    print(f"Top-{K} modes explain {explained:.1%} of variance")

    V_k = V[:, :K].astype(np.float32)
    sigma_k = np.sqrt(np.maximum(eigenvalues[:K], 0)).astype(np.float32)
    U_k = (A.T @ V_k) / sigma_k[None, :]
    _, _, piv = qr(U_k.T.astype(np.float64), pivoting=True)
    return np.sort(piv[:K])


# ── 主程式 ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="95 QR wake + 5 body-ring shear sensor placement")
    parser.add_argument("--shards", nargs="+", required=True, help="Arrow shard 路徑")
    parser.add_argument("--K", type=int, default=100, help="Sensor 總數")
    parser.add_argument("--n-shear", type=int, default=5, help="body-ring 剪切 sensor 數（剩餘由 QR 選）")
    parser.add_argument("--time-stride", type=int, default=20)
    parser.add_argument("--body-threshold", type=float, default=1e-4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    K, n_shear = args.K, args.n_shear
    n_qr = K - n_shear
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(args.shards) != 1:
        raise NotImplementedError("目前僅支援單 shard（與 CEXP-002 baseline 對齊）")

    print(f"Loading shard: {args.shards[0]}")
    shard = load_shard(Path(args.shards[0]))
    x2d, y2d = shard["x"], shard["y"]
    H, W = x2d.shape
    print(f"  Re={shard['Re']:.0f}  shape={shard['u'].shape}")

    # Body / fluid mask
    body_mask = detect_cylinder_mask(shard["u"], shard["v"], args.body_threshold)
    fluid_mask = ~body_mask
    n_body, n_fluid = int(body_mask.sum()), int(fluid_mask.sum())
    print(f"  Cylinder body: {n_body} cells  Fluid: {n_fluid} / {H * W}")

    # Stage 1: shear-ring sensors
    print(f"Stage 1: body-ring shear sensors (n_shear={n_shear}) ...")
    shear = mean_shear_field(shard, args.time_stride)
    shear_flat = shear_ring_sensors(body_mask, fluid_mask, shear, n_shear)
    shear_i = (shear_flat // W).astype(int)
    shear_j = (shear_flat % W).astype(int)
    print(f"  Shear sensors (x, y, mean_shear):")
    for fi, ii, jj in zip(shear_flat, shear_i, shear_j):
        print(f"    x={x2d[ii, jj]:.4f}  y={y2d[ii, jj]:.4f}  shear={shear.reshape(-1)[fi]:.2f}")

    # Stage 2: QR on fluid EXCLUDING shear cells
    print(f"Stage 2: QR pivot for {n_qr} wake sensors (excluding shear cells) ...")
    fluid_indices = np.argwhere(fluid_mask.reshape(-1)).ravel()
    A = build_snapshot_matrix(shard, args.time_stride, fluid_mask)
    # 把 shear 點從 fluid 候選中剔除（避免重複）
    shear_pos_in_fluid = np.searchsorted(fluid_indices, shear_flat)
    keep = np.ones(len(fluid_indices), dtype=bool)
    keep[shear_pos_in_fluid] = False
    A_qr = A[:, keep]
    fluid_idx_qr = fluid_indices[keep]
    qr_pick = qr_pivot_sensors(A_qr, n_qr)
    qr_flat = fluid_idx_qr[qr_pick]
    del A, A_qr

    # 合併
    sensor_flat = np.sort(np.concatenate([shear_flat, qr_flat]))
    if len(np.unique(sensor_flat)) != K:
        raise RuntimeError(f"shear/QR 選到重複點或數量不符（{len(np.unique(sensor_flat))} != {K}）")
    sensor_i = (sensor_flat // W).astype(int)
    sensor_j = (sensor_flat % W).astype(int)
    sensor_x = x2d[sensor_i, sensor_j]
    sensor_y = y2d[sensor_i, sensor_j]
    coords_xy = np.stack([sensor_x, sensor_y], axis=1)
    print(f"Sensor x ∈ [{sensor_x.min():.4f}, {sensor_x.max():.4f}]  "
          f"y ∈ [{sensor_y.min():.4f}, {sensor_y.max():.4f}]")

    # 時序（與 QR script 完全相同 axis convention）
    print("Extracting sensor time series ...")
    u_sensors = shard["u"][:, sensor_i, sensor_j].T.astype(np.float32)  # [K, T]
    v_sensors = shard["v"][:, sensor_i, sensor_j].T.astype(np.float32)
    t_out = shard["t"]

    # 儲存
    re_tag = f"Re{shard['Re']:.0f}"
    base = f"sensors_qr{n_qr}shear{n_shear}_K{K}_cylinder_{re_tag}"
    json_path = out_dir / f"{base}.json"
    npz_path = out_dir / f"{base}_values.npz"

    payload = {
        "K": K,
        "n_shear": n_shear,
        "n_qr": n_qr,
        "domain": "cylinder_wake",
        "grid": f"{H}x{W}",
        "n_fluid_cells": n_fluid,
        "n_body_cells": n_body,
        "body_threshold": args.body_threshold,
        "method": "qr_wake_plus_body_ring_shear",
        "features": ["u", "v", "grad_u_mag_fd", "grad_v_mag_fd"],
        "time_stride_qr": args.time_stride,
        "Re_list": [shard["Re"]],
        "source_shards": [str(p) for p in args.shards],
        "selected_coordinates": coords_xy.tolist(),
        "sensor_i": sensor_i.tolist(),
        "sensor_j": sensor_j.tolist(),
        "sensor_flat": sensor_flat.tolist(),
        "shear_sensor_flat": shear_flat.tolist(),  # 給 evaluator/plot 區分兩類
        "qr_sensor_flat": np.sort(qr_flat).tolist(),
        "values_npz": str(npz_path),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {json_path}")

    np.savez(npz_path, t=t_out, u=u_sensors, v=v_sensors, x=sensor_x, y=sensor_y)
    print(f"Saved: {npz_path}  (shape: {u_sensors.shape})")
    print("Done.")


if __name__ == "__main__":
    main()
