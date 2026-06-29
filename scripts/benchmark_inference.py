"""scripts/benchmark_inference.py

What:
    量測訓練完的 PI-CON 模型在純推論流程的耗時。
    將推論成本拆三層回報：
      (A) encode_once: sensor 時序 → hidden states 的一次性編碼成本。
      (B) query_field: 在 coarse grid (16384 points) 上 query 單一 (t, component) 場。
      (C) full_sequence: 與 `evaluate_deeponet_cfc.py` 一致的 T × 3 (u, v, p) 場評估
                        （共用同一份 hidden states，不重做 encode）。

Why:
    既有 evaluator 的 wall-time 混入 FFT/KE/IO/plot，
    無法回答「純模型推論」這個工程提問。

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/benchmark_inference.py \
        --config configs/exp_094_b3_seed2.toml \
        --checkpoint artifacts/kolmogorov/stable/exp094-b3-seed2/picon_kolmogorov_final.pt \
        --device mps \
        --output-json artifacts/benchmark_inference_exp094.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# 重用 evaluator 的 helper（同一 source of truth，避免 silent drift）
from evaluate_deeponet_cfc import (  # noqa: E402
    TRAIN_RATIO_FALLBACK,
    choose_device,
    coarse_reference_grid,
    extract_model_state,
    load_model_weights_strict,
)
from kolmogorov_dataset import KolmogorovDataset  # noqa: E402
from picon_kolmogorov import create_picon_model, load_picon_config  # noqa: E402


def _sync(device: torch.device) -> None:
    """確保 GPU/MPS 非同步 kernel 真正完成，否則計時失真。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _timed(fn, sync_device: torch.device) -> float:
    _sync(sync_device)
    t0 = time.perf_counter()
    fn()
    _sync(sync_device)
    return time.perf_counter() - t0


def _stats(samples: list[float]) -> dict[str, float]:
    return {
        "n": len(samples),
        "mean_ms": 1000.0 * statistics.fmean(samples),
        "std_ms": 1000.0 * statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "min_ms": 1000.0 * min(samples),
        "max_ms": 1000.0 * max(samples),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--batch", type=int, default=8192, help="與 evaluator 預設一致")
    p.add_argument("--n-encode", type=int, default=20)
    p.add_argument("--n-query", type=int, default=30)
    p.add_argument("--n-full", type=int, default=3)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_picon_config(args.config)
    device = choose_device(args.device)
    print(f"[setup] device={device}")

    # ── 1. 建模 + load weights ──────────────────────────────────────
    model = create_picon_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = extract_model_state(ckpt)
    load_model_weights_strict(model, state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] params={n_params/1e6:.2f}M ({n_params:,})")

    # ── 2. 載入 sensor + 建 query grid（與 evaluator 對齊）───────────
    ds = KolmogorovDataset(
        sensor_json=cfg["sensor_jsons"][0],
        sensor_npz=cfg["sensor_npzs"][0],
        dns_path=cfg["dns_paths"][0],
        re_value=float(cfg.get("re_values", [1000.0])[0]),
        observed_channel_names=tuple(cfg.get("observed_sensor_channels", ["u", "v"])),
        train_ratio=TRAIN_RATIO_FALLBACK,
        seed=int(cfg.get("seed", 42)),
    )
    sensor_pos = ds.sensor_pos.astype(np.float32)
    sensor_vals = ds.sensor_vals.astype(np.float32)
    sensor_time = ds.sensor_time.astype(np.float32)
    re_norm = ds.re_norm

    dns = np.load(cfg["dns_paths"][0], allow_pickle=True).item()
    x_g, y_g = coarse_reference_grid(
        dns["x"].astype(np.float32), dns["y"].astype(np.float32)
    )
    xx, yy = np.meshgrid(x_g, y_g, indexing="ij")
    xy_flat = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    n_grid = xy_flat.shape[0]
    n_t = len(sensor_time)
    K = sensor_pos.shape[0]
    n_components = 3  # u, v, p
    print(
        f"[setup] K={K}, T={n_t}, grid={n_grid} ({len(x_g)}x{len(y_g)}), "
        f"batch={args.batch}, components={n_components}"
    )

    sv_t = torch.tensor(sensor_vals.transpose(1, 0, 2), dtype=torch.float32, device=device)
    sp_t = torch.tensor(sensor_pos, dtype=torch.float32, device=device)
    st_t = torch.tensor(sensor_time, dtype=torch.float32, device=device)
    xy_t = torch.tensor(xy_flat, dtype=torch.float32, device=device)

    # 預先 encode 一份給 (B)(C) 共用（與 evaluator 路徑一致）
    with torch.no_grad():
        h_states, s_time = model.encode(sv_t, sp_t, re_norm, st_t)

    batch = args.batch

    # ── 3. 計時 closure ─────────────────────────────────────────────
    def encode_once() -> None:
        with torch.no_grad():
            model.encode(sv_t, sp_t, re_norm, st_t)

    t_mid = float(sensor_time[n_t // 2])

    def query_one_field() -> None:
        with torch.no_grad():
            for start in range(0, n_grid, batch):
                end = min(start + batch, n_grid)
                xy_b = xy_t[start:end]
                bs = end - start
                t_b = torch.full((bs,), t_mid, dtype=torch.float32, device=device)
                c_b = torch.zeros((bs,), dtype=torch.long, device=device)  # u channel
                model.query_decoder(xy_b, t_b, c_b, h_states, s_time, sp_t)

    def full_sequence() -> None:
        with torch.no_grad():
            for ti in range(n_t):
                t_val = float(sensor_time[ti])
                for comp_idx in range(n_components):
                    for start in range(0, n_grid, batch):
                        end = min(start + batch, n_grid)
                        xy_b = xy_t[start:end]
                        bs = end - start
                        t_b = torch.full((bs,), t_val, dtype=torch.float32, device=device)
                        c_b = torch.full((bs,), comp_idx, dtype=torch.long, device=device)
                        model.query_decoder(xy_b, t_b, c_b, h_states, s_time, sp_t)

    # ── 4. warmup（避免首次 JIT / kernel compile 污染數字）────────
    print(f"[warmup] x{args.warmup} (encode + single query)")
    for _ in range(args.warmup):
        encode_once()
        query_one_field()
    _sync(device)

    # ── 5. 三層計時 ──────────────────────────────────────────────
    print(f"[time] (A) encode x{args.n_encode} ...")
    samples_a = [_timed(encode_once, device) for _ in range(args.n_encode)]

    print(f"[time] (B) single (t, component) field x{args.n_query} ...")
    samples_b = [_timed(query_one_field, device) for _ in range(args.n_query)]

    print(f"[time] (C) full sequence (T={n_t} x {n_components} components) x{args.n_full} ...")
    samples_c = [_timed(full_sequence, device) for _ in range(args.n_full)]

    # ── 6. 彙整輸出 ──────────────────────────────────────────────
    mean_b_s = statistics.fmean(samples_b)
    mean_c_s = statistics.fmean(samples_c)
    report = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "model_params": n_params,
        "K": K,
        "T_snapshots": n_t,
        "n_grid": n_grid,
        "n_components": n_components,
        "batch": batch,
        "encode": _stats(samples_a),
        "single_field": _stats(samples_b),
        "single_field_throughput_queries_per_s": n_grid / mean_b_s,
        "full_sequence": _stats(samples_c),
        "full_sequence_per_snapshot_ms": 1000.0 * mean_c_s / n_t,
        "full_sequence_per_field_ms": 1000.0 * mean_c_s / (n_t * n_components),
    }
    print("\n=== Result ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"[save] {args.output_json}")


if __name__ == "__main__":
    main()
