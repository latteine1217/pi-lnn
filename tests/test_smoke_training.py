"""Smoke test：用真實資料跑 5 步完整訓練迴圈，驗證 Batch A+B+C 整合不爆。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from pi_lnn import DEFAULT_LNN_ARGS, train_lnn_kolmogorov

    cfg = dict(DEFAULT_LNN_ARGS)
    cfg.update({
        "sensor_jsons": [str(ROOT / "data/kolmogorov_sensors/re1000/sensors_qrpivot_K100_N128_t0-5.json")],
        "sensor_npzs": [str(ROOT / "data/kolmogorov_sensors/re1000/sensors_qrpivot_K100_N128_t0-5_dns_values.npz")],
        "dns_paths": [str(ROOT / "data/dns/kolmogorov_dns_fp64_etdrk4_Re1000_N128_T5_ds4.npy")],
        "re_values": [1000.0],
        "observed_sensor_channels": ["u", "v"],
        "d_model": 64,
        "d_time": 8,
        "num_query_mlp_layers": 1,
        "query_mlp_hidden_dim": 64,
        "operator_rank": 64,
        "iterations": 5,
        "num_query_points": 64,
        "num_physics_points": 16,
        "checkpoint_period": 0,
        "device": "cpu",   # CPU 才能跑得起來不裝 mps；MPS 一樣可，但 CI 預設 CPU
        "artifacts_dir": str(ROOT / "artifacts/_smoke_batchABC"),
        "lr_schedule": "step",
        "lr_decay_steps": 1000,
        "lr_decay_gamma": 0.9,
        "time_marching": True,
        "time_marching_start": 1.0,
        "time_marching_warmup": 0.5,
        "kolmogorov_k_f": 2.0,
    })

    metrics_log: list[tuple[int, dict[str, float]]] = []

    def log_fn(step: int, metrics: dict) -> None:
        metrics_log.append((step, metrics))

    t0 = time.perf_counter()
    train_lnn_kolmogorov(cfg, log_fn=log_fn)
    dt = time.perf_counter() - t0

    print(f"\n=== Smoke training PASSED in {dt:.1f}s, {len(metrics_log)} steps logged ===")
    for step, m in metrics_log:
        print(f"  step={step:3d}  l_data={m['l_data']:.4e}  l_physics={m['l_physics']:.4e}  l_total={m['l_total']:.4e}")


if __name__ == "__main__":
    main()
