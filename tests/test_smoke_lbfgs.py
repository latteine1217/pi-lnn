"""Smoke test for the L-BFGS path (also rewired to uvp_fn + h_states cache)."""
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
        "d_model": 32,
        "d_time": 8,
        "num_query_mlp_layers": 1,
        "query_mlp_hidden_dim": 32,
        "operator_rank": 32,
        "iterations": 2,
        "num_query_points": 32,
        "num_physics_points": 8,
        "checkpoint_period": 0,
        "device": "cpu",
        "artifacts_dir": str(ROOT / "artifacts/_smoke_lbfgs"),

        "lr_schedule": "lbfgs",
        "learning_rate": 1.0,
        "lbfgs_max_iter": 3,
        "lbfgs_history_size": 5,

        "time_marching": False,
        "kolmogorov_k_f": 2.0,
    })

    metrics_log: list[tuple[int, dict]] = []
    def log_fn(step, m): metrics_log.append((step, m))

    t0 = time.perf_counter()
    train_lnn_kolmogorov(cfg, log_fn=log_fn)
    dt = time.perf_counter() - t0
    print(f"\n=== L-BFGS smoke PASSED in {dt:.1f}s ===")
    for step, m in metrics_log:
        print(f"  step={step}  l_data={m['l_data']:.3e}  l_total={m['l_total']:.3e}")


if __name__ == "__main__":
    main()
