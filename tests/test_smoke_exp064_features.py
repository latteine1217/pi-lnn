"""Smoke test for EXP-064 production features (GradNorm + sensor_physics + SOAP)."""
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
        "fourier_embed_dim": 64,
        "num_token_attention_layers": 2,    # block_idx=1 dead code 路徑（EXP-064 設定）
        "use_temporal_anchor": True,
        "iterations": 5,
        "num_query_points": 64,
        "num_physics_points": 16,
        "checkpoint_period": 0,
        "device": "cpu",
        "artifacts_dir": str(ROOT / "artifacts/_smoke_exp064"),

        # GradNorm
        "use_gradnorm": True,
        "gradnorm_update_freq": 2,
        "gradnorm_init_weights": [1.0, 0.01, 0.01, 0.01],

        # sensor physics（uvp_fn 多算 p 的路徑）
        "use_sensor_physics": True,
        "num_sensor_physics_time_samples": 2,
        "sensor_physics_start_step": 0,

        # Poisson loss
        "poisson_loss_weight": 0.001,

        "lr_schedule": "step",
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

    print(f"\n=== EXP-064-style features PASSED in {dt:.1f}s ===")
    for step, m in metrics_log:
        print(
            f"  step={step}  l_data={m['l_data']:.3e}  l_phys={m['l_physics']:.3e}  "
            f"w_data={m.get('gn_w_data', 0):.3f}  w_cont={m.get('gn_w_cont', 0):.3f}"
        )


if __name__ == "__main__":
    main()
