"""Smoke test for the Natural Gradient (NG) training path.

驗證：
  1. lr_schedule="ng" 能完成 2 個 training steps 不報錯
  2. log_fn 收到合理的 metrics（l_data / l_phys / l_total 為 finite）

注意：NG 的 N（殘差向量大小）= num_query × num_re + 3 × num_phys × num_re
      此 smoke 設定 N = 32 + 3*8 = 56，明顯 < P，走 kernel trick 路徑。
"""
from __future__ import annotations

import math
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
        "sensor_npzs":  [str(ROOT / "data/kolmogorov_sensors/re1000/sensors_qrpivot_K100_N128_t0-5_dns_values.npz")],
        "dns_paths":    [str(ROOT / "data/dns/kolmogorov_dns_fp64_etdrk4_Re1000_N128_T5_ds4.npy")],
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
        "artifacts_dir": str(ROOT / "artifacts/_smoke_ng"),

        "lr_schedule": "ng",
        "learning_rate": 1.0,
        "ng_damping": 1.0e-3,
        "ng_damping_strategy": "fixed",
        "ng_jacobi_scaling": True,
        "ng_max_residuals": 2000,

        "time_marching": False,
        "kolmogorov_k_f": 2.0,
    })

    metrics_log: list[tuple[int, dict]] = []
    def log_fn(step, m): metrics_log.append((step, m))

    t0 = time.perf_counter()
    train_lnn_kolmogorov(cfg, log_fn=log_fn)
    dt = time.perf_counter() - t0

    assert len(metrics_log) == 2, f"預期 2 個 step 的 log，實際 {len(metrics_log)}"
    for step, m in metrics_log:
        for key in ("l_data", "l_physics", "l_total"):
            v = m[key]
            assert math.isfinite(v), f"step={step} 的 {key} 非 finite: {v}"

    print(f"\n=== NG smoke PASSED in {dt:.1f}s ===")
    for step, m in metrics_log:
        print(
            f"  step={step}  l_data={m['l_data']:.3e}  "
            f"l_phys={m['l_physics']:.3e}  l_total={m['l_total']:.3e}"
        )


if __name__ == "__main__":
    main()
