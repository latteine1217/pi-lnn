"""比較 NG vs Adam 在 Kolmogorov Re=1000 上的 loss / wall-time。

Usage:
    uv run python scripts/compare_ng_vs_adam.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NG_LOG = ROOT / "artifacts/exp_ng_001_re1000_smoke/loss_log.json"
ADAM_LOG = ROOT / "artifacts/exp_ng_001_baseline_adam/loss_log.json"


def _load(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"找不到 log：{path}")
    with open(path) as f:
        return json.load(f)


def _summary(name: str, logs: list[dict]) -> dict:
    steps = np.array([m["step"] for m in logs])
    walls = np.array([m["wall"] for m in logs])
    l_data = np.array([m["l_data"] for m in logs])
    l_phys = np.array([m["l_physics"] for m in logs])
    l_total = np.array([m["l_total"] for m in logs])

    return {
        "name": name,
        "n_steps": len(steps),
        "wall_total_s": float(walls[-1]),
        "wall_per_step_s": float(np.mean(np.diff(walls)) if len(walls) > 1 else walls[0]),
        "l_data_init": float(l_data[0]),
        "l_data_final": float(l_data[-1]),
        "l_data_min": float(l_data.min()),
        "l_data_min_step": int(steps[l_data.argmin()]),
        "l_phys_final": float(l_phys[-1]),
        "l_total_final": float(l_total[-1]),
        "l_total_min": float(l_total.min()),
    }


def main() -> None:
    print("=== NG vs Adam on Kolmogorov Re=1000 ===\n")

    ng_logs = _load(NG_LOG)
    adam_logs = _load(ADAM_LOG)

    ng = _summary("NG", ng_logs)
    adam = _summary("Adam", adam_logs)

    print(f"{'Metric':<22} {'NG':>14} {'Adam':>14}")
    print("-" * 52)
    for key in (
        "n_steps", "wall_total_s", "wall_per_step_s",
        "l_data_init", "l_data_final", "l_data_min",
        "l_phys_final", "l_total_final", "l_total_min",
    ):
        nv = ng[key]
        av = adam[key]
        if isinstance(nv, float) and abs(nv) < 1.0:
            fmt = "{:14.4e}"
        elif isinstance(nv, float):
            fmt = "{:14.2f}"
        else:
            fmt = "{:>14}"
        print(f"{key:<22} {fmt.format(nv):>14} {fmt.format(av):>14}")
    print(f"{'l_data_min_step':<22} {ng['l_data_min_step']:>14} {adam['l_data_min_step']:>14}")
    print()

    # Wall-time 對齊比較：在相同 wall-time 下 NG vs Adam 的 loss
    ng_walls = np.array([m["wall"] for m in ng_logs])
    ng_l_data = np.array([m["l_data"] for m in ng_logs])
    adam_walls = np.array([m["wall"] for m in adam_logs])
    adam_l_data = np.array([m["l_data"] for m in adam_logs])

    target_walls = [30, 60, 120, 240, 480]
    print(f"{'Wall (s)':>10} {'NG l_data':>14} {'Adam l_data':>14} {'NG/Adam ratio':>14}")
    print("-" * 56)
    for tw in target_walls:
        if tw > min(ng_walls.max(), adam_walls.max()):
            continue
        ng_idx = int(np.searchsorted(ng_walls, tw))
        adam_idx = int(np.searchsorted(adam_walls, tw))
        ng_idx = min(ng_idx, len(ng_l_data) - 1)
        adam_idx = min(adam_idx, len(adam_l_data) - 1)
        ngv, av = ng_l_data[ng_idx], adam_l_data[adam_idx]
        ratio = ngv / av if av > 0 else float("nan")
        print(f"{tw:>10} {ngv:>14.4e} {av:>14.4e} {ratio:>14.3f}")

    # 結論
    print("\n=== 結論 ===")
    if ng["l_total_final"] < adam["l_total_final"]:
        ratio = adam["l_total_final"] / max(ng["l_total_final"], 1e-30)
        print(f"NG 最終 l_total ({ng['l_total_final']:.3e}) 優於 Adam ({adam['l_total_final']:.3e})，"
              f"領先 {ratio:.2f}×")
    else:
        ratio = ng["l_total_final"] / max(adam["l_total_final"], 1e-30)
        print(f"Adam 最終 l_total ({adam['l_total_final']:.3e}) 優於 NG ({ng['l_total_final']:.3e})，"
              f"領先 {ratio:.2f}×")
    speedup = adam["wall_per_step_s"] / max(ng["wall_per_step_s"], 1e-30)
    print(f"每步成本：NG {ng['wall_per_step_s']:.3f}s vs Adam {adam['wall_per_step_s']:.3f}s "
          f"（NG 慢 {1/speedup:.1f}×）")


if __name__ == "__main__":
    main()
