"""Train DeepONet + CfC on Kolmogorov flow.

What: 提供單一 CLI 入口，明確對應目前的 DeepONet + CfC 主模型。
Why: 之後 branch/trunk 與資料流程會持續調整，獨立腳本比直接呼叫 module entry
      更方便在轉型期間維持穩定入口與實驗命名。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pi_onet.lnn_kolmogorov import DEFAULT_LNN_ARGS, load_lnn_config, train_lnn_kolmogorov


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DeepONet + CfC Kolmogorov model.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/deeponet_cfc_smoke.toml"),
        help="Path to TOML config.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default=None,
        help="Override device from config.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override iterations from config for quick smoke checks.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Override artifacts directory from config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = dict(DEFAULT_LNN_ARGS)
    config.update(load_lnn_config(args.config))
    if args.device is not None:
        config["device"] = args.device
    if args.iterations is not None:
        config["iterations"] = int(args.iterations)
    if args.artifacts_dir is not None:
        config["artifacts_dir"] = str(args.artifacts_dir.resolve())

    print("=== DeepONet+CfC Train Entry ===")
    print(f"config: {args.config.resolve()}")
    print(f"iterations: {config['iterations']}")
    print(f"device: {config['device']}")
    print(f"artifacts_dir: {config['artifacts_dir']}")
    train_lnn_kolmogorov(config)


if __name__ == "__main__":
    main()
