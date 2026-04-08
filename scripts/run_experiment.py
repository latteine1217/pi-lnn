"""Pipeline: train → eval → (optional) compare.

What: 把 train / eval / compare 三個步驟包成一個 CLI 入口。
Why: 每次實驗都需要跑三個指令，且 eval output dir 命名容易出錯；
     統一入口讓流程可重跑、可追蹤，減少人工步驟與命名錯誤。

Convention:
  - train artifact: 由 config 的 artifacts_dir 決定
  - eval  output:   <artifacts_dir>-eval
  - compare output: <artifacts_dir>-compare（若有 --compare）
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import tomllib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline: train → eval → compare.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 只跑 train + eval
  uv run python scripts/run_experiment.py --config configs/my_exp.toml

  # 跑完後對比 baseline
  uv run python scripts/run_experiment.py \\
      --config configs/my_exp.toml \\
      --compare artifacts/deeponet-cfc-eval-baseline \\
      --compare-label Baseline

  # 跳過 train，只跑 eval + compare
  uv run python scripts/run_experiment.py \\
      --config configs/my_exp.toml --skip-train \\
      --compare artifacts/deeponet-cfc-eval-baseline
""",
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="TOML config 路徑。")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"],
                        default=None, help="覆蓋 config 的 device。")
    parser.add_argument("--skip-train", action="store_true",
                        help="跳過 train，直接從 final checkpoint 跑 eval。")
    parser.add_argument("--skip-eval", action="store_true",
                        help="跳過 eval（需搭配 --skip-train 使用時已有 eval 輸出）。")
    parser.add_argument(
        "--compare",
        type=Path,
        nargs="+",
        default=None,
        help="一或多個已有的 eval 輸出目錄，用於對比。",
    )
    parser.add_argument(
        "--compare-label",
        type=str,
        nargs="+",
        default=None,
        help="--compare 目錄對應的顯示標籤（數量需與 --compare 相同）。",
    )
    return parser.parse_args()


def load_artifacts_dir(config_path: Path) -> Path:
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    section = cfg.get("train", cfg)
    raw = section.get("artifacts_dir", "artifacts/experiment")
    return Path(raw)


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    args = parse_args()
    config = args.config.resolve()
    artifacts_dir = load_artifacts_dir(config)
    checkpoint = artifacts_dir / "lnn_kolmogorov_final.pt"
    eval_dir = Path(str(artifacts_dir) + "-eval")
    compare_dir = Path(str(artifacts_dir) + "-compare")

    print("=== run_experiment pipeline ===")
    print(f"config:       {config}")
    print(f"artifacts:    {artifacts_dir}")
    print(f"eval output:  {eval_dir}")

    # --- Step 1: Train ---
    if not args.skip_train:
        train_cmd = [
            "uv", "run", "python", "scripts/train_deeponet_cfc.py",
            "--config", str(config),
        ]
        if args.device:
            train_cmd += ["--device", args.device]
        run(train_cmd)
    else:
        print("\n[skip] train")

    if not checkpoint.exists():
        print(f"[ERROR] checkpoint 不存在: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    # --- Step 2: Eval ---
    if not args.skip_eval:
        eval_cmd = [
            "uv", "run", "python", "scripts/evaluate_deeponet_cfc.py",
            "--config", str(config),
            "--checkpoint", str(checkpoint),
            "--output-dir", str(eval_dir),
        ]
        if args.device:
            eval_cmd += ["--device", args.device]
        run(eval_cmd)
    else:
        print("\n[skip] eval")

    # --- Step 3: Compare (optional) ---
    if args.compare:
        labels_this = [artifacts_dir.name]
        compare_dirs = [str(eval_dir)] + [str(d.resolve()) for d in args.compare]

        compare_labels = [artifacts_dir.name]
        if args.compare_label:
            compare_labels += args.compare_label
        else:
            compare_labels += [d.name for d in args.compare]

        compare_cmd = [
            "uv", "run", "python", "scripts/compare_experiments.py",
            "--evals", *compare_dirs,
            "--labels", *compare_labels,
            "--output-dir", str(compare_dir),
        ]
        run(compare_cmd)
        print(f"\ncompare output: {compare_dir}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
