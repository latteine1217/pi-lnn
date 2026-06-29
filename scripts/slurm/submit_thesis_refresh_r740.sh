#!/usr/bin/env bash
# What:
#   提交 thesis refresh 實驗矩陣到 lab-server Slurm r740 partition。
#
# Why:
#   所有訓練 job 共用 scripts/slurm/train_exp.sbatch.tmpl；template 明確使用半節點級別資源：
#     --cpus-per-task=8
#     --mem=48G
#     --gres=gpu:1
#   r740 節點可讓兩個訓練 job 並行時仍保留 CPU/RAM 餘裕。
#
# Usage:
#   在 lab-server repo root 執行：
#     bash scripts/slurm/submit_thesis_refresh_r740.sh
#   Dry-run:
#     DRY=1 bash scripts/slurm/submit_thesis_refresh_r740.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs logs/sbatch

declare -a JOBS=(
  "290_noise01_a configs/stable/exp_290_noise01_a.toml"
  "290_noise01_b configs/stable/exp_290_noise01_b.toml"
  "290_noise01_c configs/stable/exp_290_noise01_c.toml"
  "290_noise01_d configs/stable/exp_290_noise01_d.toml"
  "290_noise01_e configs/stable/exp_290_noise01_e.toml"
  "290_noise03_a configs/stable/exp_290_noise03_a.toml"
  "290_noise03_b configs/stable/exp_290_noise03_b.toml"
  "290_noise03_c configs/stable/exp_290_noise03_c.toml"
  "290_noise03_d configs/stable/exp_290_noise03_d.toml"
  "290_noise03_e configs/stable/exp_290_noise03_e.toml"
  "290_noise05_a configs/stable/exp_290_noise05_a.toml"
  "290_noise05_b configs/stable/exp_290_noise05_b.toml"
  "290_noise05_c configs/stable/exp_290_noise05_c.toml"
  "290_noise05_d configs/stable/exp_290_noise05_d.toml"
  "290_noise05_e configs/stable/exp_290_noise05_e.toml"
  "290_noise10_a configs/stable/exp_290_noise10_a.toml"
  "290_noise10_b configs/stable/exp_290_noise10_b.toml"
  "290_noise10_c configs/stable/exp_290_noise10_c.toml"
  "290_noise10_d configs/stable/exp_290_noise10_d.toml"
  "290_noise10_e configs/stable/exp_290_noise10_e.toml"
  "291_rho003 configs/stable/exp_291_rho003.toml"
  "291_rho03 configs/stable/exp_291_rho03.toml"
  "291_rho1 configs/stable/exp_291_rho1.toml"
  "292_cont_pure_al configs/stable/exp_292_cont_pure_al.toml"
  "292_full_physics_pure_al configs/stable/exp_292_full_physics_pure_al.toml"
  "292_ns_pure_al_cont_double configs/stable/exp_292_ns_pure_al_cont_double.toml"
  "292_full_double configs/stable/exp_292_full_double.toml"
  "293_learn_kf configs/stable/exp_293_learn_kf.toml"
  "293_learn_A configs/stable/exp_293_learn_A.toml"
  "293_learn_both configs/stable/exp_293_learn_both.toml"
  "294_smoothing_bidir configs/stable/exp_294_smoothing_bidir.toml"
)

for entry in "${JOBS[@]}"; do
  exp_id="${entry%% *}"
  cfg="${entry#* }"
  if [[ ! -f "$cfg" ]]; then
    echo "[ERR] missing config: $cfg" >&2
    exit 1
  fi
done

echo "=== thesis refresh r740 submit ==="
echo "repo: $(pwd)"
echo "jobs: ${#JOBS[@]}"
echo "resources per training job: partition=r740, gpu=1, cpus=8, mem=48G"
echo

for entry in "${JOBS[@]}"; do
  exp_id="${entry%% *}"
  cfg="${entry#* }"
  echo "--- submit exp_${exp_id}: ${cfg}"
  scripts/slurm/submit_exp.sh "$exp_id" "$cfg"
done

echo "=== submitted ${#JOBS[@]} training jobs ==="
