#!/bin/bash
#SBATCH --job-name=exp_295_baseline_squeeze
#SBATCH --output=logs/exp_295_baseline_squeeze_%j.out
#SBATCH --error=logs/exp_295_baseline_squeeze_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#
# What:
#   EXP-295 classical interpolation / squeeze baseline refresh。
#
# Why:
#   這不是 neural training job；不申請 GPU。仍在 r740 partition 使用半節點 CPU/RAM
#   (8 CPU, 48G) 讓資源規格與 thesis refresh 記錄一致。

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn}"

export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1

command -v uv >/dev/null 2>&1 || { echo "[ERR] uv 不在 PATH；先在 lab-server 裝 uv" >&2; exit 2; }
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在；請先在 head node 跑 'uv sync --python 3.12'" >&2; exit 3; }

echo "======================================================================"
echo "[EXP-295] $(date '+%F %T') slurm_job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-?}"
echo "  repo: $(pwd)"
echo "  resources: partition=r740 cpus=8 mem=48G gpu=none"
echo "======================================================================"
uv --version

uv run python -u scripts/baseline_squeeze.py

echo "[$(date '+%F %T')] EXP-295 baseline squeeze 結束"
