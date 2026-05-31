#!/bin/bash
#SBATCH --job-name=sweep_al
#SBATCH --output=logs/sweep_al_%j.out
#SBATCH --error=logs/sweep_al_%j.err
#SBATCH --time=3-00:00:00          # 30 trials × ~2.2hr（含 MedianPruner 剪枝後更少）
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn}"

export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1                 # r740 無 internet
command -v uv >/dev/null 2>&1 || { echo "[ERR] uv not in PATH" >&2; exit 2; }
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在；先在 head node 'uv sync --python 3.12'" >&2; exit 3; }

echo "======================================================================"
echo "[sweep_al] $(date '+%F %T')  slurm_job=${SLURM_JOB_ID}  node=${SLURMD_NODENAME:-?}"
echo "  trials=${TRIALS:-30}  iters=${ITERS:-16000}"
echo "======================================================================"
nvidia-smi -L 2>/dev/null || echo "[WARN] nvidia-smi 不可用"

# Optuna SQLite (artifacts/sweep_al/optuna_al.db) 為 source of truth；可斷點續跑。
uv run python -u scripts/sweep_al_re10000.py \
  --trials "${TRIALS:-30}" \
  --iterations "${ITERS:-16000}" \
  --device cuda \
  --study-name "${STUDY:-al-re10000-v1}"

echo "[$(date '+%F %T')] sweep_al 結束"
