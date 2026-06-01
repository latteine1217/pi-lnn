#!/bin/bash
# scripts/slurm/eval_capacity_sweep.sbatch
# What: 對 6 個 capacity-sweep checkpoint (EXP-280~285) 序列跑 DNS offline benchmark。
# Why: sbatch train template 只訓練不評估；判讀「容量是否飽和」需要 KE rel-err /
#      band(low/mid/high) rel-err / div_l2 等 benchmark，而非 train loss。
# 單 job 序列跑 6 個 eval（省 queue 排隊）；每個輸出 <artifacts>/<exp>-eval/summary.json。
#SBATCH --job-name=eval_cap_sweep
#SBATCH --output=logs/eval_cap_sweep_%j.out
#SBATCH --error=logs/eval_cap_sweep_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn}"

export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在；先在 head node 跑 'uv sync --python 3.12'" >&2; exit 3; }

ART="artifacts/kolmogorov/cap_sweep"
VARIANTS=(
  exp_280_cap_baseline
  exp_281_cap_width128
  exp_282_cap_width512
  exp_283_cap_rank128
  exp_284_cap_rank512
  exp_285_cap_liquid_tau
)

echo "======================================================================"
echo "[eval_cap_sweep] $(date '+%F %T')  slurm_job=${SLURM_JOB_ID}  node=${SLURMD_NODENAME:-?}"
nvidia-smi -L 2>/dev/null || echo "[WARN] nvidia-smi 不可用"
echo "======================================================================"

for v in "${VARIANTS[@]}"; do
  ckpt="${ART}/${v}/picon_kolmogorov_final.pt"
  out="${ART}/${v}-eval"
  echo ""
  echo "############### [eval] ${v}  $(date '+%F %T') ###############"
  if [ ! -f "$ckpt" ]; then
    echo "[ERR] checkpoint 不存在: $ckpt（跳過）" >&2
    continue
  fi
  uv run python -u scripts/evaluate_deeponet_cfc.py \
    --config "configs/generated/${v}.toml" \
    --checkpoint "$ckpt" \
    --output-dir "$out" \
    --device cuda
done

echo ""
echo "[$(date '+%F %T')] eval sweep done"
echo "summary 路徑: ${ART}/<exp>-eval/summary.json"
