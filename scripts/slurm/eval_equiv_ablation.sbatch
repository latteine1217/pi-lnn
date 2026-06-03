#!/bin/bash
#SBATCH --job-name=eval_equiv
#SBATCH --output=logs/eval_equiv_%j.out
#SBATCH --error=logs/eval_equiv_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#
# What: 對 EXP-281/282/283 對等 ablation 的 15 個 final.pt 在 r740 (cuda) 跑 eval，
#       輸出 series.npz 至 artifacts/eval_<id>/。skip-existing 可重投。
# Why : eval 為 autograd-heavy compute，必須走 slurm 到 compute node，不在 head node / 本地。
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn}"
export PYTHONUNBUFFERED=1 UV_OFFLINE=1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在；先在 head node 跑 'uv sync --python 3.12'" >&2; exit 3; }
nvidia-smi -L 2>/dev/null || echo "[WARN] nvidia-smi 不可用"

SPECS=(
  "281:configs/exp_281_b0_les_T50_20k.toml"
  "281b:configs/exp_281b_b0_les_T50_20k_seed1.toml"
  "281c:configs/exp_281c_b0_les_T50_20k_seed2.toml"
  "281d:configs/exp_281d_b0_les_T50_20k_seed3.toml"
  "281e:configs/exp_281e_b0_les_T50_20k_seed4.toml"
  "282:configs/exp_282_b1_les_T50_20k.toml"
  "282b:configs/exp_282b_b1_les_T50_20k_seed1.toml"
  "282c:configs/exp_282c_b1_les_T50_20k_seed2.toml"
  "282d:configs/exp_282d_b1_les_T50_20k_seed3.toml"
  "282e:configs/exp_282e_b1_les_T50_20k_seed4.toml"
  "283:configs/exp_283_b2_les_T50_20k.toml"
  "283b:configs/exp_283b_b2_les_T50_20k_seed1.toml"
  "283c:configs/exp_283c_b2_les_T50_20k_seed2.toml"
  "283d:configs/exp_283d_b2_les_T50_20k_seed3.toml"
  "283e:configs/exp_283e_b2_les_T50_20k_seed4.toml"
)

for spec in "${SPECS[@]}"; do
  id="${spec%%:*}"; cfg="${spec##*:}"
  if [ -f "artifacts/eval_$id/series.npz" ]; then echo "[skip] eval_$id 已有 series.npz"; continue; fi
  adir=$(grep '^artifacts_dir' "$cfg" | sed 's/.*= *"//; s/".*//')
  ckpt="$adir/picon_kolmogorov_final.pt"
  if [ ! -f "$ckpt" ]; then echo "[WARN] 缺 checkpoint: $ckpt (跳過)"; continue; fi
  echo "================ eval EXP-$id ================"
  uv run python -u scripts/evaluate_deeponet_cfc.py --config "$cfg" \
    --checkpoint "$ckpt" --output-dir "artifacts/eval_$id" \
    --export_arrays --device cuda
done
echo "[DONE] eval_equiv all 15"
