#!/bin/bash
# PCGrad ablation eval 提交腳本（remote, lab-server r740）。
# Usage（lab-server head node）：bash scripts/run_eval_pcgrad_ablation_remote.sh
#
# Controlled comparison：exp_pcgrad（use_pcgrad=true）vs exp_cosdiag（use_pcgrad=false,
# 標準 backward，位元等同 EXP-094 baseline）。兩者同 config / 同 seed=2，唯一差 PCGrad。
# 每個 checkpoint 一個 sbatch：載入 final.pt → evaluate_deeponet_cfc.py → eval_metrics.json。
# 架構參數用 evaluator 預設（operator-rank=256/d-model=256/fourier-harmonics=16/
# fourier-embed-dim=128/temporal-anchor），與 exp_pcgrad / exp_cosdiag config 一致。
set -euo pipefail

REPO="${HOME}/pi-lnn"
cd "$REPO"

DNS="data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy"
SENSOR_JSON="data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100.json"
SENSOR_NPZ="data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100_dns_values.npz"

# tag → artifact dir
TAGS=(pcgrad cosdiag)
declare -A ART_DIR=(
  [pcgrad]="artifacts/kolmogorov/exp_pcgrad_b3_dnspivot"
  [cosdiag]="artifacts/kolmogorov/exp_cosdiag_b3_dnspivot"
)

for TAG in "${TAGS[@]}"; do
  ART="${ART_DIR[$TAG]}"
  CKPT="${ART}/picon_kolmogorov_final.pt"
  if [[ ! -f "$CKPT" ]]; then
    echo "[skip] $CKPT 不存在"; continue
  fi
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=eval_${TAG}
#SBATCH --output=logs/eval_${TAG}_%j.out
#SBATCH --error=logs/eval_${TAG}_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G

set -euo pipefail
cd "$REPO"
export PYTHONUNBUFFERED=1 UV_OFFLINE=1
export PATH="\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH"

uv run python -u scripts/evaluate_deeponet_cfc.py \\
  --checkpoint "$CKPT" \\
  --dns "$DNS" \\
  --sensor-json "$SENSOR_JSON" \\
  --sensor-npz "$SENSOR_NPZ" \\
  --re 10000 \\
  --device cuda \\
  --n-times 50 \\
  --use-temporal-anchor \\
  --output "${ART}/eval_metrics.json"
EOF
  echo "[submit] eval_${TAG}  ckpt=$CKPT"
done
