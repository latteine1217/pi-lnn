#!/bin/bash
#SBATCH --job-name=eval_thesis_refresh
#SBATCH --output=logs/eval_thesis_refresh_%j.out
#SBATCH --error=logs/eval_thesis_refresh_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What:
#   對 EXP-290~294 thesis refresh checkpoints 跑 DNS evaluator。
#
# Why:
#   training loss 不是論文指標；必須用同一 evaluate_deeponet_cfc.py 產生
#   KE/u/v/omega/div/forcing summary，才可更新 thesis 表格。

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn}"

export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1

command -v uv >/dev/null 2>&1 || { echo "[ERR] uv 不在 PATH；先在 lab-server 裝 uv" >&2; exit 2; }
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在；請先在 head node 跑 'uv sync --python 3.12'" >&2; exit 3; }

echo "======================================================================"
echo "[eval_thesis_refresh] $(date '+%F %T') slurm_job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-?}"
echo "  repo: $(pwd)"
echo "  resources: partition=r740 gpu=1 cpus=8 mem=48G"
echo "======================================================================"
nvidia-smi -L 2>/dev/null || echo "[WARN] nvidia-smi 不可用"
uv --version

mapfile -t JOB_ROWS < <(
  uv run python - <<'PY'
from pathlib import Path
import tomllib

for path in sorted(Path("configs/stable").glob("exp_29*.toml")):
    if path.name.startswith("exp_295"):
        continue
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    cfg = payload.get("train", payload)
    artifact = cfg["artifacts_dir"]
    print(f"{path}\t{artifact}\t{path.stem}")
PY
)

for row in "${JOB_ROWS[@]}"; do
  IFS=$'\t' read -r cfg artifact stem <<< "$row"
  ckpt="${artifact}/picon_kolmogorov_final.pt"
  out_dir="artifacts/eval_thesis_refresh/${stem}"

  if [[ ! -f "$ckpt" ]]; then
    echo "[ERR] missing checkpoint for ${stem}: ${ckpt}" >&2
    exit 10
  fi

  echo "=== eval ${stem} ==="
  echo "  config:     ${cfg}"
  echo "  checkpoint: ${ckpt}"
  echo "  output:     ${out_dir}"

  uv run python scripts/evaluate_deeponet_cfc.py \
    --config "$cfg" \
    --checkpoint "$ckpt" \
    --output-dir "$out_dir" \
    --device cuda \
    --export_arrays
done

echo "[$(date '+%F %T')] eval_thesis_refresh completed (${#JOB_ROWS[@]} configs)"
