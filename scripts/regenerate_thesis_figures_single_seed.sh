#!/usr/bin/env bash
# scripts/regenerate_thesis_figures_single_seed.sh
#
# (B) Regenerate thesis figures for the LES-pivot main pipeline using EXP-245_a
# (seed=42, 20k iter) checkpoint. Overwrites PNGs in thesis/figures/results/.
#
# Prerequisite: rsync the EXP-245_a checkpoint from the lab GPU first, e.g.
#   rsync -avz lab-gpu:/home/lab/pi-lnn/artifacts/.../exp_245_seed42/step_20000.pt \
#         artifacts/lab/deeponet-cfc-re10000-exp245-b3-les-T50/seed42/
#
# Usage:
#   bash scripts/regenerate_thesis_figures_single_seed.sh                   # uses default ckpt path
#   bash scripts/regenerate_thesis_figures_single_seed.sh path/to/ckpt.pt   # overrides ckpt path

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/stable/exp_245.toml"
OUTPUT_DIR="${REPO_ROOT}/thesis/figures/results"

# --- Resolve checkpoint path -------------------------------------------------
DEFAULT_CKPT_DIR="${REPO_ROOT}/artifacts/lab/deeponet-cfc-re10000-exp245-b3-les-T50/seed42"
CKPT="${1:-}"
if [[ -z "${CKPT}" ]]; then
    CKPT="$(ls -t "${DEFAULT_CKPT_DIR}"/step_*.pt 2>/dev/null | head -1 || true)"
fi

if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
    cat <<EOF
❌ Cannot locate EXP-245_a checkpoint.

Expected one of:
  - Pass as argument:   bash $0 path/to/checkpoint.pt
  - Default location:   ${DEFAULT_CKPT_DIR}/step_*.pt

To rsync from lab GPU (example; adjust hostname/path):
  mkdir -p "${DEFAULT_CKPT_DIR}"
  rsync -avzP lab-gpu:/path/to/exp_245_seed42/step_20000.pt \\
              "${DEFAULT_CKPT_DIR}/"
EOF
    exit 1
fi

echo "=== EXP-245_a single-seed thesis figure regeneration ==="
echo "Config     : ${CONFIG}"
echo "Checkpoint : ${CKPT}"
echo "Output dir : ${OUTPUT_DIR}"
echo

mkdir -p "${OUTPUT_DIR}"

# --- Run evaluator -----------------------------------------------------------
# MPS fallback is required (SOAP eigh op falls back to CPU on Apple silicon).
PYTORCH_ENABLE_MPS_FALLBACK=1 \
    uv run python "${REPO_ROOT}/scripts/evaluate_deeponet_cfc.py" \
        --config "${CONFIG}" \
        --checkpoint "${CKPT}" \
        --output_dir "${OUTPUT_DIR}" \
        --device mps

echo
echo "✅ Thesis figures regenerated at ${OUTPUT_DIR}"
echo "   Now recompile thesis/main.tex (2-pass) to embed the refreshed figures."
