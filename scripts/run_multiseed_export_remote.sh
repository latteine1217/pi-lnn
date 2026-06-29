#!/usr/bin/env bash
# scripts/run_multiseed_export_remote.sh
#
# (C) Step 2: Run evaluator on lab GPU for all 5 EXP-245 seeds (a/b/c/d/e),
# writing series.npz per seed (needed for multi-seed envelope plotting).
#
# Usage (on lab GPU, after evaluator patch has been pulled):
#   bash scripts/run_multiseed_export_remote.sh
#
# Then on Mac:
#   rsync -avzP lab-server:/path/to/pi-lnn/artifacts/eval_245_seed{a,b,c,d,e} \
#               artifacts/lab/
#   uv run python scripts/plot_multiseed_envelope.py \
#       --seed_dirs artifacts/lab/eval_245_seed{a,b,c,d,e} \
#       --output-dir thesis/figures/results

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Seed → (config suffix, checkpoint dir) mapping.
# Adjust CKPT_BASE / CKPT_NAME to match your lab artifact layout.
CKPT_BASE="${CKPT_BASE:-artifacts/kolmogorov}"
CKPT_NAME="${CKPT_NAME:-picon_kolmogorov_final.pt}"

declare -a SEEDS=("a" "b" "c" "d" "e")
declare -A SEED_NUMS=([a]=42 [b]=1 [c]=2 [d]=3 [e]=4)

for tag in "${SEEDS[@]}"; do
    seed="${SEED_NUMS[$tag]}"
    if [[ "$tag" == "a" ]]; then
        CFG="configs/stable/exp_245.toml"
        CKPT="${CKPT_BASE}/deeponet-cfc-re10000-exp245-b3-les-T50/${CKPT_NAME}"
    else
        CFG="configs/stable/exp_245_${tag}.toml"
        CKPT="${CKPT_BASE}/deeponet-cfc-re10000-exp245-${tag}-b3-les-T50/${CKPT_NAME}"
    fi
    OUT="artifacts/eval_245_seed${tag}"

    if [[ ! -f "$CFG" ]]; then
        echo "⚠️  Skip seed=$tag: config $CFG missing"
        continue
    fi
    if [[ ! -f "$CKPT" ]]; then
        echo "⚠️  Skip seed=$tag: checkpoint $CKPT missing"
        continue
    fi

    echo "=== seed=$tag (seed=$seed) ==="
    echo "  config:     $CFG"
    echo "  checkpoint: $CKPT"
    echo "  output:     $OUT"

    uv run python scripts/evaluate_deeponet_cfc.py \
        --config "$CFG" \
        --checkpoint "$CKPT" \
        --output-dir "$OUT" \
        --device cuda \
        --export_arrays

    echo
done

echo "=== All seeds processed ==="
echo "Now rsync the artifact dirs to your Mac:"
echo "  rsync -avzP lab-server:${REPO_ROOT}/artifacts/eval_245_seed{a,b,c,d,e} \\"
echo "              <mac-repo>/artifacts/lab/"
echo "Then on Mac:"
echo "  uv run python scripts/plot_multiseed_envelope.py \\"
echo "      --seed_dirs artifacts/lab/eval_245_seed{a,b,c,d,e} \\"
echo "      --output-dir thesis/figures/results"
