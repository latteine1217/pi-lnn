#!/usr/bin/env bash
# scripts/run_multiseed_eval_271_remote.sh
#
# EXP-271 (DNS-pivot oracle, n=5) 全種 seed 評估腳本，在 lab-server 跑。
#
# What: 對 EXP-271 所有 5 個 seed 的 final checkpoint 跑評估，
#       輸出 summary.json + series.npz，供 Mac 端 plot_multiseed_envelope.py 使用。
#
# Why: EXP-271 是 EXP-245（LES-pivot）的 DNS-pivot oracle 對照組（n=5, 20k iter）；
#      只有等 5 seed 全跑完，KE mean ± std 才能與 EXP-245 做 fair 比較。
#
# Usage (在 lab-server 上，全部 seed 訓練完後執行):
#   bash scripts/run_multiseed_eval_271_remote.sh
#
# Then on Mac:
#   rsync -avzP lab-server:~/pi-lnn/artifacts/eval_271_seed{a,b,c,d,e} \
#               artifacts/lab/
#   uv run python scripts/plot_multiseed_envelope.py \
#       --seed_dirs artifacts/lab/eval_271_seed{a,b,c,d,e} \
#       --output-dir thesis/figures/results

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CKPT_BASE="artifacts/kolmogorov"
CKPT_NAME="picon_kolmogorov_final.pt"

# seed tag → (config, checkpoint dir)
declare -a SEEDS=("a" "b" "c" "d" "e")
declare -A CFG_MAP=(
    [a]="configs/exp_271_b3_dns_pivot.toml"
    [b]="configs/exp_271b_b3_dns_pivot_seed1.toml"
    [c]="configs/exp_271c_b3_dns_pivot_seed2.toml"
    [d]="configs/exp_271d_b3_dns_pivot_seed3.toml"
    [e]="configs/exp_271e_b3_dns_pivot_seed4.toml"
)
declare -A CKPT_MAP=(
    [a]="${CKPT_BASE}/deeponet-cfc-re10000-exp271-b3-dns-pivot-20k/${CKPT_NAME}"
    [b]="${CKPT_BASE}/deeponet-cfc-re10000-exp271b-b3-dns-pivot-seed1-20k/${CKPT_NAME}"
    [c]="${CKPT_BASE}/deeponet-cfc-re10000-exp271c-b3-dns-pivot-seed2-20k/${CKPT_NAME}"
    [d]="${CKPT_BASE}/deeponet-cfc-re10000-exp271d-b3-dns-pivot-seed3-20k/${CKPT_NAME}"
    [e]="${CKPT_BASE}/deeponet-cfc-re10000-exp271e-b3-dns-pivot-seed4-20k/${CKPT_NAME}"
)

for tag in "${SEEDS[@]}"; do
    CFG="${CFG_MAP[$tag]}"
    CKPT="${CKPT_MAP[$tag]}"
    OUT="artifacts/eval_271_seed${tag}"

    if [[ ! -f "$CFG" ]]; then
        echo "[SKIP] seed=$tag: config $CFG missing"
        continue
    fi
    if [[ ! -f "$CKPT" ]]; then
        echo "[SKIP] seed=$tag: checkpoint $CKPT missing"
        continue
    fi

    echo "=== EXP-271 seed=${tag} ==="
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

echo "=== All EXP-271 seeds processed ==="
echo "Rsync to Mac:"
echo "  rsync -avzP lab-server:~/pi-lnn/artifacts/eval_271_seed{a,b,c,d,e} \\"
echo "              /Users/latteine/Documents/coding/pi-lnn/artifacts/lab/"
