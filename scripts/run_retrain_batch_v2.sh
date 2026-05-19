#!/usr/bin/env bash
# scripts/run_retrain_batch_v2.sh
#
# What:
#   Sequential retrain of EXP-101 / EXP-102 / EXP-103 / EXP-106 with
#   AXIS-BUG-FIXED sensor files (bug discovered 2026-05-18).
#
# Why:
#   既有 EXP-101/102/103 用 buggy sensor (NPZ value at swapped row/col),
#   KE rel-err 退步至 37–54%. Fix axis convention + regen sensor + retrain.
#   EXP-105 v2 已驗證 fix: KE drop 53.7% → 12.36%.
#   現批次跑剩餘 4 個實驗（含 NEW EXP-106 T=30 dns-init）取得乾淨 5-way 對比。
#
# Order:
#   1. EXP-101 v2 (random)          — quick, sanity baseline
#   2. EXP-103 v2 (LES_N256 T=5)    — short-window LES
#   3. EXP-106    (LES_N256 T=30)   — NEW: longer dns-init LES
#   4. EXP-102 v2 (LES_N128 T=15)   — over-dissipated stand-alone
#
# Each takes ~2.5 hr → total ~10 hr overnight.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOGS_ROOT="logs"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1

run_one() {
  local exp_id="$1"     # 101 / 102 / 103 / 106
  local tag="$2"        # variant tag for artifact dir
  local config="$3"     # config path

  local artifact_root="artifacts/kolmogorov/deeponet-cfc-re10000-exp${exp_id}-${tag}"
  local final_pt="${artifact_root}/picon_kolmogorov_final.pt"
  local eval_dir="${artifact_root}/deeponet-cfc-eval"
  local summary="${eval_dir}/summary.json"
  local logs_dir="${LOGS_ROOT}/exp_${exp_id}"
  mkdir -p "$logs_dir"
  local timing_tsv="${logs_dir}/timing.tsv"
  local metrics_tsv="${logs_dir}/metrics.tsv"
  [ -f "$timing_tsv" ] || printf "exp\tstart_ts\tend_ts\twall_seconds\twall_human\n" > "$timing_tsv"
  [ -f "$metrics_tsv" ] || printf "exp\tu_L2\tv_L2\tomega_L2\tKE_rel_err\tek_ratio_last\tdiv_L2\n" > "$metrics_tsv"
  local train_log="${logs_dir}/exp_${exp_id}_train.log"
  local eval_log="${logs_dir}/exp_${exp_id}_eval.log"

  echo ""
  echo "==================================================================="
  echo "[EXP-${exp_id}] retrain (axis-fixed)  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "  config:    ${config}"
  echo "  artifact:  ${artifact_root}"
  echo "==================================================================="

  if [ ! -f "$config" ]; then
    echo "[ERROR] config not found: $config" >&2
    return 1
  fi

  local t0=$(date +%s)
  uv run python -u src/picon_kolmogorov.py \
    --config "$config" \
    --device mps \
    2>&1 | tee "$train_log"
  local t1=$(date +%s)
  local dur=$((t1 - t0))
  local human=$(printf "%dh %02dm %02ds" $((dur/3600)) $(((dur%3600)/60)) $((dur%60)))
  printf "EXP-%s\t%d\t%d\t%d\t%s\n" "$exp_id" "$t0" "$t1" "$dur" "$human" >> "$timing_tsv"
  echo "[EXP-${exp_id}] training wall-time: ${human}"

  if [ ! -f "$final_pt" ]; then
    echo "[EXP-${exp_id}] ERROR: final.pt missing" >&2
    return 1
  fi

  mkdir -p "$eval_dir"
  uv run python -u scripts/evaluate_deeponet_cfc.py \
    --config "$config" \
    --checkpoint "$final_pt" \
    --output-dir "$eval_dir" \
    --device mps \
    2>&1 | tee "$eval_log"

  if [ -f "$summary" ]; then
    local u_l2 v_l2 omega_l2 ke ek_ratio div_l2
    u_l2=$(jq -r '.u_rel_l2_mean // "null"' "$summary")
    v_l2=$(jq -r '.v_rel_l2_mean // "null"' "$summary")
    omega_l2=$(jq -r '.omega_rel_l2_mean // "null"' "$summary")
    ke=$(jq -r '.ke_rel_err_mean // "null"' "$summary")
    ek_ratio=$(jq -r '.ek_ratio_kf_last // "null"' "$summary")
    div_l2=$(jq -r '.div_l2_mean // "null"' "$summary")
    printf "EXP-%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$exp_id" "$u_l2" "$v_l2" "$omega_l2" "$ke" "$ek_ratio" "$div_l2" >> "$metrics_tsv"
    echo "[EXP-${exp_id}] metrics → ${metrics_tsv}"
    echo "[EXP-${exp_id}] KE rel-err: $ke"
  else
    echo "[EXP-${exp_id}] WARN: summary.json missing" >&2
  fi
}

main() {
  echo "[batch] start at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[batch] order: EXP-101 → EXP-103 → EXP-106 → EXP-102"

  # 1. Random
  run_one "101" "b3-random-seed42" "configs/exp_101_b3_random_seed42.toml"

  # 2. LES_N256 T=5 dns-init
  run_one "103" "b3-lesinformed-n256-seed2" "configs/exp_103_b3_lesinformed_n256_qrpivot.toml"

  # 3. NEW: LES_N256 T=30 dns-init
  run_one "106" "b3-les-n256-T30-dnsinit-seed2" "configs/exp_106_b3_les_n256_T30_dnsinit_qrpivot.toml"

  # 4. LES_N128 stand-alone (over-dissipated)
  run_one "102" "b3-lesinformed-seed2" "configs/exp_102_b3_lesinformed_qrpivot.toml"

  echo ""
  echo "[batch] === ALL DONE at $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo ""
  echo "=== Combined metrics ==="
  for exp_id in 101 103 106 102; do
    if [ -f "logs/exp_${exp_id}/metrics.tsv" ]; then
      cat "logs/exp_${exp_id}/metrics.tsv"
    fi
  done
}

if [ -z "${_INSIDE_CAFFEINATE:-}" ]; then
  export _INSIDE_CAFFEINATE=1
  exec caffeinate -i "$BASH_SOURCE" "$@"
fi

main "$@"
