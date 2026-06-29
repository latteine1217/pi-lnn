#!/usr/bin/env bash
# scripts/run_exp102_lesinformed.sh
#
# What:
#   EXP-102 完整 pipeline：用 LES-informed QR-pivot sensor 訓練 B3 model 10k steps
#   + evaluator → 拉出 summary metrics 對比 baseline。
#
# Why:
#   首次 REAL_WORLD_PIPELINE 完整驗證（CLAUDE.md REAL_WORLD_PIPELINE section）。
#   sensor 位置由 LES（工程現場 proxy）決定，sensor 量測值從 DNS（real-world surrogate）抽。
#
# How:
#   1. caffeinate -i 防 macOS sleep（B3 ~2 hr 訓練）
#   2. PYTORCH_ENABLE_MPS_FALLBACK=1（per KNOWN_PITFALLS）
#   3. uv run python -u → unbuffered stdout
#   4. 訓練完接 evaluator，metrics 寫進 logs/exp_102/metrics.tsv

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOGS_DIR="$REPO_ROOT/logs/exp_102"
mkdir -p "$LOGS_DIR"
TIMING_TSV="$LOGS_DIR/timing.tsv"
METRICS_TSV="$LOGS_DIR/metrics.tsv"

if [ ! -f "$TIMING_TSV" ]; then
  printf "exp\tstart_ts\tend_ts\twall_seconds\twall_human\n" > "$TIMING_TSV"
fi
if [ ! -f "$METRICS_TSV" ]; then
  printf "exp\tu_L2\tv_L2\tomega_L2\tKE_rel_err\tek_ratio_last\tdiv_L2\n" > "$METRICS_TSV"
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1

CONFIG="configs/exp_102_b3_lesinformed_qrpivot.toml"
ARTIFACT_DIR="artifacts/kolmogorov/stable/exp102-b3-lesinformed-seed2"
FINAL_PT="${ARTIFACT_DIR}/picon_kolmogorov_final.pt"
EVAL_DIR="${ARTIFACT_DIR}/deeponet-cfc-eval"
SUMMARY="${EVAL_DIR}/summary.json"
TRAIN_LOG="${LOGS_DIR}/exp_102_train.log"
EVAL_LOG="${LOGS_DIR}/exp_102_eval.log"

main () {
  echo "======================================================================"
  echo "[EXP-102] LES-informed QR-pivot, B3 seed=2  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "  config:    ${CONFIG}"
  echo "  artifact:  ${ARTIFACT_DIR}"
  echo "======================================================================"

  # ─── 1. Sanity check sensor files exist ────────────────
  local sensor_json="data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100_lesinformed.json"
  local sensor_npz="data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100_lesinformed_dns_values.npz"
  if [ ! -f "$sensor_json" ] || [ ! -f "$sensor_npz" ]; then
    echo "[EXP-102] ERROR: sensor file 不存在；請先跑 generate_sensors_qrpivot_from_les.py" >&2
    echo "  expected: $sensor_json" >&2
    echo "  expected: $sensor_npz" >&2
    exit 1
  fi
  echo "[EXP-102] sensor files OK"

  # ─── 2. Train ───────────────────────────────────────────
  local t0=$(date +%s)
  uv run python -u src/picon_kolmogorov.py \
    --config "$CONFIG" \
    --device mps \
    2>&1 | tee "$TRAIN_LOG"
  local t1=$(date +%s)
  local dur=$((t1 - t0))
  local h=$((dur/3600)); local m=$(((dur%3600)/60)); local s=$((dur%60))
  local human=$(printf "%dh %02dm %02ds" $h $m $s)
  printf "EXP-102\t%d\t%d\t%d\t%s\n" "$t0" "$t1" "$dur" "$human" >> "$TIMING_TSV"
  echo "[EXP-102] training wall-time: ${human}"

  if [ ! -f "$FINAL_PT" ]; then
    echo "[EXP-102] ERROR: ${FINAL_PT} 不存在；訓練失敗" >&2
    exit 1
  fi

  # ─── 3. Evaluate ────────────────────────────────────────
  mkdir -p "$EVAL_DIR"
  uv run python -u scripts/evaluate_deeponet_cfc.py \
    --config "$CONFIG" \
    --checkpoint "$FINAL_PT" \
    --output-dir "$EVAL_DIR" \
    --device mps \
    2>&1 | tee "$EVAL_LOG"

  # ─── 4. 抓 metrics ──────────────────────────────────────
  if [ -f "$SUMMARY" ]; then
    local u_l2 v_l2 omega_l2 ke ek_ratio div_l2
    u_l2=$(jq -r '.u_rel_l2_mean // "null"' "$SUMMARY")
    v_l2=$(jq -r '.v_rel_l2_mean // "null"' "$SUMMARY")
    omega_l2=$(jq -r '.omega_rel_l2_mean // "null"' "$SUMMARY")
    ke=$(jq -r '.ke_rel_err_mean // "null"' "$SUMMARY")
    ek_ratio=$(jq -r '.ek_ratio_kf_last // "null"' "$SUMMARY")
    div_l2=$(jq -r '.div_l2_mean // "null"' "$SUMMARY")
    printf "EXP-102\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$u_l2" "$v_l2" "$omega_l2" "$ke" "$ek_ratio" "$div_l2" >> "$METRICS_TSV"
    echo "[EXP-102] metrics → ${METRICS_TSV}"
  else
    echo "[EXP-102] WARN: ${SUMMARY} 不存在；metrics 未紀錄" >&2
  fi

  echo ""
  echo "[EXP-102] === DONE at $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo ""
  echo "=== Comparison ==="
  echo "baseline EXP-094 (DNS-pivot, seed=2):  KE ≈ 9.4%   (training placement)"
  echo "random pseed=1 (alt-eval, no retrain): KE ≈ 59.4%"
  echo "this EXP-102  (LES-informed, retrain): KE = ${ke:-?}"
}

if [ -z "${_INSIDE_CAFFEINATE:-}" ]; then
  export _INSIDE_CAFFEINATE=1
  exec caffeinate -i "$BASH_SOURCE" "$@"
fi

main "$@"
