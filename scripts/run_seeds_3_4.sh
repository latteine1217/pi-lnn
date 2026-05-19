#!/usr/bin/env bash
# scripts/run_seeds_3_4.sh
#
# What:
#   串行跑 4 個 multi-seed runs (B0/B3 × seed 3/4) → final.pt → evaluator → 收集 metrics。
#   把 EXP-093/094/095/096 的 3-seed 統計延伸到 5-seed。
#
# Why:
#   並行 (EXP-093+095 同 start) 會讓 wall-time 失真，paper 報數字不可信。
#   串行模式給乾淨的 single-run timing + 不互相 contention 的 GradNorm 動態。
#
# How:
#   1. caffeinate 防 macOS sleep (5h 訓練不能讓電腦休眠)
#   2. PYTORCH_ENABLE_MPS_FALLBACK=1 (CLAUDE.md KNOWN_PITFALLS)
#   3. uv run python -u → unbuffered stdout (EXP-059 教訓：8 KB block buffering 看似 hung)
#   4. 每 run 個別 log → logs/exp_NNN.log；wall-time 寫到 timing.tsv
#   5. evaluator 跑完後從 summary.json 抓 KE / u_L2 / v_L2 / omega_L2 / ek_ratio / div_L2
#
# Order: B0 先（每個 ~25 min），快收尾；B3 後（每個 ~2h），避免 B3 中途失敗浪費 B0 結果

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOGS_DIR="$REPO_ROOT/logs/seeds_3_4"
mkdir -p "$LOGS_DIR"
TIMING_TSV="$LOGS_DIR/timing.tsv"
METRICS_TSV="$LOGS_DIR/metrics.tsv"

# 初始化 TSV header（idempotent — 若已存在保留歷史）
if [ ! -f "$TIMING_TSV" ]; then
  printf "exp\tstart_ts\tend_ts\twall_seconds\twall_human\n" > "$TIMING_TSV"
fi
if [ ! -f "$METRICS_TSV" ]; then
  printf "exp\tu_L2\tv_L2\tomega_L2\tKE_rel_err\tek_ratio_last\tdiv_L2\n" > "$METRICS_TSV"
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1   # double-protect stdout flush（除了 -u 之外）

run_one () {
  local exp_id="$1"      # e.g. 097
  local variant="$2"     # e.g. b3-seed3
  local config="$3"      # e.g. configs/exp_097_b3_seed3.toml
  local artifact_dir="artifacts/kolmogorov/deeponet-cfc-re10000-exp${exp_id}-${variant}"
  local final_pt="$artifact_dir/picon_kolmogorov_final.pt"
  local eval_dir="$artifact_dir/deeponet-cfc-eval"
  local summary="$eval_dir/summary.json"
  local train_log="$LOGS_DIR/exp_${exp_id}_train.log"
  local eval_log="$LOGS_DIR/exp_${exp_id}_eval.log"

  echo ""
  echo "======================================================================"
  echo "[run_one] EXP-${exp_id} ${variant}  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "  config:    ${config}"
  echo "  artifact:  ${artifact_dir}"
  echo "======================================================================"

  # ─── 1. Train ───────────────────────────────────────────
  local t0=$(date +%s)
  uv run python -u src/picon_kolmogorov.py \
    --config "$config" \
    --device mps \
    2>&1 | tee "$train_log"
  local t1=$(date +%s)
  local dur=$((t1 - t0))
  local h=$((dur/3600)); local m=$(((dur%3600)/60)); local s=$((dur%60))
  local human=$(printf "%dh %02dm %02ds" $h $m $s)
  printf "EXP-%s\t%d\t%d\t%d\t%s\n" "$exp_id" "$t0" "$t1" "$dur" "$human" >> "$TIMING_TSV"
  echo "[run_one] EXP-${exp_id} training wall-time: ${human}"

  if [ ! -f "$final_pt" ]; then
    echo "[run_one] ERROR: ${final_pt} 不存在；訓練失敗" >&2
    return 1
  fi

  # ─── 2. Evaluate ────────────────────────────────────────
  mkdir -p "$eval_dir"
  uv run python -u scripts/evaluate_deeponet_cfc.py \
    --config "$config" \
    --checkpoint "$final_pt" \
    --output-dir "$eval_dir" \
    --device mps \
    2>&1 | tee "$eval_log"

  # ─── 3. 從 summary.json 抓 metric（容錯 — summary 可能 key 略不同）
  if [ -f "$summary" ]; then
    # 抓 evaluator summary.json 真實 key（見 scripts/evaluate_deeponet_cfc.py:1201–1216）
    local u_l2 v_l2 omega_l2 ke ek_ratio div_l2
    u_l2=$(jq -r '.u_rel_l2_mean // "null"' "$summary")
    v_l2=$(jq -r '.v_rel_l2_mean // "null"' "$summary")
    omega_l2=$(jq -r '.omega_rel_l2_mean // "null"' "$summary")
    ke=$(jq -r '.ke_rel_err_mean // "null"' "$summary")
    ek_ratio=$(jq -r '.ek_ratio_kf_last // "null"' "$summary")
    div_l2=$(jq -r '.div_l2_mean // "null"' "$summary")
    printf "EXP-%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$exp_id" "$u_l2" "$v_l2" "$omega_l2" "$ke" "$ek_ratio" "$div_l2" >> "$METRICS_TSV"
    echo "[run_one] EXP-${exp_id} metrics → ${METRICS_TSV}"
  else
    echo "[run_one] WARN: ${summary} 不存在；metrics 未紀錄" >&2
  fi
}

main () {
  echo "[main] start at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[main] timing.tsv → $TIMING_TSV"
  echo "[main] metrics.tsv → $METRICS_TSV"

  # 順序：B0 先（快、低風險先過 pipeline）；B3 後（重頭）
  run_one "099" "b0-seed3" "configs/exp_099_b0_seed3.toml"
  run_one "100" "b0-seed4" "configs/exp_100_b0_seed4.toml"
  run_one "097" "b3-seed3" "configs/exp_097_b3_seed3.toml"
  run_one "098" "b3-seed4" "configs/exp_098_b3_seed4.toml"

  echo ""
  echo "[main] === ALL DONE at $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo ""
  echo "=== Timing ==="
  cat "$TIMING_TSV"
  echo ""
  echo "=== Metrics ==="
  cat "$METRICS_TSV"
}

# 自動用 caffeinate -i 包裹自身（防 idle sleep），避免 user 忘記加
if [ -z "${_INSIDE_CAFFEINATE:-}" ]; then
  export _INSIDE_CAFFEINATE=1
  exec caffeinate -i "$BASH_SOURCE" "$@"
fi

main "$@"
