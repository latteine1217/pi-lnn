#!/usr/bin/env bash
# scripts/eval_alt_sensors.sh
#
# What:
#   用 EXP-094 trained model + 同一 DNS + 不同 placement 的 sensor，跑 evaluator，
#   收集 metrics 對比 training-placement baseline，回答「operator 對 unseen
#   sensor placement 是否 generalize」。
#
# Why:
#   既有 evaluator 用訓練 sensor → 只能驗 reconstruction on seen sensor，
#   不能回答 generalization。本 runner 替換 sensor input、不重訓 model。
#
# How:
#   1. cp EXP-094 config 3 份，sed 替換 sensor_jsons / sensor_npzs 為 alt placement
#   2. 跑 evaluator 用 trained final.pt + alt config
#   3. 抓 summary.json 主要 metric 進 TSV
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_CONFIG="configs/exp_094_b3_seed2.toml"
CHECKPOINT="artifacts/kolmogorov/stable/exp094-b3-seed2/picon_kolmogorov_final.pt"
ALT_OUT_ROOT="artifacts/kolmogorov/stable/exp094-b3-seed2/alt_sensor_eval"
mkdir -p "$ALT_OUT_ROOT"

ALT_METRICS_TSV="$ALT_OUT_ROOT/metrics.tsv"
if [ ! -f "$ALT_METRICS_TSV" ]; then
  printf "placement\tu_L2\tv_L2\tomega_L2\tKE_rel_err\tek_ratio_last\tdiv_L2\n" > "$ALT_METRICS_TSV"
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1

run_alt_eval () {
  local pseed="$1"     # placement seed: 1, 2, 3
  local alt_json="data/kolmogorov_sensors/re10000_alt/sensors_random_K100_N256_t0-5_si100_pseed${pseed}.json"
  local alt_npz="data/kolmogorov_sensors/re10000_alt/sensors_random_K100_N256_t0-5_si100_pseed${pseed}_dns_values.npz"
  local alt_config="$ALT_OUT_ROOT/eval_config_random_pseed${pseed}.toml"
  local eval_dir="$ALT_OUT_ROOT/random_pseed${pseed}"
  local summary="$eval_dir/summary.json"
  local log="$ALT_OUT_ROOT/eval_pseed${pseed}.log"

  echo ""
  echo "======================================================================"
  echo "[alt-eval] random placement seed=$pseed  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "  sensor JSON: $alt_json"
  echo "  sensor NPZ : $alt_npz"
  echo "======================================================================"

  # 1. cp base config + 替換 sensor path（其他 model 架構 / loss / 訓練設定都不變）
  cp "$BASE_CONFIG" "$alt_config"
  # sed 在 macOS 用 BSD 風格；先把整行 sensor_jsons / sensor_npzs 替成 single-line
  python3 - <<PY
import re, pathlib
cfg = pathlib.Path("$alt_config").read_text()
cfg = re.sub(
    r'sensor_jsons\s*=\s*\[\s*\n\s*"[^"]*"\s*,?\s*\n\s*\]',
    'sensor_jsons = [\n  "$alt_json",\n]',
    cfg, count=1,
)
cfg = re.sub(
    r'sensor_npzs\s*=\s*\[\s*\n\s*"[^"]*"\s*,?\s*\n\s*\]',
    'sensor_npzs = [\n  "$alt_npz",\n]',
    cfg, count=1,
)
pathlib.Path("$alt_config").write_text(cfg)
print(f"[cfg] replaced sensor paths in $alt_config")
PY

  # 2. 跑 evaluator
  mkdir -p "$eval_dir"
  uv run python -u scripts/evaluate_deeponet_cfc.py \
    --config "$alt_config" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$eval_dir" \
    --device mps \
    2>&1 | tee "$log"

  # 3. 抓 metric
  if [ -f "$summary" ]; then
    local u_l2 v_l2 omega_l2 ke ek_ratio div_l2
    u_l2=$(jq -r '.u_rel_l2_mean // "null"' "$summary")
    v_l2=$(jq -r '.v_rel_l2_mean // "null"' "$summary")
    omega_l2=$(jq -r '.omega_rel_l2_mean // "null"' "$summary")
    ke=$(jq -r '.ke_rel_err_mean // "null"' "$summary")
    ek_ratio=$(jq -r '.ek_ratio_kf_last // "null"' "$summary")
    div_l2=$(jq -r '.div_l2_mean // "null"' "$summary")
    printf "random_pseed%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$pseed" "$u_l2" "$v_l2" "$omega_l2" "$ke" "$ek_ratio" "$div_l2" >> "$ALT_METRICS_TSV"
    echo "[alt-eval] random_pseed${pseed} → $ALT_METRICS_TSV"
  else
    echo "[alt-eval] WARN: $summary 不存在；skip metric collection" >&2
  fi
}

main () {
  echo "[main] start at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[main] base config:    $BASE_CONFIG"
  echo "[main] checkpoint:     $CHECKPOINT"
  echo "[main] alt out root:   $ALT_OUT_ROOT"

  for pseed in 1 2 3; do
    run_alt_eval "$pseed"
  done

  echo ""
  echo "[main] === ALL ALT-SENSOR EVAL DONE at $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo ""
  echo "=== Metrics ==="
  cat "$ALT_METRICS_TSV"
}

main "$@"
