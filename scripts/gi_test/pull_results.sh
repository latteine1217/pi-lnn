#!/bin/bash
# scripts/gi_test/pull_results.sh
#
# What:
#   從 lab-server:~/gi_test_re10000/ 拉所有 GI test 產物到 pi-lnn 本地。
#   含 .npy 模擬輸出 + .log 紀錄 + sweep 控制檔。
#
# Why:
#   GI test 在 lab-server (48-core Xeon) 執行，pi-lnn (Mac local) 做 analysis。
#   每跑完一個 N 就執行此 script 拉新檔（rsync 增量），analysis script 可逐步分析。
#
# Usage:
#   bash scripts/gi_test/pull_results.sh
#
set -euo pipefail

REMOTE_DIR="home-gpu:~/gi_test_re10000/"
LOCAL_DIR="$(cd "$(dirname "$0")/../.." && pwd)/data/dns/gi_test_re10000/"

mkdir -p "$LOCAL_DIR"
echo "=== Pulling from $REMOTE_DIR → $LOCAL_DIR ==="
rsync -avzP \
    --include='*.npy' \
    --include='*.log' \
    --include='*.out' \
    --include='*.sh' \
    --exclude='__pycache__/' \
    --exclude='.venv/' \
    --exclude='pyproject.toml' \
    --exclude='uv.lock' \
    --exclude='generate_kolmogorov_dns_fp64.py' \
    "$REMOTE_DIR" "$LOCAL_DIR"
echo ""
echo "=== Local files ==="
ls -la "$LOCAL_DIR" | grep -E '\.(npy|log)$' || true
