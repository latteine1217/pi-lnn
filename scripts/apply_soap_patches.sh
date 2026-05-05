#!/usr/bin/env bash
# scripts/apply_soap_patches.sh
#
# What
# ----
# 把 patches/soap-*.patch 套用到 SOAP submodule。
# Idempotent — 已 apply 的 patch 會被跳過。
#
# Why
# ---
# SOAP 是上游 third-party submodule（github.com/nikhilvyas/SOAP），
# 沒有 push 權；我們的 MPS 顯式 CPU 搬移優化以 patch 形式維護在外層 repo，
# 由本 script 在本地 SOAP working tree 套用。
#
# Usage
# -----
#   ./scripts/apply_soap_patches.sh           # apply
#   ./scripts/apply_soap_patches.sh --revert  # revert
#   ./scripts/apply_soap_patches.sh --check   # check apply 狀態，不修改檔案
#
# 重新 init / update submodule 後請重跑本 script。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOAP_DIR="$ROOT/SOAP"
PATCH_DIR="$ROOT/patches"

if [[ ! -d "$SOAP_DIR" ]]; then
  echo "[ERROR] SOAP submodule not found at $SOAP_DIR" >&2
  echo "        Run: git submodule update --init --recursive" >&2
  exit 1
fi

shopt -s nullglob
patches=("$PATCH_DIR"/soap-*.patch)
shopt -u nullglob

if [[ ${#patches[@]} -eq 0 ]]; then
  echo "[INFO] no soap-*.patch files in $PATCH_DIR — nothing to do."
  exit 0
fi

mode="${1:-apply}"

case "$mode" in
  apply)
    for p in "${patches[@]}"; do
      name="$(basename "$p")"
      # --check 先試 forward apply：成功 → 還沒套；失敗 → 可能已套或衝突
      if (cd "$SOAP_DIR" && git apply --check "$p") 2>/dev/null; then
        (cd "$SOAP_DIR" && git apply "$p")
        echo "[OK]      applied  $name"
      elif (cd "$SOAP_DIR" && git apply --check --reverse "$p") 2>/dev/null; then
        echo "[SKIP]    $name 已套用過（reverse-check passes）"
      else
        echo "[ERROR]   $name 套用失敗：既無法 apply 也無法 reverse" >&2
        echo "          手動處理或執行 --revert 後重試" >&2
        exit 2
      fi
    done
    ;;

  --revert|revert)
    # 反向：從最後一個 patch 開始 revert
    for ((i=${#patches[@]}-1; i>=0; i--)); do
      p="${patches[$i]}"
      name="$(basename "$p")"
      if (cd "$SOAP_DIR" && git apply --check --reverse "$p") 2>/dev/null; then
        (cd "$SOAP_DIR" && git apply --reverse "$p")
        echo "[OK]      reverted $name"
      else
        echo "[SKIP]    $name 不在已套用狀態"
      fi
    done
    ;;

  --check|check)
    for p in "${patches[@]}"; do
      name="$(basename "$p")"
      if (cd "$SOAP_DIR" && git apply --check --reverse "$p") 2>/dev/null; then
        echo "[APPLIED]  $name"
      elif (cd "$SOAP_DIR" && git apply --check "$p") 2>/dev/null; then
        echo "[NOT-YET]  $name"
      else
        echo "[CONFLICT] $name (既無法 forward 也無法 reverse)"
      fi
    done
    ;;

  *)
    echo "Usage: $0 [apply|--revert|--check]" >&2
    exit 64
    ;;
esac
