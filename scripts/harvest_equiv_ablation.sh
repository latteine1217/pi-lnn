#!/usr/bin/env bash
# scripts/harvest_equiv_ablation.sh
#
# What:
#   EXP-281/282/283 對等 ablation (B0/B1/B2 × n=5 @ 20k) 跑完後的收割：
#   ① 檢查 15 個 slurm job 是否全部結束；② 在 lab 對每個 final.pt 跑 eval (cuda)；
#   ③ rsync series.npz 回本地；④ 彙總 KE/u/v/ω 的 per-architecture mean±std。
#
# Why:
#   lab slurm job 非本地背景 task，無法被 harness 自動喚醒收割。此腳本可重複執行：
#   job 未跑完則回報並退出（exit 0），跑完才進行 eval + 彙總。
#
# Usage (本地執行):
#   bash scripts/harvest_equiv_ablation.sh
#
# 前置: lab 已 git pull 到含 --export_fields 的 evaluate_deeponet_cfc.py (commit 84305b9+)。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
LAB="lab-server"

# (eval_id : config 相對路徑) — 與 submit 時一致
SPECS=(
  "281:configs/exp_281_b0_les_T50_20k.toml"
  "281b:configs/exp_281b_b0_les_T50_20k_seed1.toml"
  "281c:configs/exp_281c_b0_les_T50_20k_seed2.toml"
  "281d:configs/exp_281d_b0_les_T50_20k_seed3.toml"
  "281e:configs/exp_281e_b0_les_T50_20k_seed4.toml"
  "282:configs/exp_282_b1_les_T50_20k.toml"
  "282b:configs/exp_282b_b1_les_T50_20k_seed1.toml"
  "282c:configs/exp_282c_b1_les_T50_20k_seed2.toml"
  "282d:configs/exp_282d_b1_les_T50_20k_seed3.toml"
  "282e:configs/exp_282e_b1_les_T50_20k_seed4.toml"
  "283:configs/exp_283_b2_les_T50_20k.toml"
  "283b:configs/exp_283b_b2_les_T50_20k_seed1.toml"
  "283c:configs/exp_283c_b2_les_T50_20k_seed2.toml"
  "283d:configs/exp_283d_b2_les_T50_20k_seed3.toml"
  "283e:configs/exp_283e_b2_les_T50_20k_seed4.toml"
)

echo "=== [1/4] 檢查 lab queue 中是否仍有 exp_28x job ==="
PENDING=$(ssh "$LAB" 'squeue -u junyi -h -o "%j" | grep -cE "^exp_28[123]" || true')
if [ "${PENDING:-0}" -gt 0 ]; then
  echo "[HOLD] 仍有 ${PENDING} 個 exp_28x job 在 queue/running；稍後再執行本腳本。"
  ssh "$LAB" 'squeue -u junyi -o "%.7i %.10j %.2t %.11M" | grep -E "JOBID|exp_28[123]"' || true
  exit 0
fi
echo "[OK] 無 exp_28x job 在 queue，視為全部結束。"

echo "=== [2/4] lab 對 15 個 final.pt 跑 eval (--export_arrays, cuda) ==="
ssh "$LAB" "cd ~/pi-lnn && for spec in ${SPECS[*]}; do
    id=\${spec%%:*}; cfg=\${spec##*:};
    adir=\$(grep '^artifacts_dir' \"\$cfg\" | sed 's/.*= *\"//; s/\".*//');
    ckpt=\"\$adir/picon_kolmogorov_final.pt\";
    if [ ! -f \"\$ckpt\" ]; then echo \"[WARN] 缺 checkpoint: \$ckpt (跳過)\"; continue; fi
    echo \"--- eval EXP-\$id ---\";
    uv run python scripts/evaluate_deeponet_cfc.py --config \"\$cfg\" \
      --checkpoint \"\$ckpt\" --output-dir \"artifacts/eval_\$id\" \
      --export_arrays --device cuda 2>&1 | tail -2;
done"

echo "=== [3/4] rsync series.npz 回本地 ==="
for spec in "${SPECS[@]}"; do
  id="${spec%%:*}"
  mkdir -p "artifacts/eval_$id"
  rsync -avzP "$LAB:~/pi-lnn/artifacts/eval_$id/series.npz" "artifacts/eval_$id/" 2>/dev/null \
    || echo "[WARN] rsync 失敗: eval_$id"
done

echo "=== [4/4] 本地彙總 per-architecture mean±std ==="
uv run python - "${SPECS[@]}" <<'PYEOF'
import sys, pathlib, json
import numpy as np

specs = sys.argv[1:]
groups = {"B0 (281)": [], "B1 (282)": [], "B2 (283)": []}
for spec in specs:
    eid = spec.split(":")[0]
    p = pathlib.Path(f"artifacts/eval_{eid}/series.npz")
    if not p.exists():
        print(f"[skip] {eid}: series.npz 不存在")
        continue
    d = np.load(p)
    ke = float(d["KE_rel_err"].mean()) * 100      # time-mean KE MAPE %
    u = float(d["u_rel_L2"].mean()) * 100
    v = float(d["v_rel_L2"].mean()) * 100
    w = float(d["omega_rel_L2"].mean()) * 100
    key = "B0 (281)" if eid.startswith("281") else "B1 (282)" if eid.startswith("282") else "B2 (283)"
    groups[key].append((eid, ke, u, v, w))

print("\n=== EXP-281/282/283 對等 ablation 收割 (20k × n=5, time-mean) ===")
print(f"{'arch':<10}{'n':>3}  {'KE%':>14}  {'u-L2%':>14}  {'v-L2%':>14}  {'ω-L2%':>14}")
summary = {}
for k, rows in groups.items():
    if not rows:
        print(f"{k:<10}  (無資料)"); continue
    arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    m, s = arr.mean(0), arr.std(0, ddof=1) if len(rows) > 1 else np.zeros(4)
    print(f"{k:<10}{len(rows):>3}  {m[0]:>7.2f}±{s[0]:<5.2f}  {m[1]:>7.2f}±{s[1]:<5.2f}  "
          f"{m[2]:>7.2f}±{s[2]:<5.2f}  {m[3]:>7.2f}±{s[3]:<5.2f}")
    summary[k] = {"n": len(rows), "KE_mean": m[0], "KE_std": s[0],
                  "u_mean": m[1], "v_mean": m[2], "omega_mean": m[3],
                  "seeds": [r[0] for r in rows]}
print("\n對照 B3 (EXP-245, 20k × n=5): KE 5.71 ± 0.11 %  (Table 4.3)")
pathlib.Path("artifacts/equiv_ablation_summary.json").write_text(json.dumps(summary, indent=2))
print("[wrote] artifacts/equiv_ablation_summary.json")
PYEOF

echo "=== 完成。請依結果更新 thesis Table 4.6 + experiment_log_v2 §13 ==="
