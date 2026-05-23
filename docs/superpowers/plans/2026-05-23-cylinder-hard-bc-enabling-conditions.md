# Cylinder Hard BC Enabling Conditions — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 部署 CEXP-017/018/019 三個 single-variable diagnostic configs（vs CEXP-016 catastrophic baseline），驗證 H1/H2/H3 三個 enabling-condition hypothesis 對 hard body BC 是否充分，並把結果寫回 `docs/cylinder_log_v2.md`。

**Architecture:** 每個 config 是 `configs/exp_cylinder_016_hard_bc_fair.toml` 的單一變數變動（H1: GradNorm 4→5 task; H2: collocation random→body_aware; H3: bc_body 64→96 + bc_outlet 0→32）。本地 commit + push → lab git pull + sed arrow_shards path → SLURM r740 partition (acmt20 RTX 3090) parallel 2-job submission → evaluate_cylinder.py 產 summary.json → rsync 回本地 → 對照 spec §4 decision tree 更新 cylinder_log_v2.md。

**Tech Stack:** Python 3.12 + torch 2.7.1+cu118, SLURM (r740 partition, acmt20 RTX 3090), uv, rsync over SSH, git.

**Spec reference:** [`docs/superpowers/specs/2026-05-23-cylinder-hard-bc-enabling-conditions-design.md`](../specs/2026-05-23-cylinder-hard-bc-enabling-conditions-design.md)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `configs/exp_cylinder_017_hard_bc_5task_gn.toml` | Create | H1 config: hard BC + 5-task GradNorm（其餘對齊 CEXP-016）|
| `configs/exp_cylinder_018_hard_bc_body_aware.toml` | Create | H2 config: hard BC + body_aware collocation（其餘對齊 CEXP-016）|
| `configs/exp_cylinder_019_hard_bc_dense_bc.toml` | Create | H3 config: hard BC + bc_body=96 + bc_outlet=32（其餘對齊 CEXP-016）|
| `docs/cylinder_log_v2.md` | Modify | `[INDEX]` table 加 3 rows + `[RECORD]` 加 3 sections + `[STATE] Surprise Findings` #6 + `[STATE] Open Questions` Stage 2 plan |

**注意 lab-only edits（不 commit 進 git）**:
- 3 個新 config 在 lab 上需要 sed 改 `arrow_shards` path：`/Users/latteine/Documents/coding/RealPDEBench` → `/home/junyi/RealPDEBench`
- 因 ForcingPrior cylinder regression workaround，3 個 config 必須含 dummy `kolmogorov_A=1e-6, k_f=2.0`（從 CEXP-016 繼承）

---

## Task 1: Create CEXP-017 config (H1: 5-task GradNorm)

**Files:**
- Create: `configs/exp_cylinder_017_hard_bc_5task_gn.toml`

- [ ] **Step 1.1: 複製 CEXP-016 config 為 CEXP-017**

```bash
cp configs/exp_cylinder_016_hard_bc_fair.toml configs/exp_cylinder_017_hard_bc_5task_gn.toml
```

- [ ] **Step 1.2: 替換 GradNorm init weights（4-task → 5-task）**

```bash
sed -i.bak 's|^gradnorm_init_weights = \[1.0, 0.01, 0.01, 0.01\]$|gradnorm_init_weights = [1.0, 0.01, 0.01, 0.01, 0.1]|' \
  configs/exp_cylinder_017_hard_bc_5task_gn.toml
rm configs/exp_cylinder_017_hard_bc_5task_gn.toml.bak
```

- [ ] **Step 1.3: 替換 artifacts_dir**

```bash
sed -i.bak 's|^artifacts_dir = .*$|artifacts_dir = "artifacts/cylinder/deeponet-cfc-cylinder-exp017-hard-bc-5task-gn"|' \
  configs/exp_cylinder_017_hard_bc_5task_gn.toml
rm configs/exp_cylinder_017_hard_bc_5task_gn.toml.bak
```

- [ ] **Step 1.4: 更新 config 頂部 header 描述（保留 single-variable rationale 不混淆讀者）**

Edit `configs/exp_cylinder_017_hard_bc_5task_gn.toml` 第 1-30 行的註解，把 file 名與 hypothesis 改：

```toml
# configs/exp_cylinder_017_hard_bc_5task_gn.toml
# CEXP-017 = Hard BC + 5-task GradNorm (H1 enabling condition test)
#
# 設計目的：
#   CEXP-016 (hard BC + 4-task GN) KE 111.6% catastrophic over-predict
#   (ke_pred/ke_ref=2.12, w_ns_u GradNorm 推 209×)。
#   Hypothesis H1: BC weight 未進 GradNorm → physics weight 失控
#   暴衝 dominate → wake NN_u 過大 → KE 高估。
#
#   本實驗：CEXP-016 + 唯一變動 gradnorm_init_weights 4→5 task
#   ([data, ns_u, ns_v, cont] → [data, ns_u, ns_v, cont, bc])。
#
# 與 CEXP-016 唯一差異：
#   gradnorm_init_weights:  [1.0, 0.01, 0.01, 0.01]
#                          → [1.0, 0.01, 0.01, 0.01, 0.1]
#
# Falsifiability gates (per spec §4):
#   KE < 10%   AND w_ns_u_final < 0.5 → ✅ H1-A: 5-task is enabling condition
#   KE 10-30%                          → 🟡 H1-B: partial, combine with H2/H3
#   KE > 30%  OR ke_pred/ke_ref > 1.5  → ❌ H1-C: GradNorm not root cause
```

- [ ] **Step 1.5: 驗證 diff vs CEXP-016 只有 4 行變動（header 註解 + gradnorm + artifacts_dir）**

Run:
```bash
diff configs/exp_cylinder_016_hard_bc_fair.toml configs/exp_cylinder_017_hard_bc_5task_gn.toml | head -50
```

Expected: 只看到 (a) 註解區塊改變、(b) `gradnorm_init_weights` 變動、(c) `artifacts_dir` 變動，**無其他差異**。

---

## Task 2: Create CEXP-018 config (H2: body-aware sampling)

**Files:**
- Create: `configs/exp_cylinder_018_hard_bc_body_aware.toml`

- [ ] **Step 2.1: 複製 CEXP-016 config 為 CEXP-018**

```bash
cp configs/exp_cylinder_016_hard_bc_fair.toml configs/exp_cylinder_018_hard_bc_body_aware.toml
```

- [ ] **Step 2.2: 替換 physics_collocation_strategy（random → body_aware）**

```bash
sed -i.bak 's|^physics_collocation_strategy = "random"$|physics_collocation_strategy = "body_aware"|' \
  configs/exp_cylinder_018_hard_bc_body_aware.toml
rm configs/exp_cylinder_018_hard_bc_body_aware.toml.bak
```

- [ ] **Step 2.3: 替換 artifacts_dir**

```bash
sed -i.bak 's|^artifacts_dir = .*$|artifacts_dir = "artifacts/cylinder/deeponet-cfc-cylinder-exp018-hard-bc-body-aware"|' \
  configs/exp_cylinder_018_hard_bc_body_aware.toml
rm configs/exp_cylinder_018_hard_bc_body_aware.toml.bak
```

- [ ] **Step 2.4: 更新 config 頂部 header 描述**

Edit `configs/exp_cylinder_018_hard_bc_body_aware.toml` 第 1-30 行的註解：

```toml
# configs/exp_cylinder_018_hard_bc_body_aware.toml
# CEXP-018 = Hard BC + body-aware collocation (H2 enabling condition test)
#
# 設計目的：
#   CEXP-016 (hard BC + random collocation) KE 111.6% catastrophic over-predict。
#   Hypothesis H2: random sampling 只 ~7% 落 near-body → boundary layer
#   gradient signal 不足 → physics residual 高 → GradNorm 推 physics 暴衝。
#
#   本實驗：CEXP-016 + 唯一變動 physics_collocation_strategy
#   "random" → "body_aware" (30% near-body distance<median + 70% uniform)。
#
# 與 CEXP-016 唯一差異：
#   physics_collocation_strategy:  "random" → "body_aware"
#
# Falsifiability gates (per spec §4):
#   KE < 10%   AND w_ns_u_final < 0.5 → ✅ H2-A: body_aware is enabling condition
#   KE 10-30%                          → 🟡 H2-B: partial, combine with H1/H3
#   KE > 30%                           → ❌ H2-C: not root cause
```

- [ ] **Step 2.5: 驗證 diff vs CEXP-016**

Run:
```bash
diff configs/exp_cylinder_016_hard_bc_fair.toml configs/exp_cylinder_018_hard_bc_body_aware.toml | head -50
```

Expected: 只 (a) 註解、(b) `physics_collocation_strategy`、(c) `artifacts_dir` 變動。

---

## Task 3: Create CEXP-019 config (H3: dense BC supervision)

**Files:**
- Create: `configs/exp_cylinder_019_hard_bc_dense_bc.toml`

- [ ] **Step 3.1: 複製 CEXP-016 config 為 CEXP-019**

```bash
cp configs/exp_cylinder_016_hard_bc_fair.toml configs/exp_cylinder_019_hard_bc_dense_bc.toml
```

- [ ] **Step 3.2: 替換 bc_body_n_points（64 → 96）**

```bash
sed -i.bak 's|^bc_body_n_points = 64$|bc_body_n_points = 96|' \
  configs/exp_cylinder_019_hard_bc_dense_bc.toml
rm configs/exp_cylinder_019_hard_bc_dense_bc.toml.bak
```

- [ ] **Step 3.3: 新增 bc_outlet_n_points = 32 line（在 bc_slip_n_points 後）**

Edit `configs/exp_cylinder_019_hard_bc_dense_bc.toml`，找到 `bc_slip_n_points = 32` 那一行，在它下方新增一行：

```toml
bc_outlet_n_points = 32   # H3: cylinder outlet (x=1) ∂u/∂x≈0 BC，CEXP-016/baseline 為 0
```

- [ ] **Step 3.4: 替換 artifacts_dir**

```bash
sed -i.bak 's|^artifacts_dir = .*$|artifacts_dir = "artifacts/cylinder/deeponet-cfc-cylinder-exp019-hard-bc-dense-bc"|' \
  configs/exp_cylinder_019_hard_bc_dense_bc.toml
rm configs/exp_cylinder_019_hard_bc_dense_bc.toml.bak
```

- [ ] **Step 3.5: 更新 config 頂部 header 描述**

Edit `configs/exp_cylinder_019_hard_bc_dense_bc.toml` 第 1-30 行的註解：

```toml
# configs/exp_cylinder_019_hard_bc_dense_bc.toml
# CEXP-019 = Hard BC + dense BC supervision (H3 enabling condition test)
#
# 設計目的：
#   CEXP-016 (hard BC + bc_body=64 + bc_outlet=0) KE 111.6% catastrophic。
#   Hypothesis H3: soft body BC 點少 → body 表面 supervision 稀疏 →
#   hard BC gate 獨自扛 boundary 約束 → wake NN_u 補償壓力大 → KE 高估。
#
#   本實驗：CEXP-016 + 兩個變動：
#     bc_body_n_points:    64 → 96 (body 表面 supervision 加密 50%)
#     bc_outlet_n_points:  0  → 32 (outlet 邊界新增 32 points)
#
# 與 CEXP-016 差異 (兩個 sub-knob，都屬 BC density axis)：
#   bc_body_n_points:    64 → 96
#   bc_outlet_n_points:  (隱式 0) → 32
#
# Falsifiability gates (per spec §4):
#   KE < 10%   AND w_ns_u_final < 0.5 → ✅ H3-A: BC density is enabling condition
#   KE 10-30%                          → 🟡 H3-B: partial, combine with H1/H2
#   KE > 30%                           → ❌ H3-C: not root cause
```

- [ ] **Step 3.6: 驗證 diff vs CEXP-016**

Run:
```bash
diff configs/exp_cylinder_016_hard_bc_fair.toml configs/exp_cylinder_019_hard_bc_dense_bc.toml | head -50
```

Expected: 只 (a) 註解、(b) `bc_body_n_points`、(c) 新增 `bc_outlet_n_points`、(d) `artifacts_dir` 變動。

---

## Task 4: Commit + push 3 個新 configs

**Files:**
- Modify: git index

- [ ] **Step 4.1: Stage 3 個 configs**

```bash
git add configs/exp_cylinder_017_hard_bc_5task_gn.toml \
        configs/exp_cylinder_018_hard_bc_body_aware.toml \
        configs/exp_cylinder_019_hard_bc_dense_bc.toml
git status | head -15
```

Expected: 3 new files staged，無其他變動。

- [ ] **Step 4.2: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(cylinder): CEXP-017/018/019 hard BC enabling condition diagnostic configs

Stage 1 of cylinder hard BC enabling conditions investigation (per
docs/superpowers/specs/2026-05-23-cylinder-hard-bc-enabling-conditions-design.md).

CEXP-016 (hard BC fair single-var) catastrophic over-predict KE 111.6%,
ke_pred/ke_ref=2.12, w_ns_u GradNorm 推 209×. 三個 single-variable diagnostic:

- CEXP-017 (H1, 60% prior): + 5-task GradNorm with BC weight in dynamic balance
- CEXP-018 (H2, 25% prior): + body_aware collocation (30% near-body)
- CEXP-019 (H3, 15% prior): + bc_body 64→96 + bc_outlet 0→32

Each vs CEXP-016 changes exactly one column (5-task GN / body_aware / BC density).
Falsifiability gates per spec §4: KE < 10% confirms hypothesis as enabling
condition; KE > 30% falsifies.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4.3: Push to origin**

```bash
git push origin main 2>&1 | tail -3
```

Expected: `main -> main` 帶 commit hash 跳出。

---

## Task 5: Lab 同步 + 套用 deployment-only edits

**Files:**
- Modify on lab (not committed): `configs/exp_cylinder_017/018/019_*.toml` 的 `arrow_shards` 路徑

- [ ] **Step 5.1: Lab git pull**

```bash
ssh lab-server 'cd pi-lnn && git pull --ff-only 2>&1 | tail -5'
```

Expected: 3 個新檔案 `created mode 100644 configs/exp_cylinder_017/018/019_*.toml`。

- [ ] **Step 5.2: Lab in-place sed 改 arrow_shards path（3 configs）**

```bash
ssh lab-server 'cd pi-lnn && sed -i "s|/Users/latteine/Documents/coding/RealPDEBench|/home/junyi/RealPDEBench|g" \
  configs/exp_cylinder_017_hard_bc_5task_gn.toml \
  configs/exp_cylinder_018_hard_bc_body_aware.toml \
  configs/exp_cylinder_019_hard_bc_dense_bc.toml'
```

- [ ] **Step 5.3: 驗證 path 已改成 lab 路徑**

```bash
ssh lab-server 'cd pi-lnn && grep -A 1 arrow_shards \
  configs/exp_cylinder_017_hard_bc_5task_gn.toml \
  configs/exp_cylinder_018_hard_bc_body_aware.toml \
  configs/exp_cylinder_019_hard_bc_dense_bc.toml | head -12'
```

Expected: 3 個 config 都顯示 `"/home/junyi/RealPDEBench/data/realpdebench/cylinder/hf_dataset/numerical/data-00000-of-00092.arrow"`。

---

## Task 6: Submit CEXP-017 + CEXP-018 並行到 SLURM

**Files:**
- 產生（lab）：`logs/exp_cylinder_017_<jobid>.out/err`, `logs/exp_cylinder_018_<jobid>.out/err`

- [ ] **Step 6.1: 確認 acmt20 有 idle GPU slot**

```bash
ssh lab-server 'sinfo -N -p r740 -o "%N %P %t %G %C" 2>&1 | head -5; echo "---squeue r740---"; squeue -p r740 2>&1 | head -10'
```

Expected: acmt20 state=mix 或 idle，目前 GPU 使用 < 2 (因 GRES=gpu:2)。

- [ ] **Step 6.2: Submit CEXP-017**

```bash
ssh lab-server 'cd pi-lnn && scripts/slurm/submit_exp.sh cylinder_017 configs/exp_cylinder_017_hard_bc_5task_gn.toml 2>&1 | tail -5'
```

Expected: `Submitted batch job <jobid_017>`。記下 jobid_017。

- [ ] **Step 6.3: Submit CEXP-018**

```bash
ssh lab-server 'cd pi-lnn && scripts/slurm/submit_exp.sh cylinder_018 configs/exp_cylinder_018_hard_bc_body_aware.toml 2>&1 | tail -5'
```

Expected: `Submitted batch job <jobid_018>`。記下 jobid_018。

- [ ] **Step 6.4: 確認兩個 job 都 RUNNING（或 PENDING 短暫等待）**

```bash
ssh lab-server 'squeue --me 2>&1 | head -10'
```

Expected: jobid_017 + jobid_018 都在 squeue，R 或 PD state。

---

## Task 7: 等 CEXP-017 + 018 完成，並行跑 evaluators，submit CEXP-019

**Files:**
- 產生（lab）：`artifacts/cylinder/deeponet-cfc-cylinder-exp{017,018}-*/cylinder-eval/summary.json`
- 產生（lab）：`logs/exp_cylinder_019_<jobid>.out/err`

- [ ] **Step 7.1: 等 017 + 018 完成（background monitor）**

```bash
ssh lab-server 'until [ $(sacct -j <jobid_017>,<jobid_018> --noheader --format=State 2>/dev/null | grep -cE "PENDING|RUNNING") -eq 0 ]; do sleep 60; done && sacct -j <jobid_017>,<jobid_018> --format=JobID,State,Elapsed,ExitCode 2>&1'
```

把 `<jobid_017>` 與 `<jobid_018>` 替換為 step 6 記下的實際 jobid。

Expected: 兩個 job 都 `COMPLETED` (Elapsed ~1:30:00-1:40:00, ExitCode 0:0)。若 FAILED 立即 abort 並 debug logs/exp_cylinder_017/018_*.err。

- [ ] **Step 7.2: 並行跑 017 + 018 evaluators（lab head node）**

```bash
ssh lab-server 'cd pi-lnn && \
  nohup .venv/bin/python -u scripts/evaluate_cylinder.py \
    --config configs/exp_cylinder_017_hard_bc_5task_gn.toml \
    --checkpoint artifacts/cylinder/deeponet-cfc-cylinder-exp017-hard-bc-5task-gn/picon_kolmogorov_final.pt \
    > logs/eval_cylinder_017.out 2>&1 & echo "eval 017 PID: $!"; \
  nohup .venv/bin/python -u scripts/evaluate_cylinder.py \
    --config configs/exp_cylinder_018_hard_bc_body_aware.toml \
    --checkpoint artifacts/cylinder/deeponet-cfc-cylinder-exp018-hard-bc-body-aware/picon_kolmogorov_final.pt \
    > logs/eval_cylinder_018.out 2>&1 & echo "eval 018 PID: $!"'
```

- [ ] **Step 7.3: Submit CEXP-019（GPU slot 已 free）**

```bash
ssh lab-server 'cd pi-lnn && scripts/slurm/submit_exp.sh cylinder_019 configs/exp_cylinder_019_hard_bc_dense_bc.toml 2>&1 | tail -5'
```

Expected: `Submitted batch job <jobid_019>`。記下 jobid_019。

- [ ] **Step 7.4: 等 evaluator 017 + 018 完成（背景跑 ~5 min each）**

```bash
ssh lab-server 'while pgrep -f "evaluate_cylinder.py.*exp_cylinder_01[78]" > /dev/null; do sleep 30; done; ls artifacts/cylinder/deeponet-cfc-cylinder-exp01{7,8}-*/cylinder-eval*/summary.json 2>&1'
```

Expected: 2 個 `summary.json` 路徑列出。

---

## Task 8: 等 CEXP-019 完成 + 跑 evaluator

**Files:**
- 產生（lab）：`artifacts/cylinder/deeponet-cfc-cylinder-exp019-*/cylinder-eval/summary.json`

- [ ] **Step 8.1: 等 019 完成**

```bash
ssh lab-server 'until [ $(sacct -j <jobid_019> --noheader --format=State 2>/dev/null | grep -cE "PENDING|RUNNING") -eq 0 ]; do sleep 60; done && sacct -j <jobid_019> --format=JobID,State,Elapsed,ExitCode 2>&1'
```

Expected: 019 `COMPLETED` Elapsed ~1:30:00-1:40:00。

- [ ] **Step 8.2: 跑 019 evaluator**

```bash
ssh lab-server 'cd pi-lnn && nohup .venv/bin/python -u scripts/evaluate_cylinder.py \
    --config configs/exp_cylinder_019_hard_bc_dense_bc.toml \
    --checkpoint artifacts/cylinder/deeponet-cfc-cylinder-exp019-hard-bc-dense-bc/picon_kolmogorov_final.pt \
    > logs/eval_cylinder_019.out 2>&1 & echo "eval 019 PID: $!"'
```

- [ ] **Step 8.3: 等 evaluator 019 完成**

```bash
ssh lab-server 'while pgrep -f "evaluate_cylinder.py.*exp_cylinder_019" > /dev/null; do sleep 30; done; ls artifacts/cylinder/deeponet-cfc-cylinder-exp019-*/cylinder-eval*/summary.json 2>&1'
```

Expected: 1 個 `summary.json` 路徑列出。

---

## Task 9: Rsync 三組 artifacts 回本地

**Files:**
- Create（本地）：`artifacts/cylinder/deeponet-cfc-cylinder-exp017-hard-bc-5task-gn/`
- Create（本地）：`artifacts/cylinder/deeponet-cfc-cylinder-exp018-hard-bc-body-aware/`
- Create（本地）：`artifacts/cylinder/deeponet-cfc-cylinder-exp019-hard-bc-dense-bc/`

- [ ] **Step 9.1: Rsync 3 個 artifacts 並行**

```bash
rsync -avz lab-server:/home/junyi/pi-lnn/artifacts/cylinder/deeponet-cfc-cylinder-exp017-hard-bc-5task-gn/ \
  /Users/latteine/Documents/coding/pi-lnn/artifacts/cylinder/deeponet-cfc-cylinder-exp017-hard-bc-5task-gn/ \
  2>&1 | tail -5 &
rsync -avz lab-server:/home/junyi/pi-lnn/artifacts/cylinder/deeponet-cfc-cylinder-exp018-hard-bc-body-aware/ \
  /Users/latteine/Documents/coding/pi-lnn/artifacts/cylinder/deeponet-cfc-cylinder-exp018-hard-bc-body-aware/ \
  2>&1 | tail -5 &
rsync -avz lab-server:/home/junyi/pi-lnn/artifacts/cylinder/deeponet-cfc-cylinder-exp019-hard-bc-dense-bc/ \
  /Users/latteine/Documents/coding/pi-lnn/artifacts/cylinder/deeponet-cfc-cylinder-exp019-hard-bc-dense-bc/ \
  2>&1 | tail -5 &
wait
echo "all rsync done"
```

Expected: 3 個 rsync 完成（總 size 估 30-50 MB each，<1 min）。

- [ ] **Step 9.2: 驗證 summary.json 都在本地**

```bash
ls artifacts/cylinder/deeponet-cfc-cylinder-exp01{7,8,9}-*/cylinder-eval*/summary.json 2>&1
```

Expected: 3 個 summary.json 列出。

---

## Task 10: 抽取 metrics + 對照 spec §4 decision tree 判讀

**Files:**
- 讀取：`artifacts/cylinder/deeponet-cfc-cylinder-exp{017,018,019}-*/cylinder-eval*/summary.json`
- 讀取：`artifacts/cylinder/deeponet-cfc-cylinder-exp{017,018,019}-*/loss_log.json`（取 GradNorm w_ns_u final）

- [ ] **Step 10.1: 抽取 6 個關鍵 metrics 給每 config**

```bash
for n in 017 018 019; do
  echo "=== CEXP-${n} ==="
  jq '{ke_rel_err_mean, ke_rel_err_late, ke_pred_mean, ke_ref_mean, u_rmse_mean, v_rmse_mean, omega_rmse_mean, div_l2_mean}' \
    artifacts/cylinder/deeponet-cfc-cylinder-exp${n}-*/cylinder-eval*/summary.json
done
```

Expected: 3 個 JSON output，每個含 8 個 metric。

- [ ] **Step 10.2: 抽取每個 config 的 w_ns_u final value**

```bash
for n in 017 018 019; do
  echo "=== CEXP-${n} w_ns_u_final ==="
  tail -100 artifacts/cylinder/deeponet-cfc-cylinder-exp${n}-*/loss_log.json 2>&1 | tail -20
done
```

或者更精準（從訓練 stdout log）：

```bash
ssh lab-server 'for n in 017 018 019; do echo "=== CEXP-${n} final step weights ==="; grep "^10000" pi-lnn/logs/exp_cylinder_${n}_*.out | tail -1; done'
```

Expected: 三行 `10000 ... L_data ... w_ns_u <value> w_ns_v <value> w_cont <value> ...`。

- [ ] **Step 10.3: 對照 spec §4 decision tree 判讀每個 config 的 outcome（A/B/C）**

用 Read 工具讀 `docs/superpowers/specs/2026-05-23-cylinder-hard-bc-enabling-conditions-design.md` 的 §4 段，對照剛抽出的 metrics 寫判讀結論（自己心算或 inline 寫成 markdown 表，供 Task 11 使用）：

```
CEXP-017 (H1): KE rel-err = X.X%, w_ns_u_final = Y.Y → Outcome: H1-A / H1-B / H1-C
CEXP-018 (H2): ...
CEXP-019 (H3): ...
```

並對應 spec §4「Multi-config outcomes 判讀」表得出 Stage 2 strategy:
- 1 個 A → Stage 2 用該 condition + axis sweep
- 2-3 個 A → Pick 最簡單 condition
- 全 🟡 → Stage 1b: pairwise stacking
- 全 ❌ → Re-diagnose

---

## Task 11: 更新 cylinder_log_v2.md [INDEX] 表

**Files:**
- Modify: `docs/cylinder_log_v2.md` `[INDEX] Cylinder Active` 表

- [ ] **Step 11.1: 找到 [INDEX] Cylinder Active 表的 CEXP-016 row 之下，插入 3 個新 rows**

Edit `docs/cylinder_log_v2.md`，找到下面的表（在 `## [INDEX] Cylinder Active` 段內），在 CEXP-016 row 之後 + CEXP-001 row 之前插入：

```markdown
| **CEXP-017** | (status from §4 outcome) | Re=10031, **hard BC + 5-task GradNorm**（H1）| (KE %) | (ratio) | (ω) | (div) | 10k | (一句結論 per §4 outcome) |
| **CEXP-018** | (status) | Re=10031, **hard BC + body-aware collocation**（H2）| (KE %) | (ratio) | (ω) | (div) | 10k | (一句結論) |
| **CEXP-019** | (status) | Re=10031, **hard BC + dense BC (bc_body 96 + bc_outlet 32)**（H3）| (KE %) | (ratio) | (ω) | (div) | 10k | (一句結論) |
```

把 `(status)` 替換為 `ACTIVE_REFERENCE`（KE < 30%）或 `NEGATIVE_RESULT`（KE > 30%）;
把 `(KE %)` 等替換為 Task 10 抽出的實際數字。

- [ ] **Step 11.2: 同時 update CEXP-016 row status 為 `INCONCLUSIVE → 已由 CEXP-017/018/019 後續判讀`**

找 CEXP-016 那一 row 的最後欄位，加註解 reference Stage 1 結論。

---

## Task 12: 更新 cylinder_log_v2.md [RECORD] + [STATE] Surprise Findings + Open Questions

**Files:**
- Modify: `docs/cylinder_log_v2.md` `[RECORD]` 段 + `[STATE] Surprise Findings` + `[STATE] Open Questions`

- [ ] **Step 12.1: [RECORD] 新增 3 個 detailed sections（CEXP-017/018/019）**

在 `[RECORD]` 大段內，CEXP-013 之後新增 3 個 detail rows（模仿 CEXP-013 等既有格式）：

```markdown
### CEXP-017：Hard BC + 5-task GradNorm（H1）

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_017_hard_bc_5task_gn.toml` |
| Artifact | `artifacts/cylinder/deeponet-cfc-cylinder-exp017-hard-bc-5task-gn/` |
| Checkpoint | `picon_kolmogorov_final.pt`（step 10000） |
| KE rel-err mean / late | (Task 10 值) |
| u / v RMSE | (Task 10 值) |
| ω RMSE | (Task 10 值) |
| div L2 | (Task 10 值) |
| ke_pred / ke_ref | (Task 10 值) |
| GradNorm w_ns_u final | (Task 10 值) |
| 設計變動 | gradnorm_init_weights 4-task `[1.0, 0.01, 0.01, 0.01]` → 5-task `[1.0, 0.01, 0.01, 0.01, 0.1]` |
| 結論 | (Per §4 H1-A/B/C outcome) |

### CEXP-018：Hard BC + body-aware collocation（H2）

| 項目 | 值 |
|---|---|
... (同樣格式) ...

### CEXP-019：Hard BC + dense BC supervision（H3）

| 項目 | 值 |
|---|---|
... (同樣格式) ...
```

- [ ] **Step 12.2: [STATE] Surprise Findings #6 新增（Stage 1 結論）**

在 `[STATE] Surprise Findings` 大段最後（CEXP-016 Finding #5 後）新增：

```markdown
### Finding 6 — Stage 1 hard BC enabling conditions diagnostic（CEXP-017/018/019, 2026-05-23）

CEXP-016 catastrophic fail (KE 111.6%) 的 3 個 single-variable enabling-condition hypothesis 結果:

| Hypothesis | Config | Result | Outcome |
|---|---|---|---|
| H1: 5-task GradNorm | CEXP-017 | KE X.X%, w_ns_u Y.Y | (A/B/C per §4) |
| H2: body-aware sampling | CEXP-018 | KE X.X% | (A/B/C) |
| H3: dense BC supervision | CEXP-019 | KE X.X% | (A/B/C) |

**Multi-config 判讀 (per spec §4)**: (1 個 A / 2-3 個 A / 全 🟡 / 全 ❌ 之一)
**結論**: (per outcome) Hard BC 的 minimum enabling condition 為 (X / X+Y / 待 Stage 1b)。
**Stage 2 plan**: (per outcome 對應 strategy)
```

- [ ] **Step 12.3: [STATE] Open Questions update**

找 `[STATE] Open Questions` 表，把 CEXP-016 對應的 row 改成 `已由 Stage 1 解析 (per Finding 6)`，並新增一行 Stage 2 plan 描述（per Task 10.3 結論）。

---

## Task 13: Commit + push v2 log update + 最終 sanity check

**Files:**
- Modify: git index

- [ ] **Step 13.1: Stage v2 log update**

```bash
git add docs/cylinder_log_v2.md
git status | head -10
```

Expected: `docs/cylinder_log_v2.md` modified staged，no other changes。

- [ ] **Step 13.2: Commit**

```bash
git commit -m "$(cat <<'EOF'
docs(v2): Stage 1 hard BC enabling conditions results (CEXP-017/018/019)

CEXP-017 (H1: 5-task GradNorm): KE X.X% [outcome A/B/C]
CEXP-018 (H2: body_aware):       KE X.X% [outcome A/B/C]
CEXP-019 (H3: dense BC):         KE X.X% [outcome A/B/C]

Multi-config 判讀 per spec §4: <pattern>.
Minimum enabling condition: <condition>.
Stage 2 plan: <plan>.

[INDEX] 加 3 rows, [RECORD] 加 3 detail sections,
[STATE] Surprise Findings 新增 #6,
[STATE] Open Questions update CEXP-016 row + Stage 2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

把 placeholder（X.X%, A/B/C, <pattern> 等）替換為 Task 10 實際結論。

- [ ] **Step 13.3: Push**

```bash
git push origin main 2>&1 | tail -3
```

Expected: `main -> main`。

- [ ] **Step 13.4: 最終 sanity check — v2 log 結構完整**

```bash
grep -n "^##" docs/cylinder_log_v2.md | head -20
echo "---CEXP-017/018/019 出現次數---"
grep -c "CEXP-01[789]" docs/cylinder_log_v2.md
```

Expected:
- v2 log 仍 ~340-400 行（plus 新增段）
- CEXP-017/018/019 各出現 ≥ 3 次（INDEX + RECORD + Surprise Findings）

---

## Spec Coverage Self-Check（implementer 跑完 Task 13 後）

對照 spec §1-5 確認所有 requirement 都有對應 task：

| Spec section | Plan task | OK? |
|---|---|---|
| §1 Goal: 識別 hard BC enabling conditions | Task 1-3 (3 configs) + Task 10 (判讀) | ✅ |
| §1 Success criteria: 至少 1 config KE < 10% | Task 10 outcome check + Task 12 結論 | ✅ |
| §2 H1/H2/H3 hypothesis | Task 1/2/3 各對應 | ✅ |
| §3 Config Matrix (3 single-var) | Task 1-3 | ✅ |
| §4 Decision tree thresholds | Task 10 (per-config outcome) + Task 12 (multi-config judgment) | ✅ |
| §5 Workflow（commit → push → lab pull → submit → eval → rsync → judge → update log）| Task 4-13 完整覆蓋 | ✅ |
| §5 Prerequisites (forcing dummy, arrow_shards path) | 已由 CEXP-016 deployment 處理；Task 5 sed 維持 | ✅ |
| §5 Out-of-scope | 不出現在任何 task | ✅ |

---

## Stop Loss 條件（implementer 觀察）

訓練中**不** early stop。但若：

- Task 6/7 submit 即 FAILED (jobid 0:0 不出現, error in .err) → 立即 debug，可能 src 端 cylinder pipeline regression（同 CEXP-016 第一次 forcing prior fail）
- Task 7/8 完成但 evaluator 拋 exception → 檢查 `summary.json` 是否被產出，rsync 後手動讀 raw metrics
- Task 10 三個 config 全 catastrophic (KE > 30%) → **不**進入 Task 11-13 結論段，改寫 v2 log 為 "Stage 1 全 falsified，回 brainstorming"
