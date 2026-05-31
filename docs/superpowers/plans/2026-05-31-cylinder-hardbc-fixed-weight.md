# CEXP-037 Hard BC + Fixed Weight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 驗證 Finding #6 歸因——關掉 GradNorm 改用固定權重後，hard BC gate 是否不再爆炸（KE 不再 >100%）。

**Architecture:** Config-only 變更（無 src 改動）。CEXP-002 base + `use_hard_body_bc=true` + `use_gradnorm=false` + `bc_body_n_points=0`，固定 loss 權重。

**Tech Stack:** Python, PyTorch, SLURM (r740, RTX 3090), uv, lab-server SSH

---

## Context

| 項目 | 值 |
|---|---|
| Spec | [docs/superpowers/specs/2026-05-31-cylinder-hardbc-fixed-weight-design.md](../specs/2026-05-31-cylinder-hardbc-fixed-weight-design.md) |
| Base | CEXP-002（`configs/exp_cylinder_002_k100_bc.toml`, KE 3.54%, div 1.14） |
| 對照 | CEXP-016（hard BC + GradNorm, KE 111.6%, w_ns_u 2.09）|
| 變更 | `use_hard_body_bc` f→t, `use_gradnorm` t→f, `bc_body_n_points` 64→0 |
| Falsifiability | KE <20% ✅ Finding #6 正確 / KE >100% ❌ 歸因錯 / div<1.14 = incompressibility 改善 |

---

### Task 1: Create CEXP-037 config

**Files:**
- Create: `configs/exp_cylinder_037_hardbc_fixed_weight.toml`

- [ ] **Step 1: 建立 config**

```toml
# configs/exp_cylinder_037_hardbc_fixed_weight.toml
# CEXP-037 = CEXP-002 + hard BC gate + fixed weight (no GradNorm) + bc_body=0
#
# 設計目的（Finding #6 乾淨對照）：
#   CEXP-016 (hard BC + GradNorm) KE 111.6%, w_ns_u 推爆到 2.09。
#   Finding #6 歸因：hard BC gate + GradNorm 優化不相容。
#   本實驗：移除 GradNorm 改固定權重，驗證 hard BC 單獨是否可行。
#
# 變更 vs CEXP-002：
#   use_hard_body_bc: false → true   (Sukumar 2022 output gate, body 內 u=v=0 機器精度)
#   use_gradnorm:     true  → false  (Finding #6 元兇，改固定權重)
#   bc_body_n_points: 64    → 0      (gate 已管 body，Finding #8 一區一約束)
#
# Falsifiability：
#   KE < 20%   → ✅ Finding #6 正確：GradNorm 是 hard BC 失敗元兇
#   KE 20-100% → 🟡 部分原因
#   KE > 100% + L_phys 爆漲 → ❌ hard BC 路線徹底封閉

[train]
dataset_type = "cylinder"

sensor_jsons = [
  "data/cylinder_sensors/sensors_qrpivot_K100_cylinder_Re10031.json",
]
sensor_npzs = [
  "data/cylinder_sensors/sensors_qrpivot_K100_cylinder_Re10031_values.npz",
]
arrow_shards = [
  "/Users/latteine/Documents/coding/RealPDEBench/data/realpdebench/cylinder/hf_dataset/numerical/data-00000-of-00092.arrow",
]
re_values = [10031.0]
observed_sensor_channels = ["u", "v"]
sensor_subsample = 20

fourier_harmonics = 16
fourier_embed_dim = 128
use_periodic_domain = false
use_physics_denormalization = true

use_temporal_anchor = true
T_total = 20.0
temporal_anchor_harmonics = 2

d_model = 256
d_time = 16
num_spatial_cfc_layers = 1
num_temporal_cfc_layers = 1
num_token_attention_layers = 2
token_attention_heads = 4
num_query_mlp_layers = 1
query_mlp_hidden_dim = 256
output_head_gain = 1.0
operator_rank = 256

domain_length = 1.0
kolmogorov_k_f = 0.0
kolmogorov_A = 0.0

data_loss_weight = 1.0
physics_loss_weight = 0.01
physics_loss_warmup_steps = 0
physics_loss_ramp_steps = 0
continuity_weight = 1.0
poisson_loss_weight = 0.0

bc_loss_weight = 0.1
bc_inflow_u = 0.33
bc_n_points = 64
bc_body_n_points = 0      # ← gate 已管 body（Finding #8）
bc_slip_n_points = 32

t_early_weight = 10.0
t_early_threshold = 0.05
physics_collocation_strategy = "random"

# ===== CEXP-037 變動 =====
use_hard_body_bc = true   # ← Sukumar gate（架構級 geometry enforcement）
use_gradnorm = false      # ← 移除 GradNorm（Finding #6 元兇），改固定權重
# ==========================

time_marching = true
time_marching_start = 0.5
time_marching_warmup = 0.3

iterations = 10000
num_physics_points = 64
learning_rate = 1e-3
weight_decay = 0.0
lr_schedule = "soap"
use_schedule_free = true
soap_betas = [0.9, 0.999]
soap_precondition_frequency = 2
lr_warmup_steps = 2000
soap_use_step_decay = true
lr_decay_steps = 2000
lr_decay_gamma = 0.9
min_learning_rate = 1e-6
max_grad_norm = 1.0

use_sensor_physics = false
num_sensor_physics_time_samples = 1
sensor_physics_start_step = 1000

checkpoint_period = 500
seed = 42
device = "cuda"

artifacts_dir = "artifacts/cylinder/deeponet-cfc-cylinder-exp037-hardbc-fixed-weight"
```

- [ ] **Step 2: 驗證 vs CEXP-002 只有預期欄位不同**

Run:
```bash
diff <(grep -v "^#" configs/exp_cylinder_002_k100_bc.toml | grep -v "^$" | sed 's/  *#.*//') \
     <(grep -v "^#" configs/exp_cylinder_037_hardbc_fixed_weight.toml | grep -v "^$" | sed 's/  *#.*//')
```
Expected: 差異為 `bc_body_n_points`(64→0)、`use_hard_body_bc`(無→true)、`use_gradnorm`(true→false)、`device`(mps→cuda)、`artifacts_dir`。

---

### Task 2: Commit and push config

- [ ] **Step 1: Commit + push**

```bash
git add configs/exp_cylinder_037_hardbc_fixed_weight.toml && git commit -m "exp: CEXP-037 hard BC + fixed weight (Finding #6 control)

CEXP-002 + use_hard_body_bc=true + use_gradnorm=false + bc_body=0.
Tests whether removing GradNorm lets hard BC gate work without
the w_ns_u explosion seen in CEXP-016 (111%).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" && git push
```

---

### Task 3: Lab deploy + SLURM submit

- [ ] **Step 1: Pull on lab + sed-fix paths**

```bash
ssh lab-server 'cd pi-lnn && git stash 2>/dev/null; git pull; git stash drop 2>/dev/null; \
  sed -i "s|/Users/latteine/Documents/coding/RealPDEBench|/home/junyi/RealPDEBench|g" configs/exp_cylinder_037_hardbc_fixed_weight.toml && \
  sed -i "s|kolmogorov_A = 0.0|kolmogorov_A = 1e-6|" configs/exp_cylinder_037_hardbc_fixed_weight.toml && \
  sed -i "s|kolmogorov_k_f = 0.0|kolmogorov_k_f = 2.0|" configs/exp_cylinder_037_hardbc_fixed_weight.toml && \
  grep -E "use_hard_body_bc|use_gradnorm|bc_body_n_points|kolmogorov_A|RealPDEBench" configs/exp_cylinder_037_hardbc_fixed_weight.toml | grep -v "^#"'
```
Expected: `use_hard_body_bc = true`, `use_gradnorm = false`, `bc_body_n_points = 0`, `kolmogorov_A = 1e-6`, arrow path = `/home/junyi/...`.

- [ ] **Step 2: Submit SLURM job**

```bash
ssh lab-server 'cd pi-lnn && bash scripts/slurm/submit_exp.sh cylinder_037 configs/exp_cylinder_037_hardbc_fixed_weight.toml 2>&1 | tail -2; squeue -u junyi'
```
Expected: `Submitted batch job <ID>`。記下 job ID（注意：實際 ID 可能與預期不同，以 squeue 顯示為準）。

- [ ] **Step 3: 確認啟動 + 監控 L_phys 是否爆漲**

```bash
ssh lab-server 'JID=<JOBID>; until [ -f pi-lnn/logs/exp_cylinder_037_${JID}.out ]; do sleep 3; done; grep -E "trainable_param|hard_body_bc|^1 |^1000 " pi-lnn/logs/exp_cylinder_037_${JID}.out | head -8'
```
Expected: 看到 `hard_body_bc: enabled`，Step 1 L_phys 正常（~0.x）。**監控重點**：step 1000 後 L_phys 是否爆漲（hard BC gate 壓 physics 梯度，固定權重下若 L_phys 失控 = ❌ C）。

---

## Post-train checklist

訓練 ~1.6 hr。完成後：

1. **Eval**（r740 SLURM，checkpoint 用 `picon_kolmogorov_final.pt`）：
```bash
ssh lab-server 'cat > /tmp/eval_037.sbatch << EOF
#!/bin/bash
#SBATCH --job-name=eval_cyl037
#SBATCH --partition=r740
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/junyi/pi-lnn/logs/eval_cylinder_037_%j.out
cd /home/junyi/pi-lnn
/home/junyi/.local/bin/uv run python scripts/evaluate_cylinder.py \
  --checkpoint artifacts/cylinder/deeponet-cfc-cylinder-exp037-hardbc-fixed-weight/picon_kolmogorov_final.pt \
  --config configs/exp_cylinder_037_hardbc_fixed_weight.toml \
  --output artifacts/cylinder/deeponet-cfc-cylinder-exp037-hardbc-fixed-weight/summary.json \
  --device cuda
EOF
sbatch /tmp/eval_037.sbatch'
```

2. **讀 metrics**（注意 summary.json 可能是 nested dir，先確認真實路徑）：
```bash
ssh lab-server 'python3 << PYEOF
import json
d=json.load(open("pi-lnn/artifacts/cylinder/deeponet-cfc-cylinder-exp037-hardbc-fixed-weight/summary.json"))
kp=d.get("ke_pred_mean"); kr=d.get("ke_ref_mean")
print(f"KE={d[\"ke_rel_err_mean\"]*100:.2f}%  ke_pred/ref={kp/kr:.3f}  omega={d[\"omega_rmse_mean\"]:.2f}  div={d[\"div_l2_mean\"]:.2f}")
PYEOF'
```

3. **判讀（§4 gates）+ 更新 cylinder_log_v2.md**：
   - 核心比較：div < 1.14（baseline）？body_u_max ≈ 0？KE vs CEXP-016 111%？
   - L_phys 軌跡（從訓練 log）是否爆漲？
   - 寫入 INDEX + 變更紀錄。**必須先單獨確認 job ID 與 summary 路徑再寫數字**（CEXP-036 教訓）。
