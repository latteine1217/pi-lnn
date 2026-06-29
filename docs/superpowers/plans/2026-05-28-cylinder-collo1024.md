# CEXP-030 Collocation 1024 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以 CEXP-002 為 base，單一變數 `num_physics_points` 從 64 升至 1024，驗證更密的 physics collocation 能否降低 cylinder KE rel-err 與 div L2。

**Architecture:** Config-only change（無 src 改動）。CEXP-002 soft BC baseline 所有 hyperparameter 不動，只改 `num_physics_points = 1024`。Clean single-variable A/B comparison。

**Tech Stack:** Python, PyTorch, SLURM (r740 partition, RTX 3090), `uv`, lab-server SSH

---

## Context

| 項目 | 值 |
|---|---|
| Base experiment | CEXP-002 (`configs/exp_cylinder_002_k100_bc.toml`), KE 3.54%, div L2 1.14 |
| Single variable | `num_physics_points`: 64 → **1024** |
| Hypothesis | 更密的 random collocation 提供更完整的 NS + continuity coverage，應降低 div L2 (目前 1.14) 並可能略降 KE rel-err |
| Falsifiability | KE < 3.0% ✅ 改善；KE 3-5% 🟡 同等；KE > 5% ❌ over-regularization；div L2 < 0.8 → incompressibility 改善 |
| Risk | 訓練時間 ~2-4× 增加（physics 前向/後向 16× 點數）；GradNorm 需重新平衡 |
| Next CEXP | CEXP-030（CEXP-029 job 3694 尚在跑）|

---

### Task 1: Create CEXP-030 config

**Files:**
- Create: `configs/exp_cylinder_030_collo1024.toml`

- [ ] **Step 1: Create config 檔（複製 CEXP-002，修改 3 個欄位）**

```toml
# configs/exp_cylinder_030_collo1024.toml
# CEXP-030 = CEXP-002 + num_physics_points 64 → 1024
#
# 設計目的：
#   CEXP-002 (soft BC, KE 3.54%) 是唯一 working baseline。
#   Collocation 只有 64 點，div L2 = 1.14（比 Kolmogorov 差 17×）。
#   Hypothesis: 1024 random collo 提供更完整 NS + continuity coverage → div L2 下降。
#
# 單一變數變動 vs CEXP-002：
#   num_physics_points: 64 → 1024
#
# Falsifiability gates:
#   KE < 3.0%    → ✅ 改善（更多 physics supervision 有幫助）
#   KE 3.0-5.0%  → 🟡 同等（collocation 不是瓶頸）
#   KE > 5.0%    → ❌ over-regularization（physics 壓倒 sensor fit）
#   div L2 < 0.8 → ✅ incompressibility 改善
#   div L2 > 1.2 → ❌ 無改善（可能 random 採樣在 fluid domain 效率低）

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
bc_body_n_points = 64
bc_slip_n_points = 32

t_early_weight = 10.0
t_early_threshold = 0.05
physics_collocation_strategy = "random"

use_gradnorm = true
gradnorm_update_freq = 1000
gradnorm_ema_momentum = 0.9
gradnorm_init_weights = [1.0, 0.01, 0.01, 0.01]

time_marching = true
time_marching_start = 0.5
time_marching_warmup = 0.3

iterations = 10000
num_physics_points = 1024   # ← 唯一變動（CEXP-002 = 64）
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

artifacts_dir = "artifacts/cylinder/deeponet-cfc-cylinder-exp030-collo1024"
```

- [ ] **Step 2: 確認欄位只有 `num_physics_points` 與 CEXP-002 不同**

```bash
diff <(grep -v "^#" configs/exp_cylinder_002_k100_bc.toml | grep -v "^$") \
     <(grep -v "^#" configs/exp_cylinder_030_collo1024.toml | grep -v "^$")
```

Expected: 只有 `num_physics_points`、`device`、`artifacts_dir` 三行不同（device mps→cuda 是 lab deploy 差異，artifacts_dir 不同是預期）。

---

### Task 2: Update cylinder_log_v2.md

**Files:**
- Modify: `docs/cylinder_log_v2.md`

- [ ] **Step 1: 在 [INDEX] Active 表加 CEXP-030 行**

在 `| **CEXP-029** | ...` 那行後面插入：

```markdown
| **CEXP-030** | `PENDING_RUN` | Re=10031, **CEXP-002 + collo 1024**（single-var: num_physics_points 64→1024） | — | — | — | — | 10k | Single-var collocation ablation；目標確認 div L2 是否因 physics coverage 改善 |
```

- [ ] **Step 2: 在 [STATE] Open Questions 加 CEXP-030 行**

在 `| **Stage 4 no-GNN boundary semantics: outlet BC only (CEXP-029)** | ...` 那行後插入：

```markdown
| **CEXP-030: collo 1024 ablation** | CEXP-002 + num_physics_points 64→1024；單一變數；目標降低 div L2（目前 1.14）；falsifiability: KE <3% ✅ / KE >5% ❌ / div L2 <0.8 ✅ | `PENDING_RUN` |
```

- [ ] **Step 3: 在 變更紀錄 末尾加一行**

```markdown
- **2026-05-28 CEXP-030 設計**:
  - 新增 `configs/exp_cylinder_030_collo1024.toml`，由 CEXP-002 派生，唯一變動為 `num_physics_points: 64 → 1024`。
  - Hypothesis: 更密的 physics collocation 應降低 div L2（目前 1.14）並可能改善 KE。Falsifiability: KE > 5% → over-regularization；div L2 > 1.2 → 無改善。
```

---

### Task 3: Commit and push

**Files:** (既有 src 無改動)

- [ ] **Step 1: 確認 diff 只有 config + log**

```bash
git diff --stat
```

Expected: 只有 `configs/exp_cylinder_030_collo1024.toml`（new）和 `docs/cylinder_log_v2.md`（modified）。

- [ ] **Step 2: Stage + commit**

```bash
git add configs/exp_cylinder_030_collo1024.toml docs/cylinder_log_v2.md
git commit -m "exp: CEXP-030 collocation 1024 ablation config

Single-variable ablation from CEXP-002 baseline (KE 3.54%).
num_physics_points: 64 → 1024, all other settings identical.
Hypothesis: better physics coverage reduces div L2 (currently 1.14).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

- [ ] **Step 3: Push**

```bash
git push
```

---

### Task 4: Lab deploy + SLURM submit

**Prerequisites:** SSH access to lab-server；r740 partition 可用

- [ ] **Step 1: lab-server pull**

```bash
ssh lab-server 'cd pi-lnn && git pull'
```

Expected: `1 file changed` (config)，已見到 `exp_cylinder_030_collo1024.toml`。

- [ ] **Step 2: Sed 修正 lab 路徑（arrow_shards + kolmogorov dummies）**

```bash
ssh lab-server 'cd pi-lnn && \
  sed -i "s|/Users/latteine/Documents/coding/RealPDEBench|/home/junyi/RealPDEBench|g" \
    configs/exp_cylinder_030_collo1024.toml && \
  sed -i "s|kolmogorov_A = 0.0|kolmogorov_A = 1e-6|" \
    configs/exp_cylinder_030_collo1024.toml && \
  sed -i "s|kolmogorov_k_f = 0.0|kolmogorov_k_f = 2.0|" \
    configs/exp_cylinder_030_collo1024.toml'
```

- [ ] **Step 3: smoke test（確認 config 能被 parse + dataset 讀取）**

```bash
ssh lab-server 'cd pi-lnn && /home/junyi/.local/bin/uv run python -c "
import toml, torch
cfg = toml.load(\"configs/exp_cylinder_030_collo1024.toml\")
print(\"num_physics_points:\", cfg[\"train\"][\"num_physics_points\"])
print(\"arrow_shards:\", cfg[\"train\"][\"arrow_shards\"])
print(\"kolmogorov_A:\", cfg[\"train\"][\"kolmogorov_A\"])
print(\"OK\")
"'
```

Expected:
```
num_physics_points: 1024
arrow_shards: ['/home/junyi/RealPDEBench/...']
kolmogorov_A: 1e-06
OK
```

- [ ] **Step 4: Submit SLURM train job**

```bash
ssh lab-server 'cd pi-lnn && bash scripts/slurm/submit_exp.sh cylinder_030 configs/exp_cylinder_030_collo1024.toml'
```

Expected: `Submitted batch job XXXX`。記下 job ID。

- [ ] **Step 5: 確認 job 進入 queue**

```bash
ssh lab-server 'squeue -u junyi'
```

Expected: job 出現在 r740 partition，State = PD 或 R。

- [ ] **Step 6: 等 Step 1 output 出現確認訓練啟動**

```bash
ssh lab-server 'until [ -f pi-lnn/logs/exp_cylinder_030_<JOBID>.out ]; do sleep 5; done; head -30 pi-lnn/logs/exp_cylinder_030_<JOBID>.out'
```

Expected: 看到 `=== Configuration ===` 和 `trainable_parameters: 3172170`（與 CEXP-002 相同，config-only change）。

---

## Post-train checklist（訓練完成後）

訓練預估時間：~2-3 hr（1024 collo vs 64 的 16× physics 點，但 SOAP preconditioner 攤銷後實際可能 1.5-2×）。

完成後需執行：

1. **Eval** (SLURM r740 job)：
```bash
cat > /tmp/eval_030.sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=eval_cyl030
#SBATCH --partition=r740
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/junyi/pi-lnn/logs/eval_cylinder_030_%j.out

cd /home/junyi/pi-lnn
/home/junyi/.local/bin/uv run python scripts/evaluate_cylinder.py \
  --checkpoint artifacts/cylinder/deeponet-cfc-cylinder-exp030-collo1024/picon_kolmogorov_final.pt \
  --config configs/exp_cylinder_030_collo1024.toml \
  --output artifacts/cylinder/deeponet-cfc-cylinder-exp030-collo1024/summary.json \
  --device cuda
EOF
sbatch /tmp/eval_030.sbatch
```

2. **Rsync** back：
```bash
rsync -av lab-server:pi-lnn/artifacts/cylinder/deeponet-cfc-cylinder-exp030-collo1024/summary.json \
  artifacts/cylinder/deeponet-cfc-cylinder-exp030-collo1024/
```

3. **Judge** per falsifiability gates，更新 `docs/cylinder_log_v2.md`。
