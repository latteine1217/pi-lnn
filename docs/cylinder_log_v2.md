# Cylinder Wake 實驗紀錄 v2（Stable Phase State 主檔）

> **Status**: Cylinder stable phase 主 state（2026-05-22 啟用）。本檔是 cylinder（非週期 + 含障礙物 geometry）主線唯一 active state；legacy [`docs/archive/cylinder_log.md`](archive/cylinder_log.md)（只到 CEXP-002）已 superseded。
>
> **Scope**: 從 CEXP-001 起所有 cylinder 實驗的 state 紀錄。Kolmogorov 主線見 [`docs/experiment_log_v2.md`](experiment_log_v2.md)。
>
> **Numbering convention**:
> - `CEXP-001~099`: Re=10031（high-Re vortex shedding 主線）
> - `CEXP-100~199`: Re=1781（low-Re laminar wake / collapse 診斷）
> - Multi-seed suffix `_a~_e` 預留給後續 statistical 升級

---

## [STATE] Read Order

| 檔 | 內容 | 何時讀 |
|---|---|---|
| **本檔** `docs/cylinder_log_v2.md` | **唯一 active 主檔** — Cylinder STATE/INDEX/RECORD、confound warning | **任何 cylinder 任務前都讀這個** |
| [`docs/experiment_log_v2.md`](experiment_log_v2.md) | Kolmogorov stable phase | 跨案例對照（如 div L2 量級對比）|
| [`docs/archive/cylinder_log.md`](archive/cylinder_log.md) | Legacy（只到 CEXP-002）| 歷史追溯 |
| [`docs/archive/diagnostics_log.md`](archive/diagnostics_log.md) | physics denorm bug / Q5/Q7/Q8 | `use_physics_denormalization` cylinder 必開的根因 |
| [`CLAUDE.md`](../CLAUDE.md) `<KNOWN_PITFALLS>` | Cylinder 第二階自動微分 NaN bug + 非週期 BC loss 必要性 | 啟動新 cylinder run 前 |

---

## [STATE] Metrics Glossary

| Metric | 定義 | 解讀 |
|---|---|---|
| `ke_rel_err_mean` | `\|0.5⟨u²+v²⟩_pred − 0.5⟨u²+v²⟩_DNS\| / 0.5⟨u²+v²⟩_DNS`, mean over 40 eval frames | 全頻段 integral 能量誤差 |
| `ke_rel_err_late` | 同上，僅取 t ∈ [t_max·0.5, t_max] | 排除 spin-up，late-time 重建品質 |
| `u_rmse_mean` / `v_rmse_mean` | pointwise RMSE | 注意 u_mean ≈ 0.33，看相對量 |
| `omega_rmse_mean` | ω = ∂v/∂x − ∂u/∂y 的 RMSE | wake 渦核位置/強度 sensitivity |
| `div_l2_mean` | `‖∇·u_pred‖₂` over fluid cells | 非均勻格 + 集中尾跡 sensor 下 incompressibility 收斂緩慢 |
| **`ke_pred_mean / ke_ref_mean`** | 預測 KE / 真值 KE | **collapse 診斷量** — < 0.8 強烈暗示 trivial mean-flow 解 |
| `body_u_max` / `body_v_max` | cylinder body 內 \|u\|/\|v\| 最大值（hard BC 才有） | no-slip 違反度 |

---

## [STATE] Data Version

- DNS shard（Re=10031）: `/Users/latteine/Documents/coding/RealPDEBench/data/realpdebench/cylinder/hf_dataset/numerical/data-00000-of-00092.arrow`
- DNS shard（Re=1781）: `data-00020-of-00092.arrow`
- Domain: `[0, 0.325] × [0, 0.178]`（含 cylinder body），normalized to `[0, 1]²`（`domain_length = 1.0`）
- Grid: H=128, W=256（非均勻）
- T_full = 3990 frames（dt=0.005s），`sensor_subsample=20` → T=200（dt=0.1s）
- Inflow 量測：`u[100:, :, 0].mean() = 0.33 m/s`（Re=10031）
- Sensor:
  - `data/cylinder_sensors/sensors_qrpivot_K100_cylinder_Re10031.{json,npz}`（pure QR-pivot）
  - `data/cylinder_sensors/sensors_hybrid20qr80_K100_cylinder_Re10031.{json,npz}`（20 farthest + 80 QR hybrid）
  - `data/cylinder_sensors/sensors_hybrid20qr80_K100_cylinder_Re1781.{json,npz}`

---

## [STATE] Current Baseline

### Re=10031 主線 = **`CEXP-002` (soft inflow BC, KE 3.54%)**

```
Baseline ID:  CEXP-002
Config:       configs/exp_cylinder_002_k100_bc.toml
Artifact:     artifacts/cylinder/deeponet-cfc-cylinder-exp002-k100-bc/
              (eval at cylinder-eval-step10000/)
Architecture: B3-like (d_model=256, 2 cross-attn layers, 4 heads)
Sensor:       pure QR-pivot K=100 (集中尾跡)
BC:           soft inflow (bc_w=0.1, u_inf=0.33), bc_body=64, bc_slip=32, bc_outlet=0
Hard body BC: False
Collocation:  random, num_physics_points=64
Iterations:   10000 (1-shot)
GradNorm:     4-task [data, ns_u, ns_v, cont]
KE rel-err:   3.54 % (mean) / 3.88 % (late)
```

**唯一 working baseline**。所有後續 cylinder 實驗（CEXP-003~015）都應與此對齊比較。

### Re=1781 baseline: **尚未建立**

CEXP-013 (small capacity) ke_pred/ke_ref = **0.54** → trivial mean-flow collapse。CEXP-012 (full capacity) 無 artifact 但 config 註解暗示同樣 collapse。CEXP-014/015 (RAR adaptive) 配置完成但尚未驗證執行。

---

## [STATE] 主線固定假設

- **非週期域必須加 inflow BC loss**：CEXP-001（無 BC）KE 51% → CEXP-002（soft BC）KE 3.5%，14.5× 改善。週期域（Kolmogorov）不需要。
- **`use_physics_denormalization = true`** 對 cylinder 為必要：u_std ≈ 0.15，黏性項否則被 normalize 壓掉（詳見 [`diagnostics_log.md`](archive/diagnostics_log.md) Physics Output Denormalization）
- **Sensor placement**: K=100 QR-pivot 集中尾跡（x > 0.10）→ 來流區無 supervision，需 inflow BC 錨定
- **Physics 用 primitive momentum + continuity**（與 Kolmogorov 一致）
- **DeepONet-CfC decoder 用 smooth norm**（`rel_r = sqrt(rel² + 1e-8)`）避免 second-order autograd NaN（cylinder 觸發率 ~20%/batch，詳見 [`CLAUDE.md`](../CLAUDE.md) Physics Second-Order Autograd NaN）

---

## [STATE] ⚠️ Multi-Confound Warning（最重要！）

**CEXP-002 (baseline) 與 CEXP-007/010/013 沒有任何一對是 clean A/B comparison**。下表列出每個變數的差異：

| Variable | CEXP-002 (baseline) | CEXP-007 (distance) | CEXP-010 (hard BC) | CEXP-013 (Re=1781) |
|---|---|---|---|---|
| iter | **10k** | 3k | **5k** | **5k** |
| d_model | 256 | 256 | 256 | **128** ↓ |
| bc_body_n_points | **64** | 96 | 96 | 96 |
| bc_outlet_n_points | **0** | 32 | 32 | 32 |
| collocation strategy | **random** | body_aware | body_aware | body_aware |
| use_hard_body_bc | False | False | **True** | True |
| use_body_distance_feature | False | True (**BLOCKED**) | False | False |
| GradNorm tasks | **4** | 5 | 5 | 5 |
| Re | 10031 | 10031 | 10031 | **1781** |

**意涵**：
1. **「Hard BC 退步」(CEXP-010 KE 17.5% vs CEXP-002 KE 3.5%) 無法歸因**到 hard BC — 同時改了 iter (10k→5k 半步)、body_aware sampling、bc_outlet 新增、GradNorm 4→5 task。**iter 半步可能單一原因即足以解釋**。
2. **「Distance feature buggy」(CEXP-007 KE 32.7%)**：config header 註明 `use_body_distance_feature` key 沒落到 main 的 DEFAULT_PICON_ARGS → 此 key 訓練時靜默忽略 → CEXP-007 實質上是「**iter=3k 的 body_aware + 5-task GradNorm 配置**」，distance feature 從未生效。退步 9× 大概是 iter 太短 + 5-task GradNorm 把 BC 壓太低的副作用。
3. **「Re=1781 mean-flow collapse」(CEXP-013 ke_pred/ke_ref=0.54)** 同時 d_model 256→128 + Re drop + 5k iter — 不能單獨 claim「Re=1781 + 高容量 必然 collapse」（exp_012 full capacity Re=1781 沒留 artifact，無法驗證）。

**Paper-blocking**：任何 cylinder generalization claim 都必須先做乾淨 A/B comparison（單變數變動）才可信。

---

## [INDEX] Cylinder Active

| ID | Status | Configuration | KE rel-err | ke_pred/ke_ref | ω RMSE | div L2 | iter | 一句結論 |
|---|---|---|---|---|---|---|---|---|
| **CEXP-002** | 🥇 `ACTIVE_BASELINE` | Re=10031, soft inflow BC | **3.54 %** | **1.01** ✅ | 2.14 | 1.14 | 10k | **唯一 working baseline**；振幅高 ~10%，渦核位置可識別 |
| **CEXP-016** | 🔄 `PENDING_RUN` | Re=10031, **hard body BC + baseline 對齊**（single-variable ablation）| TBD | TBD | TBD | TBD | 10k | **CEXP-010-fair**: paper-grade hard BC effect 評估；唯一變動 `use_hard_body_bc=true`，其餘 100% 對齊 baseline |
| **CEXP-001** | `NEGATIVE_RESULT` | Re=10031, **無** BC | 51.0 % | — | — | 1.13 | — | [PHYSICAL_FAILURE] 來流 u → 0；被 CEXP-002 取代；artifact 已遺失 |
| CEXP-007 | `NEGATIVE_RESULT` | Re=10031, distance feature (BLOCKED), iter=3k | 32.7 % | 0.67 | 7.76 | **2.85** | 3k | `use_body_distance_feature` key 從未落到 main → distance 沒生效 + iter 太短 + 5-task GradNorm 壓 BC |
| CEXP-010 | `INCONCLUSIVE` | Re=10031, **hard body BC**, body_aware, iter=**5k** (半) | 17.5 % | 0.82 | 8.24 | 1.08 | 5k | KE 退步 5×、ω 退步 4× — 但與 baseline iter/strategy/GradNorm tasks 全不同, **無法歸因**；由 CEXP-016 重做乾淨 A/B |
| CEXP-013 | `NEGATIVE_RESULT` | Re=1781, d_model **128**, hard BC, body_aware | 46.3 % | **0.54** ❌ | 1.59 | 0.046 | 5k | **mean-flow collapse 直接證據** (ke_pred 僅 ref 的 54%)；div 0.046 是 trivial 解的副作用，非 incompressibility 成就 |

---

## [STATE] Surprise Findings（2026-05-22 揭露）

### Finding 1 — Hard body BC 看似退步，但 multi-confound 無法歸因

CEXP-010 (hard BC) KE 17.5% vs CEXP-002 (baseline) KE 3.54%，看似退步 5×。**但同時改了 4 個 component**（見上表 confound warning）。**iter 10k → 5k 半步即可能單獨解釋退步**。

**Action required**：跑 CEXP-010-fair（hard BC ON，其餘全部與 CEXP-002 一致：iter 10k、random sampling、4-task GradNorm、bc_outlet=0）才能 paper-grade claim hard BC effect。

### Finding 2 — Re=1781 mean-flow collapse 機制確認

CEXP-013 量化指標：
- `ke_pred_mean = 0.0011` vs `ke_ref_mean = 0.0021` → **預測 KE 僅真值 54%**
- u_RMSE 0.021 / u_mean 0.33 ≈ 6.4% 相對誤差（但 wake amplitude 完全沒重建）
- div L2 0.046 看似漂亮 → 是 trivial 解 ∇·u ≈ const 的副作用，**不是 PI-CON 在 Re=1781 滿足 incompressibility**

機制（per exp_013 config 註解 + Wang 2022 / Cao 2020 引用）：
- Re=1781 wake 是 mean flow 的 **small perturbation (u'/u_mean ≈ 15%)**
- Wide-NN implicit bias 偏好 low-frequency / minimum-norm 解
- 高容量 model 找到 trivial mean-flow 解 (u ≡ 0.33) 就能把 data loss 壓低，wake 不被學
- 即使 d_model 256 → 128 capacity 降 4× 仍 collapse（CEXP-013 已驗）

**Caveat**：CEXP-012 (full capacity Re=1781) 沒留 artifact，無法獨立驗證「full capacity 必然 collapse」。下一步若 CEXP-015 (collo 1024 + RAR) 仍 collapse，才能 claim「standard PINN training 在 Re=1781 wake 重建的 fundamental failure mode」。

### Finding 3 — Cylinder div L2 比 Kolmogorov 差兩個量級

| 案例 | div L2 (mean) | 比例 |
|---|---|---|
| Kolmogorov EXP-245 (20k n=5, 1024 collo) | **0.0039** | 1× |
| Kolmogorov EXP-200_a (legacy, 64 collo) | 0.066 | 17× |
| Cylinder CEXP-002 (working baseline) | **1.14** | **292×** |
| Cylinder CEXP-010 (hard BC, 5k iter) | 1.08 | 277× |

cylinder div L2 兩個量級 worse 的可能原因（**待 ablation**）：
- 非均勻格 → 採樣空間離散度大
- Sensor 集中尾跡 (x > 0.10) → 來流 + 障礙物上下 0 supervision，physics-only 收斂慢
- `use_physics_denormalization = true` → 黏性項量級被恢復，但 continuity loss gradient 量級可能變不平衡

**Hypothesis**：可能需要 cylinder-specific 的 continuity weight schedule 或 RAR-on-continuity-residual 才能對齊 Kolmogorov 量級。

---

## [RECORD] Cylinder 實驗詳細記錄

### CEXP-001：無 BC baseline（KE=51%，failure）

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_001_k100.toml` |
| Artifact | **已遺失**（state 結論保留） |
| KE rel-err mean | **51.0 %** |
| u RMSE | 2.47e-1 |
| v RMSE | 9.99e-2 |
| div L2 | 1.13 |
| 結論 | [PHYSICAL_FAILURE]：sensor 全集中尾跡（x>0.10）→ 來流區 u → 0 而非 u ≈ 0.33 m/s。**確立了「非週期域 + 集中 sensor 必須加 BC loss」的硬規則**。 |

### CEXP-002：Soft inflow BC（KE=3.54%，✅ working baseline）

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_002_k100_bc.toml` |
| Artifact | `artifacts/cylinder/deeponet-cfc-cylinder-exp002-k100-bc/cylinder-eval-step10000/` |
| Checkpoint | `lnn_kolmogorov_step_10000.pt`（artifact 內 ckpt 已遺失，但 metrics 保留）|
| KE rel-err mean / late | **3.54 % / 3.88 %** |
| u / v RMSE | 0.103 / 0.106 |
| ω RMSE | 2.14 |
| div L2 | 1.14 |
| ke_pred / ke_ref | 0.0680 / 0.0673 = **1.01** ✅ |
| 修改內容 | `bc_loss_weight=0.1`, `bc_inflow_u=0.33`, `bc_n_points=64`, `bc_body=64`, `bc_slip=32` |
| 訓練紀錄重點 | step 10k: L_data 1.15e-3, L_phys 3.25e-2, w_ns_u 0.108, w_cont 0.038, t_max 20.0 |
| 結論 | **Cylinder baseline 建立**。與 Kolmogorov EXP-064 (7.8%) 相當。KE(t) 振幅略高 ~10%，渦街可識別。div L2=1.14 仍待改善。 |

### CEXP-007：Distance feature（KE=32.7%，[BLOCKED + multi-confound]）

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_007_distance.toml` |
| Artifact | `artifacts/cylinder/deeponet-cfc-cylinder-exp007-distance/eval/` |
| KE rel-err mean / late | 32.7 % / 32.8 % |
| u / v RMSE | 0.091 / 0.105 |
| ω RMSE | 7.76 |
| div L2 | **2.85** ❌ |
| ke_pred / ke_ref | 0.045 / 0.067 = **0.67** ❌ |
| 設計意圖 | `use_body_distance_feature=true`（query 加 SDF feature dim +1），expect model 自學 boundary layer |
| 實際發生 | config header 標 **`BLOCKED`** — `use_body_distance_feature` key 沒落到 main 的 `DEFAULT_PICON_ARGS` / decoder。訓練時靜默忽略 → distance feature **從未生效**。 |
| 同時新增 | iter 3k（baseline 10k）、bc_body 96、bc_outlet 32、body_aware sampling、5-task GradNorm |
| 結論 | **無法獨立評估 distance feature**。退步主因可能是 iter 太短 + 5-task GradNorm 把 BC 壓太低。Distance feature 主線後被 hard BC (CEXP-010+) 路線取代。 |

### CEXP-010：Hard body BC（KE=17.5%，INCONCLUSIVE）

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_010_hard_bc.toml` |
| Artifact | `artifacts/cylinder/deeponet-cfc-cylinder-exp010-hard-bc/eval/` |
| Checkpoint | `lnn_kolmogorov_final.pt`（最末 step 5000） |
| KE rel-err mean / late | 17.5 % / 17.4 % |
| u / v RMSE | 0.095 / 0.108 |
| ω RMSE | **8.24** ❌（vs baseline 2.14）|
| div L2 | 1.08 |
| ke_pred / ke_ref | 0.055 / 0.067 = **0.82** |
| 設計意圖 | Sukumar 2022 hard body BC: `u = (φ/scale).clamp(0,1) · NN_u`, autograd-friendly SDF |
| Multi-confound | iter 半（10k→5k）+ hard BC + body_aware + bc_body 96 + bc_outlet 32 + 5-task GradNorm（vs baseline 全不同）|
| 結論 | **無法歸因到 hard BC 本身**。需要 CEXP-010-fair（iter 10k + 其餘 baseline 一致）才能 paper-grade claim。詳見 [STATE] Surprise Findings #1。 |

### CEXP-013：Re=1781 small capacity（KE=46.3%，mean-flow collapse）

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_013_re1781_small.toml` |
| Artifact | `artifacts/cylinder/deeponet-cfc-cylinder-exp013-re1781-small/cylinder-eval/` |
| Checkpoint | `lnn_kolmogorov_step_5000.pt`（final） |
| KE rel-err mean / late | 46.3 % / 46.3 % |
| u / v RMSE | 0.021 / 0.023（看似小但相對 u_mean=0.33 仍 6.4%）|
| ω RMSE | 1.59 |
| div L2 | **0.046** ⚠️（trivial 解副作用，**非** incompressibility 成就） |
| ke_pred / ke_ref | 0.0011 / 0.0021 = **0.54** ❌ |
| 設計意圖 | exp_012 (Re=1781 full capacity) collapse 後降 capacity 4-8×：d_model 256→128, operator_rank 256→128, attention layers 2→1, fourier_harmonics 16→12 |
| 結論 | **Mean-flow collapse 確認**。降 capacity 沒救。引用 Wang 2022 frequency bias 機制。詳見 [STATE] Surprise Findings #2。 |

---

## [STATE] Orphan Configs（intent-only, **無 artifact**）

下列 configs 寫好但 artifact 不存在 — 要嘛沒跑、要嘛跑了沒留、要嘛跑掛。**不該當「實驗結果」引用**。

| Config | 設計意圖（per header） | 為何無 artifact（推測）|
|---|---|---|
| `exp_cylinder_003_postfix.toml` | C2/C3/M3 修法 + artifact 改名避覆蓋 | 設計變更，未必實際執行 |
| `exp_cylinder_003b_resume_to_10k.toml` | 對齊 exp_002 iter 10k 公平比較 | resume_checkpoint 禁用（EXP-082 災難），可能因此沒跑 |
| `exp_cylinder_004_gradnorm5.toml` | 4→5 task GradNorm 把 BC 納入動態平衡 | — |
| `exp_cylinder_005_bc_body.toml` | bc_body 64→256、bc_outlet 0→64、body_aware sampling | — |
| `exp_cylinder_006_bc_lite.toml` | bc_body 96 + bc_outlet 32 lite 配置 | 可能被 010 hard BC 路線取代 |
| `exp_cylinder_008_distance_5k.toml` | exp_007 同 setup + iter 3k→5k | exp_010 註解提及 exp_008 KE 33.6%、div 2.95 → 可能跑過後刪 |
| `exp_cylinder_009_floor.toml` | gradnorm_min_weight 0→0.05 防 w_cont 被催化降到 0 | — |
| `exp_cylinder_010b_resume_to_8k.toml` | 從 010 step_5000 resume 到 8k | resume 禁用 |
| `exp_cylinder_012_re1781.toml` | Re=1781 full capacity (d_model 256) | exp_013 註解確認「spectral collapse」→ 可能跑過後刪 |
| `exp_cylinder_014_re1781_rar256.toml` | collo 64→256 + RAR adaptive，預估 wake gradient 信號 20× | 未跑 / 未留 |
| `exp_cylinder_015_re1781_rar1024.toml` | collo 1024 + RAR pool 10240，預估 wake 60× 信號 | **Task 4 候選**（待 falsifiability gate 確立後執行）|

---

## [STATE] Open Questions（含 falsifiability gates）

| 問題 | 現況 | 狀態 |
|---|---|---|
| **CEXP-016 (= CEXP-010-fair)**: hard BC ON 但 iter/sampling/GradNorm 全與 CEXP-002 對齊 | **config 完成** ([`configs/exp_cylinder_016_hard_bc_fair.toml`](../configs/exp_cylinder_016_hard_bc_fair.toml))，待執行 | **進行中** — Task 6 in progress |
| **CEXP-002 multi-seed (n=3-5)** | single seed only，無 σ | **高優先** — paper-grade rigor |
| **CEXP-015 (Re=1781, collo 1024+RAR)** | config 完備，gate 已設但**暫不執行**（per 2026-05-22 prioritization decision: hard BC 歸因比 Re=1781 collapse 重要）| `DEFERRED` — gate 見下 |
| div L2 cylinder vs Kolmogorov 兩個量級差距 | 機制不明（非均勻格 / sensor 集中 / denorm 任一） | 開放（CEXP-016+ 候選研究方向）|
| Sensor placement variability | K=100 single placement only | 待開工 |
| 與 FLRNet / Energy Transformer 比較 | baseline 已建立但未實際 benchmark | 待規劃 |
| CfC Jacobian spectral radius @ cylinder | 未寫腳本 | 同 Kolmogorov 待開工 |

### CEXP-015 Falsifiability Gate（Task 4 prerequisite）

執行 `configs/exp_cylinder_015_re1781_rar1024.toml`（1024 collo + RAR pool 10240, freq=50, iter=2500）時的 **stop-loss criteria**：

| Step | Metric | Gate | Action |
|---|---|---|---|
| 500 | RAR pool top-K 採樣分布 | 若 wake region (x > 0.10) 佔比 < 50% | RAR 失效訊號（per exp_014 trap warning：trivial 解 NS residual ≈ 0 → ranking 退化成 random）→ 評估是否 abort |
| 1000 | `ke_pred / ke_ref` | < 0.7 | 第一個 warning gate — continue but red flag |
| **2000** | `ke_pred / ke_ref` | **< 0.8** | **🛑 early stop**（避免 trivial mean-flow collapse 浪費 1-2 hr）|
| 2500 (final) | `ke_rel_err_mean` | > 30% | 結論：**collapse 在 standard PINN training + 1024 collo + RAR 下不可避免**，寫進 paper limitations 段 |
| 2500 (final) | `ke_rel_err_mean` | 15-30% | **partial improvement**，分析剩餘 collapse mechanism |
| 2500 (final) | `ke_rel_err_mean` | < 15% | **break collapse 成功**，paper 升級為 cylinder Re=1781 baseline；後續設計 ablation |

**Resource budget**：
- M2 Mac MPS: 估 4-6 sec/step × 2500 ≈ 3-4 hr wall（unified memory 12-16 GB tight）
- Lab GPU (RTX 3090): 估 1-1.5 sec/step ≈ 1 hr wall + 24 GB VRAM 充裕
- 建議 lab GPU 跑 — M2 Mac 留小 step 測試

---

## [STATE] Rejected Directions

從 cylinder 已試 / 已驗證為負的方向：

1. **無 BC loss**（CEXP-001 KE 51%）：非週期 + 集中 sensor 不可行
2. **`use_body_distance_feature` (SDF as query input)**（CEXP-007）：key 沒落到 main，**靜默忽略**。若要重啟需先在 src 端落地（cylinder_dataset 已備 SDF grid，缺 decoder 接 dim+1 路徑）
3. **降 capacity 救 Re=1781 collapse**（CEXP-013）：d_model 256→128 沒救
4. **Resume from checkpoint**：繼承 Kolmogorov EXP-082 災難（silent state corruption），必須 1-shot 訓練。Cylinder 的 `exp_003b/007b/010b` resume 配置全 invalid。

---

## [STATE] Pitfalls cross-ref（與 CLAUDE.md `<KNOWN_PITFALLS>` 對齊）

- **Physics Second-Order Autograd NaN**: cylinder 觸發率 ~20%/batch（sensor 與 collocation 都在 grid cells 上）。修法已落地：[`src/pi_con/decoder.py`](../src/pi_con/decoder.py) 用 `rel_r = sqrt((rel**2).sum + 1e-8)`。
- **非週期域必須加 Inflow BC Loss**: CEXP-001 → CEXP-002 14.5× 改善，已確立硬規則
- **`use_physics_denormalization = true` 對 cylinder 必開**: u_std ≈ 0.15 黏性項否則被壓掉
- **Resume 禁用**: 同 Kolmogorov EXP-082，cylinder `*b_resume_to_*` configs 全 invalid
- **Sensor file axis convention**: cylinder sensor 由 `scripts/generate_sensors_qrpivot_cylinder.py` 生成；應通過 `test_sensor_axis_convention.py`（待補 cylinder coverage）

---

## 變更紀錄

- **2026-05-22 v2 啟用**:
  - 從 legacy `docs/archive/cylinder_log.md` (只到 CEXP-002) 升級為 stable phase 主檔
  - 揭露 4 個 surprise findings：(1) hard BC 退步無法歸因 multi-confound、(2) Re=1781 mean-flow collapse 機制確認 ke_pred/ke_ref=0.54、(3) cylinder div L2 比 Kolmogorov 差 292×、(4) `use_body_distance_feature` key 從未落到 main 靜默忽略
  - 建立 CEXP-002/007/010/013 multi-confound table — 沒有任何一對是 clean A/B
  - 加 CEXP-015 falsifiability gate（ke_pred/ke_ref < 0.8 @ step 2000 → early stop）
  - Orphan configs 表清楚標記「無 artifact」status
- **2026-05-22 (CEXP-016 design)**:
  - 設計 CEXP-016 = CEXP-010-fair single-variable ablation：唯一變動 `use_hard_body_bc=true`，其餘 100% 與 CEXP-002 對齊
  - 加入 [INDEX] 標 `PENDING_RUN`，Open Questions 改為「進行中」
  - CEXP-015 改 `DEFERRED`（prioritization: hard BC 歸因優先於 Re=1781 collapse 解決）
  - Expected outcomes 4 種解讀寫進 config header（A 中性 / B 輕微退步 / C 實質有害 / D 實質有益），各自對應 paper claim
