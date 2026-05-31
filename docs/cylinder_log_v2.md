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
| **CEXP-023** | `NEGATIVE_RESULT` | Re=10031, **B+C geometry-aware** (`use_graph_spatial_encoder` + `use_trunk_geo_context`) | **365.4 %** ❌ | **4.65** | 34.60 | 9.93 | 10k | [PHYSICAL_FAILURE] B+C over-predict KE 4.65×，比 CEXP-002 baseline 3.54% 差兩個量級 |
| **CEXP-024** | `NEGATIVE_RESULT` | Re=10031, **B-only graph spatial encoder** (`use_graph_spatial_encoder`) | **458.6 %** ❌ | **5.59** | 40.55 | 8.69 | 10k | [PHYSICAL_FAILURE] B-only 最差，sensor-side geometry memory 造成 severe over-energy |
| **CEXP-025** | `NEGATIVE_RESULT` | Re=10031, **C-only trunk geometry context** (`use_trunk_geo_context`) | **401.1 %** ❌ | **5.01** | 32.60 | 11.68 | 10k | [PHYSICAL_FAILURE] C-only 仍 over-predict KE 5.01×，trunk geometry memory 不足 |
| **CEXP-026** | `NEGATIVE_RESULT` | Re=10031, **C-only RNG-neutral control** (`use_trunk_geo_context` + `geometry_preserve_base_rng`) | **463.7 %** ❌ | **5.64** | 38.12 | 10.61 | 10k | [PHYSICAL_FAILURE] RNG-neutral 仍 over-energy，CEXP-025 不是單純 RNG/init confound |
| **CEXP-027** | `NEGATIVE_RESULT` | Re=10031, **B-only zero-gate control** (`use_graph_spatial_gate` + `geometry_preserve_base_rng`) | **489.5 %** ❌ | **5.89** | 38.80 | 11.41 | 10k | [PHYSICAL_FAILURE] zero-gate 沒救，B path 失敗不是單純 ungated residual 初始擾動 |
| **CEXP-028** | `NEGATIVE_RESULT` | Re=10031, **hybrid20qr80 sensor baseline**（20 farthest + 80 QR；no geometry modules） | **154.4 %** ❌ | **2.54** | 44.90 | 14.48 | 10k | [PHYSICAL_FAILURE] sensor coverage alone 只把 over-energy 從 B/C 的 4.65–5.89× 降到 2.54×，仍遠離 CEXP-002 |
| **CEXP-029** | `NEGATIVE_RESULT` | Re=10031, **hybrid20qr80 + soft outlet BC**（CEXP-028 + `bc_outlet_n_points=32` only） | **164.8 %** ❌ | **~2.65** | 48.15 | 17.94 | 10k | [PHYSICAL_FAILURE] outlet BC 輕微惡化 (CEXP-028 154% → 165%)；div L2 也更差；soft outlet BC 不是 over-energy 根因 |
| **CEXP-030** | `NEGATIVE_RESULT` | Re=10031, **CEXP-002 + collo 1024**（single-var: `num_physics_points` 64→1024） | **610 %** ❌❌❌ | **~7.1** | 47.79 | 7.49 | 10k | [PHYSICAL_FAILURE] **ill-posedness（非 GradNorm 失衡）**：training 全健康（L_data 1.79e-3, L_phys 1.67e-2, w_ns_u 僅 0.65）但 eval KE 610%；強 physics 在 sparse-sensor underdetermined 系統中把場推向 spurious NS-consistent 解。見 Finding #9 |
| **CEXP-031** | `POSITIVE_FINDING` | Re=10031, **hybrid20qr80 + bc_body=0**（CEXP-028 - body soft BC） | **13.1 %** 🟡 | **~1.13** | 9.27 | 5.33 | 10k | ✅ 移除 body BC 後 KE 154%→13.1%（10× 改善）；見 Finding #8 統一 2×2 交互作用解釋 |
| **CEXP-032** | `NEGATIVE_RESULT` | Re=10031, **QR wake + bc_body=0**（CEXP-002 - body soft BC） | **177.8 %** ❌❌❌ | **~2.7** | 13.14 | 5.50 | 10k | [PHYSICAL_FAILURE] **推翻「body BC 冗余」假說**：QR body 區唯一約束就是 body BC，移除後 body interior 無拘束 → 污染整個 wake；w_ns_u 推 272× |
| **CEXP-033** | `NEGATIVE_RESULT` | Re=10031, **hybrid95downstream + bc_body=0**（CEXP-031 - 5 upstream sensors） | **12.5 %** 🟡 | **~1.12** | 9.57 | 5.33 | 10k | upstream sensor **不是** 13.1% gap 主因（CEXP-031 13.1% ≈ CEXP-033 12.5%）；gap 來自 hybrid coverage density 而非 sensor 衝突 |
| **CEXP-034** | `NEGATIVE_RESULT` | Re=10031, **CEXP-002 + K=200 QR sensor**（single-var: sensor K 100→200） | **355.5 %** ❌❌❌ | **4.55** | 46.08 | 11.69 | 10k | ❌ K=200 災難性惡化 baseline 100×（3.54%→355%）；強烈確認 Finding #8——K=200 引入 upstream 8 + within-body-x 11 sensor → 與 body BC 嚴重衝突，over-energy 4.55× |
| **CEXP-035** | `NEGATIVE_RESULT` | Re=10031, **K=200 + collo 1024**（CEXP-034 + num_physics_points 64→1024） | **375.4 %** ❌❌❌ | **4.75** | 45.78 | 8.27 | 10k | ❌ K=200+collo1024 亦災難（375%）；vs CEXP-030 (K=100+collo1024, 610%) 略好但仍崩潰 → 更多 collo 在 sensor/BC 已衝突基礎上無法挽救（div 8.27 略降是 collo 副作用，非物理正確）|
| **CEXP-036** | `NEUTRAL_RESULT` | Re=10031, **CEXP-002 + RAR collocation**（physics_collocation_strategy random→rar, freq=1000） | **3.66 %** 🟡 | **1.039** | 2.18 | 3.62 | 10k | 🟡 RAR ≈ baseline（3.54%→3.66%，持平略差）；w_ns_u=0.16 健康（SOAP+RAR freq=1000 未爆，驗證 EXP-054 下限）。**補強 Finding #9**：聰明放置 collo（不增量）不崩潰但也不改善 → physics 配置非 cylinder baseline 瓶頸，瓶頸是 sensor 資訊上限。div 3.62 > baseline 1.14（RAR 把點集中 wake 渦核 → body/邊界 continuity 監督變稀）|
| **CEXP-016** | `NEGATIVE_RESULT` | Re=10031, **hard body BC + baseline 對齊**（CEXP-010-fair single-var）| **111.6 %** ❌ | **2.12** | 12.62 | 6.93 | 10k | **Catastrophic over-predict 2.12×**, w_ns_u GradNorm 推 209× → Stage 1 diagnostic 起點 |
| **CEXP-017** | `NEGATIVE_RESULT` | Re=10031, hard BC + **5-task GradNorm** (H1) | **303.6 %** ❌❌❌ | **4.04** | 19.30 | 6.50 | 10k | **❌ H1-C falsified**：5-task GradNorm 反讓 catastrophic 推 3× (w_bc 19.5+w_ns_u 3.82) |
| **CEXP-018** | `NEGATIVE_RESULT` | Re=10031, hard BC + **body_aware sampling** (H2) | 106.3 % ❌ | 2.06 | 11.90 | 6.27 | 10k | **❌ H2-C falsified**：body_aware ≈ CEXP-016（no improvement） |
| **CEXP-019** | `NEGATIVE_RESULT` | Re=10031, hard BC + **bc_body 96 + bc_outlet 32** (H3) | 139.3 % ❌ | 2.39 | 12.70 | 6.13 | 10k | **❌ H3-C falsified**：dense BC 沒救（且更糟一點）|
| **CEXP-020** | `NEGATIVE_RESULT` | Re=10031, **trunk SDF input + hard BC OFF** (Stage 2 Option A) | **405.2 %** ❌❌❌ | **5.06** | 35.1 | 9.80 | 10k | **❌ Option C**：比 hard BC catastrophic 還差 114×；SDF input 在 sensor 不覆蓋 body 區時有害，需要 Options E/F |
| **CEXP-021** | `NEGATIVE_RESULT` | Re=10031, **SDF trunk + hard BC** (Option A+BC combined) | **174 %** ❌ | **~2.74** | — | — | 10k | [STOP-LOSS] SDF+hard BC 比 hard BC alone (111%) 更糟；w_ns_u=1.96 GradNorm 仍病態；兩者無法互補 |
| **CEXP-022** | `NEGATIVE_RESULT` | Re=10031, **cross-attn geometry tokens + hard BC** (Option E) | **99.8 %** ❌ | **~1.98** | 12.45 | 6.11 | 10k | [STOP-LOSS] Geometry tokens 輕微改善 (99.8% vs 111.6%) 但 w_ns_u=2.09 GradNorm 病態未解；hard BC 路線架構級失敗確認 |
| **CEXP-001** | `NEGATIVE_RESULT` | Re=10031, **無** BC | 51.0 % | — | — | 1.13 | — | [PHYSICAL_FAILURE] 來流 u → 0；被 CEXP-002 取代；artifact 已遺失 |
| CEXP-007 | `NEGATIVE_RESULT` | Re=10031, distance feature (BLOCKED), iter=3k | 32.7 % | 0.67 | 7.76 | **2.85** | 3k | `use_body_distance_feature` key 從未落到 main → distance 沒生效 + iter 太短 + 5-task GradNorm 壓 BC |
| CEXP-010 | `INCONCLUSIVE` (superseded) | Re=10031, **hard body BC**, body_aware, iter=**5k** (半) | 17.5 % | 0.82 | 8.24 | 1.08 | 5k | Multi-confound; **Stage 1 證明 hard BC + standard architecture incompatibility**, CEXP-010 KE 17.5% 是 multi-confound accidental survival |
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

### Finding 4 — Stage 1 全 ❌：Hard BC + standard PI-CON architecture 有 fundamental incompatibility（2026-05-23/24）

per [`docs/superpowers/specs/2026-05-23-cylinder-hard-bc-enabling-conditions-design.md`](../docs/superpowers/specs/2026-05-23-cylinder-hard-bc-enabling-conditions-design.md) §4 multi-config decision table 判讀: **全 ❌ pattern → Re-diagnose**。

Stage 1 完整 outcomes:

| Hypothesis | Config | Prior | KE rel-err | ke_pred/ke_ref | Outcome |
|---|---|---|---|---|---|
| H1: 5-task GradNorm 把 BC weight 納入動態平衡 | CEXP-017 | 60% | **303.6 %** | **4.04** | ❌ H1-C (**反更糟 3×**) |
| H2: body_aware sampling 補強 boundary gradient signal | CEXP-018 | 25% | 106.3 % | 2.06 | ❌ H2-C |
| H3: dense BC supervision (bc_body 96 + bc_outlet 32) | CEXP-019 | 15% | 139.3 % | 2.39 | ❌ H3-C |
| (control) hard BC alone | CEXP-016 | — | 111.6 % | 2.12 | catastrophic baseline |

**Key insights**:

1. **CEXP-017 顛覆 60% prior**: 5-task GradNorm 把 catastrophic 推得**更糟 3×**, 而非緩解。Training log 顯示 `w_bc` 從 0.1 → 19.5 (推 195×) + `w_ns_u` 從 0.01 → 3.82, BC + physics 同時被推高 → model 仍 over-predict 4×。**BC loss 進 GradNorm 反讓 model 找 trivial 滿足 BC 卻違反 sensor MSE 的解**。

2. **H2/H3 ≈ CEXP-016 持平**: body_aware (KE 106%) 與 dense BC (KE 139%) 都沒把 KE 拉回 baseline 量級 (3.5%)。証明這些 multi-confound 內部 component **不是 enabling conditions**，只是 accidental 條件。

3. **CEXP-010 (legacy multi-confound, KE 17.5%) 是 accidental survival**: 推測 5k iter 不夠長, 還未進入 catastrophic failure mode。10k iter 公平比較全部都 catastrophic。

4. **Hard BC + standard PI-CON architecture (SOAP + ScheduleFree + 4/5-task GradNorm) fundamental incompatibility**: 任何單一 enabling condition 無法救。Catastrophic 機制 (`ke_pred/ke_ref > 2.0` over-predict, `w_ns_u > 2.0` GradNorm pathology) 是架構級而非超參級問題。

**Root cause hypothesis (2026-05-24)**: Trunk net 目前完全沒有 geometry awareness — hard BC gate 只是 output post-hoc transformation, NN_u 對所有 query 都試圖輸出「自然 wake value」, 然後 body-adjacent 值被 gate 壓掉。Wake 區 NN_u 必須過度補償 → physics residual 暴增 → GradNorm 推 w_ns_u → over-predict。

**Stage 2 redirect (Option A)**: 加 SDF `φ` 進 trunk input concat (`query = [x, y, t, c, φ]`), 移除 hard BC gate。Trunk 自學「near-body vs far-field」區分，不依賴 output gate 救援。詳見 [STATE] Open Questions。

### Finding 5 — Stage 2 Option A (trunk SDF input) 也失敗（CEXP-020, 2026-05-24）— SDF input 在 sensor 不覆蓋 body 區時有害

CEXP-020 = CEXP-002 baseline + `use_body_distance_feature=true` + `use_hard_body_bc=false`. KE **405.2%** (比 CEXP-016 hard BC catastrophic 111% 還差 3.6×, Option C per spec §4).

| Metric | CEXP-002 baseline | CEXP-020 (Option A) | Δ |
|---|---|---|---|
| KE rel-err mean | 3.54 % | **405.2 %** | 114× 退步 |
| ke_pred / ke_ref | 1.01 | **5.06** (over-predict 5×) | 更糟 |
| u RMSE | 0.103 | 0.554 | 5.4× |
| ω RMSE | 2.14 | 35.1 | 16× |
| div L2 | 1.14 | 9.80 | 8.6× |
| GradNorm w_ns_u final | 0.108 | 1.13 (10×) | 偏高但非 catastrophic |
| L_data final | 1.15e-3 | **5.03e-3** (4× worse data fit) | — |

**根本原因分析**：SDF input 造成 **adversarial training signal**。

- φ(x,y) → 0 near body = trunk MLP 學「suppress velocity」
- K=100 sensors 全在 wake (x > 0.10), **body 區沒有 sensor supervision 糾正**
- Model 的 body-region suppression (learned from φ hint) 缺乏 corrective feedback
- 結果：SDF hint 給錯誤 prior 且無法修正 → data fit 4× 惡化 → KE 405%

**Comparison with CEXP-002** (same config without SDF input, KE 3.54%): 加 SDF input 反而比不加差 114×. SDF input 並非中性特徵——在 sensor coverage 不完整的情況下是有害特徵。

**Paper-grade insight (2026-05-24)**:
> "Raw SDF scalar as trunk input is harmful when sensors don't cover the body region (K=100 wake-concentrated placement). The φ feature provides a `suppress near body` inductive bias without corrective sensor feedback, perturbing the optimization landscape and degrading data fit 4×. Geometry-aware learning requires stronger structural priors (Options E: cross-attention with geometry tokens, or F: geometry-conditioned hypernetwork) that can leverage geometry information without creating adversarial gradients."

**Next direction (per spec §4 ❌C stop-loss)**:
- **Not** continuing to Options B/D (Fourier-on-φ / per-layer gate) — same adversarial signal problem
- **Options E/F are future paper material** per user 2026-05-24 decision
- Cylinder generalization section paper claim: revise to "geometry awareness requires full sensor coverage or architectural-level enforcement; raw SDF input alone insufficient"

### Finding 6 — Option E (cross-attention geometry tokens + hard BC) 失敗：hard BC 路線架構級問題無法透過 geometry encoding 解決（2026-05-28）

CEXP-022 = CEXP-016 + cross-attention geometry tokens（body surface points 注入為 K-V pool tokens）。

| Metric | CEXP-002 baseline | CEXP-016 (hard BC) | CEXP-022 (geo tokens + hard BC) | Δ vs CEXP-016 |
|---|---|---|---|---|
| KE rel-err mean | 3.54 % | 111.6 % | **99.8 %** | 略好 11% |
| ke_pred/ke_ref | 1.01 | 2.12 | ~1.98 | 略改善 |
| ω RMSE | 2.14 | 12.62 | 12.45 | ≈ 同等 |
| div L2 | 1.14 | 6.93 | 6.11 | 略改善 |
| **w_ns_u final** | **0.108** | **~2.09** | **2.09** | **同等（GradNorm 仍爆）** |

**關鍵觀察**：w_ns_u = 2.09 與 CEXP-016 完全相同（~2.09）。Geometry tokens 讓 KE 從 111.6% 降至 99.8%（~11% 改善），但 GradNorm 病態機制完全未被打破。

**理論分析**：

1. **Geometry tokens 提供了正確的資訊**：body surface 作為 K-V token，attention mechanism 確實在 body 附近給出 「zero-velocity 先驗」。這解釋了 KE 略降（99.8% vs 111.6%）。

2. **但資訊量仍不夠阻止 GradNorm explosion**：hard BC gate `φ(x,y)/scale * NN_u` 的梯度結構中，gate 近 body ≈ 0，physics residual 的梯度主要由 `gate * ∂NN_u/∂params` 決定。即使 NN_u 因 geometry tokens 在 body 附近輸出較小值，gate 壓縮後的殘差 *gradient magnitude* 對 GradNorm 的 signal-to-noise ratio 仍然很差。

3. **Root cause 不是 geometry awareness，是 hard BC + GradNorm 的 optimization incompatibility**：Finding #4 的結論再次被確認。所有 hard BC 變體（CEXP-016/021/022）的 w_ns_u 最終都在 ~2；這是架構級問題，不是 geometry encoding 問題。

**Paper-grade insight (2026-05-28)**:
> "Cross-attention geometry tokens provide marginal improvement (~11% KE error reduction) by encoding zero-velocity priors at body surface positions. However, the fundamental incompatibility between hard BC output gates and GradNorm-based multi-task optimization persists: the gate-suppressed physics residual gradients create an unbalanced optimization landscape that drives w_ns_u to over-weight NS momentum loss regardless of geometry encoding. Geometry-aware learning must be pursued without hard BC enforcement."

**Hard BC 路線全封閉（2026-05-28）**：
- CEXP-016/017/018/019 (Stage 1): all failed
- CEXP-021 (SDF + hard BC): 174%，w_ns_u=1.96
- CEXP-022 (geo tokens + hard BC): 99.8%，w_ns_u=2.09
- 所有 hard BC 實驗都在 90-175% 的 stop-loss zone，w_ns_u 最終都在 ~2.0

**Next direction**: 棄所有 hard BC 路線。若未來需要 strict no-slip enforcement，需要不同的 multi-task optimization（如 augmented Lagrangian 替代 GradNorm）。目前研究方向回歸 CEXP-002 soft BC base，透過 sensor coverage / boundary semantics 改善。

### Finding 7 — Sensor/BC 空間衝突機制確認（CEXP-031, 2026-05-29）

> ⚠️ **2026-05-29 修正**：本 finding 的「衝突」觀察正確，但單一框架不完整。CEXP-032 揭露了對稱的另一半（body 區無約束也會崩潰）。完整解釋見 **Finding #8（2×2 交互作用）**，本節保留為歷史推理過程。

**實驗設計**：CEXP-031 = CEXP-028（hybrid20qr80, KE 154%）+ `bc_body_n_points: 64 → 0`，單一變數。

| Metric | CEXP-002（QR wake + body BC） | CEXP-028（hybrid + body BC） | CEXP-031（hybrid + no body BC） |
|---|---|---|---|
| KE rel-err | **3.54 %** | 154.4 % | **13.1 %** |
| ω RMSE | 2.14 | 44.90 | 9.27 |
| div L2 | 1.14 | 14.48 | 5.33 |
| w_ns_u final | 0.108 | — | 2.89 |

**結論**：移除 body soft BC 後，KE 從 154.4% → 13.1%（**10× 改善**）。KE < 30% ✅，衝突假說成立。

**機制**：hybrid20qr80 的 farthest-point anchor sensors 有部分覆蓋 body 附近（upstream/near-body 區域）。這些 sensor 提供 `u ≈ small but nonzero` 的 supervision，與 `bc_body_n_points=64` 在同一空間強制 `u → 0` 形成 competing objectives。GradNorm 無法同時滿足 → optimization landscape 崩潰 → over-energy（154%）。

**剩餘 gap（13.1% vs CEXP-002 3.54%）的可能原因**：
1. **Wake coverage 稀疏化**：hybrid sensor 把 K=100 budget 分散到 near-body/inlet/outlet，wake 主要動態模式（vortex street）的 sensor density 降低，重建精度下降
2. **No-slip 缺乏 supervision**：沒有 body BC 後，body surface 附近 u=v=0 完全沒有 loss 約束；w_ns_u=2.89 顯示 physics residual 在 body 附近仍高
3. **Farthest-point anchor 帶來 far-field 不一致**：20 個 farthest 點覆蓋了 inlet/outlet 等難以重建的位置，model 在這些位置的 loss 較難收斂

**Paper-grade insight (2026-05-29)**:
> "When sensors are placed in body-adjacent regions, soft body BC creates competing objectives in the same spatial domain, causing GradNorm to diverge. The solution for deployment depends on sensor placement: wake-concentrated sensors (CEXP-002) can coexist with soft body BC; body-adjacent sensors require removing body BC supervision or using a conflict-aware multi-task optimizer."

**CEXP-002 成功的深層原因更新**：pure QR wake sensor (x > 0.10) 不只是「sensor 多」，而是在空間上與 body soft BC 完全分區（sensors 全在 x > 0.10，body BC 在 x ≈ 0.2–0.29）。這個隱式分區是 CEXP-002 穩定訓練的關鍵條件，之前未被識別。

### Finding 8 — Body 區約束的「恰好一個」原則：2×2 交互作用（CEXP-032/033, 2026-05-29）

**動機**：CEXP-031 的「衝突」框架預測「移除 body BC 應該總是改善或中性」。為驗證，跑 CEXP-032（QR + no body BC）。結果完全推翻單一框架——QR 移除 body BC 後 KE 從 3.54% 爆到 **177.8%**。這逼出一個更完整的 2×2 結構。

**完整 2×2 交互作用表**：

| | body BC = 64 | body BC = 0 |
|---|---|---|
| **QR wake**（body 區無 sensor）| **3.54 %** ✅ (CEXP-002) | **177.8 %** ❌ (CEXP-032) |
| **hybrid**（body 區有 sensor）| **154.4 %** ❌ (CEXP-028) | **13.1 %** 🟡 (CEXP-031) |

這是 **pure interaction（無主效果）**：對角線好、反對角線壞。body BC 的效果完全取決於 sensor 是否覆蓋 body 區。

**統一原則**：
> **Body 區域必須恰好有一個約束來源。零個 → 無拘束區污染整場（CEXP-032 177%）；兩個 → 競爭目標使 GradNorm 崩潰（CEXP-028 154%）。**

三個機制（互相印證）：

1. **零約束 → 崩潰（CEXP-032）**：QR sensor 全在 wake，body 區唯一約束就是 body BC。移除後 body interior 完全無拘束——physics residual 對 steady body 是 `0=0` trivially（u=任意 divergence-free 場都滿足），不強制 u=0。無拘束的 body 區誤差透過 cross-attention / physics collocation 污染鄰近 fluid cell → 傳播到整個 wake → 177.8%。w_ns_u 被推到 2.72。

2. **雙約束 → 衝突（CEXP-028）**：near-body sensor（u≈小值非零）vs body BC（u→0）同區競爭 → 154%。

3. **恰好一個 → 可行**：
   - QR + body BC（約束來自 BC）= **3.54%** ✅ 最優
   - hybrid + no body BC（約束來自 near-body sensor）= **13.1%** 🟡 可行但非最優

**CEXP-033 補充（upstream sensor 非 gap 主因）**：移除 hybrid 的 5 個 upstream sensor（x < 0.063）後 KE **12.5%** ≈ CEXP-031 的 13.1%。證明 13.1% vs 3.54% 的 gap **不是** upstream sensor 衝突造成，而是 hybrid farthest-point 佈點的 wake coverage density 低於 QR-pivot 資訊論最優佈點。

**修正我先前的錯誤預測（2026-05-29，誠實記錄）**：
- 預測 CEXP-032 ≈ 3.5%（認為 body BC 冗余、body cell 被 mask）→ 實際 177.8%，錯 40×。錯誤根源：低估了「無拘束 body 區會污染周圍 fluid cell」的傳播效應；body cell 即使被 mask，其鄰域 fluid cell 仍受污染。
- 預測 CEXP-033 ≈ 7%（認為 upstream sensor 是 gap 主因）→ 實際 12.5%，方向錯。upstream sensor 影響微乎其微。

**Paper-grade insight (2026-05-29，取代 Finding #7 的版本)**:
> "Body-region reconstruction in obstacle flows requires exactly one constraint source. With wake-concentrated sensors that leave the body region unsupervised, a soft body BC is necessary (removing it lets the unconstrained body region corrupt the entire wake, KE 3.5%→178%). With body-adjacent sensors, the soft body BC instead conflicts with sensor supervision and must be removed (KE 154%→13%). The two constraint sources are mutually exclusive, not additive. The optimal configuration (wake QR-pivot sensors + soft body BC, KE 3.54%) succeeds because the single constraint per region is spatially partitioned: inflow BC anchors the inlet, body BC anchors the obstacle, and information-optimal sensors anchor the wake."

**對 paper 的意義**：這個 2×2 是一個乾淨、可發表的 sensor-placement / BC-design ablation。它把「為何 CEXP-002 work、為何所有改進嘗試失敗」用一個原則解釋完畢。不再是「遇到瓶頸」，而是「已找到設計原則，CEXP-002 正好落在最優格」。

### Finding 9 — CEXP-030 collo 1024 失敗是 ill-posedness，非 GradNorm 失衡（2026-05-29，修正先前歸因）

> ⚠️ **修正記錄**：先前（2026-05-28）把 CEXP-030 失敗歸因為「physics 梯度放大 16× → GradNorm 失衡」。經查證 **此歸因錯誤**，本 finding 取代之。

**錯誤歸因的反證**：

1. **Physics loss 用 mean reduction**（`training.py:1179` `torch.mean(mom_u**2)`）→ loss 期望值不隨 collocation 點數變，沒有「放大 16×」。
2. **CEXP-030 訓練曲線全部健康**：

| | CEXP-002 (64 collo) | CEXP-030 (1024 collo) |
|---|---|---|
| L_data final | 1.15e-3 | 1.79e-3（相當）|
| L_phys final | 3.25e-2 | 1.67e-2（更低）|
| w_ns_u final | 0.108 | **0.65**（溫和，未爆）|
| **KE eval** | **3.54 %** | **610 %** |

w_ns_u 只到 0.65（CEXP-016 hard BC 系列才是 ~2.0 爆炸）。training loss 全綠但 eval 災難 → 排除優化失衡，確認是 **泛化 / ill-posedness 失敗**。

**真正機制（sparse-sensor PINN ill-posedness）**：
- NS + continuity 在「K=100 sensor 全集中 wake」的稀疏約束下 **underdetermined**——有無窮多 divergence-free + NS-consistent 場，只有一個是真實 DNS 場。
- **64 collo**：physics 弱正則化，data interpolation 主導 → 落在真實解附近（3.54%）。
- **1024 collo**：physics 變主導場塑造力，在全場（含無 sensor 的上游/body/far-wake/邊界）強力施加「滿足 NS」。但「滿足 NS」≠「正確解」→ model 滑到一個 spurious 解：NS 殘差低 ✓、100 sensor 吻合 ✓、但無 sensor 區能量爆 7×（ke_pred/ke_ref≈7.1）→ KE 610%。

**與 Kolmogorov 的對比（為何 K 主線加 collo 反而好）**：Kolmogorov 週期域 + 固定 forcing + sensor 全域分佈 → 解空間受限，EXP-245 用 1024 collo 改善。Cylinder 開放域 + wake-only sensor → 解空間極大，physics 過強有害。即 Krishnapriyan 2021 描述的 PINN failure mode。

**Paper-grade insight (2026-05-29)**:
> "In sparse-sensor reconstruction of open-domain flows, increasing PDE collocation density degrades accuracy: with sensors confined to the wake, the NS system is underdetermined, and strong physics enforcement drives the solution onto a spurious NS-consistent manifold whose energy is wrong by 7×, despite low training data- and physics-losses. The accuracy of the working configuration depends on physics acting as weak regularization, not a dominant field-shaping constraint."

**統一視角（Findings #8 + #9 合起來）**：CEXP-002 的成功來自一個微妙平衡——(a) 每個區域恰好一個 velocity supervision（Finding #8），(b) physics 弱到只當正則化、由 data 主導內插（Finding #9）。**任何強化 physics（加 collo）或弄亂 supervision（sensor/BC 衝突）的動作都破壞此平衡。**

---

## [RECORD] Cylinder 實驗詳細記錄

### CEXP-029：Hybrid20QR80 + soft outlet BC（eval submitted）

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_029_hybrid20qr80_outlet_bc.toml` |
| Artifact | `artifacts/cylinder/deeponet-cfc-cylinder-exp029-hybrid20qr80-outlet-bc/` |
| 派生基準 | CEXP-028；唯一變動為新增 `bc_outlet_n_points=32` |
| Sensor | `sensors_hybrid20qr80_K100_cylinder_Re10031`（與 CEXP-028 相同） |
| Geometry modules | 全部 off：`use_graph_spatial_encoder=false`, `use_trunk_geo_context=false`, `use_body_distance_feature=false`, `use_hard_body_bc=false` |
| Hypothesis | 若 CEXP-028 的 2.54× over-energy 主要來自缺 outlet semantic，soft outlet BC 應降低 `ke_pred/ke_ref` 並改善 outlet probing，且不犧牲 wake vorticity。 |
| Falsifiability | KE > 30% 或 `ke_pred/ke_ref > 1.5` → soft outlet BC alone 不足；KE 改善但 `omega_rmse` 惡化 → 可能只是數值耗散，不是物理解。 |
| Smoke | lab-server smoke passed：`sensor_pos=(100,2)`, `sensor_vals=(100,200,2)`, `bc_outlet_n_points=32`, trainable params `3,138,634`。 |
| Train | job 3694 completed，stderr 空；final step 10000: `L_data=2.5164e-03`, `L_phys=1.7861e-02`, `L_total=8.3290e-03`。Step 8000 had transient physics spike `L_phys=8.1836e+00`, then recovered by step 9000。 |
| 目前狀態 | `EVAL_SUBMITTED`；training completed；eval job 3702 submitted on r740 single-GPU。 |

### CEXP-028：Hybrid20QR80 sensor baseline（negative result）

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_028_hybrid20qr80_baseline.toml` |
| Artifact | `artifacts/cylinder/deeponet-cfc-cylinder-exp028-hybrid20qr80-baseline/` |
| 派生基準 | CEXP-002；唯一變動為 sensor placement |
| Sensor | `sensors_hybrid20qr80_K100_cylinder_Re10031` = 20 farthest-point spatial anchors + 80 QR-pivot sensors |
| Coverage vs pure QR | hybrid reaches normalized x/y boundary (`x=[0,1]`, `y=[0,1]`), with near inlet 2, near outlet 2, top/bottom 8, upstream-of-body 5；pure QR 對應為 0/0/0/1。 |
| Hypothesis | 若 CEXP-023~027 失敗主要由 pure QR downstream-only coverage 導致，CEXP-028 應接近 CEXP-002 baseline 或至少降低 inlet/body/outlet probing error。 |
| Falsifiability | KE <= 10% 且 boundary probing 改善 → hybrid coverage 是必要修正；KE > 30% 或 boundary probing 仍壞 → sensor coverage alone 不足，需 boundary-token / stronger BC semantics。 |
| Train | job 3690 completed，stderr 空；final step 10000: `L_data=4.2938e-03`, `L_phys=1.3326e-02`, `L_total=1.6004e-02`。 |
| Eval | `cylinder-eval-step10000/summary.json`；job 3691 completed cleanly，stderr 空。 |
| KE rel-err mean / late | **154.4 % / 151.6 %** (`ke_rel_err_mean=1.5444`, `ke_rel_err_late=1.5159`) |
| ke_pred / ke_ref | **2.54×** (`0.17099 / 0.06719`) |
| Velocity RMSE | `u=0.3865`, `v=0.1806` |
| Vorticity / divergence | `omega_rmse=44.90`, `div_l2=14.48`（DNS baseline `div_ref_l2=8.74`） |
| 判讀 | Hybrid sensor coverage 有降低 global KE over-prediction（B/C/controls 為 4.65–5.89×，CEXP-028 為 2.54×），但仍是 CEXP-002 baseline 的 44× KE error，且 vorticity/divergence 更差；因此 pure QR coverage 不足是因素之一，但 sensor coverage alone 不是充分解。 |

### CEXP-026：C-only RNG-neutral control（negative result）

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_026_trunk_geo_rng_control.toml` |
| Artifact | `artifacts/cylinder/deeponet-cfc-cylinder-exp026-trunk-geo-rng-control/` |
| 派生基準 | CEXP-002；對照 CEXP-025 |
| 唯一架構變動 | `use_trunk_geo_context=true`, `geometry_preserve_base_rng=true` |
| Hypothesis | 若 CEXP-025 主要是 optional module 消耗 RNG 改變 baseline layer 初始化，則 CEXP-026 應回到接近 CEXP-002；若仍 over-energy，C path 本身有害。 |
| Falsifiability | KE < 10% → RNG confound 成立；KE > 30% 或 `ke_pred/ke_ref > 2` → C path 仍失敗。 |
| Eval | `cylinder-eval-step10000/summary.json`；job 3688 completed cleanly (`stderr` empty) |
| KE rel-err | **463.7%** (`ke_rel_err_mean=4.6372`) |
| ke_pred/ke_ref | **5.64** (`0.3787 / 0.06719`) |
| ω RMSE | **38.12** |
| div L2 | **10.61** |
| 目前狀態 | `NEGATIVE_RESULT`；`geometry_preserve_base_rng` 沒有讓 C-only 回到 CEXP-002，反而比 CEXP-025 更差。 |

### CEXP-027：B-only zero-gate RNG-neutral control（negative result）

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_027_graph_spatial_zero_gate.toml` |
| Artifact | `artifacts/cylinder/deeponet-cfc-cylinder-exp027-graph-spatial-zero-gate/` |
| 派生基準 | CEXP-002；對照 CEXP-024 |
| 唯一架構變動 | `use_graph_spatial_encoder=true`, `use_graph_spatial_gate=true`, `geometry_preserve_base_rng=true` |
| Hypothesis | 若 CEXP-024 主要是 ungated graph residual 初始擾動造成，則 zero-gate 後 CEXP-027 應回到接近 CEXP-002；若仍 over-energy，B path 本身有害。 |
| Falsifiability | KE < 10% → ungated residual 是主因；KE > 30% 或 `ke_pred/ke_ref > 2` → B path 仍失敗。 |
| Eval | `cylinder-eval-step10000/summary.json`；job 3689 completed cleanly (`stderr` empty) |
| KE rel-err | **489.5%** (`ke_rel_err_mean=4.8946`) |
| ke_pred/ke_ref | **5.89** (`0.3960 / 0.06719`) |
| ω RMSE | **38.80** |
| div L2 | **11.41** |
| 目前狀態 | `NEGATIVE_RESULT`；zero-gate + RNG-neutral 仍 severe over-energy，甚至略差於 CEXP-024。 |

### CEXP-024：B-only graph spatial encoder（negative result）

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_024_graph_spatial_only.toml` |
| Artifact | `artifacts/cylinder/deeponet-cfc-cylinder-exp024-graph-spatial-only/` |
| 派生基準 | CEXP-002（Re=10031, K=100 QR-pivot, soft inflow/body/slip BC, random collocation, 10k planned） |
| 唯一架構變動 | `use_graph_spatial_encoder=true`, `graph_k_neighbors=8`, `use_trunk_geo_context=false` |
| Hypothesis | sensor tokens 在進 CfC 前聚合 body geometry，可改善 branch-side 對障礙物位置的記憶。 |
| Falsifiability | KE < 10% 且 body/wake 指標不惡化 → 有效；KE 10-30% → 部分有效；KE > 30% 或 `ke_pred/ke_ref > 2` → B-only 不足或有害。 |
| Eval | `cylinder-eval-step10000/summary.json`；job 3682 completed cleanly (`ExitCode=0:0`) |
| Metrics | `ke_rel_err_mean=4.5857`, `ke_rel_err_late=4.6610`, `ke_pred/ke_ref=5.59`, `omega_rmse=40.55`, `div_l2=8.69` |
| 目前狀態 | `NEGATIVE_RESULT`；[RESULT: PHYSICAL_FAILURE] KE > 30% gate 被大幅違反，B-only 明確失敗。 |

### CEXP-025：C-only trunk geometry context（negative result）

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_025_trunk_geo_only.toml` |
| Artifact | `artifacts/cylinder/deeponet-cfc-cylinder-exp025-trunk-geo-only/` |
| 派生基準 | CEXP-002（Re=10031, K=100 QR-pivot, soft inflow/body/slip BC, random collocation, 10k planned） |
| 唯一架構變動 | `use_graph_spatial_encoder=false`, `use_trunk_geo_context=true` |
| Hypothesis | trunk query 讀取 body geometry memory，可改善 query-side 對 boundary/wake 位置的局部感知。 |
| Falsifiability | KE < 10% 且 body/wake 指標不惡化 → 有效；KE 10-30% → 部分有效；KE > 30% 或 `ke_pred/ke_ref > 2` → C-only 不足或有害。 |
| Eval | `cylinder-eval-step10000/summary.json`；job 3683 completed cleanly (`ExitCode=0:0`) |
| Metrics | `ke_rel_err_mean=4.0114`, `ke_rel_err_late=4.0264`, `ke_pred/ke_ref=5.01`, `omega_rmse=32.60`, `div_l2=11.68` |
| 目前狀態 | `NEGATIVE_RESULT`；[RESULT: PHYSICAL_FAILURE] KE > 30% gate 被大幅違反，C-only 明確失敗。 |

### CEXP-023：B+C explicit geometry memory（negative result）

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_023_graph_spatial_trunk_geo.toml` |
| Artifact | `artifacts/cylinder/deeponet-cfc-cylinder-exp023-graph-spatial-trunk-geo/` |
| 派生基準 | CEXP-002（Re=10031, K=100 QR-pivot, soft inflow/body/slip BC, random collocation, 10k planned） |
| 唯一架構變動 | `use_graph_spatial_encoder=true`, `graph_k_neighbors=8`, `use_trunk_geo_context=true` |
| Hypothesis | explicit geometry memory 讓 sensor token 與 trunk query 都能讀取 body geometry，避免 raw SDF scalar 的 adversarial prior。 |
| Falsifiability | KE < 10% 且 body/wake 指標不惡化 → 有效；KE 10-30% → 部分有效；KE > 30% 或 `ke_pred/ke_ref > 2` → geometry prior 仍造成 over-predict，停止此路線。 |
| Eval | `cylinder-eval-step10000/summary.json`；job 3681 completed cleanly (`ExitCode=0:0`) |
| Metrics | `ke_rel_err_mean=3.6537`, `ke_rel_err_late=3.7174`, `ke_pred/ke_ref=4.65`, `omega_rmse=34.60`, `div_l2=9.93` |
| 目前狀態 | `NEGATIVE_RESULT`；[RESULT: PHYSICAL_FAILURE] KE > 30% gate 被大幅違反。B+C 是三者中 KE 最低，但仍比 CEXP-002 baseline 3.54% 差兩個量級。 |

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

### CEXP-016：Hard BC + baseline 對齊 (CEXP-010-fair, KE=111.6%, catastrophic)

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_016_hard_bc_fair.toml` |
| Artifact | `artifacts/cylinder/deeponet-cfc-cylinder-exp016-hard-bc-fair/` |
| Checkpoint | `picon_kolmogorov_final.pt`（step 10000） |
| KE rel-err mean / late | **111.6 %** / 113.7 % ❌ |
| u / v RMSE | 0.253 / 0.103 |
| ω RMSE | **12.62** (5.9× baseline) |
| div L2 | **6.93** (6× baseline) |
| ke_pred / ke_ref | 0.142 / 0.067 = **2.12** ❌ (over-predict) |
| GradNorm `w_ns_u` final | **2.09** (推 209× from 0.01) |
| 設計變動 | 唯一 `use_hard_body_bc=true`, 其餘 100% 對齊 CEXP-002 baseline |
| 結論 | **Catastrophic over-predict**。Hard BC gate 強制 body 區 = 0 → NN_u 在 wake 區補償壓力大 → physics residual 暴增 → GradNorm 推 w_ns_u → model 失控 over-predict 2.12×。Stage 1 baseline failure。詳見 Surprise Findings #4。 |

### CEXP-017：Hard BC + 5-task GradNorm (H1, KE=303.6%, **worst**)

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_017_hard_bc_5task_gn.toml` |
| Artifact | `artifacts/cylinder/deeponet-cfc-cylinder-exp017-hard-bc-5task-gn/` |
| Checkpoint | `picon_kolmogorov_final.pt`（step 10000） |
| KE rel-err mean / late | **303.6 %** / 305.6 % ❌❌❌ |
| u / v RMSE | **0.458** / 0.113 |
| ω RMSE | **19.30** |
| div L2 | 6.50 |
| ke_pred / ke_ref | 0.271 / 0.067 = **4.04** (over-predict 4×) |
| GradNorm `w_ns_u` final | **3.82** (推 382×) |
| GradNorm `w_bc` final | **19.56** (推 195× from 0.1) |
| 設計變動 (vs CEXP-016) | `gradnorm_init_weights [1.0, 0.01, 0.01, 0.01]` → `[1.0, 0.01, 0.01, 0.01, 0.1]`（H1: BC weight 進 GradNorm）|
| 結論 | **❌ H1-C falsified**。5-task GradNorm 反讓 catastrophic 推 3× — BC weight 進 GradNorm 後**反向放大** physics dominance（BC + NS 同時被推高）→ model trivial 滿足 BC 卻違反 sensor MSE → over-predict 4×。Surprise: 60% prior 顛覆。 |

### CEXP-018：Hard BC + body_aware sampling (H2, KE=106.3%, no improvement)

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_018_hard_bc_body_aware.toml` |
| Artifact | `artifacts/cylinder/deeponet-cfc-cylinder-exp018-hard-bc-body-aware/` |
| Checkpoint | `picon_kolmogorov_final.pt`（step 10000） |
| KE rel-err mean / late | 106.3 % / 108.7 % ❌ |
| u / v RMSE | 0.247 / 0.099 |
| ω RMSE | 11.90 |
| div L2 | 6.27 |
| ke_pred / ke_ref | 0.139 / 0.067 = **2.06** |
| GradNorm `w_ns_u` final | 1.65 (推 165×) |
| 設計變動 (vs CEXP-016) | `physics_collocation_strategy "random"` → `"body_aware"` (30% near-body + 70% uniform) |
| 結論 | **❌ H2-C falsified**。body_aware sampling ≈ CEXP-016 (KE 微降 5pp 但仍 catastrophic)。Boundary gradient signal 雖加強, physics dominance 機制不被打破。 |

### CEXP-019：Hard BC + dense BC supervision (H3, KE=139.3%, mild worse)

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_019_hard_bc_dense_bc.toml` |
| Artifact | `artifacts/cylinder/deeponet-cfc-cylinder-exp019-hard-bc-dense-bc/` |
| Checkpoint | `picon_kolmogorov_final.pt`（step 10000） |
| KE rel-err mean / late | 139.3 % / 141.4 % ❌ |
| u / v RMSE | 0.277 / 0.101 |
| ω RMSE | 12.70 |
| div L2 | 6.13 |
| ke_pred / ke_ref | 0.161 / 0.067 = **2.39** |
| Train wall | 2:27 hr (50% 多 vs 016/017/018 的 1:36 hr — bc_outlet 加密 + body_aware 加 collocation 成本) |
| 設計變動 (vs CEXP-016) | `bc_body_n_points 64→96` + 新增 `bc_outlet_n_points=32` |
| 結論 | **❌ H3-C falsified**。加密 soft BC supervision **微更糟** (KE 111→139%)。Hard BC over-predict 機制不被多 BC points 打破。 |

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
| **Stage 1 全 falsified（CEXP-017/018/019）** | 已驗證 hard BC + standard PI-CON architecture **fundamental incompatibility** | ✅ Closed 2026-05-24 (Finding #4) |
| **Stage 2: Option A — Trunk SDF input concat (CEXP-020)** | ❌ **Failed** (KE 405.2%, 比 hard BC catastrophic 還差 114×). SDF input 在 sensor 不覆蓋 body 區時有害 (adversarial training signal)。Finding #5 written. | ✅ Closed 2026-05-24 (❌ negative finding) |
| **Stage 3: B/C/B+C explicit geometry memory ablation (CEXP-023/024/025)** | ❌ all failed：CEXP-023 KE 365.4%, CEXP-024 KE 458.6%, CEXP-025 KE 401.1%；三者皆 severe over-energy (`ke_pred/ke_ref=4.65–5.59`)。 | ✅ Closed 2026-05-27 (`NEGATIVE_RESULT`) |
| **Stage 3 controls: RNG-neutral C and zero-gate B (CEXP-026/027)** | ❌ both failed：CEXP-026 KE 463.7%, CEXP-027 KE 489.5%；RNG/init confound 與 ungated residual 都不是主要根因，B/C geometry memory path 本身會放大能量。 | ✅ Closed 2026-05-27 (`NEGATIVE_RESULT`) |
| **Stage 3 sensor coverage control: hybrid20qr80 baseline (CEXP-028)** | ❌ CEXP-028 eval completed：KE 154.4%, `ke_pred/ke_ref=2.54`, `omega=44.90`, `div=14.48`。比 B/C over-energy 輕，但仍遠離 CEXP-002；sensor coverage alone 不足。 | ✅ Closed 2026-05-27 (`NEGATIVE_RESULT`) |
| **Stage 4 no-GNN boundary semantics: outlet BC only (CEXP-029)** | ❌ **Failed** (KE 164.8%)；soft outlet BC 輕微惡化 CEXP-028 (154%)；div L2 從 14.48 → 17.94 更差。Outlet semantics 不是 over-energy 根因。 | ✅ Closed 2026-05-28 (`NEGATIVE_RESULT`) |
| **CEXP-030: collo 1024 ablation** | ❌ **Failed** (KE 610%)；**修正歸因**：非 GradNorm 失衡（w_ns_u 僅 0.65、training loss 全健康），而是 sparse-sensor PINN 的 ill-posedness——強 physics 把 underdetermined 系統推向 spurious NS-consistent 解。見 Finding #9。 | ✅ Closed 2026-05-29 (`NEGATIVE_RESULT`) |
| **Body-region constraint 機制（CEXP-031/032/033, Finding #8）** | ✅ **解決**：2×2 交互作用——body 區必須恰好一個約束（零 → 污染整場 178%；雙 → GradNorm 衝突 154%）。CEXP-002（QR + body BC）正好落在最優格。"瓶頸"已轉為可發表的 sensor/BC design ablation。 | ✅ Closed 2026-05-29 (Finding #8) |
| **Option E: cross-attn geometry tokens + hard BC (CEXP-022)** | ❌ **Failed** (KE 99.8%, stop-loss zone)；w_ns_u=2.09 與 CEXP-016 相同，geometry tokens 輕微降低 KE error (~12%) 但未解決 GradNorm 病態；Finding #6 written。**Hard BC 路線全部封閉**。 | ✅ Closed 2026-05-28 (`NEGATIVE_RESULT`) |
| **CEXP-002 multi-seed (n=3-5)** | single seed only，無 σ | **高優先** — paper-grade rigor |
| **CEXP-015 (Re=1781, collo 1024+RAR)** | config 完備，gate 已設但**暫不執行**（per 2026-05-22 prioritization decision: hard BC 歸因比 Re=1781 collapse 重要）| `DEFERRED` |
| div L2 cylinder vs Kolmogorov 兩個量級差距 | 機制不明（非均勻格 / sensor 集中 / denorm 任一） | 開放（Stage 2 Option A 可能 indirectly fix —if trunk geometry awareness 改善 incompressibility）|
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
- **Geometry-aware flags 必須明確 opt-in**: `use_graph_spatial_encoder` / `use_trunk_geo_context` 預設 false；啟用但沒有 `body_xy` 時 fail fast，禁止 Kolmogorov 靜默退回錯誤 no-op。
- **Resume 禁用**: 同 Kolmogorov EXP-082，cylinder `*b_resume_to_*` configs 全 invalid
- **Sensor file axis convention**: cylinder sensor 由 `scripts/generate_sensors_qrpivot_cylinder.py` 生成；應通過 `test_sensor_axis_convention.py`（待補 cylinder coverage）

---

## 變更紀錄

- **2026-05-30 CEXP-036 RAR collocation 結果（neutral）**:
  - CEXP-036 = CEXP-002 + `physics_collocation_strategy: random→rar`（freq=1000, pool=640, num_physics_points 維持 64）。Slurm job 3745 train + 3760 eval。
  - 結果：KE **3.66%**（baseline 3.54%，持平略差）, ke_pred/ref 1.039, omega 2.18, div 3.62。
  - 訓練健康：w_ns_u 最終 0.16（SOAP+RAR freq=1000 未觸發 EXP-053 爆漲，再次驗證 EXP-054 freq≥1000 下限）。
  - 判讀：RAR「聰明放置同樣 64 collo」不崩潰（對比 CEXP-030 盲目增量 collo→610%），但也不超越 baseline → **補強 Finding #9**：physics collocation 的「量」與「放置」都非 cylinder baseline 瓶頸；瓶頸是 K=100 sensor 的資訊論上限。
  - 副作用：div L2 3.62 > baseline 1.14（3×）。RAR 把 collo 集中 wake 高 residual 區 → body/inlet/outlet 的 continuity 監督變稀疏，incompressibility 局部惡化。
- **2026-05-30 CEXP-034/035 K=200 sensor 系列提交**:
  - 生成 K=200 QR-pivot sensor（`sensors_qrpivot_K200_cylinder_Re10031.{json,npz}`），axis convention 驗證通過。
  - 分佈圖 `docs/figures/cylinder_sensor_K100_vs_K200.png`：K=200 仍集中 wake，但 upstream 8（vs K=100 的 1）、within-body-x 11（vs 0）；near-body dist<0.03 反而較少（4 vs 7）。
  - CEXP-034 = CEXP-002 + K=200（collo 64）；CEXP-035 = CEXP-034 + collo 1024（Finding #9 falsification：K=200 是否化解 CEXP-030 ill-posedness）。
  - Slurm r740 並行提交：CEXP-034 job 3740、CEXP-035 job 3741（8-CPU template 兩 job 同節點並行）。
  - **Eval 結果（job 3743，2026-05-30）**：
    - CEXP-034 (K=200, collo 64): KE **355.5%**, ke_pred/ref **4.55**, omega 46.08, div 11.69 → ❌ 災難
    - CEXP-035 (K=200, collo 1024): KE **375.4%**, ke_pred/ref **4.75**, omega 45.78, div 8.27 → ❌ 災難
    - **結論（呼應 Finding #8）**：K=200 反而比 K=100 baseline (3.54%) 惡化 100×。QR-pivot 在 K=200 時自然把 sensor 鋪到 upstream (8) 與 body 上下剪切層 (within-body-x 11)，這些 velocity supervision 與 body soft BC (u→0) 在重疊空間競爭 → GradNorm 撕裂 → over-energy 4.5×。**「更多 sensor」在有 body BC 的情況下有害，不是有益**。
    - **對 Finding #9 的補充**：CEXP-035 (K=200+collo1024, 375%) vs CEXP-030 (K=100+collo1024, 610%) 略好，暗示 sensor density 對 ill-posedness 有微弱緩解；但因 K=200 同時引爆 sensor/BC 衝突 (Finding #8)，無法乾淨隔離。collo 1024 在兩個 K 都有害。
    - **下一步若要測 K=200**：必須同時 `bc_body_n_points=0`（依 Finding #8，sensor 已覆蓋 body 區時不可再加 body BC）。當前 CEXP-034/035 是「雙約束衝突」的再次印證，非乾淨的 K-scaling 測試。
- **2026-05-29 CEXP-032/033 + Finding #8（body-region constraint 原則）**:
  - CEXP-032（QR + bc_body=0）KE **177.8%**、CEXP-033（hybrid95downstream + bc_body=0）KE **12.5%**。
  - CEXP-032 推翻先前「body BC 冗余」假說（預測 3.5%，實際 177.8%，錯 40×）；揭露完整 2×2 交互作用。
  - **Finding #8**：body 區必須恰好一個約束來源（零→污染整場、雙→GradNorm 衝突）。CEXP-002（QR + body BC）落在 2×2 最優格。
  - CEXP-033 證明 upstream sensor 非 13.1% gap 主因；gap 來自 hybrid coverage density。
  - Finding #7 標註為歷史推理（「衝突」觀察正確但框架不完整），由 Finding #8 取代。
  - 新增 sensor file `sensors_hybrid95downstream_K95_cylinder_Re10031.{json,npz}`（hybrid 移除 5 upstream）。
  - Eval jobs 3727 (032/033) completed cleanly；artifacts rsynced。
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
- **2026-05-23/24 Stage 1 全 ❌**:
  - CEXP-016 (hard BC fair) eval: **KE 111.6 %**, `ke_pred/ke_ref 2.12`, catastrophic over-predict — 4 種預期 outcome 全沒命中, 出現新 case E (catastrophic)
  - CEXP-017/018/019 三個 single-variable enabling-condition diagnostic 全部 falsified（per spec [`docs/superpowers/specs/2026-05-23-cylinder-hard-bc-enabling-conditions-design.md`](../docs/superpowers/specs/2026-05-23-cylinder-hard-bc-enabling-conditions-design.md) §4 multi-config decision table 判讀: 全 ❌ pattern → Re-diagnose）
  - **CEXP-017 顛覆 60% prior**: 5-task GradNorm 反讓 KE 推至 **303.6 %** (over-predict 4.04×), w_bc 飆 195× + w_ns_u 飆 382×
  - **CEXP-018 (H2 body_aware) ≈ CEXP-016**: KE 106.3 %, body_aware sampling 沒救
  - **CEXP-019 (H3 dense BC)**: KE 139.3 %, dense BC supervision 微更糟
  - **Finding 4 新增**: Hard BC + standard PI-CON architecture (SOAP + ScheduleFree + GradNorm) **fundamental incompatibility**, CEXP-010 KE 17.5 % 是 multi-confound accidental survival (5k iter 不夠長, 10k iter 公平比較全 catastrophic)
	- **Root cause hypothesis (2026-05-24)**: Trunk net 完全沒有 geometry awareness — hard BC 只是 output post-hoc gate, NN 不知道 boundary 在哪 → over-compensation 機制
	- **Stage 2 redirect (Option A)**: 加 SDF `φ` 進 trunk input concat (`query = [x, y, t, c, φ]`), 移除 hard BC gate; raw scalar concat, hard BC off (per user 2026-05-24 decision). 需先修 src `DEFAULT_PICON_ARGS` 缺失 key (~30 line patch). Options E (cross-attn geometry tokens) + F (geometry-conditioned hypernetwork) 列 long-term future paper material
	- [INDEX] CEXP-016/017/018/019 entries finalized, [RECORD] 4 個 detail tables 新增, [STATE] Open Questions Stage 2 plan 寫入
- **2026-05-27 CEXP-023 B+C geometry-aware 準備**:
  - 實作 opt-in flags：`use_graph_spatial_encoder`（B：sensor tokens 進 CfC 前聚合 geometry nodes）與 `use_trunk_geo_context`（C：trunk query 讀 geometry memory）。
  - 預設 false 時不建立 `graph_spatial_*` / `trunk_geo_*` 參數，舊 Kolmogorov / cylinder baseline checkpoint 相容；啟用但未注入 geometry positions 會 fail fast。
  - 新增 `configs/exp_cylinder_023_graph_spatial_trunk_geo.toml`，從 CEXP-002 派生，唯一架構變動為 B+C；完整 10k 訓練與 eval 尚未執行。
  - 驗證包含 unit / equivalence / config model smoke：`tests/test_geometry_graph_features.py`、`tests/test_optimization_equivalence.py`、`tests/test_cfc_pass_refactor.py`、`tests/test_make_picon_model_fn_cache.py`。
  - 本地 1-step CPU train smoke 已撤回為無效實驗證據；artifact `artifacts/_smoke/cylinder023-graph-spatial-trunk-geo` 已移除。CEXP-023 正式 smoke / train / eval 必須在 lab-server 執行。
  - lab-server 啟動紀錄：job 3674 因 config 仍指向本機 RealPDEBench path 而 fail fast；`arrow_shards` 修正為 `/home/junyi/RealPDEBench/...` 後重送 job 3675。job 3675 已在 acmt20 / RTX 3090 上進入訓練，Step 1 完成且 stderr empty。
  - 訓練完成紀錄：job 3675 completed cleanly (`ExitCode=0:0`, elapsed 01:47:39)。Final Step 10000: `L_data=4.4321e-03`, `L_phys=9.7689e-03`, `L_total=1.1669e-02`。需執行 eval 後才能判讀物理品質。
- **2026-05-27 CEXP-024/025 ablation 準備**:
  - 新增 CEXP-024 B-only：`configs/exp_cylinder_024_graph_spatial_only.toml`，只啟用 `use_graph_spatial_encoder=true`。
  - 新增 CEXP-025 C-only：`configs/exp_cylinder_025_trunk_geo_only.toml`，只啟用 `use_trunk_geo_context=true`。
  - 三組 CEXP-023/024/025 皆使用 lab-server RealPDEBench path、CEXP-002 訓練設定、r740 single-GPU Slurm template；差異只限 geometry flags 與 artifact path。
  - 提交狀態：CEXP-023 job 3675、CEXP-024 job 3676、CEXP-025 job 3677 皆 train done pending eval。
  - CEXP-024 訓練完成紀錄：job 3676 completed cleanly (`ExitCode=0:0`, elapsed 01:46:40)。Final Step 10000: `L_data=2.6593e-03`, `L_phys=8.8372e-03`, `L_total=5.9508e-03`。需執行 eval 後才能判讀物理品質。
  - CEXP-025 訓練完成紀錄：job 3677 completed cleanly (`ExitCode=0:0`, elapsed 01:46:38)。Final Step 10000: `L_data=4.9786e-03`, `L_phys=8.6168e-03`, `L_total=1.4051e-02`。需執行 eval 後才能判讀物理品質。
- **2026-05-27 CEXP-023/024/025 eval 啟動**:
  - 直接在 lab-server head node 用 `--device cuda` 跑 CEXP-023 eval 時 fail fast：head node 暴露 GTX 1050，與目前 PyTorch CUDA arch 不相容；partial eval output 已移除。
  - 初次 Slurm eval jobs 3678/3679/3680 fail fast：`evaluate_cylinder.py` 未同步 training path 的 geometry injection，導致 `use_graph_spatial_encoder` / `use_trunk_geo_context` 時 geometry_pos 為空。這是 evaluator 缺口，未作為模型結果。
  - 修補 `scripts/evaluate_cylinder.py`：重建 `CylinderDataset` 後，若 geometry-aware flags 開啟，注入 `ds.body_xy` 到 `model.set_geometry_tokens(...)`；缺 `body_xy` 時 fail fast。
  - 重送 Slurm r740 single-GPU eval jobs：CEXP-023 job 3681、CEXP-024 job 3682、CEXP-025 job 3683。
  - Eval output dirs: `cylinder-eval-step10000/` under each experiment artifact。
- **2026-05-27 CEXP-023/024/025 eval 結果**:
  - CEXP-023 B+C: `ke_rel_err_mean=365.4%`, `ke_pred/ke_ref=4.65`, `omega_rmse=34.60`, `div_l2=9.93` → `NEGATIVE_RESULT`。
  - CEXP-024 B-only: `ke_rel_err_mean=458.6%`, `ke_pred/ke_ref=5.59`, `omega_rmse=40.55`, `div_l2=8.69` → `NEGATIVE_RESULT`。
  - CEXP-025 C-only: `ke_rel_err_mean=401.1%`, `ke_pred/ke_ref=5.01`, `omega_rmse=32.60`, `div_l2=11.68` → `NEGATIVE_RESULT`。
  - 判讀：training loss 低不代表 physical correctness；B/C/B+C geometry memory 全部造成 severe over-energy，未解決 CEXP-020/SDF path 的 adversarial geometry prior 問題。
- **2026-05-27 CEXP-026/027 control 設計**:
  - 新增 `geometry_preserve_base_rng`：建立 optional geometry modules 後還原 torch RNG，確保後續 baseline layers 初始化不受影響。
  - 新增 `use_graph_spatial_gate`：B path 改成 `tokens + tanh(gate) * graph_message`，gate 初始 0；預設 false 以維持 CEXP-024 舊語義。
  - CEXP-026：C-only + RNG-neutral，用來驗證 CEXP-025 是否主要是 RNG/init confound。
  - CEXP-027：B-only + zero gate + RNG-neutral，用來驗證 CEXP-024 是否主要是 ungated graph residual 初始擾動。
  - Slurm 提交：CEXP-026 job 3685、CEXP-027 job 3684；皆使用既有 r740 single-GPU train template。
- **2026-05-27 CEXP-027 train 完成**:
  - Slurm job 3684 completed，stderr 空。
  - Final training line: step 10000, `L_data=3.1957e-03`, `L_phys=1.4935e-02`, `L_total=8.1626e-03`。
  - Checkpoint: `artifacts/cylinder/deeponet-cfc-cylinder-exp027-graph-spatial-zero-gate/checkpoints/picon_kolmogorov_step_10000.pt`。
  - Eval 暫等 CEXP-026 完成後一起送出，避免半套結果造成錯誤比較。
- **2026-05-27 CEXP-026 train 完成與 026/027 eval 提交**:
  - CEXP-026 Slurm job 3685 completed，stderr 空。
  - CEXP-026 final training line: step 10000, `L_data=3.5730e-03`, `L_phys=1.1662e-02`, `L_total=9.1182e-03`。
  - Checkpoint: `artifacts/cylinder/deeponet-cfc-cylinder-exp026-trunk-geo-rng-control/checkpoints/picon_kolmogorov_step_10000.pt`。
  - Initial eval jobs 3686/3687 failed immediately because `sbatch --wrap` used `/bin/sh` and rejected `set -o pipefail`; this is a submit wrapper error, not evaluator/model evidence.
  - Eval jobs resubmitted on r740 single-GPU with explicit bash sbatch: CEXP-026 job 3688, CEXP-027 job 3689。
  - Eval output dirs: `cylinder-eval-step10000/` under each experiment artifact。
- **2026-05-27 CEXP-026/027 eval 結果**:
  - CEXP-026 C-only RNG-neutral: `ke_rel_err_mean=463.7%`, `ke_pred/ke_ref=5.64`, `omega_rmse=38.12`, `div_l2=10.61` → `NEGATIVE_RESULT`。
  - CEXP-027 B-only zero-gate RNG-neutral: `ke_rel_err_mean=489.5%`, `ke_pred/ke_ref=5.89`, `omega_rmse=38.80`, `div_l2=11.41` → `NEGATIVE_RESULT`。
  - 判讀：CEXP-026 未回到 CEXP-002，否定「CEXP-025 只是 RNG/init confound」；CEXP-027 未回到 CEXP-002 且未改善 CEXP-024，否定「B-only 主要是 ungated graph residual 初始擾動」。
  - 結論：B/C geometry memory path 的失敗根因更接近 geometry prior 注入位置與 energy amplification，而不是開關初始值或 RNG 消耗。
- **2026-05-27 boundary semantics probing（CEXP-026/027）**:
  - Dataset geometry: physical domain `Lx=0.3223`, `Ly=0.1721`；normalized body bbox `x=[0.1961,0.2902]`, `y=[0.4016,0.5827]`；measured inlet `u_inf=0.329345`。
  - Sensor coverage: K=100 sensors have `x=[0.1843,0.9176]`, `y=[0.2756,0.7008]`；near inlet `x<0.05`: 0；near outlet `x>0.95`: 0；top/bottom `y<0.05 or >0.95`: 0；upstream of body: 1；downstream of body: 99；within 0.03 of body: 0。
  - Config semantics: `use_periodic_domain=false` lets trunk distinguish upstream/downstream coordinates, but `use_hard_body_bc=false`, `use_body_distance_feature=false`, `bc_outlet_n_points=0`。Thus no explicit outlet boundary condition and no continuous SDF/hard body semantics.
  - Boundary probing CEXP-026: inlet mean `u≈0.57` vs target `0.329` (RMSE ≈0.25), outlet `v≈0.19–0.36`, outlet `|du/dx|≈5.9–8.1`, body speed RMSE ≈1.55。Sampled region KE ratios: upstream ≈9.4×, near body ≈9.2–9.6×, far wake ≈3.4–3.6×。
  - Boundary probing CEXP-027: similar; inlet mean `u≈0.55–0.57`, outlet `v≈0.16–0.27`, outlet `|du/dx|≈5.3–7.7`, body speed RMSE ≈1.57–1.60。Sampled region KE ratios: upstream ≈9.5–9.7×, near body ≈9.2–9.6×, far wake ≈3.5–3.9×。
  - 判讀：模型「能分辨座標左右」但沒有學到 inlet/outlet/body 的 CFD semantic role；B/C body point memory 無法補足 sensor coverage 與 boundary-type prior，反而造成全域 energy amplification。
- **2026-05-27 CEXP-028 hybrid20qr80 baseline 啟動**:
  - 新增 `configs/exp_cylinder_028_hybrid20qr80_baseline.toml`，由 CEXP-002 派生，唯一實驗變數為 sensor placement：pure QR → `sensors_hybrid20qr80_K100_cylinder_Re10031`。
  - Lab smoke: `sensor_pos=(100,2)`, `sensor_vals=(100,200,2)`, `near_inlet=2`, `near_outlet=2`, `top_bottom=8`, trainable params `3,138,634`。
  - Slurm train submitted on r740 single-GPU: CEXP-028 job 3690。
- **2026-05-27 CEXP-028 train 完成與 eval 提交**:
  - Slurm job 3690 completed，stderr 空。
  - Final training line: step 10000, `L_data=4.2938e-03`, `L_phys=1.3326e-02`, `L_total=1.6004e-02`。
  - Checkpoint: `artifacts/cylinder/deeponet-cfc-cylinder-exp028-hybrid20qr80-baseline/checkpoints/picon_kolmogorov_step_10000.pt`。
  - Eval job submitted on r740 single-GPU with explicit bash sbatch: CEXP-028 job 3691。
  - Eval output dir: `artifacts/cylinder/deeponet-cfc-cylinder-exp028-hybrid20qr80-baseline/cylinder-eval-step10000/`。
- **2026-05-27 CEXP-028 eval 結果**:
  - Eval job 3691 completed cleanly，stderr 空；summary path `artifacts/cylinder/deeponet-cfc-cylinder-exp028-hybrid20qr80-baseline/cylinder-eval-step10000/summary.json`。
  - Metrics: `ke_rel_err_mean=154.4%`, `ke_rel_err_late=151.6%`, `ke_pred/ke_ref=2.54`, `u_rmse=0.3865`, `v_rmse=0.1806`, `omega_rmse=44.90`, `div_l2=14.48`。
  - 判讀：hybrid20qr80 將 B/C 系列的 severe over-energy（4.65–5.89×）降到 2.54×，說明 pure QR downstream-only coverage 是根因之一；但相對 CEXP-002（KE 3.54%, `ke_pred/ke_ref=1.01`）仍是 `[RESULT: PHYSICAL_FAILURE]`，且 vorticity/divergence 未改善。下一步不應只增加 generic geometry memory，需 explicit boundary semantic 或更強 BC/field-consistency constraint。
- **2026-05-28 CEXP-022 cross-attn geometry tokens + hard BC 結果**:
  - CEXP-022 Slurm job 3671 completed cleanly (`ExitCode=0:0`, elapsed 01:42:12)。
  - Training log 摘要：w_ns_u = 0.064 (step 1000) → 0.730 (step 5000) → **2.09 (step 10000)**；與 CEXP-016 (hard BC alone) 完全相同的 GradNorm 爆炸模式。
  - Eval job 3692 (step_10000.pt): `ke_rel_err_mean=98.2%`, `u_rmse=0.245`, `omega_rmse=12.55`, `div_l2=6.25`。
  - Eval job 3693 (final.pt): `ke_rel_err_mean=99.8%`, `u_rmse=0.245`, `omega_rmse=12.45`, `div_l2=6.11`。
  - 452 body surface points 成功注入（`geometry_context: 452 body surface points injected`），hard BC scale=0.7517。
  - 判讀：spec §4 stop-loss zone (KE > 100%)；geometry tokens 輕微改善 (99.8% vs 111.6%) 但 GradNorm 病態機制不變；Finding #6 written；hard BC 路線全封閉。
  - Artifact: `artifacts/cylinder/deeponet-cfc-cylinder-exp022-geometry-tokens/` (summary.json + summary_final.json rsynced)。
- **2026-05-28 CEXP-029 no-GNN boundary semantic probe 準備**:
  - 新增 `configs/exp_cylinder_029_hybrid20qr80_outlet_bc.toml`，由 CEXP-028 派生，唯一變動為新增 soft outlet BC：`bc_outlet_n_points=32`。
  - Hypothesis: 若 CEXP-028 的 over-energy 主要來自 outlet semantic 缺口，CEXP-029 應降低 `ke_pred/ke_ref=2.54` 並改善 outlet probing；若 KE 降但 `omega_rmse` 惡化，視為數值耗散而非物理解。
  - Lab smoke passed：`sensor_pos=(100,2)`, `sensor_vals=(100,200,2)`, `bc_outlet_n_points=32`, trainable params `3,138,634`。
  - Slurm train submitted on r740 single-GPU: CEXP-029 job 3694。
- **2026-05-28 CEXP-029 train 完成與 eval 提交**:
  - Slurm job 3694 completed，stderr 空。
  - Final training line: step 10000, `L_data=2.5164e-03`, `L_phys=1.7861e-02`, `L_total=8.3290e-03`。
  - 訓練中 step 8000 曾出現 transient physics spike：`L_phys=8.1836e+00`, `L_total=3.7722e+00`；step 9000 已恢復到 `L_phys=7.3528e-02`。
  - Checkpoint: `artifacts/cylinder/deeponet-cfc-cylinder-exp029-hybrid20qr80-outlet-bc/picon_kolmogorov_final.pt`。
  - Eval job submitted on r740 single-GPU with explicit bash sbatch: CEXP-029 job 3702。
  - Eval output dir: `artifacts/cylinder/deeponet-cfc-cylinder-exp029-hybrid20qr80-outlet-bc/cylinder-eval-step10000/`。
- **2026-05-28 CEXP-030 設計與提交**:
  - 新增 `configs/exp_cylinder_030_collo1024.toml`，由 CEXP-002 派生，唯一變動為 `num_physics_points: 64 → 1024`。
  - Hypothesis: 更密的 physics collocation 應降低 div L2（目前 1.14）並可能改善 KE rel-err。Falsifiability: KE > 5% → over-regularization；div L2 > 1.2 → 無改善。
  - Slurm train submitted on r740 single-GPU: CEXP-030 job 3695（PD，等 CEXP-029 job 3694 釋出）。
