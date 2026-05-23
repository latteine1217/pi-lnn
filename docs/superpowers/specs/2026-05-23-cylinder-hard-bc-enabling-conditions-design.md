# Design — Cylinder Hard BC Enabling Conditions Diagnostic (Stage 1)

| Field | Value |
|---|---|
| Date | 2026-05-23 |
| Status | Approved（user sign-off Section 1-5）|
| Owner | latteine |
| Related state | [docs/cylinder_log_v2.md](../../cylinder_log_v2.md) |
| Related skill | superpowers:brainstorming → writing-plans (next) |

---

## 1. Goal & Scope

**Goal**: 識別 hard body BC (Sukumar 2022 output transformation) 在 PI-CON architecture 下需要哪些 **enabling conditions** 才不會 catastrophic fail。

**Background**: CEXP-016 (hard BC ON, 其餘對齊 CEXP-002 baseline) 訓練後 eval 出 **KE rel-err 111.6 %**, `ke_pred/ke_ref = 2.12`（over-predict 2.12×）, `w_ns_u` GradNorm 推 209× — 表面 fair comparison 反而 expose hidden interaction。CEXP-010 (multi-confound hard BC, KE 17.5 %) 反而比 CEXP-016 好 6×, 暗示 multi-confound 中有「necessary enabling conditions」。

**Scope reframing**（從原本「重跑所有種類」→ 分階段執行）:
- **Stage 1（本 spec）**: 3 個 single-variable diagnostic configs 解 CEXP-016 catastrophic fail 機制
- **Stage 2（未來 session）**: 在 enabling conditions 確認後做完整 5-axis sweep

**Why 縮 scope**: 直接做 5-axis isolated 可能 5 個 catastrophic fail, spec 無 paper-grade output. 先理解 mechanism 才能設計有意義的 sweep。

**Success criteria**:
- 至少 1 個 Stage 1 config 把 KE 從 CEXP-016 的 111.6 % 拉回 baseline 量級 (KE < 10 %)
- 識別 hard BC 的「最小 working setup」(minimal enabling conditions set)
- 過程不依賴 multi-confound 救援 — 必須是 paper-grade clean attribution

**Non-goals**:
- 不打算超越 CEXP-002 baseline KE 3.5 %（geometry awareness 是品質保險, 不是性能 unblocker）
- 不在本 spec 內做 multi-seed（single seed = 42 first-pass）
- 不在本 spec 內動 axis 3 (SDF input feature) 與 axis 5 (cross-attn)（src dev 留 Stage 2）

---

## 2. Past Failure Modes & Design Rationale

下表是 cylinder CEXP-001~016 失敗模式的歸納, 每個都對應 Stage 1 一個 enabling condition 假設。

| Exp | Setup | KE rel-err | Failure mode | Stage 1 對應 hypothesis |
|---|---|---|---|---|
| CEXP-001 | 無 BC | 51 % | 來流 u→0 collapse (sensor 集中尾跡, 無 inflow supervision) | → soft inflow BC 必要（baseline 已含, 三個 H 都保留）|
| CEXP-007 | distance feature input, iter=3k | 32.7 % | `use_body_distance_feature` key **沒落到 main DEFAULT_PICON_ARGS** → 靜默忽略 + iter 太短 + 5-task GradNorm 壓 BC | → Src bug, Stage 1 不修（留 Stage 2 axis 3） |
| CEXP-010 | hard BC + body_aware + 5-task GN + bc_body 96 + bc_outlet 32 + iter=5k | 17.5 % | Multi-confound, 無法歸因; 但這是 hard BC 唯一**沒 catastrophic** 的 setup | → 「multi-confound 其實是 enabling conditions」 |
| CEXP-013 | Re=1781 small capacity + hard BC | 46.3 % | Mean-flow collapse (`ke_pred/ke_ref=0.54`); Re=1781 wake 是 15 % perturbation, trivial 解 saturate data loss | → Re=10031 主線不會有, 但暗示 hard BC + low SNR 不合 |
| **CEXP-016**（today） | **hard BC + soft body + iter=10k + 4-task GN + random sampling**（fair single-var）| **111.6 %** | **Catastrophic over-predict** `ke_pred/ke_ref=2.12`; `w_ns_u` GradNorm 推 209× → physics dominate → wake `NN_u` 失控 | → 解析此 fail 是 Stage 1 主任務 |

### 三個 enabling condition hypothesis

**H1: 5-task GradNorm**（BC weight 納入動態平衡）
- 機制：CEXP-016 用 4-task `[data, ns_u, ns_v, cont]`, BC loss 是 fixed weight 0.1 **不進 GradNorm**。Hard BC gate 後 wake 區 NN_u 需大值補償 → physics loss 大 → GradNorm 只看 4 個 task → 把 `w_ns_u` 推爆 209×。
- 若 5-task `[data, ns_u, ns_v, cont, bc]`, GradNorm 把 BC weight 一起平衡, physics 不會 dominate。
- Falsify: 若 CEXP-017 (5-task GN) KE > 30 % → 5-task 不是 enabling condition。

**H2: Body-aware collocation sampling**（30 % near-body + 70 % uniform）
- 機制：Hard BC gate 在 boundary layer 內壓制 NN_u 量級, random sampling 只 ~7 % 落 near-body → boundary layer gradient 學不好 → physics residual 在 near-body 高 → GradNorm 推 physics weight。
- 若 body_aware, physics 在 boundary 區 gradient 充分採樣, physics residual 自然下降, GradNorm 不需 over-correct。
- Falsify: 若 CEXP-018 (body_aware) KE > 30 % → body_aware 不是 enabling condition。

**H3: 加密 BC supervision points**（bc_body 64→96, bc_outlet 0→32）
- 機制：Soft body BC 點少 (64) → body 表面 supervision 稀疏 → hard BC gate 必須 carry 大部分約束 → wake 區 NN_u 補償壓力大。
- 若加密 (bc_body 96 + bc_outlet 32 覆蓋整個 body 邊界 + outlet), soft BC supervise 更密, hard BC 不需獨自扛。
- Falsify: 若 CEXP-019 (加密 BC) KE > 30 % → BC density 不是 enabling condition。

### Catastrophic over-predict 機制（為何 `ke_pred/ke_ref=2.12`）

從 CEXP-016 training log:
```
step 1:     L_data 7.37,    L_phys 0.083,  w_ns_u 0.01
step 1000:  L_data 0.049,   L_phys 1.03,   w_ns_u 0.068   ← physics loss 暴增 12×
step 5000:  L_data 2.8e-3,  L_phys 0.016,  w_ns_u 0.70
step 10000: L_data 2.5e-3,  L_phys 9.5e-3, w_ns_u 2.09    ← physics dominate
```

物理解釋：
1. step 1-1000: Hard BC gate 強制 body 區 = 0, NN_u 在 wake 區必須大 → NS residual 也大（NN gradient 量級大）
2. GradNorm 看到 ns_u residual 比 data 大 → 推 `w_ns_u` 上升
3. `w_ns_u` ↑ → optimizer 推 NN_u 更接近 NS 滿足解 → 系統「以為」可用大 u + 大 viscous 達到 NS balance
4. 結果：u 預測偏大 (over-predict ~2×), KE 整體高估

**這是 GradNorm pathology, 不是 hard BC 本質有問題**。H1 直接針對此 mechanism, H2/H3 是 indirect compensation。

---

## 3. Config Matrix

3 個 single-variable configs, 每個 vs CEXP-016 改動唯一一個 condition。並行跑（acmt20 同節點 2-job 上限 → 跑 2 + 1）。

| Config | Hard BC | GradNorm tasks | Collocation | bc_body | bc_outlet | iter | Hypothesis 隔離變數 |
|---|---|---|---|---|---|---|---|
| CEXP-002 (legacy, reference) | ❌ | 4-task | random | 64 | 0 | 10k | (baseline KE 3.5 %) |
| CEXP-016 (done) | ✅ | 4-task | random | 64 | 0 | 10k | (control: catastrophic 111.6 %) |
| **CEXP-017** | ✅ | **5-task** ⬅ | random | 64 | 0 | 10k | **H1: 5-task GradNorm** |
| **CEXP-018** | ✅ | 4-task | **body_aware** ⬅ | 64 | 0 | 10k | **H2: body-aware sampling** |
| **CEXP-019** | ✅ | 4-task | random | **96** ⬅ | **32** ⬅ | 10k | **H3: BC density** |

**統一**:
- 全部基於 CEXP-016 (hard BC ON) 出發, 只改 1 個 column
- `soft body BC` 在所有 configs 都是 ON（per cylinder pitfall, 非週期域 BC loss 必要）
- iter / d_model / sensor / seed = 42 / Re = 10031 / SOAP + ScheduleFree 全對齊
- CEXP-002 (no hard BC) 是 reference baseline, 不重跑（既有 metrics 已可用）

**5-task GradNorm tasks (H1)**: `[data, ns_u, ns_v, cont, bc]`, init weights `[1.0, 0.01, 0.01, 0.01, 0.1]`（與 legacy CEXP-010 一致）

**body_aware sampling (H2)**: `physics_collocation_strategy = "body_aware"`, 30 % 近 body (distance < median fluid SDF) + 70 % uniform（src 已實作 in `src/cylinder_dataset.py:371-379`）

**BC density (H3)**: `bc_body_n_points = 96`（從 64 增 50 %）+ `bc_outlet_n_points = 32`（從 0 新增）

### Stacking 規劃（後續, 若需要）

若 H1/H2/H3 isolated 都未把 KE 拉回 10 % 內, **下個 session** 才考慮 pairwise stacking（CEXP-020/021/022）。本 spec **不包含 stacking jobs** — 等 isolated 結果決定。

### Config 命名 / file 結構

```
configs/exp_cylinder_017_hard_bc_5task_gn.toml   # H1
configs/exp_cylinder_018_hard_bc_body_aware.toml # H2
configs/exp_cylinder_019_hard_bc_dense_bc.toml   # H3
```

---

## 4. Falsifiability Gates per Config

每個 config 訓練後 eval 出 KE / div / `ke_pred_ke_ref`, 依以下 decision tree 判讀。

### 共同 metrics 與 thresholds

| Metric | "Healthy" 範圍（vs CEXP-002 baseline 3.54 %）| 解釋 |
|---|---|---|
| `ke_rel_err_mean` | < 10 % (recovery), 10-30 % (partial), > 30 % (catastrophic) | KE 全頻段誤差 |
| `ke_pred / ke_ref` | 0.85-1.15 (健康), < 0.7 or > 1.3 (collapse / over-predict) | Trivial / over-predict 診斷量 |
| `div_l2_mean` | < 2× CEXP-002 (~1.14) | Incompressibility |
| `omega_rmse_mean` | < 2× CEXP-002 (~2.14) | Wake structure |
| GradNorm `w_ns_u` final | < 5× CEXP-002 (~0.108) → < 0.5 | Physics dominance 診斷 |

### Per-config decision tree

**CEXP-017 (H1: 5-task GradNorm)**

| Outcome | Threshold | Interpretation | Next action |
|---|---|---|---|
| ✅ H1-A | KE < 10 % AND `w_ns_u_final` < 0.5 | 5-task GradNorm fixes catastrophic — BC weight 納入動態平衡阻止 physics dominance | Stage 1 結束, spec promote 為 paper claim, plan Stage 2 axis sweep |
| 🟡 H1-B | KE 10-30 % | Partial fix — GradNorm 路徑改善但仍有其他 issue | Combine with H2 或 H3 (Stage 1b stacking) |
| ❌ H1-C | KE > 30 % OR `ke_pred/ke_ref` > 1.5 | Falsified — GradNorm 不是 root cause | Stop loss; 重新 hypothesize（SOAP / ScheduleFree / hard BC 更深層 incompatibility）|

**CEXP-018 (H2: body-aware sampling)**

| Outcome | Threshold | Interpretation | Next action |
|---|---|---|---|
| ✅ H2-A | KE < 10 % AND `w_ns_u_final` < 0.5 | Boundary-layer gradient signal 充足後 physics residual 自然下降, GradNorm 不爆 | Stage 1 結束 |
| 🟡 H2-B | KE 10-30 % | Sampling 改善 boundary 但其他 axis 也需要 | Combine with H1 |
| ❌ H2-C | KE > 30 % | Falsified — body-aware 不解 catastrophic | — |

**CEXP-019 (H3: BC density)**

| Outcome | Threshold | Interpretation | Next action |
|---|---|---|---|
| ✅ H3-A | KE < 10 % | 加密 soft BC supervision 補強 hard BC 不獨扛, physics balanced | Stage 1 結束 |
| 🟡 H3-B | KE 10-30 % | Partial | Combine |
| ❌ H3-C | KE > 30 % | Falsified — BC density 不夠 | — |

### Multi-config outcomes 判讀

3 個 configs 各跑完後：

| Pattern | Interpretation | Stage 2 strategy |
|---|---|---|
| 1 個 ✅ A, 2 個 ❌/🟡 | 單一 enabling condition 充分 → paper claim "X 是必要 + 充分" | Stage 2 該 condition + 各 axis sweep |
| 2-3 個 ✅ A | 多個獨立 condition 都 work → 都不是 unique necessary | Pick 最簡單 condition 進 Stage 2 baseline, 其他作 alternative |
| 全 🟡 | 沒有單一 condition 足夠, 需要 stacking | Stage 1b: pairwise (CEXP-017+018, 017+019, 018+019) |
| 全 ❌ | 三個 hypothesis 都錯 — fundamental incompatibility | Re-diagnose (e.g., 換 optimizer, 換 architecture, 移除 SOAP precondition) |

### Stop-loss 明確規則

訓練中**不**啟動 early stop（training 1.6 hr 不長, 跑完看 final metrics 較有 information）。Stop loss 是 **spec-level**：

- 若全部 3 config catastrophic (KE > 30 %) → **不**進入 Stage 2, 回 brainstorming 重新 design
- 若 stacking 也救不回 (Stage 1b 全失敗) → paper 改寫成「Hard BC 在 PI-CON 主訴 K=100 sparse 配置下 fundamental incompatible, 改 soft-only 路徑 (CEXP-002) 就好」的 negative finding

### Expected outcome 機率分配（personal priors）

- H1 (5-task GradNorm) confirmed: 60 % — 與 training log `w_ns_u` 推 209× 觀察最直接對應
- H2 (body_aware) confirmed: 25 % — 間接補強 mechanism
- H3 (BC density) confirmed: 15 % — 最 indirect

---

## 5. Implementation Prerequisites & Workflow

### Pre-requisites（必須在 submit 3 個 jobs 前處理）

**Src-level**

| Issue | 範圍 | Stage 1 策略 | Stage 2 處理（不在本 spec）|
|---|---|---|---|
| **ForcingPrior cylinder regression** (`A_init > 0`, `k_f ∈ (1,8)` check) | `src/pi_con/operator.py:275` 強制 attach forcing | Workaround: config 內 dummy `kolmogorov_A=1e-6, k_f=2.0`（CEXP-016 已用） | Src patch: `if kolmogorov_A > 0:` 條件 attach |
| **arrow_shards path hardcoded macOS** | `configs/*.toml` 內絕對路徑 | Lab in-place sed → `/home/junyi/RealPDEBench/...`（CEXP-016 已用） | Spec 待議：env var / repo-relative |
| **`use_body_distance_feature` key drop** (CEXP-007) | `DEFAULT_PICON_ARGS` 缺 key | **Stage 1 不需修**（H1/H2/H3 不涉及） | Stage 2 axis 3 才修 |

**Lab deployment（已驗證）**

- ✅ Lab `/home/junyi/RealPDEBench/.../data-00000-of-00092.arrow` 已存在 (1.57 GB rsync 完成)
- ✅ Lab `data/cylinder_sensors/sensors_qrpivot_K100_cylinder_Re10031_{json,npz}` 已存在
- ✅ Lab `.venv` (Python 3.12 + torch 2.7.1+cu118) 已可用
- ✅ SLURM r740 partition acmt20 (RTX 3090) 已驗證可跑 cylinder pipeline

### Workflow（每個 config 共通）

```
1. 本地：複製 CEXP-016 config → exp_cylinder_0XX_*.toml + sed 改 single var + commit + push
2. Lab：git pull + sed arrow_shards path (lab-only edit, not committed)
3. Lab：scripts/slurm/submit_exp.sh cylinder_0XX configs/exp_cylinder_0XX_*.toml
4. Wait：~1.6 hr training
5. Lab：跑 evaluate_cylinder.py 在 head node (~5 min) 產 summary.json
6. Rsync：artifacts 回本地
7. 判讀：對照 §4 decision tree
8. 更新 cylinder_log_v2.md [INDEX] 加 row + [RECORD] section
```

3 個 configs 可 **parallel 跑 (acmt20 同節點 2-job limit)**：先 submit CEXP-017+018 並行, 等 free slot submit CEXP-019。

### 與 cylinder_log_v2.md 整合

每個 config 完成後 update v2 log：
- `[INDEX]` Cylinder Active 表加 1 row (status = ACTIVE_REFERENCE 或 NEGATIVE_RESULT)
- `[RECORD]` 新增詳細 metrics 段
- `[STATE] Surprise Findings` #6 更新 (Stage 1 結論)
- `[STATE] Open Questions` Stage 2 plan 寫進去

### Out-of-scope（明確不做）

- ❌ Multi-seed n = 3/5（single seed = 42 only per user decision）
- ❌ Axis 3 (SDF input feature) — Stage 2 才動, 需先 src 修 key drop bug
- ❌ Axis 5 (geometry-aware cross-attn) — Stage 2 才動, 需 100+ line src dev
- ❌ Stacking pairs (e.g., 5-task + body_aware) — 等 3 isolated 結果 → Stage 1b 才考慮
- ❌ Re=1781 collapse 路徑 (EXP-015 仍 deferred)
- ❌ CLAUDE.md READ_PROTOCOL update（per user 之前選擇）

### Spec deliverables

| Item | Path |
|---|---|
| Design doc (本 spec) | `docs/superpowers/specs/2026-05-23-cylinder-hard-bc-enabling-conditions-design.md` |
| Config files (產生於 implementation 階段) | `configs/exp_cylinder_017_hard_bc_5task_gn.toml`<br>`configs/exp_cylinder_018_hard_bc_body_aware.toml`<br>`configs/exp_cylinder_019_hard_bc_dense_bc.toml` |
| Updated state | `docs/cylinder_log_v2.md` (results 寫回) |

### Total scope estimate

| Phase | Wall time | GPU |
|---|---|---|
| Spec write + review | 15 min | — |
| Config files create + commit | 10 min | — |
| Lab submit + train (parallel 2-job) | ~3.2 hr | RTX 3090 |
| Eval + rsync + update v2 | 30 min | — |
| **Total** | **~4 hr** | **~5 GPU-hr** |

---

## Next Steps

按 brainstorming skill 終點：spec approved 後, **invoke `superpowers:writing-plans` skill**（不是 `frontend-design` / `mcp-builder` 等）產生 detailed implementation plan, 描述 each config 的具體 sed commands、commit messages、SLURM submission orders、eval invocations。
