# 實驗紀錄 v2（Stable Phase State 主檔）

> **Status**: Stable phase（2026-05-19 啟用）。研究已脫離前期探索（EXP-001~106），進入主線收斂、論文寫作、多 seed 統計確認階段。
>
> **Scope**: 此檔負責 **EXP-200 起所有 stable phase 實驗** 的 state 紀錄。Legacy EXP-001~106 已全部移至 [`docs/archive/`](archive/)，不動。
>
> **Numbering convention（穩定階段）**:
> - `EXP-200~219`: Architecture baselines + ablations
> - `EXP-220~229`: Sensor placement ablation
> - `EXP-230~239`: Re=1000 reference baselines
> - `EXP-240~299`: 預留給後續 ablation / inference benchmark / robustness study
> - **Multi-seed naming**: `exp_{NNN}_a` ~ `exp_{NNN}_e`（最多 5 seeds），對應 `seed=42, 1, 2, 3, 4`

---

## [STATE] Read Order（穩定階段優先）

| 檔 | 內容 | 何時讀 |
|---|---|---|
| **本檔** `docs/experiment_log_v2.md` | **唯一 active 主檔** — Stable phase STATE/INDEX、legacy 對照表 | **任何實驗變更前都讀這個** |
| [`docs/archive/experiment_log.md`](archive/experiment_log.md) | Legacy STATE（EXP-001~106 結論層）| 若 stable phase 結論不足，往回查 |
| [`docs/archive/experiment_archive_kolmogorov.md`](archive/experiment_archive_kolmogorov.md) | EXP-001~063 詳細 RECORD | 早期實驗追溯 |
| [`docs/archive/experiment_archive_kolmogorov_post_k100.md`](archive/experiment_archive_kolmogorov_post_k100.md) | EXP-064~106 詳細 RECORD（含 v2 axis-fix）| 近期 ablation 判讀 |
| [`docs/archive/cylinder_log.md`](archive/cylinder_log.md) | Cylinder 主線 | Cylinder 任務 |
| [`docs/archive/diagnostics_log.md`](archive/diagnostics_log.md) | denorm bug + CFD-rigour Q5/Q7/Q8 + Forward CFD | 評估值質疑 |
| [`docs/adr/`](adr/) | 設計決策 | 設計權衡追溯 |
| [`docs/paper_framing_draft.md`](paper_framing_draft.md) | 論文 framing | 寫作 |

---

## [STATE] Data Version（與 legacy 一致，不變）

- DNS（Re=10000）: [`data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy`](../data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy)
- DNS（Re=1000）: [`data/kolmogorov_dns_fp64_etdrk4_Re1000_N128_T5_ds4.npy`](../data/kolmogorov_dns_fp64_etdrk4_Re1000_N128_T5_ds4.npy)
- Sensor（DNS QR-pivot K=100, Re=10000, default）: `data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100.{json,npz}`
- Sensor（Random K=100, Re=10000）: `data/kolmogorov_sensors/re10000/sensors_random_K100_N256_t0-5_si100_seed42.{json,npz}`（v2 fixed axis convention）
- Sensor（LES-informed series, Re=10000）: `data/kolmogorov_sensors/re10000/sensors_lesinformed_*.{json,npz}`（v2 fixed axis convention）
- Sensor（DNS QR-pivot K=100, Re=1000）: `data/kolmogorov_sensors/re1000/sensors_qrpivot_K100_N128_t0-5.json`

---

## [STATE] Current Baselines

### Re=10000 主線 = **`EXP-245`**（工程可遷移配置）

```
Baseline ID:  EXP-245
Config:       configs/stable/exp_245.toml → exp_245_b3_les_T50.toml
Architecture: B3 (1-head cross-attn, minimal)
Sensor:       LES_T50  (= EXP-221, real-world DNS-free placement)
Collocation:  1024
Seed:         42
KE rel-err:   6.92 %   (single seed, n=1; multi-seed n≥3 為 高優先 待補)
```

#### EXP-245 完整 paper-grade metrics

| Metric | 值 |
|---|---|
| **KE rel-err** (all / train / val) | **6.92 % / 6.72 / 7.71** |
| u rel-L2 | 14.51 % |
| v rel-L2 | 19.25 % |
| ω rel-L2 | 44.32 % |
| div L2 mean | 0.0492 (DNS floor 0.0923) |
| ek_ratio_kf @ last | — |
| Train wall-time (RTX 3090) | **1 h 19 m 53 s** |
| Eval wall-time (RTX 3090) | ~80 s / snapshot batch |
| Artifact | `artifacts/kolmogorov/deeponet-cfc-re10000-exp245-b3-les-T50/` |

#### EXP-245 vs 延伸/對照配置（all single seed=42, 1024 collo, 10k steps）

| 角色 | ID | Architecture | Sensor | KE rel-err | Δ vs baseline |
|---|---|---|---|---|---|
| **🥇 Main baseline** | **EXP-245** | B3 (1-head) | **LES_T50** | **6.92 %** | — |
| 4-head 延伸 | EXP-251 | B3 + **4-head** | LES_T50 | 6.68 % | **-0.24 pp** |
| DNS 對照（單頭）| EXP-241_b | B3 (1-head) | DNS | 5.97 % | **-0.95 pp** |
| DNS 對照（4-head 上限）| EXP-244 | B3 + **4-head** | DNS | 5.51 % | **-1.41 pp** |
| Legacy reference | EXP-200_a~e | B3 (1-head) | DNS | 10.77 ± 0.52 % (n=5) | +3.85 pp |

**Baseline 選擇邏輯**:

| 條件 | EXP-245 | 為何排除其他 |
|---|---|---|
| 工程可遷移（real-world 無 DNS）| ✅ LES_T50 | EXP-241_b/EXP-244/EXP-200_a~e 全用 DNS sensor |
| 主線 collocation density (1024) | ✅ 1024 | EXP-200_a~e 是 64 (legacy) |
| Minimal architecture（4-head 屬延伸）| ✅ 1-head | EXP-244/EXP-251 是 4-head (延伸論述) |

**延伸 quantification（vs baseline EXP-245 6.92 %）**:
- **4-head delta**: -0.24 pp（EXP-251 vs EXP-245，同 LES_T50）
- **DNS oracle gap**: -0.95 pp（EXP-241_b vs EXP-245，同 1-head）
- **4-head + DNS oracle combined upper bound**: -1.41 pp（EXP-244 vs EXP-245）

**結案更新（精準版，per EXP-241_b 1024 collo band-energy 分析, 2026-05-19）**:

| 主張 | 修訂 |
|---|---|
| K=100 = upper bound on **mid/high bands (k≥8)** | **仍成立** ✅ — band_mid/high @ t=5 ≈ 100 %（baseline 99.97 % → 1024 collo 99.99 %, 無改善）; energy spectrum 在 k~5.6 (Nyquist k_max=√(K/π)) 之後陡降至 DNS 之下 10⁻²~10⁻⁶ |
| K=100 = upper bound on **low band (k≤8)** | **被 falsify** ❌ — 64 collo physics estimator 不夠密集；1024 collo 把 band_low @ t=5 從 3.62 → 2.41 %（-34 %） |
| K=100 = upper bound on **整體 KE** | **被 falsify** ❌ — low band 佔 94.4 % 能量，low band 改善 dominate 整體 KE (10.77 → 5.97 %) |

**修正後 binding constraints 分層**:
1. **Mid/high band (k≥8)**: K=100 sensor → Nyquist 硬上限，**任何 collocation density 都無法突破**
2. **Low band (k≤8)**: collocation density 為 binding constraint, 1024 collo 仍未 saturated（vs 256 仍有 -0.9 pp 改善），可能 4096 還能再降
3. **div L2 / NS residual consistency**: collocation density 直接決定 estimator 密度 → 64→1024 collo 讓 div 從 0.184 → 0.046（-75 %）

### Re=1000 主線（EXP-230 reference baseline）

| 項目 | 現況 |
|---|---|
| Baseline ID | `EXP-230` |
| Config | `configs/stable/exp_230.toml`（symlink → legacy EXP-030）|
| KE rel-err | 9.61 % |
| u RMSE | 5.68e-2 |
| amp ratio | 1.027 |

---

## [STATE] 主線固定假設（Re=10000 K=100 stable phase）

- 觀測 supervision 僅使用 `u, v`（無 ω）
- Physics 使用 primitive `momentum + continuity`
- 空間編碼: `LearnableFourierEmb`(`embed_dim=128`, init σ=2.0)
- `output_head_gain = 1`
- `use_temporal_anchor = true`（`n_harmonics=2`）
- `XLarge` size（d=256）
- Forcing `k_f = 2`
- `time_marching = true`
- Optimizer: SOAP + Schedule-Free（lr=1e-3, betas=(0.9, 0.999), precond_freq=2, warmup=2000, step_decay）
- GradNorm: 4 tasks `[data, ns_u, ns_v, cont]`, init `[1, 0.057, 0.057, 0.01]`, freq=1000, momentum=0.9
- AL-continuity: ρ=0.1, λ_clip=10, freq=100, ema=0.5（`al_allow_cont_in_gradnorm = true`）
- `use_physics_denormalization = false`（Kolmogorov 預設；與 d62e698 前 byte-aligned）
- 訓練 1-shot 10000 步，禁用 `resume_checkpoint`（EXP-082 災難根因）

---

## [INDEX] Stable Phase Active

### Architecture group（EXP-200~205）

| ID | Status | 架構 | 一句結論 | KE rel-err |
|---|---|---|---|---|
| **EXP-200** _a-e_ | `ACTIVE_BASELINE` | B3 (Full: CfC + cross-attn) | Re=10000 主線, n=5 multi-seed | **10.77 ± 0.52 %** |
| **EXP-201** _a-e_ | `ACTIVE_REFERENCE` | B0 (vanilla DeepONet) | Architectural ablation, n=5 multi-seed | 18.52 ± 0.66 % |
| **EXP-202** | `ACTIVE_REFERENCE` | B1 (CfC, no cross-attn) | n=1 ablation | 14.65 % |
| **EXP-203** | `ACTIVE_REFERENCE` | B2 (cross-attn, no CfC) | n=1 ablation | 13.62 % |
| **EXP-204** | `ACTIVE_REFERENCE` | Standard PINN (SiLU) | Single-instance PINN baseline | 38.50 % |
| **EXP-205** | `ACTIVE_REFERENCE` | Standard PINN (tanh) | Single-instance PINN baseline | 39.80 % |

### Architecture × Placement 2D ablation group（EXP-240, 2026-05-19 完成）

| ID | Status | 架構 + Placement | KE rel-err | Train wall (RTX 3090) | 角色 |
|---|---|---|---|---|---|
| **EXP-240_a** | `ACTIVE_REFERENCE` | B0 + LES_T50 (seed=42) | **19.58 %** | 24:05（並行）| B0 LES placement transfer 證據 |
| **EXP-240_b** | `ACTIVE_REFERENCE` | B0 + Random (seed=42) | **21.82 %** | 28:30（並行）| B0 placement-agnostic 對照 |

完整 2×3 表見 `[STATE] Architecture × Placement 2×3 完整表` section。

### Architecture × Sensor sweep at 1024 collocation（EXP-244 + EXP-245~250, 2026-05-20 完成）— **延伸論述 group**

全 1024 collo + seed=42 對齊比較。EXP-244 為 4-head cross-attn DNS baseline; EXP-245~250 為 6 個 architecture 配 EXP-221 LES_T50 sensor（real-world DNS-free）。**Stable phase 主 baseline 仍是 EXP-200_a~e (n=5 multi-seed)**；本 group 為「collo + 4-head + LES placement」延伸論述。

| ID | Status | Architecture | Sensor | KE rel-err | div L2 | Train wall (RTX 3090) |
|---|---|---|---|---|---|---|
| **EXP-244** | **`ACTIVE_BASELINE` 🥇** | B3 + **4-head** | DNS | **5.51 %** | 0.044 | 1:16:40 |
| EXP-245 | `ACTIVE_REFERENCE` | B3 (1-head) | **LES_T50** | **6.92 %** | 0.049 | 1:19:53 |
| EXP-246 | `ACTIVE_REFERENCE` | B0 (vanilla) | LES_T50 | 9.96 % | 0.056 | 0:24:58 |
| EXP-247 | `ACTIVE_REFERENCE` | B1 (no cross-attn) | LES_T50 | 10.62 % | 0.068 | 0:52:43 |
| EXP-248 | `ACTIVE_REFERENCE` | B2 (no CfC) | LES_T50 | 8.43 % | 0.053 | 0:46:59 |
| EXP-249 | `ACTIVE_REFERENCE` | Standard PINN SiLU | LES_T50 | 10.13 % | **0.024** | 0:38:07 |
| EXP-250 | `ACTIVE_REFERENCE` | Standard PINN tanh | LES_T50 | 13.09 % | **0.016** | 0:31:15 |

**5 個 paper-grade findings**:

1. **EXP-244 (4-head) 取代 EXP-241_b 為新 stable best** — KE 5.51 % (-0.46 pp vs 1-head)。Multi-head cross-attn 不增 param 但提高 attention 表達力。

2. **1024 collo 大幅縮小 DNS↔LES_T50 gap**:
   - 64 collo: DNS 9.40% / LES_T50 12.36% → gap **2.96 pp** (EXP-220 vs EXP-221)
   - 1024 collo: DNS 5.97% / LES_T50 6.92% → gap **0.95 pp** (EXP-241_b vs EXP-245)
   - Paper 主張「LES proxy pipeline 可達 baseline quality」在 1024 collo 下進一步強化

3. **Architecture ranking 在 LES_T50 + 1024 collo 重新洗牌**:
   - B3 (6.92) > B2 (8.43) > **B0 (9.96)** > PINN-SiLU (10.13) > **B1 (10.62)**
   - **B0 vanilla DeepONet 反超 B1 (CfC, no cross-attn)** — 暗示 **cross-attention 比 CfC 更重要** 在 LES + 高 collo 環境（之前 64 collo + DNS 下 B1 14.65 < B0 18.52）

4. **PINN 1024 collo 大幅 improvement**:
   - PINN-SiLU: 38.50 % (64 collo, EXP-204) → **10.13 %** (1024 collo, EXP-249), -28.4 pp
   - 「plain MLP PINN 比 operator framework 對 collo density 更敏感」— physics regularization 對 PINN 是 dominant lever
   - 但 absolute KE 仍輸 operators (B3 6.92 < PINN-SiLU 10.13)
   - PINN div_L2 0.024/0.016 反而最低 — PINN 對 incompressibility 嚴格滿足，trade-off vs sensor data fit

5. **PINN tanh outlier 13.09 %** confirm SiLU > tanh activation choice（EXP-250 vs EXP-249 +2.96 pp）

**Take-away for paper**（2026-05-20 baseline 升級後）: 
- **主 baseline = EXP-245 (B3 + 1-head + 1024 collo + LES_T50, KE 6.92%, n=1)** — 工程可遷移配置，對齊 paper 主訴「無 DNS」
- **延伸論述**:
  - **collo density**: EXP-241_b (DNS, 1-head, 5.97%) vs baseline 6.92% — DNS sensor 換 LES proxy cost 0.95 pp
  - **4-head**: EXP-251 (LES_T50, 4-head, 6.68%) vs baseline 6.92% — multi-head inductive bias 改善 0.24 pp
  - **DNS oracle upper bound**: EXP-244 (DNS, 4-head, 5.51%) 為「omniscient sensor + 4-head」上限
- **Legacy reference**: EXP-200_a~e (DNS, 64 collo, 10.77 ± 0.52 %, n=5) — paper 寫作可引為「64 collo + DNS 早期 baseline + 統計穩定」reference
- **High priority**: EXP-245 multi-seed n=3-5 → 把主 baseline 從 n=1 升 statistical
- B0 反超 B1 (in LES_T50 + 1024 collo): **cross-attention 為 architectural sweet spot**，比 CfC 更關鍵

### Multi-constraint AL ablation group（EXP-242 + EXP-243, 2026-05-20 完成）— **NS 加 AL = anti-pattern**

| ID | Status | GN tasks | AL constraints | use_gradnorm | KE rel-err | Train wall | 一致原則 | 結論 |
|---|---|---|---|---|---|---|---|---|
| **EXP-242_a** | `ACTIVE_REFERENCE` | `[data, ns_u, ns_v]` | `[cont]` | true | **10.19 %** | 1:02:58 | ✅ | cont 純 AL ≈ baseline 雙開（in 1 std） |
| **EXP-243** | `NEGATIVE_RESULT` | `[data]` (僅 data) | `[ns_u, ns_v, cont]` | **false** | **13.33 %** | 1:03:05 | ✅ **完全** | 全 physics 純 AL, no GN — multi-AL 對 NS 仍反效果 |
| **EXP-242_c** | `NEGATIVE_RESULT` | `[data, cont]` | `[ns_u, ns_v, cont]` | true | **13.70 %** | 1:03:43 | ⚠️ cont 雙開 | NS 純 AL + cont 雙開（部分違反） |
| **EXP-242_b** | `NEGATIVE_RESULT` | `[data, ns_u, ns_v, cont]` | `[ns_u, ns_v, cont]` | true | **14.79 %** | 1:05:54 | ❌ 全雙開 | NS+cont 全雙開（GN+AL 互相 amplification）|

Decision gates 評估:

| Config | Gate | 結果 |
|---|---|---|
| 242_a (a) KE < 9 % 雙開冗餘 | (b) 9-12 % 雙開 ≈ 純 AL ✅ | ADR-001 §4 修訂中性 |
| 242_b (a) ≤ 9.5 % net positive | (c) > 11.5 % AL over-penalty ✅ | **NS 加 AL = anti-pattern** |
| 242_c vs 242_b (a) 純 AL 更乾淨 ✅ | (-1.09 pp) | 但 NS 加任何 AL 都不好 |

**Paper-grade findings**（含 EXP-243 一致原則 confirmation）:
1. **L_phys 低 ≠ KE 低**: 242_b L_phys 1.67e-2 (~9× ↓) 但 KE +4 pp 退步 — 經典 PINN over-physics 病態，AL pressure 過度 push NS → data fit 被犧牲
2. **cont AL 是 sweet spot, NS AL 不是**: cont (divergence) 是 hard 約束（incompressibility）； NS momentum 是 soft 引導，太強會 over-fit 物理解
3. **「開 AL = 拿出 GN」一致原則部分驗證 (EXP-243)**: 拿出 GN 比雙開乾淨（243 vs 242_b -1.46 pp / 243 vs 242_c -0.37 pp），符合 ADR-001 §4 motivation
4. **但 NS 加 AL 本身仍 anti-pattern**: EXP-243 (一致原則 + 全 physics 純 AL + no GN) 仍 KE +2.56 pp 退步 vs baseline — 證明問題不在 GN 處理方式，而是 NS 不適合 AL pressure
5. **ADR-001 §4 對 cont 過保守，但對 NS valid**: cont 拿出 GN (242_a 10.19%) ≈ baseline；NS 拿出 GN (243 13.33%) 仍退步
6. **Decision gate (c) confirmed for NS only**: GN 為 essential balancing **僅對 NS 成立**（對 cont 不一定）

**Take-away for paper**: 主線 EXP-200_a recipe (cont 雙開 + NS 只 GN, no NS-AL) **已是 multi-AL 配置中最佳**；不要試圖加 NS-AL（會傷害 KE，無論 GN 處理方式）。collocation density (EXP-241) 才是真正可改善的 lever。

### Collocation density sweep group（EXP-241, 2026-05-19 完成）— **新最佳結果**

| ID | Status | num_physics_points | KE rel-err | Train wall (RTX 3090) | 角色 |
|---|---|---|---|---|---|
| **EXP-241_a** | `ACTIVE_REFERENCE` | 256 (4× baseline) | **6.88 %** 🥈 | 1:04:46（並行）| collocation density 中段 |
| **EXP-241_b** | **`ACTIVE_BASELINE` 🏆** | 1024 (16× baseline) | **5.97 %** 🥇 | 1:19:30（並行）| **stable phase 新最佳**；GPU util 75% |

EXP-241 ablation 完整數值：

| 指標 | EXP-200_a baseline (64) | EXP-241_a (256) | EXP-241_b (1024) | 改善 (best vs baseline) |
|---|---|---|---|---|
| KE rel-err | 10.77 ± 0.52 % | 6.88 % | **5.97 %** | **-44.6 %** |
| u rel-L2 | 20.69 % | 17.13 % | **16.38 %** | -20.8 % |
| v rel-L2 | 24.79 % | 20.76 % | **19.77 %** | -20.3 % |
| ω rel-L2 | 52.65 % | 46.71 % | **45.14 %** | -14.3 % |
| div L2 | 0.066 | 0.0551 | **0.0460** | -30.3 % |
| ek_ratio_kf | 0.920 | 0.953 | **0.957** | +0.040 |
| kf amp ratio | 0.937 | 0.960 | **0.972** | +0.035 |
| kf phase err (rad) | -0.011 | -0.026 | -0.019 | similar |
| GPU util (RTX 3090) | 13-34 % (latency-bound) | 40 % | **75 %** | throughput-bound |
| GPU memory | 0.55 GB | 3.69 GB | 12.25 GB | |

**Decision gate (per EXP-241 falsifiability)**: 兩點都 `KE ≤ 9.5 %` ✅ → "collocation density 為 binding constraint, 主線應升級"。**EXP-241_b 取代 EXP-200_a 為 stable phase active baseline**，但保留 EXP-200_a 作 multi-seed n=5 統計參照（EXP-241_b 仍 single seed=42, 需後續 multi-seed 確認 std）。

### Inference cost benchmark（hardware-specific）

| Hardware | Model | Eval wall (full snapshot 評估) | per-snapshot avg |
|---|---|---|---|
| M3 base (4P+4E, MPS) | B3 (EXP-094, legacy) | — | encoder 71 ms + query 1.5 ms |
| RTX 3090 (lab acmt20) | B0 (EXP-240, 201 snapshots) | 16 s/model | ~80 ms/snapshot (full eval pipeline) |
| RTX 3090 (lab acmt20) | B3 (EXP-241, 201 snapshots) | 79 s/model | ~390 ms/snapshot (full eval pipeline incl. spectral) |

> **Note**: 上述 RTX 3090 數字含整套 evaluator pipeline（場重建 + 譜估 + KE/div/能譜計算 + 繪圖），非純 inference latency。Paper-grade pure encoder/query benchmark 仍以 EXP-094 M3 baseline (71+1.5 ms) 為主 reference；RTX 3090 paper-grade benchmark 待 `scripts/benchmark_inference.py` 重跑。

### Sensor placement group（EXP-220~222, EXP-224, B3 arch, seed=2, axis-fix v2）

| ID | Status | Placement strategy | KE rel-err | Δ vs oracle | 工程可遷移性 |
|---|---|---|---|---|---|
| **EXP-220** | `ACTIVE_REFERENCE` | DNS QR-pivot K=100（**oracle**）| **9.40 %** | — | 無（需 DNS）|
| **EXP-221** | `ACTIVE_REFERENCE` | LES_N256 **T=50 stat-converged, random IC** + QR-pivot | 12.36 % | +2.96 pp | **強**（real-world 完全 DNS-free，**論文 engineering pivot 主代表**）|
| **EXP-222** | `ACTIVE_REFERENCE` | LES_N128 T=15 Bardina over-disp stand-alone + QR-pivot | 12.40 % | +3.00 pp | 強（**low-fidelity LES viable**：N=DNS/2 + 計算 1/16）|
| **EXP-224** | `ACTIVE_REFERENCE` | Random uniform K=100 (seed=42) | 13.25 % | +3.85 pp | 強（無需 LES）|

> **Note**: EXP-220 與 EXP-200_c 都是 B3 + DNS QR-pivot + seed=2，差異僅在報告角度（前者 placement ablation, 後者 multi-seed group）。實質訓練 artifact 完全相同。
>
> **EXP-221 vs EXP-222 重點差異**: 兩者都「real-world DNS-free」可遷移，但 (a) EXP-221 N=256 同 DNS grid + T=50 26.5 turnovers + α=1.8 譜形接近 DNS（slope −6.46 vs DNS −4.75）；(b) EXP-222 N=128 粗網格 + T=15 8.5 turnovers + α=30 過耗散（slope −14）。KE 幾乎打平（12.36 % vs 12.40 %）→ 論文可主張「**LES 解析度與譜形對齊都不是 bottleneck**，statistical convergence + 正確 axis convention 才是」。
>
> **EXP-223 (LES_N256 T=30 dns-init) 已從 stable phase 移除（2026-05-19）**: 同時 (a) 工程不可遷移（dns-init 需偷看 DNS IC）和 (b) 效果不如 EXP-221（13.08 % > 12.36 %），無 paper value。Legacy EXP-106 archive 保留作 internal note。
>
> **EXP-225 (LES_T5) 已從 stable phase 移除（2026-05-19）**: T_end=5 < 1 large-eddy turnover (T_L≈1.88)，**非 statistically-converged LES**，KE 23.48% 為已知 outlier。完整 record 仍保留於 legacy EXP-103 v2 archive 作為「LES under-convergence 失敗教材」。

### Re=1000 baseline group（EXP-230）

| ID | Status | 主題 | KE rel-err |
|---|---|---|---|
| **EXP-230** | `ACTIVE_BASELINE` | Re=1000 SOAP+SF 5k | 9.61 % |

---

## [INDEX] Legacy ↔ Stable ID 雙向對照

### 由 stable ID 查 legacy

| Stable ID | Legacy ID | Seed | 角色 |
|---|---|---|---|
| `EXP-200_a` | `EXP-080` | 42 | B3 multi-seed #1（時間最早，AL Pareto sweet spot 首次定錨）|
| `EXP-200_b` | `EXP-093` | 1 | B3 multi-seed #2 |
| `EXP-200_c` | `EXP-094` | 2 | B3 multi-seed #3（同時為 EXP-220 DNS-pivot oracle, inference benchmark）|
| `EXP-200_d` | `EXP-097` | 3 | B3 multi-seed #4 |
| `EXP-200_e` | `EXP-098` | 4 | B3 multi-seed #5 |
| `EXP-201_a` | `EXP-088` | 42 | B0 multi-seed #1 |
| `EXP-201_b` | `EXP-095` | 1 | B0 multi-seed #2 |
| `EXP-201_c` | `EXP-096` | 2 | B0 multi-seed #3 |
| `EXP-201_d` | `EXP-099` | 3 | B0 multi-seed #4 |
| `EXP-201_e` | `EXP-100` | 4 | B0 multi-seed #5 |
| `EXP-202` | `EXP-089` | 42 | B1 ablation |
| `EXP-203` | `EXP-090` | 42 | B2 ablation |
| `EXP-204` | `EXP-091` | 42 | Standard PINN SiLU |
| `EXP-205` | `EXP-092` | 42 | Standard PINN tanh |
| `EXP-220` | `EXP-094` | 2 | DNS-pivot oracle（同 EXP-200_c）|
| `EXP-221` | `EXP-105 v2` | 2 | LES_N256 T=50 stat-conv, random IC（real-world DNS-free）|
| `EXP-222` | `EXP-102 v2` | 2 | LES_N128 over-disp stand-alone（low-fidelity LES viable）|
| `EXP-224` | `EXP-101 v2` | 42 | Random uniform |
| `EXP-230` | `EXP-030` | — | Re=1000 baseline |
| ~~`EXP-223`~~ | ~~`EXP-106`~~ | — | **移除（2026-05-19）**: T=30 dns-init 工程不可遷移（需 DNS IC）且效果不如 T=50；legacy archive 保留 |
| ~~`EXP-225`~~ | ~~`EXP-103 v2`~~ | — | **移除（2026-05-19）**: T=5 < 1 turnover，非 statistically-converged LES，legacy archive 作失敗教材 |

### 由 legacy 查 stable

| Legacy ID | Stable ID | 註 |
|---|---|---|
| `EXP-030` | `EXP-230` | Re=1000 baseline |
| `EXP-080` | `EXP-200_a` | B3 seed=42（first AL ρ=0.1 sweet spot run）|
| `EXP-088` | `EXP-201_a` | B0 seed=42 |
| `EXP-089` | `EXP-202` | B1 |
| `EXP-090` | `EXP-203` | B2 |
| `EXP-091` | `EXP-204` | Standard PINN SiLU |
| `EXP-092` | `EXP-205` | Standard PINN tanh |
| `EXP-093` | `EXP-200_b` | B3 seed=1 |
| `EXP-094` | `EXP-200_c` ≡ `EXP-220` | B3 seed=2 ≡ DNS-pivot oracle |
| `EXP-095` | `EXP-201_b` | B0 seed=1 |
| `EXP-096` | `EXP-201_c` | B0 seed=2 |
| `EXP-097` | `EXP-200_d` | B3 seed=3 |
| `EXP-098` | `EXP-200_e` | B3 seed=4 |
| `EXP-099` | `EXP-201_d` | B0 seed=3 |
| `EXP-100` | `EXP-201_e` | B0 seed=4 |
| `EXP-101 v2` | `EXP-224` | Random uniform |
| `EXP-102 v2` | `EXP-222` | LES_N128 over-disp stand-alone |
| `EXP-103 v2` | — (移除) | T=5 非 stat-converged，**不納入 stable phase**；僅 legacy archive 保留 |
| `EXP-105 v2` | `EXP-221` | LES_N256 T=50 stat-conv, random IC |
| `EXP-106` | — (移除) | T=30 dns-init 工程不可遷移且效果不如 T=50，**不納入 stable phase** |

> `EXP-101/102/103/105` v1（axis bug 受害版本）**不重新編號**，永遠以 legacy ID + "v1 buggy" 標籤存在於 archive，避免污染 stable phase。

---

## [STATE] Architectural Ablation 結論（B0/B1/B2/B3）

| Component | B0 | B1 | B2 | B3 (Ours) |
|---|---|---|---|---|
| CfC time encoding | ✗ | ✓ | ✗ | ✓ |
| Cross-attention | ✗ | ✗ | ✓ | ✓ |
| KE rel-err | 18.52 % | 14.65 % | 13.62 % | **10.77 %** |
| Δ vs B0 | — | -3.87 pp | -4.90 pp | **-7.75 pp** |

- **B3 vs B0 stat sig**: Cohen d = 13.09, p < 1e-7 (Welch's t-test, df_welch=7.6)
- **CfC contribution**: B0 → B1, ΔKE = -3.87 pp
- **Cross-attn contribution**: B0 → B2, ΔKE = -4.90 pp
- **Both components essential**: B3 - B1 = -3.88 pp（cross-attn 在 CfC 上仍有貢獻）；B3 - B2 = -2.85 pp（CfC 在 cross-attn 上仍有貢獻）
- **Operator framework >> Standard PINN**: B0 - PINN = -20.0 ~ -21.3 pp

---

## [STATE] Sensor Placement 結論（K=100 sparse regime）

修完 axis bug 後（CLAUDE.md KNOWN_PITFALLS / 2026-05-18），**僅列工程可遷移 + statistically-converged 的 LES placement**:

| Placement | KE rel-err | 工程可遷移性 | 解讀 |
|---|---|---|---|
| DNS QR-pivot (oracle) | **9.40 %** | 無（需 DNS）| 上限參考（理論上 omniscient）|
| LES_N256 **T=50 stat-conv, random IC** | 12.36 % | **強**（real-world DNS-free）| 26.5 turnovers 完全脫離 DNS 影響；**論文 engineering pivot 主代表** |
| LES_N128 Bardina over-disp stand-alone | 12.40 % | 強 | N=DNS/2 + α=30 過耗散 + spin-up 充足；**low-fidelity LES viable**（計算 1/16）|
| Random uniform | 13.25 % | 強（無需 LES）| placement-agnostic baseline |
| ~~LES_N256 T=30 dns-init~~ | ~~13.08 %~~ | ~~中（DNS IC）~~ | **已移除**: 工程不可遷移（需 DNS IC）且效果不如 T=50 |
| ~~LES_N256 T=5 short~~ | ~~23.48 %~~ | — | **已移除**: < 1 large-eddy turnover (T_L≈1.88)，非 stat-converged LES |

### Paper-grade findings（移除 T=30 dns-init + T=5 後）
1. **LES proxy pipeline viable**: 3 個 well-formed cross-source placements（EXP-221/222/224）達 KE 12-13% (gap to oracle ~3pp)
2. **LES 解析度與譜形對齊都不是 bottleneck**: EXP-221 (N=256 譜接近 DNS) ≈ EXP-222 (N=128 過耗散 slope −14) — KE 差 < 0.05 pp
3. **Statistical convergence 才是 gating**: T_end ≥ 8 turnovers 即夠（EXP-222 T=15 = 8.5 turnovers ≈ EXP-221 T=50 = 26.5 turnovers）
4. **Random ≈ well-formed LES**: K=100 sparse regime 下 placement 演算法影響有限（< 1 pp）
5. **Real-world engineering pipeline 可行**: 低成本 LES（EXP-222: N=128, T=15, 計算 1/16 DNS）+ QR-pivot + 量測 → 重建 達 baseline-quality
6. **移除項說明**: T=30 dns-init 違反「現場無 DNS」假設不納入；T=5 不滿足統計收斂不納入。兩者完整 record 見 legacy archive。

---

## [STATE] Rejected / Invalid Directions（穩定階段一致引用）

從 legacy 繼承的 reject 結論，stable phase 仍有效：

1. `omega` 作為 sensor supervision（legacy EXP-002）
2. 5k 延長訓練（legacy EXP-009）
3. top-k local attention（legacy EXP-013）
4. Re=1000 用 k_f=4 forcing
5. Physics loss 機制調整（Chebyshev / residual norm / Poisson, legacy EXP-035~039）
6. Transfer learning 跨架構（legacy EXP-040, EXP-042）
7. **6-lever pivot ablation 全 falsified**（legacy EXP-083~087）: ρ ablation、multi-head、harmonics ↑、K-scaling、trunk depth ↑、mMLP gating — 無一突破
8. AL 與 GradNorm 同時控制 cont（ADR-001 §4 escape hatch ok, 但「兩全其美」不存在）
9. **Resume from checkpoint**（legacy EXP-082 災難）: silent state corruption，必須 1-shot 訓練
10. **Sensor swap-axis convention**（CLAUDE.md AXIS BUG, 2026-05-18）: KE 30+pp 退步，必須通過 `test_sensor_axis_convention.py`

---

## [STATE] Architecture × Placement 2×3 完整表（2026-05-19）

EXP-240_a/_b 完成後 2D ablation 表已封閉：

| Architecture | DNS oracle | LES_T50 | Random | Placement gap |
|---|---|---|---|---|
| **B0** (Vanilla DeepONet, n=1@seed=42) | 18.52 ± 0.66 (n=5, EXP-201) | **19.58** (EXP-240_a) | **21.82** (EXP-240_b) | **3.30 pp** |
| **B3** (Ours, n=1@seed=2) | 9.40 (EXP-220) / 10.77 ± 0.52 (n=5 EXP-200) | 12.36 (EXP-221) | 13.25 (EXP-224) | 2.48-3.85 pp |
| **Architecture gap (B0 − B3)** | ~8 pp | **7.22 pp** | **8.57 pp** | — |

### Paper-grade findings (EXP-240 contribution)

1. **Architecture effect dominant**: B3 − B0 ~8 pp 穩定跨所有 placement，**比 placement gap (~3 pp) 大 2-3 ×**
2. **LES degradation 跨 architecture 比 B3 更輕微**:
   - B0 DNS → LES_T50: +1.06 pp（vs B3 +2.96 pp）
   - 解讀：**B3 高表達力會對 placement 更挑剔**；B0 因受限於模型 capacity，placement quality 對 KE 的邊際影響反而 saturated
3. **LES > Random 跨 architecture 成立**:
   - B0: Random → LES_T50 gain **2.24 pp**
   - B3: Random → LES_T50 gain 0.89 pp
   - 反直覺：**B0 從 LES placement 受益更多**（架構 expressivity 不足時，placement 提供更多信息成為 binding constraint）

### Falsifiability evaluation (per EXP-240 config 預設)

| Decision gate | EXP-240_a 結果 | 判讀 |
|---|---|---|
| KE ≤ 16% → "transferable" 成立 | 19.58 % | ❌ 不到 16 % |
| 16 % < KE ≤ 22% → 部分有效 | 19.58 % | ✅ **此區** |
| KE > 22% → 僅 B3 專屬 | 19.58 % | ❌ 未到 22 % |
| `|B0+Random − B0+LES_T50| < 2pp` → placement-agnostic | 2.24 pp | ❌ 略超 2 pp |
| `> 3pp` → B0 顯著更敏感 | 2.24 pp | ❌ 未到 |

結論：**Hypothesis 部分支持** — LES placement 在 B0 也帶來改善，但 absolute KE 仍受限於架構 capacity。

---

## [STATE] Open Question（stable phase, 待補）

---

## [STATE] Open Question（stable phase, 待補）

| 問題 | 現況 | 狀態 |
|---|---|---|
| **EXP-245 multi-seed (n=3-5)** | NEW main baseline, single seed=42, KE 6.92 % — paper-grade needs std confirmation | **NEW 最高優先** |
| **EXP-241_b multi-seed (n=3-5)** | DNS 對照 single seed=42, KE 5.97 % — paper-grade needs std confirmation | **NEW 高優先** |
| **EXP-241_c collocation = 4096?** | EXP-241_a (256) → EXP-241_b (1024) 還沒 saturated（5.97 vs 6.88, -0.9pp 下行）；4096 可能再降但 OOM risk | 待開工（需 split-batch fallback）|
| RTX 3090 paper-grade inference benchmark | EXP-094 M3 baseline 71+1.5 ms 為唯一參考；新 hw 待測 | 待 `benchmark_inference.py` 重跑 |
| Re=1000 stable phase multi-seed（n=5）| 尚未跑；目前只有 legacy EXP-030 single seed | 待開工 |
| `EXP-220` (= `EXP-200_c`) 5-seed sensor placement variance | 單 seed (seed=2) 結果；無法估計 placement-induced variance | 待開工（若需 paper-grade noise quantification）|
| LES robustness across LES_seed | 目前 LES generator 用 seed=42 single placement; 跨 LES seed 的 sensor variability 未測 | 待開工 |
| EXP-242 group: multi-constraint AL (cont 純 AL / NS 加 AL / NS 純 AL) | 設計完成，code change ~80 LOC pending implement | 待開工 |
| Cylinder stable phase 整併 | Cylinder 仍用 CEXP-XXX；是否要納入此 v2 system？| 開放討論 |
| CfC Jacobian spectral radius stability | 未寫腳本 | 待開工（CFD-rigour）|

---

## [HOW-TO] 新增 stable phase 實驗

1. 確認新實驗屬於哪個 group（200~239 已分配；新研究方向用 240+）
2. 在 `configs/stable/` 建立新 config（檔名 `exp_NNN.toml` 或 `exp_NNN_X.toml`，X 為 multi-seed suffix）
3. 訓練時 artifact dir 命名: `artifacts/kolmogorov/stable/exp_NNN[_X]_{描述}`
4. 訓練完成後：
   - 此檔 `[INDEX]` 新增一行
   - 詳細 RECORD 加在本檔 `[RECORD]` section（若篇幅 > 50 lines，另開 `docs/experiment_archive_stable_phase.md`）
5. 若實驗 supersedes 既有結論，更新 `[STATE] Current Baseline` 與 `[STATE] Rejected`

---

## 變更紀錄

- **2026-05-19**: v2 啟用。從 legacy EXP-001~106 完整提取 stable phase 主線（B3 multi-seed, B0 multi-seed, B1/B2/PINN ablation, sensor placement series, Re=1000 baseline），以 EXP-200 起編號。Multi-seed 統一 `_a~_e` suffix。Legacy IDs 與其 archive 不動；雙向對照表見 [INDEX] Legacy ↔ Stable ID 雙向對照。
