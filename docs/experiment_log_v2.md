# 實驗紀錄 v2（Stable Phase State 主檔）

> **Status**: Stable phase（2026-05-19 啟用，2026-05-27 narrative 重組）。研究已脫離前期探索（EXP-001~106），進入主線收斂、論文寫作、多 seed 統計確認階段。
>
> **Scope**: 此檔負責 **EXP-200 起所有 stable phase Kolmogorov 實驗** 的 state 紀錄。Cylinder 主線見 [`docs/cylinder_log_v2.md`](cylinder_log_v2.md)（獨立支線）。Legacy EXP-001~106 已全部移至 [`docs/archive/`](archive/)，不動。
>
> **本檔組織原則**（2026-05-27 重組後）:
> - §1 **主線** — Re=10⁴ baseline + 主線固定假設（含 collocation density 證據鏈）
> - §2 **延伸驗證** — Re=10⁶ ablation ladder
> - §3~§7 **五條對照證據** — 架構 / Sensor placement / Sensor amount (K-scaling) / Sensor noise / Classical interpolation
> - §8 **Inference cost benchmark**
> - §9 **Diagnostics / Negative findings** — multi-AL anti-pattern、forcing identifiability ill-posed
> - §10+ 結論摘要 / Legacy 對照 / Open questions / 變更紀錄

---

## [STATE] Read Order

| 檔 | 內容 | 何時讀 |
|---|---|---|
| **本檔** `docs/experiment_log_v2.md` | **唯一 active 主檔（Kolmogorov）** — Stable phase STATE/INDEX、legacy 對照表 | **任何 Kolmogorov 實驗變更前都讀這個** |
| [`docs/cylinder_log_v2.md`](cylinder_log_v2.md) | **Cylinder 獨立支線** — CEXP 系列 | Cylinder 任務 |
| [`docs/archive/experiment_log.md`](archive/experiment_log.md) | Legacy STATE（EXP-001~106 結論層）| 若 stable phase 結論不足，往回查 |
| [`docs/archive/experiment_archive_kolmogorov.md`](archive/experiment_archive_kolmogorov.md) | EXP-001~063 詳細 RECORD | 早期實驗追溯 |
| [`docs/archive/experiment_archive_kolmogorov_post_k100.md`](archive/experiment_archive_kolmogorov_post_k100.md) | EXP-064~106 詳細 RECORD（含 v2 axis-fix）| 近期 ablation 判讀 |
| [`docs/archive/squeeze_report_2026-05-11.md`](archive/squeeze_report_2026-05-11.md) | Classical interpolation baseline 詳細推導與 SVD null-space 證明 | §7 數據來源；想看完整 fair-baseline 方法/scripts |
| [`docs/archive/diagnostics_log.md`](archive/diagnostics_log.md) | denorm bug + CFD-rigour Q5/Q7/Q8 + Forward CFD | 評估值質疑 |
| [`docs/adr/`](adr/) | 設計決策 | 設計權衡追溯 |
| [`docs/paper_framing_draft.md`](paper_framing_draft.md) | 論文 framing | 寫作 |

---

## [STATE] Metrics Glossary

| Metric | 定義 | 解讀 |
|---|---|---|
| `KE rel-err` | `|0.5⟨u²+v²⟩_pred − 0.5⟨u²+v²⟩_DNS| / 0.5⟨u²+v²⟩_DNS`, 取 t=5 | 全頻段 integral 能量誤差 |
| `u/v/ω rel-L2` | `‖field_pred − field_DNS‖₂ / ‖field_DNS‖₂` | pointwise 場誤差 |
| `div L2 mean` | `‖∇·u_pred‖₂` over t | incompressibility 違反度（DNS floor ~0.09 為 numerical truncation）|
| **`ek_ratio_kf_last`** | **`E_pred(k=k_f) / E_DNS(k=k_f)`** at t=5（spectrum value 比）| forcing-injection wavenumber 的能量是否精準（1.0 = perfect, <1 under-driven, >1 over-shoot）|
| `kf_amplitude_ratio` | `|û_pred(k_f)| / |û_DNS(k_f)|` (mode coefficient amplitude) | 同上但用複數 mode coeff 拆 amplitude/phase |
| `kf_phase_err` | `arg(û_pred) − arg(û_DNS)` at k=k_f (radians) | forcing mode 相位差，0 = 同相 |
| `band_energy_rel_err_last` | 低/中/高 band 各自 KE rel-err at t=5 | band_low (k≤5, **= K=100 Nyquist cutoff $\lfloor\sqrt{K/\pi}\rfloor$**) / band_mid (5<k≤16) / band_high (k>16) — 與 evaluator `BAND_EDGES_K_LOW=5.0` 對齊 |

---

## [STATE] Data Version

- DNS（Re=10000）: [`data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy`](../data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy)
- DNS（Re=1000）: [`data/kolmogorov_dns_fp64_etdrk4_Re1000_N128_T5_ds4.npy`](../data/kolmogorov_dns_fp64_etdrk4_Re1000_N128_T5_ds4.npy)
- Sensor（DNS QR-pivot K=100, Re=10000, default）: `data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100.{json,npz}`
- Sensor（Random K=100, Re=10000）: `data/kolmogorov_sensors/re10000/sensors_random_K100_N256_t0-5_si100_seed42.{json,npz}`（v2 fixed axis convention）
- Sensor（LES-informed series, Re=10000）: `data/kolmogorov_sensors/re10000/sensors_lesinformed_*.{json,npz}`（v2 fixed axis convention）
- Sensor（DNS QR-pivot K=100, Re=1000）: `data/kolmogorov_sensors/re1000/sensors_qrpivot_K100_N128_t0-5.json`

---

## [STATE] Grid Independence Validation (Re=10⁴, 2026-05-24)

| Field | Value |
|---|---|
| Status | ✅ **PASS** — N=256 baseline grid-converged for paper §Methods |
| Report | [`docs/grid_independence_re10000.md`](grid_independence_re10000.md) |
| Spec | [`docs/superpowers/specs/2026-05-24-kolmogorov-re10000-grid-independence-design.md`](superpowers/specs/2026-05-24-kolmogorov-re10000-grid-independence-design.md) |
| Data | `data/dns/gi_test_re10000/` (7 .npy + 2 JSON) |
| Figures | `docs/figures/grid_independence/` (6 PNG) |
| Compute | home-gpu (i7-11700, 12-core CPU), ~5 hr wall total |

**Setup**: N ∈ {128, 256, 512, 1024}, ETDRK4 spectral fp64, dt=2.5e-4, T=5, dealias 2/3, deterministic `spectral_seeded` IC (cross-N bit-exact in `k ≤ k_cutoff=8`, pytest 12/12 PASS)

**Main metrics (N=256 vs ref=N=1024)**:

| Metric | Value | Threshold | Verdict |
|---|---|---|---|
| `rel_L2(u) @ t=0.5` | 0.113 % | < 1 % | ✅ PASS (10× margin) |
| `KE max rel diff (post-spinup t≥2)` | **0.064 %** | < 2 % | ✅ PASS (35× margin) |
| `Enstrophy max rel diff (post-spinup)` | **0.24 %** | < 2 % | ✅ PASS (8× margin) |
| `max|∇·u|` | 3.76e-13 | < 1e-10 | ✅ PASS (machine ε) |
| `k_max / k_eta` (dissipation resolution) | 2.06 (k_η=41.5) | ≥ 1.5 | ✅ PASS |

**Killer claims for §Methods**:
- N=512 vs N=1024 KE diff = **3.87e-7 (machine ε)** → 直接證明 ref converged，defang「ref unverified」reviewer attack
- K=100 sensor Nyquist `k ≤ √(K/π) = 5.64`，**99.32% energy** in this band; N=256 vs N=1024 在此 band 收斂到 **0.05%** → grid 對 sparse-sensor training 完全 adequate
- dt=2.5e-4 temporally converged: dt-halved 比 spatial error 小 **160×**

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
- 訓練 1-shot 20000 步（baseline 升級至 20k），禁用 `resume_checkpoint`（EXP-082 災難根因）

---

# §1 主線（Main Line）: Re=10⁴ Kolmogorov

## 1.1 主 baseline = **EXP-245** (B3 + LES_T50 + K=100 + 20k n=5)

```
Baseline ID:  EXP-245 (n=5 multi-seed group _a~_e)
Config:       configs/stable/exp_245.toml → exp_245_b3_les_T50.toml (seed=42 = _a)
              configs/stable/exp_245_{b,c,d,e}.toml (seed=1/2/3/4)
Architecture: B3 (1-head cross-attn, minimal)
Sensor:       LES_T50  (= EXP-221, real-world DNS-free placement)
Collocation:  1024
Iterations:   20000     (升級自 10k, per 收斂分析: L_phys step 10k 仍 monotone 下降)
Warmup (all): 2000 steps fixed (lr_warmup, time_marching_warmup_steps, lr_decay)
Seeds:        42 / 1 / 2 / 3 / 4 (n=5)
KE rel-err:   5.71 ± 0.11 %   (n=5, σ=0.11 pp, 95% CI [5.61, 5.81] %)
```

### EXP-245 n=5 multi-seed metrics

| Seed | KE rel-err | u L₂ | v L₂ | ω L₂ | Ens rel-err | div ratio | k_f amp |
|---|---|---|---|---|---|---|---|
| _a (42) | 5.9035 % | 13.59 % | 17.53 % | 41.66 % | 24.41 % | 0.39 % | 0.9973 |
| _b (1)  | 5.6751 % | 13.74 % | 17.70 % | 41.95 % | 24.14 % | 0.40 % | 0.9852 |
| _c (2)  | 5.6491 % | 13.63 % | 17.48 % | 41.67 % | 23.85 % | 0.40 % | 0.9871 |
| _d (3)  | 5.7144 % | 13.66 % | 17.46 % | 41.83 % | 24.18 % | 0.39 % | 0.9915 |
| _e (4)  | 5.5882 % | 13.65 % | 17.44 % | 41.75 % | 23.99 % | 0.39 % | 0.9957 |
| **mean ± std** | **5.71 ± 0.11 %** | **13.65 ± 0.06 %** | **17.52 ± 0.10 %** | **41.79 ± 0.12 %** | **24.11 ± 0.21 %** | **0.39 ± 0.006 %** | **0.991 ± 0.005** |

### 10k → 20k upgrade summary（baseline 升級的關鍵改善）

| Metric | 10k single seed=42 | 20k n=5 mean | Δ |
|---|---|---|---|
| KE rel-err | 5.97 % | **5.71 ± 0.11 %** | **−4.3 %** relative |
| u rel-L₂ | 14.46 % | 13.65 % | −5.6 % |
| v rel-L₂ | 19.07 % | 17.52 % | −8.1 % |
| ω rel-L₂ | 43.95 % | 41.79 % | −4.9 % |
| Ens rel-err | 27.51 % | 24.11 % | **−12.4 %** |
| div ratio | 2.41 % | **0.39 %** | **−84 %** （**< DNS floor 1.04 %**）|
| k_f amp ratio | 0.926 | **0.991** | +7.0 % |
| Train wall-time | ~80 min | ~150 min | +88 % |

**Headline finding**: 20k baseline 三個 metric 出現質變（不只 marginal 改善）:
1. **div ratio 0.39 % < DNS floor 1.04 %**: PI-CON 在 sensor-only 訓練下達成 **sub-DNS divergence 控制** — paper §Discussion 強 claim
2. **k_f amp 0.991 ≈ 1.0**: forcing-mode recover 接近完美
3. **σ = 0.11 pp**: n=5 統計顯著確立, KE 5.71 % 為 publication-grade 數字

> **DNS oracle fair comparison（EXP-271, 2026-05-29）**: 完全相同 config（20k n=5）換回 DNS QR-pivot sensor → KE **4.68 ± 0.06 %**。但 trade-off：DNS 贏整體能量(KE +1.03 pp)、**LES 贏逐點場(u L2 13.65 vs 15.34)**。原「no measurable penalty」claim 已改為 trade-off framing（詳見 §4.3）。

## 1.2 證據鏈：collocation density 為 binding constraint（EXP-241）

64 → 256 → 1024 collocation density sweep 確認「density-bound」假說：

| 指標 | EXP-200_a baseline (64) | EXP-241_a (256) | EXP-241_b (1024, single seed) | 改善 (best vs baseline) |
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

**結論**: 兩點 `KE ≤ 9.5 %` ✅ → "collocation density 為 binding constraint, 主線應升級"。EXP-241_b (1024 collo DNS oracle) 為 EXP-245 升級的根據；EXP-245 進一步把 DNS sensor 替換為 LES_T50（工程可遷移）並升 20k n=5。

**結案更新（per EXP-241_b 1024 collo band-energy 分析, 2026-05-19）— K=100 上限分層**:

| 主張 | 狀態 |
|---|---|
| K=100 upper bound on **mid/high (k>5)** | ✅ **仍成立**（Nyquist 硬上限 $k_{\max}=\lfloor\sqrt{K/\pi}\rfloor=5$）|
| K=100 upper bound on **low (k≤5)** | ❌ falsify（band_low 3.62→2.41 %）|
| K=100 upper bound on **整體 KE** | ❌ falsify（low band 佔 ~99 % 能量, 10.77→5.97 %）|

## 1.3 Re=10³ reference baseline = **EXP-230**

| 項目 | 現況 |
|---|---|
| Baseline ID | `EXP-230` |
| Config | `configs/stable/exp_230.toml`（symlink → legacy EXP-030）|
| KE rel-err | 9.61 % |
| u RMSE | 5.68e-2 |
| amp ratio | 1.027 |

**角色**: Re=10⁴ 主線完成前的 sanity baseline；stable phase 尚未跑 multi-seed 版本（Open Question）。

---

# §2 延伸驗證（Extended Validation）: Re=10⁶

> **2026-05-23 finalized**: 完整 Re=10⁶ ablation ladder (EXP-262/264/265/267/268), LES T=50 home-gpu 7.18 hr 完成 (50 T_L stat-converged)。

## 2.1 Ablation ladder

| ID | Status | Re | K | LES | d_model | iter | KE rel-err | u L₂ | ω L₂ | Ens rel-err | div ratio | k_f amp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **EXP-245** (20k n=5) | `ACTIVE_BASELINE` | 10⁴ | 100 | T=50 | 256 | 20k | **5.71 ± 0.11 %** | 13.65 | 41.79 | 24.11 | **0.39 %** | **0.991** |
| EXP-262 | `REFERENCE` | 10⁶ | 100 | T=5 | 256 | 10k | 23.73 % | 32.92 % | 71.17 % | 60.93 % | 0.67 % | 0.919 |
| EXP-264 | `REFERENCE` | 10⁶ | 100 | T=5 | 384 | 50k | 19.02 % | 29.69 % | 67.84 % | 54.99 % | 0.37 % | 0.746 ⚠️ |
| EXP-265 | `REFERENCE` | 10⁶ | 200 | T=5 | 384 | 50k | 11.39 % | 21.64 % | 62.56 % | 48.11 % | 0.37 % | 0.849 |
| EXP-267 | `ACTIVE_REFERENCE` | 10⁶ | 100 | **T=50** | 256 | 10k | **14.58 %** | 25.61 % | 67.43 % | 55.05 % | 0.69 % | 1.147 |
| **EXP-268** | **`ACTIVE_REFERENCE` 🥇** | 10⁶ | 200 | **T=50** | 384 | 50k | **6.10 %** ⭐ | **15.62 %** | 58.17 % | 42.06 % | **0.37 %** | **1.035** |

## 2.2 Lever contributions

| Step | Change | KE | ΔKE |
|---|---|---|---|
| EXP-262 (baseline) | K=100, T=5, d=256, 10k | 23.73 % | — |
| → EXP-267 | **LES T=5 → T=50** (quality) | 14.58 % | **−9.15 pp** |
| → EXP-264 | **d=256→384 + 10k→50k** (capacity) | 19.02 % | −4.71 pp (from EXP-262) |
| → EXP-265 | **K=100→200 + T=5** | 11.39 % | −7.63 pp (from EXP-264) |
| → **EXP-268** | **K=200 + LES T=50 + d=384 + 50k** (全升) | **6.10 %** | **−17.63 pp from EXP-262** |

## 2.3 Findings

**Finding 1 — 🌟 EXP-268 KE 6.10 % ≈ Re=10⁴ baseline 5.71 %: cross-Re 主訊息確立**

```
Re=10⁴ (EXP-245, K=100, LES T=50, 20k): KE 5.71 ± 0.11 %
Re=10⁶ (EXP-268, K=200, LES T=50, 50k): KE 6.10 %   ← 差距僅 0.39 pp ≈ 1 σ_training
```

→ Paper §Cross-Re 最終 claim:
> "With quality LES placement (T=50, 50 T_L), K=200 sensors, and XL capacity (d=384, 50k steps), PI-CON achieves **KE rel-err 6.10 %** at Re=10⁶ — comparable to the Re=10⁴ baseline (5.71 ± 0.11 %) using K=100 sensors. This demonstrates that the framework generalizes across two orders of magnitude in Reynolds number, requiring only sensor budget scaling and commensurate training resources."

**Finding 2 — LES quality (T=5 → T=50) is the single largest lever**:
- EXP-262 → EXP-267: LES T=5 → T=50, same K=100/d=256/10k → KE **−9.15 pp** (−38.6 %)
- LES placement quality **dominates** capacity/training length lever (−4.71 pp)
- 說明 high-quality sensor placement 是 cross-Re performance 的 **critical prerequisite**

**Finding 3 — k_f amp 回到接近 1**:
- EXP-267: 1.147 (slight overshoot, K=100 limits forcing-mode fitting accuracy)
- EXP-268: **1.035** ≈ 1 → forcing mode recover 完美 cross-Re

**Finding 4 — div control cross-Re robust**:
- EXP-268 div ratio **0.37 %** vs DNS floor 3.31 % → **9× under floor**
- 跟 Re=10⁴ baseline 0.39 % 幾乎相同 → **sub-DNS divergence control 是 architecture 固有特性, not Re-specific**

**Finding 5 — ω / Ens still bounded by Layer 1 truncation**:
- EXP-268 ω 58.17 %, Ens 42.06 % — LES T=50 vs T=5 改善 ~4-7 pp
- High-band 受 K=200 Nyquist k_max=7.98 << Re=10⁶ dissipation k~100 限制, Layer 1 truncation dominant
- ω / Ens 的 absolute level 仍高, 對應 Re=10⁶ 高 dynamic range 物理預期 (不是 failure)

**Caveats**:
1. `num_physics_points = 512` (vs EXP-245 1024), OOM constraint for N=512
2. DNS frames 101 (vs Re=10⁴ 201)
3. Single seed for Re=10⁶ series — σ 未估計（Open Question）

---

# §3 對照群 A：架構（Architecture Ablation）

## 3.1 B0/B1/B2/B3 + Standard PINN（legacy multi-seed, 64 collo）

| ID | Status | 架構 | KE rel-err |
|---|---|---|---|
| **EXP-200** _a-e_ | `ACTIVE_BASELINE` (legacy 64 collo) | B3 (Full: CfC + cross-attn) | **10.77 ± 0.52 %** (n=5) |
| **EXP-201** _a-e_ | `ACTIVE_REFERENCE` | B0 (vanilla DeepONet) | 18.52 ± 0.66 % (n=5) |
| **EXP-202** | `ACTIVE_REFERENCE` | B1 (CfC, no cross-attn) | 14.65 % (n=1) |
| **EXP-203** | `ACTIVE_REFERENCE` | B2 (cross-attn, no CfC) | 13.62 % (n=1) |
| **EXP-204** | `ACTIVE_REFERENCE` | Standard PINN (SiLU) | 38.50 % (n=1) |
| **EXP-205** | `ACTIVE_REFERENCE` | Standard PINN (tanh) | 39.80 % (n=1) |

| Component | B0 | B1 | B2 | B3 (Ours) |
|---|---|---|---|---|
| CfC time encoding | ✗ | ✓ | ✗ | ✓ |
| Cross-attention | ✗ | ✗ | ✓ | ✓ |
| KE rel-err | 18.52 % | 14.65 % | 13.62 % | **10.77 %** |
| Δ vs B0 | — | -3.87 pp | -4.90 pp | **-7.75 pp** |

**Findings**:
- **B3 vs B0 stat sig**: Cohen d = 13.09, p < 1e-7 (Welch's t-test, df_welch=7.6)
- **CfC contribution**: B0 → B1, ΔKE = -3.87 pp
- **Cross-attn contribution**: B0 → B2, ΔKE = -4.90 pp
- **Both components essential**: B3 - B1 = -3.88 pp（cross-attn 在 CfC 上仍有貢獻）；B3 - B2 = -2.85 pp（CfC 在 cross-attn 上仍有貢獻）
- **Operator framework >> Standard PINN**: B0 - PINN = -20.0 ~ -21.3 pp

## 3.2 Architecture × Sensor sweep at 1024 collocation（EXP-244 + EXP-245~251）

全 1024 collo + seed=42 對齊比較。EXP-245 為工程可遷移主 baseline（B3 + 1-head + LES_T50），EXP-244 為 4-head cross-attn DNS oracle upper reference；EXP-246~250 為同一 LES_T50 sensor 下的架構對照。EXP-200_a~e 保留為 DNS-sensor legacy multi-seed statistical reference。

| ID | Status | Architecture | Sensor | KE rel-err | u L₂ | v L₂ | ω L₂ | div L₂ | div_ratio | band low (t=5) | Train wall (RTX 3090) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-244 | `ORACLE_REFERENCE` | B3 + **4-head** | DNS | **5.51 %** | 16.30 % | 19.69 % | 44.95 % | 0.0436 | 0.0049 | **1.46 %** | 1:16:40 |
| **EXP-245** (10k single, archived) | `HISTORICAL` | B3 (1-head) | **LES_T50** | 6.92 % | 14.51 % | 19.25 % | 44.32 % | 0.0492 | 0.0055 | 2.85 % | 1:19:53 |
| **EXP-245** (**20k n=5**) | **`ACTIVE_BASELINE` 🥇** | B3 (1-head) | **LES_T50** | **5.71 ± 0.11 %** | 13.65 ± 0.06 % | 17.52 ± 0.10 % | 41.79 ± 0.12 % | — | **0.0039 ± 6e-5** | — | ~2:30:00 |
| EXP-251 | `ACTIVE_REFERENCE` | B3 + **4-head** | LES_T50 | **6.68 %** | 14.36 % | 19.03 % | 43.89 % | 0.0481 | 0.0054 | 2.62 % | (parallel run) |
| EXP-246 | `ACTIVE_REFERENCE` | B0 (vanilla) | LES_T50 | 9.96 % | 16.59 % | 22.55 % | 47.94 % | 0.0557 | 0.0063 | **0.72 %** | 0:24:58 |
| EXP-247 | `ACTIVE_REFERENCE` | B1 (no cross-attn) | LES_T50 | 10.62 % | 18.55 % | 25.51 % | 52.27 % | 0.0677 | 0.0076 | 3.85 % | 0:52:43 |
| EXP-248 | `ACTIVE_REFERENCE` | B2 (no CfC) | LES_T50 | 8.43 % | 15.94 % | 21.30 % | 47.26 % | 0.0528 | 0.0059 | 5.54 % | 0:46:59 |
| EXP-249 | `ACTIVE_REFERENCE` | Standard PINN SiLU | LES_T50 | 10.13 % | 14.35 % | 19.19 % | 44.35 % | **0.0244** | **0.0027** | 3.01 % | 0:38:07 |
| EXP-250 | `ACTIVE_REFERENCE` | Standard PINN tanh | LES_T50 | 13.09 % | 17.37 % | 22.98 % | 49.41 % | **0.0157** | **0.0018** | 4.50 % | 0:31:15 |

**5 個 paper-grade findings**:

1. **EXP-244 (4-head) 取代 EXP-241_b 為新 stable best** — KE 5.51 % (-0.46 pp vs 1-head)。Multi-head cross-attn 不增 param 但提高 attention 表達力。

2. **1024 collo 大幅縮小 DNS↔LES_T50 gap, 20k baseline 完全 close gap**:
   - 64 collo: DNS 9.40% / LES_T50 12.36% → gap **2.96 pp** (EXP-220 vs EXP-221)
   - 1024 collo 10k: DNS 5.97% / LES_T50 6.92% → gap **0.95 pp** (EXP-241_b vs EXP-245 10k)
   - **1024 collo 20k**: DNS 10k 5.97% vs LES_T50 20k **5.71 ± 0.11 %** → **LES 20k 已優於 DNS 10k** (paper claim: LES proxy pipeline 在足夠訓練後 match DNS oracle)

3. **Architecture ranking 在 LES_T50 + 1024 collo 重新洗牌**:
   - B3 (5.71 @ 20k / 6.92 @ 10k) > B2 (8.43) > **B0 (9.96)** > PINN-SiLU (10.13) > **B1 (10.62)**
   - **B0 vanilla DeepONet 反超 B1 (CfC, no cross-attn)** — 暗示 **cross-attention 比 CfC 更重要** 在 LES + 高 collo 環境

4. **PINN 1024 collo 大幅 improvement**:
   - PINN-SiLU: 38.50 % (64 collo, EXP-204) → **10.13 %** (1024 collo, EXP-249), -28.4 pp
   - 「plain MLP PINN 比 operator framework 對 collo density 更敏感」— physics regularization 對 PINN 是 dominant lever
   - 但 absolute KE 仍輸 operators (B3 5.71 ± 0.11 @ 20k < PINN-SiLU 10.13)
   - PINN div_L2 0.024/0.016 反而最低 — PINN 對 incompressibility 嚴格滿足，trade-off vs sensor data fit; 但 EXP-245 20k 已達 div ratio 0.39 % < DNS floor 1.04 %，**operator + 長訓 = best of both**

5. **PINN tanh outlier 13.09 %** confirm SiLU > tanh activation choice（EXP-250 vs EXP-249 +2.96 pp）

## 3.3 Architecture × Placement 2×3 ablation（EXP-240, 2026-05-19 完成）

| ID | Architecture + Placement | KE rel-err |
|---|---|---|
| **EXP-240_a** | B0 + LES_T50 (seed=42) | **19.58 %** |
| **EXP-240_b** | B0 + Random (seed=42) | **21.82 %** |

完整 2×3 表：

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

---

# §4 對照群 B：Sensor Placement

## 4.1 Placement strategy 比較（EXP-220~224, B3 + 64 collo + seed=2, axis-fix v2）

| ID | Status | Placement strategy | KE rel-err | Δ vs oracle | 工程可遷移性 |
|---|---|---|---|---|---|
| **EXP-220** | `ACTIVE_REFERENCE` | DNS QR-pivot K=100（**oracle**）| **9.40 %** | — | 無（需 DNS）|
| **EXP-221** | `ACTIVE_REFERENCE` | LES_N256 **T=50 stat-converged, random IC** + QR-pivot | 12.36 % | +2.96 pp | **強**（real-world 完全 DNS-free，**論文 engineering pivot 主代表**）|
| **EXP-222** | `ACTIVE_REFERENCE` | LES_N128 T=15 Bardina over-disp stand-alone + QR-pivot | 12.40 % | +3.00 pp | 強（**low-fidelity LES viable**：N=DNS/2 + 計算 1/16）|
| **EXP-224** | `ACTIVE_REFERENCE` | Random uniform K=100 (seed=42, 10k) | 13.25 % | +3.85 pp | 強（無需 LES）|

> **Note**: EXP-220 與 EXP-200_c 都是 B3 + DNS QR-pivot + seed=2，差異僅在報告角度（前者 placement ablation, 後者 multi-seed group）。實質訓練 artifact 完全相同。
>
> **EXP-221 vs EXP-222 重點差異**: 兩者都「real-world DNS-free」可遷移，但 (a) EXP-221 N=256 同 DNS grid + T=50 26.5 turnovers + α=1.8 譜形接近 DNS（slope −6.46 vs DNS −4.75）；(b) EXP-222 N=128 粗網格 + T=15 8.5 turnovers + α=30 過耗散（slope −14）。KE 幾乎打平（12.36 % vs 12.40 %）→ 論文可主張「**LES 解析度與譜形對齊都不是 bottleneck**，statistical convergence + 正確 axis convention 才是」。
>
> **EXP-223 (LES_N256 T=30 dns-init) 已移除**: 同時工程不可遷移（需 DNS IC）+ 效果不如 EXP-221（13.08 % > 12.36 %）。
> **EXP-225 (LES_T5) 已移除**: T_end < 1 turnover，非 statistically-converged LES，KE 23.48% outlier。

### Paper-grade findings
1. **LES proxy pipeline viable**: 3 個 well-formed cross-source placements（EXP-221/222/224）達 KE 12-13% (gap to oracle ~3pp)
2. **LES 解析度與譜形對齊都不是 bottleneck**: EXP-221 (N=256 譜接近 DNS) ≈ EXP-222 (N=128 過耗散 slope −14) — KE 差 < 0.05 pp
3. **Statistical convergence 才是 gating**: T_end ≥ 8 turnovers 即夠
4. **Random ≈ well-formed LES**: K=100 sparse regime 下 placement 演算法影響有限（< 1 pp）
5. **Real-world engineering pipeline 可行**: 低成本 LES + QR-pivot + 量測 → 重建 達 baseline-quality

## 4.2 Placement variance（EXP-266_a~e, Random K=100 × n=5 placement seeds, 2026-05-22）

**Setup**: training seed=42 固定（隔離 training stochasticity），改 placement seed 42/1/2/3/4；對齊 EXP-245 baseline 20k iter + warmup all 2000。

| ID | Placement seed | KE rel-err | u L₂ | v L₂ | ω L₂ | Ens rel-err | div ratio | k_f amp |
|---|---|---|---|---|---|---|---|---|
| EXP-266_a | 42 | 7.24 % | 16.14 % | 20.33 % | 44.89 % | 28.49 % | 0.37 % | **1.001** |
| EXP-266_b | 1 | **9.18 %** ⚠️ outlier | 19.80 % | 25.64 % | 49.66 % | 32.15 % | 0.34 % | 0.941 |
| EXP-266_c | 2 | 7.89 % | 16.48 % | 20.86 % | 44.96 % | 27.89 % | 0.36 % | 0.962 |
| EXP-266_d | 3 | 8.03 % | 16.10 % | 20.16 % | 44.73 % | 27.86 % | 0.35 % | 0.966 |
| EXP-266_e | 4 | 7.40 % | 17.47 % | 21.11 % | 46.10 % | 28.82 % | 0.34 % | 0.995 |
| **mean ± std** | n=5 | **7.95 ± 0.68 %** | 17.20 ± 1.42 % | 21.62 ± 2.07 % | 46.07 ± 1.92 % | 29.04 ± 1.66 % | **0.35 ± 0.01 %** | 0.973 ± 0.024 |

### Headline — Placement variance vs Training variance comparison

| Group | Variance source | n | KE mean | KE σ |
|---|---|---|---|---|
| **EXP-245** (a~e) | Training seed (LES_T50 placement 固定) | 5 | **5.71 %** | **0.11 %** |
| **EXP-266** (a~e) | Placement seed (Random K=100, training seed=42 固定) | 5 | **7.95 %** | **0.68 %** |

**Key findings (paper-grade)**:

1. **σ_placement / σ_training = 0.68 / 0.11 = 6.2×** — **placement variance dominate, 6 倍於 training stochasticity**
2. **LES_T50 vs Random K=100 mean gap = 2.24 pp** (Welch t-test: gap / σ_random ≈ 3.3, **statistically significant p < 0.01**)
3. **LES_T50 placement 不只 mean better, σ 也小 6×** — placement strategy 既影響 KE 平均值也影響 reproducibility
4. **div ratio 0.35 ± 0.01 % 在所有 placement seeds 都 < DNS floor 1.04 %** — continuity 控制 **placement-invariant** (sub-DNS divergence claim 對 placement robust)
5. **EXP-266_b (pseed=1) 是 outlier (9.18 %, +1.23 pp vs group mean)** — Random placement 偶爾 hit bad spatial coverage; LES-derived placement 避免此 outlier (EXP-245 σ=0.11 % 無 outlier)

**Paper §Sensor Placement 新主張**:

> "Sensor placement strategy contributes a variance source 6.2× larger than training stochasticity (σ_placement 0.68 % vs σ_training 0.11 %, both n=5 at fixed K=100). LES-derived QR-pivot placement also improves mean KE by 2.24 percentage points (5.71 % vs 7.95 %, z ≈ 3.3). Engineering deployment should prioritize placement optimization over training repetition."

## 4.3 DNS-pivot oracle multi-seed（EXP-271, B3 + DNS QR-pivot + 1024 collo + 20k n=5, 2026-05-29）

> **目的**: EXP-245 使用 LES_T50 sensor（工程可遷移）；EXP-271 在**完全相同訓練 config**（B3 / 1024 collo / 20k / seeds 42/1/2/3/4）下換回 DNS QR-pivot sensor，作為「oracle upper bound」。§4.1 的 oracle（EXP-220）使用 64 collo / 10k / 1 seed，無法直接與 EXP-245 做 fair 統計比較；EXP-271 補全此缺口。

**Setup**:
- Config: `configs/exp_271_b3_dns_pivot.toml` (seed=42) + `exp_271{b,c,d,e}_b3_dns_pivot_seed{1,2,3,4}.toml`
- Sensor: `data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100.{json,npz}`（DNS QR-pivot oracle）
- Architecture / collocation / iter / seeds: **完全對齊 EXP-245** — B3 1-head cross-attn, 1024 collo, 20k iter, seeds 42/1/2/3/4
- Artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp271{,b,c,d,e}-b3-dns-pivot-*-20k/`
- Eval: `artifacts/_lab_rsync/eval_271_seed{a,b,c,d,e}/summary.json`（rsync 2026-05-29）
- Slurm: jobs 3696/3707~3710 train + 3713/3715/3717~3719 eval, acmt20 (RTX 3090)

### EXP-271 n=5 multi-seed metrics

| Seed | KE rel-err | band_low rel-err (last) | div ratio |
|---|---|---|---|
| _a (42) | 4.69 % | 2.77 % | 0.362 % |
| _b (1)  | 4.70 % | 2.78 % | 0.363 % |
| _c (2)  | 4.66 % | 2.75 % | 0.360 % |
| _d (3)  | 4.77 % | 2.86 % | 0.370 % |
| _e (4)  | 4.60 % | 2.75 % | 0.368 % |
| **mean ± std** | **4.68 ± 0.06 %** | **2.78 ± 0.04 %** | **0.365 ± 0.004 %** |

### EXP-245 vs EXP-271 fair comparison（同配置 20k n=5，僅 sensor 來源不同；全用論文一致 time-mean 欄位）

| 實驗 | Sensor 來源 | KE | u L2 | v L2 | ω L2 | div ratio | kf amp |
|---|---|---|---|---|---|---|---|
| **EXP-271** | DNS QR-pivot oracle | **4.68 ± 0.06 %** | 15.34 ± 0.06 % | 18.10 ± 0.03 % | 42.41 ± 0.12 % | 0.36 ± 0.004 % | 0.986 |
| **EXP-245** | LES_T50（工程可遷移）| 5.71 ± 0.11 % | **13.65 ± 0.06 %** | **17.52 ± 0.10 %** | **41.79 ± 0.12 %** | 0.39 ± 0.006 % | 0.991 |
| **誰贏** | — | DNS +1.03 pp | **LES +1.69 pp** | **LES +0.58 pp** | LES 略 | ~平 | ~平 |

### Paper-grade findings（2026-05-29 修正：先前誤用 u_rel_l2_last，已改用論文一致的 time-mean u_rel_l2_mean）

1. **整體 vs 逐點 trade-off**：DNS oracle 整體能量(KE)更準(4.68 vs 5.71)，但 **LES placement 逐點場(u/v/ω L2)反而更準**(u 13.65 vs 15.34)。兩者各有所長，無人全面碾壓。
2. **「no measurable penalty」claim 已死**：兩組 KE 95% CI 不重疊([4.62,4.74] vs [5.61,5.81])，不能宣稱 statistically indistinguishable。改用 **trade-off framing**（已寫入 paper：abstract / ch4:45 / ch5:7,11）。
3. **gap 量級對比**：random placement KE ~59 % → 兩種 well-formed placement 都壓到 4.7–5.7 %；DNS↔LES 之間僅 1 pp 量級差異，placement strategy 的「演算法選擇」遠不如「有沒有 well-formed placement」重要。
4. **div_ratio**: EXP-271 0.36 % vs EXP-245 0.39 %（已降級為 diagnostic，見 §9.4，**不寫 sub-DNS**）。
5. **Paper claim（§Results）**: 「LES-derived placement 與 DNS oracle 互有取捨（DNS 贏整體能量、LES 贏逐點），LES 競爭力足且無需 DNS 全場 → REAL_WORLD_PIPELINE 工程可遷移。」

---

# §5 對照群 C：Sensor Amount（K-scaling）

> **2026-05-21 finalized (10k)**: 三點 K-scaling curve 完成。
> **2026-05-24 升級 (20k)**: plateau 分析顯示 EXP-256/257 在 10k 均未收斂；重跑 EXP-269 (K=200 20k) + EXP-270 (K=400 20k)，warmup 顯式固定 2000 step 對齊 EXP-245。KE 大幅改善：K=200 3.91→2.47%，K=400 2.90→1.76%。

## 5.1 K=100 → 200 → 400 sweep

| ID | Status | Configuration | K | collo | iter / n | KE rel-err | u L₂ (last) | v L₂ (last) | ω L₂ (last) | div ratio | k_f amp ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-245 (20k n=5) | `ACTIVE_BASELINE` | B3 + LES_T50, K=100 reference | 100 | 1024 | 20k / 5 | **5.71 ± 0.11 %** | 13.65 ± 0.06 | 17.52 ± 0.10 | 41.79 ± 0.12 | **0.39 ± 0.006 %** | **0.991 ± 0.005** |
| EXP-245 (20k seed=42) | `ACTIVE_REFERENCE` | 同上, single seed | 100 | 1024 | 20k / 1 | 5.90 % | 7.10 % | 16.08 % | 38.03 % | 0.39 % | 0.969 |
| **EXP-269** | **`ACTIVE_REFERENCE`** | 同上, **K=200**, tm_warmup=2000 fixed | 200 | 1024 | **20k / 1** | **2.47 %** | **4.95 %** | **10.30 %** | **31.36 %** | 0.40 % | **0.998** |
| **EXP-270** | **`ACTIVE_REFERENCE`** | 同上, **K=400**, collo=512, tm_warmup=2000 fixed | 400 | **512** | **20k / 1** | **1.76 %** | **4.11 %** | **8.64 %** | **29.29 %** | 0.43 % | 0.975 |
| EXP-256 | `HISTORICAL` | K=200, 10k single seed (未 plateau) | 200 | 1024 | 10k / 1 | ~~3.91 %~~ | 10.84 % | 13.92 % | 38.84 % | 2.18 % | 0.989 |
| EXP-257 | `HISTORICAL` | K=400, 10k single seed, collo=512 (未 plateau) | 400 | 512 | 10k / 1 | ~~2.90 %~~ | 9.46 % | 12.15 % | 36.78 % | 0.56 % | 0.965 |

## 5.2 Two-layer framing: Spectrum cut-off Nyquist vs scalar KE

⚠️ **REVISED 2026-05-22 v2 (二次修正; 區分 spectrum-domain vs scalar KE)**: 早期將 KE ∝ 1/√K 標為「Nyquist 帶寬律完美吻合」混淆了兩個 independent claim。

### Layer 1 (✅ strong, paper-grade): Spectrum-domain Nyquist cut-off ∝ √K

| K | Sensor spacing d = L/√K | Nyquist k_max ≈ π/d ~ √(K/π) | Spectrum visual verification |
|---|---|---|---|
| 100 | 0.10 | **5.64** | E(k) reconstruction 在 k ≤ 5.64 緊貼 DNS, k > 5.64 開始 deviate |
| 200 | 0.071 | **7.98** | cut-off 推至 k ≈ 8, 對應 inertial/dissipation 邊界附近 |
| 400 | 0.050 | **11.28** | cut-off 進入 forward enstrophy cascade |

- **這是嚴謹 sampling theorem 在 spectrum domain 的應用**（random sampling 推廣, Cohen 2009 / Manohar 2018 compressive sensing bound）
- **Paper §Theory 強 claim**: 「PI-CON spectrum reconstruction follows Nyquist k_max ≈ √(K/π) — the cut-off separates accurately reconstructed (k ≤ k_max) from irrecoverable (k > k_max) bands.」

### Layer 2 (❌ wrong, 之前 over-claim): Scalar KE ∝ 1/√K 不是 universal scaling

KE 是 spectrum 的積分量，對 K 的 scaling 不繼承 spectrum cut-off 的 √K：

$$
\text{KE rel-err} = \frac{|\int_0^\infty [E_\text{pred}(k) - E_\text{DNS}(k)] \, dk|}{\int_0^\infty E_\text{DNS}(k) \, dk}
$$

| 誤差來源 | 形式 | 對 K 的真實 scaling |
|---|---|---|
| (A) k > k_max truncation | $\int_{k_\max}^\infty E_\text{DNS}(k)\,dk$ | $\propto K^{-(p-1)/2}$ |
| (B) k ≤ k_max reconstruction imperfection | $\int_0^{k_\max} \|E_\text{pred} - E_\text{DNS}\|\,dk$ | 跟 spatial sampling d/δ_ω 有關 |

**Layer 2 修正**: KE rel-err 在我們的 case 由 **(B) 主導**（不是 (A)）：

| Metric | K=100 | K=200 | K=400 | Δ (100→400) | Layer 2 interpretation |
|---|---|---|---|---|---|
| Sensor spacing d = L/√K | 0.1 | 0.071 | 0.05 | **−50 %** | spatial sampling lever |
| Re=10⁴ vorticity layer δ_ω ~ Re^{−1/2} | 0.01 | 0.01 | 0.01 | flat | characteristic flow scale |
| **d/δ_ω under-sampling (Re=10⁴)** | **10×** | **7.1×** | **5×** | −50 % | (B) reconstruction quality lever |
| KE rel-err (20k single seed) | 5.90 % | 2.47 % | **1.76 %** | **−70 %** | (B) dominant |
| ω rel-L₂ | 44.32 % | 38.84 % | 36.78 % | −17 % | high-band tail bounded by spectrum + Layer 1 truncation |
| Ens rel-err | 27.51 % | 22.20 % | 20.47 % | −26 % | ∫k²E(k) high-band dominated (Layer 1 truncation) |
| k_f amp ratio | 0.926 | **0.989** | 0.965 | +4.2 % | forcing-mode 在 k=2 ≪ k_max 已 well-resolved |
| div ratio | 2.41 % | 2.18 % | 0.56 % | −77 % | (B) continuity 大幅改善 |

**核心 framing (UNIFIED)**:
- **(I) Spectrum cut-off k_max ∝ √K is rigorous** — visual confirm in spectrum plots, **strong paper claim**
- **(II) KE rel-err 不繼承 √K scaling** — 因 KE = ∫E(k)dk 是 integral; 改善由 (B) reconstruction quality (d/δ_ω) 主導
- **Re=10⁴ KE 三點看似「1/√K fit」是 narrow-range coincidence**

**Caveat — 不嚴格對齊**:
- collo: EXP-245/269 用 1024, EXP-270 用 **512**（K=400 + 1024 collo OOM at RTX 3090 22.69/24 GB）
- iter / seed: EXP-245 為 20k n=5 multi-seed baseline; EXP-269/270 為 20k single seed (seed=42)
- **warmup**: EXP-269/270 顯式設 `time_marching_warmup_steps = 2000`（fixed）

---

# §6 對照群 D：Sensor Noise Robustness

> **2026-05-21 finalized**: 4 個 noise level (1 % / 3 % / 5 % / 10 %), base on EXP-245 baseline (B3 + LES_T50 + 1024 collo + seed=42), per-channel std-relative Gaussian additive injection。

| ID | Status | Noise σ | KE rel-err | Δ vs clean | u L₂ | v L₂ | ω L₂ | Ens rel-err | k_f amp ratio |
|---|---|---|---|---|---|---|---|---|---|
| EXP-245 (10k n=1, archived) | `HISTORICAL` | 0 % (clean, **10k baseline**) | 6.92 % | — | 14.51 % | 19.25 % | 44.32 % | 27.51 % | 0.926 |
| EXP-245 (20k n=5, current) | `ACTIVE_BASELINE` | 0 % (clean, **20k baseline**) | **5.71 ± 0.11 %** | — | 13.65 ± 0.06 | 17.52 ± 0.10 | 41.79 ± 0.12 | 24.11 ± 0.21 | **0.991 ± 0.005** |
| EXP-258 | `ACTIVE_REFERENCE` | 1 % | 6.89 % | -0.03 pp | 14.48 % | 19.09 % | 44.17 % | 27.83 % | 0.971 |
| EXP-259 | `ACTIVE_REFERENCE` | 3 % | 6.84 % | -0.08 pp | 14.49 % | 19.20 % | 44.31 % | 27.96 % | 0.974 |
| EXP-260 | `ACTIVE_REFERENCE` | 5 % | 7.07 % | +0.15 pp | 14.71 % | 19.47 % | 44.67 % | 28.56 % | **0.982** |
| EXP-261 | `ACTIVE_REFERENCE` | 10 % | 7.14 % | +0.22 pp | 15.18 % | 20.12 % | 45.49 % | 29.44 % | 0.959 |

**Finding 1 — PI-CON 對 sensor noise 高度 robust**:
- 1–10 % noise 範圍 KE rel-err 變化僅 **-0.08 到 +0.22 pp absolute**；single-seed 下可視為對 noise 高度 robust，而非明顯 monotone degradation
- 即使 10 % noise（量級 = sensor std 的 10 %, 工程現場 worst case），KE 7.14 % 仍 < EXP-224 random K=100 placement 13.25 % → **architecture 退步 < placement 退步**
- ω rel-L₂ 43.95 → 45.49 % (10 % noise)，**+1.54 pp** absolute → noise 影響的也是 low-band，high-band 已被 K=100 Nyquist 限制

**Finding 2 — Noise 的 implicit regularization 效果（surprising）**:
- 1–5 % noise 區段 k_f amp ratio **比 clean baseline 略好**（0.97~0.98 vs clean 0.926）
- 解讀：sensor noise 對 over-fit sensor MSE 起 weak regularization 作用，使 model 更貼近 forcing prior 而非 fit 個別 sensor 量值的細節
- 但 10 % noise k_f amp 0.959 略 regress → trade-off curve 存在 sweet spot

**Finding 3 — 1 % vs 3 % statistically indistinguishable**:
- EXP-258 (1 %) 6.89 % vs EXP-259 (3 %) 6.84 % — non-monotone
- Single seed 隨機性 mask 小 noise level 差異; 5 % 之後 (7.07 → 7.14) monotone
- Paper-grade claim 需 multi-seed n ≥ 3 確認 noise scaling 是否 linear

**Take-away**:
1. **「PI-CON robust to 10 % sensor noise (KE +0.22 pp vs clean engineering baseline)」是 strong engineering claim** — 可寫入 §Discussion 與 §Conclusion 的 deployability 段
2. Noise injection 對 forcing-mode recovery 有 weak regularization 效果是 surprising side-finding

---

# §7 對照群 E：vs Classical Interpolation

> **數據來源**: [`docs/archive/squeeze_report_2026-05-11.md`](archive/squeeze_report_2026-05-11.md) (Phase 1-7, 2026-05-11)。Subject model = **EXP-080 (= legacy EXP-200_a, B3 single seed=42, 64 collo, KE 10.68%)**。比較基準是早期 EXP-080 設定，**不是當前 EXP-245 20k baseline**；EXP-245 升級後 PI-CON 在 Pareto 上會進一步往「lower KE + better pointwise」靠（KE 10.68 → 5.71%, u L2 17.0 → 13.65%）。

## 7.1 Fair baselines（no DNS access during training/inference）

| Method | KE % | u L2 % | v L2 % | ω L2 % | Notes |
|---|---|---|---|---|---|
| **EXP-080 (CfC-DeepONet-PINN, ours)** | 10.68 | **17.0** ⭐ | **20.2** ⭐ | **47.6** ⭐ | Our method (legacy single seed) |
| RBF Gaussian (ε=10) | 6.83 | 33.81 | 38.69 | 59.59 | Smooth interpolation |
| **RBF Multiquadric (ε=10)** | **4.10** ⭐ | 32.84 | 37.70 | 58.38 | **Lowest KE among classical** |
| RBF Thin-plate-spline | 8.60 | 31.48 | 35.93 | 58.67 | Smooth interpolation |
| IDW (p=2) | 62.95 | 53.70 | 61.99 | 81.20 | Catastrophic over-localization |
| **Div-free trig LSQ, k ≤ 5 (80 modes)** | **3.93** ⭐ | 28.19 | 34.39 | 64.78 | **Mathematical optimum for KE** (over-determined LSQ at sensor info bound) |
| Div-free trig LSQ, k ≤ 8 (196 modes) | 6337.45 | 607.45 | 916.10 | 1259.56 | ❌ Numerical explosion (just-determined, ill-conditioned) |
| Div-free trig LSQ, k ≤ 12 (440 modes) | 72.92 | 145.98 | 184.29 | 520.02 | ❌ Under-determined → high-k noise |

## 7.2 DNS-supervised reference（engineering non-transferable, upper bound）

| Method | KE % | u L2 % | v L2 % | Notes |
|---|---|---|---|---|
| Gappy POD r=50 | 0.38 | 2.72 | 2.72 | Cheats with DNS-trained basis |
| Gappy POD r=100 | 0.12 | 0.85 | 0.85 | Same |
| Gappy POD r=150 | 0.04 | 0.37 | 0.37 | Same |

## 7.3 Mathematical ill-posedness（SVD null-space proof）

For periodic Fourier basis up to $k_{\max} = 16$:
- Total div-free degrees of freedom: $M_{\rm div\text{-}free} = 1{,}592$
- Sensor rank constraint: $K = 100$
- **Null-space dimension**: $1{,}592 - 100 \times 2 = 1{,}392$ (87.4%)

**87.4% of the structurally valid (divergence-free) field components are completely invisible to K=100 sensors**. Sparse-sensor reconstruction at K=100 is **provably ill-posed**: the role of any method is to choose a preferred element from the 1,392-dim null space; the implicit **prior** determines which element.

## 7.4 Pareto trade structure (key finding)

```
                       KE rel-err
                            ↑
                     10% ●  EXP-080 (Ours)
                            ↓ Pareto frontier
                      5% ●  Multiquadric, Trig LSQ k≤5
                            ↓
                      1% ●  (spectral bound k≥6)
                            ↓
                      0%·····················
                              ↓
                   0%   10%  20%  30%  40%   pointwise u rel-L2 →
                          ↑
                      Our model (17%)        ↑ Multiquadric (33%), Trig LSQ (28%)
```

**Headline finding (paper §Discussion)**: Among 7 fair (no-DNS) baselines, our PINN exhibits a **Pareto-favorable trade**:
- Sacrifices ~7pp on scalar KE (10.68% vs best 3.93%)
- Gains **11–14pp on all pointwise field metrics** (u, v, vorticity rel-L2)

The trade exposes a **previously undocumented systematic phenomenon**: classical methods optimize KE through systematic over-smoothing (predicting essentially the spatial mean). Linear Fourier basis at the sensor's information bound (k_max ≈ 5.64) is mathematically optimal for KE but structurally suboptimal for pointwise reconstruction.

## 7.5 Why classical methods over-smooth

All classical methods (RBF, IDW, trig LSQ at low k_max) implement variants of weighted averaging:
- **RBF**: weighted by kernel decay
- **IDW**: weighted by $1/r^p$
- **Trig LSQ k ≤ 5**: weighted by Fourier basis up to k=5 only

These methods produce smooth reconstructions because their basis functions are smooth. Predicting $\mathbf{u} \approx \langle \mathbf{u}_{\rm DNS} \rangle$ minimizes $|KE_{\rm pred} - KE_{\rm DNS}|$ but destroys pointwise structure.

## 7.6 Paper-grade claims

1. **KE-as-misleading-metric**: classical methods achieve low KE through over-smoothing; multi-metric evaluation (KE + u/v/ω rel-L2 + ek_ratio_kf + div) should be standard for sparse-sensor benchmarks
2. **Our PINN on Pareto frontier (pointwise side)**: 11-14 pp better pointwise accuracy than fair baselines
3. **Provably ill-posed at K=100**: 87.4% null-space; any algorithm's output is determined by its implicit prior
4. **Updated baseline section template for paper**:
   ```
   Group A — Classical interpolation: RBF, IDW, Div-free Trigonometric LSQ
   Group B — Recent neural operators (literature): Physics-Constrained CNN [arXiv:2409.00260],
             RecFNO [arXiv:2302.09808], FLRONet [arXiv:2412.08009]
   Note: no published work matches our setup (Kolmogorov Re=10000, K=100 sensors,
   no DNS field loss) — literature gap this paper fills.
   ```

> **Scripts / artifacts**: `scripts/baseline_squeeze.py`, `scripts/under_determined_proof_divfree.py`, JSON 結果在 `artifacts/under_determined_proof/`，主圖 `under_determined_demo.png`。

---

# §8 Inference Cost Benchmark

| Hardware | Model | Eval wall (full snapshot 評估) | per-snapshot avg |
|---|---|---|---|
| M3 base (4P+4E, MPS) | B3 (EXP-094, legacy) | — | encoder 71 ms + query 1.5 ms |
| RTX 3090 (lab acmt20) | B0 (EXP-240, 201 snapshots) | 16 s/model | ~80 ms/snapshot (full eval pipeline) |
| RTX 3090 (lab acmt20) | B3 (EXP-241, 201 snapshots) | 79 s/model | ~390 ms/snapshot (full eval pipeline incl. spectral) |

> **Note**: 上述 RTX 3090 數字含整套 evaluator pipeline（場重建 + 譜估 + KE/div/能譜計算 + 繪圖），非純 inference latency。Paper-grade pure encoder/query benchmark 仍以 EXP-094 M3 baseline (71+1.5 ms) 為主 reference；RTX 3090 paper-grade benchmark 待 `scripts/benchmark_inference.py` 重跑。

---

# §8.5 Uncertainty Quantification — Split Conformal Prediction（EXP-245 post-hoc, 2026-05-29）

> **[STATUS: VALIDATED · PARKED]** — 結果已驗證且 reproducible，但**暫不寫入論文**，保留待後續使用。
> Post-hoc UQ：不重訓、不改架構，對 EXP-245 (seed=42, `multiseed/seeda`) 套 split conformal。
> 腳本 [`scripts/conformal_prediction.py`](../scripts/conformal_prediction.py)；artifact `artifacts/conformal_exp245/`。
> 內建 sanity gate（KE rel-err 6.72% PASS，確認 `multiseed/seeda` = EXP-245 LES_T50，非 legacy EXP-200）。

**兩條路徑（對齊 ENGINEERING_VISION）**：
- **Path A (transferable, headline)**：calibration = 均勻隨機 held-out 位置（排除訓練 100 sensor），只用點量測 → 工程可遷移。n_cal=200（現實額外 sensor 預算）。
- **Path B (oracle, 僅研究用)**：calibration = DNS 全場隨機點，n_cal=5000 → full-field 保證，**工程不可遷移**。

**Marginal coverage（multi-draw n=50，mean±std；u/v 平均）**：

| α | target | Path A fixed | Path B fixed |
|---|---|---|---|
| 0.05 | 0.95 | 0.955 ± 0.015 | 0.950 ± 0.004 |
| 0.10 | 0.90 | 0.906 ± 0.020 | 0.899 ± 0.007 |
| 0.20 | 0.80 | 0.805 ± 0.029 | 0.801 ± 0.009 |

→ 兩路徑 marginal coverage 皆精準命中 target。Path A std 較大（n_cal=200 有限樣本），但仍 center 在 1−α，證明**工程可遷移 CP 成立**（不需 DNS 即可給統計保證）。

**Adaptive σ 比較（不重訓；三個工程可遷移候選 + fixed）**：
以「到最近 sensor 距離」分四分位 bin，coverage spread = max−min（越低越均勻）。α=0.1, u, Path B oracle（n_cal=5000, 50 draws）：

| 方法 σ | marginal cov | mean halfwidth | **conditional spread** |
|---|---|---|---|
| fixed-width | 0.900 | 0.0962 | 0.093 |
| adaptive: dist（原始距離）| 0.901 | 0.1136 (+18%) | 0.160 ❌ over-correct |
| **adaptive: √dist（tempered）** | 0.901 | **0.0960（持平）** | **0.022** ✅ 3-4× 更均勻 |
| adaptive: PDE residual | 0.900 | 0.1161 (+21%) | 0.099 ≈ fixed |

conditional coverage by σ-quartile（α=0.1, u, Path B, near→far sensor）：

| | Q1 (near) | Q2 | Q3 | Q4 (far) |
|---|---|---|---|---|
| fixed-width | 0.952 | 0.909 | 0.880 | 0.859 |
| √dist (tempered) | 0.892 | 0.898 | 0.898 | 0.914 |
| PDE residual | 0.952 | 0.912 | 0.884 | 0.853 |

**發現（推翻原假設 "residual 最佳"）**：
1. **fixed-width 揭示誤差非空間均勻**：近 sensor over-cover (0.952)、遠 sensor under-cover (0.859) → 重建誤差確實隨離 sensor 距離增大。
2. **√dist 是贏家**：conditional spread 0.093→0.022（3-4× 更均勻）且平均寬度持平（誤差隨距離成長但**比距離緩**，√ 是對的 power）。工程可遷移（只需 sensor 座標）。
3. **PDE residual σ 失敗**：stratified 曲線幾乎完全疊在 fixed 上（residual 空間近乎常數），寬度 +21%。**根因**：(a) 模型被 physics loss + GradNorm 驅使 residual 空間均勻；(b) K=100 ill-posed null-space（§7.3）下**低 PDE residual ≠ 低重建誤差**，residual 與 error 解耦 → 反過來是 §7.3 ill-posedness 的另一證據。

**√dist power-law 檢驗（避免 diffusion overclaim）**：fit `|error| ~ dist^p` → p_pointwise 0.42 (u)/0.44 (v)，p_binned_median 0.45/0.46（接近 0.5，**溫和支持 √**）；但 log-log corr 僅 0.18（距離只解釋少量 pointwise 誤差變異）。→ √dist 是 **population-level scaling**，非 pointwise 物理定律；論文寫「empirically √dist tempering」，diffusion 機制列 plausible interpretation，不可當證明。

**時間外推 / 交換性壓測（α=0.1, u, target 0.90）**：

| t_split | i.i.d. (參考) | forward 早→晚 | reverse 晚→早 |
|---|---|---|---|
| 2.5 | 0.901 ± 0.009 | 0.997（保守）| 0.616（anti-conservative）|
| 4.0 | 0.902 ± 0.011 | 0.991 | 0.711 |

- **i.i.d. 校準 coverage 精準命中 0.90**（窗內交換性成立）。
- **時間外推破壞交換性，且方向不對稱**：forward（早校準→晚測試）over-cover ~0.99（保守，安全）；reverse（晚校準→早測試）崩到 0.62–0.71（**少 20–28 pp**，危險）。
- **根因**：重建誤差強烈時間非平穩——早期 t 誤差高（CfC temporal encoder warm-up / transient），晚期低。
- **operating-regime 邊界（寫入 paper）**：CP 保證僅在「calibration 與 target 同一時間窗（交換）」成立。部署時校準資料必須與預測窗交換，**禁止跨時間 regime 校準**。這也獨立佐證模型重建品質隨 CfC context 累積而改善。

**Paper framing**：headline = Path A + **√dist tempered adaptive**（工程可遷移、marginal coverage 命中 target、conditional coverage 均勻、寬度無代價）；Path B 為 oracle 上限；PDE-residual σ 為誠實負面結果連結 ill-posedness；時間外推為 operating-regime 邊界 + 非平穩性證據。
圖（`artifacts/conformal_exp245/`）：`calibration_curve.png`、`stratified_coverage.png`（紅=residual 疊在藍=fixed，綠=√dist 水平）、`time_extrapolation.png`（紅 reverse 崩至 0.62）、`residual_width_field.png`、`error_vs_time.png`（相對誤差 21%→5%，真實場 RMS 全程平穩 → 隔離 CfC context warm-up，非 flow transient）。

---

# §9 Diagnostics / Negative Findings

> 本章節記錄「跑過但證實不該往那走」的負面結果 — 對論文 §Discussion 仍有價值（誠實揭露所做的 ablation），但不該被誤讀為主線進度。

## 9.1 Multi-constraint AL ablation — **NS 加 AL = anti-pattern**（EXP-242 + EXP-243, 2026-05-20）

| ID | Status | GN tasks | AL constraints | use_gradnorm | KE rel-err | u L₂ | v L₂ | ω L₂ | div L₂ | band low (t=5) | Train wall | 一致原則 | 結論 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **EXP-242_a** | `ACTIVE_REFERENCE` | `[data, ns_u, ns_v]` | `[cont]` | true | **10.19 %** | 20.07 % | 24.28 % | 51.78 % | 0.0721 | 6.68 % | 1:02:58 | ✅ | cont 純 AL ≈ baseline 雙開（in 1 std） |
| **EXP-243** | `NEGATIVE_RESULT` | `[data]` (僅 data) | `[ns_u, ns_v, cont]` | **false** | **13.33 %** | 21.98 % | 26.45 % | 54.17 % | 0.0769 | 11.62 % | 1:03:05 | ✅ **完全** | 全 physics 純 AL, no GN — multi-AL 對 NS 仍反效果 |
| **EXP-242_c** | `NEGATIVE_RESULT` | `[data, cont]` | `[ns_u, ns_v, cont]` | true | **13.70 %** | 22.44 % | 27.09 % | 54.92 % | 0.0703 | 11.34 % | 1:03:43 | ⚠️ cont 雙開 | NS 純 AL + cont 雙開（部分違反） |
| **EXP-242_b** | `NEGATIVE_RESULT` | `[data, ns_u, ns_v, cont]` | `[ns_u, ns_v, cont]` | true | **14.79 %** | 23.01 % | 28.06 % | 55.77 % | 0.0712 | 12.99 % | 1:05:54 | ❌ 全雙開 | NS+cont 全雙開（GN+AL 互相 amplification）|

Decision gates 評估:

| Config | Gate | 結果 |
|---|---|---|
| 242_a (a) KE < 9 % 雙開冗餘 | (b) 9-12 % 雙開 ≈ 純 AL ✅ | ADR-001 §4 修訂中性 |
| 242_b (a) ≤ 9.5 % net positive | (c) > 11.5 % AL over-penalty ✅ | **NS 加 AL = anti-pattern** |
| 242_c vs 242_b (a) 純 AL 更乾淨 ✅ | (-1.09 pp) | 但 NS 加任何 AL 都不好 |

**Paper-grade findings**（含 EXP-243 一致原則 confirmation）:
1. **L_phys 低 ≠ KE 低**: 242_b L_phys 1.67e-2 (~9× ↓) 但 KE +4 pp 退步 — 經典 PINN over-physics 病態，AL pressure 過度 push NS → data fit 被犧牲
2. **cont AL 是 sweet spot, NS AL 不是**: cont (divergence) 是 hard 約束（incompressibility）； NS momentum 是 soft 引導，太強會 over-fit 物理解
3. **「開 AL = 拿出 GN」一致原則部分驗證 (EXP-243)**: 拿出 GN 比雙開乾淨，符合 ADR-001 §4 motivation
4. **但 NS 加 AL 本身仍 anti-pattern**: EXP-243 (一致原則 + 全 physics 純 AL + no GN) 仍 KE +2.56 pp 退步 vs baseline — 證明問題不在 GN 處理方式，而是 NS 不適合 AL pressure
5. **ADR-001 §4 對 cont 過保守，但對 NS valid**: cont 拿出 GN ≈ baseline；NS 拿出 GN 仍退步

**Take-away for paper**: 主線 EXP-200_a recipe (cont 雙開 + NS 只 GN, no NS-AL) **已是 multi-AL 配置中最佳**；不要試圖加 NS-AL。

## 9.2 Forcing-prior identifiability — **ill-posed (two-sided verified)**（EXP-252~255）

全 LES_T50 sensor + 1024 collo + seed=42 + 10k steps（同 EXP-245 setup）對齊比較。Forcing parameters 用 zero-ish init（`forcing_A_init=0.001, k_f_init=0.01`）— 與前次 high-init run（A=0.05, k_f=2.5）合併形成 two-sided test。

| ID | Configuration | A learned (truth 0.1) | k_f learned (truth 2.0) | KE rel-err | u L₂ | v L₂ | ω L₂ | Ens rel-err | div ratio |
|---|---|---|---|---|---|---|---|---|---|
| EXP-252 | forcing hardcoded（≡ EXP-245 10k single）| 0.1 (fixed) | 2.0 (fixed) | 6.92 % | 14.51 % | 19.25 % | 44.32 % | 27.51 % | 2.41 % |
| EXP-253 | learn k_f only, A fixed 0.1 | 0.1 (fixed) | **0.0102 (err 99.49 %)** | 6.82 % | 14.45 % | 19.01 % | 44.11 % | 27.78 % | 2.20 % |
| EXP-254 | learn A only, k_f fixed 2.0 | **0.00133 (err 98.67 %)** | 2.0 (fixed) | 6.84 % | 14.54 % | 19.21 % | 44.30 % | 27.88 % | 2.18 % |
| EXP-255 | learn both A + k_f | **0.00100 (err 98.96 %)** | **0.0103 (err 99.48 %)** | 6.82 % | 14.49 % | 19.09 % | 44.22 % | 27.83 % | 2.20 % |

**Finding 1 — Forcing identifiability ill-posed (two-sided verified)**:
- **k_f from zero-ish init**: 0.0100 → 0.0102（變化 < 0.001 over 10k steps），梯度完全卡死於 sigmoid flat region
- **A from zero-ish init**: 0.0010 → 0.00133（+33% 但仍離 truth 0.1 兩個量級）
- **vs 前次 high-init run**（已 archived）：k_f oscillate ±0.05 不收斂、A 反向漂移 0.05→0.045
- → **兩端 init 都驗證**：僅靠 sensor MSE + PDE residual，對 (A, k_f) 的 gradient signal **不足以 separately identify**

**Finding 2 — Forcing parameters wrong does *not* break flow reconstruction**:
- EXP-253/254/255 學到的 (A, k_f) 全錯，但 KE rel-err 6.82~6.84% 與 baseline 6.92%（EXP-245 10k single）幾乎一樣
- Model 直接 fit sensor data，PDE residual 用 wrong forcing 也能 self-consistent
- 物理解釋：forcing 對 u/v 的 contribution 量級小於 advection/diffusion 項 → flow reconstruction quality 對 forcing identification accuracy 不敏感
- **paper-grade claim**: 「sensor MSE-driven reconstruction is forcing-agnostic at K=100 budget; forcing identification requires either ① larger K or ② explicit forcing-mode supervision (e.g. spectral peak prior)」

## 9.3 RAR（Residual Adaptive Resampling）ablation — **FAILED → 已修正待重跑**（EXP-272, 2026-05-29）

> **背景**: EXP-054（舊架構 64 collo，無 GradNorm/AL）顯示 RAR freq=1000 可帶來 -2.2 pp；但現行架構（1024 collo + GradNorm + AL + SOAP）從未重測 RAR。本實驗驗證 RAR 在當前 stable config 下是否仍有益。

**Setup**:
- Config: `configs/exp_272_b3_les_T50_rar1000.toml`
- Base: EXP-245_a (seed=42, KE 5.90%) — **byte-identical 除以下三項**：
  - `physics_collocation_strategy = "rar"` (was "random")
  - `rar_update_freq = 1000`（freq < 1000 已知 SOAP preconditioner 失效）
  - `rar_pool_multiplier = 10`（pool = 1024 × 10 = 10240 候選點）
  - `rar_exploration_ratio = 0.2`（20% 隨機點防 mode collapse）
- Sensor: LES_T50（工程可遷移，同 EXP-245）
- Collocation: 1024 / Iter: 20k / Seed: 42
- Slurm: job 3721, acmt20 (RTX 3090), 2026-05-29

**Hypothesis**: RAR 偏向高殘差區域採樣，有助於高梯度時刻的 physics residual 收斂。

**Falsifiability**: 若 KE ≥ 5.90% 或訓練過程 L_phys 出現 spike（同 EXP-053 模式），則 RAR 在當前架構無效/有害。

**Status**: `NEGATIVE_RESULT` — RAR 在現行架構下淨有害（2026-05-30 重跑完成 + 評估）

**修 bug 過程**（首跑 job 3721 FAILED）:
- **首跑（job 3721, 2026-05-29）FAILED**：啟動 9 秒崩於第一次 RAR pool 更新。
  `RuntimeError: Trying to backward through the graph a second time`
  （[physics.py](../src/pi_con/physics.py) `_rar_update_pool` → `_g1`）。
- **根因**：u/v/p 是同一次 forward `uvp` 的 slice，共用同一張 autograd graph；
  `_g1` 三次一階 grad 未設 `retain_graph=True`，第一次 backward 釋放 saved tensors
  後第二次即 double-backward。RAR 路徑（`physics_collocation_strategy="rar"`）在現行
  架構從未被執行過，首次啟用即踩中，與 RAR 物理假設無關（純工程 bug）。
- **修正**：前兩次 `_g1` 改 `retain_graph=True`，最後一次釋放（commit `4c76b6b`）。
  regression test [`tests/test_rar_pool_autograd.py`](../tests/test_rar_pool_autograd.py)
  monkeypatch uvp_fn 重現崩潰並守住修正（修正前 raise、修正後 PASS）。

**重跑（job 3739, 2026-05-30, COMPLETED 2h43m）+ 評估結果**（同 evaluator, time-mean 欄位）:

| 指標 (time-mean) | EXP-245_a (random collo) | EXP-272 (rar freq=1000) | 判讀 |
|---|---|---|---|
| **KE rel-err** | **5.90 %** | **10.28 %** | +4.38 pp（1.74× 變差）|
| KE rel-err (val) | 6.53 % | 11.58 % | 變差 |
| u rel-L2 | 13.59 % | 18.59 % | 變差 |
| v rel-L2 | 17.53 % | 24.97 % | 變差 |
| omega rel-L2 | 41.66 % | 50.37 % | 變差 |
| Ens rel-err | 24.41 % | 34.54 % | 變差 |
| div_ratio (pred) | 0.385 % | 6.66 % | 17× 變差 |
| E(k_f=2) ratio @last | 0.969 | 0.856 | 偏離 1 |
| sensor MSE | 5.63e-4 | 1.14e-3 | 2× 變差 |
| EXP-054 (舊架構參考) | 19.6 % (from 21.8 %) | — | rar 64 collo, 無 GradNorm/AL |

**判讀（Falsifiability 命中）**：§9.3 預設「KE ≥ 5.90% 或 L_phys spike → RAR 無效/有害」。
- KE = 10.28% ≥ 5.90% → **Falsified，RAR 淨有害**，且**每一項指標都變差**（非單一指標雜訊）。
- 但**非經由 L_phys spike**：訓練 l_physics 單調下降（step 2000 的 30.5 → 20000 的 4.58），
  無 EXP-053 式 spike → 失效機制不是 SOAP preconditioner 失效/優化不穩。
- **物理機制（hypothesis）**：RAR 把 collocation 集中到高殘差區（高梯度/小尺度/早期 t），
  在 K=100 sparse-sensor underdetermined 系統中，physics residual 是 bulk 場的主要正則化；
  集中採樣使平滑 bulk 與 continuity 失去均勻約束 → 全域能量分布漂移（KE↑）、
  散度約束惡化（div_ratio 0.385%→6.66%）。連 sensor MSE 也 2× 變差，顯示非均勻 collocation
  破壞了 data/physics 的權衡平衡。
- **工程結論**：現行 B3（1024 uniform collocation + GradNorm + AL + SOAP）下，均勻隨機採樣已對
  underdetermined 場提供平衡覆蓋；RAR 的高殘差集中反而**移除 bulk 覆蓋**，淨有害。
  **保留 `physics_collocation_strategy="random"`**；RAR 寫入 ablation 作 NEGATIVE_RESULT，
  支持 baseline 設計選擇。eval artifact: `artifacts/eval_272/`（lab-server）。

## 9.4 「sub-DNS divergence」是 band-limiting 假象 — **claim 降級**（2026-05-29）

> **觸發**: 使用者質疑「重建場散度怎麼可能比 DNS 還低」。執行對照實驗 [`scripts/divergence_smoothness_control.py`](../scripts/divergence_smoothness_control.py) 驗證。

**先確認不是計算 bug**: evaluator [`evaluate_deeponet_cfc.py:1112-1113`](../scripts/evaluate_deeponet_cfc.py:1112) 對 pred 與 DNS **用同一個 `divergence_fd`(2 階中心差分) + 同網格(128² block-avg) + 同分母(DNS strain-rate Frob 8.898)**。報告的 div_ratio 兩邊皆 FD，autograd 只用於 training loss。比較公平，0.39% < 1.04% 數字正確。

**Control 結果**（DNS 譜空間 isotropic 低通後，用同款 FD 算散度比）:

| 場 | div_ratio (mean over t) | 說明 |
|---|---|---|
| DNS full (block-avg 128, k≤64) | **1.037 %** | 重現 evaluator 的 1.04% floor（驗證 replication）|
| DNS 低通 **k≤5** | **0.069 %** | 真正 k≤5 的場散度 |
| DNS 低通 k≤8 | 0.151 % | |
| DNS 低通 **k≤16** | **0.376 %** | |
| **PI-CON 實測 (EXP-245/271)** | **0.36–0.39 %** | ≈ k≤16 band-limited DNS |

**結論**:
1. **FD 散度截斷誤差 ∝ 場的高波數含量**（中心差分截斷誤差 $O(\Delta x^2 \partial^3 u)$）。DNS 的 1.04% 全是「完整 cascade(k→64)被 FD 微分」的截斷誤差；DNS 在譜空間守恆到 ~1e-13。
2. **PI-CON 的 0.39% ≈ k≤16 band-limited DNS 的 0.376%**，而非接近 k≤5（0.069%）。代表 PI-CON 有效頻段到 ~k=16（與 spectrum floor 一致），其低散度**主要來自「場比 DNS 平滑(缺 k=16~64 高頻)」**，不是「比 DNS 更守恆」。
3. **"sub-DNS divergence" framing 必須降級**：不能當 contribution / headline；任何 CFD 審稿人一眼識破「band-limited 場 FD 散度天生低」。
4. **可保留的誠實 claim**：AL-continuity 把重建散度壓到「其 resolved bandwidth(k≲16)的 FD 截斷 floor」(0.39% ≈ 0.376%)，證明約束 active；但這是 secondary diagnostic，不是「優於 DNS 的不可壓縮性」。

**論文影響（2026-05-29 降級已執行）**: chapter03 floor 定義加 band-limiting 說明；chapter04 §sub_dns_div 重寫為 diagnostic（移除 "X× below floor"）；chapter05 contribution/implication 移除 sub-DNS 條目；C/Eng abstract 移除 sub-DNS 句；thesis/CLAUDE.md 主訊息 contribution #2 改寫。

## 9.5 訓練長度：20k 是早停，40k 真實改善且非 overfit — **POSITIVE_FINDING**（EXP-273, 2026-05-31）

> **觸發**: 分析 EXP-245_a metrics.jsonl 後段斜率，發現 20k 尚未飽和（最後 5k 步 l_data −20.6%、l_physics −32.6%）。EXP-273 = EXP-245_a byte-identical config，唯一改 `iterations` 20k→40k（lr_decay_steps 維持 2000 + ScheduleFree → 前 20k 軌跡完全相同，乾淨延伸；step 20000 l_data 3.229e-3 ≈ EXP-245 的 3.233e-3 驗證吻合）。job 3746, acmt20, 5h28m COMPLETED。

**結果**（同 evaluator, time-mean, seed=42 paired comparison）:

| 指標 | EXP-245_a (20k) | EXP-273 (40k) | Δ |
|---|---|---|---|
| **KE rel-err (all)** | **5.90 %** | **4.95 %** | **−0.95 pp（−16% 相對）** |
| KE rel-err (train) | 5.74 % | 4.84 % | −0.90 pp |
| KE rel-err (val) | 6.53 % | 5.39 % | −1.14 pp |
| u rel-L2 | 13.59 % | 13.10 % | −0.5 pp |
| v rel-L2 | 17.53 % | 16.63 % | −0.9 pp |
| Ens rel-err | 24.41 % | 21.44 % | −3.0 pp |
| div ratio (pred) | 0.385 % | 0.32 % | 更好 |
| E(k_f=2) ratio @last | 0.969 | 0.976 | 更接近 1 |

**判讀**:
1. **真效果非 seed 雜訊**：paired（seed=42，唯一差 iterations），−0.95 pp ≫ training σ=0.11 pp；4.95% 亦贏 multi-seed mean 5.71%。
2. **非 sensor overfit**（關鍵反向假設已排除）：val KE 也降（6.53→5.39%，幅度 > train），train/val gap 縮小（0.79→0.55 pp）。overfit 應 val↑/gap↑，觀察到相反 → 真泛化。
3. **物理同步改善**：l_physics 後半減半、div 0.385→0.32%、E(k_f) 0.969→0.976 → 往「更滿足 NS」收斂，非擬合 sensor。

**結論**: 20k baseline 是早停，PI-CON 被低估。`iterations` 是唯一實測有效的訓練超參（cf. §9.3 RAR NEGATIVE）。

[RISK: 成本] 2× 算力（5.5hr vs 2.7hr）換 −0.95 pp；K-scaling 仍是更大 lever（K=200→2.47%）。是否把 baseline 升級為 40k 需 multi-seed n=5 確立 σ 後再決定（見 §13）。eval artifact: `artifacts/eval_273/`（lab-server）。

---

# §10 Summary Tables

## 10.1 Architectural Ablation 結論摘要（B0/B1/B2/B3 + PINN）

| Component | B0 | B1 | B2 | B3 (Ours) |
|---|---|---|---|---|
| CfC time encoding | ✗ | ✓ | ✗ | ✓ |
| Cross-attention | ✗ | ✗ | ✓ | ✓ |
| KE rel-err (64 collo, n=5 / n=1) | 18.52 ± 0.66 % | 14.65 % | 13.62 % | **10.77 ± 0.52 %** |
| KE rel-err (1024 collo, LES_T50, seed=42) | 9.96 % | 10.62 % | 8.43 % | **6.92 % (10k) / 5.71 ± 0.11 % (20k n=5)** |

- **B3 vs B0 stat sig (64 collo)**: Cohen d = 13.09, p < 1e-7
- **CfC contribution** (64 collo): -3.87 pp
- **Cross-attn contribution** (64 collo): -4.90 pp
- **Operator framework >> Standard PINN**: B0 - PINN = -20.0 ~ -21.3 pp (64 collo)
- **1024 collo + LES_T50 重新洗牌**: cross-attention 比 CfC 更關鍵（B0 反超 B1）

## 10.2 Sensor Placement 結論摘要（K=100 sparse regime）

修完 axis bug 後（CLAUDE.md KNOWN_PITFALLS / 2026-05-18），**僅列工程可遷移 + statistically-converged 的 LES placement**:

| Placement | KE rel-err (64 collo seed=2) | 工程可遷移性 | 解讀 |
|---|---|---|---|
| DNS QR-pivot (oracle) | **9.40 %** | 無（需 DNS）| 上限參考（理論上 omniscient）|
| LES_N256 **T=50 stat-conv, random IC** | 12.36 % | **強**（real-world DNS-free）| 26.5 turnovers 完全脫離 DNS 影響；**論文 engineering pivot 主代表** |
| LES_N128 Bardina over-disp stand-alone | 12.40 % | 強 | N=DNS/2 + α=30 過耗散 + spin-up 充足；**low-fidelity LES viable** |
| Random uniform | 13.25 % | 強（無需 LES）| placement-agnostic baseline |

**Placement variance** (EXP-266 n=5, training seed=42 固定):
- Random K=100: KE 7.95 ± 0.68 %
- LES_T50 K=100: KE 5.71 ± 0.11 % (EXP-245 同 n=5)
- σ_placement / σ_training = **6.2×** — placement variance dominant

## 10.3 Sensor Amount (K-scaling) 結論摘要

| K | KE rel-err (20k single seed) | 角色 |
|---|---|---|
| 100 | 5.90 % (single) / 5.71 ± 0.11 % (n=5 baseline) | 工程主線 |
| 200 | **2.47 %** (EXP-269) | K-scaling 中段 |
| 400 | **1.76 %** (EXP-270) | K-scaling 最佳 |

- Spectrum cut-off k_max ∝ √K **嚴謹 paper claim**
- Scalar KE 由 d/δ_ω reconstruction quality 主導，不繼承 √K

## 10.4 Sensor Noise 結論摘要

| Noise σ | KE rel-err | Δ vs clean |
|---|---|---|
| 0 % (clean 10k) | 6.92 % | — |
| 1 % | 6.89 % | -0.03 pp |
| 3 % | 6.84 % | -0.08 pp |
| 5 % | 7.07 % | +0.15 pp |
| 10 % | 7.14 % | +0.22 pp |

- **PI-CON robust to 10 % sensor noise** (KE +0.22 pp absolute)
- 1–5 % noise 對 k_f amp 有 implicit regularization 效果

## 10.5 vs Classical Interpolation 結論摘要

| Method | KE % | u L2 % | ω L2 % |
|---|---|---|---|
| Ours (EXP-080 legacy single seed) | 10.68 | **17.0** | **47.6** |
| Ours (EXP-245 20k n=5) | **5.71 ± 0.11** | **13.65 ± 0.06** | 41.79 ± 0.12 |
| RBF Multiquadric (best classical KE) | 4.10 | 32.84 | 58.38 |
| Div-free trig LSQ k ≤ 5 (math KE optimum) | 3.93 | 28.19 | 64.78 |

- Pareto trade: classical 低 KE 但 over-smoothing (pointwise 差 11-14 pp)
- KE-as-misleading-metric → multi-metric evaluation 為 paper recommendation

---

# §11 Rejected / Invalid Directions

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
11. **NS 加 AL**（§9.1 EXP-242/243）: NS momentum 不適合 AL pressure，cont AL 是 sweet spot
12. **Forcing identification from sensor MSE + PDE residual alone**（§9.2 EXP-252~255）: ill-posed; 需要 explicit forcing-mode supervision

---

# §12 Legacy ↔ Stable ID 雙向對照

## 由 stable ID 查 legacy

| Stable ID | Legacy ID | Seed | 角色 |
|---|---|---|---|
| `EXP-200_a` | `EXP-080` | 42 | B3 multi-seed #1（時間最早，AL Pareto sweet spot 首次定錨；同 EXP-080 squeeze report subject）|
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
| `EXP-240_a` | — (new 2026-05-19) | 42 | B0 + LES_T50 (2×3 ablation) |
| `EXP-240_b` | — (new 2026-05-19) | 42 | B0 + Random (2×3 ablation) |
| `EXP-241_a` | — (new 2026-05-19) | 42 | Collo density 256 |
| `EXP-241_b` | — (new 2026-05-19) | 42 | Collo density 1024 (DNS oracle) |
| `EXP-242_a/b/c` | — (new 2026-05-20) | 42 | Multi-AL ablation (cont AL / NS+cont AL / 純 cont AL) |
| `EXP-243` | — (new 2026-05-20) | 42 | Full physics 純 AL, no GN |
| `EXP-244` | — (new 2026-05-20) | 42 | B3 + 4-head + DNS sensor (oracle reference) |
| `EXP-245` _a~e_ | — (new 2026-05-20, upgraded 2026-05-21) | 42/1/2/3/4 | **🥇 ACTIVE_BASELINE** (B3 + LES_T50 + 20k n=5) |
| `EXP-246~250` | — (new 2026-05-20) | 42 | Architecture × LES_T50 sensor sweep (B0/B1/B2/PINN-SiLU/PINN-tanh) |
| `EXP-251` | — (new 2026-05-20) | 42 | B3 + 4-head + LES_T50 |
| `EXP-252` | — (≡ `EXP-245` 10k) | 42 | Forcing hardcoded reference |
| `EXP-253` | — (new 2026-05-20) | 42 | Forcing: learn k_f only, A fixed; zero-ish init |
| `EXP-254` | — (new 2026-05-20) | 42 | Forcing: learn A only, k_f fixed; zero-ish init |
| `EXP-255` | — (new 2026-05-20) | 42 | Forcing: learn both A + k_f; zero-ish init |
| `EXP-256` | — (new 2026-05-20, redefined 2026-05-21) | 42 | **K=200 LES sensor, 10k**（HISTORICAL; 未 plateau）|
| `EXP-257` | — (new 2026-05-21) | 42 | **K=400 LES sensor, 10k, collo=512**（HISTORICAL; 未 plateau）|
| `EXP-258` | — (new 2026-05-21) | 42 | Sensor noise robustness 1 % |
| `EXP-259` | — (new 2026-05-21) | 42 | Sensor noise robustness 3 % |
| `EXP-260` | — (new 2026-05-21) | 42 | Sensor noise robustness 5 % |
| `EXP-261` | — (new 2026-05-21) | 42 | Sensor noise robustness 10 % |
| `EXP-262` | — (new 2026-05-21) | 42 | **Re=10⁶ baseline**（DNS jaxpi + LES T=5）|
| `EXP-264` | — (new 2026-05-22) | 42 | Re=10⁶ Path 2: d_model=384 + 50k iter |
| `EXP-265` | — (new 2026-05-22) | 42 | **Re=10⁶ Path A**: K=200 + d=384 + 50k (KE 11.39 %) |
| `EXP-266_a~e` | — (new 2026-05-22) | training=42 / placement=42/1/2/3/4 | Random K=100 placement variance n=5 |
| `EXP-267` | — (new 2026-05-23) | 42 | Re=10⁶ K=100 LES T=50 ablation (KE 14.58 %) |
| `EXP-268` | — (new 2026-05-23) | 42 | **🥇 Re=10⁶ K=200 LES T=50 + d=384 + 50k (KE 6.10 %)** |
| `EXP-269` | — (new 2026-05-24) | 42 | K-scaling K=200 LES sensor, 20k (KE 2.47 %) |
| `EXP-270` | — (new 2026-05-24) | 42 | K-scaling K=400 LES sensor, 20k, collo=512 (KE 1.76 %) |
| ~~`EXP-223`~~ | ~~`EXP-106`~~ | — | **移除（2026-05-19）**: T=30 dns-init 工程不可遷移且效果不如 T=50 |
| ~~`EXP-225`~~ | ~~`EXP-103 v2`~~ | — | **移除（2026-05-19）**: T=5 < 1 turnover，非 stat-converged LES |

## 由 legacy 查 stable

| Legacy ID | Stable ID | 註 |
|---|---|---|
| `EXP-030` | `EXP-230` | Re=1000 baseline |
| `EXP-080` | `EXP-200_a` | B3 seed=42（first AL ρ=0.1 sweet spot run；同 squeeze report subject model）|
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
| `EXP-103 v2` | — (移除) | T=5 非 stat-converged，**不納入 stable phase** |
| `EXP-105 v2` | `EXP-221` | LES_N256 T=50 stat-conv, random IC |
| `EXP-106` | — (移除) | T=30 dns-init 工程不可遷移且效果不如 T=50 |

> `EXP-101/102/103/105` v1（axis bug 受害版本）**不重新編號**，永遠以 legacy ID + "v1 buggy" 標籤存在於 archive，避免污染 stable phase。

---

# §13 Open Questions / Pending TODO

| 問題 | 現況 | 狀態 |
|---|---|---|
| **EXP-245 multi-seed (n=5, 20k)** | **已完成** (5.71 ± 0.11 %, σ=0.11 pp 統計顯著確立) | ✅ 2026-05-21 |
| **EXP-241_b multi-seed (n=3-5)** | DNS 對照 single seed=42, KE 5.97 % — paper-grade needs std confirmation | **NEW 高優先** |
| **EXP-241_c collocation = 4096?** | EXP-241_a (256) → EXP-241_b (1024) 還沒 saturated（5.97 vs 6.88, -0.9pp 下行）；4096 可能再降但 OOM risk | 待開工（需 split-batch fallback）|
| RTX 3090 paper-grade inference benchmark | EXP-094 M3 baseline 71+1.5 ms 為唯一參考；新 hw 待測 | 待 `benchmark_inference.py` 重跑 |
| Re=1000 stable phase multi-seed（n=5）| 尚未跑；目前只有 legacy EXP-030 single seed | 待開工 |
| EXP-266 Random K=100 placement variance (n=5) | **已完成** (KE 7.95 ± 0.68 %, σ_placement / σ_training = 6.2×) | ✅ 2026-05-22 |
| `EXP-220` (LES_T50) 5-seed placement variance (LES 5 seeds × home-gpu CPU 50+ hr) | 仍 single seed=2; LES-derived placement σ 未估計 | 待開工（paper §Sensor Placement extension）|
| LES robustness across LES_seed | 目前 LES generator 用 seed=42 single placement; 跨 LES seed 的 sensor variability 未測 | 待開工 |
| EXP-242 group: multi-constraint AL | **已完成** (EXP-242a/b/c + EXP-243 全 rsync 回, metrics 完整) | ✅ 2026-05-20 |
| EXP-252~255 forcing-prior identifiability | **已完成** (rerun zero-ish init 全 finalize；finding: identifiability ill-posed two-sided verified) | ✅ 2026-05-21 |
| EXP-258~261 sensor noise robustness | **已完成** (1/3/5/10 % noise, KE -0.08~+0.22 pp; PI-CON 高度 robust) | ✅ 2026-05-21 |
| Sensor noise multi-seed n ≥ 3 | 確認 noise scaling 是否 linear | 待開工 |
| EXP-262 Re=10⁶ baseline | **已完成 (single seed)** | ✅ 2026-05-21 |
| EXP-264/265 Re=10⁶ Path 2/A | **已完成** | ✅ 2026-05-22 |
| EXP-267/268 Re=10⁶ LES T=50 | **已完成** (EXP-268 KE 6.10 % ≈ Re=10⁴ baseline) | ✅ 2026-05-23 |
| EXP-268 multi-seed n=3 (Re=10⁶ KE σ 估計) | single seed only; n=3 × 6 hr = +18 hr | 待開工 (paper §Cross-Re σ extension) |
| EXP-269/270 K=200/400 sweep 20k | **已完成** (KE 2.47 / 1.76 %) | ✅ 2026-05-24 |
| EXP-269/270 multi-seed | single seed only | 待開工 |
| Grid independence (Re=10⁴) | **已完成** (PASS, N=256 ref converged at machine ε) | ✅ 2026-05-24 |
| **EXP-273 training length 20k→40k (single seed)** | **已完成** (KE 5.90→4.95%, −0.95pp, 非 overfit；20k 是早停, 見 §9.5) | ✅ 2026-05-31 |
| **EXP-273 40k multi-seed n=5** | single seed only; 需 σ 才能把新 baseline (40k) 寫進 paper | 待開工（高優先, paper baseline 升級用）|
| RAR collocation ablation (EXP-272) | **已完成** (NEGATIVE: KE 5.90→10.28%, 見 §9.3) | ✅ 2026-05-30 |
| Classical interpolation re-benchmark vs EXP-245 20k | 目前 squeeze report 對比 EXP-080 legacy 10.68%；新 baseline 5.71% 下 Pareto trade 重算 | 待開工（paper polish 用）|
| Cylinder stable phase 整併 | Cylinder 仍用 CEXP-XXX；是否要納入此 v2 system？| 開放討論（傾向維持獨立 v2）|
| CfC Jacobian spectral radius stability | 未寫腳本 | 待開工（CFD-rigour）|

---

# §14 HOW-TO 新增 stable phase 實驗

1. 確認新實驗屬於哪個 group（200~239 已分配；新研究方向用 240+）
2. 在 `configs/stable/` 建立新 config（檔名 `exp_NNN.toml` 或 `exp_NNN_X.toml`，X 為 multi-seed suffix）
3. 訓練時 artifact dir 命名: `artifacts/kolmogorov/stable/exp_NNN[_X]_{描述}`
4. 訓練完成後：
   - 此檔對應的 §3~§7 對照群（或 §9 diagnostics）新增 row + finding 段
   - §12 legacy 對照表補入新 ID
   - §13 Open Question 表標 ✅ completed（若有對應 entry）
5. 若實驗 supersedes 既有結論，更新 §1 主 baseline / §11 Rejected
6. 若篇幅 > 50 lines，另開 `docs/experiment_archive_stable_phase.md`

---

# §15 變更紀錄

- **2026-05-27 (narrative 重組)**:
  - 把 v2 從「按時間/編號群組」改為「按主線敘事」結構：§1 主線 / §2 延伸 / §3-§7 五條對照 / §8 inference / §9 diagnostics
  - 新增 §7 **vs Classical Interpolation**：從 `docs/archive/squeeze_report_2026-05-11.md` 整合 RBF×3 / IDW / div-free trig LSQ + SVD null-space + Pareto trade
  - 把 EXP-242/243 (multi-AL anti-pattern) 與 EXP-252~255 (forcing identifiability) 統一歸 §9 Diagnostics
  - 把 EXP-241 (collocation density) 整合進 §1.2 主線證據鏈（之前是獨立 group）
  - 新增 §10 Summary Tables（5 對照群各一表）
  - §13 Pending TODO 表清理；§14 HOW-TO 對應新結構更新
  - 備份：`docs/experiment_log_v2.md.bak`
- **2026-05-23 (Cross-Re ablation ladder 完結, Re=10⁶ ≈ Re=10⁴ baseline)**:
  - LES T=50 Re=10⁶ home-gpu 7.18 hr 完成 (50 T_L)
  - EXP-267 (Re=10⁶ K=100 LES T=50 ablation): KE 23.73 → **14.58 %** (−9.15 pp, LES quality lever > capacity lever)
  - EXP-268 (Re=10⁶ K=200 LES T=50 + d=384 + 50k): KE **6.10 %** ⭐ ≈ Re=10⁴ baseline 5.71 ± 0.11 %
  - **Paper milestone**: "PI-CON generalizes across Re=10⁴ → Re=10⁶"
- **2026-05-22 v3 (Placement variance + Re=10⁶ Path A 結案)**:
  - EXP-266_a~e (Random K=100 × 5 placement seeds): σ_placement / σ_training = **6.2×**, LES_T50 vs Random gap 2.24 pp z≈3.3
  - EXP-265 (Re=10⁶ K=200 + XL + 50k): KE 11.39 % ✓ engineering viable
  - EXP-264 (Path 2 capacity+step): KE 19.02 %, k_f amp 退步 0.746
- **2026-05-22 v2 (K-scaling two-layer framing 再修正)**:
  - 區分 **Layer 1 (spectrum cut-off ∝ √K, ✅ 嚴謹)** vs **Layer 2 (scalar KE, ❌ 不繼承 √K)**
  - Re=10⁴ KE 三點「1/√K 看似 fit」= d/δ_ω 50% 改善的 numerical coincidence
- **2026-05-21 (EXP-245 baseline 升級 10k → 20k n=5, KE 5.71 ± 0.11 %)**:
  - EXP-245 升級 iterations 10k → 20k, time_marching_warmup_steps 改 fixed 2000
  - 5 seeds 全跑完，KE = 5.71 ± 0.11 %
  - **三個 metric 出現質變**：div ratio 2.41 % → 0.39 % (< DNS floor 1.04 %); k_f amp 0.926 → 0.991; Ens 27.51 → 24.11 %
- **2026-05-21 (K-scaling + noise + Re=10⁶ baseline)**: EXP-257/258/259/260/261/262 完成
- **2026-05-20 (artifacts rsync + forcing + K-scaling K=200 finalize)**:
  - 從 lab-server rsync 補完 EXP-242a/b/c/243/244/245/246/247/248/249/250/251 + EXP-252~256 完整 metrics
  - EXP-253/254/255 zero-ish init rerun + EXP-256 (K=200 LES sensor) finalize
- **2026-05-19**: v2 啟用。從 legacy EXP-001~106 完整提取 stable phase 主線（B3/B0 multi-seed, B1/B2/PINN ablation, sensor placement series, Re=1000 baseline），以 EXP-200 起編號。Multi-seed 統一 `_a~_e` suffix。Legacy IDs 與其 archive 不動。

---

## EXP-274 — AL delayed-start + Phase2 L-BFGS finetune（訓練策略探索）

**日期**: 2026-05-31 ｜ **狀態**: ✅ 已評估（job 3750, 4h49m）— **neutral result：與 baseline 統計不可區分，不採用（無可測量增益）**
**Config**: `configs/exp_274_al_delay_lbfgs.toml`（派生 EXP-271, DNS QR-pivot oracle, seed=42）

**Why**: 探索兩個訓練策略：(1) AL dual update 延後到 step≥10000、freq 100→500（早期 λ 凍結在 0，
僅留 ρ 二次罰，讓 data/NS 先收斂，後期才用 λ 線性項收緊 continuity）；(2) 主 phase (SOAP 20k)
後同進程切 L-BFGS 在 eval-mode(y_t) finetune 5000 步（max_iter=20, λ 凍結），用二階收斂壓低 residual。

**程式變更（向後相容，預設停用）**:
- `config.py`: 新增 `al_start_step`(預設 0) + `lbfgs_finetune_steps`(預設 0)
- `training.py`: 兩處 AL dual update 加 `step >= al_start_step` gate；main loop 後新增 Phase2
  L-BFGS finetune block（eval-mode、GradNorm 凍結、λ 凍結、loss 組法同 non-gradnorm AL path）
- `tests/test_al_delay_lbfgs_finetune.py`: 4 passed（gate 凍結/解凍、phase 接續、λ 凍結、向後相容）

**變更（vs EXP-271）**: `al_update_freq` 100→500；`al_start_step` 0→10000；`lbfgs_finetune_steps` 0→5000

**訓練軌跡（從 metrics.jsonl，已驗證為真實數據）**:

| step | l_data | l_cont | λ_cont | 備註 |
|---|---|---|---|---|
| 1 | 2.53e+0 | 1.88e-1 | 0.000 | λ 凍結 |
| 5000 | 8.12e-3 | 1.99e-2 | 0.000 | λ 凍結（< al_start_step） |
| 9500 | 4.51e-3 | 7.56e-3 | 0.000 | λ 凍結 |
| 10000 | 9.55e-3 | 6.97e-3 | 0.0007 | **dual update 開啟** |
| 15000 | 2.17e-3 | 3.40e-3 | 0.0057 | λ 累積中 |
| 20000 | 3.34e-3 | 2.56e-3 | 0.0085 | phase1 結束 |
| 20001 (ft) | 5.94e-4 | 2.52e-3 | 0.0085 | phase2 起，λ 凍結 |
| 25000 (ft) | 2.77e-3 | 2.69e-3 | 0.0085 | phase2 結束 |

**控制流驗證 ✅（皆為真實 log 數據）**:
- λ 在 step 1–9999 嚴格凍結為 0（僅 ρ=0.1 二次罰生效）；step 10000 起 dual update 開啟，freq 500
- 注意 λ 終值僅 **0.0085**（遠未達 clip=10）— EMA momentum 0.5 + freq 500 稀疏更新下累積緩慢
- phase2 (20001–25000) λ 凍結在 0.0085；L-BFGS 全程 l_data/l_cont 在同量級震盪，**無單調下降**（curvature history 因每步重採樣 collocation 而失效，與 SOAP+RAR 失效機制相同）

**DNS 評估（2026-05-31, `evaluate_deeponet_cfc.py`, 兩者皆 seed=42 final.pt — 真實同seed對照）**:

| 指標 | EXP-271 baseline | EXP-274 | Δ |
|---|---|---|---|
| KE rel-err (all) | 4.682% | **4.571%** | −0.11pp（noise 內）|
| u rel-L2 (all) | 15.34% | 15.08% | −0.26pp |
| v rel-L2 (all) | 17.90% | 17.75% | −0.15pp |
| ω rel-L2 (all) | 41.41% | 41.33% | −0.08pp |
| enstrophy rel-err | 22.42% | 22.32% | −0.10pp |
| div ratio (pred) | 0.66% | 0.69% | +0.03pp（皆 < DNS floor 1.04%）|
| k_f amplitude ratio | 0.9944 | 0.9945 | 持平 |
| E(k_f) ratio | 0.9931 | 0.9931 | 持平 |

> 兩欄皆 single seed=42、DNS QR-pivot oracle、20k，取自各自 `eval/summary.json`（evaluator 真實輸出）。
> EXP-271 n=5 mean = 4.68 ± 0.06%；EXP-274 的 4.571% 落在此帶內，單 seed 差異不可宣稱顯著。

**結論（neutral result，假設未被支持）**:
1. **兩個策略無可測量增益**：EXP-274 在 KE/u/v/ω/Ens 全部微好 0.08–0.26pp，但全落在單 seed noise
   （n=5 σ≈0.06pp）內 → 與 baseline **統計不可區分**。div 微升 0.03pp，無實質意義（皆 < DNS floor）。
2. **AL delayed-start 原假設未被支持**：預期「早期 λ=0 → 模型卡 high-div basin → div 惡化」未發生
   （div 0.66→0.69%，幾乎不變）。根因：ρ=0.1 二次罰在早期已足夠約束 continuity；且 λ 終值僅 0.0085
   （freq 500 + EMA 0.5 累積過慢，遠未達 clip=10）→ AL 線性項本來就幾乎沒發揮，延不延遲都無感。
3. **Phase2 L-BFGS 無效**：5000 步 training loss 同量級震盪無單調下降，DNS 指標無改善。
   curvature history 因每步重採樣 collocation 失效（同 SOAP+RAR 失效機制），且 ScheduleFree y_t 已近最優。
4. **處置：不採用**。+5k 步（≈+25% wall）+ 流程複雜度換不到可測量指標，違反 Simplicity。
   `al_start_step` / `lbfgs_finetune_steps` 預設 0（停用）保留為可選旋鈕，現有實驗不受影響。

**控制流驗證 ✅（真實 log）**: λ step 1–9999 嚴格凍結 0；step 10000 起開啟 dual update；phase2 λ 凍結 0.0085。實作與設計一致。

---

## EXP-275 — L-BFGS fixed-batch 診斷（驗證 EXP-274 phase2 失效機制）

**日期**: 2026-05-31 ｜ **狀態**: ✅ 根因確認 + 已修（bug）— **phase2 L-BFGS lr 誤用 1e-3，深收斂點零更新致 loss 凍結**
**Config**: `configs/exp_275_lbfgs_fixed_batch.toml`

### 🐛 根因（systematic-debugging 確認，2026-05-31）

**症狀（已驗證, metrics.jsonl 實讀）**: job 3766 phase2 l_data 連續 2186 步位元級不變（`5.938034e-04`）。

**根因（已確認）**: phase2 的 `torch.optim.LBFGS` 在 [training.py:1677] 用 `lr=learning_rate=1e-3`
（SOAP 一階法 LR）。但 **L-BFGS+strong_wolfe 是 Newton step，標準 lr=1.0**（line search 自決步長）。
lr=1e-3 把步長縮 1000×，進入深收斂區（梯度小）後 L-BFGS step 退化成**零權重更新** → loss 凍結。

**決定性證據（lab-server GPU 真實 scale, d_model=256, CUDA, job 3790）**:

| 設定 | phase2 l_data | param_change/step | 相異值 |
|---|---|---|---|
| `lr=1e-3`（原 bug）| 1.35e-2 → 1.98e-4 | 末兩步 **4e-06 → 0.0**（凍結 onset）| 7/8 |
| `lr=1.0`（修法）| 4.65e-5 → **4.34e-7**（低 450×）| [81,61,40,31,23,18,14,13] 全大 | 8/8 |

→ lr=1e-3 在梯度變小時 param_change 歸零（凍結）；lr=1.0 全程大幅更新且 loss 降更深。
job 3766 跑滿 20000 步（l_data 已到 5.9e-4，比此處 3000 步更收斂 68×）→ 從 phase2 第一步就**永久**凍結。
本地 smoke 不凍結因 d_model=32 短訓練，phase2 起點梯度大（~0.4–1.0），1e-3 步仍足以動。

**先前錯誤假說的更正**: 「float32 精度飢餓」framing **不準** — param_change 是**精確 0.0**（L-BFGS line
search 回傳零步），非浮點微小累積。「已達最優」也已被 PROBE 排除（凍結點 grad_max=0.47，GD 能降 loss）。

### 修法（已套用 + 測試）

- `config.py`: 新增 `lbfgs_finetune_lr`（預設 **1.0**）
- `training.py:1677`: phase2 L-BFGS 改用 `lbfgs_finetune_lr` 而非 `learning_rate`
- `tests/`: 新增 2 個回歸測試（lr 預設=1.0 解耦於 learning_rate / 可 config 覆寫）— **修法前 RED 修法後 GREEN**，7 passed
- 回歸：既有 lbfgs/al/smoke 19 passed

**對先前結論的影響（重大更正）**: EXP-274（lr 同樣=1e-3 + 每步換 batch）與 EXP-275 的 phase2
**兩次都是 no-op**（權重未更新）。先前所有「phase2 neutral / L-BFGS 對此問題無增益」的結論**前提錯誤**，
作廢。**「L-BFGS finetune 是否真能改善 DNS 重建」這個原始問題，修法後尚未測試** → 需重跑 EXP-275（lr=1.0）才有答案。

**下一步（待使用者決定）**: 用修法後程式碼重跑 EXP-275（phase2 lr=1.0），首次真正測試 L-BFGS finetune 效果。

---

#### （以下為 bug 發現前的原始診斷假設，已被上方根因取代，保留供追溯）

**診斷假設**: EXP-274 phase2 L-BFGS 無增益的根因 = **每 outer step 重採樣**（freq=1）使
L-BFGS curvature history `(s_k, y_k)` 跨不同 batch 累積 → Hessian 近似失效（同 SOAP+RAR
freq≥1000 才穩的機制）。phase2 逐步 l_data 軌跡證實：5000 步在 5.94e-4~1.08e-2 鋸齒震盪，
前1000步均 2.75e-3 ≈ 後1000步均 2.74e-3，**完全無下降**。

**程式變更（向後相容，預設 false）**:
- `config.py`: 新增 `lbfgs_finetune_fixed_batch`(預設 False)
- `training.py`: phase2 採樣改快取式 — fixed_batch=true 時只在第一個 outer step 採一次後跨步重用
- `tests/`: 新增 `test_lbfgs_finetune_fixed_batch_runs`（5 passed）

**唯一變因（vs EXP-274）**: `lbfgs_finetune_fixed_batch` False→True（steps/max_iter/history/AL 全同）

**Falsifiability（三分支判讀）**:
- (a) l_data 仍鋸齒不降 → curvature 假設錯，失效另有原因（y_t 已最優 / 無可榨空間）
- (b) l_data 單調降但 DNS KE/div 退步 → 過擬合固定 collocation（救優化傷泛化）→ phase2 路線不可行
- (c) l_data 單調降且 DNS 指標改善 → fixed batch 為正確修法，EXP-274 確為設計錯配
