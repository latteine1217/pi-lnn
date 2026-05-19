# 實驗紀錄（Legacy State 主檔，精簡版）

> **⚠️ 2026-05-19 起：研究進入 stable phase，**主要 state 入口已轉至 [`docs/experiment_log_v2.md`](experiment_log_v2.md)**（含 EXP-200+ 重編號 + legacy 對照表）。
>
> 本檔（v1）保留為 **legacy state** 供 EXP-001~106 的結論層查詢與雙向追溯。任何新 stable phase 實驗變更、評估、對照前，應**優先讀 v2**。

本文件是本 repo 的 **legacy 實驗 state 入口**，不再放完整 RECORD。歷史與詳細紀錄已拆檔，依需求按下方 [Read Order](#read-order) 載入。

主要用途：

- 快速回答 legacy 主線是什麼
- 判斷哪些方向已被支持、證偽或取代
- 讓 agent 在續跑或比較前先自讀，不靠記憶腦補
- 提供 stable phase（v2）「往回查」的窗口

---

## [STATE] 拆檔導引（Read Order）

> 2026-05-15 主檔再次拆分：把 `[STATE] Supported Decisions` 全部詳細條目、`[DIAGNOSTIC]` 報告、Cylinder 主線、與所有 `[RECORD]` 條目搬到專屬檔，主檔僅保留 `[STATE]/[INDEX]` 結論層。
>
> 2026-05-19 stable phase 啟用：主要 state 入口轉至 [`docs/experiment_log_v2.md`](experiment_log_v2.md)；本檔降為 legacy（EXP-001~106）查詢介面。

| 檔 | 內容 | 何時讀 |
|---|---|---|
| **[`docs/experiment_log_v2.md`](experiment_log_v2.md)** | **Stable phase 主檔**（EXP-200+ 重編號 + multi-seed `_a~_e` + legacy 雙向對照表）| **stable phase 任何實驗變更前先讀** |
| **本檔** `docs/experiment_log.md` | Legacy STATE/INDEX 結論層、Open Question、Rejected（EXP-001~106）| stable phase 結論不足、需 legacy 追溯時 |
| [`docs/experiment_archive_kolmogorov.md`](experiment_archive_kolmogorov.md) | **EXP-001~063** 詳細 RECORD（Re=1000 主線 + Re=10000 早中期）| 早期實驗追溯 |
| [`docs/experiment_archive_kolmogorov_post_k100.md`](experiment_archive_kolmogorov_post_k100.md) | **EXP-064~101** 詳細 RECORD + GROUP（K=100 結案後 AL/pivot/multi-seed/benchmark）| 近期實驗判讀 |
| [`docs/cylinder_log.md`](cylinder_log.md) | Cylinder Wake 主線（CEXP-001/002 + BC loss + NaN 診斷）| Cylinder 任務 |
| [`docs/diagnostics_log.md`](diagnostics_log.md) | Physics denorm silent regression + CFD-rigour Q5/Q7/Q8 + Forward CFD baseline | 評估值 / div / ∇p 質疑、CFD-rigour 問答 |
| [`docs/analysis_reports.md`](analysis_reports.md) | Wavelet sparsity + AIM diagnostic（早期）| 資訊論硬上限論述 |
| [`docs/adr/`](adr/) | 設計決策紀錄（ADR-001/002）| 設計權衡追溯 |
| [`docs/paper_framing_draft.md`](paper_framing_draft.md) | 論文 framing v2（engineering pivot, 5-seed stats）| 論文寫作 |

---

## [SCHEMA]

### 欄位定義

- `ID`: 穩定實驗編號，供後續引用
- `Status`:
  - `ACTIVE_BASELINE`: 當前主基準
  - `ACTIVE_REFERENCE`: 仍有效的對照或關鍵依據
  - `NEGATIVE_RESULT`: 已證偽或明確負收益
  - `MIXED_RESULT`: 部分假設成立、部分證偽
  - `ARCHIVED_CONTEXT`: 保留背景脈絡，但已被更新主線取代
  - `PENDING`: 訓練未完成或評估未跑
- `Decision`: 這筆紀錄最後支撐的結論
- `Supersedes / Superseded_By`: 用於追蹤哪條線已被後續結果覆蓋

### 讀取建議

1. 先看 `## [INDEX] Active`
2. 再看 `## [STATE] Current Baseline`
3. 若要判斷某改動是否已被做過，看 `## [INDEX] Negative` 與對應 archive 的 RECORD
4. 若仍不足，依 Read Order 載入對應 archive / log

---

## [STATE] Data Version

### 資料條件

- domain: `[0, 1]^2`
- DNS（Re=1000）:
  [`data/kolmogorov_dns_fp64_etdrk4_Re1000_N128_T5_ds4.npy`](../data/kolmogorov_dns_fp64_etdrk4_Re1000_N128_T5_ds4.npy)
- DNS（Re=10000）:
  [`data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy`](../data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy)
- Re=1000 sensors:
  [`data/kolmogorov_sensors/re1000/sensors_qrpivot_K100_N128_t0-5.json`](../data/kolmogorov_sensors/re1000/sensors_qrpivot_K100_N128_t0-5.json)
- Re=10000 sensors (K=100, QR-pivot, default):
  `data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100.{json,npz}`
- Re=10000 sensors (K=100, random, EXP-101 only):
  `data/kolmogorov_sensors/re10000/sensors_random_K100_N256_t0-5_si100_seed42.{json,npz}`

Cylinder 主線資料設定見 [`docs/cylinder_log.md`](cylinder_log.md)。

---

## [STATE] Current Baseline

### Re=1000 Baseline（EXP-030）

| 項目 | 現況 |
|---|---|
| Baseline ID | `EXP-030` |
| 主線 config | [`configs/exp_030_re1000_soap_sf_5k.toml`](../configs/exp_030_re1000_soap_sf_5k.toml) |
| train artifact | `artifacts/deeponet-cfc-re1000-soap-sf-5000` |
| eval checkpoint | `artifacts/deeponet-cfc-re1000-soap-sf-5000/checkpoints/lnn_kolmogorov_step_5000.pt` |
| 目前判讀 | `SOAP + Schedule-Free` + `5000 steps`（EXP-028 resume）是目前最佳主線；首次突破 KE 10% 門檻 |
| 主要優勢 | KE rel-err **9.61%**（vs EXP-025 SF AdamW: 12.06%，**-20%**）、u RMSE **5.68e-2**（最低）、amp ratio **1.027** |
| 主要已解問題 | t=3.5∼4.5 的 phase 高峰為 Re=1000 chaotic divergence 物理本質，非表徵問題 |

詳見 [`docs/experiment_archive_kolmogorov.md`](experiment_archive_kolmogorov.md) G7。

### Re=10000 Baseline（雙線：EXP-064 KE-optimal / EXP-080 Pareto sweet spot）

| 項目 | EXP-064（KE-optimal）| EXP-080（Pareto sweet spot）|
|---|---|---|
| Baseline ID | `EXP-064` | `EXP-080` |
| 主線 config | `configs/exp_064_re10000_xlarge_sensor_physics.toml` | `configs/exp_079_re10000_al_4task_gradnorm.toml`（ρ=0.1 變體）|
| train artifact | `artifacts/deeponet-cfc-re10000-exp064-sensor-physics` | `artifacts/kolmogorov/deeponet-cfc-re10000-exp080-al-4task-rho01` |
| KE rel-err | **7.80%** | **10.68%** |
| div L2 | 0.184 | **0.067** |
| ek_ratio_last | 0.938 | 0.911 |
| kf_amp ratio | 0.962 | 0.937 |
| 設計目標 | 4-task GradNorm + sensor continuity → best KE | 4-task GradNorm + AL ρ=0.1 → KE-div trade-off sweet spot |
| 結案聲明 | **K=100 結案（2026-04-26）**：中高頻 ≈100% 為 K=100 資訊論硬上限；架構/optimizer 無法突破 | 9-point AL Pareto frontier 中真正 Pareto-optimal point；6-lever pivot ablation 後 near-optimal |

詳見 [`docs/experiment_archive_kolmogorov_post_k100.md`](experiment_archive_kolmogorov_post_k100.md)。

### 主線固定假設（Re=10000 K=100）

- 觀測 supervision 僅使用 `u, v`
- physics 使用 primitive `momentum + continuity`
- 空間編碼：`LearnableFourierEmb`（`embed_dim=128`，init σ=2.0）for Re=10000；`periodic_fourier_encode`（`fourier_harmonics=8`）for Re=1000
- `relpos_bias`：純距離輸入 `|rel|`（等向），不含方向向量
- `output_head_gain = 1`
- `use_temporal_anchor = true`（`n_harmonics=2`）：為 trunk 提供 `sin/cos(2π n t/T)` 絕對時間座標
- `Small` 尺寸（d=64）在 Re=1000 已足夠；Re=10000 需 `XLarge`（d=256）
- `Re=1000/10000` forcing mode 均為 `k_f = 2`
- `time_marching` 應保留
- 優化器主線（Re=10000）：`SOAP + Schedule-Free`（`lr=1e-3`，`betas=(0.9,0.999)`，`precond_freq=2`，`step_decay`，`warmup=2000`）
- `GradNorm`（`update_freq=1000`，`momentum=0.9`）自動均衡 data/physics task 梯度比例
- `use_physics_denormalization = False`（Kolmogorov 預設，與 d62e698 前 byte-aligned）；Cylinder 主動 `= true`，詳見 [`docs/diagnostics_log.md`](diagnostics_log.md)

---

## [STATE] Rejected Directions

1. 把 `omega` 當作 sensor data supervision（EXP-002）。
2. 只靠降載期待自動修復 collapse（EXP-004/005）。
3. 單純延長訓練步數到 `5k`（EXP-009）。
4. `top-k local attention` 作為 decoder 讀 branch token 機制（EXP-013）。
5. 在 `Re=1000` 上使用錯誤 forcing mode `k_f=4`。
6. Physics loss 機制調整（Re=10000）：Chebyshev collocation、residual normalization、壓力 Poisson 約束（weight=0.1~1.0）均無法突破 EXP-031 基準。在 K=100 sparse sensors 的資訊量限制下，physics loss 設計已非主要瓶頸。
7. Transfer learning 需要 source/target 架構完全相同（EXP-040）。EXP-030（d=64）→ Re=10000 Wide-v2（d=128）直接 transfer 因架構不匹配失敗。
8. Transfer learning（EXP-042）在 source 品質不足時產生負遷移：EXP-041 KE=24.5% 作為 source，transfer 後 KE 40.2%，差於隨機初始化。
9. **6-lever pivot ablation 全 falsified**（2026-05-11，EXP-083~087）：ρ ablation、multi-head cross-attention、fourier_harmonics ↑、K-scaling、trunk depth ↑、mMLP gating — 均無法在 K=100 + 當前架構下顯著突破 EXP-080。
10. **AL 與 GradNorm 同時控制 cont（ADR-001 §4 禁令）**：EXP-079 證實禁令過於保守但無害；違反禁令既不破壞訓練也不解決 KE-div trade-off。「兩全其美」不存在於 AL 任何配方。
11. **Resume from checkpoint**（EXP-082 災難）：silent state corruption（ScheduleFree internal step / GradNorm init_loss / RAR timing 等未復原）；強制 1-shot 訓練，禁用 `resume_checkpoint`。

---

## [STATE] K=100 稀疏重建結案（2026-04-26）

**EXP-064 為 K=100 sensor 配置的最終接受結果。稀疏重建主線結案。**

K=100 已達資訊論硬上限：Wavelet 分析顯示 mid (k~8..16) 需 ~588 自由度、high (k~16..32) 需 ~1452，K=100 均欠定；CS 精確重建需 M ≥ O(s log N) ≈ 5000 sensors，K=100 差約 50 倍。

完整數據、頻帶能量佔比、結案判斷見 [`docs/experiment_archive_kolmogorov_post_k100.md`](experiment_archive_kolmogorov_post_k100.md) 的「K=100 稀疏重建結案聲明」section。

進一步提升高頻需要根本性增加感測器覆蓋（K≥5000）或引入 DNS 高頻先驗（工程不可遷移）。

---

## [STATE] Open Question

| 問題 | 現況 | 狀態 |
|---|---|---|
| amplitude ratio=0.9965 是否 overfitting | EXP-015 更高，需 OOD 測試 | 開放（低優先）|
| K=200 band_mid 突破後低頻退步是否可恢復 | EXP-066/EXP-085 均未充分收斂 | **CLOSED**：K=100 結案，K=200 屬另一配置 |
| 高頻重建可行路徑 | CS 理論：K=100/200 均遠低於 ~5000 門檻 | **CLOSED**：高頻不可達為數學必然 |
| EXP-070 KE=84% 是否 AL 設計失敗 | **重訪（2026-05-07）**：evaluator double-scale 假象；真實 KE=6.30% 優於 baseline；div_l2 退步 3.7× trade-off 真實但非「失敗」 | **REOPENED**（2026-05-07）— ADR-001 §7.2 結論待重評；詳見 [`docs/diagnostics_log.md`](diagnostics_log.md) |
| `physics_output_denormalization` silent regression | 訓練端升格 config flag；evaluator default 反轉 + opt-in flag | **CLOSED**（2026-05-07）— Step 2 修補 + Round 7 evaluator 雙向驗證 |
| EXP-101 random sensor placement vs QR-pivot | **完成（2026-05-17, 1-shot 10k 步 2 h 26 m）**：KE **37.20 %**（vs EXP-080 10.68 %，+26.5 pp 3.5×）、u/v rel-L² **122/130 %**（error > reference, phase decorrelated）、kf phase ≈ −π（forcing mode 反相）、ek_ratio 0.39。Random sensor 在 sensor MSE 收斂、AL dual variable、continuity residual 三個獨立信號全劣於 QR-pivot；sensor placement 為架構之外 critical engineering lever。詳見 [`docs/experiment_archive_kolmogorov_post_k100.md`](experiment_archive_kolmogorov_post_k100.md) EXP-101 RECORD。 | **CLOSED**（2026-05-17）— Manohar 2018 QR-pivot 優越性在 K=100 / Re=10⁴ / PINN 設定下 5-6× pointwise 數值再驗證 |
| CfC Jacobian spectral radius stability | 未寫腳本 | 待開工（CFD-rigour pending task）|
| EXP-102 LES-informed QR-pivot pipeline | KE 44.3% — placement spectral coverage 與 DNS-pivot 等價（fourier pseudo-inverse k=1..30 acc 幾乎相同），失敗根因疑為 model 對 sensor measurement distribution overfit | 待 sanity training（EXP-103 DNS-downsampled-pivot N=128 對照）|

---

## [INDEX] Cylinder Active

| ID | Status | 主題 | 一句結論 |
|---|---|---|---|
| `CEXP-002` | `ACTIVE_BASELINE` | Cylinder, K=100, inflow BC loss | KE 3.5%（14.5× 改善 vs CEXP-001）|
| `CEXP-001` | `NEGATIVE_RESULT` | Cylinder, K=100, 無 BC loss | KE 51%（PHYSICAL_FAILURE）|

完整資料設定、訓練紀錄、NaN 診斷見 [`docs/cylinder_log.md`](cylinder_log.md)。

---

## [INDEX] Active

> 僅列出當前主線與最近實驗。完整 RECORD 與 GROUP 詳情見各 archive。

### Kolmogorov 當前主線

| ID | Status | 主題 | 一句結論 |
|---|---|---|---|
| `EXP-030` | `ACTIVE_BASELINE` | Re=1000：SOAP+SF resume → 5000 steps | KE 9.61%、amp 1.027；首次破 10%（archive G7）|
| `EXP-064` | `ACTIVE_BASELINE` | Re=10000：EXP-063 + sensor continuity | KE 7.80%、div_l2 0.184；K=100 結案值（post_k100 archive）|
| `EXP-080` | `ACTIVE_REFERENCE` | Re=10000：AL Pareto sweet spot（4-task GN + AL ρ=0.1）| KE 10.68%、div 0.067、ek_ratio 0.911；論文主要 result（post_k100 archive）|

### 最近實驗概要

| 群組 | EXP 範圍 | 一句結論 | 完整位置 |
|---|---|---|---|
| AL series H | EXP-070~077 | AL-continuity 系列，div 突破 0.05 但 KE 退步 | [`post_k100 archive`](experiment_archive_kolmogorov_post_k100.md) G_AL_H |
| 9-point Pareto I | EXP-078, 079 | 「兩全其美」不存在 | [`post_k100 archive`](experiment_archive_kolmogorov_post_k100.md) G_AL_I |
| AL strength J | EXP-080~082 | ρ ablation 找到 sweet spot，EXP-082 resume invalid | [`post_k100 archive`](experiment_archive_kolmogorov_post_k100.md) G_AL_J |
| 6-lever pivot | EXP-083~087 | 6 個架構 lever 全 falsified | [`post_k100 archive`](experiment_archive_kolmogorov_post_k100.md) G_pivot |
| Architectural ablation | EXP-088~090 | 2×2 B0/B1/B2/B3 — CfC + cross-attn 都 essential | [`post_k100 archive`](experiment_archive_kolmogorov_post_k100.md) G_arch |
| Standard PINN baseline | EXP-091, 092 | Operator framework 勝 single-instance PINN 15-25pp；SiLU > tanh | [`post_k100 archive`](experiment_archive_kolmogorov_post_k100.md) G_arch |
| Multi-seed reproducibility | EXP-093~100 | n=5 per arch；B3 vs B0 KE +7.75pp Cohen d=13 | [`post_k100 archive`](experiment_archive_kolmogorov_post_k100.md) G_seed |
| Inference benchmark | EXP-094 sub | encoder 71 ms + query 1.5 ms（B3 seed=2）| [`post_k100 archive`](experiment_archive_kolmogorov_post_k100.md) G_bench |
| EXP-101 (pending) | random sensor vs QR-pivot | 訓練中斷 step 4500/10000 | [`post_k100 archive`](experiment_archive_kolmogorov_post_k100.md) PENDING |
| **EXP-102** | LES-informed QR-pivot pipeline（N=128, stand-alone）| KE **44.3%** — 不過 sanity check 顯示問題非 placement informativeness（4 組 placement spectral coverage 等價）| [`post_k100 archive`](experiment_archive_kolmogorov_post_k100.md) G_pipeline |
| **EXP-103** | LES-informed QR-pivot pipeline（N=256, dns-init，confound-removed retry）| KE **52.0%** — **反而比 EXP-102 退步 8pp**；information-content 分析揭露 LES_N256 effective rank 最低、redundancy 最高，falsify「LES quality 是 bottleneck」假設 | [`post_k100 archive`](experiment_archive_kolmogorov_post_k100.md) G_pipeline |
| **EXP-105** | T=50 statistically-converged LES + QR-pivot（v1 buggy 53.7%）| **v2 fixed KE 12.36%** — 修完 axis-swap bug 後接近 baseline | [`post_k100 archive`](experiment_archive_kolmogorov_post_k100.md) G_pipeline |
| **EXP-101 v2** | Random uniform placement | v1 37.2% → **v2 13.25%** | 同上 |
| **EXP-102 v2** | LES_N128 stand-alone (α=30) | v1 44.3% → **v2 12.40%** | 同上 |
| **EXP-103 v2** | LES_N256 T=5 dns-init (short) | v1 52.0% → **v2 23.48%** (短窗 outlier) | 同上 |
| **EXP-106** | LES_N256 T=30 dns-init (NEW) | **v1=13.08%** | 同上 |
| **AXIS BUG** | sensors_qrpivot_from_les / sensors_random_from_dns 使用 swap row/col convention | **修完後 KE 平均改善 ~32pp**；5/5 unit test PASS guard 已加 | [`post_k100 archive`](experiment_archive_kolmogorov_post_k100.md) `[CRITICAL] AXIS BUG DISCOVERY` |

### 歷史群組（archive 內含完整 RECORD）

| GROUP | EXP 範圍 | 群組角色 | 群組 status |
|---|---|---|---|
| **G14** | EXP-062~064 | LearnableFourier 演進 → EXP-064 baseline | `ACTIVE`（EXP-064）|
| **G13** | EXP-057~061 | 冷啟動 IC weight 系列；EXP-057 為冷啟動 + IC weight 最佳 | `RESOLVED` |
| **G12** | EXP-049~056 | EXP-048 resume 變體；EXP-055 IC weight 為主要正向 | `RESOLVED` |
| **G11** | EXP-044~047 | d=256 從頭 3k 失敗群（GradNorm/sweep/locality 證偽）| `RESOLVED` |
| **G10** | EXP-043, 048 | d=256 漸進收斂線（3k→5k→10k：31.5%→27.2%→21.8%）| `SUPERSEDED`（被 G14）|
| **G9** | EXP-040~042 | Transfer learning 失敗（架構不匹配 + 負遷移）| `RESOLVED` |
| **G8** | EXP-031~033, 035~039 | Re=10000 新資料容量 + physics loss 失敗 | `RESOLVED` |
| **G7** | EXP-026~030 | Re=1000 SOAP+SF 主線 → EXP-030 baseline | `ACTIVE`（EXP-030）|
| **G6** | EXP-023~025 | Re=1000 SF vs stepLR 5k 對照；前主線 EXP-025 | `SUPERSEDED`（被 G7）|
| **G5** | EXP-021~022 | Re=1000 spatial encoding；KE 0.251→0.153（-39%）| `SUPERSEDED`（被 G6/G7）|
| **G4** | EXP-016~020 | Re=10000 舊資料容量探索；舊 DNS 已棄用 | `SUPERSEDED`（舊 DNS）|
| **G3** | EXP-013~015 | Re=1000 anchor 系列（top-k 失敗 + phase + temporal）| `SUPERSEDED`（被 G5）|
| **G2** | EXP-007, 008, 010~012 | Re=1000 baseline 確立；前主線 EXP-012 | `SUPERSEDED`（被 G3）|
| **G1** | EXP-001~006, 009 | Re=1000 早期 smoke + collapse 尺度診斷 | `RESOLVED`（根因定位）|

G1~G14 詳細 RECORD 見 [`docs/experiment_archive_kolmogorov.md`](experiment_archive_kolmogorov.md)。

---

## [INDEX] Negative

| ID | Status | 主題 | 一句結論 |
|---|---|---|---|
| `EXP-040` | `NEGATIVE_RESULT` | Re=10000 transfer from EXP-030（架構不匹配）| `size mismatch`：直接 transfer 不可行 |
| `EXP-027` | `NEGATIVE_RESULT` | `SOAP resume → 5000 steps`（已取消）| 改做 SOAP+SF；無有效結果 |
| `EXP-002` | `NEGATIVE_RESULT` | `omega` 作為 data supervision | 設定不合理且數值明顯失控 |
| `EXP-004` | `NEGATIVE_RESULT` | 低載 baseline | 能跑，但仍 near-zero collapse |
| `EXP-005` | `NEGATIVE_RESULT` | momentum smoke + curriculum off | 問題是尺度爆量，不是 physics 啟動太早 |
| `EXP-009` | `NEGATIVE_RESULT` | 5k 長訓練 | 訓練更久沒有帶來更好物理解 |
| `EXP-013` | `NEGATIVE_RESULT` | `top-k local attention` | 主模態與整體品質都下降 |
| `EXP-016` | `NEGATIVE_RESULT` | Re=10000 baseline (σ_max=16, small) | early-time catastrophic failure |
| `EXP-017` | `NEGATIVE_RESULT` | Re=10000 + σ_max=32 (small) | σ 擴展反而惡化 |
| `EXP-066` | `MIXED_RESULT` | Re=10000, K=200 sensor 冷啟動 10k | band_mid 突破但低頻退步、L_phys 未收斂 |
| `EXP-065` | `NEGATIVE_RESULT` | Re=10000, trunk MLP 1→2 層 | KE 7.74% 持平；band_mid/high≈100% 未改善 |
| `EXP-082` | `INVALID` | AL ρ=0.02（resume 災難）| Resume catastrophic state corruption；不寫進 ablation curve |
| `EXP-085` | `INVALID` | K=200 recipe-K mismatch | EXP-080 recipe 無法 transfer K=200；artifact 已刪 |

---

## [INDEX] Archived Context

| ID | Status | 主題 | 一句結論 |
|---|---|---|---|
| `EXP-001` | `ARCHIVED_CONTEXT` | 早期 `uvomega` 中長訓練 | 可跑，但後期收縮到低能量保守解 |
| `EXP-003` | `ARCHIVED_CONTEXT` | 改回 `u,v-only` smoke | 是必要修正，但當時仍不夠穩 |
| `EXP-006` | `ARCHIVED_CONTEXT` | `rff=32/gain=5` vs `rff=4/gain=1` 診斷 | 已定位 physics 爆量根因 |

---

## [INDEX] Context Missing

| 項目 | 缺口 |
|---|---|
| `EXP-012` | 精確主線 TOML 未存於 repo，只能由 `small.toml + k_f=2 + stepLR(500x0.9) + 3000 steps` 重建 |
| `EXP-013` | 精確 `top-k` config 未存於 repo，目前僅能由 artifact 名稱與紀錄描述回推 |
| `EXP-004` | `lowload` 專用 TOML 未存於 repo |
| `EXP-001` | 早期 `deeponet_cfc_midlong_uvomega.toml` 未存於 repo |

---

## 拆檔歷史

- **2026-05-06**：第一次拆檔 — `[ANALYSIS]` 條目（Wavelet/AIM/Denorm 早期）→ [`docs/analysis_reports.md`](analysis_reports.md)；EXP-001~063 詳細 RECORD → [`docs/experiment_archive_kolmogorov.md`](experiment_archive_kolmogorov.md)。
- **2026-05-15**：第二次拆檔 — `[STATE] Supported Decisions` 詳細條目 + EXP-064~101 RECORD → [`docs/experiment_archive_kolmogorov_post_k100.md`](experiment_archive_kolmogorov_post_k100.md)；`[STATE] Cylinder Wake` → [`docs/cylinder_log.md`](cylinder_log.md)；兩份 `[DIAGNOSTIC]` → [`docs/diagnostics_log.md`](diagnostics_log.md)。主檔精簡至 STATE/INDEX 結論層。
