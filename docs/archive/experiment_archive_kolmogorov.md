# Kolmogorov 實驗紀錄歷史檔（Archive）

本檔以「實驗群組」（GROUP）格式保留 Kolmogorov 主線歷史 RECORD（EXP-001 ~ EXP-063），按時序與主題合併。

從 `docs/experiment_log.md` 拆出（2026-05-06），同日重組為 GROUP 樣式以濃縮重複 metadata。

**讀取建議：**

1. 一般查詢 → 先讀 [`docs/experiment_log.md`](experiment_log.md) 的 `[STATE]` 與 `[INDEX]`
2. 需要 **EXP-064 ~ EXP-101** 詳細 RECORD（K=100 結案後 AL/pivot/multi-seed/benchmark）→ [`docs/experiment_archive_kolmogorov_post_k100.md`](experiment_archive_kolmogorov_post_k100.md)
3. 需要 **EXP-001 ~ EXP-063** 早期 EXP 細節（含 Hypothesis / Discussion / metric 比較）→ 才讀本檔對應 GROUP
4. Cylinder Wake（CEXP 系列）→ [`docs/cylinder_log.md`](cylinder_log.md)
5. 量化分析（wavelet, AIM, 早期 denorm diagnostic）→ [`docs/analysis_reports.md`](analysis_reports.md)
6. CFD-rigour validation / silent regression 細節 → [`docs/diagnostics_log.md`](diagnostics_log.md)

**GROUP 樣式約定：**

每組保留：Status / Time / Topic → Hypothesis → 共同設定 → 個別實驗 metric 表 → Discussion → Configs / Artifacts → Supersedes 鏈。
個別 EXP 的 Decision 收斂在 metric 表的 Status 欄與 Discussion 段。

---

## [GROUP INDEX]

| GROUP | 時間 | EXP 範圍 | 主題 | 群組結論 |
|---|---|---|---|---|
| G1 | 2026-03-30 | EXP-001~006, 009 | Re=1000 早期 smoke + near-zero collapse 尺度診斷 | 根因為 `rff_sigma=32 + gain=5` 過大；u,v-only supervision 必要 |
| G2 | 2026-03-30~31 | EXP-007, 008, 010, 011, 012 | Re=1000 baseline 確立（rff=4→Small→k_f=2→stepLR） | EXP-012 為前主線；time_marching 不可關 |
| G3 | 2026-03-31 | EXP-013~015 | Re=1000 anchor 系列（top-k/phase/temporal） | phase+temporal anchor 改善 amp/KE；t=3.5~4.5 phase 高峰為 Lyapunov 物理本質 |
| G4 | 2026-03-31 | EXP-016~020 | Re=10000 舊資料容量探索（σ/Wide+resume） | 舊 DNS 已棄用；確立「容量是 early-time 貢獻因子，σ 不是根因」 |
| G5 | 2026-04-01 | EXP-021~022 | Re=1000 spatial encoding（periodic Fourier + isotropic relpos） | KE 0.251→0.153，-39%；條紋根源確認為 relpos 方向輸入 |
| G6 | 2026-04-01 | EXP-023~025 | Re=1000 Schedule-Free vs stepLR 5k 對照 | SF Polyak 平均優於 stepLR（KE -13%, amp 0.995） |
| G7 | 2026-04-02~ | EXP-026~030 | Re=1000 SOAP+SF 主線（→ EXP-030 baseline） | EXP-030 KE **9.61%**，首次破 10%；SOAP 二階曲率 + SF 雙效 |
| G8 | 2026-04-08~ | EXP-031~033, 035~039 | Re=10000 新資料容量 + physics loss 機制變更失敗 | d=256 為下限；physics loss 設計改動均無法突破資訊論上限 |
| G9 | 2026-04-09 | EXP-040~042 | Re=10000 transfer learning 失敗 | 架構不匹配 + source 品質不足 → transfer 證偽 |
| G10 | 2026-04-13~15 | EXP-043, 048 | Re=10000 d=256 漸進收斂線（resume）| 3k→5k→10k：KE 31.5%→27.2%→21.8% 邊際遞減但持續 |
| G11 | 2026-04-10~14 | EXP-044~047 | Re=10000 d=256 從頭 3k 失敗群（locality/sweep/GradNorm）| GradNorm/sweep 在 sparse 設定下證偽 |
| G12 | 2026-04-15~21 | EXP-049~056 | Re=10000 EXP-048 resume 變體（RAR/L-BFGS/IC weight）| **IC weight 為單一最有效改動（KE 21.8%→17.1%）**；RAR freq=1000 有效；組合產生負干擾 |
| G13 | 2026-04-21~22 | EXP-057~061 | Re=10000 冷啟動 IC weight 系列（雙向 CfC/h32/jaxpi 對齊）| t=0 問題根因為訓練訊號不足；雙向 CfC/h32 證偽 |
| G14 | 2026-04-23~24 | EXP-062~063 | Re=10000 LearnableFourier 演進（→ EXP-064 baseline）| LearnableFourierEmb + GradNorm + sensor continuity 達 KE **7.80%**（K=100 結案值）|

---

## [GROUP G1] Re=1000 早期 smoke + near-zero collapse 尺度診斷（EXP-001 ~ EXP-006, EXP-009）

- **Status**: `RESOLVED`（根因已定位）
- **Time**: 2026-03-30 03:12 ~ 05:00 +0800
- **Topic**: 確認 sparse u,v sensor 重建可行性、定位「near-zero collapse」失敗模式根因

### Hypothesis

1. 早期 `uvomega` 主線可跑通 sparse + physics 訓練流程（EXP-001 驗證）。
2. 將 `omega` 加入 sensor data supervision 可加速收斂（EXP-002 探索）。
3. 觀測到的 near-zero collapse 是 batch/算力不足造成（EXP-004 假設，後證偽）。
4. 真正的失敗根因可能是 physics 啟動時機、初始化增益、或頻率尺度（EXP-005/006 診斷）。

### 共同設定

- domain `[0,1]^2`、Re=1000、K=100 sensors
- 早期 RFF: `rff_sigma=32, output_head_gain=5`（後證為爆量根因）
- physics: `momentum + continuity`（EXP-005 之後）

### 個別實驗

| ID | 改動 | 結果指標 | Status |
|---|---|---|---|
| EXP-001 | 早期 uvomega midlong（thin sensor tokens + cross-attn）| step_500: KE 0.668, ens 2.80；final: KE 0.913, ens 0.541（收縮至低能量保守解）| `ARCHIVED_CONTEXT` |
| EXP-002 | 切換新機制 + omega 入 supervision | L_data step1=1.09e3, step3=6.85e3（爆量）| `NEGATIVE_RESULT` |
| EXP-003 | 改回 u,v-only smoke | L_data step1=38.7, step3=256（仍不穩，但是必要修正）| `ARCHIVED_CONTEXT` |
| EXP-004 | 低載 baseline 1000 step | KE 0.9995, ens 0.9990, u_std 2.04e-3（**near-zero collapse**）| `NEGATIVE_RESULT` |
| EXP-005 | momentum smoke + 關 curriculum | L_phys step1=3.54e5, step3=7.17e5（physics 啟動時量級已爆）| `NEGATIVE_RESULT` |
| EXP-006 | rff=32+gain=5 vs rff=4+gain=1 尺度診斷 | rff=32+gain=5: u_std=1.08, mom_residual=476；rff=4+gain=1: u_std=0.063, mom_residual=0.89~1.44 | `ARCHIVED_CONTEXT` |
| EXP-009 | 5k 長訓練 final | KE 0.953, u_std 0.0162（near-zero, 訓練更久反更平滑）| `NEGATIVE_RESULT` |

### Discussion

1. **Near-zero collapse 根因定位**：EXP-006 對照確認 `rff_sigma=32 + output_head_gain=5` 把 spatial gradient 的 RMS 推到 180~250（vs 健康範圍 1.2~1.4），lap_u_rms 達 1.11e5；momentum residual 必爆。降載（EXP-004）只能讓程序活下來，無法修正尺度。
2. **omega-as-supervision 證偽**：EXP-002 的 L_data 量級顯示 omega（高階導數）的數值範圍與 u,v 不匹配，當作直接 supervision 會主導 loss。`omega` 自此僅作物理量與診斷量。
3. **訓練時間並非解方**：EXP-009 在 5k 步後仍是 near-zero，反而 u_std 更低；確認延長訓練不能修復尺度問題，必須從表徵層動手。
4. **永久結論**：`u,v-only sensor supervision` 與 `rff_sigma ≤ 4 + output_head_gain = 1` 為後續所有實驗的硬約束。

### Configs / Artifacts

| ID | Config | Artifact | 備註 |
|---|---|---|---|
| EXP-001 | `[CONTEXT_MISSING]` `deeponet_cfc_midlong_uvomega.toml` | `[CONTEXT_MISSING]` `deeponet-cfc-midlong-uvomega` | sweep 評估 step_250/500/750/final |
| EXP-002 | — | `[CONTEXT_MISSING]` `deeponet-cfc-smoke-uvomega-mechcheck` | smoke only |
| EXP-003 | — | `[CONTEXT_MISSING]` `deeponet-cfc-smoke-uvonly-check2` | smoke only |
| EXP-004 | `[CONTEXT_MISSING]` `deeponet_cfc_midlong_uvomega_lowload.toml` | `[CONTEXT_MISSING]` `deeponet-cfc-midlong-uvomega-lowload-baseline-1000` | — |
| EXP-005 | — | `[CONTEXT_MISSING]` `deeponet-cfc-smoke-uvonly-momentum-check2` | — |
| EXP-006 | — | — | 尺度量化診斷無 train artifact |
| EXP-009 | — | `[CONTEXT_MISSING]` `deeponet-cfc-midlong-uvomega-5k` | — |

### Supersedes / Superseded_By

- **Supersedes**: 無（最早 group）
- **Superseded_By**: G2（EXP-007 起，採用 rff=4+gain=1 進入有效 regime）

---

## [GROUP G2] Re=1000 baseline 確立（EXP-007, 008, 010, 011, 012）

- **Status**: `SUPERSEDED`（前主線，被 G3+G5+G6+G7 取代）
- **Time**: 2026-03-30 ~ 03-31
- **Topic**: 從尺度修正後逐步確立 Re=1000 工作主線（rff=4 → Small → k_f=2 → stepLR）

### Hypothesis

1. `rff_sigma=4 + gain=1` 修正 G1 的尺度爆量後，模型應跳出 near-zero collapse（EXP-007）。
2. `Small` 尺寸（d=64）已足以進入有效 regime，不需 wider model（EXP-008）。
3. Re=1000 forcing mode 應為 `k_f=2`（EXP-010 修正先前錯誤）。
4. `time_marching` 是必要技巧，不是可有可無的 trick（EXP-011 對照）。
5. 延長訓練 + stepLR 衰減能持續改善精度（EXP-012）。

### 共同設定

- domain `[0,1]^2`、Re=1000、K=100 sensors
- `rff_sigma=4, output_head_gain=1`（G1 結論）
- u,v-only sensor supervision（G1 結論）
- physics: `momentum(u,v,p) + continuity`，`p` 為 latent 場

### 個別實驗

| ID | 時間 | 改動 vs prev | KE | Ens | u_rmse | kf_amp | kf_phase | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| EXP-007 | 03-30 05:19 | rff=4+gain=1 step600 | 0.312 | 0.259 | 0.139 | — | — | `ACTIVE_REFERENCE`（已脫離 collapse）|
| EXP-008 | 03-30 12:21 | Small 1000-step（d_model=64, op_rank=64）| 0.340 | 0.270 | 0.142 | — | — | `ACTIVE_REFERENCE` |
| EXP-010 | 03-31 00:18 | k_f=4 → 2 修正 | 0.454 | 0.227 | 0.142 | 0.779 | -0.753 rad | `ACTIVE_REFERENCE`（k_f bug 修正）|
| EXP-011 | 03-31 00:22 | 關掉 time_marching | 0.458 | 0.508 | 0.155 | 0.273 | -0.880 rad | `ACTIVE_REFERENCE`（負對照）|
| EXP-012 | 03-31 01:28 | stepLR(500×0.9) + 3000 steps | **0.318** | **0.192** | **0.108** | **0.781** | -0.531 rad | `ACTIVE_REFERENCE`（前主線）|

EXP-007 為診斷 checkpoint（step_600，未走完），參數量 182,226（EXP-008+）。

### Discussion

1. **EXP-007 確認尺度修正即可脫離 collapse**：u_std 從 EXP-006 的 0.063 直接進入 RMSE ~0.14 的有效範圍；`rff_sigma=4 + gain=1` 為後續硬約束。
2. **EXP-008 確認小模型即足夠**：d_model=64 + op_rank=64 在 1000 step 內可達 KE 34%、ens 27%；不需先放大架構。
3. **EXP-010 的 k_f=2 修正**：先前認為 Re=1000 forcing mode 為 k_f=4 是錯的，修正後 KE 變高（0.454 vs EXP-008 的 0.340）— 看似退步，但問題從 amplitude 轉成 phase 對齊（kf_amp_ratio 0.779），更易處理。
4. **EXP-011 為時域結構提供負證據**：關掉 time_marching 後 amp_ratio 從 0.78 崩到 0.27，phase err -0.88 rad，整體場品質明顯變差。`time_marching` 自此為硬約束。
5. **EXP-012 為前主線**：stepLR(500×0.9) + 3000 步把 KE 從 EXP-008 的 0.340 → 0.318，ens 從 0.270 → 0.192；u_rmse 0.108（最好）。後續 anchor、spatial encoding 改進均以此為比較起點。

### Configs / Artifacts

| ID | Config | Artifact |
|---|---|---|
| EXP-007 | — | `[CONTEXT_MISSING]` `deeponet-cfc-midlong-uvomega-lowload-rff4-gain1-1000/checkpoints/picon_kolmogorov_step_600.pt` |
| EXP-008 | [`configs/exp_008_re1000_small_baseline.toml`](../configs/exp_008_re1000_small_baseline.toml) | `[CONTEXT_MISSING]` `deeponet-cfc-midlong-uvomega-small-1000` |
| EXP-010 | — | [`artifacts/deeponet-cfc-midlong-uvomega-small-kf2-1000`](../artifacts/deeponet-cfc-midlong-uvomega-small-kf2-1000) |
| EXP-011 | — | [`artifacts/deeponet-cfc-midlong-uvomega-small-notm-1000`](../artifacts/deeponet-cfc-midlong-uvomega-small-notm-1000) |
| EXP-012 | `[CONTEXT_MISSING]` 精確主線 toml 未存於 repo | [`artifacts/deeponet-cfc-midlong-uvomega-small-step500x0p9-3000`](../artifacts/deeponet-cfc-midlong-uvomega-small-step500x0p9-3000) |

### Supersedes / Superseded_By

- **Supersedes**: G1（EXP-007 採用 G1 診斷出的 rff=4+gain=1 設定）
- **Superseded_By**: G3（EXP-014/015 加入 anchor → KE 0.318→0.251, -21%）

---

## [GROUP G3] Re=1000 anchor 系列（EXP-013 ~ EXP-015）

- **Status**: `SUPERSEDED`（EXP-015 為當期主線，整組後被 G5/G6 取代）
- **Time**: 2026-03-31
- **Topic**: trunk 注入 phase / temporal 錨點，與 top-k local attention 對 forcing mode 重建的影響

### Hypothesis

1. phase anchor `sin/cos(2π k_f y/L)` 注入 branch+trunk 應改善 forcing mode amplitude 與 phase 對齊。
2. temporal anchor `sin/cos(2π n t / T_total)` 進一步補充絕對時間座標，可降低 KE/Ens 並消除 t=3.5~4.5 phase 高峰。
3. top-k local attention 可降計算成本同時改善 phase alignment（順帶探索）。

### 共同設定（vs EXP-012 baseline）

- d_model=64（Small），multi-scale RFF bands `[[16,4.0],[8,8.0],[8,16.0]]`
- AdamW + stepLR(500×0.9)，3000 steps
- 觀測 supervision：u, v；physics：momentum + continuity
- 只改動 anchor 或 attention 機制，其餘超參與 EXP-012 完全相同

### 個別實驗

| ID | 改動 vs prev | KE | Ens | u_rmse | kf_amp | kf_phase | Status |
|---|---|---:|---:|---:|---:|---:|---|
| EXP-013 | top-k=16 local attn（取代 full attn） | 0.398 | 0.352 | 0.145 | 0.552 | -0.870 rad | `NEGATIVE_RESULT` |
| EXP-014 | + use_phase_anchor=true | 0.277 | 0.192 | 0.099 | 0.993 | -0.153 rad | `ACTIVE_REFERENCE` |
| EXP-015 | EXP-014 + use_temporal_anchor（n_harmonics=2） | **0.251** | **0.172** | 0.098 | **0.997** | -0.218 rad | `ACTIVE_BASELINE`（當期）|

EXP-015 額外指標：`phase_err_|max| = 0.6554 @ t=3.5∼4.5`、`phase_err_std = 0.185`、+256 params。

### Discussion

1. **EXP-013（top-k local attention）證偽**：KE 0.398 顯著差於 EXP-012 baseline（0.318），主模態與整體場品質皆下降。確認 full attention 對全域 sensor token 聚合是必要的，不可用 locality 優化取代。

2. **EXP-014（phase anchor）顯著正向**：amplitude +27%（0.78 → 0.993）、phase err -71%（-0.53 → -0.15 rad）、t=0 初始偏移 -0.555 → -0.109 rad。branch encoder 的 phase 表徵能力直接提升。
   - **[RISK]** amp_ratio=0.993 過高，疑似訓練時段過擬合（無 OOD 驗證）。

3. **EXP-015（+ temporal anchor）關鍵物理洞見**：
   - KE/Ens 各降 ~10%，整體積分量改善顯著。
   - 但 t=3.5∼4.5 phase 高峰（0.64 rad）**未改善**，原 hypothesis 被否定。
   - **確認 t=3.5∼4.5 ~0.64 rad phase 偏差為 Re=1000 Kolmogorov flow 的 Lyapunov 不穩定極限**，非表徵或訓練策略問題。後續不再以表徵改動追求此點。

### Configs / Artifacts

| ID | Config | Artifact |
|---|---|---|
| EXP-013 | `[CONTEXT_MISSING]`（不在 repo） | `artifacts/deeponet-cfc-midlong-uvomega-small-topk16-step500x0p9-3000` |
| EXP-014 | [`configs/exp_014_re1000_phaseanchor.toml`](../configs/exp_014_re1000_phaseanchor.toml) | `artifacts/deeponet-cfc-midlong-uvomega-small-phaseanchor-3000` |
| EXP-015 | [`configs/exp_015_re1000_temporal_anchor.toml`](../configs/exp_015_re1000_temporal_anchor.toml) | `artifacts/deeponet-cfc-midlong-uvomega-small-temporal-anchor-3000` |

### Supersedes / Superseded_By

- **Supersedes**: G2（EXP-007~012, baseline 確立）
- **Superseded_By**: G5（EXP-021~022 spatial encoding，KE 0.251 → 0.153，-39%）
- **永久結論**：t=3.5∼4.5 phase 高峰為 chaotic divergence 物理本質（Lyapunov 不穩定性），無後續實驗能解決

---

## [GROUP G4] Re=10000 舊資料容量探索（EXP-016 ~ EXP-020）

- **Status**: `SUPERSEDED`（舊 DNS 已棄用，新資料 si100 由 G8 起接手）
- **Time**: 2026-03-31
- **Topic**: 將 Re=1000 EXP-015 主線架構移植至 Re=10000，逐步加大容量並 resume 延長訓練

### Hypothesis

1. EXP-015 主線架構（Small + RFF + anchors）可直接應用於 Re=10000（EXP-016 假設，後證偽）。
2. 擴展 RFF σ_max 至 32 提供更高頻覆蓋可改善 early-time phase（EXP-017 假設，後證偽）。
3. 增加模型容量（d_model 64 → 128）可解決 early-time catastrophic failure（EXP-018 假設）。
4. 延長訓練（5k → 10k steps）持續單調改善（EXP-019/020 假設）。

### 共同設定

- 舊 DNS：Re=10000, ν=0.0001, 256×256, 41 frames, dt=0.125, T=5.0（時間遠比 Re=1000 的 101 frames 稀疏）
- u,v-only supervision（G1 結論）
- AdamW + stepLR(500×0.9)
- 注意：所有 G4 config 引用舊 DNS，後已隨舊 DNS 一併刪除

### 個別實驗

| ID | 步數 | 改動 vs prev | KE | u_rmse | kf_amp | kf_phase | max_phase@t≤1.0 | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| EXP-016 | 3000 | 直接移植 EXP-015（Small + σ=16）| 0.645 | 0.310 | 0.416 | 1.420 rad | **2.50 rad** | `NEGATIVE_RESULT`（catastrophic）|
| EXP-017 | 3000 | + σ_max=32（Small）| 0.591 | 0.284 | 0.530 | 0.992 rad | **5.37 rad**（更差）| `NEGATIVE_RESULT` |
| EXP-018 | 3000 | + Wide d=128, op_rank=128, σ_max=32 | 0.573 | 0.261 | 0.461 | 0.154 rad | 0.71 rad | `ACTIVE_REFERENCE`（容量假設成立）|
| EXP-019 | 5000（resume EXP-018）| + 延長至 5k | 0.536 | 0.235 | **0.595** | 0.048 rad | 0.46 rad | `ACTIVE_REFERENCE`（舊資料最佳）|
| EXP-020 | 10000（resume EXP-019）| + 延長至 10k | 0.540 | 0.260 | 0.548（**退化**）| -0.079 rad | 0.43 rad | `ACTIVE_REFERENCE`（飽和震盪）|

EXP-020 L_data 軌跡：step 7000=2.03e-2, step 8000=**1.27e-2 觸底**, step 9000=1.34e-2, step 10000=1.89e-2（震盪）。

### Discussion

1. **EXP-016 catastrophic 證實 Re=1000 主線不能直接移植**：max_phase@t≤1.0 達 2.50 rad（vs Re=1000 的 ~0.32），early-time 重建完全失敗。時間稀疏（41 vs 101 frames, dt=0.125 vs 0.05）+ 容量不足（Small）共同造成。
2. **EXP-017 σ 擴展證偽**：σ_max 從 16 擴至 32，max_phase 不降反升至 5.37 rad；low-frequency representation 能量被稀釋，confirming **σ 不是 root cause**。
3. **EXP-018 容量假設成立**：d_model=64→128（+ op_rank+QMLP 一併）使 max_phase@t≤1.0 從 >2 rad 降至 0.71 rad；**確認模型容量是 early-time failure 的貢獻因子**。amp_ratio=0.461 仍低，時間稀疏為剩餘主因。
4. **EXP-019 resume 帶來明顯且單調收益**：amp +13.4%、phase -69%、max_phase -35%；模型在 5k 仍未飽和。
5. **EXP-020 揭示飽和點**：step 8000 觸底後震盪，amp_ratio 退化（0.595→0.548）；確認此 config 的有效訓練上限約 5k 步。**繼續訓練無益**，應換方向（資料條件、loss 設計）。

### Configs / Artifacts

| ID | Config | Artifact |
|---|---|---|
| EXP-016 | `[STATUS: DELETED]` `deeponet_cfc_midlong_uvomega_small_re10000.toml`（隨舊 DNS 清除）| [`artifacts/deeponet-cfc-midlong-uvomega-small-re10000-3000`](../artifacts/deeponet-cfc-midlong-uvomega-small-re10000-3000) |
| EXP-017 | `[STATUS: DELETED]` `deeponet_cfc_midlong_uvomega_small_re10000_sigma32.toml` | [`artifacts/deeponet-cfc-re10000-sigma32-3000`](../artifacts/deeponet-cfc-re10000-sigma32-3000) |
| EXP-018 | `[STATUS: DELETED]` `deeponet_cfc_midlong_uvomega_wide_re10000_sigma32.toml` | [`artifacts/deeponet-cfc-re10000-wide-sigma32-3000`](../artifacts/deeponet-cfc-re10000-wide-sigma32-3000) |
| EXP-019 | 同上（resume 用同 config）| [`artifacts/deeponet-cfc-re10000-wide-sigma32-5000`](../artifacts/deeponet-cfc-re10000-wide-sigma32-5000) |
| EXP-020 | 同上 | [`artifacts/deeponet-cfc-re10000-wide-sigma32-10000`](../artifacts/deeponet-cfc-re10000-wide-sigma32-10000) |

### Supersedes / Superseded_By

- **Supersedes**: 移植自 G3（EXP-015 主線）
- **Superseded_By**: G8（EXP-031 起切換新 DNS si100，dt=0.025，201 frames）
- **永久結論**：時間稀疏 + 容量都是 Re=10000 的瓶頸；σ 擴展非解方

---

## [GROUP G5] Re=1000 spatial encoding 改進（EXP-021 ~ EXP-022）

- **Status**: `SUPERSEDED`（被 G6/G7 取代）
- **Time**: 2026-04-01
- **Topic**: 用確定性 periodic Fourier 取代隨機 RFF + 等向 relpos_bias 消除 x 條紋偽影

### Hypothesis

1. RFF（seed=42）有 8/32 個近純 x 方向頻率向量（angle <20°），造成流場直條紋偽影；改用確定性週期 Fourier 編碼可消除（EXP-021）。
2. EXP-021 後仍殘存 x 條紋，根因為 `relpos_bias` 的方向輸入 `(rel_x, rel_y)` 把感測器 x 非均勻分佈（66/128 columns）注入 attention bias；改純距離 `|rel|`（等向）可消除（EXP-022）。

### 共同設定（vs EXP-015 baseline）

- d_model=64（Small），AdamW + stepLR(500×0.9)，3000 steps
- 移除 RFF B matrix buffer 與 `use_phase_anchor`（k_f=2 已包含在 k=2 諧波）
- `fourier_harmonics=8`（共 32 特徵）

### 個別實驗

| ID | 改動 vs prev | KE | Ens | u_rmse | v_rmse | kf_amp | kf_phase | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| EXP-021 | RFF → periodic_fourier_encode（h=8）| **0.153** | 0.137 | 0.081 | 0.087 | 1.152 | -0.381 rad | `ACTIVE_REFERENCE` |
| EXP-022 | + relpos_bias `(rel_x,rel_y,\|rel\|)` → 純 `\|rel\|`（等向）| 0.153 | **0.119** | 0.081 | 0.087 | 1.152 | -0.381 rad | `ACTIVE_BASELINE`（當期）|

### Discussion

1. **EXP-021 確認空間編碼品質直接影響整體精度**：KE 從 EXP-015 的 0.251 → 0.153（**-39%**），u_rmse -23%；確定性 periodic Fourier 完全消除 RFF 角度偏差。
2. **EXP-021 殘留 x 條紋的根因診斷**：vorticity field 仍見 x 方向條紋；分析 sensor 分佈為 66/128 x-columns covered（非均勻），這個資訊經 `relpos_bias` 的方向輸入 `(rel_x, rel_y)` 進入 attention，產生方向偏差。
3. **EXP-022 等向 attention 確認**：純距離輸入後 vorticity error 場轉為隨機分佈，KE 持平（0.153 → 0.153），ens 改善（0.137→0.119）。**確認 Kolmogorov 等向 attention 是正確設計**：感測器貢獻只與距離相關，方向資訊無收益且是條紋根源。
4. **永久結論**：`relpos_bias` 必須只用 `|rel|`（純距離），不可加方向向量。

### Configs / Artifacts

| ID | Config | Artifact |
|---|---|---|
| EXP-021 | [`configs/exp_021_re1000_periodicfft.toml`](../configs/exp_021_re1000_periodicfft.toml) | [`artifacts/deeponet-cfc-re1000-periodicfft-3000`](../artifacts/deeponet-cfc-re1000-periodicfft-3000) |
| EXP-022 | [`configs/exp_022_re1000_isotropicattn.toml`](../configs/exp_022_re1000_isotropicattn.toml) | [`artifacts/deeponet-cfc-re1000-isotropicattn-3000`](../artifacts/deeponet-cfc-re1000-isotropicattn-3000) |

### Supersedes / Superseded_By

- **Supersedes**: G3（EXP-015，KE 0.251 → 0.153, -39%）
- **Superseded_By**: G6（EXP-023~025 SF AdamW），G7（EXP-026~030 SOAP+SF）

---

## [GROUP G6] Re=1000 Schedule-Free vs stepLR 5k 對照（EXP-023 ~ EXP-025）

- **Status**: `SUPERSEDED`（被 G7 SOAP+SF 取代）
- **Time**: 2026-04-01
- **Topic**: 5000 steps 訓練長度下 Schedule-Free AdamW vs stepLR 對照，確認 Polyak 平均的獨立收益

### Hypothesis

1. EXP-022 在 3k 步未飽和，延長至 5k 應持續收益（EXP-023）。
2. Schedule-Free AdamW 的 Polyak 平均應在不變 KE 的前提下改善 amp/phase 推理品質（EXP-024）。
3. SF 延長至 5k 可整體優於同步數 stepLR（EXP-025）。

### 共同設定（vs EXP-022 baseline）

- d_model=64（Small），periodic_fourier h=8，等向 relpos_bias，u,v-only supervision
- `lr=1e-3`，3000 步從頭或 resume → 5000

### 個別實驗

| ID | 模式 | optimizer | 步數 | KE | Ens | u_rmse | kf_amp | kf_phase | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| EXP-023 | resume EXP-022 → 5k | AdamW + stepLR(500×0.9) | 5000 | 0.140 | 0.119 | 0.073 | 1.072 | -0.361 rad | `ACTIVE_REFERENCE` |
| EXP-024 | 從頭 3k | SF AdamW（warmup 300）| 3000 | 0.150 | 0.121 | 0.080 | 0.950 | -0.222 rad | `ACTIVE_REFERENCE` |
| EXP-025 | resume EXP-024 → 5k | SF AdamW | 5000 | **0.121** | **0.109** | **0.072** | **0.995** | -0.293 rad | `ACTIVE_BASELINE`（當期）|

L_data 軌跡：EXP-023: 3k→1.75e-2, 5k→1.32e-2；EXP-025: 3k→2.04e-2, 5k→1.40e-2。

### Discussion

1. **EXP-023 延長帶來單調收益**：KE -8.5%、u/v RMSE -10%/-11%；EXP-022 未飽和。
2. **EXP-024 SF 在 3k 不換 KE 換 amp/phase**：KE 0.150 vs EXP-022 0.153 持平；但 amp 0.950 vs 1.152（更接近 1）、phase err -0.222 vs -0.381（-42%）。**確認 Polyak 平均在推理品質上有獨立收益**。
3. **EXP-025 SF 延長 5k 全面優於 stepLR**：相比 EXP-023，KE -13%、Ens -8%、amp 達 **0.995（最接近 1.0）**。L_data 5k 步 1.40e-2 仍未飽和，可繼續 resume。
4. **永久結論**：SF AdamW 在 sparse-data PINN + 5k 步設定下優於 stepLR；後續主線採用。

### Configs / Artifacts

| ID | Config | Artifact |
|---|---|---|
| EXP-023 | [`configs/exp_023_re1000_isotropicattn_5k.toml`](../configs/exp_023_re1000_isotropicattn_5k.toml) | [`artifacts/deeponet-cfc-re1000-isotropicattn-5000`](../artifacts/deeponet-cfc-re1000-isotropicattn-5000) |
| EXP-024 | [`configs/exp_024_re1000_schedulefree.toml`](../configs/exp_024_re1000_schedulefree.toml) | [`artifacts/deeponet-cfc-re1000-schedulefree-3000`](../artifacts/deeponet-cfc-re1000-schedulefree-3000) |
| EXP-025 | [`configs/exp_025_re1000_schedulefree_5k.toml`](../configs/exp_025_re1000_schedulefree_5k.toml) | [`artifacts/deeponet-cfc-re1000-schedulefree-5000`](../artifacts/deeponet-cfc-re1000-schedulefree-5000) |

### Supersedes / Superseded_By

- **Supersedes**: G5（EXP-022, KE 0.153 → 0.121, -21%）
- **Superseded_By**: G7（EXP-030 SOAP+SF resume EXP-028 → 5k, KE 0.0961, -20% vs EXP-025）

---

## [GROUP G7] Re=1000 SOAP+SF 主線（EXP-026 ~ EXP-030）

- **Status**: `ACTIVE`（EXP-030 為當前 Re=1000 baseline）
- **Time**: 2026-04-02 ~
- **Topic**: SOAP（二階曲率估計）導入，搭配 Schedule-Free Polyak 平均，建立目前 Re=1000 最佳主線

### Hypothesis

1. SOAP 的 Kronecker factor preconditioner 在 sparse-data PINN 上能優於 first-order Adam 系（EXP-026）。
2. 進一步加 Schedule-Free 可疊加 Polyak 平均收益（EXP-028）。
3. 增加 CfC 深度（1→2 layer）可改善 KE 但需觀察是否損害 amp（EXP-029）。
4. SOAP+SF 在 5k 步下能突破 SF AdamW（EXP-025 KE 12.06%）（EXP-030）。

### 共同設定（vs EXP-022 架構）

- d_model=64, periodic_fourier h=8, 等向 relpos_bias, u,v-only
- 學習率 `lr=1e-3`，3000 步從頭或 resume → 5000
- 注意：早期評估腳本有 checkpoint 載入 bug（讀 `model` 而非 `model_state_dict`）→ EXP-026/028/029 評估指標經 bug 修正後重評

### 個別實驗

| ID | optimizer / 配置 | 步數 | KE | Ens | u_rmse | kf_amp | Status |
|---|---|---:|---:|---:|---:|---:|---|
| EXP-026 | SOAP（無 SF）| 3000 | 0.124 | 0.152 | — | 0.925 | `ACTIVE_REFERENCE`（SOAP 對照基準）|
| EXP-027 | SOAP resume → 5000 | — | — | — | — | — | `NEGATIVE_RESULT`（取消，改做 SOAP+SF）|
| EXP-028 | SOAP + Schedule-Free（從頭 3k）| 3000 | 0.122 | 0.134 | — | 1.039 | `ACTIVE_REFERENCE` |
| EXP-029 | 2-layer TemporalCfC + SF AdamW（從頭 3k）| 3000 | 0.111 | 0.124 | — | 0.759 | `ACTIVE_REFERENCE`（amp 退化）|
| EXP-030 | SOAP+SF resume EXP-028 → 5k | 5000 | **0.0961** | 0.119 | **0.0568** | **1.027** | `ACTIVE_BASELINE`（當前 Re=1000）|

EXP-030 為 Re=1000 全實驗中：
- 首次突破 KE 10% 門檻（9.61%）
- u_rmse_mean = 0.0568（全實驗最低）
- amp_ratio = 1.027（接近 1）
- vs EXP-025（SF AdamW 5k, KE 0.121）改善 -20%

### Discussion

1. **EXP-026 SOAP alone 略優於 SF AdamW（KE 0.124 vs EXP-024 0.150）**：但 amp 0.925 偏低、ens 0.152 偏高，整體優勢不明顯。
2. **EXP-027 取消的原因**：先做了 EXP-028（SOAP+SF）發現組合更佳，故 EXP-027（純 SOAP resume）無有效訓練結果。
3. **EXP-028 SOAP+SF 疊加成功**：比 EXP-026（SOAP only）KE 持平、ens -12%、amp 1.039 超過 1.0；確認 SOAP 二階曲率 + SF Polyak 平均雙效。
4. **EXP-029 加深 CfC 有副作用**：KE 0.111 為 3k 步最低（優於所有 1-layer），但 amp 0.759 是所有 1-layer 中最低。CfC 深度與能量幅值重建有 trade-off，需延長訓練或加層間殘差才能評估。
5. **EXP-030 為當前主線**：5k 步下 SOAP+SF 全面優於 SF AdamW（KE -20%, u_rmse -21%）。Ens 11.85% 較 EXP-025 10.93% 略差，這是唯一退步指標；其他全面改善。
6. **重要 bug 修正**：evaluate_deeponet_cfc.py 早期版本只處理 `state["model"]` key，但訓練腳本使用 `state["model_state_dict"]`，導致 EXP-026/028/029 早期評估顯示 KE ~97%（廢值）。已修正為優先讀 `model_state_dict`。

### Configs / Artifacts

| ID | Config | Artifact |
|---|---|---|
| EXP-026 | `[CONTEXT_MISSING]` | `[CONTEXT_MISSING]` |
| EXP-027 | — | — （取消）|
| EXP-028 | `[CONTEXT_MISSING]` | `[CONTEXT_MISSING]` |
| EXP-029 | `[CONTEXT_MISSING]`（2-layer TemporalCfC） | `[CONTEXT_MISSING]` |
| EXP-030 | [`configs/exp_030_re1000_soap_sf_5k.toml`](../configs/exp_030_re1000_soap_sf_5k.toml) | [`artifacts/deeponet-cfc-re1000-soap-sf-5000`](../artifacts/deeponet-cfc-re1000-soap-sf-5000) |

EXP-030 evaluated checkpoint: `artifacts/deeponet-cfc-re1000-soap-sf-5000/checkpoints/picon_kolmogorov_step_5000.pt`

### Supersedes / Superseded_By

- **Supersedes**: G6（EXP-025, KE 0.121 → 0.0961）
- **Superseded_By**: 無（Re=1000 主線結案於 EXP-030）
- **永久結論**：SOAP + Schedule-Free 為 Re=1000 sparse-data PINN 的最佳組合；後續 Re=10000 系列（G14）基於此改 LearnableFourier + GradNorm

---

## [GROUP G8] Re=10000 新資料容量基準 + physics loss 機制變更失敗（EXP-031 ~ EXP-033, EXP-035 ~ EXP-039）

- **Status**: `RESOLVED`（容量結論定，physics 改動全部證偽）
- **Time**: 2026-04-08 ~
- **Topic**: 切換新 DNS（si100, 201 frames）後的容量基準確立 + physics loss 設計探索

### Hypothesis

1. 新 DNS（dt=0.025, 201 frames vs 舊 dt=0.125, 41 frames）能解決 G4 的時間稀疏瓶頸（EXP-031 假設）。
2. 加深 CfC（1→2 layer）改善 KE（EXP-032，後證偽）。
3. 擴大容量（d=128 → 256, op_rank=256）優於增加深度（EXP-033 假設）。
4. Chebyshev collocation / residual normalization / 壓力 Poisson 約束可改善 physics 訊號品質（EXP-035~039）。

### 共同設定

- 新 DNS：si100, dt=0.025, 201 frames, T=5.0
- AdamW Schedule-Free 或 SOAP+SF（EXP-031~033 為 SOAP+SF）
- 3000 steps 從頭訓練
- u,v-only sensor supervision

### 個別實驗

#### 子組 8a: 容量基準（EXP-031~033）

| ID | 改動 vs prev | KE | Status |
|---|---|---:|---|
| EXP-031 | d=128, fourier_h=16, 1-layer attn | 0.394 | `ACTIVE_REFERENCE`（新資料基準）|
| EXP-032 | d=128, **2-layer CfC + 層間殘差** | 0.551 | `NEGATIVE_RESULT` |
| EXP-033 | d=256, op_rank=256, 1-layer | **0.315** | `ACTIVE_REFERENCE`（容量加倍最佳）|

EXP-033 額外指標：ens 0.489, amp 0.875；參數量 3.09M。

#### 子組 8b: Physics loss 機制變更（EXP-035~039）

| ID | 改動 vs EXP-031 | KE | Status |
|---|---|---:|---|
| EXP-035 | chebyshev-256 + residual normalize | 0.661 | `NEGATIVE_RESULT` |
| EXP-036 | normalize only | 0.867 | `NEGATIVE_RESULT` |
| EXP-037 | chebyshev only | 0.625 | `NEGATIVE_RESULT` |
| EXP-038 | + Poisson weight=1.0 | 0.560 | `NEGATIVE_RESULT` |
| EXP-039 | + Poisson weight=0.1 | 0.416 | `NEGATIVE_RESULT` |

### Discussion

1. **EXP-031 確立新資料基準**：KE 39.4% 為 d=128 首次基準；新 DNS 把時間稀疏問題解決，但 G4 的 K=100 sparse 限制仍在。
2. **EXP-032 證偽「加深 CfC」假設**：KE 55.1% > EXP-031（39.4%），增加 CfC 深度反而退步；層間殘差不足以補救 d=128 的容量瓶頸。
3. **EXP-033 確認「擴大寬度」是正確方向**：d=128→256 + op_rank 同步加倍，KE 31.5%（-7.9pp）；3.09M params 雖大，但物理可行性遠優於 d=128 + 2-layer。後續 Re=10000 主線統一採用 d=256。
4. **G8b physics loss 變更全部失敗**：
   - 結構化 collocation（chebyshev）：EXP-035/037 KE 66/63%，遠差於 EXP-031（39%）；確認均勻採樣已足夠。
   - 殘差正規化（normalize）：EXP-036 KE 87%，最差；正規化破壞物理量級資訊。
   - 壓力 Poisson 約束：EXP-038/039 KE 56/42%，無權重可改善；**確認壓力自由度不是 Re=10000 的主要瓶頸**。
5. **永久結論**：在 K=100 sparse sensors（覆蓋 k≤5）的資訊量限制下，**改善 physics loss 設計無法突破資訊理論上限**；下一步應從資料密度（K↑）或架構表達力（容量、Fourier embedding）著手。

### Configs / Artifacts

| ID | Config | Artifact |
|---|---|---|
| EXP-031 | 基於 EXP-008 wide-v2 設定 | `[CONTEXT_MISSING]` `deeponet-cfc-re10000-wide-v2-3000`（已刪除）|
| EXP-032 | `[CONTEXT_MISSING]` | `[CONTEXT_MISSING]`（已刪除）|
| EXP-033 | `[CONTEXT_MISSING]` | `[CONTEXT_MISSING]`（已刪除）|
| EXP-035~039 | `[CONTEXT_MISSING]` | `[CONTEXT_MISSING]`（已刪除）|

### Supersedes / Superseded_By

- **Supersedes**: G4（EXP-019/020 舊資料 Wide d=128, KE ~54%）
- **Superseded_By**: G10（EXP-043, 048 漸進式 d=256 → KE 21.8%）；physics loss 結論被 G14 sensor continuity 部分修正
- **永久結論（被列入 [STATE] Rejected Directions）**：Chebyshev / normalize / Poisson 在 K=100 sparse 設定下均無效

---

## [GROUP G9] Re=10000 transfer learning 失敗（EXP-040 ~ EXP-042）

- **Status**: `RESOLVED`（transfer 路徑證偽）
- **Time**: 2026-04-09
- **Topic**: 嘗試從 Re=1000 weights transfer 到 Re=10000，驗證 inductive bias 遷移可行性

### Hypothesis

1. EXP-030（Re=1000 d=64）的 inductive bias 可加速 Re=10000 收斂（EXP-040 假設）。
2. 即使 source 未完全收斂（EXP-041 KE 24.5%），仍可提供有用 inductive bias（EXP-042 假設）。

### 共同設定

- target: Re=10000, d=128, fourier_h=16, 2-layer attn（EXP-031 wide-v2 架構）
- optimizer 從頭啟動（不延續 source 的 optimizer state）

### 個別實驗

| ID | 步驟 | 結果 | Status |
|---|---|---|---|
| EXP-040 | 從 EXP-030（d=64, h=8, 1-layer）transfer 到 d=128/h=16/2-layer | `RuntimeError: size mismatch for spatial_encoder.base_norm.weight: shape [34] vs [66]` + missing keys for `token_blocks.1` | `NEGATIVE_RESULT`（架構不匹配）|
| EXP-041 | 先在 Re=1000 訓練 d=128 wide 架構（同 EXP-031）3k 步 | KE 24.5%, amp 0.950, phase -0.503 rad（顯著差於 EXP-030 KE 9.61%；d=128 在 Re=1000 + 3k 步未收斂）| `ACTIVE_REFERENCE`（作為 EXP-042 source）|
| EXP-042 | 從 EXP-041 transfer 至 Re=10000 wide-v2, SOAP+SF 3k | KE **0.402**（差於 EXP-031 隨機初始化 0.394）、amp 0.686、phase 0.282 rad | `NEGATIVE_RESULT`（負遷移）|

### Discussion

1. **EXP-040 揭示 transfer 硬約束**：source/target 架構必須完全相同（d_model, fourier_harmonics, num_attn_layers 都不能變）；spatial_encoder 維度從 34 變 66 直接 size_mismatch。
2. **EXP-041 為驗證用 source**：EXP-030（d=64）達 KE 9.61%，但 d=128 wide 架構在 Re=1000 + 3000 步只到 24.5%；超參化導致更大架構在短訓練下未收斂，weights 品質次佳但仍可作為 transfer 出發點。
3. **EXP-042 證偽「即使 source 不完美仍有 inductive bias 收益」**：transfer 後 KE 40.2% 略差於隨機初始化（39.4%），amp 從 0.950 → 0.686 大幅退化；**確認 source 必須充分收斂才有正向遷移**。EXP-041 的 24.5% 品質不足以提供有用 prior。
4. **永久結論（被列入 [STATE] Rejected Directions）**：
   - Transfer 需 source/target 架構完全一致
   - Source 品質不足會產生負遷移，差於隨機初始化

### Configs / Artifacts

| ID | Config | Artifact |
|---|---|---|
| EXP-040 | [`configs/exp_040_re10000_transfer.toml`](../configs/exp_040_re10000_transfer.toml) | — （載入失敗）|
| EXP-041 | [`configs/exp_041_re1000_wide.toml`](../configs/exp_041_re1000_wide.toml) | [`artifacts/deeponet-cfc-re1000-wide-3000`](../artifacts/deeponet-cfc-re1000-wide-3000) |
| EXP-042 | [`configs/exp_042_re10000_transfer_wide.toml`](../configs/exp_042_re10000_transfer_wide.toml) | [`artifacts/deeponet-cfc-re10000-transfer-wide-3000`](../artifacts/deeponet-cfc-re10000-transfer-wide-3000) |

### Supersedes / Superseded_By

- **Supersedes**: 無（與 G8 並行的探索方向）
- **Superseded_By**: G10（EXP-043 從 EXP-033 隨機初始化漸進式訓練達 KE 27.2%）

---

## [GROUP G10] Re=10000 d=256 漸進收斂線（EXP-043, EXP-048）

- **Status**: `SUPERSEDED`（被 G14 EXP-064 取代）
- **Time**: 2026-04-13 ~ 04-15
- **Topic**: d=256 架構下逐步延長訓練（3k → 5k → 10k）的 KE 收斂軌跡

### Hypothesis

1. EXP-033（KE 31.5%）在 3k 未飽和，resume 至 5k 應持續收益（EXP-043）。
2. EXP-043 在 5k 仍未飽和（L_data 仍下降），延至 10k 持續改善（EXP-048）。

### 共同設定

- d=256, op_rank=256, fourier_h=16, 1-layer attn
- SOAP + Schedule-Free, lr=1e-3
- 新 DNS si100

### 個別實驗

| ID | 步驟 | KE | Ens | amp | phase | t=0 KE 估計 | u_rmse | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| EXP-033 | 從頭 3k（G8）| 0.315 | 0.489 | 0.875 | — | — | — | (G8) |
| EXP-043 | resume EXP-033 → 5k | 0.272 | — | 0.931 | -0.025 rad | — | — | `ACTIVE_REFERENCE` |
| EXP-048 | resume EXP-043 → 10k | **0.218** | 0.437 | 0.899 | 0.039 rad | **58%（嚴重低估）** | 0.106 | `ACTIVE_REFERENCE`（被超越）|

EXP-048 視覺診斷：
- 大尺度流場（低 k）重建良好，誤差場呈隨機高頻分佈
- 渦量峰值系統性低估 50%（DNS ±30 vs PI-CON ±15），spectral bias，sensors 僅覆蓋 k≤5
- 能譜 k<10 端與 DNS 高度吻合；k>20 端略低估
- KE 曲線：t=0 PI-CON=0.068 vs DNS=0.161（-58%），t>1 穩定在 DNS 的 85%

### Discussion

1. **3k→5k→10k 軌跡 KE 31.5%→27.2%→21.8%**：邊際遞減但持續改善。L_data 在 10k 步仍 9.78e-3（未飽和）。
2. **主要殘差來源**：
   - **t=0 KE 低估 58%**（初始條件重建不足）→ 後續 G12 EXP-055 IC weight 機制針對解決
   - **高 k 渦量結構 spectral bias**（K=100 資訊論硬上限）→ G14 wavelet 診斷確認
3. **永久結論**：漸進式訓練（多次 resume）的累積收益約 10pp KE，相當於架構或 loss 層面的顯著改動；任何「從頭訓練」的架構改動需先承受這個路徑代價（EXP-051 後來證實此點）。

### Configs / Artifacts

| ID | Config | Artifact |
|---|---|---|
| EXP-043 | `[CONTEXT_MISSING]` | `[CONTEXT_MISSING]` |
| EXP-048 | [`configs/exp_048_re10000_xlarge_10k.toml`](../configs/exp_048_re10000_xlarge_10k.toml) | [`artifacts/deeponet-cfc-re10000-xlarge-20000/deeponet-cfc-re10000-xlarge-10000`](../artifacts/deeponet-cfc-re10000-xlarge-20000/deeponet-cfc-re10000-xlarge-10000) |

### Supersedes / Superseded_By

- **Supersedes**: G8 子組 8a（EXP-033 為 EXP-043 的 source）
- **Superseded_By**: G12（EXP-055 IC weight resume EXP-048 → KE 17.1%, -4.7pp），G14（EXP-064, KE 7.80%, 取代為當前 baseline）

---

## [GROUP G11] Re=10000 d=256 從頭 3k 失敗群（EXP-044 ~ EXP-047）

- **Status**: `RESOLVED`（locality / sweep / GradNorm 全部證偽）
- **Time**: 2026-04-10 ~ 04-14
- **Topic**: d=256 架構下從頭 3000 步的多種改動探索（與 G10 漸進式線並行對照）

### Hypothesis

1. 加可學習距離衰減（locality decay）引入近鄰優先 inductive bias 可改善 KE（EXP-044）。
2. Optuna sweep（lr/cont_w/soap_freq/locality）找到的最佳參數可優於手調（EXP-045）。
3. GradNorm 自動平衡 data/physics task 梯度比例優於固定權重（EXP-046/047）。

### 共同設定

- d=256, op_rank=256, fourier_h=16, 1-layer attn
- 從頭訓練 3000 steps（vs G10 漸進式對照）
- 新 DNS si100

### 個別實驗

| ID | 改動 | KE | Ens | amp | phase | Status |
|---|---|---:|---:|---:|---:|---|
| EXP-044 | + log_locality_decay（可學習，初始 -2.0）| —（中止 ~500/3000）| — | — | — | `ARCHIVED_CONTEXT`（無收斂結論）|
| EXP-045 | sweep best: lr=4.75e-3, cont_w=0.509, soap_freq=20, locality=True | 0.354 | 0.502 | 0.866 | 0.096 rad | `NEGATIVE_RESULT` |
| EXP-046 | GradNorm 3-task [data, ns, cont]（初始等權）| 0.599 | 0.346 | 0.560 | -0.067 rad | `NEGATIVE_RESULT` |
| EXP-047 | GradNorm 4-task [data, ns_u, ns_v, cont]（初始 [1, 0.01, 0.01, 0.01]）| **0.721**（最差紀錄）| 0.858 | 0.339 | 0.009 rad | `NEGATIVE_RESULT` |

### Discussion

1. **EXP-044 因其他工作中止**：~500/3000 steps，架構正確但無收斂結論；可後續 resume 但優先級低。
2. **EXP-045 揭示 sweep 代理指標失敗**：sweep v2 以 1500-step l_data 作為代理指標，但 l_data@1500 與最終 KE 無相關性；cont_w=0.509 太弱導致散度約束被犧牲，KE 35.4% > EXP-043（27.2%, 漸進式）。
3. **EXP-046/047 GradNorm 兩種設定均失敗**：
   - 3-task: w_cont 持續下降至 0.750，散度約束被系統性壓制
   - 4-task: GradNorm 把 w_ns 推至 0.37，物理過強壓制資料；KE 72.1% 為全實驗最差
   - **GradNorm 等梯度範數目標 ≠ 物理可行性**；初始等權從物理上不合理的起點調整；NS x/y 混合為單一 task 無法感知 forcing 不對稱性
4. **G10 vs G11 對照**：EXP-048 漸進式 10k KE 21.8% vs EXP-046/047 從頭 3k KE 60~72%；確認**漸進式訓練路徑優於從頭訓練的單次大改動**。

### Configs / Artifacts

| ID | Config | Artifact |
|---|---|---|
| EXP-044 | [`configs/exp_044_re10000_xlarge_locality.toml`](../configs/exp_044_re10000_xlarge_locality.toml) | `artifacts/deeponet-cfc-re10000-xlarge-locality-3000`（部分）|
| EXP-045 | `[CONTEXT_MISSING]` `exp_045_re10000_xlarge_sweep_best.toml` | `[CONTEXT_MISSING]`（已刪除）|
| EXP-046 | `[CONTEXT_MISSING]`（已被 EXP-047 config 取代）| `[CONTEXT_MISSING]`（已刪除）|
| EXP-047 | [`configs/exp_047_re10000_xlarge_gradnorm4.toml`](../configs/exp_047_re10000_xlarge_gradnorm4.toml) | `artifacts/deeponet-cfc-re10000-xlarge-gradnorm4-3000` |

### Supersedes / Superseded_By

- **Supersedes**: 無（與 G10 並行）
- **Superseded_By**: G14（EXP-063 GradNorm 在 LearnableFourier 架構 + 正確初始權重下成功）
- **永久結論**：在 sparse-data PINN + 從頭 3k 設定下，GradNorm 與 sweep 代理指標均無效

---

## [GROUP G12] Re=10000 EXP-048 resume 變體探索（EXP-049 ~ EXP-056）

- **Status**: `RESOLVED`（IC weight 為主要正向結論，已被 G14 超越）
- **Time**: 2026-04-15 ~ 04-21
- **Topic**: 以 EXP-048（KE 21.8%, d=256 漸進式 10k）為起點，並行測試 K=200/curriculum/L-BFGS/RAR/IC weight 等多項改動

### Hypothesis

1. K=200 sensor 在頻譜覆蓋 acc>0.8 的 k_cutoff 從 20 提升至 41，可改善 KE（EXP-049 假設）。
2. Physics collocation curriculum（8→128 點）可在 resume 後改善物理可行性（EXP-050 假設）。
3. fourier_h=20 + t0_boost×3 可同時改善 IC 與高頻覆蓋（EXP-051 假設）。
4. L-BFGS 二階曲率資訊在 EXP-048 收斂局部盆地中可細調（EXP-052 假設）。
5. RAR（Residual Adaptive Refinement）偏向高殘差區域可改善 KE（EXP-053/054）。
6. IC Loss Weight（λ=10, t≤0.05）強制學習 t=0 重建可解決 t=0 KE 58% 低估（EXP-055）。
7. RAR + IC weight 組合產生加性收益（EXP-056）。

### 共同設定

- Resume from EXP-048 step_9500（d=256, KE 21.8%）→ +10000 steps → step 19500
- SOAP + Schedule-Free, fourier_h=16
- 評估腳本 v2（2026-04-20 修正 Re 正規化 + cell-center grid + spectrum k-axis）

### 個別實驗

| ID | 改動 vs EXP-048 | KE | Ens | div_l2 | amp | phase | Status |
|---|---|---:|---:|---:|---:|---:|---|
| EXP-049 | K=200 sensor，**從頭** 10k（注意：與 EXP-048 的漸進式對照不公平）| 0.439 | 0.518 | — | 0.084 | -0.190 | `NEGATIVE_RESULT` |
| EXP-050 | + physics curriculum（8→128 點）| 0.259 | 0.474 | — | **0.213**（崩潰）| 0.650 | `NEGATIVE_RESULT` |
| EXP-051 | fourier_h 16→20 + t≤0.1 weight×3，**從頭** 10k | 0.278 | 0.490 | — | 0.670 | 0.004 rad | `NEGATIVE_RESULT` |
| EXP-052 | L-BFGS 100 步（lr=0.1, max_iter=20, time_marching=false）| 0.241 | — | — | 0.889 | — | `NEGATIVE_RESULT`（22h/2000 步不可行）|
| EXP-053 | + RAR freq=50 | 0.252 | 0.487 | — | 0.878 | — | `NEGATIVE_RESULT`（L_phys 7.96→19.27 爆漲）|
| EXP-054 | + RAR freq=1000 | **0.196** | 0.364 | — | **0.961** | 0.064 rad | `POSITIVE_RESULT`（突破 -2.2pp）|
| EXP-055 | + IC weight λ=10（t≤0.05）| **0.171** | 0.384 | 1.612 | **0.970** | 0.034 rad | `POSITIVE_RESULT`（突破 -4.7pp，當期最佳）|
| EXP-056 | RAR freq=1000 + IC weight | 0.194 | 0.363 | — | 0.955 | 0.084 rad | `NEGATIVE_RESULT`（負干擾）|

EXP-055 額外指標：`ek_ratio_kf_last = 0.934`（全系列最佳）、`L_data 4.83e-3, L_phys 9.24e-1`。

### Discussion

1. **EXP-049 confounded**：K=200 從頭 10k（KE 43.9%）vs K=100 漸進式 10k（21.8%）不可直接比較；K=200 從頭 10k 甚至差於 K=100 從頭 3k（31.5%）。**確認瓶頸不在 sensor 頻譜覆蓋**，而在訓練收斂或架構容量。
2. **EXP-050 揭示 SOAP+SF resume 限制**：preconditioner + Polyak 平均無法從 checkpoint 重建，curriculum 重啟 time-marching warmup 破壞 EXP-048 的時域泛化；amp_ratio 0.927 → 0.213 崩潰。
3. **EXP-051 揭示路徑代價**：從頭 10k（KE 27.81%）反而差於 EXP-048 漸進式 10k（21.8%）；hr=20 + t0_boost 的潛在改善被「從頭 vs 漸進」混淆變數覆蓋。**任何需要從頭訓練的架構改動需先承受 ~10pp KE 路徑代價**。
4. **EXP-052 L-BFGS 證偽**：100 步耗時 68 分鐘（41 s/step），2000 步需 22h；KE 24.07% > 21.8% 退步方向明確。L-BFGS 對 stochastic mini-batch + 3M params 的 PINN 益處有限，計算代價是 SOAP 的 20×。
5. **EXP-053/054 揭示 RAR freq 是關鍵參數**：
   - freq=50：L_phys 7.96→19.27 爆漲，L_data 1.20e-2→2.64e-2 上升；SOAP+SF preconditioner 無法跟上每 50 步的 collocation 變動
   - freq=1000：模型在固定 collocation 上充分收斂後再重新採樣，KE 19.6%（突破 -2.2pp）
6. **EXP-055 IC weight 為主要正向結論**：λ=10 在 t≤0.05 加權，KE 從 21.8% → 17.1%（-4.7pp），優於 RAR alone（-2.2pp）。kf_amp_ratio 0.970 與 E(k_f) 0.934 全系列最佳；強制學習 t=0 IC 連帶改善 forcing mode 重建。
7. **EXP-056 揭示組合干擾**：KE 19.4% > IC alone（17.1%）。L_data 最終 2.59e-3 為全系列最低，但 KE 反而更高；RAR 每 1000 步重新分配 collocation（改變 loss landscape），與 IC weight 依賴的穩定梯度方向衝突。**RAR + IC weight 不能同時用**。

### Configs / Artifacts

| ID | Config | Artifact |
|---|---|---|
| EXP-049 | [`configs/exp_049_re10000_xlarge_k200.toml`](../configs/exp_049_re10000_xlarge_k200.toml) | [`artifacts/deeponet-cfc-re10000-xlarge-k200-10000`](../artifacts/deeponet-cfc-re10000-xlarge-k200-10000) |
| EXP-050 | [`configs/exp_050_re10000_xlarge_20k.toml`](../configs/exp_050_re10000_xlarge_20k.toml) | [`artifacts/deeponet-cfc-re10000-xlarge-20000`](../artifacts/deeponet-cfc-re10000-xlarge-20000) |
| EXP-051 | [`configs/exp_051_re10000_xlarge_harmonics20_t0boost.toml`](../configs/exp_051_re10000_xlarge_harmonics20_t0boost.toml) | [`artifacts/deeponet-cfc-re10000-exp051`](../artifacts/deeponet-cfc-re10000-exp051) |
| EXP-052 | [`configs/exp_052_re10000_xlarge_lbfgs.toml`](../configs/exp_052_re10000_xlarge_lbfgs.toml) | [`artifacts/deeponet-cfc-re10000-exp052-lbfgs`](../artifacts/deeponet-cfc-re10000-exp052-lbfgs) |
| EXP-053 | [`configs/exp_053_re10000_xlarge_rar.toml`](../configs/exp_053_re10000_xlarge_rar.toml) | [`artifacts/deeponet-cfc-re10000-exp053-rar`](../artifacts/deeponet-cfc-re10000-exp053-rar) |
| EXP-054 | [`configs/exp_054_re10000_xlarge_rar_1k.toml`](../configs/exp_054_re10000_xlarge_rar_1k.toml) | [`artifacts/deeponet-cfc-re10000-exp054-rar-1k`](../artifacts/deeponet-cfc-re10000-exp054-rar-1k) |
| EXP-055 | [`configs/exp_055_re10000_xlarge_rar_ic.toml`](../configs/exp_055_re10000_xlarge_rar_ic.toml) | [`artifacts/deeponet-cfc-re10000-exp055-ic`](../artifacts/deeponet-cfc-re10000-exp055-ic) |
| EXP-056 | [`configs/exp_056_re10000_xlarge_rar_ic.toml`](../configs/exp_056_re10000_xlarge_rar_ic.toml) | [`artifacts/deeponet-cfc-re10000-exp056-rar-ic`](../artifacts/deeponet-cfc-re10000-exp056-rar-ic) |

### Supersedes / Superseded_By

- **Supersedes**: G10 EXP-048（resume 起點）
- **Superseded_By**: G14（EXP-064 KE 7.80%，combine LearnableFourier + GradNorm + sensor continuity）
- **永久結論**：
  - IC weight 是 sparse-data PINN 解 t=0 IC 重建問題的主要工具
  - RAR freq ≥ 1000 才能與 SOAP+SF 共存
  - RAR + IC weight 不可同時使用
  - L-BFGS 在此規模 stochastic 設定下不適用

---

## [GROUP G13] Re=10000 冷啟動 IC weight 系列（EXP-057 ~ EXP-061）

- **Status**: `RESOLVED`（CfC freerun / 雙向 CfC / h32 / jaxpi 對齊單獨上場全部證偽）
- **Time**: 2026-04-21 ~ 04-22
- **Topic**: 在 G12 確認 IC weight 有效後，從頭實驗各種 t=0 重建與架構/optimizer 改動

### Hypothesis

1. IC weight 在冷啟動下仍有效（不依賴 EXP-048 的 warm state）（EXP-057）。
2. CfC 自由積分（freerun）可改善時序連貫性（EXP-058，後因診斷推翻）。
3. 雙向 CfC 使 h_states[t=0] 同時看到未來觀測，消除因果編碼資訊不對稱（EXP-059）。
4. fourier_h 16→32 擴大 trunk 頻率覆蓋（EXP-060）。
5. jaxpi 對齊 SOAP（去 SF, betas=(0.9,0.999), warmup=2000）改善 L_phys 穩定性（EXP-061）。

### 共同設定

- 冷啟動，10000 steps（無 EXP-048 warm state）
- d=256, fourier_h=16（EXP-060 為 32）
- IC weight λ=10（t≤0.05）作為基礎（EXP-059 改用 t_early_weight=1.0 以隔離效果）
- num_physics_points=64（EXP-061）/ 32（其他）

### 個別實驗

| ID | 改動 vs EXP-057 | KE | Ens | div_l2 | amp | phase | t=0 KE | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| EXP-057 | baseline（IC weight + 冷啟動）| **0.206** | 0.413 | **0.796**（最佳）| 0.981 | 0.008 rad | 55.5% | `COMPLETED`（IC weight 在冷啟動有效）|
| EXP-058 | + use_cfc_freerun=true | — | — | — | — | — | — | `PAUSED`（hypothesis 被 EXP-057 時序診斷推翻：誤差遞減非遞增）|
| EXP-059 | + use_bidirectional_cfc, t_early_weight=1.0 | 0.191 | 0.404 | 0.701 | 1.009 | 0.031 rad | **60.4%（惡化）** | `NEGATIVE_RESULT` |
| EXP-060 | + fourier_h=32 | **0.976**（崩潰）| — | 4.523 | 0.140 | 1.049 rad | — | `NEGATIVE_RESULT`（L_phys 週期爆漲）|
| EXP-061 | + jaxpi SOAP（去 SF, betas=(0.9,0.999), warmup 2000）| 0.294 | 0.452 | 0.835 | 0.905 | 0.029 rad | — | `NEGATIVE_RESULT`（去 SF 退步 +8.8pp）|

EXP-060 訓練過程：step 3000 L_phys=59.5, step 6000=517.6, step 9000=95.2（週期爆漲）。

### Discussion

1. **EXP-057 確認 IC weight 在冷啟動有效**：KE 20.6% vs EXP-051（冷啟動無 IC, 27.81%）改善 7.2pp；vs EXP-055（resume + IC, 17.1%）差 3.5pp（warm state 額外貢獻）。div 0.796 全系列最佳，phase 0.008 rad 全系列最低；冷啟動不繼承 resume 的方向偏誤。
2. **EXP-058 自我推翻**：分析 EXP-057 時序數據後發現誤差**隨時間遞減**（t=0: KE 55.5% → t=5: KE 9.3%），代表模型隨觀測累積而改善，非因時序連貫性缺失而惡化。方案 B 針對的問題不存在；移除相關 freerun_gate/freerun_value 模組。
3. **EXP-059 雙向 CfC 證偽**：mean KE 微改（19.1%, -1.5pp），但 t=0 KE 60.4%（**惡化 +4.9pp**）；去除 IC weight 後 t=0 反而更差。**根因確認：t=0 問題核心是訓練訊號（IC weight）不足，而非資訊存取（因果編碼）不足**。同步診斷 band_high 87% 根因為 fourier_h=16 缺乏 k=17..32 基函數。
4. **EXP-060 h32 崩潰**：trunk input 從 96 → 160 後，SOAP+SF 在高頻方向曲率估計失效；betas=(0.95, 0.95) 過激進 + precond_freq=10 太慢 + 無 LR warmup。L_phys 週期爆漲，ScheduleFree Polyak 平均累積壞歷史後收斂至 near-zero（KE 97.6%）。
5. **EXP-061 jaxpi SOAP 對齊不足以單獨工作**：去除 SF（betas=0.9/0.999, warmup 2000, step decay）後 KE 退步至 29.4%（vs EXP-057 20.6%, +8.8pp）。**確認 SF Polyak 平均對 Re=10000 chaotic flow 是不可或缺的**；jaxpi 參數本身無法替代。

### Configs / Artifacts

| ID | Config | Artifact |
|---|---|---|
| EXP-057 | [`configs/exp_057_re10000_xlarge_cold_ic.toml`](../configs/exp_057_re10000_xlarge_cold_ic.toml) | [`artifacts/deeponet-cfc-re10000-exp057-cold-ic`](../artifacts/deeponet-cfc-re10000-exp057-cold-ic) |
| EXP-058 | [`configs/exp_058_re10000_xlarge_cfc_freerun.toml`](../configs/exp_058_re10000_xlarge_cfc_freerun.toml) | — （暫停）|
| EXP-059 | [`configs/exp_059_re10000_xlarge_bidir_cfc.toml`](../configs/exp_059_re10000_xlarge_bidir_cfc.toml) | [`artifacts/deeponet-cfc-re10000-exp059-bidir-cfc`](../artifacts/deeponet-cfc-re10000-exp059-bidir-cfc) |
| EXP-060 | [`configs/exp_060_re10000_xlarge_harmonics32.toml`](../configs/exp_060_re10000_xlarge_harmonics32.toml) | `artifacts/deeponet-cfc-re10000-exp060-harmonics32` |
| EXP-061 | [`configs/exp_061_re10000_xlarge_soap_aligned.toml`](../configs/exp_061_re10000_xlarge_soap_aligned.toml) | [`artifacts/deeponet-cfc-re10000-exp061-soap-aligned`](../artifacts/deeponet-cfc-re10000-exp061-soap-aligned) |

### Supersedes / Superseded_By

- **Supersedes**: G12（EXP-055 IC weight 為冷啟動實驗的基礎假設驗證）
- **Superseded_By**: G14（EXP-062 LearnableFourierEmb 後不再需要 fourier_h 手調 + jaxpi SOAP 在新架構下成功）

---

## [GROUP G14] Re=10000 LearnableFourier 演進（EXP-062 ~ EXP-063 → EXP-064 為主檔 baseline）

- **Status**: `ACTIVE`（演進至主檔 EXP-064 為當前 Re=10000 baseline）
- **Time**: 2026-04-23 ~ 04-24
- **Topic**: 引入 LearnableFourierEmb 取代固定 periodic Fourier，搭配正確 jaxpi SOAP + GradNorm，建立 K=100 結案值

### Hypothesis

1. 可學習 Fourier 投影（embed_dim=128, init σ=2.0）讓模型自適應覆蓋 k>16，band_high_last 顯著低於 EXP-057（~87%）（EXP-062）。
2. 正確 jaxpi SOAP 對齊（保留 SF, betas=(0.9,0.999), precond_freq=2, wd=0, decay=2000）+ GradNorm（freq=1000, momentum=0.9）改善 KE 與 div（EXP-063）。

### 共同設定

- 冷啟動 10000 steps，d=256, op_rank=256, 1-layer attn
- LearnableFourierEmb（embed_dim=128, init σ=2.0），spatial dim 64→128
- u,v-only sensor supervision

### 個別實驗

| ID | 改動 vs prev | KE | Ens | div_l2 | amp | phase | band_low/mid/high@last | Status |
|---|---|---:|---:|---:|---:|---:|---|---|
| EXP-062 | LearnableFourierEmb（128, σ=2）+ jaxpi 設定（precond_freq=5, wd=1e-4, decay=1000）| **0.104** | 0.322 | 0.571 | 0.949 | -0.063 rad | 5.8% / 99.8% / 99.98% | `POSITIVE_RESULT`（KE 紀錄）|
| EXP-063 | + use_schedule_free=true, precond_freq=2, wd=0, decay=2000, GradNorm（freq=1000, mom=0.9, init [1, 0.01, 0.01, 0.01]）| **0.0865** | 0.304 | **0.204**（-64%）| 0.9636 | -0.0489 rad | 5.0% / 99.97% / 100% | `POSITIVE_RESULT`（KE 紀錄）|
| EXP-064 | + sensor continuity（n_t=1, start=1000）| **0.0780** | 0.291 | **0.184** | 0.9615 | -0.0228 rad | 3.6% / 99.97% / 100% | `ACTIVE_BASELINE`（**主檔 RECORD**）|

EXP-062/063 額外指標：
- EXP-062: u_rmse 0.0809, v_rmse 0.0744, worst_u_rmse 0.194 @ t=0
- EXP-063: u_rmse 0.0709 (-12% vs EXP-062), v_rmse 0.0636, worst_u_rmse 0.1646 @ t=0

EXP-063 額外修正：`ScheduleFreeWrapper` 非 `torch.optim.Optimizer` 子類，LR scheduler 改綁 `base_optimizer`。

### Discussion

1. **EXP-062 確認可學習 Fourier 顯著優於固定 periodic**：KE 10.4%（vs EXP-055 17.1%, -6.7pp），是 Re=10000 當時新紀錄。但 hypothesis 部分 falsified：
   - band_high_last=99.98% 未改善（hypothesis 期望 <87%）
   - KE 改善源自**低頻精度大幅提升**（band_low@t=5: 5.8%），非頻率覆蓋擴展
   - 模型收斂至低頻能量主導解，中高頻能量近乎全滅（spectral 結構不完整）
2. **EXP-063 GradNorm 自動強化 continuity**：KE 8.65%（-1.75pp）；div_l2 0.204（-64% vs EXP-062 0.571）為全系列最低；u_rmse 全時段 -12%；t=0 worst_rmse -15%。**確認 GradNorm 在 LearnableFourier 架構 + 正確初始權重下成功**（vs G11 GradNorm 在 d=128 失敗）。但 band_mid/high@t=5 仍 ≈100%，**確認 K=100 感測器對 k>5 模態的覆蓋不足為資訊論硬上限**，非 optimizer/架構問題。
3. **EXP-064（主檔）sensor continuity 完成最終結案值**：KE 7.80%（-0.85pp）；div_l2 0.184（-9.6%, 全系列最低）；phase_err -0.0228 rad（-53% vs EXP-063）。band_mid/high@t=5 仍 ≈100% → hypothesis falsified。**確認 K=100 配置已達資訊論硬上限**（後由 G analysis_reports 的 wavelet diagnostic 量化）。
4. **永久結論**：LearnableFourier + 正確 jaxpi SOAP + GradNorm + sensor continuity 是 K=100 配置的最終形式；後續提升需 K↑（EXP-066 探索）或 DNS 高頻先驗（工程不可遷移）。

### Configs / Artifacts

| ID | Config | Artifact |
|---|---|---|
| EXP-062 | [`configs/exp_062_re10000_xlarge_learnable_fourier.toml`](../configs/exp_062_re10000_xlarge_learnable_fourier.toml) | [`artifacts/deeponet-cfc-re10000-exp062-learnable-fourier`](../artifacts/deeponet-cfc-re10000-exp062-learnable-fourier) |
| EXP-063 | [`configs/exp_063_re10000_xlarge_soap_gradnorm.toml`](../configs/exp_063_re10000_xlarge_soap_gradnorm.toml) | `artifacts/deeponet-cfc-re10000-exp063-soap-gradnorm` |
| EXP-064 | 見主檔 RECORD | 見主檔 RECORD |

### Supersedes / Superseded_By

- **Supersedes**: G12（EXP-055 KE 17.1% → EXP-062 10.4%, -6.7pp）；G13（EXP-061 jaxpi SOAP 失敗 → EXP-063 jaxpi SOAP 成功，差別在保留 SF 與正確初始權重）
- **Superseded_By**: 無（K=100 配置結案）；EXP-066 K=200 探索屬另一資料密度配置，不取代 EXP-064
- **永久結論**：
  - LearnableFourierEmb 取代固定 periodic Fourier 為 Re=10000 標配
  - GradNorm 需在「正確初始權重 [1, 0.01, ...]」+「LearnableFourier」雙重條件下才能正向工作
  - K=100 KE 7.80% 為資訊論硬上限下的最佳可達值

---

> **完整 GROUP 索引見本檔最上方 [GROUP INDEX] 表格。**
>
> **更晚實驗**（EXP-064 ~ EXP-101 含 K=100 結案、AL series、6-lever pivot、multi-seed、benchmark）已搬到 [`docs/experiment_archive_kolmogorov_post_k100.md`](experiment_archive_kolmogorov_post_k100.md)（2026-05-15 拆出）。
>
> **量化分析報告**（Wavelet Sparsity / AIM / 早期 Physics Denorm Diagnostic）：見 [`docs/analysis_reports.md`](analysis_reports.md)。
>
> **CFD-rigour validation / silent regression 細節**（Q5/Q7/Q8、Forward CFD baseline、physics denorm bug 翻轉）：見 [`docs/diagnostics_log.md`](diagnostics_log.md)。
