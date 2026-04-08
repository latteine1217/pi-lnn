# 實驗紀錄

本文件是本 repo 的實驗 state 檔，不是 protocol。

只記錄目前 repo 內可核對的 `artifact / config / summary.json / checkpoint`，用於：

- 快速回答目前主線是什麼
- 判斷哪些方向已被支持、證偽或取代
- 讓 agent 在續跑或比較前先自讀，不靠記憶腦補

---

## [SCHEMA]

### 欄位定義

- `ID`: 穩定實驗編號，供後續引用
- `Status`:
  - `ACTIVE_BASELINE`: 當前主基準
  - `ACTIVE_REFERENCE`: 仍有效的對照或關鍵依據
  - `NEGATIVE_RESULT`: 已證偽或明確負收益
  - `ARCHIVED_CONTEXT`: 保留背景脈絡，但已被更新主線取代
- `Decision`: 這筆紀錄最後支撐的結論
- `Supersedes / Superseded_By`: 用於追蹤哪條線已被後續結果覆蓋

### 讀取建議

1. 先看 `## [INDEX] Active`
2. 再看 `## [STATE] Current Baseline`
3. 若要判斷某改動是否已被做過，再看 `## [INDEX] Negative` 與對應紀錄
4. 若仍不足，再往下讀詳細 `## [RECORD]`

---

## [STATE] Data Version

### 資料條件

- domain: `[0, 1]^2`
- DNS:
  [`/Users/latteine/Documents/coding/pi-lnn/data/kolmogorov_dns_fp64_etdrk4_Re1000_N128_T5_ds4.npy`](/Users/latteine/Documents/coding/pi-lnn/data/kolmogorov_dns_fp64_etdrk4_Re1000_N128_T5_ds4.npy)
- sensors:
  [`/Users/latteine/Documents/coding/pi-lnn/data/kolmogorov_sensors/re1000/sensors_qrpivot_K100_N128_t0-5.json`](/Users/latteine/Documents/coding/pi-lnn/data/kolmogorov_sensors/re1000/sensors_qrpivot_K100_N128_t0-5.json)
- sensor values:
  [`/Users/latteine/Documents/coding/pi-lnn/data/kolmogorov_sensors/re1000/sensors_qrpivot_K100_N128_t0-5_dns_values.npz`](/Users/latteine/Documents/coding/pi-lnn/data/kolmogorov_sensors/re1000/sensors_qrpivot_K100_N128_t0-5_dns_values.npz)

---

## [STATE] Current Baseline

| 項目 | 現況 |
|---|---|
| Baseline ID | `EXP-030` |
| 主線 config | [`/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_small_soap_sf_5000.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_small_soap_sf_5000.toml) |
| train artifact | [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re1000-soap-sf-5000`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re1000-soap-sf-5000) |
| eval checkpoint | `artifacts/deeponet-cfc-re1000-soap-sf-5000/checkpoints/lnn_kolmogorov_step_5000.pt` |
| 目前判讀 | `SOAP + Schedule-Free` + `5000 steps`（EXP-028 resume）是目前最佳主線；首次突破 KE 10% 門檻 |
| 主要優勢 | KE rel-err **9.61%**（vs EXP-025 SF AdamW: 12.06%，**-20%**）、u RMSE **5.68e-2**（最低）、amp ratio **1.027** |
| 主要改變 | EXP-028 step 3000 resume → 5000 steps；SOAP 曲率估計 + Polyak 平均雙效帶來 KE 突破 |
| 主要已解問題 | t=3.5∼4.5 的 phase 高峰為 Re=1000 chaotic divergence 物理本質，非表徵問題 |

### 主線固定假設

- 觀測 supervision 僅使用 `u, v`
- physics 使用 primitive `momentum + continuity`
- 空間編碼：`periodic_fourier_encode`（`fourier_harmonics=8`，共 32 特徵），嚴格週期邊界
- `relpos_bias`：純距離輸入 `|rel|`（等向），不含方向向量
- `output_head_gain = 1`
- `use_temporal_anchor = true`（`n_harmonics=2`）：為 trunk 提供 `sin/cos(2π n t/T)` 絕對時間座標
- `Small` 尺寸已足夠進入穩定 regime
- `Re=1000` forcing mode 應為 `k_f = 2`
- `time_marching` 應保留
- 優化器主線：`SOAP + Schedule-Free`（`lr=3e-3`，`betas=(0.95,0.95)`，`precondition_freq=10`）

---

## [STATE] Supported Decisions

1. `u,v-only` sensor supervision 是必要前提。
2. physics 應回到 primitive momentum form，不應用錯誤 supervision 掩蓋尺度問題。
3. `rff_sigma=32 + output_head_gain=5` 會把導數與 residual 推到不可訓練量級。
4. `Small` 尺寸模型已足夠，不需要先加大模型。（在 Re=1000 下成立；Re=10000 見第 10 條）
5. `time_marching=true` 比直接全時段訓練更好。
6. `stepLR(500 x 0.9) + 3000 steps` 是目前最佳已驗證訓練策略。
7. `use_phase_anchor=true` 對 forcing mode amplitude（+27%）與 phase error（-71%）都有明顯改善；sensor 可觀測性分析排除了 data 本身看不到 phase 的假設。
8. `use_temporal_anchor=true`（`n_harmonics=2`）在 EXP-015 帶來 KE/Ens 各降 ~10%，但對 t=3.5∼4.5 phase 高峰無效；已確認該高峰為 Re=1000 chaotic divergence 物理本質。
9. t=3.5∼4.5 的 ~0.64 rad phase 偏差高峰是 Re=1000 Kolmogorov flow 的 Lyapunov 不穩定性極限，非表徵或訓練策略問題。
10. Re=10000 下，Small 模型（d_model=64）在 t≤1.0 就出現 max phase_err=2.50 rad 的 early-time catastrophic failure（EXP-016）；擴展 RFF σ_max 至 32 無法解決，反而使 max phase_err 升至 5.37 rad（EXP-017），確認 σ 覆蓋不是根因。
11. Re=10000 下，Wide 模型（d_model=128, EXP-018）將 early-time max phase_err@t≤1.0 從 >2 rad 降至 0.71 rad，確認模型容量是 early-time phase failure 的貢獻因子。時間稀疏（41 frames, dt=0.125 vs Re=1000 的 101 frames, dt=0.05）仍是主因，寬模型部分補償。amp_ratio=0.461、KE err=57%，整體問題尚未解決。
12. RFF（seed=42）產生 8/32 個近純 x 方向頻率向量（angle < 20°），造成流場直條紋偽影；改用確定性週期 Fourier 編碼（`periodic_fourier_encode`，`fourier_harmonics=8`）後 KE rel-err 從 25.1% 降至 15.3%（-39%），確認空間編碼品質直接影響整體物理精度。
13. EXP-021 確認週期 Fourier 消除 RFF 角度偏差後，仍殘存 x 方向條紋；根因為 `relpos_bias` 的方向輸入 `(rel_x, rel_y)` 將感測器 x 非均勻分佈（66/128 x-columns covered）注入 attention bias。EXP-022 改用純距離 `|rel|` 後，vorticity error 場轉為隨機分佈，KE rel-err 維持 15.25%，確認方向信息對精度無貢獻且是條紋偏差的根源。
14. Schedule-Free AdamW（EXP-024/025）在 3k 步時與 stepLR（EXP-022）KE 持平（15.04% vs 15.25%），但 amp ratio 更接近 1.0（0.950 vs 1.152）、phase err 改善 42%。延長至 5k 步後 KE 12.06%（vs stepLR 5k 的 13.95%，**-13%**），amp ratio 達 0.995，確認 Polyak 平均在推理品質上優於 stepLR。
15. evaluate_deeponet_cfc.py 存在 checkpoint 載入 bug：舊邏輯只處理 `state["model"]` key，但訓練腳本儲存的 periodic checkpoint 使用 `state["model_state_dict"]`，導致整個 dict 被誤傳入 load_state_dict，所有 periodic checkpoint 評估結果均為廢值（KE ~97%）。已修正為優先讀 `model_state_dict`。**EXP-026、EXP-028、EXP-029 先前的評估失敗均為此 bug 所致，非真實訓練失敗。**
16. SOAP（EXP-026）3k 步：KE 12.39%、Ens 15.23%、amp ratio 0.925。與 SF AdamW（EXP-024）相比，KE 相當但 Ens 偏高、amp 偏低，優勢不明顯。
17. SOAP+Schedule-Free（EXP-028）3k 步：KE 12.24%、Ens 13.35%、amp ratio 1.039。優於 EXP-026（SOAP 無 SF），Ens 改善 12%、amp ratio 超過 1.0。先前誤判為失敗（eval bug），已訂正。
18. 2-layer TemporalCfC（EXP-029，SF AdamW 3k）：KE 11.14%（優於 EXP-024 3k 的 15.04%），Ens 12.38%，但 amp ratio 僅 0.759（差於所有 1-layer 實驗）。增加 CfC 深度降低 KE 但損害能量幅值重建，需延長訓練或加層間殘差才能判斷是否值得。
19. SOAP+SF 5k（EXP-030，resume EXP-028）：KE **9.61%**，首次突破 10% 門檻；u RMSE 5.68e-2（全實驗最低）；amp ratio 1.027；Ens 11.85%（較 EXP-025 的 10.93% 略差）。確認 SOAP 二階曲率估計 + Polyak 平均在 5k 步能顯著優於 SF AdamW（-20% KE）。
20. Re=10000 新資料（si100，dt=0.025，201 frames）搭配 Wide-v2 架構（EXP-031，d_model=128，fourier_harmonics=16，SOAP+SF 3k）：KE 39.4%，確立新資料條件下的基準線。
21. Re=10000 EXP-032（d_model=128，2-layer CfC + 層間殘差，SOAP+SF 3k）：KE 55.1%（差於 EXP-031），確認在 d_model=128 容量下增加 CfC 深度反而退步；層間殘差不足以補救容量瓶頸。
22. Re=10000 EXP-033（d_model=256，operator_rank=256，1-layer，SOAP+SF 3k）：KE 31.5%，為目前 Re=10000 最佳結果；確認容量加倍（3.09M params）是正確方向，優於增加 CfC 深度。

---

## [STATE] Rejected Directions

1. 把 `omega` 當作 sensor data supervision。
2. 只靠降載期待自動修復 collapse。
3. 單純延長訓練步數到 `5k`。
4. `top-k local attention` 作為 decoder 讀 branch token 機制。
5. 在 `Re=1000` 上使用錯誤 forcing mode `k_f=4`。

---

## [STATE] Open Question

| 問題 | 現況 | 下一步方向 |
|---|---|---|
| amplitude ratio=0.9965 是否 overfitting | EXP-015 更高（0.9965），需確認是否對訓練時段過度擬合 | 若有新時段資料，做 out-of-distribution 測試；否則視為未解疑慮 |

---

## [INDEX] Active

| ID | Status | 主題 | 一句結論 |
|---|---|---|---|
| `EXP-030` | `ACTIVE_BASELINE` | `SOAP+SF resume EXP-028 → 5000 steps` | **目前最佳主線（Re=1000）：KE 9.61%、amp 1.027、u RMSE 5.68e-2；首次突破 10%** |
| `EXP-033` | `ACTIVE_REFERENCE` | Re=10000, d_model=256, 1-layer, SOAP+SF 3k | **目前最佳 Re=10000（新資料）：KE 31.5%、Ens 48.9%、amp 0.875**；EXP-034 resume 待跑 |
| `EXP-032` | `NEGATIVE_RESULT` | Re=10000, d_model=128, 2-layer CfC+殘差, SOAP+SF 3k | KE 55.1%（差於 EXP-031）；d_model=128 下加深無益 |
| `EXP-031` | `ACTIVE_REFERENCE` | Re=10000, d_model=128, 1-layer, SOAP+SF 3k（新資料基準） | KE 39.4%；EXP-033 的容量對照基準 |
| `EXP-025` | `ACTIVE_REFERENCE` | `Schedule-Free AdamW + isotropic relpos_bias + 5000 steps` | 前主線；EXP-030 的對照基準（KE 12.06%、amp 0.995）|
| `EXP-029` | `ACTIVE_REFERENCE` | `2-layer TemporalCfC + SF AdamW + 3000 steps` | KE 11.14%（3k 最低）但 amp ratio 0.759，待進一步觀察 |
| `EXP-028` | `ACTIVE_REFERENCE` | `SOAP + Schedule-Free 3000 steps` | KE 12.24%、Ens 13.35%、amp 1.039；優於純 SOAP（EXP-026）|
| `EXP-026` | `ACTIVE_REFERENCE` | `SOAP 3000 steps` | KE 12.39%、Ens 15.23%、amp 0.925；SOAP 無 SF 的對照基準 |
| `EXP-024` | `ACTIVE_REFERENCE` | `Schedule-Free AdamW + isotropic relpos_bias + 3000 steps` | EXP-025 直接前驅；SF 3k 基準（KE 15.04%、amp 0.950）|
| `EXP-023` | `ACTIVE_REFERENCE` | `stepLR + isotropic relpos_bias + 5000 steps` | SF 的對照基準（KE 13.95%、amp 1.072）|
| `EXP-022` | `ACTIVE_REFERENCE` | `periodic_fourier + isotropic relpos_bias + temporal_anchor + 3000 steps` | EXP-023 直接前驅；等向 attn 效益的對照基準（KE 15.25%，無 x 條紋） |
| `EXP-021` | `ACTIVE_REFERENCE` | `periodic_fourier_encode + temporal_anchor + 3000 steps` | EXP-022 直接前驅；週期 Fourier 效益的對照基準（KE 15.3%，仍有殘留 x 條紋） |
| `EXP-015` | `ACTIVE_REFERENCE` | `RFF + phase+temporal anchor + 3000 steps` | EXP-021 直接前驅；RFF 的對照基準 |
| `EXP-020` | `ACTIVE_REFERENCE` | Re=10000, wide + 10000 steps（resume EXP-019） | 在 step 8000 觸底後震盪退化，EXP-019 仍為最佳 |
| `EXP-019` | `ACTIVE_REFERENCE` | Re=10000, wide + 5000 steps（resume EXP-018，舊資料） | 舊資料最佳：amp_ratio=0.595，max phase_err@t≤1.0=0.46 rad；已被 EXP-031+ 系列取代 |
| `EXP-018` | `ACTIVE_REFERENCE` | Re=10000, wide (d_model=128) + σ_max=32 + 3000 steps | 確認容量為 early-time failure 貢獻因子；EXP-019 的直接前驅 |
| `EXP-014` | `ACTIVE_REFERENCE` | `phaseanchor + 3000 steps` | EXP-015 的直接前驅；作為 no-temporal-anchor 對照組 |
| `EXP-012` | `ACTIVE_REFERENCE` | `stepLR(500 x 0.9) + 3000 steps` | no-anchor 基準；phase/temporal anchor 效益的比較起點 |
| `EXP-010` | `ACTIVE_REFERENCE` | `k_f=2` 修正 | 明確證明先前 forcing mode 設錯 |
| `EXP-011` | `ACTIVE_REFERENCE` | `time_marching` 對照 | 關掉後品質變差，應保留 |
| `EXP-008` | `ACTIVE_REFERENCE` | `Small` 1000-step | 證明小模型已足夠進入有效 regime（Re=1000） |
| `EXP-007` | `ACTIVE_REFERENCE` | `rff=4 + gain=1` step600 | 證明 near-zero collapse 的主要根因已被定位 |

---

## [INDEX] Negative

| ID | Status | 主題 | 一句結論 |
|---|---|---|---|
| `EXP-027` | `NEGATIVE_RESULT` | `SOAP resume → 5000 steps`（已取消）| 先取消改做 SOAP+SF；無有效訓練結果 |
| `EXP-002` | `NEGATIVE_RESULT` | `omega` 作為 data supervision | 設定不合理且數值明顯失控 |
| `EXP-004` | `NEGATIVE_RESULT` | 低載 baseline | 能跑，但仍 near-zero collapse |
| `EXP-005` | `NEGATIVE_RESULT` | momentum smoke + curriculum off | 問題是尺度爆量，不是 physics 啟動太早 |
| `EXP-009` | `NEGATIVE_RESULT` | 5k 長訓練 | 訓練更久沒有帶來更好物理解 |
| `EXP-013` | `NEGATIVE_RESULT` | `top-k local attention` | 主模態與整體品質都下降 |
| `EXP-016` | `NEGATIVE_RESULT` | Re=10000 baseline (σ_max=16, small) | early-time catastrophic failure：max phase_err@t≤1.0=2.50 rad |
| `EXP-017` | `NEGATIVE_RESULT` | Re=10000 + σ_max=32 (small) | σ 擴展反而惡化：max phase_err@t≤1.0=5.37 rad，確認 σ 不是根因 |

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

## [RECORD] EXP-012

- Status: `ACTIVE_REFERENCE`
- Time: `2026-03-31 01:28:00 +0800`
- Topic: `stepLR(500 x 0.9) + 3000 steps`
- Train Artifact:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-midlong-uvomega-small-step500x0p9-3000`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-midlong-uvomega-small-step500x0p9-3000)
- Eval:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-eval-midlong-uvomega-small-step500x0p9-3000/summary.json`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-eval-midlong-uvomega-small-step500x0p9-3000/summary.json)
- Change:
  - optimizer 維持 `AdamW`
  - scheduler 改成每 `500` 步衰減 `0.9`
  - 訓練長度從 `1000` 拉到 `3000`
- Metrics:
  - `u_rmse_mean = 0.1080`
  - `v_rmse_mean = 0.0941`
  - `u_std_mean = 0.1401`
  - `v_std_mean = 0.1911`
  - `ke_rel_err_mean = 0.3178`
  - `ens_rel_err_mean = 0.1922`
  - `k_f amplitude ratio @ last = 0.7812`
  - `k_f phase error @ last = -0.5308 rad`
- Decision:
  - 首個完整驗證 phase anchor 前的最佳結果
  - 改善不只出現在 RMSE，也反映到 KE / Enstrophy / phase
- Supersedes:
  - `EXP-008`
  - `EXP-010`
- Superseded_By:
  - `EXP-014`

---

## [RECORD] EXP-010

- Status: `ACTIVE_REFERENCE`
- Time: `2026-03-31 00:18:00 +0800`
- Topic: 修正 `Re=1000` forcing mode 為 `k_f=2`
- Train Artifact:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-midlong-uvomega-small-kf2-1000`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-midlong-uvomega-small-kf2-1000)
- Eval:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-eval-midlong-uvomega-small-kf2-1000/summary.json`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-eval-midlong-uvomega-small-kf2-1000/summary.json)
- Change:
  - `kolmogorov_k_f: 4.0 -> 2.0`
- Metrics:
  - `u_rmse_mean = 0.1419`
  - `v_rmse_mean = 0.1308`
  - `u_std_mean = 0.1205`
  - `v_std_mean = 0.1733`
  - `ke_rel_err_mean = 0.4542`
  - `ens_rel_err_mean = 0.2265`
  - `k_f amplitude ratio @ last = 0.7794`
  - `k_f phase error @ last = -0.7534 rad`
- Decision:
  - 先前 `Re=1000` 的偏差確實被錯 forcing mode 汙染
  - 修正後，問題主軸由 amplitude 轉成 phase 對齊不足
- Superseded_By:
  - `EXP-012`

---

## [RECORD] EXP-011

- Status: `ACTIVE_REFERENCE`
- Time: `2026-03-31 00:22:00 +0800`
- Topic: `time_marching` 對照
- Train Artifact:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-midlong-uvomega-small-notm-1000`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-midlong-uvomega-small-notm-1000)
- Eval:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-eval-midlong-uvomega-small-notm-1000/summary.json`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-eval-midlong-uvomega-small-notm-1000/summary.json)
- Change:
  - 關掉 `time_marching`
- Metrics:
  - `u_rmse_mean = 0.1545`
  - `v_rmse_mean = 0.1631`
  - `ke_rel_err_mean = 0.4577`
  - `ens_rel_err_mean = 0.5079`
  - `k_f amplitude ratio @ last = 0.2726`
  - `k_f phase error @ last = -0.8799 rad`
- Decision:
  - `time_marching` 不是可有可無的技巧
  - 關掉後 forcing mode 振幅與整體場品質都變差

---

## [RECORD] EXP-008

- Status: `ACTIVE_REFERENCE`
- Time: `2026-03-30 12:21:09 +0800`
- Topic: `Small` 尺寸 1000-step
- Config:
  [`/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_small.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_small.toml)
- Train Artifact:
  - `[STATUS: CONTEXT_MISSING]` `deeponet-cfc-midlong-uvomega-small-1000` 不在目前工作樹
- Eval:
  - `[STATUS: CONTEXT_MISSING]` `deeponet-cfc-eval-midlong-uvomega-small-1000/summary.json` 不在目前工作樹
- Change:
  - `d_model = 64`
  - `d_time = 8`
  - `num_spatial_cfc_layers = 1`
  - `num_temporal_cfc_layers = 1`
  - `query_mlp_hidden_dim = 64`
  - `operator_rank = 64`
  - 保留 `rff_sigma = 4`、`output_head_gain = 1`
- Metrics:
  - `trainable_parameters = 182226`
  - `u_rmse_mean = 0.1419`
  - `v_rmse_mean = 0.1345`
  - `u_std_mean = 0.1445`
  - `v_std_mean = 0.1852`
  - `ke_rel_err_mean = 0.3402`
  - `ens_rel_err_mean = 0.2700`
  - `E(k_f=4) = 0.0`
- Decision:
  - `Small` 版成立
  - 不需要更大的 hidden width 才能進入有效 regime
- Superseded_By:
  - `EXP-012`

---

## [RECORD] EXP-007

- Status: `ACTIVE_REFERENCE`
- Time: `2026-03-30 05:19:00 +0800`
- Topic: `rff=4 + gain=1` 低載版 step600 checkpoint
- Checkpoint:
  - `[STATUS: CONTEXT_MISSING]` `.../deeponet-cfc-midlong-uvomega-lowload-rff4-gain1-1000/checkpoints/lnn_kolmogorov_step_600.pt` 不在目前工作樹
- Eval:
  - `[STATUS: CONTEXT_MISSING]` `deeponet-cfc-eval-midlong-uvomega-lowload-rff4-gain1-step600/summary.json` 不在目前工作樹
- Change:
  - 沿用低載版
  - `rff_sigma = 4.0`
  - `output_head_gain = 1.0`
- Metrics:
  - `u_rmse_mean = 0.1393`
  - `v_rmse_mean = 0.1312`
  - `u_std_mean = 0.1538`
  - `v_std_mean = 0.1867`
  - `ke_rel_err_mean = 0.3119`
  - `ens_rel_err_mean = 0.2594`
  - `E(k_f=4) = 0.0`
- Decision:
  - 這版已不再是 near-zero collapse
  - 已證明 root cause 不是單純算力或 batch，而是初始化與頻率尺度

---

## [RECORD] EXP-015

- Status: `ACTIVE_BASELINE`
- Time: `2026-03-31`
- Topic: `use_phase_anchor + use_temporal_anchor` + 3000 steps
- Config:
  [`/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_small_temporal_anchor.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_small_temporal_anchor.toml)
- Train Artifact:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-midlong-uvomega-small-temporal-anchor-3000`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-midlong-uvomega-small-temporal-anchor-3000)
- Eval:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-midlong-uvomega-small-temporal-anchor-3000-eval/summary.json`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-midlong-uvomega-small-temporal-anchor-3000-eval/summary.json)
- Change:
  - 在 EXP-014 基礎上加入 `use_temporal_anchor=true`
  - trunk 注入 `sin/cos(2π n t / T_total)`，`n_harmonics=2`，`T_total=5.0`
  - 新增 `temporal_phase_anchor()` 函式；`query_in` 由 82 → 86（+4 維）
  - 參數量：173,766（+256 vs EXP-014）
- Metrics:
  - `u_rmse_mean = 0.0983`
  - `v_rmse_mean = 0.0919`
  - `u_std_mean = 0.1561`
  - `v_std_mean = 0.1917`
  - `ke_rel_err_mean = 0.2507`
  - `ens_rel_err_mean = 0.1715`
  - `kf_amp_ratio @ last = 0.9965`
  - `kf_phase_err @ last = -0.2176 rad`
  - `phase_err_std = 0.1848`
  - `phase_err_|max| = 0.6554`（@ t=3.5∼4.5）
- Decision:
  - temporal anchor 帶來 KE -9.6%、Ens -10.8%、v RMSE -6.1%，整體積分量改善顯著
  - t=3.5∼4.5 的 phase 高峰（0.64→0.66 rad）未改善 → hypothesis 已被否定
  - 確認 t=3.5∼4.5 phase 偏差為 Re=1000 chaotic divergence 物理本質，不再以表徵改動追求
- Supersedes:
  - `EXP-014`

---

## [RECORD] EXP-014

- Status: `ACTIVE_REFERENCE`
- Time: `2026-03-31`
- Topic: `use_phase_anchor=true` + 3000 steps
- Config:
  [`/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_small_phaseanchor_3000.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_small_phaseanchor_3000.toml)
- Train Artifact:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-midlong-uvomega-small-phaseanchor-3000`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-midlong-uvomega-small-phaseanchor-3000)
- Eval:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-eval-phaseanchor-3000/summary.json`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-eval-phaseanchor-3000/summary.json)
- Change:
  - `use_phase_anchor = true`：注入 `sin/cos(2π k_f y/L)` 至 branch encoder 與 trunk
  - multi-scale RFF bands 保留：`[[16,4.0],[8,8.0],[8,16.0]]`
  - 其餘超參與 EXP-012 相同
- Metrics:
  - `u_rmse_mean = 0.0989`
  - `v_rmse_mean = 0.0979`
  - `ke_rel_err_mean = 0.2774`
  - `ens_rel_err_mean = 0.1923`
  - `kf_amp_ratio @ last = 0.9934`
  - `kf_phase_err @ last = -0.1526 rad`
  - `phase_err_std = 0.1993`（vs baseline 0.3238）
  - `phase_err_|max| = 0.6401`（vs baseline 1.0984）
- Decision:
  - `phase_anchor` 對 forcing mode 有顯著改善：amplitude +27%、phase err -71%
  - 初始偏移（t=0）從 -0.555 → -0.109 rad，branch encoder 的 phase 表徵能力提升
  - t=3.5∼4.5 的高峰偏差從 1.10 → 0.64 rad，結構仍存在但幅度減輕
  - amplitude ratio 0.9934 極高，標注為待確認的 overfitting 疑慮
- [RISK: amplitude=0.9934 可能反映模型對訓練時段的過度擬合，而非真正的物理理解；無 out-of-distribution 驗證前不應宣稱完全解決]
- Supersedes:
  - `EXP-012`
- Superseded_By:
  - `EXP-015`

---

## [RECORD] EXP-013

- Status: `NEGATIVE_RESULT`
- Time: `2026-03-31 02:58:00 +0800`
- Topic: `top-k local attention`
- Train Artifact:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-midlong-uvomega-small-topk16-step500x0p9-3000`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-midlong-uvomega-small-topk16-step500x0p9-3000)
- Eval:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-eval-midlong-uvomega-small-topk16-step500x0p9-3000/summary.json`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-eval-midlong-uvomega-small-topk16-step500x0p9-3000/summary.json)
- Change:
  - decoder 改成每個 query 僅對最近 `16` 個 sensor tokens 做 softmax
- Metrics:
  - `u_rmse_mean = 0.1447`
  - `v_rmse_mean = 0.1418`
  - `ke_rel_err_mean = 0.3978`
  - `ens_rel_err_mean = 0.3522`
  - `k_f amplitude ratio @ last = 0.5520`
  - `k_f phase error @ last = -0.8698 rad`
- Decision:
  - 局部 attention 沒有改善 phase alignment
  - 反而使主模態與整體場品質下降

---

## [RECORD] EXP-009

- Status: `NEGATIVE_RESULT`
- Time: `2026-03-30 03:12:23 +0800`
- Topic: 5k 長訓練 final
- Train Artifact:
  - `[STATUS: CONTEXT_MISSING]` `deeponet-cfc-midlong-uvomega-5k` 不在目前工作樹
- Eval:
  - `[STATUS: CONTEXT_MISSING]` `deeponet-cfc-eval-midlong-uvomega-5k-final/summary.json` 不在目前工作樹
- Change:
  - `uvomega` 主線延長到 `5000` steps
- Metrics:
  - `u_rmse_mean = 0.1952`
  - `v_rmse_mean = 0.2282`
  - `u_std_mean = 0.0162`
  - `v_std_mean = 0.0195`
  - `ke_rel_err_mean = 0.9533`
  - `ens_rel_err_mean = 0.8840`
  - `E(k_f=4) = 0.0`
- Decision:
  - 訓練更久只會把解推得更平滑
  - 不是目前主線的答案

---

## [RECORD] EXP-005

- Status: `NEGATIVE_RESULT`
- Time: `2026-03-30 04:49:05 +0800`
- Topic: momentum form smoke + 關掉 curriculum
- Train Artifact:
  - `[STATUS: CONTEXT_MISSING]` `deeponet-cfc-smoke-uvonly-momentum-check2` 不在目前工作樹
- Change:
  - physics 改回 `momentum(u,v,p) + continuity`
  - `p` 回到模型內部 latent 場
  - 關掉 `physics curriculum`
- Metrics:
  - step 1: `L_data = 2.39e+01`, `L_phys = 3.54e+05`
  - step 2: `L_data = 6.82e+01`, `L_phys = 2.72e+06`
  - step 3: `L_data = 4.39e+01`, `L_phys = 7.17e+05`
- Decision:
  - 真正問題不是 physics 太早啟動
  - 而是啟動時量級已經爆掉

---

## [RECORD] EXP-004

- Status: `NEGATIVE_RESULT`
- Time: `2026-03-30 04:40:23 +0800`
- Topic: 低載 1000-step baseline
- Config:
  - `[STATUS: CONTEXT_MISSING]` `deeponet_cfc_midlong_uvomega_lowload.toml` 未存於目前 repo
- Train Artifact:
  - `[STATUS: CONTEXT_MISSING]` `deeponet-cfc-midlong-uvomega-lowload-baseline-1000` 不在目前工作樹
- Eval:
  - `[STATUS: CONTEXT_MISSING]` `deeponet-cfc-eval-midlong-uvomega-lowload-baseline-1000/summary.json` 不在目前工作樹
- Change:
  - 降低 query/physics batch 與 decoder hidden 寬度
  - 當時仍保留 `rff_sigma=32`、`output_head_gain=5`
- Metrics:
  - `u_rmse_mean = 0.1956`
  - `v_rmse_mean = 0.2235`
  - `u_std_mean = 2.04e-03`
  - `v_std_mean = 1.75e-03`
  - `ke_rel_err_mean = 0.9995`
  - `ens_rel_err_mean = 0.9990`
  - `E(k_f=4) = 0.0`
- Decision:
  - 降載只能讓程序活下來
  - 不能修復 near-zero collapse

---

## [RECORD] EXP-002

- Status: `NEGATIVE_RESULT`
- Time: `2026-03-30 03:30:17 +0800`
- Topic: `omega` 當 data supervision 的 smoke
- Train Artifact:
  - `[STATUS: CONTEXT_MISSING]` `deeponet-cfc-smoke-uvomega-mechcheck` 不在目前工作樹
- Change:
  - 切換到新機制
  - 但仍錯誤地把 `omega` 放進 data supervision
- Metrics:
  - step 1: `L_data = 1.09e+03`
  - step 2: `L_data = 5.13e+03`
  - step 3: `L_data = 6.85e+03`
- Decision:
  - 機制本身可接上
  - 但 `omega` 當 supervision 是錯誤設定

---

## [RECORD] EXP-006

- Status: `ARCHIVED_CONTEXT`
- Time: `2026-03-30 05:00:00 +0800`
- Topic: `rff_sigma=32 + output_head_gain=5` 尺度診斷
- Change:
  - 對照 `rff_sigma=32, gain=5` 與 `rff_sigma=4, gain=1`
- Metrics:
  - 在 `rff_sigma=32`, `gain=5` 下：
    - `u_std ≈ 1.08`
    - `v_std ≈ 1.38`
    - `p_std ≈ 1.08`
    - 一階導數 RMS 約 `180 ~ 250`
    - `lap_u_rms ≈ 1.11e5`
    - `lap_v_rms ≈ 1.32e5`
    - `momentum residual rms ≈ 476`
  - 在 `rff_sigma=4`, `gain=1` 下：
    - `u_std ≈ 0.063`
    - `v_std ≈ 0.052`
    - `p_std ≈ 0.040`
    - 一階導數 RMS 約 `1.2 ~ 1.4`
    - `momentum residual rms ≈ 0.89 ~ 1.44`
- Decision:
  - physics 爆量主因已定位為高頻空間基底與過大初始化增益
- Related:
  - `EXP-007`

---

## [RECORD] EXP-003

- Status: `ARCHIVED_CONTEXT`
- Time: `2026-03-30 04:08:42 +0800`
- Topic: 改回 `u,v-only` sensor supervision
- Train Artifact:
  - `[STATUS: CONTEXT_MISSING]` `deeponet-cfc-smoke-uvonly-check2` 不在目前工作樹
- Change:
  - sensor data supervision 只保留 `u,v`
  - `omega` 僅保留為物理量與診斷
- Metrics:
  - step 1: `L_data = 3.87e+01`
  - step 2: `L_data = 1.05e+02`
  - step 3: `L_data = 2.56e+02`
- Decision:
  - 這是必要修正
  - 但當時整體仍偏不穩，尚不能直接長訓練

---

## [RECORD] EXP-001

- Status: `ARCHIVED_CONTEXT`
- Time: `2026-03-30 03:12:23 +0800`
- Topic: 早期 `uvomega` 主線中長訓練與 checkpoint sweep
- Config:
  - `[STATUS: CONTEXT_MISSING]` `deeponet_cfc_midlong_uvomega.toml` 未存於目前 repo
- Train Artifact:
  - `[STATUS: CONTEXT_MISSING]` `deeponet-cfc-midlong-uvomega` 不在目前工作樹
- Eval Sweep:
  - `[STATUS: CONTEXT_MISSING]` `deeponet-cfc-eval-midlong-uvomega-sweep` 不在目前工作樹
- Change:
  - 新資料格式 `domain=[0,1]`
  - branch 使用 thin sensor tokens + token self-attention + temporal CfC
  - decoder 使用 query-to-branch cross-attention
- Metrics:
  - `step_250`: `u_rmse=0.2287`, `v_rmse=0.2371`, `ke=0.7566`, `ens=2.4015`
  - `step_500`: `u_rmse=0.2282`, `v_rmse=0.2534`, `ke=0.6682`, `ens=2.8044`
  - `step_750`: `u_rmse=0.1997`, `v_rmse=0.2410`, `ke=0.8664`, `ens=0.3841`
  - `final`: `u_rmse=0.1979`, `v_rmse=0.2352`, `ke=0.9130`, `ens=0.5407`
- Decision:
  - 主線可跑
  - 但後期仍向低能量保守解收縮

---

## [RECORD] EXP-016

- Status: `NEGATIVE_RESULT`
- Time: `2026-03-31`
- Topic: Re=10000 baseline（EXP-015 等級，σ_max=16，d_model=64）
- Config:
  [`/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_small_re10000.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_small_re10000.toml)
- Train Artifact:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-midlong-uvomega-small-re10000-3000`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-midlong-uvomega-small-re10000-3000)
- Eval:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-midlong-uvomega-small-re10000-3000-eval/summary.json`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-midlong-uvomega-small-re10000-3000-eval/summary.json)
- Change:
  - DNS: Re=10000（ν=0.0001, 256×256, 41 frames, dt=0.125, T=5.0）
  - `rff_sigma_bands = [[16,4.0],[8,8.0],[8,16.0]]`（沿用 EXP-015）
  - `d_model = 64`（Small）
  - 其他超參與 EXP-015 相同
- Metrics:
  - `u_rmse_mean = 0.3102`
  - `ke_rel_err_mean = 0.6453`
  - `kf_amp_ratio_last = 0.4160`
  - `kf_phase_err_last = 1.4198 rad`
  - `max phase_err @ t≤1.0 = 2.4996 rad`（catastrophic failure）
- Decision:
  - EXP-015 主線架構直接移植至 Re=10000 失敗
  - early-time phase error 在 t≤1.0 就達 2.50 rad，遠超 Re=1000 的 0.32 rad
  - 時間稀疏（41 vs 101 frames）與容量不足共同導致 early-time catastrophic failure
- [RESULT: PHYSICAL_FAILURE] max phase_err@t≤1.0 = 2.50 rad > 2 rad 閾值

---

## [RECORD] EXP-017

- Status: `NEGATIVE_RESULT`
- Time: `2026-03-31`
- Topic: Re=10000 + σ_max=32（Small 模型）
- Config:
  [`/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_small_re10000_sigma32.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_small_re10000_sigma32.toml)
- Train Artifact:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re10000-sigma32-3000`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re10000-sigma32-3000)
- Eval:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re10000-sigma32-3000-eval/summary.json`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re10000-sigma32-3000-eval/summary.json)
- Change:
  - `rff_sigma_bands = [[8,4.0],[8,8.0],[8,16.0],[8,32.0]]`（σ_max 16→32，band 重均分）
  - `d_model = 64`（Small，與 EXP-016 相同）
  - 其他超參不變
- Metrics:
  - `u_rmse_mean = 0.2837`
  - `ke_rel_err_mean = 0.5914`
  - `kf_amp_ratio_last = 0.5303`
  - `kf_phase_err_last = 0.9923 rad`
  - `max phase_err @ t≤1.0 = 5.3713 rad`（較 EXP-016 更差）
- Hypothesis:
  - 高頻空間基底不足是振幅崩潰的貢獻因子；若 amp_ratio 顯著提升，確認 σ 是瓶頸。
- Falsifiability 判定:
  - `kf_amp_ratio = 0.530`（EXP-016 為 0.416），略有改善，但 max phase_err@t≤1.0 暴增至 5.37 rad。
  - **σ 擴展不是根因，甚至損害 early-time phase stability**
  - 寬頻 band 重新分配後，低頻 representation 能量被稀釋，導致 early-time 更不穩定
- Decision:
  - σ_max=32 + Small 模型的組合對 Re=10000 不可行
  - 確認頻率覆蓋不是 early-time failure 的根因
- [RESULT: PHYSICAL_FAILURE] max phase_err@t≤1.0 = 5.37 rad（更差）

---

## [RECORD] EXP-018

- Status: `ACTIVE_REFERENCE`
- Time: `2026-03-31`
- Topic: Re=10000 + σ_max=32 + Wide 模型（d_model=128）
- Config:
  [`/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_wide_re10000_sigma32.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_wide_re10000_sigma32.toml)
- Train Artifact:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re10000-wide-sigma32-3000`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re10000-wide-sigma32-3000)
- Eval:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re10000-wide-sigma32-3000/summary.json`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re10000-wide-sigma32-3000/summary.json)
- Change:
  - EXP-017 config 基礎上 `d_model: 64 → 128`
  - `query_mlp_hidden_dim: 64 → 128`
  - `operator_rank: 64 → 128`
  - 其他超參不變
- Metrics:
  - `u_rmse_mean = 0.2610`
  - `v_rmse_mean = 0.2256`
  - `ke_rel_err_mean = 0.5728`
  - `ens_rel_err_mean = 0.6576`
  - `kf_amp_ratio_last = 0.4612`（t=5.0）
  - `kf_phase_err_last = 0.1537 rad`（t=5.0）
  - `max phase_err @ t≤1.0 = 0.7103 rad`（t=0.0 最大）
- Hypothesis:
  - 容量是 early-time phase failure 的次要貢獻因子；若 phase_err@t≤1.0 < 2 rad → 確認
- Falsifiability 判定:
  - `max phase_err@t≤1.0 = 0.71 rad < 2 rad`（EXP-016: 2.50、EXP-017: 5.37）
  - **假設確認：模型容量是 early-time failure 的貢獻因子**
  - 寬模型顯著改善 early-time phase stability，但振幅仍嚴重低估（amp_ratio=0.461）
- Decision:
  - Wide 模型對 Re=10000 的 early-time phase 問題有顯著改善
  - 時間稀疏（41 frames）仍是振幅低估的主因；需要更密集的時間 anchor 或更多訓練步數
  - 目前為 Re=10000 最佳結果，標為 ACTIVE_REFERENCE
- [RISK: amp_ratio=0.461 仍遠低於 Re=1000 的 0.9965，振幅低估根因尚未解決]
- Superseded_By:
  - `EXP-019`

---

## [RECORD] EXP-019

- Status: `ACTIVE_REFERENCE`
- Time: `2026-03-31`
- Topic: Re=10000, wide 模型（d_model=128）+ 5000 steps（resume from EXP-018 step 3000）
- Config:
  [`/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_wide_re10000_sigma32.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_wide_re10000_sigma32.toml)
- Train Artifact:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re10000-wide-sigma32-5000`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re10000-wide-sigma32-5000)
- Eval:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re10000-wide-sigma32-5000/summary.json`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re10000-wide-sigma32-5000/summary.json)
- Change:
  - EXP-018 step 3000 checkpoint resume，延長至 5000 steps
  - 訓練時 L_data 軌跡：3000→1.05e-01, 3500→8.29e-02, 4000→6.36e-02, 4500→4.19e-02, 5000→3.63e-02（持續收斂）
- Metrics:
  - `u_rmse_mean = 0.2348`
  - `v_rmse_mean = 0.2088`
  - `ke_rel_err_mean = 0.5361`
  - `ens_rel_err_mean = 0.6099`
  - `kf_amp_ratio_last = 0.5955`（vs EXP-018: 0.461, **+13.4%**）
  - `kf_phase_err_last = 0.0482 rad`（vs EXP-018: 0.154 rad, **-68.8%**）
  - `max phase_err @ t≤1.0 = 0.4631 rad`（vs EXP-018: 0.710 rad, **-34.9%**）
- Decision:
  - 步數延長帶來明顯且單調的收益：振幅、phase、RMSE 全面改善
  - 模型仍未飽和（L_data 仍在下降，訓練曲線無停滯跡象）
  - 目前為 Re=10000 最佳結果；建議下一步繼續 resume 至 7000-10000 steps 觀察飽和點
  - KE err=53.6%、amp_ratio=0.595，與 Re=1000（KE 25%、amp 0.997）仍有差距，振幅低估根因（時間稀疏）尚未解決
- Supersedes:
  - `EXP-018`
- Superseded_By:
  - `EXP-020`（但 EXP-019 指標優於 EXP-020，EXP-019 仍為此 config 最佳）

---

## [RECORD] EXP-020

- Status: `ACTIVE_REFERENCE`
- Time: `2026-03-31`
- Topic: Re=10000, wide 模型（d_model=128）+ 10000 steps（resume from EXP-019 step 5000）
- Config:
  [`/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_wide_re10000_sigma32.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_wide_re10000_sigma32.toml)
- Train Artifact:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re10000-wide-sigma32-10000`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re10000-wide-sigma32-10000)
- Eval:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re10000-wide-sigma32-10000/summary.json`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re10000-wide-sigma32-10000/summary.json)
- Change:
  - EXP-019 step 5000 checkpoint resume，延長至 10000 steps
- L_data 軌跡:
  - step 7000: 2.03e-02, step 8000: **1.27e-02（最低點）**, step 9000: 1.34e-02, step 10000: 1.89e-02（震盪）
- Metrics:
  - `u_rmse_mean = 0.2602`
  - `v_rmse_mean = 0.2109`
  - `ke_rel_err_mean = 0.5396`
  - `ens_rel_err_mean = 0.6338`
  - `kf_amp_ratio_last = 0.5482`（vs EXP-019: 0.595, **退化**）
  - `kf_phase_err_last = -0.0794 rad`
  - `max phase_err @ t≤1.0 = 0.4260 rad`（vs EXP-019: 0.463 rad，略微改善）
- Decision:
  - L_data 在 step 8000 觸底後震盪，kf_amp_ratio 退回 0.548（EXP-019 為 0.595）
  - KE/Ens rel-err 也略微惡化
  - 確認此 config 的有效訓練上限約在 step 5000；更多步數帶來學習率過大造成的震盪
  - **EXP-019（5k steps）為此 config 與 wide 架構的最佳結果**
  - 下一步方向：解決時間稀疏根因，而非繼續增加步數
- [RISK: amp_ratio=0.548 退化表明模型已飽和，繼續訓練無益]

---

## [RECORD] EXP-021

- Status: `ACTIVE_REFERENCE`
- Time: `2026-04-01`
- Topic: 週期 Fourier 編碼（`periodic_fourier_encode`）取代 RFF + 移除 `use_phase_anchor`
- Config:
  [`/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_small_periodicfft.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_small_periodicfft.toml)
- Train Artifact:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re1000-periodicfft-3000`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re1000-periodicfft-3000)
- Change:
  - `rff_encode` → `periodic_fourier_encode`（`fourier_harmonics=8`，共 32 特徵）
  - 移除 `use_phase_anchor`（k_f=2 已包含在 k=2 諧波）
  - 移除 B matrix buffer
- Metrics:
  - `u_rmse_mean = 0.0814`（vs EXP-015: 0.1057，**-23%**）
  - `v_rmse_mean = 0.0868`
  - `ke_rel_err_mean = 0.1530`（vs EXP-015: 0.2510，**-39%**）
  - `ens_rel_err_mean = 0.1370`（vs EXP-015: 0.1720）
  - `kf_amp_ratio_last = 1.1524`
  - `kf_phase_err_last = -0.3809 rad`
- Hypothesis 判定:
  - **週期 Fourier 確認消除 RFF 角度偏差 → KE -39%**
- 殘留問題:
  - vorticity field 視覺化仍有 x 方向條紋偽影（用戶回報 Image #2）
  - 根因為 `relpos_bias` 的方向輸入 `(rel_x, rel_y)` 將感測器 x 非均勻分佈注入 attention
- Decision:
  - 本實驗確認空間編碼品質直接影響整體物理精度
  - x 條紋根源已從 RFF 角度偏差轉移到 relpos_bias 方向偏差
- Supersedes:
  - `EXP-015`
- Superseded_By:
  - `EXP-022`

---

## [RECORD] EXP-022

- Status: `ACTIVE_BASELINE`
- Time: `2026-04-01`
- Topic: 等向 relpos_bias（純距離輸入）消除 x 條紋偽影
- Config:
  [`/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_small_isotropicattn.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_small_isotropicattn.toml)
- Train Artifact:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re1000-isotropicattn-3000`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re1000-isotropicattn-3000)
- Eval:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re1000-isotropicattn-3000/eval/summary.json`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re1000-isotropicattn-3000/eval/summary.json)
- Change:
  - `relpos_bias` 輸入從 `(rel_x, rel_y, |rel|)` 改為純距離 `|rel|`（維度 3→1）
  - 其他超參與 EXP-021 完全相同
- Metrics:
  - `u_rmse_mean = 0.0815`
  - `v_rmse_mean = 0.0868`
  - `ke_rel_err_mean = 0.1525`（vs EXP-021: 0.1530，**持平**）
  - `ens_rel_err_mean = 0.1193`（vs EXP-021: 0.137，**改善**）
  - `kf_amp_ratio_last = 1.1524`
  - `kf_phase_err_last = -0.3809 rad`
- Hypothesis 判定:
  - **條紋根源確認為 relpos_bias 方向輸入，而非 Fourier 編碼本身**
  - 純距離不損失精度（KE 持平），且 vorticity error 轉為隨機分佈
- Decision:
  - 等向 relpos_bias 是正確設計：Kolmogorov flow 的感測器貢獻與方向無關，只與距離相關
  - 目前為 Re=1000 最佳主線
  - [RISK: v field 左下角仍有輕微垂直帶狀，需用更多感測器或更高 fourier_harmonics 驗證是否為真實流場結構]
- Supersedes:
  - `EXP-021`

---

## [RECORD] EXP-023

- Status: `ACTIVE_BASELINE`
- Time: `2026-04-01`
- Topic: EXP-022 resume → 5000 steps
- Config:
  [`/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_small_isotropicattn_5000.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_small_isotropicattn_5000.toml)
- Train Artifact:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re1000-isotropicattn-5000`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re1000-isotropicattn-5000)
- Eval:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re1000-isotropicattn-5000/eval/summary.json`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re1000-isotropicattn-5000/eval/summary.json)
- Change:
  - EXP-022 step 3000 checkpoint resume，延長至 5000 steps
  - L_data 軌跡：3000→1.75e-02, 3500→2.34e-02, 4000→2.38e-02, 4500→1.89e-02, 5000→**1.32e-02**（持續收斂）
- Metrics:
  - `u_rmse_mean = 0.07325`（vs EXP-022: 0.0815，**-10%**）
  - `v_rmse_mean = 0.07708`（vs EXP-022: 0.0868，**-11%**）
  - `ke_rel_err_mean = 0.1395`（vs EXP-022: 0.1525，**-8.5%**）
  - `ens_rel_err_mean = 0.1194`（vs EXP-022: 0.1193，持平）
  - `kf_amp_ratio_last = 1.0721`（vs EXP-022: 1.152，**更接近 1.0**）
  - `kf_phase_err_last = -0.3606 rad`（vs EXP-022: -0.381 rad，略改善）
- Decision:
  - 步數延長帶來單調收益（KE -8.5%、RMSE -10%），模型在 5000 步仍未飽和
  - vorticity error 場維持隨機分佈，確認 x 條紋已消除
  - 目前為 Re=1000 最佳結果
  - [RISK: amp_ratio 1.072 仍略大於 1.0，表示 forcing mode 能量略微高估；尚在可接受範圍]
- Supersedes:
  - `EXP-022`

---

## [RECORD] EXP-024

- Status: `ACTIVE_REFERENCE`
- Time: `2026-04-01`
- Topic: Schedule-Free AdamW 取代 stepLR，3000 steps（從頭訓練）
- Config:
  [`/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_small_schedulefree.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_small_schedulefree.toml)
- Train Artifact:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re1000-schedulefree-3000`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re1000-schedulefree-3000)
- Eval:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re1000-schedulefree-3000/eval/summary.json`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re1000-schedulefree-3000/eval/summary.json)
- Change:
  - `lr_schedule: "step"` → `lr_schedule: "schedulefree"`，`lr_warmup_steps=300`
  - 其他超參與 EXP-022 完全相同（`lr=1e-3`）
  - L_data 3000步：2.04e-02（仍收斂）
- Metrics:
  - `u_rmse_mean = 0.0801`（vs EXP-022: 0.0815，**-2%**）
  - `v_rmse_mean = 0.0818`（vs EXP-022: 0.0868，**-6%**）
  - `ke_rel_err_mean = 0.1504`（vs EXP-022: 0.1525，持平）
  - `ens_rel_err_mean = 0.1207`
  - `kf_amp_ratio_last = 0.9502`（vs EXP-022: 1.152，**更接近 1.0**）
  - `kf_phase_err_last = -0.2219 rad`（vs EXP-022: -0.381 rad，**-42%**）
- Decision:
  - KE 持平，但 amp/phase 顯著改善，確認 Polyak 平均對推理品質有獨立收益
  - 模型未飽和（L_data 2.04e-02），適合 resume
- Supersedes: —
- Superseded_By: `EXP-025`

---

## [RECORD] EXP-025

- Status: `ACTIVE_BASELINE`
- Time: `2026-04-01`
- Topic: Schedule-Free AdamW，5000 steps（resume from EXP-024 step 3000）
- Config:
  [`/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_small_schedulefree_5000.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/deeponet_cfc_midlong_uvomega_small_schedulefree_5000.toml)
- Train Artifact:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re1000-schedulefree-5000`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re1000-schedulefree-5000)
- Eval:
  [`/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re1000-schedulefree-5000/eval/summary.json`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re1000-schedulefree-5000/eval/summary.json)
- Change:
  - EXP-024 step 3000 checkpoint resume，延長至 5000 steps
  - L_data 軌跡：3000→2.04e-02, 3500→1.77e-02, 4000→2.45e-02, 4500→2.18e-02, 5000→**1.40e-02**
- Metrics:
  - `u_rmse_mean = 0.07185`（vs EXP-023 stepLR 5k: 0.07325，**-2%**）
  - `v_rmse_mean = 0.07357`（vs EXP-023: 0.07708，**-5%**）
  - `ke_rel_err_mean = 0.1206`（vs EXP-023: 0.1395，**-13%**）
  - `ens_rel_err_mean = 0.1093`（vs EXP-023: 0.1194，**-8%**）
  - `kf_amp_ratio_last = 0.9949`（vs EXP-023: 1.072，**最接近 1.0**）
  - `kf_phase_err_last = -0.2934 rad`（vs EXP-023: -0.361 rad，**-19%**）
- Decision:
  - Schedule-Free 在所有指標全面優於同步數 stepLR（EXP-023）
  - amp ratio 0.995 是目前所有實驗中最接近 1.0 的結果
  - 目前為 Re=1000 最佳主線
  - [RISK: L_data 5000步仍為 1.40e-02，模型未完全飽和，可繼續 resume]
- Supersedes:
  - `EXP-023`
  - `EXP-024`
