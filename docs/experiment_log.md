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
| 主線 config | [`/Users/latteine/Documents/coding/pi-lnn/configs/exp_030_re1000_soap_sf_5k.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/exp_030_re1000_soap_sf_5k.toml) |
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
23. Physics loss 機制變更系列（EXP-035~039）全部無法突破 EXP-031 基準（KE 39.4%）：
    - EXP-035（chebyshev-256 + normalize）：KE 66.1%；EXP-036（normalize only）：KE 86.7%；EXP-037（chebyshev only）：KE 62.5%。結構化 collocation 與殘差正規化均有害。
    - EXP-038（Poisson weight=1.0）：KE 56.0%；EXP-039（Poisson weight=0.1）：KE 41.6%。壓力 Poisson 約束在任何權重下均未改善 KE，確認壓力自由度不是 Re=10000 的主要瓶頸。
    - 根本結論：在 K=100 sparse sensors（覆蓋 k≤5）的資訊量限制下，改善 physics loss 設計無法突破資訊理論上限。下一步需從資料密度或架構表達力著手。
24. Transfer learning 前置實驗（EXP-041，Re=1000 d=128 wide，3000 steps）：KE 24.5%，顯著差於 EXP-030（d=64，KE 9.61%）。確認在 3000 steps 內，較大架構（d=128）在 Re=1000 上未完全收斂，超參化程度造成 KE 倒退。Weights 品質次佳但仍作為 EXP-042 transfer 出發點，檢驗 pre-training 是否提供有用 inductive bias。
25. EXP-040 Transfer 失敗（架構不匹配）：嘗試從 EXP-030（d=64, harmonics=8, 1-layer attn）transfer 至 Re=10000 Wide-v2 架構（d=128, harmonics=16, 2-layer attn），導致 `size mismatch` 與 missing keys。確認 transfer learning 要求 source 與 target 架構完全相同。

---

## [STATE] Rejected Directions

1. 把 `omega` 當作 sensor data supervision。
2. 只靠降載期待自動修復 collapse。
3. 單純延長訓練步數到 `5k`。
4. `top-k local attention` 作為 decoder 讀 branch token 機制。
5. 在 `Re=1000` 上使用錯誤 forcing mode `k_f=4`。
6. Physics loss 機制調整（Re=10000）：Chebyshev collocation、residual normalization、壓力 Poisson 約束（weight=0.1~1.0）均無法突破 EXP-031 基準。在 K=100 sparse sensors 的資訊量限制下，physics loss 設計已非主要瓶頸。
7. Transfer learning 需要 source/target 架構完全相同（EXP-040）。EXP-030（d=64）→ Re=10000 Wide-v2（d=128）直接 transfer 因架構不匹配失敗。
8. Transfer learning（EXP-042）在 source 品質不足時產生負遷移：EXP-041（Re=1000, d=128）以 KE=24.5% 作為 source，transfer 後 Re=10000 KE 40.2%，差於隨機初始化（EXP-031 39.4%）。確認 transfer 有效的前提是 source 本身已充分收斂。

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
| `EXP-055` | `RUNNING` | Re=10000, resume EXP-048 step_9500 + IC Loss Weight（λ_IC=10，t≤0.05），再跑 10000 步至 step 19500 | 訓練中（step_14000 / 19500）；單一變數對照 EXP-048，隔離 IC weight 效果 |
| `EXP-054` | `POSITIVE_RESULT` | Re=10000, resume EXP-048 step_9500 + RAR freq=1000，再跑 10000 步至 step 19500 | **KE 19.6%（突破 EXP-048 21.8%，−2.2pp）**；amp 0.961；phase 0.064 rad；RAR freq=1000 有效，為新 Re=10000 最佳基準 |
| `EXP-053` | `NEGATIVE_RESULT` | Re=10000, resume EXP-048 step_9500 + RAR freq=50，再跑 10000 步至 step 19500 | KE 25.2%（差於 EXP-048 21.8%）；RAR 更新頻率過高（50步），持續擾動 loss landscape，L_phys 爆漲 7.96→19.27 |
| `EXP-052` | `NEGATIVE_RESULT` | Re=10000, resume EXP-048 step_9500 + L-BFGS 100步（實際完成） | KE 24.07%（差於 EXP-048 21.8%）；每 100 步耗時 68 分鐘，2000 步需 22 小時，計算成本不可行；L-BFGS 不適用於此規模的隨機 mini-batch 設定 |
| `EXP-051` | `NEGATIVE_RESULT` | Re=10000, d_model=256, 從頭 10k，harmonics=20 + t0_boost×3 | KE 27.81%（差於 EXP-048 21.8%）；t=0 KE err 67.4%（更差）；漸進式訓練路徑（3k→5k→10k）的優勢被從頭訓練抵消，無法評估個別改動貢獻 |
| `EXP-050` | `NEGATIVE_RESULT` | Re=10000, d_model=256, resume EXP-048 + physics curriculum (8→128)，10000 steps | KE 25.87%（差於 EXP-048 21.8%）；kf_amp_ratio 0.927→0.213 崩潰；SOAP+SF resume 無法延續學習狀態，physics curriculum 在 resume 情境有害 |
| `EXP-049` | `NEGATIVE_RESULT` | Re=10000, d_model=256, K=200 sensor，從頭訓練 10000 steps | KE 43.9%（大幅差於 EXP-048 的 21.8%）；混淆：EXP-048 為漸進式訓練，EXP-049 為冷啟動；K=200 從頭 10k 甚至差於 K=100 從頭 3k（31.5%），sensor 覆蓋非瓶頸 |
| `EXP-048` | `ACTIVE_REFERENCE` | Re=10000, d_model=256, resume EXP-043 → 10000 steps | KE 21.8%；amp 0.899；phase 0.039 rad；t=0 KE 低估 58%；已被 EXP-054（KE 19.6%）超越 |
| `EXP-047` | `NEGATIVE_RESULT` | Re=10000, d_model=256 + GradNorm 4-task [data,ns_u,ns_v,cont]，從頭 3000 steps | KE 72.1%（最差）；GradNorm 把 w_ns 推至 0.37，物理過強壓制資料；phase err 0.009 rad（改善）但 KE/Ens 全面退步 |
| `EXP-046` | `NEGATIVE_RESULT` | Re=10000, d_model=256 + GradNorm 3-task，從頭 3000 steps | KE 59.9%（大幅退步）；GradNorm 梯度範數平衡與物理可行性無直接對應；w_cont 降至 0.750，l_phys 全程偏高 |
| `EXP-045` | `NEGATIVE_RESULT` | Re=10000, d_model=256 + sweep 最佳參數（lr=4.75e-3, cont_w=0.509, soap_freq=20, locality=True），3000 steps | KE 35.4%（退步，差於 EXP-043 27.2%）；sweep 1500-step l_data 代理指標與 KE 不相關 |
| `EXP-044` | `ARCHIVED_CONTEXT` | Re=10000, d_model=256 + locality decay，3000 steps（從頭訓練） | **訓練中止**（~500/3000 steps）；架構正確，無收斂結論；待後續決定是否 resume |
| `EXP-043` | `ACTIVE_REFERENCE` | Re=10000, d_model=256, EXP-033 resume → 5000 steps | KE 27.2%、amp 0.931、phase -0.025 rad；已被 EXP-048 超越 |
| `EXP-033` | `ACTIVE_REFERENCE` | Re=10000, d_model=256, 1-layer, SOAP+SF 3k | KE 31.5%、Ens 48.9%、amp 0.875；已被 EXP-048 超越 |
| `EXP-042` | `NEGATIVE_RESULT` | Re=10000, transfer from EXP-041 (Re=1000 d=128), SOAP+SF 3k | KE 40.2%（差於 EXP-031 39.4%）；source 品質不足（EXP-041 KE=24.5%）導致負遷移，transfer 無效 |
| `EXP-041` | `ACTIVE_REFERENCE` | Re=1000, d=128 wide (同 EXP-031 架構), SOAP+SF 3k | KE 24.5%（差於 EXP-030 的 9.61%）；d=128 在 3000 步未完全收斂；作為 EXP-042 transfer source |
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
| `EXP-040` | `NEGATIVE_RESULT` | Re=10000 transfer from EXP-030（架構不匹配）| `size mismatch`：EXP-030 d=64/harmonics=8 vs target d=128/harmonics=16；直接 transfer 不可行 |
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

## [RECORD] EXP-046

- Status: `NEGATIVE_RESULT`
- Time: `2026-04-13`
- Topic: Re=10000, d_model=256 + GradNorm 3-task [data, ns, cont]，從頭訓練 3000 steps
- Config: `configs/exp_047_re10000_xlarge_gradnorm4.toml`（已更新為 EXP-047 設定，原始版本已不在 repo）
- Train Artifact: `artifacts/deeponet-cfc-re10000-xlarge-gradnorm-3000`（已刪除）
- Key Change: 以 GradNorm 自動平衡 l_data / l_ns / l_cont 三個 task；初始等權 [1,1,1]
- Metrics:
  - `u_rmse_mean = 3.0414e-01`
  - `v_rmse_mean = 2.4088e-01`
  - `ke_rel_err_mean = 0.5987`（KE **59.9%**）
  - `ens_rel_err_mean = 0.3458`
  - `kf_amp_ratio_last = 0.5602`
  - `kf_phase_err_last = -0.0673 rad`
- Decision: **Falsified**。KE 59.9% 大幅差於 EXP-033（31.5%）。GradNorm 等梯度範數目標不等同物理可行性；w_cont 在訓練過程中持續下降至 0.750，散度約束被系統性壓制，與 EXP-045 的 cont_w gaming 失敗模式相同。根本原因：NS x/y 混合為單一 task，無法感知 Kolmogorov forcing 的不對稱性；初始等權導致 GradNorm 從物理上不合理的起點調整。

---

## [RECORD] EXP-045

- Status: `NEGATIVE_RESULT`
- Time: `2026-04-12`
- Topic: Re=10000, d_model=256 + Optuna sweep 最佳參數（lr=4.75e-3, cont_w=0.509, soap_freq=20, locality=True），3000 steps
- Config: `configs/exp_045_re10000_xlarge_sweep_best.toml`（已不在 repo）
- Train Artifact: `artifacts/deeponet-cfc-re10000-xlarge-sweep-best-3000`（已刪除）
- Key Change: sweep v2 以 1500-step l_data 作為代理指標，搜尋最佳 lr / cont_w / soap_freq / locality
- Metrics:
  - `u_rmse_mean = 1.5146e-01`
  - `v_rmse_mean = 1.3408e-01`
  - `ke_rel_err_mean = 0.3535`（KE **35.4%**）
  - `ens_rel_err_mean = 0.5015`
  - `kf_amp_ratio_last = 0.8655`
  - `kf_phase_err_last = 0.0956 rad`
- Decision: **Falsified**。KE 35.4% 差於 EXP-033（31.5%）；sweep 代理指標（l_data@1500）與最終 KE 無相關性。cont_w=0.509 使散度約束比 EXP-033 弱，物理一致性犧牲。確認 sweep 以 l_data 為目標的策略根本上無效。

---

## [RECORD] EXP-031

- Status: `ACTIVE_REFERENCE`（已被 EXP-033 取代）
- Time: `2026-04-08`
- Topic: Re=10000, d_model=128, 2-layer token attention, SOAP+SF 3000 steps（新資料基準）
- Config: 基於 `exp_008_re1000_small_baseline.toml` 的 wide-v2 設定
- Train Artifact: `artifacts/deeponet-cfc-re10000-wide-v2-3000`（已刪除）
- Key Change: 首次使用新資料（si100，dt=0.025，201 frames），d_model=128，fourier_harmonics=16，2-layer token attn
- Metrics:
  - `ke_rel_err_mean ≈ 0.394`（KE **39.4%**）（無完整 eval summary，數值來自訓練觀察）
  - `kf_amp_ratio_last ≈ 0.70`（估計）
- Decision: 確立新資料條件下的 d=128 基準線；KE 39.4% 為 EXP-033（d=256，KE 31.5%）的容量對照組。確認擴展模型容量（d=128 → d=256）是正確方向。

---

## [RECORD] EXP-055

- Status: `RUNNING`
- Time: `2026-04-20` 啟動（目前至 step_14000 / 19500）
- Topic: Re=10000, resume EXP-048 step_9500 + IC Loss Weight（t≤0.05 × 10），再跑 10000 步至 step 19500
- Config: [`configs/exp_055_re10000_xlarge_rar_ic.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/exp_055_re10000_xlarge_rar_ic.toml)
- Resume Checkpoint: `artifacts/deeponet-cfc-re10000-xlarge-20000/deeponet-cfc-re10000-xlarge-10000/checkpoints/lnn_kolmogorov_step_9500.pt`
- Change vs EXP-048：在 data loss 中對 t≤0.05 的 query points 加乘 λ_IC=10；無 RAR（單一變數）
- Hypothesis: t=0 KE rel-err < 58%（EXP-048）；整體 KE rel-err < 21.8%（EXP-048 基準）
- Falsifiability: 若 step_19500 KE > 21.8%，IC weighting 單獨無益

---

## [RECORD] EXP-054

- Status: `POSITIVE_RESULT`
- Time: `2026-04-19` 訓練與評估完成
- Topic: Re=10000, resume EXP-048 step_9500 + RAR collocation（freq=1000），再跑 10000 步至 step 19500
- Config: [`configs/exp_054_re10000_xlarge_rar_1k.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/exp_054_re10000_xlarge_rar_1k.toml)
- Artifact: `artifacts/deeponet-cfc-re10000-exp054-rar-1k`
- Resume Checkpoint: `artifacts/deeponet-cfc-re10000-xlarge-20000/deeponet-cfc-re10000-xlarge-10000/checkpoints/lnn_kolmogorov_step_9500.pt`
- Evaluated Checkpoint: `checkpoints/lnn_kolmogorov_step_19500.pt`
- Change vs EXP-048：加入 RAR（Residual Adaptive Refinement）；每 1000 步從 320 候選點評估近似 NS 殘差，選 top-80% 高殘差點 + 20% 隨機點作為下一批 collocation；無架構改動
- Change vs EXP-053：`rar_update_freq` 從 50 改為 1000
- Hypothesis: RAR 偏向高殘差區域（含 t≈0）可降低整體 KE rel-err；更新頻率降低可避免擾動已收斂的優化路徑
- Falsifiability: KE > 21.8% → RAR 在此設定無益
- Metrics（step_19500，v2 評估腳本 2026-04-20 重跑；Re 正規化 + cell-center grid + spectrum k-axis 全部修正）：
  - `ke_rel_err_mean = 0.1958`（KE **19.6%**，優於 EXP-048 的 21.8%，**突破基準 −2.2pp**）
  - `ens_rel_err_mean = 0.3636`（EXP-048 為 43.7%，改善 7pp）
  - `kf_amp_ratio_last = 0.9609`（EXP-048 為 0.899，明顯改善）
  - `ek_ratio_kf_last = 0.889`（EXP-048 為 0.803）
  - `kf_phase_err_last = 0.064 rad`（舊評估為 0.039 rad，k 軸 bug 修正後改變）
  - `u_rmse_mean = 0.1064`、`v_rmse_mean = 0.0951`
  - L_data 最終 = 2.35e-3（遠低於 EXP-048 的 ~1e-2 量級）
- Analysis:
  - RAR freq=1000 成功：L_data 穩定下降（1.03e-2→2.35e-3），L_phys 僅在每次 RAR 更新點短暫跳升後回落
  - 對比 EXP-053（freq=50）：頻繁更新 collocation 導致 L_phys 持續爆漲（7.96→19.27）、L_data 上升（1.20e-2→2.64e-2），說明 SOAP+SF preconditioner 需要足夠步數適應新 loss landscape
  - Ens rel-err 改善幅度（7pp）與 kf_amp_ratio 大幅改善（0.899→0.961）顯示 RAR 對 forcing mode 重建有顯著幫助
  - t=0 KE 低估是否改善尚未量化（EXP-048 為 58%），待後續實驗跟蹤
- Decision: **Confirmed**。RAR 有效，成為新的 Re=10000 最佳基準（KE 19.6%）。下一步：EXP-055 加入 IC Loss Weight，針對 t=0 KE 低估問題進一步介入。

---

## [RECORD] EXP-053

- Status: `NEGATIVE_RESULT`
- Time: `2026-04-19` 訓練與評估完成
- Topic: Re=10000, resume EXP-048 step_9500 + RAR collocation（freq=50），再跑 10000 步至 step 19500
- Config: [`configs/exp_053_re10000_xlarge_rar.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/exp_053_re10000_xlarge_rar.toml)
- Artifact: `artifacts/deeponet-cfc-re10000-exp053-rar`
- Resume Checkpoint: `artifacts/deeponet-cfc-re10000-xlarge-20000/deeponet-cfc-re10000-xlarge-10000/checkpoints/lnn_kolmogorov_step_9500.pt`
- Evaluated Checkpoint: `checkpoints/lnn_kolmogorov_step_19500.pt`
- Change vs EXP-048：加入 RAR；每 50 步從 320 候選點評估近似 NS 殘差，選 top-80% 高殘差點 + 20% 隨機點
- Hypothesis: RAR 偏向高殘差區域可降低 KE rel-err；t=0 KE err < 58%
- Falsifiability: KE > 22% → RAR 在此設定無益
- Metrics（step_19500）：
  - `ke_rel_err_mean = 0.2524`（KE **25.2%**，**差於 EXP-048 的 21.8%，+3.4pp**）
  - `ens_rel_err_mean = 0.4866`
  - `kf_amp_ratio_last = 0.878`
  - L_data 最終 = 2.64e-2（明顯上升）；L_phys = 19.27（從 7.96 爆漲 2.4×）
- Analysis:
  - 每 50 步更新 collocation 頻率過高：SOAP 的 Kronecker factor preconditioner 與 SF 的 lookahead buffer 均需數百步適應新 loss landscape；50 步遠不足
  - L_phys 持續爆漲（7.96→19.27）說明模型被迫追逐不斷變化的高殘差區域，優化路徑受到破壞
  - L_data 同步上升（1.20e-2→2.64e-2）確認 data supervision 也因此退步
  - 後續 EXP-054 將 freq=1000 驗證了頻率是根本原因
- Decision: **Falsified**。RAR 更新頻率過高是唯一根因；EXP-054 以 freq=1000 改善並確認。

---

## [RECORD] EXP-052

- Status: `NEGATIVE_RESULT`
- Time: `2026-04-18` 訓練啟動，100 步後手動終止
- Topic: Re=10000, resume EXP-048 step_9500 → L-BFGS 細調 2000 步（實際完成 100 步）
- Config: [`configs/exp_052_re10000_xlarge_lbfgs.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/exp_052_re10000_xlarge_lbfgs.toml)
- Artifact: `artifacts/deeponet-cfc-re10000-exp052-lbfgs`
- Resume Checkpoint: `artifacts/deeponet-cfc-re10000-xlarge-20000/deeponet-cfc-re10000-xlarge-10000/checkpoints/lnn_kolmogorov_step_9500.pt`
- Evaluated Checkpoint: `checkpoints/lnn_kolmogorov_step_9600.pt`（100 L-BFGS meta-steps）
- Change vs EXP-048：優化器從 SOAP+SF 切換至 L-BFGS（lr=0.1, max_iter=20, history_size=10, strong Wolfe line search）；batch 在每個 meta-step 固定（pre-sampling）；time_marching=false（固定全時段）
- Hypothesis: KE rel-err < 21.8%；L-BFGS 曲率資訊可在已收斂的局部盆地內進一步細調
- Falsifiability: 若 500 步後 L_data 未低於 EXP-048 的 9.78e-3，L-BFGS 無益
- Metrics（step_9600，100 L-BFGS 步）：
  - `ke_rel_err_mean = 0.2407`（KE **24.07%**，差於 EXP-048 21.8%）
  - `kf_amp_ratio_last = 0.889`（EXP-048 為 0.927，略退步）
  - 100 步耗時 ~68 分鐘（~41 s/step）；2000 步預估需 22 小時
- Analysis:
  - L-BFGS 在 mini-batch 設定下違反 Wolfe 條件成立假設：每個 closure 呼叫雖使用固定 batch，但不同 meta-step 之間 batch 仍切換，導致曲率估計不一致
  - `max_iter=20` 使每個 meta-step 實際等於最多 20 次完整 forward+backward，計算成本比 first-order 高 20×
  - 3.09M 參數模型在 MPS 上的 L-BFGS 開銷遠超過 SOAP 的 Kronecker factor 更新
  - step_9600 的 KE 24.07% 較 EXP-048 的 21.8% 退步，確認 100 步 L-BFGS 未能改善，而非尚未收斂
  - 根本限制：L-BFGS 的二階近似對 high-dimensional + non-convex + stochastic landscape 的 PINNs 益處有限，而計算代價是 SOAP 的 20 倍
- Decision: **Falsified**。(1) KE 24.07% > 21.8%（基準），步數僅 100，尚未達到 falsifiability 定義的 500 步，但退步方向明確；(2) 計算成本不可行（22 小時/2000 步），即使最終有益亦無法落地。L-BFGS 不適用於此規模的 sparse-data PINN + mini-batch 設定。

---

## [RECORD] EXP-051

- Status: `NEGATIVE_RESULT`
- Time: `2026-04-18` 訓練與評估完成
- Topic: Re=10000, d_model=256, 從頭訓練 10000 steps；fourier_harmonics 16→20 + t_early_weight=3.0（t≤0.1）
- Config: [`configs/exp_051_re10000_xlarge_harmonics20_t0boost.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/exp_051_re10000_xlarge_harmonics20_t0boost.toml)
- Artifact: `artifacts/deeponet-cfc-re10000-exp051`
- Evaluated Checkpoint: `checkpoints/lnn_kolmogorov_step_9500.pt`
- Change vs EXP-048：(1) fourier_harmonics 16→20；(2) t≤0.1 data loss × 3.0；(3) 從頭訓練（因架構維度改變，無法 resume）
- Hypothesis: KE rel-err < 21.8%；t=0 ke_rel_err < 50%（vs 估計 58%）
- Falsifiability: KE > 22% 或 t=0 KE err 仍 > 50% → 改動無效
- Metrics:
  - `ke_rel_err_mean = 0.2781`（KE **27.81%**，差於 EXP-048 21.8%）
  - `ens_rel_err_mean = 0.4904`
  - `kf_amp_ratio_last = 0.670`（EXP-048 為 0.927）
  - `kf_phase_err_last = 0.004 rad`（改善，EXP-048 為 -0.014 rad）
  - `ke_rel_err @ t=0 = 0.674`（比 EXP-048 更差）
  - `u_rmse_mean = 0.148`
- Analysis:
  - 從頭 10k 訓練（KE=27.81%）劣於 EXP-033 從頭 3k（KE=31.5%）✗ 不成立——等等，EXP-033=31.5% 更差，EXP-048=21.8% 是漸進式 3k→5k→10k 累積的結果
  - 根本混淆變數：EXP-048 的優勢來自漸進式訓練路徑（多次 resume 累積），而非單純 10k 步的訓練量
  - harmonics=20 與 t0_boost 的效果被「從頭 vs 漸進」這個混淆變數完全覆蓋，無法獨立歸因
  - phase_err 改善（0.004 rad）可能是 harmonics=20 的正面貢獻，但在 kf_amp 大幅下滑下意義不大
- Decision: **Falsified**。關鍵教訓：漸進式訓練路徑（EXP-033→043→048）的累積收益約 10% KE，等效於架構或 loss 層面的顯著改動。任何需要「從頭訓練」的架構改動，都必須先承受這個路徑代價。

---

## [RECORD] EXP-050

- Status: `NEGATIVE_RESULT`
- Time: `2026-04-17` 訓練，`2026-04-18` 評估完成
- Topic: Re=10000, d_model=256, resume EXP-048（KE=21.8%）+ physics collocation curriculum，10000 steps
- Config: [`configs/exp_050_re10000_xlarge_20k.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/exp_050_re10000_xlarge_20k.toml)
- Artifact: `artifacts/deeponet-cfc-re10000-xlarge-20000`
- Evaluated Checkpoint: `checkpoints/lnn_kolmogorov_step_9500.pt`（step_10000.pt 因 SIGTERM 損毀）
- Change vs EXP-048：加入 physics collocation curriculum（8→128 點，warmup 1000 步，ramp 5000 步）；其餘架構/超參數完全不變
- Curriculum Schedule:
  - step 1–1000: n_phys=8（time-marching warmup 期間固定）
  - step 1001–6000: n_phys 線性 8→128
  - step 6001–9500: n_phys=128（固定）
- Hypothesis: KE rel-err < 21.8%（EXP-048 基準）
- Falsifiability: 若 KE > 22%，表示 physics curriculum 在 resume 情境有害
- Metrics:
  - `ke_rel_err_mean = 0.2587`（KE **25.87%**，差於 EXP-048 21.8%）
  - `ens_rel_err_mean = 0.4736`
  - `kf_amp_ratio_last = 0.2133`（EXP-048 為 0.927，崩潰）
  - `kf_phase_err_last = 0.650 rad`
  - `u_rmse_mean = 0.129`、`v_rmse_mean = 0.120`
- Analysis:
  - SOAP+SF optimizer 無法正確延續 resume checkpoint 的學習狀態（preconditioner、Polyak 平均均從零重建）
  - time-marching warmup 重啟（1000 步短窗口）使模型從長窗口退回短窗口優化，破壞 EXP-048 的時域泛化
  - kf_amp_ratio 從 0.927 崩至 0.213，確認 forcing 模態結構被破壞而非僅精度退步
- Decision: **Falsified**。KE 25.87% > 22%；physics curriculum resume 方案對 SOAP+SF 有害。根本限制在於 SOAP+SF 無 checkpoint-resumable optimizer state，而非 curriculum 設計本身。

---

## [RECORD] EXP-049

- Status: `PENDING`
- Time: `2026-04-15` 設計，訓練中
- Topic: Re=10000, d_model=256, K=200 sensor（含 p 特徵，6 特徵集），從頭訓練 10000 steps
- Config: [`configs/exp_049_re10000_xlarge_k200.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/exp_049_re10000_xlarge_k200.toml)
- Artifact: `artifacts/deeponet-cfc-re10000-xlarge-k200-10000`
- Single Variable vs EXP-048: sensor 集 K=100 → K=200；架構/超參數/優化器完全不變；從頭訓練（無 resume），以排除 pretrained weight 干擾
- Spectral Coverage Diagnostic（Fourier pseudo-inverse, 8 snapshots, k≤50）：
  - K=100：acc>0.8 上限 k=20，acc>0.5 上限 k=32
  - K=200：acc>0.8 上限 k=41，acc>0.5 上限 k=50
  - K=200 在 k=20–41 慣性範圍提供額外可重建資訊
- Hypothesis: KE rel-err < 21.8%（EXP-048 的 K=100 基準）
- Falsifiability: 若 5000 步後 KE 與 EXP-043（K=100, 5k, KE=27.2%）差距 < 2%，表示瓶頸非 sensor 覆蓋
- Metrics:
  - `ke_rel_err_mean = 0.4389`（KE **43.9%**）
  - `ens_rel_err_mean = 0.5175`
  - `kf_amp_ratio_last = 0.0837`（幾乎為 0，forcing 頻率完全失效）
  - `kf_phase_err_last = -0.190 rad`
  - `u_rmse_mean = 0.480`、`v_rmse_mean = 0.340`
  - 最終 `L_data = 5.76e-2`（EXP-048 最終 `L_data = 9.78e-3`，差 6 倍）
- Visual Diagnostics:
  - KE 曲線全程低估 DNS 約 50%，t=0 更嚴重（LNN=0.048 vs DNS=0.161，-70%）
  - 能譜形狀（斜率）尚可，但絕對能量量級偏低
- Analysis:
  - 混淆變數：EXP-048 = 漸進訓練（3k→5k→10k），EXP-049 = 冷啟動 10k
  - K=200 從頭 10k（43.9%）差於 K=100 從頭 3k（31.5%）——更多步數仍更差
  - 可能原因：400-dim branch net 輸入比 200-dim 優化景觀更難；QR-pivoting K=200 後段的 sensors 在慣性範圍外，資訊重疊
- Decision: **Falsified**。Falsifiability 條件成立：K=200 在同等訓練量下未改善 KE。瓶頸不在 sensor 頻譜覆蓋（acc>0.8 k_cutoff 20→41），而在訓練收斂或架構容量。

---

## [RECORD] EXP-048

- Status: `ACTIVE_REFERENCE`
- Time: `2026-04-14` 訓練，`2026-04-15` 完成
- Topic: Re=10000, d_model=256, resume EXP-043（5000 steps, KE=27.2%）→ 10000 steps
- Config: [`configs/exp_048_re10000_xlarge_10k.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/exp_048_re10000_xlarge_10k.toml)
- Artifact: `artifacts/deeponet-cfc-re10000-xlarge-20000/deeponet-cfc-re10000-xlarge-10000`
- Change vs EXP-043：iterations 5000 → 10000；checkpoint_period 100 → 500；其餘完全不變
- Metrics（v2 評估腳本 2026-04-20 重跑；Re 正規化 + cell-center grid + spectrum k-axis 全部修正）：
  - `ke_rel_err_mean = 0.2181`（KE **21.8%**，與舊值一致）
  - `ens_rel_err_mean = 0.4366`
  - `kf_amp_ratio_last = 0.899`（舊評估為 0.927，舊值因 Re bug 偏高）
  - `ek_ratio_kf_last = 0.803`（舊評估為 0.000，k 軸 bug 已修正）
  - `kf_phase_err_last = 0.039 rad`（舊評估為 0.015 rad，k 軸 bug 修正後改變）
  - `u_rmse_mean = 0.1059`、`v_rmse_mean = 0.0930`（cell-center 修正後微幅改善）
- Visual Diagnostics:
  - 大尺度流場結構（低 k）重建良好；誤差場呈隨機高頻分佈
  - 渦量峰值系統性低估 50%（DNS ±30 vs LNN ±15）：spectral bias，sensors 僅覆蓋 k≤5
  - 能譜 k<10 端與 DNS 高度吻合；k>20 端略低估（預期）
  - KE 曲線：t=0 嚴重低估（LNN=0.068 vs DNS=0.161，-58%）；t>1 穩定在 DNS 的 85%
  - u/v RMSE 從 ~0.25（t=0）降至 ~0.08（t>3），無晚期發散
- Decision: **Confirmed positive**。KE 21.8% 突破 EXP-043（27.2%），訓練曲線顯示 L_data 在 10k 步仍未完全收斂（9.78e-3）。
  主要殘差來源：(1) t=0 KE 低估（初始條件重建不足）；(2) 高 k 渦量結構 spectral bias（資訊理論上限）。
  3k→5k→10k 的 KE 軌跡：31.5%→27.2%→21.8%，邊際遞減但持續改善。

---

## [RECORD] EXP-047

- Status: `NEGATIVE_RESULT`
- Time: `2026-04-14`
- Topic: Re=10000, d_model=256 + GradNorm 4-task [data, ns_u, ns_v, cont]，從頭訓練 3000 steps
- Config: [`configs/exp_047_re10000_xlarge_gradnorm4.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/exp_047_re10000_xlarge_gradnorm4.toml)
- Artifact: `artifacts/deeponet-cfc-re10000-xlarge-gradnorm4-3000`
- Key Changes from EXP-046:
  - NS residual 拆成 `l_ns_u` / `l_ns_v` 兩個獨立 task（4-task: data, ns_u, ns_v, cont）
  - GradNorm 改為直接公式（無 Adam optimizer）+ EMA（momentum=0.5）
  - 初始權重 [1.0, 0.01, 0.01, 0.01]；normalize_to_data_()（w_data=1 為基準）
  - 優化器：AdamWScheduleFree（lr=1e-3）
- Metrics:
  - `ke_rel_err_mean = 0.7212`（KE **72.1%**，最差紀錄）
  - `ens_rel_err_mean = 0.8578`
  - `kf_amp_ratio_last = 0.339`
  - `kf_phase_err_last = 0.009 rad`（異常好，但 KE 仍退步）
  - `u_rmse_mean = 0.252`、`v_rmse_mean = 0.202`
- Decision: **Falsified**。KE 72.1% 為目前最差結果，大幅差於 EXP-033（31.5%）。
  GradNorm 把 w_ns_u/w_ns_v 從 0.01 推至 ~0.37，相對 data（固定=1.0）高達 37%；
  在 K=100 sparse sensor（覆蓋 k≤5）的資訊量限制下，過強的 NS 約束使模型離開資料拉力，
  落入「物理自洽但與真實流場不符」的解空間——與 EXP-035~039 系列的失敗模式相同。
  w_cont 僅上升至 0.028，散度約束仍被壓制。
  **結論：GradNorm（不論 3-task 或 4-task）在 Re=10000 sparse-data 設定下均無效。**

---

## [RECORD] EXP-044

- Status: `ARCHIVED_CONTEXT`
- Time: `2026-04-10`（訓練中止，~500/3000 steps）
- Topic: Re=10000, d_model=256 + 可學習距離衰減（log_locality_decay），從頭訓練
- Config: [`configs/exp_044_re10000_xlarge_locality.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/exp_044_re10000_xlarge_locality.toml)
- Train Artifact: `artifacts/deeponet-cfc-re10000-xlarge-locality-3000`（部分，~500 steps checkpoint 可能存在）
- Change vs EXP-043：
  - 新增 `use_locality_decay = true`
  - 新增可學習參數 `log_locality_decay`（初始 -2.0，α≈0.135）
  - Cross-attention scores: `score_ij += -α * r_ij`（對數空間距離懲罰）
  - 從頭訓練（非 resume），消除 optimizer 狀態不對齊問題
- Metrics: 無（訓練中止，未完成評估）
- Hypothesis: locality decay 引入近鄰優先 inductive bias，KE < 27.2%（優於 EXP-043）
- Falsifiability:
  - 若 KE >= 27.2%，或 log_locality_decay 收斂至 << -2，則 Re=10000 非局域性使 locality bias 無益
- Decision: **未決定**（訓練因其他工作優先而手動中止，非因結果判定）；架構修改已實作並驗證可執行
- Supersedes: —
- Superseded_By: —

---

## [RECORD] EXP-042

- Status: `ACTIVE_REFERENCE`
- Time: `2026-04-09 19:45:00 +0800`（訓練中）
- Topic: Re=10000 transfer fine-tune from EXP-041 (Re=1000, d=128)
- Config: [`configs/exp_042_re10000_transfer_wide.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/exp_042_re10000_transfer_wide.toml)
- Train Artifact: [`artifacts/deeponet-cfc-re10000-transfer-wide-3000`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re10000-transfer-wide-3000)
- resume_checkpoint: `artifacts/deeponet-cfc-re1000-wide-3000/lnn_kolmogorov_final.pt`
- Change:
  - source: EXP-041 Re=1000 weights (KE=24.5%，未完全收斂)
  - target: Re=10000，完全相同架構（d=128, harmonics=16, 2-layer attn）
  - optimizer 從頭開始（不延續 Re=1000 optimizer 狀態）
- Metrics:
  - `u_rmse_mean = 1.6911e-01`
  - `v_rmse_mean = 1.4663e-01`
  - `u_std_mean = 3.0864e-01`
  - `v_std_mean = 2.3200e-01`
  - `ke_rel_err_mean = 0.4015`
  - `ens_rel_err_mean = 0.5750`
  - `k_f amplitude ratio @ last = 0.6856`
  - `k_f phase error @ last = 0.2824 rad`
- Hypothesis: 即使 source 未完全收斂，pre-training 提供的 inductive bias（感測器聚合、低頻流場、time-marching）可加速 Re=10000 收斂，KE < 39.4%
- Falsifiability: 若 KE >= 39.4%，表示 KE=24.5% 品質的 Re=1000 weights 無正向遷移效果
- Decision: **Falsified**。KE 40.2% 略差於 EXP-031（39.4%），amp ratio 大幅退步（0.686）。EXP-041 source 品質不足（KE=24.5%），Re=1000 未收斂的偏誤對 Re=10000 產生負遷移。Transfer learning（A 方案）在此條件下無效。

---

## [RECORD] EXP-041

- Status: `ACTIVE_REFERENCE`
- Time: `2026-04-09 ~18:00 +0800`
- Topic: Re=1000，使用與 EXP-031 完全相同的 d=128 架構，作為 EXP-042 transfer source
- Config: [`configs/exp_041_re1000_wide.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/exp_041_re1000_wide.toml)
- Train Artifact: [`artifacts/deeponet-cfc-re1000-wide-3000`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-re1000-wide-3000)
- Change:
  - d_model: 64 → 128, fourier_harmonics: 8 → 16, num_token_attention_layers: 1 → 2
  - 目的：與 EXP-031 架構對齊，使 transfer 可行
- Metrics:
  - `u_rmse_mean = 8.37e-02`
  - `v_rmse_mean = 8.22e-02`
  - `u_std_mean = 1.60e-01`
  - `v_std_mean = 1.94e-01`
  - `ke_rel_err_mean = 0.2451`
  - `ens_rel_err_mean = 0.1604`
  - `k_f amplitude ratio @ last = 0.9500`
  - `k_f phase error @ last = -0.5028 rad`
- Decision:
  - KE 24.5% 顯著差於 EXP-030（d=64，KE 9.61%）；確認 d=128 在 Re=1000 + 3000 steps 下過度參數化，未完全收斂
  - 儘管品質次佳，仍作為 EXP-042 的 transfer checkpoint
- Supersedes: —
- Superseded_By: EXP-042（transfer 結果待定）

---

## [RECORD] EXP-040

- Status: `NEGATIVE_RESULT`
- Time: `2026-04-09`
- Topic: Re=10000 transfer from EXP-030（架構不匹配，失敗）
- Config: [`configs/exp_040_re10000_transfer.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/exp_040_re10000_transfer.toml)
- resume_checkpoint: `artifacts/deeponet-cfc-re1000-soap-sf-5000/lnn_kolmogorov_final.pt`
- Error: `RuntimeError: size mismatch for spatial_encoder.base_norm.weight: shape [34] vs [66]` + missing keys for `token_blocks.1`
- Root Cause:
  - EXP-030（source）：d=64, fourier_harmonics=8, 1-layer attn → spatial_encoder input dim=34
  - EXP-031 架構（target）：d=128, fourier_harmonics=16, 2-layer attn → spatial_encoder input dim=66
  - 完全不同架構，無法直接載入
- Decision: Transfer learning 必須保證 source 與 target 架構完全相同；EXP-041 先訓練 Re=1000 wide 作為正確 source

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
  [`/Users/latteine/Documents/coding/pi-lnn/configs/exp_008_re1000_small_baseline.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/exp_008_re1000_small_baseline.toml)
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
  [`/Users/latteine/Documents/coding/pi-lnn/configs/exp_015_re1000_temporal_anchor.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/exp_015_re1000_temporal_anchor.toml)
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
  [`/Users/latteine/Documents/coding/pi-lnn/configs/exp_014_re1000_phaseanchor.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/exp_014_re1000_phaseanchor.toml)
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
  `deeponet_cfc_midlong_uvomega_small_re10000.toml` `[STATUS: DELETED — 引用舊版 DNS，已隨舊 DNS 一併清除]`
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
  `deeponet_cfc_midlong_uvomega_small_re10000_sigma32.toml` `[STATUS: DELETED — 引用舊版 DNS，已隨舊 DNS 一併清除]`
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
  `deeponet_cfc_midlong_uvomega_wide_re10000_sigma32.toml` `[STATUS: DELETED — 引用舊版 DNS，已隨舊 DNS 一併清除]`
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
  `deeponet_cfc_midlong_uvomega_wide_re10000_sigma32.toml` `[STATUS: DELETED — 引用舊版 DNS，已隨舊 DNS 一併清除]`
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
  `deeponet_cfc_midlong_uvomega_wide_re10000_sigma32.toml` `[STATUS: DELETED — 引用舊版 DNS，已隨舊 DNS 一併清除]`
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
  [`/Users/latteine/Documents/coding/pi-lnn/configs/exp_021_re1000_periodicfft.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/exp_021_re1000_periodicfft.toml)
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
  [`/Users/latteine/Documents/coding/pi-lnn/configs/exp_022_re1000_isotropicattn.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/exp_022_re1000_isotropicattn.toml)
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
  [`/Users/latteine/Documents/coding/pi-lnn/configs/exp_023_re1000_isotropicattn_5k.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/exp_023_re1000_isotropicattn_5k.toml)
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
  [`/Users/latteine/Documents/coding/pi-lnn/configs/exp_024_re1000_schedulefree.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/exp_024_re1000_schedulefree.toml)
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
  [`/Users/latteine/Documents/coding/pi-lnn/configs/exp_025_re1000_schedulefree_5k.toml`](/Users/latteine/Documents/coding/pi-lnn/configs/exp_025_re1000_schedulefree_5k.toml)
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
