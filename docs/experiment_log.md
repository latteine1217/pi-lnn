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

### Re=1000 Baseline（EXP-030）

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

### Re=10000 Baseline（EXP-064）

| 項目 | 現況 |
|---|---|
| Baseline ID | `EXP-064` |
| 主線 config | `configs/exp_064_re10000_xlarge_sensor_physics.toml` |
| train artifact | `artifacts/deeponet-cfc-re10000-exp064-sensor-physics` |
| eval checkpoint | `artifacts/deeponet-cfc-re10000-exp064-sensor-physics/checkpoints/lnn_kolmogorov_step_10000.pt` |
| 目前判讀 | EXP-063（GradNorm）+ sensor 位置 continuity physics；**KE 7.80%（Re=10000 歷史最佳）**；div_l2 0.184；phase_err -0.0228 rad |
| 主要優勢 | KE **7.80%**（-0.85pp vs EXP-063）、div_l2 **0.184**（-9.6%）、kf_phase_err **-0.0228 rad**（-53%） |
| 已確認上限 | band_mid/high@t=5 ≈100% 為 K=100 感測器資訊論硬上限；sensor physics continuity 已無法突破此限 |
| **結案狀態** | **K=100 稀疏重建結案（2026-04-26）**：此結果接受為最終主線，中高頻不可達為數學必然（CS 需 ~5000 sensors，K=100 差 50 倍），後續提升需 K≥5000 感測器或 DNS 高頻先驗 |

### 主線固定假設

- 觀測 supervision 僅使用 `u, v`
- physics 使用 primitive `momentum + continuity`
- 空間編碼：`LearnableFourierEmb`（`embed_dim=128`，σ=2.0）for Re=10000；`periodic_fourier_encode`（`fourier_harmonics=8`）for Re=1000
- `relpos_bias`：純距離輸入 `|rel|`（等向），不含方向向量
- `output_head_gain = 1`
- `use_temporal_anchor = true`（`n_harmonics=2`）：為 trunk 提供 `sin/cos(2π n t/T)` 絕對時間座標
- `Small` 尺寸（d=64）在 Re=1000 已足夠；Re=10000 需 `XLarge`（d=256）
- `Re=1000/10000` forcing mode 均為 `k_f = 2`
- `time_marching` 應保留
- 優化器主線（Re=10000）：`SOAP + Schedule-Free`（`lr=1e-3`，`betas=(0.9,0.999)`，`precond_freq=2`，`step_decay`，`warmup=2000`）
- `GradNorm`（`update_freq=1000`，`momentum=0.9`）自動均衡 data/physics task 梯度比例

---

## [STATE] Supported Decisions

> 按主題分組整理；個別 EXP 細節見 archive 對應 GROUP（[`docs/experiment_archive_kolmogorov.md`](experiment_archive_kolmogorov.md)）。

### A. Re=1000 主線（已結案於 EXP-030 KE 9.61%）

**觀測與 physics 基礎（G1, G2）**：
- `u,v-only` sensor supervision 是必要前提；`omega` 不可作 supervision（量級失控，EXP-002）。
- physics 維持 primitive `momentum + continuity`，不應用錯誤 supervision 掩蓋尺度問題。
- `rff_sigma=32 + output_head_gain=5` 把導數與 residual 推到不可訓練量級（mom_residual ~476 vs 健康 ~1）；改 `rff_sigma=4 + gain=1` 為硬約束（EXP-006/007）。
- `Small`（d=64）在 Re=1000 已足夠；`time_marching=true` 必要（關掉後 amp_ratio 從 0.78 崩到 0.27，EXP-011）。

**Spatial encoding（G5）**：
- RFF（seed=42）有近純 x 方向頻率向量造成直條紋；改 `periodic_fourier_encode`（h=8）使 KE 從 25.1% → 15.3%（EXP-021，**-39%**）。
- `relpos_bias` 方向輸入 `(rel_x, rel_y)` 將感測器 x 非均勻分佈注入 attention bias；改純距離 `|rel|`（等向）後 vorticity error 轉為隨機分佈（EXP-022）。

**Anchor 系列（G3）**：
- `use_phase_anchor=true` 對 forcing mode amplitude（+27%）與 phase err（-71%）有顯著改善（EXP-014）。
- `use_temporal_anchor=true`（n_harmonics=2）帶來 KE/Ens 各降 ~10%（EXP-015）。
- **t=3.5∼4.5 的 ~0.64 rad phase 偏差為 Re=1000 chaotic divergence 的 Lyapunov 不穩定極限**，非表徵或訓練策略問題；後續不再以表徵改動追求此點。

**Optimizer 演進（G6, G7）**：
- Schedule-Free AdamW 在 5k 步下優於 stepLR：KE -13%、amp 0.995（EXP-025 vs EXP-023）；Polyak 平均提供獨立的推理品質收益。
- SOAP（二階曲率）+ Schedule-Free 在 5k 步下首次突破 KE 10%（EXP-030 KE **9.61%**, -20% vs EXP-025）。
- 2-layer TemporalCfC 降低 KE 但損害 amp（EXP-029 amp 0.759），加深 CfC 與能量幅值有 trade-off。

**重要 bug 修正**：
- `evaluate_deeponet_cfc.py` 早期版本只處理 `state["model"]` key，但訓練腳本儲存 `state["model_state_dict"]` → EXP-026/028/029 早期評估顯示 KE ~97%（廢值，非真實訓練失敗）。已修正為優先讀 `model_state_dict`。

### B. Re=10000 容量與資料條件（G4, G8）

- 舊 DNS（41 frames, dt=0.125）下，Small（d=64）有 max_phase_err@t≤1.0 = 2.50 rad 的 catastrophic failure（EXP-016）；σ_max 16→32 反而惡化至 5.37 rad（EXP-017）；Wide（d=128）改善至 0.71 rad，**確認模型容量是 early-time failure 的貢獻因子**（EXP-018）。
- 新 DNS（si100, 201 frames, dt=0.025）下 d=128 1-layer 為 KE 39.4%（EXP-031）；d=128 + 2-layer CfC 退步至 55.1%（EXP-032）；d=256 1-layer 改善至 31.5%（EXP-033）。**確認擴大寬度優於增加深度**。

### C. Re=10000 失敗探索（並見 [STATE] Rejected Directions）

- **Physics loss 機制變更（EXP-035~039）全部失敗**（G8）：Chebyshev / residual normalize / Poisson 約束（任何權重）都無法突破 EXP-031 baseline。在 K=100 sparse 限制下 physics loss 設計已非主要瓶頸。
- **Transfer learning 證偽**（G9）：架構必須完全相同（EXP-040 size mismatch）；source 品質不足會產生負遷移：EXP-041（KE 24.5%）→ EXP-042（KE 40.2%）差於隨機初始化（EXP-031 39.4%）。
- **去 Schedule-Free 退步**（G13）：jaxpi 純 SOAP 即使搭配 betas/warmup/decay 對齊仍從 KE 20.6% 退至 29.4%（EXP-061）。**SF Polyak 平均對 Re=10000 chaotic flow 不可或缺**。
- **雙向 CfC 無法解決 t=0 重建**（G13）：EXP-059 t=0 KE 60.4% 比 EXP-057（55.5%）惡化；確認 t=0 問題核心是**訓練訊號（IC weight）不足**，非資訊存取（因果編碼）不足。
- **Trunk MLP 加深無效**（主檔 EXP-065）：1→2 層後 band_mid/high@t=5 仍 ≈100%；至此通過四次 falsifiability（optimizer / physics 密度 / sensor 位置 / trunk 表達力），確認為 K=100 資訊論硬上限。
- **GradNorm 在 d=128 sparse 設定失敗**（G11 EXP-046/047）：等權初始或 [1,0.01,...] 從不合理起點調整，w_ns 推至 0.37 物理過強壓制資料；KE 60~72%。但 GradNorm 在 G14（LearnableFourier + 正確初始權重）下成功，差別在架構容量與初值。

### D. Re=10000 EXP-048 resume 系列突破（G12）

- **IC Loss Weight（λ=10, t≤0.05）為單一最有效改動**：KE 從 EXP-048 的 21.8% → EXP-055 的 17.1%（**-4.7pp**），優於 RAR alone（EXP-054, -2.2pp）。kf_amp_ratio 0.970 與 E(k_f) 0.934 全系列最佳。
- **RAR freq 是關鍵**：freq=50（EXP-053）擾亂 SOAP+SF preconditioner（L_phys 7.96→19.27）；freq=1000（EXP-054）才能與 SOAP+SF 共存，KE 19.6%。
- **RAR + IC weight 不可同時使用**（EXP-056 KE 19.4% > IC alone 17.1%）：RAR 週期性更新 collocation 改變 loss landscape，與 IC weight 依賴的穩定梯度方向衝突。

### E. Re=10000 LearnableFourier 演進 → K=100 結案（G14, EXP-064 主檔）

- `LearnableFourierEmb`（embed_dim=128, init σ=2.0）取代固定 periodic Fourier：KE 從 EXP-055 的 17.1% → EXP-062 的 **10.4%**（-6.7pp）。但改善源自低頻精度提升（band_low@t=5: 5.8%），非頻率覆蓋擴展（band_high 99.98% 未改善）。
- 正確 jaxpi SOAP（保留 SF, betas=(0.9,0.999), precond_freq=2, wd=0, decay=2000）+ GradNorm（freq=1000, momentum=0.9, init [1,0.01,0.01,0.01]）：KE **8.65%**（EXP-063, -1.75pp）；div_l2 0.204 全系列最佳（-64%）。
- Sensor 位置 continuity physics 點（n_t=1, start=1000，僅 continuity）：KE **7.80%**（EXP-064, -0.85pp）；div_l2 **0.184**（-9.6%）；phase_err -53%。**為 K=100 配置的最終結案值**。
- band_mid/high@t=5 ≈100% 經四次 falsifiability 驗證後，確認為 **K=100 sensor 的資訊論硬上限**。Wavelet 稀疏性診斷量化確認 CS 精確重建需 M ≥ O(s log N) ≈ 5000 sensors，K=100 差約 50 倍；換 wavelet 基底不改變上限量級。詳見 [`docs/analysis_reports.md`](analysis_reports.md)。
- K=200（EXP-066）部分突破 band_mid（32.90% vs 99.97%），但低頻退步（38.65% vs 3.62%）+ 整體 KE 退步（29.94%）；L_phys@10k 未充分收斂，需延伸訓練驗證。
- AIM（Approximate Inertial Manifold）zeroth-order 後處理已證偽（τ_visc/τ_NL ≈ 215，quasi-static 假設違反）。詳見 [`docs/analysis_reports.md`](analysis_reports.md)。

### F. Cylinder Wake（非週期域）

- **非週期域必須加 inflow BC loss**：CEXP-001（無 BC）KE 51%，根因感測器 100% 集中尾跡，來流區無 supervision；加 `bc_loss_weight=0.1, bc_inflow_u=0.33 m/s, bc_n_points=64` 後 CEXP-002 KE 降至 3.5%（**14.5× 改善**）。Kolmogorov（週期域）不需要 BC。

### G. 後 K=100 結案實驗（運行於 denorm 路徑下，待重評）

> ⚠️ **以下三組均跑在 `physics_output_denormalization` 啟用路徑下**（自 d62e698 commit 自動觸發），物理 NS residual 量級被改變。完整 diagnostic 與 `PINN_DISABLE_PHYS_DENORM=1` 對照見 [`docs/analysis_reports.md`](analysis_reports.md)。重跑後需重評本節結論。

- **EXP-067**（CfC log_tau (-3,1) + 頻率分層 LearnableFourier (1,4,12)/(50/37.5/12.5%), 10k 步）：KE **11.20%**（vs EXP-064 7.80%，+3.40pp）；band_low 退步（7.19% vs 3.62%）。診斷：(a) 頻率分層 σ=12 高頻段微改善 band_mid 但犧牲 12.5% 通道；(b) CfC fast channels（τ≈0.05）相對 sensor dt=0.025 過敏感。**建議拆解 EXP-067a/b 單獨測試**。
- **EXP-068**（PINN causal weighting eps=1.0 num_bins=16, 10k 步）：KE 9.73%（+1.93pp）；div_l2 **0.680（+269% 嚴重退步）**。當前實作以「所有殘差項之和」做 cumsum，量級較大的 momentum 殘差主導權重曲線，continuity 約束被進一步壓制。**修正建議**：改 per-task cumsum 或僅以 momentum 殘差驅動權重。
- **EXP-069**（三項組合：CfC tau + 頻率分層 + causal weighting, 10k 步）：KE **20.13%（+12.33pp 災難）**；div_l2 1.404（+663%）。三項負面交互證實；皆需單獨修正後再組合。

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

> **註**：原 `[ANALYSIS] Wavelet Sparsity Diagnostic`（2026-04-26）與 `[ANALYSIS] AIM Diagnostic`（2026-04-26）已搬移至 [`docs/analysis_reports.md`](analysis_reports.md)（2026-05-06 拆檔）。其結論被引用於下方 K=100 結案聲明。

---

## [STATE] K=100 稀疏重建結案聲明（2026-04-26）

**EXP-064 為 K=100 sensor 配置的最終接受結果。稀疏重建主線結案。**

### 量化結論

K=100 已達資訊論硬上限，由 Wavelet 稀疏性診斷（item 35）量化確認：

| 頻帶 | 能量佔比 | 所需 wavelet 自由度 | K=100 可行性 | EXP-064 誤差 |
|------|----------|---------------------|-------------|-------------|
| Low（k≤8） | 94.4% | ~196 | ✓ 可重建 | **3.62%** |
| Mid（k~8..16） | 4.8% | ~588 | ✗ 超出容量 | ~100% |
| High（k~16..32） | 0.8% | ~1452 | ✗ 遠超容量 | ~100% |

CS 精確重建需 M ≥ O(s log N) ≈ 5000 sensors（s≈328，N=65536）；K=100 差約 50 倍。
換成 Fourier 基底不改變結論，自由度上限與基底選擇無關。

### 結案判斷

- 所有已試優化方向（optimizer、physics loss 密度、sensor continuity、trunk 加深）皆無法突破 band_mid/high 上限
- 低頻主能量帶（94.4%）已被可靠重建；整體 KE 7.80% 是此設定下的最佳可達值
- 進一步提升高頻需要根本性增加感測器覆蓋（K≥5000）或引入 DNS 高頻先驗

---

## [STATE] Cylinder Wake — 新主線建立（2026-04-27）

### 背景與目標

完成 Kolmogorov 稀疏重建研究後，轉向 RealPDEBench Cylinder Wake 案例：
- 非週期非均勻格（domain: [0, 0.325] × [0, 0.178]，含 cylinder body）
- K=100 QR-pivot sensor（Re=10031，T=3990 frames，dt=0.005s）
- 目標：驗證 Pi-LNN 能否在非週期域建立 baseline，為與 FLRNet / Energy Transformer 比較做準備

### 資料設定

- Arrow shard: `RealPDEBench/data/realpdebench/cylinder/hf_dataset/numerical/data-00000-of-00092.arrow`
- Re=10031（sim_id=10031.h5），T=3990, H=128, W=256（非均勻格）
- sensor 生成：`scripts/generate_sensors_qrpivot_cylinder.py`，`data/cylinder_sensors/`
- `sensor_subsample=20`：T=3990 → T=200（dt=0.1s），對齊 Kolmogorov 計算量
- 座標正規化至 [0,1]²（domain_length=1.0）

### Cylinder 實驗結果

#### CEXP-001：無 BC baseline（KE=51%，失敗）

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_001_k100.toml` |
| Artifact | `artifacts/deeponet-cfc-cylinder-exp001-k100-warmup` |
| Checkpoint | `checkpoints/lnn_kolmogorov_step_10000.pt` |
| KE rel-err mean | **51.0%** |
| u RMSE mean | 2.47e-1 |
| v RMSE mean | 9.99e-2 |
| div L2 mean | 1.13 |
| 結論 | [RESULT: PHYSICAL_FAILURE]：感測器全部集中尾跡（x>0.10），無 inflow BC 約束，模型在來流區輸出 u≈0 而非 u≈0.33 m/s，導致 KE 系統性低估 50%。 |

#### CEXP-002：Inflow BC Loss（KE=3.5%，成功）

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_002_k100_bc.toml` |
| Artifact | `artifacts/deeponet-cfc-cylinder-exp002-k100-bc` |
| Checkpoint | `checkpoints/lnn_kolmogorov_step_10000.pt` |
| KE rel-err mean | **3.5%** |
| KE rel-err late | 3.9% |
| u RMSE mean | 1.03e-1 |
| v RMSE mean | 1.06e-1 |
| div L2 mean | 1.14 |
| 修改內容 | `bc_loss_weight=0.1`，`bc_inflow_u=0.33 m/s`，`bc_n_points=64`（x=0 均勻採樣） |
| 結論 | **BC loss 錨定來流速度後，KE 從 51% 降至 3.5%（14.5× 改善）**，與 Kolmogorov EXP-064（7.8%）相當。KE(t) 振盪幅值略大（峰值比 DNS 高 ~10%），渦街結構可識別。div L2=1.14 仍高，Kármán 渦核位置有偏移，但整體可視為 cylinder 稀疏重建 **baseline 建立**。 |

#### 訓練紀錄摘要

| Step | L_data | L_phys | w_ns_u | w_cont | t_max |
|---|---|---|---|---|---|
| 1 | 6.676e+0 | 2.64e-1 | 0.010 | 0.010 | 0.5 |
| 1000 | 6.74e-3 | 9.99e-1 | 0.016 | 0.012 | 7.0 |
| 3000 | 1.46e-3 | 3.30e-1 | 0.024 | 0.016 | 20.0 |
| 6000 | 9.49e-4 | 1.01e-1 | 0.074 | 0.023 | 20.0 |
| 10000 | 1.15e-3 | 3.25e-2 | 0.108 | 0.038 | 20.0 |

### NaN 根因診斷（已修復）

**症狀**：CEXP-001 早期 step_500 checkpoint 有 83/95 個參數是 NaN。

**診斷流程**：
1. Physics OFF → 訓練穩定（L_data 在 step 400 降至 0.088）→ NaN 來自 physics
2. 物理殘差分解 → second derivatives（du_dx2, du_dy2）有 NaN，first derivatives 正常
3. NaN 點的 nearest_sensor_distance = 0 → collocation point 落在 sensor 位置

**根本原因**：`torch.linalg.norm(rel, dim=-1)` 在 `rel=0`（query = sensor）時，second-order autograd 計算 `∂²|r|/∂r² = (|r|²I - rr^T)/|r|³`，在 r=0 為 0/0 = NaN。

**修法**（`src/lnn_kolmogorov.py` DeepONetCfCDecoder.forward）：
```python
# 舊：rel_r = torch.linalg.norm(rel, dim=-1, keepdim=True)
# 新：
rel_r = torch.sqrt((rel**2).sum(dim=-1, keepdim=True) + 1e-8)
```
20 trials 驗證，NaN rate 0/20。

---

## [DIAGNOSTIC] Physics Output Denormalization Silent Regression（2026-05-06，進行中）

`d62e698 feat(cylinder+physics)` 為 cylinder 加入的 `physics_output_denormalization` 在 [`src/pi_lnn/training.py:178`](../src/pi_lnn/training.py) 沒有 opt-out flag，**Kolmogorov 主線完全滿足自動觸發條件**，導致 EXP-070+ 所有 Re=10000 主線實驗都跑在 denorm 啟用路徑下，物理 NS residual 量級被改變。

**已驗證**：smoke 對照（`PINN_DISABLE_PHYS_DENORM=1`）證明 denorm OFF 時 step 1 L_phys byte-identical EXP-064 baseline；denorm ON 時 L_phys 縮 ~5×、w_ns_u 漲 4.5×。

**已修補**（diagnostic toggle）：
- `src/pi_lnn/training.py`：加 `PINN_DISABLE_PHYS_DENORM=1` 環境變數 toggle（預設不變）
- `SOAP/soap.py`：修 `_linalg_eigh_mps` dtype/device 順序 bug

**待驗證**：用 `PINN_DISABLE_PHYS_DENORM=1` 重跑 EXP-070 滿 10k 步，判斷 KE=84% 是 denorm 路徑量級不匹配還是 AL 設計本身失敗。

**完整診斷報告**（含量級分析、時間線、修法計畫）：見 [`docs/analysis_reports.md`](analysis_reports.md)。

---

## [STATE] Open Question

| 問題 | 現況 | 狀態 |
|---|---|---|
| amplitude ratio=0.9965 是否 overfitting | EXP-015 更高（0.9965），需確認是否對訓練時段過度擬合；若有新時段資料可做 OOD 測試 | 開放（低優先） |
| K=200 band_mid 突破後，低頻退步是否可藉延伸訓練恢復 | EXP-066 L_phys@10k=2.95（未充分收斂）；K=200 主線暫停 | **CLOSED**：K=100 主線結案，K=200 屬另一資料密度配置，如重啟需獨立實驗 |
| 高頻重建的可行路徑 | CS 理論確認：K=100/200 均遠低於 ~5000 門檻；zeroth-order AIM 已證偽 | **CLOSED**：高頻不可達為數學必然，未來路徑需 DNS POD 先驗或 4D-Var（工程不可遷移）|
| EXP-070 KE=84% 是否因 denorm 路徑量級不匹配（vs AL 設計失敗） | DIAGNOSTIC 區證實 denorm 改變 baseline 量級；EXP-070 重跑加 `PINN_DISABLE_PHYS_DENORM=1` 進行中 | **進行中**（2026-05-06）|

---

## [INDEX] Cylinder Active

| ID | Status | 主題 | 一句結論 |
|---|---|---|---|
| `CEXP-002` | `ACTIVE_BASELINE` | Cylinder, K=100, **inflow BC loss**（bc_w=0.1, u_inf=0.33） | **KE 3.5%（Cylinder baseline）**；BC loss 從根本解決來流 collapse；振盪幅值略大（+10%）；div L2=1.14 仍有改善空間 |
| `CEXP-001` | `NEGATIVE_RESULT` | Cylinder, K=100, 無 BC loss | KE 51%（[PHYSICAL_FAILURE]）；無 inflow BC 導致來流區 u→0；已被 CEXP-002 取代 |

---

## [INDEX] Active

> 僅列出當前主線與最近 4 筆實驗。歷史 EXP 整理為 GROUP 索引；個別 RECORD 詳情見 [`docs/experiment_archive_kolmogorov.md`](experiment_archive_kolmogorov.md)。

### 當前主線

| ID | Status | 主題 | 一句結論 |
|---|---|---|---|
| `EXP-030` | `ACTIVE_BASELINE` | Re=1000：SOAP+SF resume EXP-028 → 5000 steps | **KE 9.61%、amp 1.027、u RMSE 5.68e-2；首次破 10%**（archive G7）|
| `EXP-064` | `ACTIVE_BASELINE` | Re=10000：EXP-063 + sensor continuity（n_t=1, start=1000）| **KE 7.80%、div_l2 0.184、phase_err -53%；K=100 結案值**（archive G14，主檔有完整 RECORD）|

### 最近實驗（K=100 結案後探索 + K=200 嘗試）

| ID | Status | 主題 | 一句結論 |
|---|---|---|---|
| `EXP-066` | `MIXED_RESULT` | Re=10000, K=200 sensor 冷啟動 10k | band_mid_last **32.90%（首次突破）**；KE mean 29.94%（退步）；L_phys@10k 未飽和，需延伸訓練（主檔有完整 RECORD）|
| `EXP-065` | `NEGATIVE_RESULT` | Re=10000, EXP-064 + trunk MLP 1→2 層 | KE 7.74%（持平）；band_mid/high@t=5 仍 ≈100%；**K=100 資訊論硬上限再次確認**（主檔有完整 RECORD）|
| `EXP-063` | `POSITIVE_RESULT` | Re=10000, jaxpi SOAP + GradNorm（→ EXP-064 直接前驅）| KE 8.65%；div_l2 0.204 全系列最佳（-64% vs EXP-062）（archive G14）|
| `EXP-062` | `POSITIVE_RESULT` | Re=10000, LearnableFourierEmb（embed_dim=128, σ=2.0）| KE 10.4%；band_low 5.8% 但 mid/high≈100%（頻譜集中低頻）（archive G14）|

### 歷史群組（archive 內含完整 RECORD）

| GROUP | EXP 範圍 | 群組角色 | 群組 status |
|---|---|---|---|
| **G14** | EXP-062~063 | LearnableFourier 演進 → EXP-064 baseline（已上方列出）| `ACTIVE`（EXP-064）|
| **G13** | EXP-057~061 | 冷啟動 IC weight 系列；EXP-057 為冷啟動 + IC weight 最佳（KE 20.6%, div 0.796 全系列最低）；EXP-058 暫停；EXP-059~061 證偽 | `RESOLVED` |
| **G12** | EXP-049~056 | EXP-048 resume 變體；**EXP-055 IC weight（KE 17.1%）為主要正向**；EXP-054 RAR freq=1000（KE 19.6%）；EXP-049~053, 056 證偽 | `RESOLVED` |
| **G11** | EXP-044~047 | d=256 從頭 3k 失敗群（GradNorm/sweep/locality 證偽）| `RESOLVED` |
| **G10** | EXP-043, 048 | d=256 漸進收斂線（3k→5k→10k：31.5%→27.2%→21.8%）| `SUPERSEDED`（被 G14）|
| **G9** | EXP-040~042 | Transfer learning 失敗（架構不匹配 + 負遷移）| `RESOLVED` |
| **G8** | EXP-031~033, 035~039 | Re=10000 新資料容量 + physics loss 失敗；EXP-031（d=128, KE 39.4%）/ EXP-033（d=256, KE 31.5%）為基準 | `RESOLVED` |
| **G7** | EXP-026~030 | Re=1000 SOAP+SF 主線 → EXP-030 baseline（已上方列出）| `ACTIVE`（EXP-030）|
| **G6** | EXP-023~025 | Re=1000 SF vs stepLR 5k 對照；前主線 EXP-025（KE 12.06%, amp 0.995）| `SUPERSEDED`（被 G7）|
| **G5** | EXP-021~022 | Re=1000 spatial encoding（periodic Fourier + isotropic relpos）；KE 0.251→0.153（-39%）| `SUPERSEDED`（被 G6/G7）|
| **G4** | EXP-016~020 | Re=10000 舊資料容量探索；舊 DNS 已棄用；舊資料最佳 EXP-019 amp 0.595 | `SUPERSEDED`（舊 DNS）|
| **G3** | EXP-013~015 | Re=1000 anchor 系列（top-k 失敗 + phase + temporal）| `SUPERSEDED`（被 G5）|
| **G2** | EXP-007, 008, 010~012 | Re=1000 baseline 確立（rff=4→Small→k_f=2→stepLR）；前主線 EXP-012（KE 0.318）| `SUPERSEDED`（被 G3）|
| **G1** | EXP-001~006, 009 | Re=1000 早期 smoke + collapse 尺度診斷 | `RESOLVED`（根因定位）|

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

## [RECORD] EXP-066

- Status: `MIXED_RESULT`
- Time: `2026-04-26` 設計，`2026-04-26` 完成訓練與評估
- Topic: Re=10000, 冷啟動 10000 步，**EXP-064 recipe + K=200 sensor set**
- Config: `configs/exp_066_re10000_xlarge_k200.toml`
- Artifact: `artifacts/deeponet-cfc-re10000-exp066-k200`
- Evaluated Checkpoint: `artifacts/deeponet-cfc-re10000-exp066-k200/checkpoints/lnn_kolmogorov_step_10000.pt`

- Compare vs EXP-064（唯一改動）：
  - `sensor_jsons/npzs` → K=200 QR-pivot（`sensors_qrpivot_K200_N256_t0-5_si100`）
  - sensor_physics: K=200 × n_t=1 = 200 continuity points/step
  - 其他所有 hyperparameter 與 EXP-064 完全相同

- Hypothesis: K=200 QR-pivot 分析 acc>0.8 上限 k=41（K=100 為 k=20）；band_mid@last < 90%；KE ≤ 7.80%
- Falsifiability: 若 band_mid/high@last 仍 ≈100% → 確認 sensor 數量非瓶頸，需考慮 sensor 布局策略或其他先驗

- Training Progress：

| Step | L_data | L_phys | L_total | t_max |
|------|--------|--------|---------|-------|
| 1000 | 2.46e-1 | 24.67 | 5.10e-1 | 2.0 |
| 3000 | 1.19e-1 | 7.03 | 2.14e-1 | 5.0 |
| 5000 | 6.67e-2 | 5.40 | 1.51e-1 | 5.0 |
| 10000 | **4.62e-2** | **2.95** | **1.01e-1** | 5.0 |

- Evaluation Metrics（step_10000）：
  - `ke_rel_err_mean = 0.2994`（KE **29.94%**，退步 vs EXP-064 7.80%）
  - `ens_rel_err_mean = 0.2760`（27.6%）
  - `div_l2_mean = 2.493`（退步，vs EXP-064 0.184；vs DNS 0.092）
  - `band_low_last = 38.65%`（退步，EXP-064: 3.62%）
  - `band_mid_last = 32.90%`（**首次突破！EXP-064: 99.97%**）
  - `band_high_last = 91.41%`（部分改善，EXP-064: 99.99%）

- Comparison Table：

| Metric | EXP-064 (K=100) | EXP-066 (K=200) | Δ |
|--------|:---------------:|:---------------:|:--:|
| KE mean | **7.80%** | 29.94% | +22.1pp |
| band_low@last | **3.62%** | 38.65% | +35.0pp |
| band_mid@last | 99.97% | **32.90%** | -67.1pp ← 突破 |
| band_high@last | 99.99% | 91.41% | -8.6pp |

- Decision: **Mixed Result（Hypothesis partially confirmed）**。
  1. band_mid_last 確實從 99.97% 降至 32.90%——確認資訊論假設：K=200 覆蓋更多中頻模態，資訊論瓶頸是 K 而非架構。
  2. 但 K=100→K=200 同時造成低頻退步（3.62%→38.65%）和整體 KE 惡化（7.80%→29.94%）。
  3. L_phys@10k=2.95 未充分收斂（EXP-064 對應值更低），推測低頻退步主要源於訓練步數不足，而非 K=200 的根本性缺陷。
  4. **後續：** 考慮 resume EXP-066 至 20k 步，驗證低頻是否能在更長訓練後收斂；若 20k 步後 KE mean < 15% 且 band_mid_last < 40%，K=200 策略值得進一步投入。

---

## [RECORD] EXP-065

- Status: `NEGATIVE_RESULT`
- Time: `2026-04-25` 設計與訓練，`2026-04-25` 評估完成
- Topic: Re=10000, 冷啟動 10000 步，**EXP-064 + trunk query MLP 加深 1→2 層**
- Config: `configs/exp_065_re10000_xlarge_trunk2.toml`
- Artifact: `artifacts/deeponet-cfc-re10000-exp065-trunk2`

- Compare vs EXP-064（唯一改動）：
  - `num_query_mlp_layers = 2`（EXP-064: 1）
  - 新增 1 個 `ResidualMLPBlock(256, 256)` → +132K params（3.14M → 3.27M）

- Hypothesis: 更深 trunk query MLP 使模型具備表達 k>5 空間模態的能力；band_mid/high@t=5 < 100%；KE ≤ 7.80%
- Falsifiability: 若 band_mid/high@t=5 仍 ≈100% → 確認根本瓶頸為 K=100 sensor 資訊論硬上限，架構表達力非解法

- Results:
  - `ke_rel_err_mean = 0.0774`（KE **7.74%**，微幅改善 -0.06pp vs EXP-064）
  - `u_rmse_mean = 0.0679`，`v_rmse_mean = 0.0617`（小幅改善）
  - `div_l2_mean = 0.196`（退步，+6.5% vs EXP-064 0.184）
  - `kf_amp_ratio_last = 0.965`（持平）；`kf_phase_err_last = -0.021 rad`（微改善）
  - `band_low@last = 4.82%`；**`band_mid@last = 99.88%`**；**`band_high@last = 99.998%`**（**≈100%，完全無改善**）

- Decision: **Negative（Hypothesis falsified）**。trunk 加深對中高頻重建完全無效。結合 EXP-062~065 的系列實驗，band_mid/high ≈100% 已通過四次 falsifiability 驗證（optimizer、physics 密度、sensor 位置、trunk 表達力），**最終確認根本原因是 K=100 sensor 的資訊論硬上限**，非任何訓練或架構問題。唯一出路是增加 sensor 覆蓋（K↑）或引入高頻先驗（e.g. spectral regularization）。EXP-064 維持 Re=10000 baseline。

---

## [RECORD] EXP-064

- Status: `ACTIVE_BASELINE`
- Time: `2026-04-24` 設計，`2026-04-25` 完成訓練與評估
- Topic: Re=10000, 冷啟動 10000 步，**EXP-063 + 感測器位置 continuity physics 點**
- Config: `configs/exp_064_re10000_xlarge_sensor_physics.toml`
- Evaluated Checkpoint: `artifacts/deeponet-cfc-re10000-exp064-sensor-physics/checkpoints/lnn_kolmogorov_step_10000.pt`
- Compare vs EXP-063（唯一改動）：
  - `use_sensor_physics = true`（EXP-063: false）
  - `num_sensor_physics_time_samples = 1`（每步 K=100 感測器 × 1 時間步 = 100 額外 continuity 點）
  - `sensor_physics_start_step = 1000`（前 1000 步只用隨機 collocation）
  - Note: sensor physics 只計算 continuity（`∂u/∂x + ∂v/∂y=0`），不計算 momentum（∂²u/∂x² 在 sensor 位置溢位 float32）
- Hypothesis: 感測矩陣分析（K=100, k≤16, 條件數≈11）確認感測器位置對中高頻有最佳分辨率；加入 sensor continuity 期望 band_mid@t=5 從 99.97% 改善，KE ≤ EXP-063（8.65%）
- Falsifiability: 若 band_mid/high@t=5 仍 ≈100% → 問題不在感測器物理約束密度，需從模型表達力（trunk 深度/寬度）或感測器策略著手
- Metrics（step_10000）：
  - `ke_rel_err_mean = 0.0780`（KE **7.80%**，**Re=10000 新紀錄**，-0.85pp vs EXP-063）
  - `ens_rel_err_mean = 0.2910`（29.1%）
  - `div_l2_mean = 0.1843`（**Re=10000 全系列最佳**，-9.6% vs EXP-063 0.204）
  - `kf_amp_ratio_last = 0.9615`（持平 EXP-063 0.9636）
  - `kf_phase_err_last = -0.0228 rad`（**-53% vs EXP-063 -0.0489 rad**）
  - `u_rmse_mean = 0.0689`，`v_rmse_mean = 0.0621`
  - `band_energy_rel_err_last`: low **3.6%** / mid **99.97%** / high **100.0%**
  - worst_time = t=0.10，worst_ke = 29.96%（t=0 附近 IC 仍不穩）
  - `ns_u_rms_mean = 0.523`，`ns_v_rms_mean = 0.506`（NS residual 仍非零，物理約束未完全滿足）
- Decision: **Positive（新 Re=10000 baseline）**。KE 7.80% 為 Re=10000 新紀錄；sensor continuity 對 div_l2（-9.6%）與 phase reconstruction（-53%）有實質貢獻。**Hypothesis falsified**：band_mid/high@t=5 ≈100% 無改善，確認中高頻失敗根源為模型表達力或感測器策略問題，非 physics 約束密度問題。下一步方向：增加 trunk MLP 深度（num_query_mlp_layers 1→2 或 3）或考慮更深的 query encoder。
- Supersedes: EXP-063

---

> **歷史 RECORD 已搬移**：EXP-001 ~ EXP-063（Re=1000 主線、Re=10000 早期/中期、所有失敗診斷）詳細指標與超參見 [`docs/experiment_archive_kolmogorov.md`](experiment_archive_kolmogorov.md)（2026-05-06 拆檔）。本主檔僅保留近期 RECORD（EXP-066, 065, 064）作為當前主線基準，更早期細節需查詢時再讀 archive。

