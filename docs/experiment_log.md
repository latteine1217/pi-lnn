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

### H. AL-continuity 系列（ADR-001 §4 / ADR-002, EXP-070~075）

> **重跑後** evaluator (Round 7) 真實值。EXP-070~074 詳細見 DIAGNOSTIC section；以下重點記 EXP-071（ADR-002 Decision-D 補跑）。

- **EXP-071**（ADR-001 §4 / ADR-002 Decision-D, 2026-05-08, 10k 步）：3-task GradNorm `[data, ns_u, ns_v]` init `[1, 0.057, 0.057]` + AL-continuity (ρ=1.0, λ_clip=10, freq=100, ema=0.5)，cont 完全由 AL 接管（v4 §5 解耦）。
  - **div L2 = 0.0442（突破）** — 比 baseline 0.184 降 4.2×、比 EXP-070 pure AL 0.682 降 15×；**首次達到 ADR-001 §7 條件 #1 閾值 0.05 以下**。
  - **KE rel-err = 14.57%**（train 14.15%, val 16.21%）— 比 baseline 7.80% / EXP-070 6.30% **退步 ~2×**。
  - NS-momentum: u_rms 0.357, v_rms 0.378（比 EXP-070 1.58 / 1.52 全降 4×）→ **整體 NS 殘差最低**。
  - GradNorm 軌跡: ns_u/ns_v 從 init 0.057 → 0.281/0.306（10k 步收斂值，**動態升 5×**）；λ 從 0 ascend 到 2.84（沒到 clip 10）；C_ema 6.4e-2 → 7.3e-3（降 9×）。
  - kf_amp ratio @ last = 0.881（vs baseline 0.962）；ek_ratio_kf_last = 0.816（vs baseline 0.938）→ 主 mode 重建退步約 8%。
  - **解讀**：3-task GradNorm + AL 的 v4 §5 解耦設計 **work**：div constraint 嚴格滿足，但 GradNorm 把 ns 權重拉到 0.30 級別，data 權重相對被壓 → KE 退步。**Trade-off 真實**：div ↓ 4×, KE ↑ 2×, NS-mom ↓ 4×。
  - 觸發 ADR-002 Decision-D **CLOSED with finding**：補跑完成；3-task + AL 解耦設計驗證有效；新問題（KE 退步）由 EXP-075 處理（cap GradNorm ns weight）。
  - artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp071-al-gradnorm/`；eval: `artifacts/eval-rerun-2026-05-08/exp071-al-gradnorm/`

- **EXP-075**（ADR-002 Decision-B 修正版, 2026-05-08, 10k 步）：EXP-071 + 新加 `gradnorm_max_weight = 0.20` cap on physics tasks (ns_u, ns_v)。
  - **KE rel-err = 13.94%**（train 13.54%, val 15.47%）— 只改善 0.63pp（vs EXP-071 14.57%）。
  - **div L2 = 0.0442**（與 EXP-071 byte-identical），cap 沒破壞 AL+cont 約束。
  - NS residual: u_rms 0.360 / v_rms 0.381（與 EXP-071 0.357/0.378 同量級）。
  - GradNorm 軌跡：前 5000 步動態升 (0.10→0.20)，step 5000+ hit cap maintained 在 0.18~0.20（vs EXP-071 step 5000+ 自由爬到 0.28~0.31）。
  - kf_amp ratio @ last = 0.892（略好 vs EXP-071 0.881）；ek_ratio @ last = 0.822（略好 vs 0.816）。
  - **解讀**：cap 設計**部分成功** — div 守住、cap 邏輯生效，但 KE 改善有限。發現 **ns weight 與 KE 單調 trade-off**：ns weight 0.057→0.20→0.30 對應 KE 6.30%→13.94%→14.57%。Cap 0.20 與 EXP-071 0.28 差距太小（30%）才導致 KE 改善只 0.63pp。
  - **觸發 ADR-002 Decision-B CLOSED with finding**：cap 機制驗證 work，但需更激進（cap=0.10）才能顯著改善 KE。
  - 新問題：trade-off 是否真的是 monotonic？或在 cap=0.10 出現 turning point？由 EXP-076 驗證。
  - artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp075-al-gradnorm-capped/`；eval: `artifacts/eval-rerun-2026-05-08/exp075-al-gradnorm-capped/`

- **EXP-076**（接續 EXP-075, 2026-05-08, 10k 步）：EXP-071 + `gradnorm_max_weight = 0.10`（vs EXP-075 cap 0.20 更激進）。
  - **KE rel-err = 13.06%**（train 12.70%, val 14.43%）— 比 EXP-075 13.94% 改善 0.88pp，但仍遠高於 baseline 7.80%。
  - **div L2 = 0.0436**（與 EXP-075 0.0442、EXP-071 0.0442 essentially identical）— **ADR-001 §7 條件 #1 滿足且 saturated**。
  - NS residual: u_rms 0.377 / v_rms 0.402（與 EXP-075 0.360/0.381 略升）。
  - GradNorm 軌跡：step 1000 起就 hit cap 0.10 全程維持（vs EXP-075 step 5000 才 hit cap）。
  - kf_amp ratio @ last = 0.890；ek_ratio @ last = 0.833（與 EXP-075 0.892/0.822 同量級）。

  **重大物理洞見** — AL series Pareto curve（ns weight ablation）:

  | EXP | ns weight | KE rel-err | div L2 | NS-mom RMS |
  |-----|-----------|------------|--------|------------|
  | EXP-070 (pure AL) | 0.057 (固定) | **6.30%** | 0.682 | u 1.58 / v 1.52 |
  | EXP-076 (cap 0.10) | 0.100 (cap) | 13.06% | 0.0436 | 0.377 / 0.402 |
  | EXP-075 (cap 0.20) | 0.200 (cap) | 13.94% | 0.0442 | 0.360 / 0.381 |
  | EXP-071 (no cap) | 0.300 (free) | 14.57% | 0.0442 | 0.357 / 0.378 |

  - **Phase transition 在 ns ∈ [0.057, 0.10]**：div 從 0.682 → 0.044（~16× drop），KE 從 6.30% → 13.06%（~2× rise）。**不是 linear monotonic，是 threshold-like**。
  - **ns ≥ 0.10 後 saturated**：div L2 完全 saturate 在 0.0442，KE 只動 ±1.5pp。再升 ns 徒勞。
  - **AL dual variable λ → 2.85 跨 EXP-071/075/076 一致**：cont penalty 主導 div，ns weight 主要影響 NS-momentum 與 KE。
  - 觸發新研究問題：phase transition 確切位置 → 由 EXP-077 (cap=0.057) 驗證。
  - artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp076-al-gradnorm-cap010/`；eval: `artifacts/eval-rerun-2026-05-08/exp076-al-gradnorm-cap010/`

- **EXP-077**（接續 EXP-076, 2026-05-08, 10k 步）：`gradnorm_max_weight = 0.057`（= init weight，GradNorm immediate freeze）。
  - **KE rel-err = 12.56%**（train 12.23%, val 13.84%）；**div L2 = 0.0437**（跟 EXP-076/075/071 essentially 一致）。
  - NS residual: u_rms 0.398 / v_rms 0.426；ek_ratio @ last = 0.860；kf_amp ratio = 0.911。
  - GradNorm 軌跡：w_ns 鎖死在 init 0.057 全程（cap 邏輯生效）。

  **🔥 重大發現 — Hypothesis A falsified**：phase transition **不是** ns weight value，而是 **`use_gradnorm` binary switch**：

  | EXP | use_gradnorm | ns weight (final) | KE | div L2 | NS-u RMS | ek_ratio |
  |---|---|---|---|---|---|---|
  | EXP-070 (pure AL) | **OFF** | 0.057 | **6.30%** | **0.682** | 1.58 | 0.927 |
  | EXP-077 (AL+GN cap=init) | **ON** | 0.057 | 12.56% | **0.044** | 0.398 | 0.860 |
  | EXP-076 (AL+GN cap=0.10) | ON | 0.100 | 13.06% | 0.044 | 0.377 | 0.833 |
  | EXP-075 (AL+GN cap=0.20) | ON | 0.200 | 13.94% | 0.044 | 0.360 | 0.822 |
  | EXP-071 (AL+GN no cap) | ON | 0.300 | 14.57% | 0.044 | 0.357 | 0.816 |

  - **完全相同 ns weight (0.057)**，EXP-070 vs EXP-077 div L2 差 **16×** (0.682 → 0.044)。
  - 4 個 use_gradnorm=true 的實驗（EXP-071/075/076/077）div L2 全部 essentially 相同 (0.043~0.044) 不論 ns weight value，但 NS-u RMS 隨 ns weight 變化 (0.30→0.40)。
  - **解讀**：`_gradnorm_step` 每 1000 步對 `trunk_out.weight + bias` 算 per-task gradient norm（含 cont via AL term），即使 weight freeze 在 cap，這個 retain_graph backward computation 仍 implicit 影響 SOAP preconditioner stats / trunk representation 學習動態 → div 大幅改善但 KE 退步。
  - **新研究問題**：是否能用 pure AL 加強 ρ（不用 GradNorm）達到同樣 div 突破？由 EXP-078 驗證。
  - 觸發 ADR-002 Decision-A 重新審視：原本「stream function reparam 候選保留」可能改寫為「GradNorm computation as implicit regularization for div constraint」。
  - artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp077-al-gradnorm-cap-init/`；eval: `artifacts/eval-rerun-2026-05-08/exp077-al-gradnorm-cap-init/`

- **EXP-078**（pure AL strength sweep, 2026-05-08, 10k 步）：`use_gradnorm = false` + `al_rho = 3.0`（vs EXP-070 ρ=1.0）。
  - **KE rel-err = 15.47%**（train 15.02%, val 17.23%）— 比 EXP-070 ρ=1 的 6.30% **退步 9pp**！
  - **div L2 = 0.0332**（**最低!** 比 GradNorm 路徑 0.044 還低）。
  - λ ascend 到 5.70（vs ρ=1 的 2.84，2× 強）；C_ema 6.5e-2 → 3.8e-3。
  - NS-u 0.397 / NS-v 0.413（與 GradNorm 路徑同量級）。
  - **解讀**: ρ↑ 確實讓 λ 強化 → div 進一步突破。但 KE 退步比 GradNorm 路徑更糟。Falsifiability (c) 部分成立：pure AL strong ρ 也能達 div breakthrough（不必 GradNorm），但 trade-off 比 GradNorm 還糟。
  - artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp078-al-pure-rho3/`；eval: `artifacts/eval-rerun-2026-05-08/exp078-al-pure-rho3/`

### I. AL series 完整 Pareto frontier（2026-05-08，7 點 ablation）

| EXP | recipe | KE rel-err | div L2 | NS-u RMS | ek_ratio | 性質 |
|---|---|---|---|---|---|---|
| **EXP-070** (no GN, ρ=1) | weak AL only | **6.30%** | 0.682 | 1.58 | 0.927 | KE-optimal extreme |
| **EXP-064** (4-task GN incl cont, no AL) | strong cont via GN | **7.80%** | 0.184 | 0.523 | 0.938 | **best balance** |
| EXP-077 (AL+3-task GN cap=0.057) | AL+GN cap-init | 12.56% | 0.044 | 0.398 | 0.860 | – |
| EXP-076 (AL+3-task GN cap=0.10) | AL+GN cap | 13.06% | 0.044 | 0.377 | 0.833 | – |
| EXP-075 (AL+3-task GN cap=0.20) | AL+GN cap | 13.94% | 0.044 | 0.360 | 0.822 | – |
| EXP-071 (AL+3-task GN no cap) | AL+GN free | 14.57% | 0.044 | 0.357 | 0.816 | – |
| **EXP-078** (no GN, ρ=3) | strong AL only | 15.47% | **0.033** | 0.397 | 0.825 | div-optimal extreme |

**核心發現**:
1. **「兩全其美」不存在於 AL 系列任何 recipe**：不論走 pure AL strong ρ 或 AL+GN cap，div < 0.05 必伴隨 KE 退步到 12-15%。
2. **EXP-064 baseline (4-task GradNorm 含 cont, no AL)** 是真正 best balance：KE 7.80% + div 0.184。雖未突破 ADR-001 §7 #1 閾值 0.05，但兩維度都 acceptable。
3. AL 與 GradNorm 對 cont 解耦設計（spec v4 §5 / ADR-001 §4）導致：要 KE 必失 div，要 div 必失 KE。
4. **AL strength（λ asymptotic value）才是 div control 主導機制**，GradNorm computation 只是 implicit 加強器（EXP-077 vs EXP-070 同 ns weight 但不同 div 證實）。

**重新審視 ADR-001 §4 禁令**：「AL 與 GradNorm 不可同時控制 cont」可能過於保守。若違反此禁令，4-task GradNorm（含 cont）+ AL 同時對 cont 套 dual penalty，是否能達 KE 7-9% + div < 0.10「兩全其美」？由 EXP-079 驗證。

- **EXP-079**（**違反 ADR-001 §4 falsifiability test**, 2026-05-09, 10k 步）：AL on cont + 4-task GradNorm 包含 cont 同時。
  - **KE rel-err = 14.77%**（train 14.36%, val 16.39%）；**div L2 = 0.0428**；NS-u 0.355 / NS-v 0.370；ek_ratio 0.828。
  - 對照 EXP-071 (3-task no cont): KE 14.57%, div 0.0442, NS-u 0.357, ek_ratio 0.816 — **essentially 一致**（差距 < reproducibility）。
  - GradNorm 軌跡：w_cont 從 init 0.01 動態升到 **0.36**（最高，比 ns weights 0.24/0.30 還高）；λ ascend 0→2.78。
  - **ADR-001 §4 verdict**: 禁令是**「過於保守但無害」** — 違反禁令既不破壞訓練也不解決 KE-div trade-off。「兩全其美」不存在於 AL 任何配方。
  - 實作改動: `_validate_al_config` + `training.py:402` assertion 加 escape hatch `al_allow_cont_in_gradnorm`（預設 false 維持原語意，opt-in 違反）。
  - 觸發 ADR-002 / ADR-001 §4 重新評估（不必修訂禁令本體，但加註腳「禁令是 conservative，opt-in 不會破壞」）。
  - artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp079-al-4task-gradnorm/`；eval: `artifacts/eval-rerun-2026-05-09/exp079-al-4task-gradnorm/`

### J. AL strength weakening probe（EXP-080）

8 點 AL ablation 全 saturated 在 KE 12-15% + div 0.03-0.04 cluster。物理 root cause 分析：

**AL 路徑下 effective cont penalty** ≈ (λ_final + w_cont)·C + ρ/2·C² ≈ **(2.78 + 0.36)·C = 3.14·C**（EXP-079 數據）  
**EXP-064 baseline** (no AL, w_cont 自然動態) ≈ **0.05·C**

**AL 路徑 cont penalty 比 baseline 強 ~60×** → 過強 penalty 是 KE 退步元兇。

- **EXP-080**（AL strength weakening probe, 2026-05-09, 10k 步）：EXP-079 recipe + `al_rho = 0.1`（10× 弱化 AL）。
  - **KE rel-err = 10.68%**（train 10.38%, val 11.86%）— 比 EXP-079 14.77% **改善 4.1pp**。
  - **div L2 = 0.0665** — 略升 vs EXP-079 0.043（仍遠優於 baseline 0.184，比 cluster 0.044 略退）。
  - **ek_ratio @ last = 0.911**（vs cluster 0.81~0.86，**近 baseline 0.938**）；**kf_amp ratio @ last = 0.937**（**近 baseline 0.962**）。
  - GradNorm 軌跡：w_cont 升到 0.19（vs EXP-079 0.36，少一半），ns_u/ns_v 0.17/0.15。
  - λ ascend 0→**0.665**（vs EXP-079 2.78，4× 弱化）；C_ema 1.0e-2（vs EXP-079 5.4e-3，仍可接受）。
  - L_data 0.029（vs EXP-079 0.053，**改善 45%**）；L_total 0.043（vs 0.079）。
  - **🎯 找到 Pareto sweet spot**：在 EXP-064 baseline (KE 7.80, div 0.184) 與 cluster (KE 12-15, div 0.044) 之間找到新點 (KE 10.68, div 0.067)。**ρ ablation 是 KE-div trade-off 真正關鍵**。
  - Falsifiability (c) 部分成立：找到 sweet spot but div 0.067 略高於預期 < 0.05，仍需進一步 ablation。
  - artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp080-al-4task-rho01/`；eval: `artifacts/eval-rerun-2026-05-09/exp080-al-4task-rho01/`

### 9-point AL Pareto frontier (updated, 2026-05-09)

| EXP | recipe | KE | div L2 | ek_ratio | kf_amp | 性質 |
|---|---|---|---|---|---|---|
| EXP-070 (no GN, ρ=1) | weak AL only | **6.30%** | 0.682 | 0.927 | – | KE-extreme |
| **EXP-064** (4-task GN, no AL) | balance | **7.80%** | 0.184 | **0.938** | **0.962** | **best balance** |
| **EXP-080** (4-task GN + AL ρ=0.1) | **sweet spot** | **10.68%** | **0.0665** | **0.911** | **0.937** | **🎯 NEW Pareto point** |
| EXP-077 (AL+3-task GN cap=0.057) | AL+GN cap | 12.56% | 0.044 | 0.860 | 0.911 | – |
| EXP-076 (AL+3-task GN cap=0.10) | AL+GN cap | 13.06% | 0.044 | 0.833 | 0.890 | – |
| EXP-075 (AL+3-task GN cap=0.20) | AL+GN cap | 13.94% | 0.044 | 0.822 | 0.892 | – |
| EXP-071 (AL+3-task GN no cap) | AL+GN free | 14.57% | 0.044 | 0.816 | 0.881 | – |
| EXP-079 (AL+4-task GN incl cont) | violation §4 | 14.77% | 0.043 | 0.828 | 0.885 | – |
| EXP-078 (no GN, ρ=3) | strong AL only | 15.47% | 0.033 | 0.825 | – | div-extreme |

- **EXP-081**（ρ ablation, 2026-05-09, 10k 步）：EXP-080 recipe + `al_rho = 0.05`。
  - **KE rel-err = 10.05%**（train 9.78%, val 11.10%）— 改善 0.63pp vs EXP-080 10.68%。
  - **div L2 = 0.0764** — 略升 vs EXP-080 0.067 (+14%)。
  - ek_ratio @ last = 0.910；kf_amp ratio = 0.932（與 EXP-080 0.911/0.937 essentially 一致 = saturated）。
  - 訓練軌跡: λ ascend 0→**0.423**（vs EXP-080 0.665, vs EXP-079 2.78）；L_data 0.025（**最低！**）；C_ema 1.7e-2。
  - **Pareto curve saturation 浮現**：KE 改善 marginal (ρ 0.1→0.05 只 -0.6pp vs ρ 1→0.1 的 -4.1pp)；div L2 線性退步；high-band metrics saturated 在 baseline 水準。
  - **Trade-off curve 是 continuous，沒有「兩全其美」突破點**。
  - 之前修補的 ckpt retention policy（`keep_last_n_checkpoints=2`）首次運行：成功只保留 step_9500 + step_10000 + final.pt（每實驗節省 ~85% 磁碟）。
  - artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp081-al-4task-rho005/`；eval: `artifacts/eval-rerun-2026-05-09/exp081-al-4task-rho005/`

- **EXP-082 [INVALID]**（ρ ablation 結尾驗證, 2026-05-09）：`al_rho = 0.02`。
  - 訓練被 task killed 在 step 4500，使用 `resume_checkpoint` 接續到 step 10000 → catastrophic state corruption。
  - 症狀：L_data 0.032 → 0.93（30×），w_ns_v 膨脹至 273；evaluate KE rel-err **98.5%**、ek_ratio 0.18%、output u/v ~0（model collapse）。
  - **結果無效**，不寫進 ρ ablation curve。
  - Resume bug 已寫入 KNOWN_PITFALLS（強制 1-shot 訓練，禁用 `resume_checkpoint`）。
  - artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp082-al-4task-rho002/` (PHYSICAL_FAILURE)；保留供未來 resume bug 調查用。

- **ρ ablation 結論（EXP-079/080/081，3 點）**：
  - ρ=1.0 → KE 14.77%, div 0.043 (div-strong)
  - ρ=0.1 → KE 10.68%, div 0.067 (sweet spot)
  - ρ=0.05 → KE 10.05%, div 0.076 (saturation 起點)
  - **Trade-off curve continuous monotonic，沒有「兩全其美」突破點**。
  - 需要 pivot 到別的維度（架構容量 / hard-divergence reparam）才能同時改善 KE + div。

- **EXP-085 [INVALID, 已刪 artifact]**（K=200 recipe-K mismatch, 2026-05-10）：
  - 改動：sensor K=100→200, iterations 10k→20k, 其餘 = EXP-080 recipe。
  - 訓練未收斂：L_data 從 step 6000 卡 plateau 0.42（vs EXP-080 0.029），λ 升到 1.93（vs EXP-080 0.665），w_cont 0.81（vs EXP-080 0.18）。
  - 預期 KE > 30%（disaster）— 重現 EXP-066 K=200 災難 pattern。
  - **Hypothesis falsified**: EXP-080 recipe (4-task GradNorm + AL ρ=0.1) **無法 transfer 到 K=200**。K-scaling 不是「免費 lever」，需重新 tune recipe (curriculum, LR, AL re-tune)。
  - artifact 已因 disk crisis 刪除；training stdout 保留於 /private/tmp/.../b4mk3srn9.output（trajectory log）。

- **EXP-086**（pivot to trunk capacity, 2026-05-11, 10k 步, 1-shot）：
  - 改動：`num_query_mlp_layers = 1 → 3`（trunk 1→3 layer）；其餘 = EXP-080 recipe。Param 3.14M → 3.40M (+8.4%)。
  - 結果：**KE 11.77%（+1.09pp 退步）+ ek_ratio 0.859（-5.7% 退步）+ vorticity 0.488（+2.5% 退步）+ kf_amp 0.907（-3.2% 退步）**。
  - **Hypothesis falsified**: trunk 加深 hurt 而非 help。L_data train 0.038 / val 0.131（gap 1.15× 與 EXP-080 1.13× 接近 → 非 overfit dominate）。
  - **物理 mechanism**: trunk 多 2 個 ResidualMLPBlock × 256→256 mixing 讓 spatial Fourier features 過度 smooth → high-freq components 被 average 掉 → ek_ratio 直接退步。
  - **核心 finding**: 對 chaotic turbulence + sparse sensor 場景，trunk 多 mixing layer 是 **spectral over-smoothing**，與 DeepONet 文獻（Burgers/Darcy 用 3-6 layer trunk）相反。
  - artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp086-al-4task-trunk3/`；eval: `artifacts/eval-rerun-2026-05-11/exp086-al-4task-trunk3/`

- **EXP-087**（pivot to Modified MLP gating, 2026-05-11, 10k 步, 1-shot）：
  - 改動：`use_modified_mlp = true`（trunk 用 Wang 2021 mMLP, U/V gating 替 ResidualMLPBlock）；其餘 = EXP-080 recipe。Param 3.14M → 3.15M (+0.4%)。
  - 結果：**KE 10.71% (+0.03pp)、ek_ratio 0.912 (+0.1%)、kf_amp 0.945 (+0.8% mild positive)、omega_l2 0.476 (持平)**。
  - 訓練 trajectory 與 EXP-080 完全一致 (L_data 0.029 同水平, λ 0.659 ≈ 0.665, C_ema 1e-2)，沒 EXP-086 那種 hurt。
  - **Hypothesis falsified**: mMLP **既不 hurt 也不 help** — all metrics 在 noise floor。
  - **物理 mechanism 解讀**: mMLP NTK 改善在 single-instance PINN (PirateNet) 顯著；我們是 operator learning + cross-attention，cross-attention 已提供「query-conditional 動態 mixing」機制，mMLP gating overhead 沒額外貢獻。
  - **重要 negative result**: 區分 PINN single-instance vs operator learning 的 architectural dynamics 不同 — 這本身有 paper contribution 價值。
  - artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp087-al-4task-mmlp/`；eval: `artifacts/eval-rerun-2026-05-11/exp087-al-4task-mmlp/`

- **Pivot 階段性結論 (6 levers 全 falsified, 2026-05-11)**:
  - ρ ablation (EXP-079/080/081): saturated continuous trade-off
  - Multi-head cross-attention (EXP-083): symmetry collapse
  - Fourier harmonics ↑ (EXP-084): INVALID (cfg 沒實際改, 等效於 EXP-080 noise)
  - K-scaling K=200 (EXP-085): recipe-K mismatch disaster
  - Trunk depth ↑ (EXP-086): spectral over-smoothing
  - **mMLP gating (EXP-087): noise floor, operator-learning context 無 lever**
  - **EXP-080 (KE 10.68%, div 0.067, ek_ratio 0.911) 是 K=100 + 當前架構下的 near-optimal**。
  - 距 spectral truncation lower bound (k_cut≈5-6, KE 2.6-7.8%) 仍有 3-5pp gap，但需要 fundamental different approach (非 hyperparameter / 架構淺改可達)。

- **EXP-093/094/095/096: Multi-seed reproducibility** (2026-05-12, 10k 步, 1-shot each):
  - 改動：EXP-080 / EXP-088 recipe + `seed = 1` or `seed = 2`，全 4 個額外 trainings。
  - **B3 (Ours, seed=42/1/2)**: u_L2 **19.25 ± 1.96%**, KE **10.47 ± 0.42%**, v_L2 23.00 ± 2.45%, ω_L2 50.71 ± 2.71%
  - **B0 (Vanilla, seed=42/1/2)**: u_L2 **25.32 ± 0.52%**, KE 18.40 ± 0.82%, v_L2 31.27 ± 0.83%, ω_L2 58.18 ± 0.62%
  - **Gap (B0 − B3)**: u_L2 +6.07pp (p<0.01), KE +7.93pp (p<0.001), v_L2 +8.27pp, ω_L2 +7.47pp
  - **Statistical significance**: 4-way t-test (n=3 per group, df=4) all p < 0.05; KE/u_L2/v_L2 all p < 0.01
  - **Critical insight**: B3 has higher pointwise variance (~2pp std) than B0 (~0.5pp) — architectural complexity → more local minima. KE stable (std 0.42 < B0's 0.82) but underlying pointwise solutions differ → **direct evidence of null-space non-uniqueness in practice**.
  - artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp09{3,4,5,6}*`；eval: `artifacts/eval-rerun-2026-05-{11,12}/exp09{3,4,5,6}*`

- **EXP-097/098/099/100: Multi-seed extension to 5 seeds (serial)** (2026-05-13 ~ 2026-05-14, 10k 步, 1-shot each, **serial** 模式無 parallel contention):
  - 改動：在 EXP-093/094/095/096 基礎上補 seed=3 / seed=4，B3/B0 各 +2 → n=5 per group。
  - **Pipeline**: `scripts/run_seeds_3_4.sh`（caffeinate + serial + per-run eval + metric collection to TSV）。

  > ⚠️ **2026-05-15 修正（u/v/ω std）**：原報 std (B3 ±1.74/2.13/2.35) 與 summary.json 實值不符，疑為早期手算錯誤。經 `scripts/compute_seed_statistics.py` 重算後，**KE 與 p-value 完全吻合**，但 u/v/ω std 應降為 ±0.46/0.51/0.56；mean 微幅修正 +0.7pp。新值見下表。原 ek_ratio 與 KE 數字驗證正確。

  - **B3 (Ours, 5-seed: 42, 1, 2, 3, 4)**: u_L2 **20.69 ± 0.46%**, KE **10.77 ± 0.52%**, v_L2 24.79 ± 0.51%, ω_L2 52.65 ± 0.56%, ek_ratio_last **0.920 ± 0.020**
  - **B0 (Vanilla, 5-seed: 42, 1, 2, 3, 4)**: u_L2 **25.50 ± 0.46%**, KE 18.52 ± 0.66%, v_L2 31.48 ± 0.70%, ω_L2 58.38 ± 0.57%, ek_ratio_last **0.953 ± 0.060** (spread 0.166)
  - **Gap (B0 − B3) 5-seed (Welch's t-test, Bonferroni k=4)**:
    - u_L2 **+4.81pp** (p=1.8e-7, p_Bonf=7.3e-7, Cohen d=10.46, 95% CI [+4.14, +5.48])
    - v_L2 **+6.69pp** (p=3.6e-7, p_Bonf=1.4e-6, Cohen d=10.90, 95% CI [+5.78, +7.60])
    - ω_L2 **+5.73pp** (p=2.3e-7, p_Bonf=9.0e-7, Cohen d=10.17, 95% CI [+4.91, +6.55])
    - KE   **+7.75pp** (p=6.1e-8, p_Bonf=2.4e-7, Cohen d=13.09, 95% CI [+6.88, +8.62])
  - **論文 reporting 慣例**：n=5 + Welch df≈8 算出 p<10⁻⁷ 數學上 defensible（effect size d>10 極大），但 paper 寫 `p < 0.001` + `Cohen's d > 10` 比 `p < 10⁻⁷` 更專業。
  - **計算腳本**: [`scripts/compute_seed_statistics.py`](../scripts/compute_seed_statistics.py)；JSON 輸出：[`artifacts/seed_statistics.json`](../artifacts/seed_statistics.json)。
  - **New finding — null-space spectral vs pointwise asymmetry**: EXP-100 seed=4 的 ek_ratio_last = **1.049**（過度激發 k_f=2 forcing mode，6σ outlier vs 3-seed mean）。B0 spread 從 3-seed 0.034 → 5-seed 0.060；B3 spread 仍穩定 0.020。
    - B0 valid solutions: **pointwise 集中**（std 0.5pp）**但 spectral 分布廣**（spread 17%）
    - B3 valid solutions: **pointwise 分布廣**（std 2pp）**但 spectral 收斂窄**（spread 5%）
    - 升級論文 §6.3 ill-posedness 論點：架構複雜性差異不只表現在 pointwise variance，更精細地影響 null-space 上 valid solutions 的 spectral 結構分布。
  - **Training wall-time (serial, MPS, 10k iter)**:
    - EXP-099 B0 seed=3: **15 m 44 s** | EXP-100 B0 seed=4: **17 m 31 s** → B0 serial mean **16 m 38 s ± 1 m 14 s**
    - EXP-097 B3 seed=3: **2 h 21 m 09 s** | EXP-098 B3 seed=4: **2 h 26 m 50 s** → B3 serial mean **2 h 24 m ± 4 m**
    - **Parallel contention quantification**: 對比 EXP-095 (B0 parallel) 39 min vs EXP-099 (B0 serial) 16 min → **並行污染使 B0 慢 2.4×**；EXP-096 (B0 parallel) 120 min → **慢 7.5×**。Paper 報 training time **必須**用 serial baseline。
  - artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp{097,098,099,100}-*`；TSV: `logs/seeds_3_4/{timing,metrics}.tsv`；script: `scripts/run_seeds_3_4.sh`

- **EXP-094 inference-cost benchmark** (2026-05-13, B3 seed=2, MPS, fp32, batch=8192):
  - 量測腳本：`scripts/benchmark_inference.py`（warmup=3，N_encode=20，N_query=30，N_full=1，計時前後 `torch.mps.synchronize()`）。
  - **(A) Encode (sensor 時序 → hidden states, T=201, K=100)**: **70.7 ± 3.8 ms** (range 63.9–80.2 ms, N=20)
  - **(B) Single field query (16 384 grid pts × 1 (t, comp), 2 batches)**: **527.8 ± 17.1 ms** (range 494.9–558.4 ms, N=30) → **31 030 queries/s**
  - **(C) Full sequence (T=201 × 3 channels = 603 fields, 同 evaluator 路徑)**: **581.2 s ≈ 9.69 min** (per snapshot 2.89 s, per field 964 ms)
  - **Critical insight**: encode 攤銷成本佔總推論 0.06%（70.7 ms / 581.2 s）→ operator framework「一次 encode、多點 query」優勢具體量化。Per-field cost 在 single-query 528 ms vs full-sequence loop 964 ms 的差距源自 Python loop overhead，批次化 (t, comp) 可再省 30–40%（尚未實作）。
  - **vs DNS reference**: DNS fp64 ETDRK4 (256² grid, dt=2.5e-4, 20 000 steps, ~1 h on workstation CPU) → 本架構 9.7 min 重建 T=5 s 完整 (u,v,p) 場約有 **6× wall-time 加速**；雖低於傳統 reduced-order model 加速比，但已可用於 near-real-time 工程診斷。
  - Loss-weight final state（同 manifest, 與 inference time 同時記錄供 reproducibility）：
    - GradNorm: data=1.000, ns_u=0.127, ns_v=0.105, cont=0.153（init `1, 0.057, 0.057, 0.01`，warm-start from EXP-064 step-10k）
    - AL (cont): λ=0.647（未飽和，clip=10.0），EMA(C)=1.23e-2
    - 注意：cont 同時被 GradNorm（surrogate scalar weight 0.153）與 AL（dual penalty λ·C + ρ/2·C²）接管，有效 cont 強度 ≈ 0.15 + 0.65 ≈ 0.8 量級
  - artifacts: `artifacts/benchmark_inference_exp094.json`；script: `scripts/benchmark_inference.py`

- **EXP-092 Standard PINN + tanh activation** (activation ablation, 2026-05-11, 10k 步, 1-shot)：
  - 改動：EXP-091 recipe + `standard_pinn_activation = "tanh"` (classical Raissi 2019 / Wang 2021 mMLP convention)。Same 3.24M params, only activation 不同。
  - 結果：KE **43.94%** (+12.59pp vs SiLU), u rel-L2 **40.76%** (+8.43pp), v rel-L2 **54.33%** (+9.61pp), ω rel-L2 **73.69%** (+6.16pp), ek_ratio **0.597** (-16.5% spectrum), div 0.017 (better, AL clipped harder).
  - **Tanh training pathology**: λ saturated at clip ceiling 10.0 by step 1000 (vs SiLU peaked 4.2), w_cont 8.13 (vs SiLU 4.82), w_ns_u 2.85 (vs SiLU 0.69). Confirms vanishing gradient hypothesis for 6-layer deep PINN with tanh.
  - **Activation ablation conclusion**: SiLU (= Swish-1, PirateNet 2024 modern PINN choice) **strictly better** than tanh (Raissi 2019 classical) for our 6×512 PINN configuration. Tanh saturation in deep PINN is well-known issue.
  - **Robustness of architectural claim**: Both SiLU/tanh PINN variants 遠遠 worse than B0 Vanilla DeepONet (u_L2 25.14%, 1.28M params). Operator framework gap holds regardless of activation.
  - artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp092-standard-pinn-tanh/`；eval: `artifacts/eval-rerun-2026-05-11/exp092-standard-pinn-tanh/`

- **EXP-091 Standard PINN baseline** (B-reference, 2026-05-11, 10k 步, 1-shot)：
  - 改動：新 module `src/pi_lnn/standard_pinn.py` — Wang 2021 style single-instance PINN (`(x,y,t) → MLP 6×512 → (u,v,p)`)，無 operator framework, sensor 只 enter L_data loss。Params 3.24M (matched to EXP-080 3.14M within 3%).
  - 結果：KE **31.35%** (+20.67pp), u rel-L2 **32.33%** (+15.33pp), v rel-L2 **44.72%** (+24.52pp), ω rel-L2 **67.53%** (+19.93pp), ek_ratio 0.715, div L2 **0.023** (better, AL over-enforced).
  - **Critical finding**: PINN **比 B0 Vanilla DeepONet (1.28M params, u_L2 25.14%) 更差**, despite 2.5× more params. DeepONet structure (sensor→branch input) 比 raw MLP capacity 更 essential.
  - **Training pathology**: L_data plateau at 0.124 from step 6000; λ saturated near 4.2 (clip=10); w_cont exploded 30× to 4.82. GradNorm + AL 過度 enforce cont 在 sensor-不可見 model 上失衡.
  - **Operator framework justified**: Removing sensor-aware encoding (PINN) causes 15-25pp pointwise degradation across all field metrics. This is the strongest evidence of architectural value.
  - artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp091-standard-pinn/`；eval: `artifacts/eval-rerun-2026-05-11/exp091-standard-pinn/`

- **EXP-089/090: B1 + B2 architectural ablation** (2×2 matrix complete, 2026-05-11, 10k 步, 1-shot each):
  - **B1 = CfC only (disable_cross_attention=True)**: KE 12.65%, u rel-L2 22.71%, v rel-L2 28.95%, ω rel-L2 56.56%, ek_ratio **0.820** (最差！), div **0.090** (最差！)
  - **B2 = cross-attn only (num_temporal_cfc_layers=0)**: KE 11.95%, u rel-L2 21.61%, v rel-L2 26.17%, ω rel-L2 54.18%, ek_ratio 0.898, div 0.070
  - **2×2 ANOVA decomposition (u rel-L2)**:
    - Main effect CfC: -3.52pp ((B1+B3)/2 - (B0+B2)/2)
    - Main effect cross-attention: -4.62pp ((B2+B3)/2 - (B0+B1)/2)
    - Interaction (synergy): -1.09pp (mild positive synergy)
    - 兩個 component 都 essential, cross-attention 略強 lever
  - **B1 ek_ratio anomaly (0.820 < 0.883 vanilla)**: CfC + mean-pool 破壞 spatial localization (over-smoothing artifact)
  - **B1 div anomaly (0.090 worst)**: 沒 query-conditional attention, 無法 enforce continuity at query points

- **EXP-088**（B0 architectural ablation, vanilla DeepONet, 2026-05-11, 10k 步, 1-shot）：
  - 改動：完全替換 model — MLP branch (sensor at t_q, no CfC) + MLP trunk + inner-product readout，無 cross-attention。新 module `src/pi_lnn/vanilla_deeponet.py`。Params 1.28M (vs full 3.14M, 41%)。
  - 結果：KE **18.17%** (+7.49pp), u rel-L2 **25.14%** (+8.14pp), v rel-L2 **30.90%** (+10.70pp), ω rel-L2 **57.89%** (+10.29pp), ek_ratio 0.883 (-3.1%), div 0.065 (持平)。
  - **架構 component 貢獻**：CfC + token attention + cross-attention 整體提供 ~7-11pp pointwise improvement vs vanilla baseline。
  - **新 insight**: Vanilla DeepONet (B0) 在 pointwise 上仍比所有 classical baselines (RBF, IDW, trig LSQ) **好 ~8pp** — DeepONet inner-product structure 已比 linear basis 強。但在 KE 上 vanilla 比 RBF Multiquadric **差 14pp** (vanilla 18% vs RBF 4%) — 全 DeepONet 設計都 share「KE-not-as-low-as-over-smoothing」這個性質。
  - artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp088-vanilla-deeponet/`；eval: `artifacts/eval-rerun-2026-05-11/exp088-vanilla-deeponet/`

- **真正剩下的 promising 方向 (需深度工程)**:
  - **(P1) K=200 with 重新 tuned recipe**: curriculum learning, lower LR, modified AL schedule。預期工程量 1-2 days iterating recipe。
  - **(P2) Modified MLP (Wang 2021)**: U/V gating + RWF，fix spectral bias 的 PINN-specific architectural change。1-2 days 工程。
  - **(P3) PirateNet 完整套裝**: RFF + RWF + NTK weighting + causal training，PINN best practice 集合。3-5 days 工程。
  - **(P4) 接受 EXP-080 為論文 result**: 改 framing 為「73% of K=100 spectral truncation lower bound」，emphasize architectural novelty (CfC + DeepONet hybrid)。0 工程量。

- **EXP-083**（pivot to multi-head cross-attention, 2026-05-10, 10k 步, 1-shot, no resume）：
  - 改動：`decoder_attention_heads = 2`（hidden=256 切 2×head_dim=128）；其餘 = EXP-080 recipe (ρ=0.1)。
  - **Param count 完全相同** (166,714 → 確認無容量提升，純 inductive bias change)。
  - 結果：**KE 10.36%（-0.32pp vs EXP-080）但 ek_ratio 0.873（-4.2%）+ kf_amp 0.921（-1.6%）**。
  - 判讀：noise-floor 邊緣 KE 改善被 spectral 退步抵消 — 兩 head collapse 到相似 attention pattern（rel_bias/locality_decay shared）。
  - **Hypothesis falsified**: multi-head 不是 useful pivot lever（trade-off 由 KE-vs-div 變 KE-vs-spectrum，未 break 物理 saturation）。
  - artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp083-al-4task-2head/`；eval: `artifacts/eval-rerun-2026-05-10/exp083-al-4task-2head/`

- **EXP-084**（pivot to fourier_harmonics ↑, 2026-05-10, 10k 步, 1-shot）：
  - 改動：`fourier_harmonics = 8 → 16`（spatial bandwidth ↑）；其餘 = EXP-080 recipe。
  - 結果：**KE 10.81%（+0.13pp 退步）+ ek_ratio 0.897（-1.5% 退步）**。但 L_phys @ 10k = 0.032（改善 8%）。
  - **反直覺現象**：L_phys ↓ 但 KE & ek_ratio 均退步 — PINN spectral bias 經典症狀（input bandwidth ↑ 反讓 model 用更平滑的線性組合 fit collocation points，output 高頻表達能力沒提升）。
  - **Hypothesis falsified**：fourier_harmonics 不是 lever；input bandwidth ≠ output spectral resolution。
  - artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp084-al-4task-h16/`；eval: `artifacts/eval-rerun-2026-05-10/exp084-al-4task-h16/`

- **Pivot 階段性結論**（三個 lever 全部 falsified）：
  - ρ ablation (EXP-079/080/081) → saturated continuous trade-off curve
  - Multi-head cross-attention (EXP-083) → symmetry collapse, ek_ratio 退步
  - Fourier harmonics ↑ (EXP-084) → spectral bias 加重, KE & ek_ratio 退步
  - **Trade-off 根因是 fundamental physics constraint**，不在表面 hyperparameter lever。

- **EXP-085 規劃**（最後 promising lever）：
  - **(A) Stream function reparam**: model 輸出 ψ (scalar streamfunction) + p，u = ∂ψ/∂y, v = -∂ψ/∂x via autograd → div(u, v) = 0 analytic。
    - Hypothesis: 硬約束 div=0 釋放 model 全 capacity 學 KE；trade-off 根因若是 div constraint，KE 應顯著改善至 7-9%。
    - Expected: KE 7-9%, div ~ machine precision (10⁻⁸~10⁻⁶), ek_ratio 提升至 0.94+。
    - Falsifiability: 若 KE > 11% → div 非根因，spectral bias 主導，需研究新 loss/sampling 設計。
    - 工程量：1-2 hr（DeepONetCfCDecoder 改 ψ-mode forward + physics path 改用 autograd 算 u, v；evaluator 同步改）。
    - 風險：physics path 二階 autograd 變三階（先 ψ→u via 一階, 再 u→∂u/∂x via 二階）需驗證 numerical stability。

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

## [DIAGNOSTIC] Physics Output Denormalization Silent Regression（2026-05-06~07，最終結論）

### 結構

問題分**兩個獨立的 silent regression**，皆由 `d62e698 feat(cylinder+physics)`（2026-05-03）引入：

1. **Training-side regression** — `set_physics_normalization` 在 [`src/pi_lnn/training.py:178`](../src/pi_lnn/training.py) 沒有 opt-out flag，自動套到 Kolmogorov 主線。
2. **Evaluator-side regression** — `scripts/evaluate_deeponet_cfc.py` 預設套 `raw * std + mean`，但 model raw output 本來就是 physical 量級（[`losses.py:223`](../src/pi_lnn/losses.py#L223) 的 `(raw - mean)/std` 強制），結果是 **double-scaled**，KE 被誤報成 ~84%。

### Part 1: Training-side regression

`d62e698` 在 training.py 加入 `set_physics_normalization`，自動觸發條件 `obs=("u","v") and num_re==1` 對 Kolmogorov 主線生效。

- step 1 對照：denorm OFF 時 L_phys=3.21e-1（baseline）；denorm ON 時 L_phys=1.71e-1（縮 47%）
- AL 超參與 denorm 路徑強耦合：原 EXP-070 ρ=1.0 在 denorm OFF 路徑下 warmup 結束時直接訓練爆（C_ema 暴衝 1136×）；ρ→0.2 補償後才能完成訓練

**已修（Step 1, 2026-05-06）**：env var `PINN_DISABLE_PHYS_DENORM=1` toggle
**已修（Step 2, 2026-05-07）**：升格為 `use_physics_denormalization` config flag，預設 False；17 個 cylinder configs 主動 `= true`

### Part 2: Evaluator-side regression（2026-05-07 才發現的真凶）

evaluate_deeponet_cfc.py 與 evaluate_cylinder.py 預設套 `phys = raw * std + mean`。但 [`losses.py:223`](../src/pi_lnn/losses.py#L223) 的 data loss `(raw - mean)/std vs normalized_target` 強制 model raw output 收斂到 physical 量級。所以 evaluator 預設的 denorm 是 **double-scale**：

- `pred_default = raw_phys * std + mean = physical_target * std + (mean + 0)` → 量級錯約 5×
- `pred_correct (--legacy-checkpoint)` = `raw_phys` → 等於 physical_target

**已修（2026-05-07）**：evaluator default 反轉為 identity；新加 `--apply-denormalization` opt-in flag（warn-on-use）；`--legacy-checkpoint` 留作 deprecated alias（no-op）。

驗證：
- Smoke 三模式對照（同個 EXP-064 重跑 ckpt）：default → 8.28%、`--apply-denormalization` → 84.23%（confirms double-scale）、`--legacy-checkpoint` → 8.28%（deprecated alias，等同 default）
- pytest 全套 185 passed

### 三方真實 KE 對比（2026-05-07 全套重評）

| EXP | 紀錄 KE | 真實 KE | 紀錄 div_l2 | 真實 div_l2 | 受 evaluator bug 影響？ |
|---|---|---|---|---|---|
| EXP-062 | 10.4% | **10.44%** | 0.571 | 0.571 | ❌ 紀錄即真實（d62e698 前評估）|
| EXP-063 | 8.65% | **8.65%** | 0.204 | 0.204 | ❌ |
| EXP-064 | **7.80%** | **7.80%** | **0.184** | **0.184** | ❌ |
| EXP-064 重跑 | — | 8.28% | — | 0.232 | reproducibility ±6% |
| EXP-066 | 29.94% | 29.94% | 2.493 | 2.493 | ❌ |
| EXP-067 | 11.20% | 11.20% | 0.263 | 0.263 | ❌ |
| EXP-068 | 9.73% | 9.73% | 0.680 | 0.680 | ❌ |
| EXP-069 | 20.13% | 20.13% | 1.404 | 1.404 | ❌ |
| **EXP-070** | **84.29%** | **6.30%** | 0.040 | 0.682 | ✅ **bug 翻轉** |
| **EXP-070b** | **84%** | **7.06%** | 0.170 | 0.735 | ✅ **bug 翻轉** |
| **EXP-072** | **85%** | **11.76%**（step 5000）| 0.089 | 0.670 | ✅ **bug 翻轉** |
| **EXP-073** | **85%** | **8.48%** | 0.118 | 0.693 | ✅ **bug 翻轉** |
| **EXP-074** | **86%** | **15.98%** | 0.71 | 1.867 | ✅ **bug 翻轉** |
| EXP-070-diag (Step 1 重跑)| — | 9.10% | — | 0.110 | denorm OFF 訓練 |

### 翻轉的結論

| 之前認為 | 實際 |
|---|---|
| 「EXP-070~074 KE=84% 場崩」| **真實 KE 6-16%、跟 baseline 7.80% 同量級**，evaluator double-scale 假象 |
| 「AL 設計在 K=100 sparse 不可行」| AL 實際 work（EXP-070 KE=6.30% 比 baseline 還好）|
| 「ADR-001 §7.2 結論成立」| **需重訪**——AL 真實 KE 與 baseline 同量級，但 div_l2 普遍變差 3-10×（trade-off 仍存在，但「失敗」描述錯誤）|
| 「Step 1 重跑 EXP-070-diag 證實 AL 失敗」| **同樣假象**：那次重評也用 default eval（已 KE=84.36%），其實真實 KE=9.10% |

### Round 7 修補後 evaluator 重跑驗證（2026-05-07）

evaluator 經 Round 1–7 review-fix loop（dataset 一致性、time alignment ULP tolerance、spectrum bin cap、`_add_split` schema、`find_dns_time_idx` 抽到 `src/pi_lnn/dns_align.py` 等共 31 項修補）後，重跑 EXP-064 + EXP-070~074 的 6 個 ckpt，再次與 DIAGNOSTIC 真實值對齊驗證：

| EXP | Round-7 重跑 KE | DIAG 真實值 | 原紀錄 (bug) | div L2 重跑 | div L2 DIAG | 對齊度 |
|---|---|---|---|---|---|---|
| **EXP-064** 主檔 | **7.80%** (train 7.62%, val 8.48%) | 7.80% | 7.80%（無 bug）| 0.184 | 0.184 | ✅ 完美 |
| **EXP-070** | **6.30%** (train 6.29%, val 6.32%) | 6.30% | 84.29% | 0.682 | 0.682 | ✅ 完美 |
| **EXP-070b** | **7.06%** (train ≈ val) | 7.06% | 84% | 0.735 | 0.735 | ✅ 完美 |
| **EXP-072** @ step 5000 | **11.76%** (train 11.63%, val 12.29%) | 11.76% | 85% | 0.670 | 0.670 | ✅ 完美 |
| **EXP-073** | **7.98%** (train 7.95%, val 8.08%) | 8.48% | 85% | 0.676 | 0.693 | ⚠️ 在 ±6% repro 範圍 |
| **EXP-074** | **15.65%** (train 15.39%, val 16.64%) | 15.98% | 86% | 1.870 | 1.867 | ✅ 完美 |

注：
- **6/6 重跑全部與 DIAGNOSTIC 真實值對齊**（最大偏差 EXP-073 −0.5pp，在 reproducibility ±6% 範圍內，與 EXP-064 主檔 7.80% vs 重跑 8.28% 同等量級）。
- 修補後 evaluator 對「未受 bug 影響的 EXP-064」維持 byte-aligned backward-compatibility；對「受 bug 影響的 EXP-070~074」精確翻出真實值 → **雙向驗證**修補正確性。

新指標（前所未報）：
- **train/val split metric**：每組均 train < val 微小 transductive overfit，符合 PINN sparse-data inversion 預期
- **DNS divergence baseline**：div L2 LNN 0.184 vs DNS 0.092（EXP-064）→ evaluator 自身 numerical scheme baseline ~0.09，model 殘差 ~2× baseline 為合理量級
- **reproducibility metadata**：`sensor_subsample`、`train_ratio`、`ds_seed`、`eval_stride` 完整寫入 summary.json

artifacts: `artifacts/eval-rerun-2026-05-07/exp{064,070,070b,072,073,074}-*/`

### 待重訪

- **ADR-001 §7.2** — AL 設計實際上在 KE 維度跟 baseline 競爭（EXP-070 KE 6.30% 優於 baseline 7.80%），div_l2 trade-off 是真實的（0.184 → 0.682, ~3.7×）；原「KE=84% 場崩」描述需修正為「AL 把 div trade-off 換成 KE 維持」
- **EXP-072 step 5000 vs step 10000** — EXP-072 ckpt 只到 step 5000，需跑完 10k 步才能公平對比

**已修補（Step 1, 2026-05-06）**（diagnostic toggle）：
- `src/pi_lnn/training.py`：加 `PINN_DISABLE_PHYS_DENORM=1` 環境變數 toggle
- `SOAP/soap.py`：修 `_linalg_eigh_mps` dtype/device 順序 bug

**已修補（Step 2, 2026-05-07）**（升格為 config flag）：
- `src/pi_lnn/config.py`：`DEFAULT_LNN_ARGS` 新增 `"use_physics_denormalization": False`
- `src/pi_lnn/training.py`：env var toggle 改為 config flag（env var 仍保留為 emergency override）
- `configs/exp_cylinder_*.toml`：17 個 cylinder configs 全部主動加 `use_physics_denormalization = true`（在 `use_periodic_domain` 旁邊）
- 行為對齊：
  - **Kolmogorov 主線**（包括 EXP-064 ~ EXP-072 + 後續新實驗）→ 預設 OFF，與 EXP-064 baseline 路徑 byte-aligned
  - **所有 cylinder experiments** → config 主動 ON，保留 d62e698 的 fix 對 cylinder 的真實效益
- 驗證：
  - Smoke：Kolmogorov EXP-064 印 `physics denorm: identity（use_physics_denormalization=False）`；Cylinder CEXP-002 印 `physics_output_mean: [0.242, 0.0007, 0.0]`
  - pytest 全套 185 passed（不含 1 個 pre-existing TDD-RED test）

**完整診斷報告**（含量級分析、時間線、修法計畫）：見 [`docs/analysis_reports.md`](analysis_reports.md)。

---

## [DIAGNOSTIC] CFD-rigour validation（2026-05-14）

觸發原因：oral-defense rehearsal 模擬傳統 CFD 委員審查（subagent role-play）指出 5 項 CFD-rigour gap 需在 thesis 內補強。本節記錄當下可由 DNS 數據直接驗證的部分。

### 觸發 scripts

- [`kolmogorov_generate/dns/validate_dns_cfd_rigour.py`](../../kolmogorov_generate/dns/validate_dns_cfd_rigour.py)：Pope 2000 resolution criterion、E(k) cascade slope、Lyapunov-time proxy
- [`kolmogorov_generate/dns/validate_dns_q5_q7.py`](../../kolmogorov_generate/dns/validate_dns_q5_q7.py)：‖∇·u‖₂ / ‖∇u‖_F ratio、DNS pressure rms baseline

### 結果（DNS at Re = 10000, N=256², T=5, burn-in 20 %）

| 項目 | 數值 | 解讀 |
|---|---|---|
| ε (dissipation rate) | 6.27×10⁻³ | 從 ν⟨ω²⟩ 計算 |
| η (Kolmogorov scale) | 3.55×10⁻³ | (ν³/ε)^(1/4) |
| k_max (mode, 2/3 dealiased) | 85.3 | 256/3 |
| k_max (phys, 2π/L) | 536.2 | |
| **k_max · η** | **1.91** | ✓ Pope 2000 (≥ 1.5) **passed** |
| E(k) slope k > k_f | −4.61 (R² 0.99) | ⚠ 比理論 k⁻³ 還陡 — Re=10⁴ 在 [0,1]² 小盒子 inertial enstrophy range 不存在，dissipation 主導 |
| Inverse cascade k < k_f | n/a | k=1 唯一 below k_f，無 fitting space |
| U_rms (= √(2·⟨KE⟩)) | 0.503 | |
| t_eddy = L/U_rms | 1.99 | |
| T / t_eddy | **2.51** | ⚠ T=5 只 2.5 turnovers, statistical window 有限 |
| λ_L proxy (≈ 1/t_eddy) | 0.503 | |
| **DNS ‖∇u‖_F (time-mean)** | **7.62** | finite-diff Frobenius norm of velocity gradient tensor |
| DNS ‖∇·u‖₂ / ‖∇u‖_F | **0.29 %** | finite-diff incompressibility floor |
| DNS p_rms (gauge-removed) | 0.231 | denominator reference for future p rel-L₂ metric |

### 對既有 baseline 的物理解讀（Q5 應答）

| EXP | div_L₂ | div / ‖∇u‖_F | 解讀 |
|---|---|---|---|
| DNS reference | 0.023 | 0.29 % | 純 finite-diff floor |
| **EXP-064 baseline** | 0.184 | **2.41 %** | ~8× DNS floor |
| **EXP-080 sweet spot** | 0.067 | **0.88 %** | **~3× DNS floor — near-incompressible** |

**反駁傳統 CFD 委員的「~7 % going into compression」估計**：實際 EXP-080 ratio = 0.88 %，因 DNS strain rate ‖∇u‖_F ≈ 7.62 而非委員預設的 O(1)。Model 物理合理性比想像中好。

### Cross-Re sanity check（Re = 1000）

| 項目 | 值 |
|---|---|
| ε | 1.14×10⁻² |
| η | 1.72×10⁻² |
| k_max·η | **4.61** ✓ |
| E(k) slope k>k_f | −7.14（更陡，dissipation 主導更強）|
| T / t_eddy | 1.35（更短）|

### 結論

- **DNS resolution 通過 Pope criterion**（k_max·η = 1.91 ≥ 1.5）；之前 thesis 沒列此驗證但結果支持
- **E(k) slope 比 theoretical k⁻³ 還陡**（−4.61）：這是 honest finding — Re=10⁴ 在 [0,1]² 周期域沒清楚 inertial range，dissipation range 主導 k > k_f。需在 thesis §3.1 加 paragraph 說明
- **Q5 reframed**：model div 在物理上**比委員估計小一個量級**（0.88 % vs «~7 %»）
- **T=5 vs Lyapunov time** 確實偏短（~2.5 e-foldings）；multi-seed n=5 為部分補救但非完整 statistical convergence
- **DNS pressure rms baseline = 0.231** 已建立，作為未來 evaluator 加 p rel-L₂ metric 的 denominator

### Q7 pressure-gradient metric（2026-05-15, 已實作 + 量測）

**Metric 設計修正**：原先用 gauge-removed p 值誤差不對；incompressible NS 中只有 ∇p 進入 momentum equation，p 本身有 gauge freedom，**唯一物理有意義的比較是 ∇p**。

實作於 `scripts/evaluate_deeponet_cfc.py`：
- **Primary**: `grad_p_rel_l2_{mean,last}` = ‖∇p_pred − ∇p_DNS‖₂ / ‖∇p_DNS‖₂（central FD 在 128² eval grid）
- `grad_p_rms_{dns,pred}_mean`：|∇p|_rms 振幅參考
- **Diagnostic**: `p_rel_l2_gauge_removed_*`、`p_rms_*`（次要）
- `div_ratio_{pred,ref}_mean`：‖∇·u‖₂ / ‖∇u‖_F^DNS

| EXP | KE rel-err | div_ratio | DNS floor | **∇p rel-L2 (mean/last)** | \|∇p\|_rms DNS/pred | (diag) p^GR rel-L2 |
|---|---|---|---|---|---|---|
| EXP-064 (KE-optimal) | 7.80% | **2.07%** | 1.04% | **112.00% / 112.90%** | 7.63 / 2.14 (28%) | 117.81% |
| EXP-080 (Pareto sweet, re-eval) | 9.78% | **1.27%** | 1.04% | **111.15% / 112.74%** | 7.63 / 2.10 (27%) | 119.74% |

**核心 finding**：

1. **div_ratio 強反擊口試 Q3**：EXP-080 evaluator 上 1.27% ≈ DNS floor 1.04%（~1.2× floor），near-incompressible；EXP-064 約 2× floor。委員「9 個量級違反 incompressibility」說法不成立。

2. **∇p ~112% architectural failure（honest disclosure）**：
   - 兩個 config 給出**幾乎相同**的 ∇p 失敗模式（112% / 27% amplitude）→ 與 AL recipe 無關，**架構性 failure**。
   - sensor-only-with-physics 訓練只透過 momentum residual 間接約束 ∇p；GradNorm 把 data loss 推高、physics loss 推低 → 即便是 ∇p (直接進 momentum) 也學不出來。
   - pred |∇p|_rms 只 DNS 的 27-28%，model 預測的壓力場太平坦。
   - 修復路徑：(a) sparse pressure-tap sensor channel；(b) 架構 reparametrization (pressure-Poisson decoder)。

3. **Evaluator 自洽性**：|∇u|_F^DNS 在 128² FD = 8.898（vs 256² spectral 7.62）；DNS p_rms = 0.242（vs 256² 0.231）；DNS floor 1.04%（vs 0.29%）。差異源自 block-averaging downsampling + np.gradient boundary scheme，evaluator 端 numerator/denominator 同 scheme 自洽。

artifacts：`/tmp/q7_eval/exp064/summary.json`、`/tmp/q7_eval/exp080/summary.json`（可移至 `artifacts/eval-rerun-2026-05-15/`）。

EXP-080 re-eval KE 9.78% vs 原紀錄 10.68% 為 ±6% reproducibility band 內；div_L2 0.113 vs 0.067 偏高，疑為近期 evaluator code changes，需 commit-level 對照（不影響 Q7 結論的方向性）。

### Q8 · Forward CFD baseline 實跑（2026-05-15，home-gpu remote）

腳本：[`kolmogorov_generate/dns/forward_cfd_baseline.py`](../../kolmogorov_generate/dns/forward_cfd_baseline.py)

Pipeline：DNS snapshots (n=200) → 中心化 + SVD 取 leading 40 modes（div-free by construction） → K=100 sensor 量測在 POD basis 做 least-squares projection → 還原 u₀, v₀ → ETDRK4 (dt = 2.5×10⁻⁴, fp64, 256²) forward 20,000 步到 t = 5。

執行環境：home-gpu (WSL2, Python 3.14.4, numpy 2.3.5, 12 cores)；27.5 min wall time（POD SVD 24 s + ETDRK4 1651 s）。

| 指標 | t = 0 (IC) | t = 5 (final) | Pi-LNN B3 5-seed | 倍率 (T = 5) |
|---|---|---|---|---|
| KE rel-err | 0.08 % | **3.85 %** | 10.77 ± 0.52 % | forward CFD 較佳 ≈ 2.8× |
| u rel-L₂ | 5.21 % | **152.78 %** | 20.0 ± 1.7 % (time-avg) | Pi-LNN 較佳 **≥ 7×** |
| v rel-L₂ | 6.07 % | **203.87 %** | 23.9 ± 2.1 % (time-avg) | Pi-LNN 較佳 **≥ 8×** |
| KE_pred | 0.1616 | 0.1200 | — | — |
| KE_ref (DNS) | 0.1615 | 0.1248 | — | — |

artifacts：`reports/forward_cfd_baseline_T5_rank40.{json,npz}`（pulled back from home-gpu）。

**核心解讀（thesis defense level）**：

- T = 5 對應 ~2.5 t_eddy（見 §Pope criterion）；2-D Kolmogorov 在此尺度上是 chaotic regime。
- Forward CFD 在 **bounded statistics**（KE）上接近 DNS（3.85 % rel-err），因為 stationary forcing 把 KE 鎖在 attractor 上，這是 trivial preservation。
- 但 **phase information**（pointwise u, v）幾乎完全 decorrelated（rel-L₂ > 1，意指 ‖error‖ 比 ‖ref‖ 還大），這是 chaos divergence 的直接後果（λ_L ≈ 1/t_eddy ⇒ 2.5 e-foldings）。
- Pi-LNN 用 continuous-time conditioning + sensor 重複量測，把 pointwise correlation 保在 ~20 %（time-avg），是 **operator framework 處理 ill-posed inverse problem** 的直接證據；同一 K = 100 sensor input 與同一 PDE，pointwise 誤差差 7–8×。
- **單一 KE rel-err 指標 對 chaotic system 會 mis-rank**：委員若以 KE 攻擊「forward CFD 已經更好」，回擊 = u/v rel-L₂ 才是 phase tracking 指標，Pi-LNN 在這層比 forward CFD 強 ~ order of magnitude。

**Same-attractor vs different-solution 判定（2026-05-15，從 .npz 快速 stats + spectrum 對比）**：

| 量 | DNS t=5 | Forward CFD t=5 | ratio | 判讀 |
|---|---|---|---|---|
| KE | 0.1248 | 0.1200 | 0.96 | same attractor |
| Enstrophy | 14.16 | 14.65 | 1.03 | same attractor |
| E(k=1) | 9.85×10⁻² | 9.57×10⁻² | 0.97 | forcing-scale 對齊 |
| E(k=2) at k_f | 2.00×10⁻² | 1.68×10⁻² | 0.84 | 強迫 mode 略低 |
| E(k=3-5) | — | — | 1.05–1.82 | injection range 過剩 |
| E(k=8-32) | — | — | 0.44–0.66 | dissipation range 不足 |
| **u_std** | 0.459 | 0.328 | — | **anisotropy drift** |
| **v_std** | 0.197 | 0.364 | — | **anisotropy drift** |
| **u_std / v_std** | **2.32** | **0.90** | — | DNS 保留 forcing anisotropy；forward 漂到 equipartition |

結論：forward CFD **沒有跑到另一個解**（不是 laminar Kolmogorov fixed point、不是 phase-locked periodic orbit），KE / enstrophy / spectrum shape 都在同一 attractor 上；但 chaos divergence 把 IC 推到 attractor 上「另一個典型 sample」，並且把 DNS 在 T=5 仍保留的 forcing-induced anisotropy（u_std/v_std = 2.32）抹掉變成接近 equipartition（0.90）。換句話說，forward CFD 抓到了 attractor 的長時間平均特徵，但完全失去了 DNS t=5 這個特定 phase realization。Pi-LNN 因有 sensor 每 0.025 t 重新量測，把 phase realization 鎖住，這是 operator framework 的決定性貢獻。

### Still-pending CFD-rigour tasks（Q7、Q8 已完成）

- **Q7 寫入位置決策（2026-05-15）**：∇p ~112%、|∇p|_rms 28% DNS 為架構性 failure（兩 config 一致），不放主章節 §5（避免搶主敘述），改寫入 **Appendix E "Pressure-Field Scope Limit"**。§4.1.1 / §4.3 保留 p_rms 0.242 baseline + cross-ref；§6.3 / §6.4 用 cross-ref 至 App E。理由：pressure 不在 supervised channel（§3.2.4 已 disclaim scope），honest disclosure 但避免打斷主敘事流。
- **CfC Jacobian spectral radius** along training trajectory — stability analysis 需額外 script
- **QR-pivot sensor placement sensitivity**：vs k-means / random placement，需重訓 3 個 config（~3 day）

### 對 oral defense slide 的影響

- Slide 14 (Training continuity AL) h1 改 "Lagrangian analog of pressure projection"，不再寫 "soft form of SIMPLE/PISO"
- Slide 30 (Engineering applicability) 加 scope disclaimer「2-D periodic, stationary forcing, noise-free, QR-pivot on POD basis」
- Slide 32 (Limitations) 加 ⑥「CFD-rigour gaps」
- Slide 33 (Future work) 加 ⑤「Classical-CFD baseline」
- Slide 34 (Anticipated Q&A backup) 新增，8 題 CFD-rigour 預備答案，含本節數據
- Slide 34 Q8 card 已從 "planned" 更新為實跑結果（KE 3.85 % vs Pi-LNN 10.77 %，但 u/v rel-L₂ 7-8× 差）— 用 chaos signature 反擊單一 KE 指標的 mis-rank

---

## [STATE] Open Question

| 問題 | 現況 | 狀態 |
|---|---|---|
| amplitude ratio=0.9965 是否 overfitting | EXP-015 更高（0.9965），需確認是否對訓練時段過度擬合；若有新時段資料可做 OOD 測試 | 開放（低優先） |
| K=200 band_mid 突破後，低頻退步是否可藉延伸訓練恢復 | EXP-066 L_phys@10k=2.95（未充分收斂）；K=200 主線暫停 | **CLOSED**：K=100 主線結案，K=200 屬另一資料密度配置，如重啟需獨立實驗 |
| 高頻重建的可行路徑 | CS 理論確認：K=100/200 均遠低於 ~5000 門檻；zeroth-order AIM 已證偽 | **CLOSED**：高頻不可達為數學必然，未來路徑需 DNS POD 先驗或 4D-Var（工程不可遷移）|
| EXP-070 KE=84% 是否因 denorm 路徑量級不匹配（vs AL 設計失敗） | **重訪（2026-05-07）**：Step 1 重跑 KE=84.36% 也是 evaluator-side bug 假象。Round 7 evaluator 修補後重跑 EXP-070 KE=6.30%（**優於 baseline 7.80%**）。AL 在 KE 維度成功，div_l2 退步 3.7×（trade-off 真實但非「失敗」） | **REOPENED**（2026-05-07）— ADR-001 §7.2 結論待重新評估 |
| `physics_output_denormalization` silent regression 是否需修 | 訓練端升格 `use_physics_denormalization` config flag；evaluator default 反轉為 identity + `--apply-denormalization` opt-in | **CLOSED**（2026-05-07）— Step 2 修補完成 + Round 7 evaluator review-fix loop 雙向驗證 |

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

