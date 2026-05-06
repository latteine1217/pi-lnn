# Pi-LNN Improvement Plan — Post Literature Review

> **Date**: 2026-05-06
> **Source**: 配合 [`docs/literature_review.md`](./literature_review.md) 與 [`docs/experiment_log.md`](./experiment_log.md)
> **Audience**: 共同研究員（含本人未來 self-review）
> **Constraint**: 嚴格遵守 CLAUDE.md 之 ENGINEERING_VISION（sensor-only + physics, no DNS full-field supervision），並依 SCIENTIFIC_HYPOTHESIS_PROTOCOL 為每項變更提出 Hypothesis / Expected_Change / Falsifiability。

---

## [SECTION 0] 改善方案三層優先級總覽

| Tier | 方向 | 預期 ROI | 工程可遷移性 | 估算工時 |
|---|---|---|---|---|
| **Tier 1（高優先）** | Sensor 時間軌跡編碼（SHRED-inspired branch encoder） | KE -1.0 ~ -3.0 pp 可能 | ✅ | 2–3 週 |
| **Tier 1** | Per-task Causal Weighting（修 EXP-068） | t=0 KE 與 div_l2 改善 | ✅ | 3–5 天 |
| **Tier 2（中優先）** | Mean-enforced Sensor Loss + Noise Robustness 評測 | 強化 paper 工程價值 | ✅ | 1–2 週 |
| **Tier 2** | Learned Sensor Placement（NeurIPS 2025 PhySense-style） | KE -1.0 ~ -2.0 pp 可能 | △ | 3–4 週 |
| **Tier 3（研究探索，需 research-only 標籤）** | DNS Latent Prior pre-training + sensor-only fine-tune | 可能突破 band_mid 上限 | ❌ training, ✅ inference | 4–6 週 |
| **Tier 3** | State-space (Mamba) trunk for long-time chaotic dynamics | 對 t>3 chaotic divergence 可能有效 | ✅ | 2–3 週 |

---

## [SECTION 1] Tier 1 — 立即可執行

### 1.1 Sensor 時間軌跡編碼（SHRED-Inspired Branch Encoder）

#### Hypothesis

> 我們 K=100 / 41 frames 的 sensor 資訊量遠超過 SHRED 在 K=3 的成功設定；
> 把 branch 從 **per-snapshot Sensor → token** 改為 **per-sensor time-series → token**，
> 可顯著提升中頻動力學（band_mid, k≈8..16）的可重建度，
> 因為 PDE 動力學的時間-空間耦合提供了對未觀測空間頻率的隱含資訊。

#### Expected_Change

| 指標 | 現況（EXP-064） | 預期 |
|---|---|---|
| KE rel-err mean | 7.80% | 4.5–7.0% |
| band_mid_last（k≈8..16） | 99.97% | **80–95%**（首次可量測突破） |
| band_high_last | 99.99% | 不變（仍受 K=100 CS 上限） |
| t=0 worst RMSE | ~0.1 | -10–20% |

#### Falsifiability

- 若 band_mid_last **無變化（保持 ≈100%）**：證明 K=100 + 41 frames 的時間軌跡資訊不足以解碼 k>5 模態，**SHRED 路徑在 sensor-only setup 下不適用**——此時將此實驗結論寫入 `[STATE] Rejected Directions`，並重新檢視「真正的 K=100 上限」。
- 若 band_mid 改善但 band_low 退步（總 KE 退步）：類似 EXP-066（K=200）的 trade-off，代表 channel allocation 衝突，需要更謹慎設計 head-split 機制。

#### 實作細節

**現況 branch 架構**（`src/lnn_kolmogorov.py` `DeepONetBranch`）：
- 輸入：`sensor_values[B, K, 2]`（u, v at K sensors, **單時刻或當前 query time**）
- 處理：positional encoding(sensor_position) + sensor_value → linear → tokens
- 輸出：`tokens[B, K, d_model]`

**改造後 branch 架構**：
- 輸入：`sensor_history[B, K, T_hist, 2]`（K sensors, **過去 T_hist frames** 的 u, v）
- 處理：
  1. 對每個 sensor i，取其時序 `s_i(t)[T_hist, 2]` 用 **CfC** 編碼為 `latent_i[d_temporal]`
  2. concat positional encoding(sensor_position_i) + `latent_i` → linear → token_i
  3. 整體 `tokens[B, K, d_model]` 進入 attention（與目前一致）
- 關鍵超參：`T_hist`（建議從 8 frames 起步），`d_temporal=32`，CfC `tau` 範圍 (-1, 1)（避開 EXP-067 失敗的 (-3, 1) fast 範圍）

**注意事項**：
1. **rollout 邊界**：目前 `time_marching=true`，在 t<T_hist 時 sensor history 不足。需明確定義 padding / teacher forcing 策略，並通過 ROLLOUT_PROTOCOL 檢查。
2. **causal vs bidirectional**：先測 causal CfC（只用 t≤t_query 的 sensor history）；EXP-059 已證實 bidirectional CfC 對 t=0 重建無益，**避免重複錯誤**。
3. **記憶體成本**：sensor history 把 batch tensor 從 `[B, K, 2]` 升維到 `[B, K, T_hist, 2]`，估算 GPU 記憶體 8x 增長；可能需要 reduce per-device batch 或 gradient checkpointing。

**驗證計畫**：
- Step 0：smoke test，確認 forward / backward 不 NaN
- Step 1（baseline 對照）：先在 Re=1000 下跑 5000 步，比較 EXP-030（KE 9.61%）
- Step 2（主實驗）：Re=10000 / K=100 / 10000 步，與 EXP-064（KE 7.80%）對照
- Step 3：若 Step 2 通過，掃 `T_hist ∈ {4, 8, 16, 32}` 找最優

#### Risk Tags

- `[RISK: ROLLOUT_PROTOCOL]`：rollout 邊界處 sensor history 不足，需設計 padding；偷看未來資訊會 invalidate 工程價值
- `[RISK: CfC_FAST_CHANNELS]`：tau 範圍若太低，重複 EXP-067 / EXP-069 的 dt=0.025 過敏感問題
- `[RISK: MEMORY]`：記憶體升維，可能需要 batch / accumulation 調整

---

### 1.2 Per-Task Causal Weighting（修正 EXP-068）

#### Hypothesis

> EXP-068 的 causal weighting 之所以讓 div_l2 +269% 退步，
> 根因為「all-residual cumsum」讓量級較大的 momentum 殘差主導 `w_t` 曲線，
> continuity 殘差（量級小一個數量級）的相對重要度被進一步壓低。
> 改用 **per-task cumsum**（對 momentum_u, momentum_v, continuity 各自獨立計算 `w_t`）後，
> 應能保留 causal weighting 加速早期收斂的優勢，同時不犧牲 continuity 約束品質。

#### Expected_Change

| 指標 | EXP-068（broken impl） | EXP-064（baseline） | 預期（per-task fix） |
|---|---|---|---|
| KE rel-err | 9.73% | 7.80% | **6.5–7.5%** |
| div_l2 | 0.680 | 0.184 | **<0.20** |
| t=0 KE | (degraded) | (baseline) | 改善 5–15% |

#### Falsifiability

- 若 per-task causal 仍讓 div_l2 退步：證明 causal weighting 在多 task PDE residual 設定下根本不適用——**寫入 Rejected Directions**，並考慮 alternative：對 momentum 啟用 causal、對 continuity 維持均勻平均。
- 若 t=0 KE 改善但 mean KE 持平：causal 主要作用是「重新分配時間 budget」，非提升整體精度——可作為「t≈0 重建問題」的專用 trick，但不是主線改進。

#### 實作細節

**現況實作（broken）**:
```python
# 偽碼，broken
all_residuals_t = ns_u_t + ns_v_t + cont_t  # [B, T]
cum_loss = cumsum(all_residuals_t, dim=1)
w_t = exp(-eps * cum_loss / (cum_loss + 1e-8))  # 由 ns_u/ns_v 主導
weighted_loss = (w_t * all_residuals_t).mean()
```

**修正實作（per-task）**:
```python
# 對每個 task 獨立
def causal_weight(residual_t, eps):
    cum = torch.cumsum(residual_t, dim=1)
    return torch.exp(-eps * cum.detach() / (cum.detach() + 1e-8))

w_ns_u = causal_weight(ns_u_t, eps=1.0)
w_ns_v = causal_weight(ns_v_t, eps=1.0)
w_cont = causal_weight(cont_t, eps=1.0)

L_phys = (w_ns_u * ns_u_t + w_ns_v * ns_v_t + w_cont * cont_t).mean()
```

**配套**：
- 保留 GradNorm（EXP-064 已驗證 -64% div_l2 改善）
- `eps` 不變（1.0），先驗證實作正確；若改善則掃 `eps ∈ {0.5, 1.0, 2.0, 5.0}`

**驗證計畫**：
- Step 0：smoke test，確認 per-task `w_t` 三條曲線量級可比較
- Step 1：Re=10000 / K=100 / 10000 步，對照 EXP-064 與 EXP-068

#### Risk Tags

- `[RISK: GRADNORM_CAUSAL_INTERACTION]`：GradNorm 在 task gradient norm 平衡，causal 在 task 內時間平衡；兩者共存可能產生意外耦合，需要謹慎觀察 `w_ns / w_cont` 與 GradNorm 推算之 task weight 的演化

---

## [SECTION 2] Tier 2 — 中期改善

### 2.1 Mean-Enforced Sensor Loss + Noise Robustness 評測

#### Hypothesis

> Mons et al. (PRF 2025) 證實 mean-enforced loss（強制 sensor 空間平均匹配 measurement 平均）
> 比 snapshot-enforced loss 對 noise 更 robust。
> Pi-LNN 目前用 snapshot-enforced（即 sensor MSE 在每個位置嚴格匹配），
> **加入 mean-enforced 作為輔助項**，並在 noise={0, SNR=20, SNR=10} 三種設定下評測，
> 可建立 Pi-LNN 對量測 noise 的健全度，補強 paper 的工程價值論述。

#### Expected_Change

| 指標 | clean (EXP-064) | + noise SNR=10（snapshot only） | + noise SNR=10（snapshot + mean） |
|---|---|---|---|
| KE rel-err | 7.80% | ~25–35%（推估） | **~10–15%** |

（具體數字需實測；上述為基於 PRF 2025 trend 的預估）

#### Falsifiability

- 若 mean-enforced 不改善 noise robustness：可能 Pi-LNN 的 GradNorm 已隱式做了 robust averaging，mean loss 是 redundant
- 若 clean case 因 mean loss 退步：代表兩種 loss 衝突，需要 task weight 調整

#### 實作細節

**Loss 修改**：
```python
# 現況（snapshot-enforced only）
L_data = ((sensor_pred - sensor_true) ** 2).mean()

# 加入 mean-enforced 輔助
L_data_snapshot = ((sensor_pred - sensor_true) ** 2).mean()
L_data_mean = ((sensor_pred.mean(dim=1) - sensor_true.mean(dim=1)) ** 2).mean()
L_data = L_data_snapshot + lambda_mean * L_data_mean
```

**初始 `lambda_mean`**：0.1（在 `configs/` 加新欄位 `mean_enforced_weight`，**記得同步更新 `DEFAULT_LNN_ARGS`**——KNOWN_PITFALLS 已記錄此規則）。

**Noise 注入**：
- 在 `dataset` 載入 sensor values 時，加上 `noise_snr` config 欄位
- noise sample 使用 fixed seed，確保跨實驗可重現
- 三組評測：clean / SNR=20 / SNR=10

**驗證計畫**：
- Step 0：smoke test
- Step 1：clean baseline 確認不退步
- Step 2：SNR=20 / SNR=10 對照 snapshot-only vs snapshot+mean

---

### 2.2 Learned Sensor Placement（NeurIPS 2025 PhySense-Style）

#### Hypothesis

> QR-pivot on POD modes 是 1990s–2010s 的最佳作法，但只是「reconstruction-agnostic 的線性代數最優」。
> NeurIPS 2025 兩篇 oral（Liu et al., Kim et al.）證明
> **reconstruction-aware 的學習式 sensor placement** 在同樣 K 下可達 1–3 pp 改善。
> Pi-LNN 應在 K=100 fixed 下，用 projected gradient descent 學一組更佳的 sensor 位置。

#### Expected_Change

| 指標 | EXP-064（QR-pivot） | 預期（learned placement） |
|---|---|---|
| KE rel-err | 7.80% | **5.5–6.8%** |
| band_low_last | 3.62% | **不變或微改善**（QR-pivot 已對 low 接近最優） |
| band_mid_last | 99.97% | **可能 micro 突破至 95–99%** |

#### Falsifiability

- 若 learned placement 不超越 QR-pivot：證明 QR-pivot 在 K=100 forced Kolmogorov 設定已是近似最優——這本身是有研究價值的 negative result，可寫進 paper
- 若 learned placement 在 train 時收斂但 test 時退步：placement 對 source DNS 過擬合，需要更強 regularization（PhySense 提到 variance minimization 的理論一致性）

#### 實作細節

**兩階段 PhySense-style training**：
1. **Stage 1**：固定 sensor 為 QR-pivot K=100，訓練 Pi-LNN 至收斂
2. **Stage 2**：固定 Pi-LNN weights，**將 sensor positions 設為可訓練 parameters**，用 projected gradient descent 在 [0, 1]² domain 上優化 sensor positions（projection = 對 grid cell center 取最近）
3. **Stage 3**：用 stage-2 學到的 placement 重新訓練 Pi-LNN（可從 stage-1 weights resume）

**重要約束**：
- Sensor positions 必須對應 DNS grid 上有量測值的 cell——這個約束來自我們的 sensor data 設定（從 DNS 取點）；學習式 placement 必須 project 回 grid cells
- 與 EXP-066（K=200）類似，需要重新生成 sensor values；但這次是 K=100 固定，只是 positions 變

**Risk**：訓練 pipeline 變動較大，需要 sensor generation script 對「learnable position」適配。

---

## [SECTION 3] Tier 3 — 研究探索（需嚴格 research-only 標籤）

### 3.1 DNS Latent Prior Pre-training + Sensor-only Fine-tune

#### Hypothesis

> CoNFiLD（Nature Comm. 2024）證明 DNS 預訓練 latent diffusion prior + sensor-conditional inference
> 可在 sparse sensor 下生成高品質 turbulence。
> 若我們在 **訓練階段**（research-only）使用 DNS pre-trained latent prior（如 VAE 或 diffusion），
> 並在 **inference / deployment** 仍只用 sensor + physics residual，
> 可能突破 band_mid/high 的 K=100 CS 上限——因為 prior 內含高頻先驗。

#### CRITICAL Engineering Compliance Note

> **此 direction 違反 ENGINEERING_VISION 的「training 時不使用 full-field DNS supervision」原則**。
>
> 必要條件（CLAUDE.md 已寫明）：
> 1. **必須在 config 與 experiment_log 明確標注「僅研究用，工程不可遷移」**
> 2. **不可作為主線 baseline**——只能作為對照組
> 3. **deployment 時須驗證 prior 來源 DNS 與 target case 的物理對應關係**
> 4. **paper 中需明確區分「engineering-transferable」與「research-only」結果**

#### Expected_Change

如果 prior 設計合理：
- KE rel-err 可能突破 5%（vs EXP-064 的 7.80%）
- band_mid_last 可能首次顯著突破至 50% 以下

如果 prior 與 inference 時的 PDE residual 不一致（distribution shift）：
- KE 改善但 div_l2 退步——表示 prior 引入了非 NS-consistent 的高頻成分

#### Falsifiability

- 若加 prior 後 band_mid 仍 ≈100%：prior 沒有真正提供高頻先驗（可能 prior 學到了 mode collapse 或太 smooth）
- 若 KE 改善但 div_l2 大幅退步：prior 與 sensor + physics 不一致，需要重新設計 prior 訓練 loss
- 若 inference 時 prior latent 與 sensor encoding 不能對齊：需要設計 cross-attention 橋接，可能引入新 hyperparameter

#### 建議架構

**option A：Latent VAE prior**
1. Train VAE on DNS full field（research only），latent dim=64
2. Pi-LNN trunk 接 VAE decoder：query → latent → field
3. Inference 階段，VAE 凍結；sensor + physics 只訓練 query→latent 路徑

**option B：Latent diffusion prior（CoNFiLD-style）**
1. Train conditional latent diffusion on DNS（research only）
2. Inference 階段，sensor 作為 conditional signal；diffusion sampling 取代 forward pass

→ 建議先試 option A（簡單，計算成本低，可快速 falsify hypothesis）

---

### 3.2 State-Space (Mamba) Trunk for Long-Time Chaotic Dynamics

#### Hypothesis

> EXP-015 確認 t=3.5–4.5 的 0.64 rad phase 偏差高峰是 Re=1000 Kolmogorov chaotic divergence 的 Lyapunov 物理本質。
> Mamba State-Space Model（NeurIPS 2024）在長序列 chaotic 系統上比 Transformer / FNO 更穩，
> 用 Mamba 取代或補強 CfC 作為 trunk 的時間 backbone，
> 可降低 t>3 區段的 phase error 累積。

#### Expected_Change

- t>3 phase_err 改善 20–40%
- KE rel-err 微改善 0.5–1.5 pp

#### Falsifiability

- 若 Mamba 與 CfC 性能無差別：證明在我們 t∈[0, 5]、41 frames 的中等長度下，State-Space 的長序列優勢未顯現
- 若 Mamba 實作 NaN 不穩定：MPS 可能不支援，需 fallback（KNOWN_PITFALLS 已記錄類似 SOAP eigh 案例）

---

## [SECTION 4] 立即不建議執行的方向

依據文獻與我們的 experiment_log 雙重驗證，以下方向應**列入 [STATE] Rejected Directions**：

| 方向 | 不建議原因 |
|---|---|
| 加大 model 至 d=512+ | EXP-065 證實 trunk 加深無效；capacity 不是瓶頸 |
| 更多 physics loss 變體（chebyshev / pressure poisson） | EXP-035..EXP-039 全 falsified |
| L-BFGS optimizer | EXP-052 計算成本不可行 |
| Transfer learning Re=1000 → Re=10000 | EXP-040/EXP-042 證實架構不匹配 + source 品質會破壞 |
| 純擴大 K（K=200, 500） | 違背「K=100 為固定研究設定」；除非整體 reframe |
| Bidirectional CfC | EXP-059 證實對 t=0 重建無益 |
| 直接套用「all-residual cumsum causal weighting」 | EXP-068 已 falsify；只能用 per-task 修正版 |

---

## [SECTION 5] 立即可下手的具體 commit 計畫

### Sprint 1（1 週）— 修 EXP-068

1. 在 `src/lnn_kolmogorov.py` 改 `causal_weight` 為 per-task
2. 寫 unit test 驗證 `w_t` 三條曲線量級可比較
3. 跑 EXP-070（暫定）：Re=10000 / K=100 / 10000 步 / per-task causal eps=1.0
4. 對照 EXP-064 與 EXP-068，更新 experiment_log 第 41 條

### Sprint 2（2–3 週）— SHRED-inspired branch

1. 在 `DeepONetBranch` 新增 `sensor_history_encoder`（CfC）
2. 修改 dataset：對每個 query (t, x, y) 提供 sensor history `[K, T_hist, 2]`
3. config 新增 `sensor_history_length`、`sensor_temporal_encoder` 兩欄位（同步 `DEFAULT_LNN_ARGS`）
4. Smoke test 在 Re=1000 / K=100 / 5000 步
5. 主實驗 EXP-071（暫定）：Re=10000 / K=100 / 10000 步 / T_hist=8

### Sprint 3（2 週）— Noise robustness 評測

1. 在 dataset 加 `noise_snr` config 欄位
2. 跑 EXP-072 / EXP-073 / EXP-074：clean / SNR=20 / SNR=10，三組對照 mean-enforced 與否
3. 整理 noise robustness 表格作為 paper 附錄

### Sprint 4+（後續）— Learned sensor placement / Latent prior

依 Sprint 2 結果決定優先級：
- 若 SHRED-inspired 改善 ≥1.5 pp KE：先做 noise robustness（Sprint 3），再考慮 placement
- 若 SHRED-inspired 改善 <0.5 pp KE：直接跳 placement 或 latent prior

---

## [SECTION 6] 投稿策略與 paper outline 草案

### 投稿目標

**Primary target**: **Journal of Computational Physics**（Energy Transformer 同 venue，accepts methodology + benchmark framing）

**Secondary**: Phys. Rev. Fluids（Mons 同 venue）

### Paper title 候選

> **"Engineering-Transferable Sparse Sensor Reconstruction of 2D Forced Turbulence: A DeepONet–CfC Framework with Information-Theoretic Bounds"**

### Section outline

1. **Introduction**
   - 問題定義：sparse sensor + PDE residual → flow field
   - 工程動機：拒絕 DNS supervision；強調 K=100 設定的真實性
   - Contributions（建議三點）：
     1. New architecture: DeepONet + CfC + sensor-temporal encoding
     2. SOTA on K=100 / Re=10000 (KE 7.80% → 期望 Sprint 2 後 <6%)
     3. **Wavelet sparsity diagnostic** 量化 K=100 的 information-theoretic ceiling
2. **Related Work**
   - 三軸分類：sensor-only+physics / DNS-supervised / DNS-pretrained operator
   - 強調我們屬於第一類，與 PRF 2025 直接可比
3. **Method**
   - Pi-LNN 架構細節
   - Sensor-temporal encoding（Sprint 2 成果）
   - Per-task causal weighting（Sprint 1 成果）
   - GradNorm + LearnableFourierEmb
4. **Information-Theoretic Bound Analysis**
   - Wavelet sparsity diagnostic
   - CS bound 計算
   - 為什麼 band_mid/high 不可達
5. **Experiments**
   - Re=1000 / Re=10000 main results
   - Noise robustness（Sprint 3）
   - Cylinder wake transferability（CEXP-002）
   - Ablations
6. **Discussion**
   - 工程可遷移性原則
   - 未來方向：learned placement / latent prior（明確標 research-only）
7. **Limitations**
   - K=100 在 Re>10000 不適用
   - 2D periodic domain only
   - Forcing mode 已知

---

## [SECTION 7] 核心建議的優先級宣告

> **如果只能執行一項改善：**
>
> 應該執行 **Sprint 1（per-task causal weighting）**——成本最低（3–5 天）、風險最小、預期改善最確定（直接修 EXP-068 的 falsified bug）、且結果無論正負都會更新 experiment_log 第 41 條的判讀。
>
> **如果只能執行兩項：**
>
> 加上 **Sprint 2（SHRED-inspired sensor temporal encoder）**——這是文獻搜尋中**唯一發現我們完全沒涉及的有力方向**，且 hypothesis 在 K=100 / 41 frames 設定下尤其合理。
>
> **如果有充裕時間（2–3 個月）：**
>
> 全部執行 Tier 1 + Tier 2，Tier 3 只在 Tier 1 + 2 結果不足以支撐 paper SOTA 主張時啟動，並嚴格遵守 research-only 標籤。

---

Check: [Protocol_Adhered]
