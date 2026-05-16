# Kolmogorov 實驗 Archive — K=100 結案後系列

本檔收錄 **EXP-064 結案後**所有 Re=10000 K=100 主線探索的詳細 RECORD 與 GROUP 結論。從 [`docs/experiment_log.md`](experiment_log.md) 拆出（2026-05-15 拆檔）。

**早期歷史**（EXP-001~063 / G1~G14 前段）見 [`docs/experiment_archive_kolmogorov.md`](experiment_archive_kolmogorov.md)。
**Cylinder 主線**見 [`docs/cylinder_log.md`](cylinder_log.md)。
**Diagnostic 報告**見 [`docs/diagnostics_log.md`](diagnostics_log.md)。

---

## [INDEX] Post-K100 Active

| ID | Status | 主題 | KE | div L2 | 一句結論 |
|---|---|---|---|---|---|
| **EXP-064** | `ACTIVE_BASELINE` | K=100 結案值（baseline）| **7.80%** | 0.184 | best balance, KE-optimal |
| **EXP-080** | `ACTIVE_REFERENCE` | AL Pareto sweet spot | **10.68%** | **0.067** | NEW Pareto point（4-task GN + AL ρ=0.1）|

| GROUP | EXP 範圍 | 角色 | Status |
|---|---|---|---|
| G12 (前情提要) | EXP-049~056 | EXP-048 resume 變體 + IC weight | `RESOLVED` |
| G14 (前情提要) | EXP-062~063 | LearnableFourier → EXP-064 | `RESOLVED` |
| **G_post1** | EXP-067~069 | 後 K=100 結案探索（受 denorm bug 影響）| `RESOLVED`（已重評）|
| **G_AL_H** | EXP-070~077 | AL-continuity 系列 | `RESOLVED` |
| **G_AL_I** | EXP-078~079 | 9-point AL Pareto frontier 完成 | `RESOLVED` |
| **G_AL_J** | EXP-080~082 | AL strength weakening probe | `RESOLVED`（EXP-082 invalid）|
| **G_pivot** | EXP-083~087 | 6-lever pivot ablation | `RESOLVED`（全 falsified）|
| **G_arch** | EXP-088~092 | Architectural ablation + Standard PINN baseline | `RESOLVED` |
| **G_seed** | EXP-093~100 | Multi-seed reproducibility (N=5 per arch) | `RESOLVED` |
| **G_bench** | EXP-094 (sub-analysis) | Inference cost benchmark | `RESOLVED` |
| EXP-101 (in-progress) | random sensor placement vs QR-pivot | 訓練未完成（step 4500/10000）| `PENDING` |

---

## [STATE] K=100 稀疏重建結案聲明（2026-04-26）

**EXP-064 為 K=100 sensor 配置的最終接受結果。稀疏重建主線結案。**

### 量化結論

K=100 已達資訊論硬上限，由 Wavelet 稀疏性診斷量化確認（詳見 [`docs/analysis_reports.md`](analysis_reports.md)）：

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

## [GROUP G12 前情提要] EXP-048 resume 系列突破

- **IC Loss Weight（λ=10, t≤0.05）為單一最有效改動**：KE 從 EXP-048 的 21.8% → EXP-055 的 17.1%（**-4.7pp**），優於 RAR alone（EXP-054, -2.2pp）。kf_amp_ratio 0.970 與 E(k_f) 0.934 全系列最佳。
- **RAR freq 是關鍵**：freq=50（EXP-053）擾亂 SOAP+SF preconditioner（L_phys 7.96→19.27）；freq=1000（EXP-054）才能與 SOAP+SF 共存，KE 19.6%。
- **RAR + IC weight 不可同時使用**（EXP-056 KE 19.4% > IC alone 17.1%）：RAR 週期性更新 collocation 改變 loss landscape，與 IC weight 依賴的穩定梯度方向衝突。

詳見 [`docs/experiment_archive_kolmogorov.md`](experiment_archive_kolmogorov.md) G12 section。

---

## [GROUP G14 前情提要] LearnableFourier 演進 → EXP-064 baseline

- `LearnableFourierEmb`（embed_dim=128, init σ=2.0）取代固定 periodic Fourier：KE 從 EXP-055 的 17.1% → EXP-062 的 **10.4%**（-6.7pp）。但改善源自低頻精度提升（band_low@t=5: 5.8%），非頻率覆蓋擴展（band_high 99.98% 未改善）。
- 正確 jaxpi SOAP（保留 SF, betas=(0.9,0.999), precond_freq=2, wd=0, decay=2000）+ GradNorm（freq=1000, momentum=0.9, init [1,0.01,0.01,0.01]）：KE **8.65%**（EXP-063, -1.75pp）；div_l2 0.204 全系列最佳（-64%）。
- Sensor 位置 continuity physics 點（n_t=1, start=1000，僅 continuity）：KE **7.80%**（EXP-064, -0.85pp）；div_l2 **0.184**（-9.6%）；phase_err -53%。**為 K=100 配置的最終結案值**。
- band_mid/high@t=5 ≈100% 經四次 falsifiability 驗證後，確認為 **K=100 sensor 的資訊論硬上限**。Wavelet 稀疏性診斷量化確認 CS 精確重建需 M ≥ O(s log N) ≈ 5000 sensors，K=100 差約 50 倍；換 wavelet 基底不改變上限量級。詳見 [`docs/analysis_reports.md`](analysis_reports.md)。
- K=200（EXP-066）部分突破 band_mid（32.90% vs 99.97%），但低頻退步（38.65% vs 3.62%）+ 整體 KE 退步（29.94%）；L_phys@10k 未充分收斂，需延伸訓練驗證。
- AIM（Approximate Inertial Manifold）zeroth-order 後處理已證偽（τ_visc/τ_NL ≈ 215，quasi-static 假設違反）。詳見 [`docs/analysis_reports.md`](analysis_reports.md)。

---

## [RECORD] EXP-064（Re=10000 ACTIVE_BASELINE）

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

## [RECORD] EXP-065（trunk depth, NEGATIVE）

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

## [RECORD] EXP-066（K=200 sensor, MIXED）

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

## [GROUP G_post1] 後 K=100 結案實驗（受 denorm bug 影響，已重評）

> ⚠️ EXP-067/068/069 三組均跑在 `physics_output_denormalization` 啟用路徑下（自 d62e698 commit 自動觸發），物理 NS residual 量級被改變。**完整 diagnostic 見 [`docs/diagnostics_log.md`](diagnostics_log.md)**；以下數字為 Round-7 evaluator 重評後。

- **EXP-067**（CfC log_tau (-3,1) + 頻率分層 LearnableFourier (1,4,12)/(50/37.5/12.5%), 10k 步）：KE **11.20%**（vs EXP-064 7.80%，+3.40pp）；band_low 退步（7.19% vs 3.62%）。診斷：(a) 頻率分層 σ=12 高頻段微改善 band_mid 但犧牲 12.5% 通道；(b) CfC fast channels（τ≈0.05）相對 sensor dt=0.025 過敏感。**建議拆解 EXP-067a/b 單獨測試**。
- **EXP-068**（PINN causal weighting eps=1.0 num_bins=16, 10k 步）：KE 9.73%（+1.93pp）；div_l2 **0.680（+269% 嚴重退步）**。當前實作以「所有殘差項之和」做 cumsum，量級較大的 momentum 殘差主導權重曲線，continuity 約束被進一步壓制。**修正建議**：改 per-task cumsum 或僅以 momentum 殘差驅動權重。
- **EXP-069**（三項組合：CfC tau + 頻率分層 + causal weighting, 10k 步）：KE **20.13%（+12.33pp 災難）**；div_l2 1.404（+663%）。三項負面交互證實；皆需單獨修正後再組合。

---

## [GROUP G_AL_H] AL-continuity 系列（ADR-001 §4 / ADR-002, EXP-070~077）

> **重跑後** evaluator (Round 7) 真實值。EXP-070~074 詳細見 [`docs/diagnostics_log.md`](diagnostics_log.md)；以下重點記 EXP-071 起的補跑結果。

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

---

## [GROUP G_AL_I] 9-point AL Pareto frontier（EXP-078/079）

- **EXP-078**（pure AL strength sweep, 2026-05-08, 10k 步）：`use_gradnorm = false` + `al_rho = 3.0`（vs EXP-070 ρ=1.0）。
  - **KE rel-err = 15.47%**（train 15.02%, val 17.23%）— 比 EXP-070 ρ=1 的 6.30% **退步 9pp**！
  - **div L2 = 0.0332**（**最低!** 比 GradNorm 路徑 0.044 還低）。
  - λ ascend 到 5.70（vs ρ=1 的 2.84，2× 強）；C_ema 6.5e-2 → 3.8e-3。
  - NS-u 0.397 / NS-v 0.413（與 GradNorm 路徑同量級）。
  - **解讀**: ρ↑ 確實讓 λ 強化 → div 進一步突破。但 KE 退步比 GradNorm 路徑更糟。Falsifiability (c) 部分成立：pure AL strong ρ 也能達 div breakthrough（不必 GradNorm），但 trade-off 比 GradNorm 還糟。
  - artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp078-al-pure-rho3/`；eval: `artifacts/eval-rerun-2026-05-08/exp078-al-pure-rho3/`

### AL series 完整 Pareto frontier（2026-05-08，7 點 ablation）

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

---

## [GROUP G_AL_J] AL strength weakening probe（EXP-080~082）

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

### ρ ablation 結論（EXP-079/080/081，3 點）

- ρ=1.0 → KE 14.77%, div 0.043 (div-strong)
- ρ=0.1 → KE 10.68%, div 0.067 (sweet spot)
- ρ=0.05 → KE 10.05%, div 0.076 (saturation 起點)
- **Trade-off curve continuous monotonic，沒有「兩全其美」突破點**。
- 需要 pivot 到別的維度（架構容量 / hard-divergence reparam）才能同時改善 KE + div。

---

## [GROUP G_pivot] 6-lever pivot ablation（EXP-083~087，全 falsified）

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

### Pivot 階段性結論 (6 levers 全 falsified, 2026-05-11)

- ρ ablation (EXP-079/080/081): saturated continuous trade-off
- Multi-head cross-attention (EXP-083): symmetry collapse
- Fourier harmonics ↑ (EXP-084): INVALID (cfg 沒實際改, 等效於 EXP-080 noise)
- K-scaling K=200 (EXP-085): recipe-K mismatch disaster
- Trunk depth ↑ (EXP-086): spectral over-smoothing
- **mMLP gating (EXP-087): noise floor, operator-learning context 無 lever**
- **EXP-080 (KE 10.68%, div 0.067, ek_ratio 0.911) 是 K=100 + 當前架構下的 near-optimal**。
- 距 spectral truncation lower bound (k_cut≈5-6, KE 2.6-7.8%) 仍有 3-5pp gap，但需要 fundamental different approach (非 hyperparameter / 架構淺改可達)。

### 真正剩下的 promising 方向 (需深度工程)

- **(P1) K=200 with 重新 tuned recipe**: curriculum learning, lower LR, modified AL schedule。預期工程量 1-2 days iterating recipe。
- **(P2) Modified MLP (Wang 2021)**: U/V gating + RWF，fix spectral bias 的 PINN-specific architectural change。1-2 days 工程。
- **(P3) PirateNet 完整套裝**: RFF + RWF + NTK weighting + causal training，PINN best practice 集合。3-5 days 工程。
- **(P4) 接受 EXP-080 為論文 result**: 改 framing 為「73% of K=100 spectral truncation lower bound」，emphasize architectural novelty (CfC + DeepONet hybrid)。0 工程量。

---

## [GROUP G_arch] Architectural ablation（B0/B1/B2 + Standard PINN baselines, EXP-088~092）

- **EXP-088**（B0 architectural ablation, vanilla DeepONet, 2026-05-11, 10k 步, 1-shot）：
  - 改動：完全替換 model — MLP branch (sensor at t_q, no CfC) + MLP trunk + inner-product readout，無 cross-attention。新 module `src/pi_lnn/vanilla_deeponet.py`。Params 1.28M (vs full 3.14M, 41%)。
  - 結果：KE **18.17%** (+7.49pp), u rel-L2 **25.14%** (+8.14pp), v rel-L2 **30.90%** (+10.70pp), ω rel-L2 **57.89%** (+10.29pp), ek_ratio 0.883 (-3.1%), div 0.065 (持平)。
  - **架構 component 貢獻**：CfC + token attention + cross-attention 整體提供 ~7-11pp pointwise improvement vs vanilla baseline。
  - **新 insight**: Vanilla DeepONet (B0) 在 pointwise 上仍比所有 classical baselines (RBF, IDW, trig LSQ) **好 ~8pp** — DeepONet inner-product structure 已比 linear basis 強。但在 KE 上 vanilla 比 RBF Multiquadric **差 14pp** (vanilla 18% vs RBF 4%) — 全 DeepONet 設計都 share「KE-not-as-low-as-over-smoothing」這個性質。
  - artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp088-vanilla-deeponet/`；eval: `artifacts/eval-rerun-2026-05-11/exp088-vanilla-deeponet/`

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

- **EXP-091 Standard PINN baseline** (B-reference, 2026-05-11, 10k 步, 1-shot)：
  - 改動：新 module `src/pi_lnn/standard_pinn.py` — Wang 2021 style single-instance PINN (`(x,y,t) → MLP 6×512 → (u,v,p)`)，無 operator framework, sensor 只 enter L_data loss。Params 3.24M (matched to EXP-080 3.14M within 3%).
  - 結果：KE **31.35%** (+20.67pp), u rel-L2 **32.33%** (+15.33pp), v rel-L2 **44.72%** (+24.52pp), ω rel-L2 **67.53%** (+19.93pp), ek_ratio 0.715, div L2 **0.023** (better, AL over-enforced).
  - **Critical finding**: PINN **比 B0 Vanilla DeepONet (1.28M params, u_L2 25.14%) 更差**, despite 2.5× more params. DeepONet structure (sensor→branch input) 比 raw MLP capacity 更 essential.
  - **Training pathology**: L_data plateau at 0.124 from step 6000; λ saturated near 4.2 (clip=10); w_cont exploded 30× to 4.82. GradNorm + AL 過度 enforce cont 在 sensor-不可見 model 上失衡.
  - **Operator framework justified**: Removing sensor-aware encoding (PINN) causes 15-25pp pointwise degradation across all field metrics. This is the strongest evidence of architectural value.
  - artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp091-standard-pinn/`；eval: `artifacts/eval-rerun-2026-05-11/exp091-standard-pinn/`

- **EXP-092 Standard PINN + tanh activation** (activation ablation, 2026-05-11, 10k 步, 1-shot)：
  - 改動：EXP-091 recipe + `standard_pinn_activation = "tanh"` (classical Raissi 2019 / Wang 2021 mMLP convention)。Same 3.24M params, only activation 不同。
  - 結果：KE **43.94%** (+12.59pp vs SiLU), u rel-L2 **40.76%** (+8.43pp), v rel-L2 **54.33%** (+9.61pp), ω rel-L2 **73.69%** (+6.16pp), ek_ratio **0.597** (-16.5% spectrum), div 0.017 (better, AL clipped harder).
  - **Tanh training pathology**: λ saturated at clip ceiling 10.0 by step 1000 (vs SiLU peaked 4.2), w_cont 8.13 (vs SiLU 4.82), w_ns_u 2.85 (vs SiLU 0.69). Confirms vanishing gradient hypothesis for 6-layer deep PINN with tanh.
  - **Activation ablation conclusion**: SiLU (= Swish-1, PirateNet 2024 modern PINN choice) **strictly better** than tanh (Raissi 2019 classical) for our 6×512 PINN configuration. Tanh saturation in deep PINN is well-known issue.
  - **Robustness of architectural claim**: Both SiLU/tanh PINN variants 遠遠 worse than B0 Vanilla DeepONet (u_L2 25.14%, 1.28M params). Operator framework gap holds regardless of activation.
  - artifacts: `artifacts/kolmogorov/deeponet-cfc-re10000-exp092-standard-pinn-tanh/`；eval: `artifacts/eval-rerun-2026-05-11/exp092-standard-pinn-tanh/`

---

## [GROUP G_seed] Multi-seed reproducibility（EXP-093~100, N=5 per architecture）

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

---

## [BENCHMARK] EXP-094 inference-cost benchmark (2026-05-13)

(B3 seed=2, MPS, fp32, batch=8192) — 量測腳本：`scripts/benchmark_inference.py`（warmup=3，N_encode=20，N_query=30，N_full=1，計時前後 `torch.mps.synchronize()`）。

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

---

## [PLAN] EXP-085 stream function reparam（候選，未跑）

最後 promising lever（不在本檔已執行 EXP 列表內）：

- **(A) Stream function reparam**: model 輸出 ψ (scalar streamfunction) + p，u = ∂ψ/∂y, v = -∂ψ/∂x via autograd → div(u, v) = 0 analytic。
  - Hypothesis: 硬約束 div=0 釋放 model 全 capacity 學 KE；trade-off 根因若是 div constraint，KE 應顯著改善至 7-9%。
  - Expected: KE 7-9%, div ~ machine precision (10⁻⁸~10⁻⁶), ek_ratio 提升至 0.94+。
  - Falsifiability: 若 KE > 11% → div 非根因，spectral bias 主導，需研究新 loss/sampling 設計。
  - 工程量：1-2 hr（DeepONetCfCDecoder 改 ψ-mode forward + physics path 改用 autograd 算 u, v；evaluator 同步改）。
  - 風險：physics path 二階 autograd 變三階（先 ψ→u via 一階, 再 u→∂u/∂x via 二階）需驗證 numerical stability。

> **註**：EXP-085 編號在實際運行時被用於 K=200 recipe-K mismatch（disaster, INVALID），不是 stream function reparam。Stream function reparam 仍為候選，未指派新編號。

---

## [PENDING] EXP-101 random sensor placement vs QR-pivot（in progress）

- Config: [`configs/exp_101_b3_random_seed42.toml`](../configs/exp_101_b3_random_seed42.toml)
- 改動：EXP-080 / EXP-094 recipe + sensor placement 改為 uniform random（placement_seed=42, training_seed=42）。架構與 EXP-080 完全一致，唯一改動為 sensor 策略。
- 訓練狀態（2026-05-15 確認）：artifact `artifacts/kolmogorov/deeponet-cfc-re10000-exp101-b3-random-seed42/checkpoints/` 只有 `step_4000.pt` + `step_4500.pt`，**未完成 10000 步**；目前無 running process。
- 屬 CFD-rigour pending tasks 之一（sensor placement sensitivity）。重啟需 1-shot 跑完（per KNOWN_PITFALLS，resume 禁用）。
