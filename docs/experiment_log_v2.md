# 實驗紀錄 v2（Stable Phase State 主檔）

> **Status**: Stable phase（2026-05-19 啟用）。研究已脫離前期探索（EXP-001~106），進入主線收斂、論文寫作、多 seed 統計確認階段。
>
> **Scope**: 此檔負責 **EXP-200 起所有 stable phase 實驗** 的 state 紀錄。Legacy EXP-001~106 全部保留在 [`docs/experiment_log.md`](experiment_log.md) 與其 archive，不動。
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
| **本檔** `docs/experiment_log_v2.md` | Stable phase STATE/INDEX、legacy 對照表 | **stable phase 任何實驗變更前** |
| [`docs/experiment_log.md`](experiment_log.md) | Legacy STATE（EXP-001~106 結論層）| 若 stable phase 結論不足，往回查 |
| [`docs/experiment_archive_kolmogorov.md`](experiment_archive_kolmogorov.md) | EXP-001~063 詳細 RECORD | 早期實驗追溯 |
| [`docs/experiment_archive_kolmogorov_post_k100.md`](experiment_archive_kolmogorov_post_k100.md) | EXP-064~106 詳細 RECORD（含 v2 axis-fix）| 近期 ablation 判讀 |
| [`docs/cylinder_log.md`](cylinder_log.md) | Cylinder 主線 | Cylinder 任務 |
| [`docs/diagnostics_log.md`](diagnostics_log.md) | denorm bug + CFD-rigour Q5/Q7/Q8 + Forward CFD | 評估值質疑 |
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

### Re=10000 主線（EXP-200, B3 architecture, n=5 multi-seed）

| 項目 | 現況 |
|---|---|
| Baseline ID | `EXP-200`（multi-seed group, 5 seeds）|
| 架構 | B3 = CfC + cross-attn + AL ρ=0.1 + 4-task GradNorm |
| Multi-seed configs | `configs/stable/exp_200_a.toml` ~ `exp_200_e.toml`（symlink → legacy）|
| 訓練 wall-time | ~2 h 24 m ± 4 m / seed（MPS, M3 Pro, fp32）|
| **KE rel-err (n=5)** | **10.77 ± 0.52 %** |
| u L2 (n=5) | 20.69 ± 0.46 % |
| v L2 (n=5) | 24.79 ± 0.51 % |
| ω L2 (n=5) | 52.65 ± 0.56 % |
| div L2 (n=5) | 0.066 ± 0.001 |
| ek_ratio (n=5) | 0.920 ± 0.020 |
| 統計分析 | [`artifacts/seed_statistics.json`](../artifacts/seed_statistics.json) |
| Inference cost | encoder 71 ms + query 1.5 ms / snapshot（seed=2, MPS fp32, batch=8192）|
| 結案聲明 | K=100 sparse reconstruction 已達資訊論硬上限；中高頻 ≈100% 為 K=100 Nyquist 限制（不可突破）|

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

## [INDEX] Pending（待跑）

| ID | Status | 配置 | Hypothesis | 對標 |
|---|---|---|---|---|
| **EXP-240_a** | `PENDING` | B0 + LES_T50 placement (seed=42) | LES placement 改善跨 architecture 仍有效 → KE ~15-17% | vs EXP-201_a (B0+DNS 18.5%), EXP-221 (B3+LES 12.4%) |
| **EXP-240_b** | `PENDING` | B0 + Random placement (seed=42) | placement-agnostic 性質跨 architecture 成立 → KE ≈ EXP-240_a | vs EXP-201_a (B0+DNS 18.5%), EXP-224 (B3+Random 13.3%) |

Decision gates（per config falsifiability section）:
- `B0+LES_T50 ≤ 16%`: claim 「LES placement effect 跨 architecture transferable」成立（gap 同 B3 受益 ~3pp）
- `B0+LES_T50 ∈ [16, 22]%`: 部分有效
- `B0+LES_T50 > 22%`: LES placement 改善僅 B3 專屬

---

## [STATE] Open Question（stable phase, 待補）

| 問題 | 現況 | 狀態 |
|---|---|---|
| Re=1000 stable phase multi-seed（n=5）| 尚未跑；目前只有 legacy EXP-030 single seed | 待開工 |
| `EXP-220` (= `EXP-200_c`) 5-seed sensor placement variance | 單 seed (seed=2) 結果；無法估計 placement-induced variance | 待開工（若需 paper-grade noise quantification）|
| LES robustness across LES_seed | 目前 LES generator 用 seed=42 single placement; 跨 LES seed 的 sensor variability 未測 | 待開工 |
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
