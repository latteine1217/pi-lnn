# ADR-001 — 研究路線優先序與架構擴張凍結

**Date:** 2026-05-04
**Status:** Accepted
**Scope:** Pi-LNN 專案下一階段（Phase 2）的研究方向與「不做什麼」清單。
**Supersedes:** —
**Reviewers:** latteine（Owner），GPT-5（external review），Claude-Opus-4.7（codebase review）

---

## 1. 背景

EXP-064 (Kolmogorov, K=100, Re=10000) 已建立基線：
- KE relative error ≈ 7.80%
- div L2 ≈ 0.184
- u/v RMSE ≈ 0.0689 / 0.0621

兩輪外部架構審查（GPT-5）提出 weak form / FVM / temporal cross-attention / geometry tokens / Augmented Lagrangian / smoothing reframing 等方向。經 Claude codebase 對照與 GPT 修正，**多數提案被降級或凍結**。本 ADR 鎖定共識，避免後續又重新爭論同一批選項。

---

## 2. 研究線分割（Decision-1）

**將專案明確切成兩條獨立研究線，禁止互相借用主張。**

| 線 | Dataset | 主張 | 主要 metric |
|---|---|---|---|
| **A. Kolmogorov sparse inverse** | Kolmogorov Re=10000, K=100, periodic | sparse-sensor physics-constrained reconstruction 受 information-theoretic limit 約束（low-band recoverable, mid/high-band 不可恢復是 observation operator 的本質瓶頸） | KE error / spectral band error / div L2 / sensor-count ablation |
| **B. Cylinder geometry-aware** | Cylinder flow, hard BC | geometry-aware constraints（hard BC, body-aware sampling, outlet BC, coordinate chain rule）對複雜邊界 reconstruction 比 pointwise residual 重要 | div / vorticity / near-wall error / hard-BC ablation |

**Why:** 兩條線的物理結構與資訊瓶頸不同。混合主張會讓論文 framing 自我矛盾（用 periodic 證明 geometry-aware 沒意義；用 cylinder hard BC 解釋 K=100 欠定問題不合理）。

**How to apply:**
- 任何實驗的 README / spec / paper draft 必須標明屬於 A 或 B。
- 不要寫「我們的方法在 Kolmogorov 與 Cylinder 都有效」這種跨線主張，除非雙線實驗矩陣完整。

---

## 3. Filtering vs Smoothing 定位（Decision-2）

當前 `DeepONetCfCDecoder` 採 `searchsorted(right=True) + h_states[idx]`，搭配 `use_bidirectional_cfc=False` 預設 → 嚴格意義上是 **causal filtering**，不是 smoothing。

**決策：**
- 短期不立刻做 smoothing 實驗（owner 決定延後）。
- README / paper framing **必須誠實標註**目前主線是 **causal sensor-conditioned filtering / reconstruction**，不可宣稱是 smoothing。
- 若未來需要 smoothing，**單純打開 `use_bidirectional_cfc=true` 即可**（`encoders.py:133–139, 188–190` 已實作），不需新 encoder、不需 task_mode flag、不需 temporal window attention。
- **Forecasting 主動排除**，不寫成 future work — K=100 sparse + Re=10000 turbulent 的 Lyapunov 時間 + 資訊上限雙重夾擊，本來就不可行。

---

## 4. 首要 Actionable：AL-continuity（Decision-3）

`continuity = ∇·u` 的特性適合 Augmented Lagrangian：scalar、無 gauge 自由度、與 div_L2 metric 直接對應。

**規則：**
- AL 與 GradNorm **不可同時控制同一 task**。
  - 若採 AL-cont，GradNorm tasks 必須移除 cont → 變 3-task `[data, ns_u, ns_v]`。
  - 移除後務必檢查 `GradNormWeights.normalize_to_data_()` 在 3-task 下的行為（`losses.py:31`），確認 ns_u/ns_v 不會被自動放大。
- AL constraint 形式採 `C = mean((∇·u)²)`，不採 `mean(∇·u)`（後者 batch 中正負抵消）。
- 標準 AL 更新：$\lambda \leftarrow \lambda + \rho \cdot C$，加 `al_lambda_clip` 防爆。

**實驗矩陣（編號從 EXP-070 起，因 067/068/069 已被 cfc_tau / causal_weighting / combined 佔用）：**

| Exp | Base | 變動 | 目的 |
|---|---|---|---|
| EXP-070 | EXP-064 | GradNorm off + AL-only-on-cont | 確認 AL 本身能否降 div_L2 |
| EXP-071 | EXP-064 | 3-task GradNorm + AL-cont | 確認 AL 與 GradNorm 相容 |
| EXP-072 | EXP-064 | 5-task GradNorm + `poisson_loss_weight=0.1`（無 AL） | 對照組：壓 p 結構 vs 壓 ∇·u 哪個對 div 更有效 |

---

## 5. 後續研究序（Decision-4）

凍結之後，明確排序：

1. **AL-continuity** + Pressure Poisson 對照（§4，EXP-070..072）
2. **Ensemble uncertainty**（3–5 seeds，支撐欠定論述）— EXP-073
3. **K-ablation** K=50（K=100/200 已有 EXP-064/EXP-066）— EXP-074
4. **Cylinder 線**（hard BC ablation, body-aware sampling, outlet BC）獨立推進
5. **Smoothing**（`use_bidirectional_cfc=true`），延後到上述完成且仍需要時再做
6. **FVM / weak residual**：last resort，1–4 全做完且仍卡在 physics constraint 才考慮

---

## 6. 凍結清單 — 短期不做（Decision-5）

明確標記為「**不要再被任何後續審查重新提案**」的方向：

| 提案 | 凍結理由 |
|---|---|
| Temporal cross-attention to time window | 與 `use_bidirectional_cfc` 高度重疊；先做 bidirectional ablation 才判斷是否需要 |
| Geometry tokens (SDF / boundary tag as sensor input) | Kolmogorov periodic 無意義；Cylinder 已有 hard BC + locality decay + relative pos bias 覆蓋 |
| Global physics tokens (A, k_f, Re as token) | EXP-064 為單一 (Re, A, k_f) 配置，Re 已有 `re_proj` bias；只在 multi-Re/multi-forcing 訓練才值得升級 |
| Pseudo-FVM weak residual | 不解決 K=100 資訊瓶頸；每點變 5–9 quadrature points，二階 autograd 成本上升 5–9× |
| Forecasting 模式 | Lyapunov + 資訊上限雙重不可行，主動排除 |
| Stream function reparam（Decision-3 替代方案） | 暫不採用 — 與 `forward_uvp` 直接學 (u,v,p) 的現有架構衝突大；若 AL-cont 失敗才考慮 |

---

## 7. 觸發重新評估的條件 / 已執行實驗回饋

### 7.1 EXP-070..073 實驗結果（2026-05-04 ~ 2026-05-06）

| Exp | Setup | div L2 | u RMSE | KE rel-err | 結論 |
|---|---|---|---|---|---|
| EXP-064 (baseline) | GradNorm 4-task, sensor_physics=true | 0.184 | 0.069 | 7.8% | 場品質保住 |
| EXP-070 | AL ρ=1.0, clip=10, **sensor_physics=false** | 0.040 ✓ | 0.252 | **84.3%** | 場崩潰 |
| EXP-070b | AL ρ=0.1, clip=0.05, **sensor_physics=false** | 0.170 | 0.251 | **84.4%** | 場崩潰（不是 AL 強度問題）|
| EXP-072 | Poisson + GradNorm 5-task, sensor_physics=true | 0.089 | 0.253 | **84.7%** | 場崩潰（GradNorm w_ns 暴走至 ~2.0）|
| **EXP-073** (diagnostic) | **EXP-064 同設定 - sensor_physics**(只關此一) | 0.118 | 0.251 | **84.5%** | **崩潰** — 確認 sensor_physics 是關鍵 |

**核心發現**：四個崩潰的 EXP（070/070b/072/073）的 u/v/KE/ω metrics 像素級相同，只有 div_L2 隨 physics 強度浮動。**真正的差異變數不是 AL/Poisson/GradNorm，而是 `use_sensor_physics`**。

**機制**：K=100 sensor 位置在 wavelet 域對 k≤16 條件數約 11（well-conditioned），是稀疏場景下唯一能穩定 anchoring 物理約束的點集。隨機 collocation 64 點/step 沒有此 conditioning 保證。EXP-064 GradNorm 收斂到 w_ns≈0.057 / w_cont≈0.039 是 **K=100 場景下能保住場品質的最強物理壓力**；任何把 effective physics gradient 推遠超此值的設計（AL ramp、5-task GradNorm 推飛、隨機 collocation 主導）都會把模型推離 informationally feasible region。

### 7.2 結論：AL spec v5 (Option 2) 取代 v4

v4 設計把 `use_sensor_physics=false` 寫進 pre-condition assert（理由是 sum-of-two-means 污染 EMA）。v5 反向：**AL constraint C 必須從 sensor 位置 cont² 計算，`use_sensor_physics=true` 為必要條件**。

此修正與 ADR-001 §1 主張（sparse-sensor reconstruction 受 information-theoretic limit 約束）邏輯一致：物理約束的有效性受限於 sensor 條件數，而非可由模型強行壓出來。

### 7.3 解凍清單（保留原條款 + 新增）

下列任一發生時，本 ADR 部分條款可重新討論：

- **EXP-074 (AL Option 2) 仍未能達 div L2 < 0.10** → AL 機制本身與 sensor_physics 共存仍有問題 → 重新考慮 stream function reparam 或 pseudo-FVM
- **K-ablation 顯示 K=200 對 mid-band 有顯著改善** → information bottleneck 論述需修正，可能值得做 K=500 並重新評估 sensor placement uncertainty
- **multi-Re / multi-forcing 訓練成為主線** → 解凍 global physics tokens 的討論
- **bidirectional CfC ablation 顯示 t≈0 仍有顯著誤差** → 解凍 temporal window attention

---

## 8. 編號規範

避免 GPT 等外部審查者再用過時編號：

- 已佔用：EXP-001 ~ EXP-069，EXP-cylinder-001/002
- **下一個可用：EXP-070**
- 任何外部建議若指定 EXP-067/068/069，需轉譯成 EXP-070+。

---

## 9. 不在本 ADR 範圍

- 個別實驗的詳細 config（屬於 spec 文件，放 `docs/superpowers/specs/`）
- AL infrastructure 的具體實作（`AugmentedLagrangianMultiplier` class 介面、training loop hook、config schema）— 待 EXP-070 開工前另寫 design spec
- Cylinder 線的 BC ablation 矩陣 — 由 cylinder commit 系列獨立追蹤
