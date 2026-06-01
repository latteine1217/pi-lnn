# CfC-DeepONet 容量榨取研究筆記

> **Date**: 2026-06-01
> **Scope**: 判斷 CfC 作為 branch encoder 的容量是否為 operator learning 瓶頸、本專案是否已榨乾其能力。
> **Method**: deep-research harness（5 angles × 並行 WebSearch → 21 sources → 100 claims → 25 條 3-票對抗驗證 → 15 confirmed / 10 killed），輔以官方 `raminmh/CfC` 原始碼比對與本 codebase 串接核對。
> **Cross-ref**: 感測器資訊論上限與 sensor-baseline 比對見 `docs/literature_review.md`（本筆記不重複 K=100 ceiling 推導，只引用其結論）。
> **Status note**: 本檔為 research note（架構容量分析），非 experiment state。sweep 實測結果填入 `docs/experiment_log_v2.md`。

---

## [SECTION 0] Executive Summary

| 核心問題 | 文獻/證據結論 | 本專案現況 |
|---|---|---|
| CfC「表達容量」夠不夠 | CfC 與 ODE-based peers **同級表達力**（非更強）；容量本身少有成為瓶頸 | 容量不太可能是主瓶頸 |
| 有沒有「榨乾」CfC | **沒有** — 本專案把 CfC 定義性的 liquid time-constant **凍結成 static 參數**（§1） | 已補 opt-in flag 恢復，待 sweep 驗證 |
| 怎麼診斷飽和 | 文獻**無**經對抗驗證的飽和診斷（SVD-decay 等被 0-3 駁回，§5）| 只能自做 capacity sweep（§4） |

**可靠立足點**：不能宣稱「CfC 表達力可證明更高」（多條被駁回，§5）。可用的只有「與 ODE peers 同級」＋「瓶頸常在梯度（trainability）而非容量」。

---

## [SECTION 1] 核心發現：被凍結的 liquid time-constant

CfC/LTC 的表達力**整個來源**是 input- 與 state-dependent 的 per-neuron 變動時間常數（Hasani et al., Nature MI 2022；LTC AAAI 2021，verified 3-0）。

| | 官方 CfC (`raminmh/CfC/torch_cfc.py`) | 本專案原實作 (`blocks.py`) |
|---|---|---|
| 時間常數係數 | `t_a = self.time_a(x)` — **input-dependent**（backbone 函數）| `tau_a = exp(self.log_tau_a)` — **static `nn.Parameter`** |
| 時間 bias | `t_b = self.time_b(x)` — input-dependent | `t_b = self.time_b(xh)` — input-dependent |
| gate | `sigmoid(t_a·ts + t_b)` | `sigmoid(-tau_a·dt + t_b)` |

→ 本專案只保留 `t_b` 是 input-dependent，把 `t_a`（液態時間常數）降級為靜態參數。**CfC 的「液態」核心在原實作裡是關掉的** = 「沒榨乾」的最直接證據。

**已落地修法**（opt-in，Never Break Userspace）：新增 config `cfc_input_dependent_tau`（預設 `False`）。True 時以
`log τ = log_tau_a + tau_mod_scale · tanh(time_a(xh))` 恢復 input-dependent τ：
- `time_a` **zero-init** → 啟動時調制為 0、τ 等於 static 路徑（與既有實驗位元級一致，平滑啟動）。
- `tanh` 有界 → τ ∈ [τ₀·e⁻ˢ, τ₀·e⁺ˢ]，避免 `exp` 對 input-dependent 項的梯度爆炸。

驗證：`tests/test_cfc_input_dependent_tau.py`（11 passed），含「zero-init 數值相容」「time_a≠0 時輸出改變」「tanh 有界」「梯度可回傳」。

---

## [SECTION 2] CfC 容量旋鈕 + 可訓練性診斷

### 2.1 容量擴張旋鈕（官方驗證 3-0）

| 旋鈕 | 官方機制 | 本專案現況 | 對策 |
|---|---|---|---|
| backbone width/depth | `backbone_units`/`backbone_layers`，輸出被 ff/time heads 共享 | 無獨立 backbone（`ff1=ff2=Linear(combined→hidden)`）| 可加共享 backbone（CfC 內的「放大再重組」）|
| gated vs minimal | 必須 gated（reversed-sigmoid）避免梯度塌縮致容量損失 | ✅ 已用 gated | 保持，勿改 minimal |
| mixed-memory (CfC-mmRNN) | `--use_mixed` 包進 LSTM，解長序列 vanishing gradient | ❌ 未用 | §2.2 診斷協議 |
| stacking 深度 | 多層堆疊 | `num_temporal_cfc_layers=1` | 可試 2 層 + 殘差（已有層間殘差）|
| hidden width | state dim | `d_model=256` | §4 sweep |

### 2.2 可訓練性 vs 容量診斷協議（3-0）

原作者明確：CfC/LTC 長序列失效是**梯度消失，不是容量不足**。因此判斷「沒榨乾」的**第一步不是加參數**：

> 加 mixed-memory（CfC hidden 包進 LSTM-style cell）：
> - **有改善** → 瓶頸在 **trainability（梯度）**，加參數無效。
> - **無改善** → 才往容量擴張。

本專案打折點：序列長度 = sensor 時間步數、單層、bidirectional 可選，長序列梯度未必嚴重。但這正好讓「mmRNN 無改善 → 排除梯度瓶頸」成為乾淨的排除實驗。

---

## [SECTION 3] DeepONet 端：「放大表達空間、組合時縮回低秩」

### 3.1 operator rank = branch-trunk 共享瓶頸（3-0）
輸出 `G(u)(y)=Σ_{k=1}^p b_k(u)·t_k(y)`，`p` = operator rank = basis 數。已知失效：vanilla DeepONet 學到的 basis **高度線性相依、有效維度 << 名目 p**（QR-DeepONet 動機）。本專案 `operator_rank=256=d_model`，有效秩可能遠低 → 屬「容量未啟用」而非「容量不夠」。

### 3.2 Separable (PI-)DeepONet：大 p 小 r（CMAME 2024，3-0）
trunk latent 分解成 hidden dim `p` × tensor rank `r`。**實證最佳 = 大 p + 小 r；同時放大 p 與 r 會塌成 trivial solution。**
→ 啟示：擴容量放在 **hidden 寬度**，不是無腦放大 rank；rank 過大有明確退化風險。
⚠️ 此結論在 Burgers 方程得到，外推到 2D Kolmogorov 需自證（故 sweep 含 rank 上界 exp_264 監測退化）。

### 3.3 POD-DeepONet 對照（3-0）
固定 POD basis 取代學習式 trunk、branch 只學係數。可作「固定 vs 學習 trunk」ablation。
⚠️ 工程可遷移性：用 DNS 全場做 POD basis 屬**工程不可遷移**，僅能 offline 診斷，不可進訓練（見 AGENTS.md ENGINEERING_VISION）。

### 3.4 PI-DeepONet 合法性背書（3-0）
PI-DeepONet 可在**完全無 paired full-field 監督**下只靠 PDE residual 學 operator（Wang et al., Sci. Adv. 2021）→ 與本專案 sensor MSE + physics residual 設定完全相容。但有 spectral bias 高頻限制，與 K=100 Nyquist ceiling 是**兩個不同來源**，需分離歸因（§4 K-scaling）。

---

## [SECTION 4] 容量診斷 sweep 設計

**腳本**: `scripts/sweep_capacity.py`（dry-run 生成 config；`--run` 執行）。
**生成 config**: `configs/generated/exp_260~265_cap_*.toml`（base = EXP-094 B3 dns-pivot seed=2, KE 9.4%）。
固定 K=100 / DNS / seed=2，掃三軸：

| EXP | 變動 | 監測重點 |
|---|---|---|
| 260 baseline | 無（對照錨點）| 基準 KE/E(k) |
| 261 / 262 | d_model 128 / 512（width）| width plateau 點 |
| 263 / 264 | operator_rank 128 / 512（rank）| rank plateau；264 監測 separable 退化 |
| 265 | `cfc_input_dependent_tau=true`（liquid τ）| §1 修法是否改善中頻 |

### 判讀表（plateau vs 資訊上限）

| 觀察 | 結論 | 下一步 |
|---|---|---|
| 加 width/rank/liquid 後 metric 持續改善 | 容量未榨乾 | 繼續擴 |
| metric plateau 但離 K=100 Nyquist ceiling（KE 7.77%）仍有距離 | 架構/訓練瓶頸 | 回 §1/§2 |
| metric plateau 且貼近 ceiling | 已達資訊論硬上限 | K-scaling，非改架構 |

**關鍵分離實驗**：固定架構容量做 **K-scaling（K ∈ {100, 200, 400}）**。中高頻隨 K 改善 → 資訊上限；不隨 K 改善 → 架構 spectral bias。目前本專案論述把兩者混在一起，此實驗可分離歸因。

---

## [SECTION 5] 被駁回的主張與 Open Questions（誠實標註）

**被對抗驗證駁回（不可引用為依據）**：
- LTC「可證明表達力更高 / universal approximator」— 1-2、0-3 ✗
- trunk basis 的 **SVD 奇異值衰減率**當容量利用率診斷 — 0-3 ✗
- expansion-coefficient 衰減到機器精度當訓練有效性測試 — 0-3 ✗
- latent dim 越大越好 — 0-3 ✗
- 「超參數對 CfC 影響量化」(IEEE 10826128) — 0-3 ✗

→ **最想要的「容量飽和診斷」恰恰證據最弱**：文獻沒有經驗證的方法，只能靠 §4 sweep 的 plateau 經驗判斷。

**Open questions**：
1. CfC/RNN 當 branch 時，separable DeepONet 的「大 p 小 r」是否仍成立（branch 是 sequence encoder 而非靜態 MLP）？無直接證據。
2. PI-DeepONet spectral bias 與 K=100 Nyquist ceiling 如何分離 → §4 K-scaling。
3. time-constant 初始範圍、bidirectional、stacking 深度的具體推薦值，本輪無可靠 primary source（config 現用 default `(-1,1)`，註解建議 turbulence 用 `(-3,1)`，仍待自證）。

---

## [SECTION 6] 引用來源（primary，已通過驗證）

- **CfC**: Hasani et al., *Nature Machine Intelligence* 2022 — [s42256-022-00556-7](https://www.nature.com/articles/s42256-022-00556-7) / [arXiv:2106.13898](https://arxiv.org/pdf/2106.13898)；官方碼 [raminmh/CfC](https://github.com/raminmh/CfC/blob/main/torch_cfc.py)
- **LTC**: Hasani et al., AAAI 2021 — [arXiv:2006.04439](https://arxiv.org/abs/2006.04439) / [arXiv:1811.00321](https://arxiv.org/abs/1811.00321)
- **mixed-memory**: Lechner & Hasani 2020 — [arXiv:2006.04418](https://arxiv.org/abs/2006.04418)
- **POD-DeepONet**: Lu et al., CMAME 2022 — [arXiv:2111.05512](https://arxiv.org/abs/2111.05512)
- **PI-DeepONet**: Wang, Wang & Perdikaris, *Sci. Adv.* 2021 — [PMID 34586842](https://pubmed.ncbi.nlm.nih.gov/34586842/)
- **Separable PI-DeepONet（大p小r）**: Mandl et al., CMAME 2024 — [arXiv:2407.15887](https://arxiv.org/abs/2407.15887)

---

## [SECTION 7] 本次落地的程式變更

| 項目 | 檔案 | 狀態 |
|---|---|---|
| input-dependent τ opt-in flag | `src/pi_con/blocks.py` (CfCCell)、`encoders.py`、`operator.py`、`config.py` | ✅ 已實作 |
| 回歸測試（11 cases） | `tests/test_cfc_input_dependent_tau.py` | ✅ 11 passed |
| 容量 sweep generator/driver | `scripts/sweep_capacity.py` | ✅ dry-run 驗證 |
| 生成 sweep config（6 變體） | `configs/generated/exp_260~265_cap_*.toml` | ✅ load+build OK |

**新增 config keys**（已註冊於 `DEFAULT_PICON_ARGS`）：`cfc_input_dependent_tau`（預設 False）、`cfc_tau_mod_scale`（預設 2.0）。
