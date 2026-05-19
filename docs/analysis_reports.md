# 分析報告（Analysis Reports）

本檔收錄獨立分析與診斷報告，從 `docs/experiment_log.md` 拆出（2026-05-06 拆檔）。

每份報告獨立成篇，可按需讀取，不需與主實驗紀錄一起載入。

| 報告 | 日期 | 結論 |
|---|---|---|
| Wavelet Sparsity Diagnostic | 2026-04-26 | K=100 sparse reconstruction 上限為資訊論硬上限（CS 需 ~5000 sensor，差 50×） |
| AIM Diagnostic | 2026-04-26 | Zeroth-order Approximate Inertial Manifold 已證偽（quasi-static 假設違反 215×） |
| Physics Output Denormalization Silent Regression | 2026-05-06 | 進行中：denorm 改變 Kolmogorov NS residual 量級，影響 EXP-070+ 對照 |

---

## [ANALYSIS] Wavelet Sparsity Diagnostic（2026-04-26）

- **目的**：量化 Re=10000 Kolmogorov flow 在 wavelet 域的稀疏性，評估 Compressed Sensing 方法突破 band_mid/high 上限的可行性。
- **資料**：`data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy`
- **方法**：db4 2D DWT，max level（=5），Gini 係數、能量累積曲線、level-wise 分解
- **輸出圖**：`artifacts/wavelet_sparsity_diagnostic.png`

**主要數字（u 分量, t=0–5）：**

| 指標 | 值 |
|---|---|
| Gini 係數（u, v） | **0.983 / 0.985**（極度稀疏） |
| Gini 係數（ω） | 0.942（較低，高頻放大） |
| 99% 能量所需係數比例（u） | **0.5%**（約 328 個/65536） |
| 99% 能量所需係數比例（ω） | 2.9% |
| Fourier vs Wavelet 稀疏度 | **幾乎相同**（top-0.5% 係數均攜帶 ~99% 能量） |

**Level-wise 能量分佈（u, t=2.5）：**

| 頻帶 | 能量佔比 | Gini | CS 可及性（K=100） |
|---|---|---|---|
| approx k≤8 | **94.4%** | 0.50 | 196 係數，K=100 欠定 |
| level 5, k~8..16 | 4.8% | 0.57 | 588 係數，超出 K=100 |
| level 4, k~16..32 | 0.8% | 0.60 | 1452 係數，遠超 K=100 |
| level 3+, k>32 | ≈0.1% | 0.63+ | 能量可忽略 |

**結論：**

1. **稀疏性假設成立**：流場在 wavelet 域 Gini≈0.98，CS 前提條件滿足。
2. **但 K=100 仍不足**：CS 精確重建需 M ≥ O(s log N) ≈ 5000 個感測器（s≈328, N=65536）。K=100 差了約 50 倍。
3. **Fourier 與 Wavelet 稀疏度等價**：換成 wavelet 基底不改變資訊論上限的量級；真正的自由度數量與基底選擇無關。
4. **渦度更難**：ω 的稀疏度明顯低於 u/v，因 ω ∝ k² 放大高頻，渦旋結構重建從根本上更難。
5. **量化確認**：band_mid/high 的資訊論硬上限由此分析得到數量級解釋——要重建 k~8..16（4.8% 能量），需要額外 588 個 wavelet 係數的自由度，遠超 K=100 的觀測容量。


---

## [ANALYSIS] AIM Diagnostic（2026-04-26）

- **目的**：驗證 Approximate Inertial Manifold（zeroth-order）後處理能否從低頻重建結果恢復高頻分量。
- **方法**：公式 `û_k = P(N̂_k(û_{≤8})) / (νk²)`（Leray 投影），k_max_low=8，Re=10000
- **資料**：EXP-064 picon_kolmogorov_step_10000.pt，10 frame，t=0..5
- **輸出圖**：`artifacts/aim_diagnostic/aim_diagnostic_klow8.png`

**診斷結果（k_max_low=8）：**

| Band | PI-CON 原始 | AIM 修正後 | 改善 |
|------|------------|-----------|------|
| low（k≤8） | 6.4% | 6.4% | +0.0pp |
| mid（k~8..16） | 50.7% | 2985.9% | **-2935pp（嚴重惡化）** |
| high（k>16） | 14844616% | 1941356% | 改善幅度無意義 |

**根本原因分析：**

- **Quasi-static 假設違反**：AIM 公式假設 τ_visc << τ_NL。在 Re=10000，k=10 時：
  - τ_visc = 1/(νk²) ≈ 10000/(2π²k²) ≈ 5.07
  - τ_NL ≈ 1/(k·u_rms) ≈ 0.024
  - **τ_visc/τ_NL ≈ 215**（viscosity 幾乎無關緊要）
- **AIM 有效範圍**：k >> k_d ≈ 1780（dissipation scale），遠超 DNS grid k_max=128
- **實際效果**：N̂_k 被 νk² 除→把 noise 放大兩個量級，完全惡化輸出

**結論：Zeroth-order AIM 路徑已明確證偽。** 後處理思路需改用 4D-Var 或動態模式分解等不依賴 quasi-static 假設的方法。

---

## [DIAGNOSTIC] Physics Output Denormalization Silent Regression（2026-05-06）

### 摘要

`d62e698 feat(cylinder+physics): mainline overhaul` 為 cylinder 加入的
`physics_output_denormalization` 在 [`src/pi_con/training.py:178`](src/pi_con/training.py#L178)
**沒有 opt-out flag**，只要 `observed_sensor_channels=("u","v") and num_re==1`
就自動觸發。**Kolmogorov 主線完全滿足這個條件**，導致 EXP-070 之後所有
Re=10000 主線實驗都跑在「denorm 啟用」路徑下，物理 NS residual 量級被改變。

### 已驗證證據（schedule-aligned smoke：1100 步 + warmup=3000）

三方對比（artifact 路徑 `artifacts/_smoke_denorm_check_runA`、`...runB`，log `logs/_smoke/runA.log`、`runB.log`）：

| Step | t_max | EXP-064 baseline (2026-04-25, monolith, denorm 路徑不存在) | Run A (2026-05-06, denorm **ON**) | Run B (2026-05-06, denorm **OFF** via `PINN_DISABLE_PHYS_DENORM=1`) |
|---|---|---|---|---|
| **1** | 0.5 | L_phys = **3.2052e-01** | L_phys = **1.7091e-01**（縮 47%） | L_phys = **3.2052e-01** ✅ byte-identical |
| 1 | 0.5 | L_data = 2.5309e+00 | L_data = 2.5309e+00 | L_data = 2.5309e+00 |
| 1100 | 2.1 | L_phys @1000 = 3.7327e+00 | L_phys = **8.32e-01**（縮 ~5×） | L_phys = 3.42e+00（baseline ±10%） |
| 1100 | — | w_ns_u@1000 = 0.0225 | w_ns_u = **0.1015**（4.5×） | w_ns_u = 0.0229 ✅ baseline ±2% |
| 1100 | — | w_cont@1000 = 0.0142 | w_cont = **0.0277**（2×） | w_cont = 0.0143 ✅ baseline ±1% |

**結論**：

- 重構（aeb3b43）、vectorize（13af6fa）、smooth norm（55ae53a）、
  CfC tau / Fourier bands / causal weighting 等改動 **均不改變 Kolmogorov 結果**
  （Run B step 1 byte-identical EXP-064 baseline）。
- **唯一改變數值的是 d62e698 的 physics_output_denormalization**。

### 量級分析

對 Kolmogorov K=100 dataset：`u_std=0.4394, v_std=0.3321, p_std=1.0`。

denorm 把 `u_norm` 反算成 `u_phys = u_norm × 0.44 - 0.04`，套進 NS：

| NS 殘差項 | normalized 路徑（EXP-064） | denorm 路徑（EXP-070+） | 比例 |
|---|---|---|---|
| 對流 `u·∂u/∂x` | O(1) | O(0.44²) = 0.19 | 縮 5× |
| 黏性 `ν∇²u`, ν=1e-4 | O(1e-4) | O(4.4e-5) | 縮 2× |
| Forcing `A sin(k_f y)` | O(0.1) | O(0.1) | 不變 |
| Pressure `∂p/∂x`, p_std=1 | O(1) | O(1) | 不變 |

實測 step 1 `L_phys` 比例 0.171 / 0.321 = 0.534（與量級分析一致）。

### 影響範圍（時間線確認）

`d62e698` commit time = **2026-05-03 17:21 +0800**。

|  | 跑時 codebase | denorm path | baseline 是否被影響 |
|---|---|---|---|
| EXP-001 ~ EXP-064（2026-04-25 前） | monolith `picon_kolmogorov.py`，無 buffer | OFF（路徑不存在） | 否 |
| EXP-065 / EXP-066（2026-04-25, 04-26） | monolith | OFF | 否 |
| EXP-067 / EXP-068 / EXP-069（2026-04-29） | pi_con 重構後 + vectorize | OFF（buffer 還沒加） | 否 |
| **EXP-070 / 070b / 072 / 073 / 074**（2026-05-04~06，worktree `claude/vigilant-easley-516efb`） | d62e698 後 | **ON**（自動注入） | **是** |
| **EXP-070+ 的 ADR-001 §7.2 結論** | — | — | **基於 denorm 路徑下的 baseline 對照** |

### 已修補（diagnostic toggle）

- [`src/pi_con/training.py`](src/pi_con/training.py)：加 `PINN_DISABLE_PHYS_DENORM=1` 環境變數
  toggle，預設不變（denorm 仍 ON），設此 env var 才跳過 `set_physics_normalization()`，
  buffer 留在 (mean=0, std=1) ≡ identity，等價於 EXP-064 路徑。
- [`SOAP/soap.py`](SOAP/soap.py)：修 `_linalg_eigh_mps` 的 dtype/device 順序 bug
  （fp64 tensor `.to(MPS)` 直接 crash；改為先 `.to(dtype)` 再 `.to(device)`）。

### 待驗證（推論 vs 實證的分界）

**已驗證**：denorm 確實改變 Kolmogorov NS residual 量級（量化見上表）。

**未實證**：

- EXP-070 KE=84% 是否因 denorm 路徑下 `physics_loss_weight=0.057`（沿用 EXP-064 GradNorm
  收斂值，但該 weight 是 normalized 路徑下取得）量級不匹配所致。
- 還是 AL 設計本身在 K=100 sparse Re=10000 場景就不可行（與 ADR-001 §7.2 結論一致）。

**驗證計畫**：用 `PINN_DISABLE_PHYS_DENORM=1` 重跑 EXP-070 滿 10k 步：

- 若 KE 顯著改善（例如 < 20%）→ ADR-001 §7.2 結論需修訂、EXP-070~074 對照不公平
- 若 KE 仍 ~84% → AL 設計確實失敗，denorm 不是根因；但 silent regression 仍要修

### 為什麼 Cylinder 需要 denorm，但 Kolmogorov 不需要

| 場景 | u std | 黏性項 / 對流項比例 | 結果 |
|---|---|---|---|
| Cylinder Re=10000 | ~0.15 | 1500× 差距 | denorm 修一個本來就不 work 的 baseline |
| Kolmogorov Re=10000 | ~0.44 | 4300× 差距 | EXP-064 GradNorm 已自然平衡到一個 work 的點，denorm 反而打破 |

修法（Step 2，未實作）：把 env var toggle 升格為 config flag
`use_physics_denormalization: bool = False`（預設 EXP-064 行為），
cylinder configs 主動 `= true`。

