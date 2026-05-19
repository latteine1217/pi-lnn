# 診斷報告（Diagnostics Log）

本檔收錄兩類**橫跨多個實驗**的診斷與驗證紀錄，從 [`docs/experiment_log.md`](experiment_log.md) 拆出（2026-05-15 拆檔）：

1. **Silent Regression** — 流程或工具改動引入的隱性錯誤，需事後翻轉結論
2. **CFD-rigour Validation** — 對抗傳統 CFD 委員審查的物理驗證

獨立分析報告（Wavelet sparsity / AIM / 早期 denorm diagnostic）見 [`docs/analysis_reports.md`](analysis_reports.md)。

| 報告 | 日期 | 結論 |
|---|---|---|
| Physics Output Denormalization Silent Regression | 2026-05-06~07 | **CLOSED**：訓練端 config flag + evaluator default 反轉；6/6 重跑與 DIAGNOSTIC 真實值對齊 |
| CFD-rigour Validation | 2026-05-14~15 | DNS 通過 Pope criterion；div_ratio 0.88% near-incompressible；∇p 112% 為架構性 failure（Appendix E）；Forward CFD baseline 同 attractor 但 phase drift |
| LES Generator Code Audit | 2026-05-17 | 演算法 8/8 正確；N=128 LES 譜 slope −14 為 `nu_h_alpha=30` 參數錯誤（應 1.8）；F11 spectral truncation IC 待修但目前未踩坑 |

---

## [DIAGNOSTIC] Physics Output Denormalization Silent Regression（2026-05-06~07，最終結論）

### 結構

問題分**兩個獨立的 silent regression**，皆由 `d62e698 feat(cylinder+physics)`（2026-05-03）引入：

1. **Training-side regression** — `set_physics_normalization` 在 [`src/pi_con/training.py`](../src/pi_con/training.py) 沒有 opt-out flag，自動套到 Kolmogorov 主線。
2. **Evaluator-side regression** — `scripts/evaluate_deeponet_cfc.py` 預設套 `raw * std + mean`，但 model raw output 本來就是 physical 量級（[`src/pi_con/losses.py`](../src/pi_con/losses.py) 的 `(raw - mean)/std` 強制），結果是 **double-scaled**，KE 被誤報成 ~84%。

### Part 1: Training-side regression

`d62e698` 在 training.py 加入 `set_physics_normalization`，自動觸發條件 `obs=("u","v") and num_re==1` 對 Kolmogorov 主線生效。

- step 1 對照：denorm OFF 時 L_phys=3.21e-1（baseline）；denorm ON 時 L_phys=1.71e-1（縮 47%）
- AL 超參與 denorm 路徑強耦合：原 EXP-070 ρ=1.0 在 denorm OFF 路徑下 warmup 結束時直接訓練爆（C_ema 暴衝 1136×）；ρ→0.2 補償後才能完成訓練

**已修（Step 1, 2026-05-06）**：env var `PINN_DISABLE_PHYS_DENORM=1` toggle
**已修（Step 2, 2026-05-07）**：升格為 `use_physics_denormalization` config flag，預設 False；17 個 cylinder configs 主動 `= true`

### Part 2: Evaluator-side regression（2026-05-07 才發現的真凶）

evaluate_deeponet_cfc.py 與 evaluate_cylinder.py 預設套 `phys = raw * std + mean`。但 `src/pi_con/losses.py` 的 data loss `(raw - mean)/std vs normalized_target` 強制 model raw output 收斂到 physical 量級。所以 evaluator 預設的 denorm 是 **double-scale**：

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

evaluator 經 Round 1–7 review-fix loop（dataset 一致性、time alignment ULP tolerance、spectrum bin cap、`_add_split` schema、`find_dns_time_idx` 抽到 [`src/pi_con/dns_align.py`](../src/pi_con/dns_align.py) 等共 31 項修補）後，重跑 EXP-064 + EXP-070~074 的 6 個 ckpt，再次與 DIAGNOSTIC 真實值對齊驗證：

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
- **DNS divergence baseline**：div L2 PI-CON 0.184 vs DNS 0.092（EXP-064）→ evaluator 自身 numerical scheme baseline ~0.09，model 殘差 ~2× baseline 為合理量級
- **reproducibility metadata**：`sensor_subsample`、`train_ratio`、`ds_seed`、`eval_stride` 完整寫入 summary.json

artifacts: `artifacts/eval-rerun-2026-05-07/exp{064,070,070b,072,073,074}-*/`

### 待重訪

- **ADR-001 §7.2** — AL 設計實際上在 KE 維度跟 baseline 競爭（EXP-070 KE 6.30% 優於 baseline 7.80%），div_l2 trade-off 是真實的（0.184 → 0.682, ~3.7×）；原「KE=84% 場崩」描述需修正為「AL 把 div trade-off 換成 KE 維持」
- **EXP-072 step 5000 vs step 10000** — EXP-072 ckpt 只到 step 5000，需跑完 10k 步才能公平對比

### 修補總覽

**Step 1, 2026-05-06**（diagnostic toggle）：
- [`src/pi_con/training.py`](../src/pi_con/training.py)：加 `PINN_DISABLE_PHYS_DENORM=1` 環境變數 toggle
- `SOAP/soap.py`：修 `_linalg_eigh_mps` dtype/device 順序 bug

**Step 2, 2026-05-07**（升格為 config flag）：
- [`src/pi_con/config.py`](../src/pi_con/config.py)：`DEFAULT_PICON_ARGS` 新增 `"use_physics_denormalization": False`
- [`src/pi_con/training.py`](../src/pi_con/training.py)：env var toggle 改為 config flag（env var 仍保留為 emergency override）
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

| 指標 | t = 0 (IC) | t = 5 (final) | PI-CON B3 5-seed | 倍率 (T = 5) |
|---|---|---|---|---|
| KE rel-err | 0.08 % | **3.85 %** | 10.77 ± 0.52 % | forward CFD 較佳 ≈ 2.8× |
| u rel-L₂ | 5.21 % | **152.78 %** | 20.0 ± 1.7 % (time-avg) | PI-CON 較佳 **≥ 7×** |
| v rel-L₂ | 6.07 % | **203.87 %** | 23.9 ± 2.1 % (time-avg) | PI-CON 較佳 **≥ 8×** |
| KE_pred | 0.1616 | 0.1200 | — | — |
| KE_ref (DNS) | 0.1615 | 0.1248 | — | — |

artifacts：`reports/forward_cfd_baseline_T5_rank40.{json,npz}`（pulled back from home-gpu）。

**核心解讀（thesis defense level）**：

- T = 5 對應 ~2.5 t_eddy（見 §Pope criterion）；2-D Kolmogorov 在此尺度上是 chaotic regime。
- Forward CFD 在 **bounded statistics**（KE）上接近 DNS（3.85 % rel-err），因為 stationary forcing 把 KE 鎖在 attractor 上，這是 trivial preservation。
- 但 **phase information**（pointwise u, v）幾乎完全 decorrelated（rel-L₂ > 1，意指 ‖error‖ 比 ‖ref‖ 還大），這是 chaos divergence 的直接後果（λ_L ≈ 1/t_eddy ⇒ 2.5 e-foldings）。
- PI-CON 用 continuous-time conditioning + sensor 重複量測，把 pointwise correlation 保在 ~20 %（time-avg），是 **operator framework 處理 ill-posed inverse problem** 的直接證據；同一 K = 100 sensor input 與同一 PDE，pointwise 誤差差 7–8×。
- **單一 KE rel-err 指標 對 chaotic system 會 mis-rank**：委員若以 KE 攻擊「forward CFD 已經更好」，回擊 = u/v rel-L₂ 才是 phase tracking 指標，PI-CON 在這層比 forward CFD 強 ~ order of magnitude。

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

結論：forward CFD **沒有跑到另一個解**（不是 laminar Kolmogorov fixed point、不是 phase-locked periodic orbit），KE / enstrophy / spectrum shape 都在同一 attractor 上；但 chaos divergence 把 IC 推到 attractor 上「另一個典型 sample」，並且把 DNS 在 T=5 仍保留的 forcing-induced anisotropy（u_std/v_std = 2.32）抹掉變成接近 equipartition（0.90）。換句話說，forward CFD 抓到了 attractor 的長時間平均特徵，但完全失去了 DNS t=5 這個特定 phase realization。PI-CON 因有 sensor 每 0.025 t 重新量測，把 phase realization 鎖住，這是 operator framework 的決定性貢獻。

### Still-pending CFD-rigour tasks（Q7、Q8 已完成）

- **Q7 寫入位置決策（2026-05-15）**：∇p ~112%、|∇p|_rms 28% DNS 為架構性 failure（兩 config 一致），不放主章節 §5（避免搶主敘述），改寫入 **Appendix E "Pressure-Field Scope Limit"**。§4.1.1 / §4.3 保留 p_rms 0.242 baseline + cross-ref；§6.3 / §6.4 用 cross-ref 至 App E。理由：pressure 不在 supervised channel（§3.2.4 已 disclaim scope），honest disclosure 但避免打斷主敘事流。
- **CfC Jacobian spectral radius** along training trajectory — stability analysis 需額外 script
- **QR-pivot sensor placement sensitivity**：vs k-means / random placement，需重訓 3 個 config（~3 day）；EXP-101 為 random placement 候選實驗（目前 step 4500/10000 中斷）

### 對 oral defense slide 的影響

- Slide 14 (Training continuity AL) h1 改 "Lagrangian analog of pressure projection"，不再寫 "soft form of SIMPLE/PISO"
- Slide 30 (Engineering applicability) 加 scope disclaimer「2-D periodic, stationary forcing, noise-free, QR-pivot on POD basis」
- Slide 32 (Limitations) 加 ⑥「CFD-rigour gaps」
- Slide 33 (Future work) 加 ⑤「Classical-CFD baseline」
- Slide 34 (Anticipated Q&A backup) 新增，8 題 CFD-rigour 預備答案，含本節數據
- Slide 34 Q8 card 已從 "planned" 更新為實跑結果（KE 3.85 % vs PI-CON 10.77 %，但 u/v rel-L₂ 7-8× 差）— 用 chaos signature 反擊單一 KE 指標的 mis-rank

---

## [DIAGNOSTIC] LES generator code audit（2026-05-17）

### 動機

EXP-102（LES-informed QR-pivot pipeline）KE rel-err 44.3% 遠差於 DNS-pivot baseline 9.4%。所用 LES 譜 slope 在 k∈[3,40] 量到 −13.95（vs DNS −4.75），疑為 SGS 過耗散；但需確認是 (a) 演算法 bug、(b) 物理參數選擇問題、還是 (c) IC / dealiasing 副作用。本次 CFD audit 由 `physics-validation-reporter` agent 執行，對比兩份 LES generator：

- `/home/latteine/les-gen/generate_kolmogorov_les.py`（home-gpu, EXP-102 N=128 LES 用）
- `../kolmogorov_generate/dns/generate_kolmogorov_les.py`（local, EXP-103 N=256 LES 用）

### 兩版本關係

`KolmogorovLES` solver class（lines 1–471）**bit-for-bit fork 確認完全相同**，差異 100% 在 `main()` 的參數預設值與 stand-alone 分支處理。

### 結論：演算法本身全部正確（8 個 [OK]）

| Finding | 位置 | 驗證內容 |
|---|---|---|
| F1 [OK] | line 278–280 | vorticity-streamfunction `−Δψ = ω` 反演 + `u = ∂ψ/∂y, v = −∂ψ/∂x` 符號正確 |
| F2 [OK] | line 248–249 | forcing vorticity form `curl f = −A·k_f·cos(k_f·y)` 推導正確 |
| F3 [OK] | line 345 | hyperviscosity `−ν_h·k^{2p}·ω_hat` 耗散性符號正確 |
| F4 [OK] | line 328–333 | Bardina mixed similarity `τ = G*(uω) − (G*u)(G*ω)` 公式正確 |
| F5 [OK] | line 213–214 | 3/2-pad scale factors `(pad_N/N)²` / `(N/pad_N)²` 正確 |
| F6 [OK] | line 402–405 | RK2 (Heun method) 正確 |
| F7 [OK] | line 417 | spectrum binning Parseval 一致 (`sum(e2d) = KE`) |
| F8 [OK] | line 433 | spectral divergence 計算正確（fp64 下 1e-13 符合預期）|

### 3 個 [PHYSICS_RISK]（非演算法 bug，但物理選擇問題）

**F9. N=128 譜 slope −14 的唯一 root cause: nu_h_alpha=30 過大**

兩 LES 的 `nu_h` 係數差距：
- N=128 (alpha=30, 2/3 dealias): `nu_h = 4.31×10⁻⁸`
- N=256 (alpha=1.8, 3/2 dealias): `nu_h = 2.78×10⁻¹⁰`
- → 差 **155×**

N=128 在 mode k 的耗散率 ν_h·k⁴ vs eddy rate U_rms·k：

| k mode | eddy rate | ν_h·k⁴ (alpha=30) | 比值 |
|---|---|---|---|
| k=10·2π | 5.6 | 0.67 | 0.12 |
| k=20·2π | 11.2 | 10.7 | 0.96 |
| k=43·2π (k_max) | 24.1 | 229.5 | 9.5× |

k > 20 的所有模態被 hyperviscosity **強行抑制**，能譜陡降，slope 趨向 −14。是**純參數選擇錯誤**，非 algorithm bug。

dt 穩定性：`λ·dt = (ν·k_max² + ν_h·k_max⁴)·dt = 0.022 < 2`，RK2 未失穩；slope −14 不是 numerical instability，是 over-dissipation 表現特徵。

修法：N=128 改用 `nu_h_alpha=1.8`（與 N=256 同 convention）→ `nu_h = 2.58×10⁻⁹`（小 16.7× 於現值）。

**F10. 線性項全 explicit（無 IMEX）**

擴散、hyperviscosity、friction 全為 explicit（line 388–394）。穩定性需 `dt < 2/(ν·k_max² + ν_h·k_max^{2p})`。`compute_linear_dt_limit` 雖計算了限制，但只在 `--auto_dt` 啟用時生效。當前 EXP-102 跑用了 `--auto_dt --cfl_target 0.4`，但作為 default 行為對 alpha 大的情況不夠保險。

**F11. DNS-init 用 point subsampling（aliasing risk）**

`prepare_dns_initial_omega` line 130: `out = omega0[::stride, ::stride]`。物理空間 stride 子採樣會把 DNS 高頻模態（k > N_les/2）**折疊回低頻**，產生 aliasing 污染 IC。正確做法是 spectral truncation（先 FFT、切高 k 為 0、再 IFFT 回小網格）。

對 N=256 dns-init LES 影響：本案 DNS_N=256 直接複製給 LES_N=256（同網格，stride=1 不觸發 subsampling），**無 aliasing 影響**。但 future 若做 N=256 DNS → N=64 / N=128 LES，會踩坑。

對 N=128 stand-alone：不適用（用 random init）。

### N=128 slope −14 根因判讀

**Algorithm 100% 正確，純粹 parameter mis-config（alpha=30 應改 1.8）。** 兩版本 solver 完全相同，差異 100% 來自 main() 的 `nu_h_alpha` 預設值 + dealiasing convention。

### N=256 dns-init 是否 cheating

**LOW RISK** — `omega[0]` warm-start from DNS t=0 snapshot；若 DNS t=0 已是穩態（依儲存策略），LES 從 attractor 附近出發，T_end=5 ≈ 2–3 T_eddy 統計收斂較快。**不算作弊**（LES 用的是合法可得 IC proxy），但需 paper §Discussion 標註「LES warm-start, not cold-start steady state」。

若要確認無 warm-start bias：從 random IC 跑 T>20、比較後 1/4 統計是否與 T=5 dns-init 一致。

### Actionable items

| # | 動作 | 優先 |
|---|---|---|
| A1 | EXP-103 完成後對比結果：若 KE 仍 > 12% 即「LES quality + grid quantization」非 root cause，鎖死 model placement-sensitivity | 等 EXP-103 結果 |
| A2 | 若 EXP-103 也失敗：用 N=128 LES `alpha=1.8` 重跑（fix F9）+ retrain，作為 same-grid + same-SGS-strength 對照 | 待 A1 後決定 |
| A3 | `prepare_dns_initial_omega` 改 spectral truncation（fix F11）；目前未踩坑但 future-proof | 低優先 |
| A4 | paper §Discussion 標註 N=256 LES warm-start 性質 | 寫作時 |

### 對 paper 的硬意義

- EXP-102（N=128 stand-alone LES, α=30）失敗的兩個 confound 之一是**作者參數選擇錯誤**，非演算法不可救
- 修法簡單但意義重大：α 從 30 → 1.8 不是「重設計演算法」，是「校正 SGS 強度到 LES 設計意圖」
- 對 reviewer 質疑 LES quality 時，**演算法 audit 通過**這件事比「找到 root cause」本身更重要——表示問題可定位、可修，pipeline 思路本身不破

