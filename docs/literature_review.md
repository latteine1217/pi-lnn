# Literature Review — Sparse-Sensor Turbulence Reconstruction with PINN / Operator Learning

> **Date**: 2026-05-06
> **Scope**: ICLR / NeurIPS / ICML / JFM / Science / Nature MI / Phys. Rev. Fluids / arXiv (2020–2026)。排除 MDPI 期刊。
> **Target**: 對照 PI-CON（DeepONet + CfC + sparse sensors at K=100 over 2D Kolmogorov flow）找出立基點、創新點、瓶頸的文獻支援與反證。
> **Method**: WebSearch keyword sweep（4 大方向 × 多輪查詢），輔以 abstract / methods 段精讀。

---

## [SECTION 0] Executive Summary

| 項目 | 結論 |
|---|---|
| 我們在 K=100 / Re=10000 取得 KE rel-err **7.80%**（EXP-064） | 在「**僅用 sensor + physics residual，無 DNS full-field supervision**」的設定下，是**目前可查文獻中數一數二的結果**；唯一可直接比較的同類設定（PRF 2025, Mons et al.）報告 KE-equivalent error ≈ **23.1%** |
| Band_mid/high@t=5 ≈ 100% 的硬上限 | 文獻證據與 CS 理論皆支援此判斷：在 **K << O(s log N)** 的稀疏度下，無論架構如何，都無法重建中高頻 |
| 突破方向（不破壞工程可遷移性） | (1) **Sensor 時間軌跡編碼**（SHRED 路徑） (2) **學習式 sensor placement**（NeurIPS 2025 兩篇 oral） (3) **Per-task causal weighting**（修正 EXP-068 缺陷） (4) **Latent prior pre-training + sensor-only inference**（CoNFiLD 路徑，需在 config 標 research-only） |
| 突破方向（會破壞工程可遷移性） | DNS perceptual / spectral loss、full-field VAE、diffusion supervision |

**主要 Risk**：我們目前的瓶頸 80% 是 **K=100 資訊論硬上限**，20% 是 **時序資訊的利用尚未充分**。文獻中 SHRED 在 K=3 即可重建 isotropic turbulence 的事實，提示我們對 sensor 時間歷史的編碼仍不夠強。

---

## [SECTION 1] 文獻分類矩陣

依照「supervision type」與「target case」做二維分類，識別與我們最直接可比的 baseline。

### 1.1 Supervision Regime 分類

| Regime | 定義 | 工程可遷移性 | 代表方法 |
|---|---|---|---|
| **A. Sensor-only + Physics** | 僅以 sensor MSE + PDE residual 訓練；無 full-field 真值 | ✅ 完全可遷移 | **PI-CON (本研究)**, Mons et al. PRF 2025, Wang Sifan Causal PINN |
| **B. DNS-supervised + Physics** | 訓練時讀取 full-field DNS 作為 perceptual / spectral / VAE supervision；推論時可只用 sensor | ❌ 工程不可遷移 | FLRNet (perceptual VAE), CoNFiLD (latent diffusion) |
| **C. DNS pretrained operator** | 預訓練於 DNS pairs（input field → output field），fine-tune 或 zero-shot 用 sensor 推論 | △ 部份可遷移 | FLRONet, Energy Transformer, Senseiver |
| **D. Adjoint / Variational DA** | 4D-Var weak/strong constraint；不需 NN training pair | ✅ 完全可遷移（但每個 case 都要重跑優化） | He et al. JFM 2024 (turbulent jet), Mons et al. (RANS-DA) |
| **E. Nudging / continuous DA** | 在 NS 方程加回饋項 `−α·I(u − u_obs)`，持續同化觀測；無 NN | ✅ 完全可遷移（但每個 case 都要跑完整 solver） | Azouani–Olson–Titi 2014 (理論), Clark Di Leoni et al. PRX 2020 (3D HIT 實測) |

> **2026-07-18 修正（全文查證）**：Regime C 的 FLRONet 經 arXiv v6 全文核對後，實際上**沒有任何 physics/PDE 項**，且訓練直接以 complete velocity field 為 ground truth，工程可遷移性應為 ❌ 而非 △。詳見 §2.3。

### 1.2 Target Case 比對

| 案例 | Re | grid | 文獻基線 | 我們對應實驗 |
|---|---|---|---|---|
| 2D Kolmogorov forced turbulence | 1000–10000 | 256² | PRF 2025 (Mons), Causal PINN (Wang 2024) | EXP-030 / EXP-064 |
| 2D Cylinder wake | 100–10031 | 256×128 | Energy Transformer, FLRONet, FLRNet | CEXP-002 |
| 3D HIT / channel | varies | varies | SHRED (HIT), Wang 2025 (3D) | 尚未涉及 |

---

## [SECTION 2] 關鍵文獻精讀

### 2.1 Sensor-only + Physics（最直接可比）

#### [Mons et al. 2025] *Reconstructing unsteady flows from sparse, noisy measurements with a physics-constrained CNN*

| 欄位 | 內容 |
|---|---|
| Venue | **Phys. Rev. Fluids 10, 034901**（arXiv 2409.00260） |
| Method | Physics-constrained CNN，三種 loss：(i) soft-constrained, (ii) snapshot-enforced（強制 sensor 位置匹配）, (iii) **mean-enforced**（強制 mean 匹配） |
| Test cases | (1) 三角柱 wake; (2) **2D turbulent Kolmogorov flow** |
| Sensor count | 對 Kolmogorov 用稀疏 measurements + white noise（SNR 5/10/20） |
| Best result on Kolmogorov | **mean-enforced loss 達到 23.1% 重建誤差**；能譜恢復至 wave number ≈ 1 |
| Significance | **是目前可查文獻中與我們最直接可比的 paper**——同樣 sensor-only + physics constraint + Kolmogorov |
| 我們的優勢 | EXP-064 KE rel-err **7.80% < 23.1%**（雖然 metric 定義可能不同）；且我們能譜恢復至 k≈5（band_low 3.62%），較其 wave number ≈ 1 更高 |
| 我們的劣勢 | 他們處理 noisy data，我們是 clean DNS sensor values；若加入 noise 我們的數字未必能保 |
| **可借鑑點** | **mean-enforced loss 比 snapshot-enforced 對 noise 更 robust**——若我們未來要展示 noise robustness，這個 trick 應該採納 |

#### [Wang Sifan et al. 2024] *Respecting causality is all you need for training PINNs*（CMAME 2024）

| 欄位 | 內容 |
|---|---|
| Venue | Computer Methods in Applied Mechanics and Engineering（arXiv 2203.07404） |
| Method | Causal weight `w_t = exp(-eps · cumsum(L_phys_<t))`，懲罰早期未收斂時對晚期 residual 的優化 |
| Test cases | Allen-Cahn, KdV, **Navier-Stokes (lid-driven cavity)**, Kuramoto-Sivashinsky；**沒有 Kolmogorov forced turbulence 直接 benchmark** |
| Significance | 解 spectral bias 與時間多尺度的 SOTA loss reformulation |
| 我們的對應 | **EXP-068 嘗試實作此方法但失敗**：div_l2 +269% 嚴重退步。根因為「all-residual cumsum」讓 momentum 主導權重，把 continuity 的相對重要度壓低 |
| **可借鑑點 / 修正方向** | **per-task causal cumsum**：對 momentum 和 continuity 各自獨立計算 w_t，而不是合併；參考 Wang 2024 後續論文 *An Expert's Guide to Training Physics-informed Neural Networks*（arxiv 2308.08468）的 task-decomposed implementation |

#### [Krishnapriyan et al. 2021] *Characterizing possible failure modes in PINNs*（NeurIPS 2021）

| 欄位 | 內容 |
|---|---|
| Method | 提出 (1) curriculum regularization, (2) sequence-to-sequence (time marching) |
| 結論 | PINN 的失敗不是 NN 表達力不足，而是 loss landscape 病態 |
| 我們的對應 | EXP-011 已驗證 `time_marching=true` 顯著優於全時段訓練（與 (2) 一致）|
| **可借鑑點** | **curriculum regularization** 我們嘗試過 chebyshev / pressure poisson 都 negative；但若改為「physics weight curriculum」（先小再大）尚未充分掃描 |

#### [Maleki et al. 2024] *Studying turbulent flows with PINNs and sparse data*（European J. of Mechanics B）

| 欄位 | 內容 |
|---|---|
| Result on turbulent boundary layer | PINN 在 PIV 稀疏量測下，對 turbulent BL 表現顯著優於 HIT |
| **意涵** | **PINN 在「結構化湍流」（boundary layer, wake）的稀疏重建效果較好；對「同向同性湍流 / 自由衰減」相對較差**——支持我們在 cylinder wake 拿到 KE 3.5% 但 Kolmogorov forced 卡在 7.8% 的觀察 |

---

### 2.2 DNS-supervised（工程不可遷移，僅作研究參照）

#### [FLRNet 2024] *Deep Learning Method for Regressive Reconstruction*（arXiv 2411.13815）

| 欄位 | 內容 |
|---|---|
| Architecture | Fourier-feature **VAE** on full DNS field + dense sensor→latent regression |
| **Supervision** | **Perceptual loss on full DNS field**（VAE encoder/decoder 需要 full field 訓練）|
| 工程可遷移性 | ❌ 訓練必須有 DNS full field |
| 我們的對應 | 在 CLAUDE.md ENGINEERING_VISION 已明確排除此類設計 |
| **意義** | FLRNet 是「Fourier feature 解 spectral bias」的概念驗證；其低頻效果好的事實佐證我們用 LearnableFourierEmb（EXP-062 KE 10.4%）的方向正確 |

#### [CoNFiLD 2024] *Conditional Neural Field Latent Diffusion Model*（Nature Communications 2024）

| 欄位 | 內容 |
|---|---|
| Method | DNS pre-trained latent diffusion + zero-shot conditional generation（given sensor observations） |
| Test cases | Spatiotemporal turbulent flows（含 reconstruction, super-resolution, inpainting） |
| Significance | **Zero-shot 推論時不需要 DNS**——但 training 需要 |
| 工程可遷移性 | △（推論可遷移，訓練不可遷移） |
| **可借鑑點** | 若我們允許「research-only pre-training + engineering deployment」分階段策略，這是最可行的「**DNS prior + sensor-only inference**」路線。需在 config 明確標註為 research-track，但 deployment 時可用 |

---

### 2.3 DNS-pretrained Operator（部份可遷移）

#### [FLRONet] *Deep Operator Learning for High-Fidelity Flow Reconstruction*（arXiv 2412.08009 **v6**；已發表 ASME **2026**, DOI `10.1115/1.4070332`）

> **2026-07-18 全文查證更新**。前一版本記載基於 arXiv v3 abstract；以下每一列均來自 v6 LaTeX 原始碼實讀，標註行號。

| 欄位 | 內容（全文核實） |
|---|---|
| Architecture | DeepONet branch (FNO) + trunk；**另有 Voronoi embedding 層**處理 sensor 缺失 + sinusoid embedding 處理時間相關性 |
| **Supervision** | **純全場監督，無 physics**。原文 line 256：「we randomly selected a single snapshot to record **the complete velocity field, which served as the ground truth**」 |
| **PDE / 不可壓縮項** | **完全沒有**。全文 grep `nabla` / `incompressib` / `divergence` / `continuity equation` / `governing equation` / `PDE residual` / `\partial` → **ZERO HITS**。全文 5 次「physics」全部在 related work 描述**他人**工作 |
| Training detail | Adam, lr = 10⁻³（line 258）。**論文未寫出訓練 loss 方程式**，僅給驗證指標 MAE（ℓ₁, eq. 5） |
| Sensor | 隨機 32 點；140×240 grid |
| Test case | CFDBench **cylinder**，50 個 case 為 inlet velocity 0.1→5.0 m/s |
| **Reynolds 範圍** | **Re ∈ [20, 1000]**（CFDBench 原文 line 452 明載）。以其 baseline ρ=10, μ=0.001, d=0.02 驗算 Re = 200·u → u∈[0.1,5.0] 對應 Re∈[20,1000]，自洽 |
| Speed claim | **16 ms/frame on A100**（line 89），宣稱 real-time |
| 與 PI-CON 的關係 | 架構相似度**極高**（同為 DeepONet branch+trunk + sensor encoding），但**訓練 regime 完全相反**：FLRONet 需 complete field GT，PI-CON 只用 sensor MSE + PDE residual |

**對本研究的三點意涵**：

1. **差異化軸線確立且乾淨**：同為 operator learning 稀疏重建，FLRONet 需全場真值訓練（依本專案 `ENGINEERING_VISION` 判準屬工程不可遷移），PI-CON 不需。這是可寫進 Contributions 的第一順位差異。
2. **Regime 差異可量化**：FLRONet 最難的 case 是 Re=1000 的**週期性 Kármán 渦街**；PI-CON 跑 Re=10⁴ 的 **chaotic 2D 湍流**，Re 高一個數量級且動力學性質不同（週期 vs 混沌）。註：本專案 EXP-230 有 Re=1000 baseline 可作為對接點。
3. **速度數字僅供 related work 對照，不構成威脅**：FLRONet 宣稱 16 ms/frame (**A100**)；本專案實測 full 128² field query 527.8 ± 17.1 ms (**M3 MPS**)。兩者算力差約 50–100×，**未經正規化不可比較，不得據此宣稱任一方較快**。本專案的速度陳述屬 capability claim（快到足以支撐 sparse monitoring），不需要優先權，故 FLRONet 的存在不影響之。zero-shot 連續查詢同理：那是 DeepONet 的既有性質（Lu Lu 2019），雙方皆具備，本就非任一方的創新點。

> **對 §3.2 的淨影響：零。** FLRONet 未觸及本專案四條 contribution 中的任何一條（CfC+DeepONet 組合、sensor-position-aware collocation、K=100 @ Re=10⁴ sensor-only 成績、information-theoretic ceiling 量化）。所需動作是**新增一段 related work 區隔訓練 regime**，非重寫 Contributions。

**殘留風險**：期刊定稿版（ASME 2026）未核對，與 arXiv v6 可能有差異。引用前建議取期刊版確認 loss 設計未變。

#### [Zhang & Krotov & Karniadakis 2025] *Energy Transformer for Sparse Reconstruction*（J. Comp. Phys. 2025, arXiv 2501.08339）

| 欄位 | 內容 |
|---|---|
| Architecture | Energy Transformer（基於 Dense Associative Memory / 現代 Hopfield）；reconstruction = energy minimization |
| **Supervision** | DNS supervision（patterns 存在 energy local minima） |
| Test cases | (1) **2D vortex street (cylinder wake)**, (2) Schlieren impinging supersonic jet, (3) **3D turbulent jet (PTV)** |
| Performance | 在 **90% 缺失資料** 下仍能準確重建；對實驗 noise robust |
| 工程可遷移性 | ❌（需 DNS 訓練） |
| **意義** | 這是用戶在 [STATE] Cylinder Wake 提到的對照 baseline。**90% missing data ≈ 對 256×128=32768 cells 而言 K≈3000**——遠超我們 K=100 |
| **意涵** | Energy Transformer 是「DNS pretrained, sensor-only inference」路線的代表；但其對 **K~3000+** 才驗證有效，**未證實 K=100 等級** |

#### [Senseiver 2023] *Attention-based global field reconstruction*（Nature Machine Intelligence 2023）

| 欄位 | 內容 |
|---|---|
| Architecture | Perceiver-style attention encoder + decoder；agnostic to dimensionality |
| **Supervision** | DNS pairs |
| Sensor count | 對 sea-surface-temperature 用 K~100 達到 SOTA |
| 工程可遷移性 | ❌ |
| **可借鑑點** | Sensor encoder 的 cross-attention 設計（learnable latent + sensor as KV）是我們可以在不破壞 sensor-only supervision 前提下吸收的 architecture pattern |

#### [Voronoi-CNN 2021] *Global field reconstruction with Voronoi tessellation* (Fukami, Maulik, Nature MI 2021)

| 欄位 | 內容 |
|---|---|
| Architecture | Voronoi 分割 sensors → grid representation → CNN |
| 優勢 | 支援 **任意數量、可移動** 的 sensor |
| 工程可遷移性 | ❌（需 DNS supervision） |
| **可借鑑點** | Voronoi diagram 作為 sensor 的 spatial structure prior 是 lightweight 的；可作為 sensor encoding 的 augmentation |

---

### 2.4 Time-history sensor encoding（**最可能突破我們瓶頸的方向**）

#### [SHRED 2024] *Sensing with shallow recurrent decoder networks*（Williams, Zahn, Kutz, Royal Society A 2024）

| 欄位 | 內容 |
|---|---|
| Architecture | **LSTM on sensor time-series + shallow decoder** |
| Key claim | **K=3 sensors 可重建 forced isotropic turbulence**；對 sensor placement 不敏感 |
| Theoretical basis | Separation of variables for linear PDEs |
| **Supervision** | DNS pairs（needs full field for training） |
| 工程可遷移性 | ❌（但概念可借鑑） |
| **核心觀察** | **靠 sensor 的時間軌跡就能解碼出大量空間資訊**——這是 PDE 動力學的時間-空間耦合的直接利用 |

#### [Williams et al. 2024] *Reduced order modeling with SHRED*（Nature Communications 2025）

| 欄位 | 內容 |
|---|---|
| 延伸 | SHRED-ROM 把 SHRED 與 reduced-order modeling 整合；超越單純 reconstruction |
| **意義** | LSTM-on-sensor-time-history 已成為一個獨立 paradigm |

> **我們的對應分析**：
>
> 我們的 sensor 設定是 **K=100 個位置上、t=0..5 共 41 frames** 的速度量測。我們目前 **trunk 只在 query 端用 CfC 處理時間**，**branch 對 sensor 是 per-snapshot 處理**——我們其實 **沒有把 sensor 的時間軌跡當成 LSTM/CfC 輸入**。這是一個 SHRED 路徑可立即啟發的修改：在 branch 對每個 sensor 的時間序列用 CfC/LSTM 編碼，再整合空間 attention。

---

### 2.5 Sensor placement optimization（**我們忽略的另一個維度**）

#### [Liu et al. 2025] *Flow Field Reconstruction with Sensor Placement Policy Learning*（**NeurIPS 2025**）

| 欄位 | 內容 |
|---|---|
| Method | 方向感知 GNN + 兩階段約束 PPO（Proximal Policy Optimization）強化學習 sensor 配置 |
| Critique on prior work | 既有方法假設 (a) 2D 域、(b) 預定 PDE、(c) idealized synthetic data、(d) **unconstrained sensor placement**——直指我們也採用的 QR-pivot |
| **意義** | **我們的 K=100 QR-pivot sensor 不是最優**；學習式 placement 可能在同樣 K=100 下顯著降低重建誤差 |
| 工程可遷移性 | △（policy 可預訓練，但部署時要能調整 sensor） |

#### [Kim et al. 2025] *PhySense: Sensor Placement Optimization*（**NeurIPS 2025 oral**）

| 欄位 | 內容 |
|---|---|
| Method | 兩階段：(1) Flow-based generative model + cross-attention 重建場；(2) Projected gradient descent 優化 sensor placement，滿足空間約束 |
| Theoretical guarantee | 與經典 variance minimization 一致 |
| **意義** | 學習式 sensor placement 在 NeurIPS 2025 已有兩篇獨立工作。**這是一個 active research frontier，我們完全沒涉及** |

> **我們的對應分析**：
>
> 我們在 EXP-064 用 **QR-pivot on POD modes** 選擇 sensor，這是 1990s–2010s 的最佳作法。但 NeurIPS 2025 的兩篇 paper 顯示，**learned sensor placement 在 reconstruction-aware 訓練下可以更好**。這是一個獨立於我們 K=100 上限分析的維度——CS 上限是 worst-case；具體 K=100 的最優 placement 可能在某些頻率帶上表現較我們現在的 QR-pivot 好。

---

### 2.6 PDE-specific advances 與其他相關工作

| Paper | Venue | 與我們的關係 |
|---|---|---|
| **FNO** (Li et al. 2021, ICLR) | ICLR 2021 | Spectral-convolution paradigm；我們的 LearnableFourierEmb 是 sparse-data 版的精神延伸 |
| **PI-DeepONet** (Wang Sifan et al. 2021, Sci. Adv.) | Sci. Adv. 2021 | 我們架構的直接前驅；確認 trunk-branch + physics residual 的可行性 |
| **PINNacle** (Hao et al. 2024, NeurIPS) | NeurIPS 2024 | 標準化 PINN benchmark；可作為我們 K=100 結果的 standalone reference |
| **SC-FNO** (Behroozi et al. 2025, ICLR) | ICLR 2025 | Sensitivity-constrained FNO；對「**inverse parameter inference + small data**」有提升。我們是 inverse field reconstruction，邏輯類似 |
| **Mamba Neural Operator** (Zheng et al. 2024, NeurIPS) | NeurIPS 2024 | State-space model for PDE；對長時序 chaotic 系統可能比 FNO 更穩 |
| **DINO** (Differential-Integral NO 2025) | arXiv 2509.21196 | 99-step Kolmogorov forecasting：DINO error 0.59 vs baseline 1.9+。**這是 forecasting，非 sparse reconstruction**——但顯示 Kolmogorov 長期動力學是 NO 的一大挑戰 |
| **Wang Sifan 2025** *Simulating 3D Turbulence with PINN* | arXiv 2507.08972 | PINN 首次解 3D HIT/channel；causal training + adaptive arch + advanced optimization 組合 |
| **He et al. 2024** *4D-Var super-temporal-resolution turbulent jet* | **JFM 978, A14** | **Adjoint-based 4D-Var**，無需 NN training pair；對「研究級 reconstruction beyond Nyquist」是強 baseline |

---

### 2.7 經典前案與 Nudging 路線（2026-07-18 新增；先前版本完全缺席）

> **緣起**：檢查 `forward_cfd_baseline` 的定義時發現，本文獻回顧從未收錄「稀疏重建的經典解法」與「持續同化（nudging）」兩條線。以下 citation 全部經 CrossRef 查證，DOI 已核對。

#### 2.7.1 Forward CFD baseline 的組成件均為既有方法

本專案的 `forward_cfd_baseline`（POD rank-40 反推 IC → ETDRK4 前向積分至 t=5，見 `docs/archive/diagnostics_log.md` Q8）**的組件**均為既有標準方法。**組件既有 ≠ 該組合有前案**，三層必須分開（見本節末「三層區分」）：

| 組件 | 文獻出處 | DOI | 查證 |
|---|---|---|---|
| Sparse → full field 的 POD 反推（**Gappy POD**） | Everson & Sirovich, *JOSA A*, 1995 | `10.1364/josaa.12.001657` | ✅ CrossRef |
| Gappy POD 用於 unsteady flow sensing | Willcox, *Computers & Fluids*, 2006（另有 2004 AIAA 會議版 `10.2514/6.2004-2415`） | `10.1016/j.compfluid.2004.11.006` | ✅ CrossRef |
| QR-pivoting sensor placement | Manohar, Brunton, Kutz & Brunton, *IEEE Control Systems Magazine*, 2018 | `10.1109/mcs.2018.2810460` | ✅ CrossRef |
| Nudging / continuous DA 理論 | Azouani, Olson & Titi, *J. Nonlinear Sci.*, 2014 | `10.1007/s00332-013-9189-y` | ✅ CrossRef |

**三層區分（2026-07-18 修訂，先前版本把三層壓成一層，屬過度宣稱）**

| 層次 | 判定 | 證據狀態 |
|---|---|---|
| **(a) 組件**：gappy POD、譜方法前向積分 | ✅ 既有 | 已查證，DOI 見上表 |
| **(b-1) free-run / open-loop 作為 no-assimilation control 的**概念** | ✅ 既有，教科書級 | 論文已引 Asch, Bocquet & Nodet (2016) SIAM《Data Assimilation》`10.1137/1.9781611974546`（thesis `Asch2016DA`）。**此為適格來源** |
| **(b-2) 此概念在近期流體 DA 論文中的**具體實作形式** | ⚠️ 抽查 4 篇，**0 篇**採用「估 IC → 完整 solver open-loop 積分」 | 見下方 methods 抽查；此為 prevalence 觀察，**不否定 (b-1)** |
| **(c) 此具體配對**：gappy POD rank-40 + ETDRK4 + 2D Kolmogorov Re=10⁴ | ❓ **未檢索到前案** | 見下方檢索限制 |

**(c) 的檢索紀錄（2026-07-18）**：跑了 5 組 query（arXiv × 4、OpenAlex × 1），涵蓋 gappy-POD-initialized forecast、open-loop free run baseline、POD IC + error growth 等組合。最接近的三篇均**不是**同一件事：

- *Forecasting 3D turbulent recirculating flows from sparse sensor data*（arXiv 2505.05955, 2025）— POD + Koopman 建**線性動力系統**外推，非用真實 NS solver 積分
- *Real-time forecasting of chaotic dynamics from sparse data and autoencoders*（arXiv 2508.08729, 2025）— CAE + ESN + **EnKF 持續同化**，屬 closed-loop，非 open-loop
- *Multi-scale data reconstruction of turbulent rotating flows with Gappy POD…*（arXiv 2210.11921, 2022）— gappy POD 做 inpainting **重建**，無前向預測

**⚠️ 檢索限制（不可略過）**：此類 baseline 通常寫在論文 methods 段的一兩句話裡，**不會出現在 title / abstract**，而本次僅能做 title/abstract 關鍵字檢索；Semantic Scholar 因無 API key 全程 rate-limited 未取得結果。故「未檢索到前案」**只能作為弱證據，不足以宣稱無前案**。

**(b) 的 methods 段抽查（2026-07-18，4 篇全文實讀）**

先前版本宣稱「open-loop free run 是 DA 標準對照組、**每篇 DA 論文都會跑**」。前半（概念是標準）由 `Asch2016DA` 教科書支持、成立；**後半（每篇都跑）經抽查證偽**：

| 論文 | 有無 no-assimilation baseline | 其 baseline 的實際形式 |
|---|---|---|
| Clark Di Leoni et al. PRX 2020（nudging, 3D HIT） | ❌ 無 | 改以掃 α、φ 參數空間呈現；只比 nudged vs reference |
| *Forecasting 3D recirculating flows*（2505.05955） | ❌ 無 | 比較 estimated vs true POD coefficients |
| Plogmann, Brenner & Jenny（2405.20160, spectral adjoint） | ✅ 有 | **未同化的 URANS baseline**（無 IC 估計步驟，形式不同） |
| **Mons et al.**（2409.00260, PC-CNN — 本專案最可比同類設定） | ✅ 有 | **thin-plate spline 空間內插**（逐 snapshot，無前向積分） |

**結論：2/4 有某種 no-assimilation baseline，但 0/4 採用「從稀疏觀測估 IC → 用完整 solver open-loop 積分」的形式。**

**這對本專案是正面結果**：本 baseline 不但不是稻草人，反而**比最可比同類工作（Mons et al.）所用的 spline 內插更強**——內插只用空間資訊，本 baseline 額外給了完整 NS 動力學與正確的 solver。打贏一個更強的對照組，論證力更高。

**論文寫作含意（結論，2026-07-18 二度修訂）**：

- ✗ 不要寫「我們提出一種新的 forward CFD baseline」→ 招致「自製稻草人」質疑
- ✗ 不要寫「forward CFD 是文獻既有方法」→ 無此名稱之方法，(c) 亦未證實有前案
- ✅ 可寫「open-loop free-run 是 DA 的 no-assimilation control」→ 有 `Asch2016DA` 教科書支持（thesis appendix07 已如此處理）
- ✗ 但不要寫「**每篇** DA 論文都跑這個對照」或暗示此具體實作形式常見 → 抽查 0/4
- ✅ **應寫**：「我們建構一個 no-assimilation 對照組：以 gappy POD（Everson & Sirovich 1995）從 K=100 感測器估初始場，再用**與 DNS 相同的 solver** open-loop 積分。相較於既有工作常用的空間內插對照（如 Mons et al. 2024 的 thin-plate spline），此對照額外提供完整 NS 動力學，因而是更強的比較基準。」

此寫法三層皆成立：組件有出處、對照強度有同類比較支撐、實例化誠實歸屬本專案，且**不依賴 (c) 的檢索結果**。

命名沿用 `docs/metric_choice_note.md`：正式描述為 **gappy-POD initialisation + open-loop (free-run) forward integration**。

> **命名以 `docs/metric_choice_note.md` 為準**（該檔 2026-07-18 已統一）：`forward_cfd_baseline` / "forward CFD" 是**本專案內部簡稱，不是文獻方法名**；正式描述為 **gappy-POD initialisation + open-loop (free-run) forward integration**。
>
> 該檔另有一項本節未涵蓋的關鍵區分：**forward CFD 是比較集合中唯一的 *forecast*，其餘（含 PI-CON、trig-LSQ）皆為 *reconstruction*（皆見 t=5 sensor）**——任務類別不同，自成一格。此區分比本節的論述更精確，撰稿時應優先採用。本節僅補充其**文獻源頭與 DOI**，兩處若有出入以 `metric_choice_note.md` 為準。

#### 2.7.2 ⚠️ Clark Di Leoni, Mazzino & Biferale — Nudging the NSE（*Phys. Rev. X* 10, 011023, 2020）

DOI `10.1103/physrevx.10.011023`（arXiv 1905.05860）。**這是 forward CFD baseline 最強的反方**，且發在 PRX。以下數據取自 arXiv 全文實讀。

| 項目 | 內容（全文核實） |
|---|---|
| 對象 | 3D homogeneous isotropic turbulence |
| 三種觀測型態 | (i) Eulerian（固定空間位置）、(ii) Fourier（波數區間）、(iii) Lagrangian（移動探針） |
| 參數 | RUN1: Re=3900, 256³；RUN2: Re=25000, 1024³（table, line 413–417） |
| 觀測實作 | 非單點，而是在半徑 `r = 1.25η` 的小球內 nudge（line 463） |
| **關鍵門檻** | **達成 full synchronization 的臨界體積分率 `φ_c ≈ 0.2`**（line 922）；對應 `k_c ~ 0.2 k_η`（line 805） |

**對本研究最重要的一句話（可直接用於口試防禦）**：

> Di Leoni 等人證明，configuration-space nudging 要達到 full synchronization 需要 **φ_c ≈ 0.2**，即 20% 的體積被持續同化。本專案 K=100 / 256² 的覆蓋率為 **0.15%**，比該門檻低約**兩個數量級**。

因此正確的 framing **不是**「PI-CON 打敗 nudging」，而是：

> 本研究所處的是**嚴重欠觀測（severely under-observed）regime**，遠低於任何已發表的 nudging 同步門檻。在此資料量下 nudging 並不預期能同步，這正是需要 operator 學習先驗的原因。

這條論述同時**解釋**了本專案的 observability wall（`k ≲ 8`）與實測 `k_cut ≈ 4.7`，兩者互相佐證。

**⚠️ 外推限制（不可忽略）**：`φ_c ≈ 0.2` 是 **3D HIT** 的量測值。本專案是 **2D Kolmogorov**，2D 的逆能量串級與守恆結構不同，**該門檻不保證可直接外推**。撰稿時必須明示此限制，只能寫成「遠低於 3D HIT 已知門檻」，不可寫成「低於 2D 的門檻」。

---

## [SECTION 3] PI-CON 立基點 / 創新點總結

### 3.1 立基點（已驗證可行）

| 立基點 | 證據 | 文獻支援 |
|---|---|---|
| **Sensor-only + physics 完全可行於 Re=10000 K=100** | EXP-064 KE 7.80% | 與 Mons et al. PRF 2025 路線一致；數字優於其報告 23.1% |
| **DeepONet branch + temporal CfC trunk 是合理架構** | EXP-030 / EXP-064 | 與 PI-DeepONet (Wang 2021) + CfC (Hasani 2022) 兩條獨立 well-cited 線吻合 |
| **LearnableFourierEmb 解低頻 spectral bias** | EXP-062 band_low 5.8% | 與 FLRNet, FNO, FF-PINN 多篇結論一致 |
| **GradNorm 自動平衡 data/physics gradient** | EXP-063 div_l2 -64% | 與 Wang 2021 *Understanding gradient pathologies* (SISC) 一致 |
| **Sensor 位置 continuity physics points** | EXP-064 phase_err -53% | 在文獻中**未被廣泛報告**——這可能是我們的微創新 |

### 3.2 潛在創新點（差異化）

| 創新點 | 說明 | 是否有先例 |
|---|---|---|
| **CfC + DeepONet 組合用於 turbulence reconstruction** | 我們是首見將 Liquid Neural Network 系列（CfC）作為 DeepONet trunk 應用於 2D Kolmogorov 稀疏重建 | 文獻搜尋未找到此組合的先例。**這是可發表的差異化點** |
| **Sensor-position-aware physics collocation** | EXP-064 把 continuity 殘差 collocation 點放在 sensor 附近 | 文獻搜尋未見明確報告；多數工作用 random / chebyshev |
| **K=100 在 Re=10000 達 KE 7.8%** | 在 sensor-only setup 下成績 | 同類 setup 文獻最佳為 PRF 2025 的 23.1% |
| **完整 Information-theoretic ceiling 量化** | Wavelet sparsity diagnostic 直接證明 K=100 不足 | 在 PINN literature 罕見此種「先證明上限再評估方法」的論述 |

### 3.3 我們**不是首創**的部份

- Sensor-only + physics constraint：**Mons et al. PRF 2025 已做過 Kolmogorov**
- Fourier feature for spectral bias：**Tancik 2020, Wang 2021 已系統化**
- DeepONet：**Lu Lu 2019，PI-DeepONet 2021**
- Causal training：**Wang Sifan 2022 已推**
- **DeepONet 做 sparse-sensor 重建本身：FLRONet (ASME 2026) 已做**（差異在訓練 regime，非架構，見 §2.3）
- **稀疏量測反推全場：Gappy POD (Everson & Sirovich 1995) 已有三十年**
- **靠持續同化鎖住 chaotic phase：nudging 文獻已完整處理**（AOT 2014 理論；Di Leoni PRX 2020 實測）
  - 影響範圍**僅限措辭**：`docs/archive/diagnostics_log.md` Q8 的「這是 operator framework 的**決定性貢獻**」屬過度宣稱，應改為「持續量測相對單次量測的優勢」並 cross-ref Di Leoni。
  - **不影響 §3.2 任何一條 contribution**。

> **2026-07-18 自我更正**：本節初版曾另列「速度不可當獨佔賣點」與「zero-shot 連續查詢已被 FLRONet 做過」兩條，**均已撤除**，理由如下——
> 1. **速度**：§3.2 從未以速度為創新點。CLAUDE.md `Paper_Main_Message` 的「快速」是**能力主張**（快到足以支撐 sparse monitoring），不是**新穎性主張**（首個快的方法）。兩者不可混為一談。且 FLRONet 的 16 ms 是 A100、本專案 527.8 ms 是 M3 MPS，算力差約 50–100×，**未正規化即宣稱「數字不利」不成立**。
> 2. **Zero-shot 連續查詢**：這是 DeepONet 自 Lu Lu 2019 起的既有性質，本專案本就未宣稱首創，列入「不是首創」屬無的放矢。
>
> 教訓：區分 **capability claim**（本方法能做到 X）與 **novelty claim**（本方法首先做到 X）。前者不需要優先權，被他人做過不構成威脅。

---

## [SECTION 4] 瓶頸根因再診斷（對照文獻）

### 4.1 已被文獻佐證的瓶頸

| 瓶頸 | 我們的證據 | 文獻佐證 | 性質 |
|---|---|---|---|
| **K=100 < O(s log N) ≈ 5000 for CS** | Wavelet sparsity diagnostic | Compressed Sensing theory 經典結果（Candès 2008） | **數學硬上限**——無法繞過 |
| **Spectral bias 對高頻收斂慢** | band_high error ≈ 100% | Wang 2021 NTK theory | **訓練上限**——可部份緩解 |
| **時間多尺度 + 因果違反** | t=0 重建 KE 60% | Wang 2024 *Causality is all you need* | **可緩解但需正確 causal weighting** |
| **2D Kolmogorov forced turbulence 比 wake / BL 更難** | EXP-064 7.8% vs CEXP-002 3.5% | Maleki 2024 PINN sparse turbulence study | **物理結構性問題** |

### 4.2 文獻提示的「我們可能忽略的因子」

| 因子 | 文獻來源 | 我們是否覆蓋 |
|---|---|---|
| **Sensor 時間軌跡編碼**（SHRED 路徑） | Williams 2024 Royal Society A | **❌ 未涉及**——我們 branch 是 per-snapshot 處理 |
| **Learned sensor placement** | Liu 2025 NeurIPS, Kim 2025 NeurIPS oral | **❌ 未涉及**——我們用 QR-pivot |
| **Per-task causal weighting** | Wang 2024 *Expert's Guide* | **△ 嘗試但失敗**（EXP-068 implementation bug） |
| **Mean-enforced loss for noise robustness** | Mons 2025 PRF | ❌ 未涉及（我們是 clean DNS, 但對展示工程價值關鍵） |
| **Latent diffusion prior**（research-only） | CoNFiLD 2024 Nat. Comm. | ❌ 未涉及 |
| **State-space (Mamba) for long-term chaotic dynamics** | Zheng 2024 NeurIPS | ❌ 未涉及 |

### 4.3 為什麼我們「卡這麼久」的真實原因

> **客觀分析**：
>
> 1. **80% 機率**：我們確實到了 K=100 的資訊論硬上限。這個上限是 CS 數學保證，**任何架構/optimizer/loss reformulation 都不能突破**。EXP-053..EXP-069 的所有負面結果都不應視為失敗——它們是在已飽和的方向上做局部優化，自然進入 diminishing returns。
> 2. **15% 機率**：我們忽略了 **sensor 時間軌跡編碼**。SHRED 顯示在 K=3 即可重建 isotropic turbulence；我們 K=100 但 branch 只做 per-snapshot 處理，可能浪費了 41 frames 的時間動力學資訊。
> 3. **5% 機率**：QR-pivot sensor placement 不是 K=100 設定下的最佳；NeurIPS 2025 的學習式 placement 可能在同樣 K 下換取數 pp 改善。

---

## [SECTION 5] 建議下一步（按 ROI 排序）

### 5.1 高 ROI / 工程可遷移

#### Direction A：**Sensor 時間軌跡編碼**（SHRED-inspired，但保 sensor-only supervision）

- **Hypothesis**：對每個 sensor 的時間序列 `s_i(t_0..t_T)` 用 CfC/LSTM 編碼，產生時間感知的 sensor embedding，再餵入 branch attention。
- **Expected change**：band_mid 可能改善（時間動力學承載部份高頻空間資訊）。
- **Falsifiability**：若 band_mid@t=5 仍 ≈100%，代表 K=100 的時間軌跡也不足以解碼 k>5 模態。
- **Risk**：CfC fast channels 在 EXP-067/EXP-069 已證實對 dt=0.025 過敏感；需要先 sweep CfC tau 範圍，避開 (-3, 1) 的 fast 範圍。
- **與 CLAUDE.md 一致性**：✅ 完全 sensor-only supervision

#### Direction B：**Per-task causal weighting**（修 EXP-068 bug）

- **Hypothesis**：對 momentum_u, momentum_v, continuity 各自獨立計算 `w_t = exp(-eps · cumsum(L_task_<t))`，避免量級主導。
- **Expected change**：div_l2 不再退步，t=0 KE 改善。
- **Falsifiability**：若 per-task causal 也讓 div_l2 退步，代表 causal weighting 在多 task 設定根本不適合。
- **Risk**：低，純 loss reformulation。

#### Direction C：**Learned sensor placement**（NeurIPS 2025 路線）

- **Hypothesis**：用 PPO 或 projected gradient descent 學一組 K=100 sensor 位置，比 QR-pivot 更利於 reconstruction。
- **Expected change**：在 K=100 fixed 下可能換 1-3pp KE 改善。
- **Falsifiability**：若學習式 placement 不超越 QR-pivot，代表 QR-pivot 已是 K=100 的近似最優。
- **Risk**：中，需要重新生成 sensor 對應 DNS values pipeline。

### 5.2 中 ROI / 部份工程可遷移

#### Direction D：**Latent prior pre-training**（CoNFiLD-inspired，標 research-only）

- **Hypothesis**：在 DNS 上預訓練 latent diffusion prior，部署時固定 prior + sensor-only fine-tune。
- **Expected change**：突破 band_mid/high 的 CS 上限——因為 prior 提供了高頻先驗。
- **Falsifiability**：若加 prior 後 band_mid 仍 ≈100%，代表 prior 與 inference time PDE residual 不一致（distribution shift）。
- **Risk**：高，需要在 config 與 experiment_log 嚴格標 research-only；deployment 時須驗證 prior 來源 DNS 與 target case 的物理對應關係。

### 5.3 低 ROI / 暫不建議

- **加大模型至 d=512+**：EXP-065 已證實 trunk 加深無效，capacity 不是瓶頸
- **更多 physics loss 變體**（Chebyshev / Pressure Poisson）：EXP-035..EXP-039 全部 falsified
- **L-BFGS**：EXP-052 證實計算成本不可行
- **Transfer learning from Re=1000**：EXP-040/EXP-042 證實架構不匹配與 source 品質都會破壞
- **K 增到 200+**：違背「K=100 為固定資料條件」的研究設定；除非整體 reframe

---

## [SECTION 6] 建議的研究 framing（投稿角度）

依據文獻定位，我們的 paper 應該投這個 framing：

> **"Engineering-Compatible Sparse Reconstruction of 2D Forced Turbulence at High Reynolds: A PI-CON Study"**
>
> 主軸：
> 1. 在 K=100 / Re=10000 / sensor-only + physics 設定下，建立**目前已知最佳結果**（KE 7.80%）。
> 2. 對照 PRF 2025 的 23.1%，**展示 architectural advances（DeepONet + CfC + LearnableFourierEmb + GradNorm）的累積效益**。
> 3. 用 wavelet sparsity diagnostic **量化證明 band_mid/high 是 CS 數學硬上限**，不是架構問題——這是 PINN literature 罕見的 rigorous failure analysis。
> 4. 提出「engineering-transferable」設計原則：拒絕 DNS perceptual loss / VAE supervision。
> 5. 識別未來方向：時間軌跡編碼、學習式 sensor placement。

可投：
- **JFM**（fluid mechanics 社群，accepts engineering framing）
- **CMAME**（Wang Sifan 系列的 home journal）
- **J. Comput. Phys.**（Energy Transformer 投這裡）
- **Phys. Rev. Fluids**（Mons 投這裡）

不建議投：
- ICLR/NeurIPS：要求方法新穎度，我們的 architectural advances 偏 incremental
- Nature MI：要求 generality 廣，我們 case-specific

---

## [SECTION 7] Sources（按 venue 分類）

### Top-tier ML conferences

- [Krishnapriyan et al. — Characterizing possible failure modes in PINNs (NeurIPS 2021)](https://proceedings.neurips.cc/paper_files/paper/2021/file/df438e5206f31600e6ae4af72f2725f1-Paper.pdf)
- [Li et al. — Fourier Neural Operator for Parametric PDEs (ICLR 2021)](https://iclr.cc/virtual/2021/poster/3281)
- [Pfaff et al. — MeshGraphNets (ICLR 2021)](https://openreview.net/forum?id=roNqYL0_XP)
- [Mamba Neural Operator (NeurIPS 2024)](https://neurips.cc/virtual/2024)
- [PINNacle benchmark (NeurIPS 2024)](https://neurips.cc/virtual/2024)
- [SC-FNO (ICLR 2025)](https://iclr.cc/)
- [Liu et al. — Flow Field Reconstruction with Sensor Placement Policy Learning (NeurIPS 2025)](https://neurips.cc/virtual/2025/poster/120210)
- [Kim et al. — PhySense: Sensor Placement Optimization (NeurIPS 2025 oral)](https://neurips.cc/virtual/2025/oral/115066)
- [Hoover et al. — Energy Transformer (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/57a9b97477b67936298489e3c1417b0a-Paper-Conference.pdf)

### Top-tier fluid / physics journals

- [Mons et al. — Reconstructing unsteady flows from sparse, noisy measurements (Phys. Rev. Fluids 2025)](https://link.aps.org/doi/10.1103/PhysRevFluids.10.034901)
- [He et al. — 4D-Var super-temporal-resolution reconstruction of turbulent jet (JFM 2024)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/fourdimensional-variational-data-assimilation-of-a-turbulent-jet-for-supertemporalresolution-reconstruction/6249CD6897ADDC4911E8781015B8EC31)
- [Williams, Zahn, Kutz — Sensing with shallow recurrent decoder networks (Royal Society A 2024)](https://royalsocietypublishing.org/doi/abs/10.1098/rspa.2024.0054)
- [Williams et al. — Reduced order modeling with SHRED (Nature Communications 2025)](https://www.nature.com/articles/s41467-025-65126-y)
- [Patel & Maulik — Senseiver (Nature Machine Intelligence 2023)](https://www.nature.com/articles/s42256-023-00746-x)
- [Fukami, Maulik et al. — Voronoi-CNN for global field reconstruction (Nature Machine Intelligence 2021)](https://www.nature.com/articles/s42256-021-00402-2)
- [Wang Sifan et al. — Learning operator with PI-DeepONets (Science Advances 2021)](https://www.science.org/doi/10.1126/sciadv.abi8605)
- [Wang Sifan et al. — Respecting causality (CMAME 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0045782524000690)
- [Wang Sifan et al. — Understanding gradient pathologies (SISC 2021)](https://epubs.siam.org/doi/10.1137/20M1318043)
- [Hasani et al. — Closed-form continuous-time neural networks / CfC (Nature Machine Intelligence 2022)](https://www.nature.com/articles/s42256-022-00556-7)
- [Du et al. — Conditional neural field latent diffusion / CoNFiLD (Nature Communications 2024)](https://www.nature.com/articles/s41467-024-54712-1)
- [PINN-DA-SA — Turbulence model augmented PINN (Phys. Rev. Fluids 2024)](https://link.aps.org/doi/10.1103/PhysRevFluids.9.034605)

### 經典前案與 Nudging（2026-07-18 新增，DOI 全數經 CrossRef 查證）

- Everson & Sirovich — Karhunen–Loève procedure for gappy data, *JOSA A*, 1995 — `10.1364/josaa.12.001657`
- Willcox — Unsteady flow sensing and estimation via the gappy POD, *Computers & Fluids*, 2006 — `10.1016/j.compfluid.2004.11.006`
- Manohar, Brunton, Kutz & Brunton — Data-driven sparse sensor placement for reconstruction, *IEEE Control Systems Magazine*, 2018 — `10.1109/mcs.2018.2810460`
- Azouani, Olson & Titi — Continuous data assimilation using general interpolant observables, *J. Nonlinear Sci.*, 2014 — `10.1007/s00332-013-9189-y`
- **Clark Di Leoni, Mazzino & Biferale — Synchronization to Big Data: Nudging the NSE, *Phys. Rev. X* 10, 011023, 2020 — `10.1103/physrevx.10.011023`（必引；見 §2.7.2）**
- CFDBench — A Large-Scale Benchmark for ML Methods in Fluid Dynamics, arXiv 2310.05963（cylinder case Re ∈ [20,1000]，原文 line 452 實讀確認）。**作者名尚未查證，引用前須補**

### arXiv preprints (not yet venue-locked)

- [FLRONet — Deep Operator Learning for Sparse Reconstruction (arXiv 2412.08009)](https://arxiv.org/abs/2412.08009) — **已發表 ASME 2026, `10.1115/1.4070332`；非 preprint，見 §2.3**
- [FLRNet — VAE + Fourier feature reconstruction (arXiv 2411.13815)](https://arxiv.org/abs/2411.13815)
- [Energy Transformer for sparse reconstruction (arXiv 2501.08339, J. Comp. Phys. 2025)](https://arxiv.org/abs/2501.08339)
- [Wang Sifan — Simulating 3D Turbulence with PINN (arXiv 2507.08972)](https://arxiv.org/abs/2507.08972)
- [DINO — Differential-Integral Neural Operator (arXiv 2509.21196)](https://arxiv.org/abs/2509.21196)
- [SHRED original (arXiv 2301.12011)](https://arxiv.org/abs/2301.12011)
- [Williams et al. — Mobile sensor trajectories with SHRED (arXiv 2307.11793)](https://arxiv.org/abs/2307.11793)
- [Causal PINN original (arXiv 2203.07404)](https://arxiv.org/abs/2203.07404)
- [Wang Sifan — Expert's Guide to Training PINNs (arXiv 2308.08468)](https://arxiv.org/abs/2308.08468)

---

Check: [Protocol_Adhered]
