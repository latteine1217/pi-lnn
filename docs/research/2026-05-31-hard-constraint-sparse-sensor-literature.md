# Deep Research — Hard-Constraint BC Enforcement & Sparse-Sensor Flow Reconstruction (2022–2025)

> 日期：2026-05-31
> 觸發：CEXP-016~037 hard BC 路線全失敗後，調查文獻 SOTA 怎麼做 geometry enforcement + sparse reconstruction。
> 方法：deep-research workflow（run wf_c6a8ca74）— 5 角度 fan-out → 21 來源 → claims 抽取 → 3-vote 對抗驗證。
> **狀態：VERIFIED**（本文件內容為 workflow 實際輸出，非先驗腦補。每條 claim 附 vote 與 source）。

---

## 最重要發現（直接命中本專案）

### ★ Zhu et al. 2025 (arXiv:2503.24074) — 獨立記錄了我們的 over-energy 失敗機制

penalty-based immersed/porous-media PINN，flow past cylinder。**這是整個 corpus 中最對症的一篇**：
- 它在 NS momentum 嵌入連續 body-fraction φ：`(1−φ)·(NS residual) + α·φ·(u−U) = 0`（φ=0 流體, φ=1 固體, U=0 no-slip）。**不是 output transform gate**。
- 它**明確記錄**：圓柱後方速度低 → solid penalty `α·φ·(u−U)` 相對 fluid residual 變小 → 形成「competitive relationship」→ **圓柱後緣成為主要誤差區**。
- **這正是我們 CEXP-037 的 over-energy（ke_pred/ref 3.83）機制**：body 區 penalty 太弱、wake 過度補償。
- 它的解法：**提高 α 到 ~10 重新平衡**（penalty rebalancing），**不是用更硬的 gate**。
- vote 3-0，single-source 但與 abstract 直接核對。

→ **我們撞的牆，文獻已記錄，且修法是「調 penalty 平衡」或「換 div-free 參數化」，不是 hard gate。**

### ★ 真正的研究 gap（= 我們的 novelty）

> workflow 結論原文：「No source was found that combines ALL three of the project's hard constraints — **sparse wake-only sensors + hard solid-body BC + PDE-residual-only (no full-field prior)**. This exact configuration is a genuine research gap.」

reconstruction SOTA（FLRONet/Voronoi-CNN）與 physics-enforcement SOTA（hard-BC/div-free PINN）是**兩個幾乎不相交的文獻**；我們正好坐在交集，而交集人煙稀少。**這是合法的論文定位，不是缺陷。**

---

## 1. Hard-constraint / boundary enforcement（已驗證 claims）

| Claim | vote | 內容 |
|---|---|---|
| **Sukumar ADF hard BC**：`u = g + φ·NN`，φ 是 approximate distance function（R-functions + transfinite interpolation），body 上 φ=0 乘性歸零，BC by construction、移除 BC penalty | **3-0 (×4)** | Sukumar & Srivastava CMAME 2022 / arXiv:2104.08426。我們的 hard BC gate 就是這個。canonical。 |
| **Div-free 架構強制**（3 種）：(a) stream-function/curl `u=curl(ψ)`，div(curl)=0 恒等（2D 單一 ψ exact）；(b) spectral Leray (Helmholtz-Hodge) projection 限制 hypothesis space 到 div-free 到機器精度；(c) div-free matrix-valued RBF kernel（Wendland C⁴）。全部移除 divergence loss term + 改善 2D NS 穩定性 | **3-0 / 2-1** | stream-function PINN: Horne et al. arXiv:2601.06244。Leray: 'Project and Generate' arXiv:2603.24500。DFK: arXiv:2504.01913。理論根據 Neural Conservation Laws NeurIPS 2022。 |
| **SDF-conditioned operator**：SDF 當 input 給 Geometric-DeepONet → boundary-layer 精度 +32% vs 標準 DeepONet（steady 3D, Re 10-1000）；加 Sobolev gradient 約束再 +25~45% | 3-0 / 2-1 | Rabeh et al. 2025 arXiv:2503.17289。⚠️ **trained from FULL fields, 非 sparse sensor**；自報「up to」最佳值。呼應我們 CEXP-020 SDF input 在 sparse 下失敗。 |
| **HCP-PINN projection layer**：把 (u,v,p) 投影到「只容許離散化 NS 精確解」的 hyperplane，硬約束 PDE 而非 soft loss | 2-1 | Horne et al. arXiv:2601.06244。NUANCE: exact 在離散 affine 形式，非連續 PDE pointwise。 |
| **純 hard BC 在複雜幾何會 degrade interior**：hard particular-solution network 被迫精確滿足 BC → 內部「disordered」高頻失真；改用 **soft** particular-solution net 反而更準 | 2-1 (medium) | arXiv:2411.08122 (Nov 2024)。**直接反駁「hard BC 一定更好」**，呼應我們 hard BC 系列失敗。single-source。 |

## 2. Sparse-sensor reconstruction（已驗證 claims）

| Claim | vote | 內容 |
|---|---|---|
| **Reconstruction SOTA = supervised full-field operator，不用 physics residual / hard BC** | **3-0 / 2-1** | FLRONet 2024 arXiv:2412.08009（Voronoi sensor encoding + FNO branch-trunk）；Voronoi-CNN Fukami Nat. Commun. 2021。**plain Adam + MSE/perceptual loss，無 GradNorm/NTK/AL，無 hard BC**。⚠️ 全靠 full-field snapshot library 學 data prior = 我們明確拒絕的工程不可遷移假設。 |
| ~~QR-pivot sensor placement 標準最優佈點~~ | **未驗證** | ⚠️ workflow caveat #1：**sensor-placement 細節（QR-pivot/POD/greedy）沒有被任何 surviving claim 證實**。驗證到的 reconstruction 方法（FLRONet/FLRNet）改用 placement-agnostic Voronoi encoding，不做 placement 最優化。我們的 QR-pivot 路線在此 corpus **未被覆蓋**——這是 open question 不是已驗證事實。 |
| **Global/spectral div cleanup 勝過 local collocation penalty**（long-rollout 穩定）| 2-1 (medium) | 'Project and Generate' arXiv:2603.24500。single-source, generative-turbulence context（非 sparse reconstruction）。 |

## 3. Optimizer / loss-balancing

- workflow **未**確立 SOAP 為 PINN 標準（我先前草稿的這點未被支持）。
- reconstruction 主流用 plain Adam + MSE（無動態 weighting）。
- **過度強調 augmented Lagrangian 是我先前草稿的錯誤**——workflow 找到的 over-energy 修法是 **penalty rebalancing（提高 α）** 與 **div-free 參數化**，不是 AL。AL 在此 corpus 未被當成主流 fix。

---

## 對 cylinder 問題的行動建議（基於已驗證證據）

1. **最強訊號 = stream-function / div-free 架構**（3-0 多來源）。同時解 (a) CEXP-037 over-energy（無 velocity gate → 無過度補償，約束在 ψ）、(b) CEXP-002 div=1.14 over-smoothing 假解（div=0 by construction）、(c) geometry awareness（body = ψ level-set）。⚠️ 與使用者「NS primitive only」決定衝突，但這是文獻最一致的方向。
2. **若堅持 primitive variable**：Zhu 2025 的 immersed body-fraction `(1−φ)·NS + α·φ·(u−U)` + 提高 α（~10）重新平衡，是有 citation 的對症做法（取代我先前誤推的 AL）。注意它仍會在圓柱後緣留誤差。
3. **論文定位**：sparse wake-only + hard BC + PDE-only 三者交集是 genuine gap → 合法 novelty。

## Key sources（已驗證）
- Sukumar & Srivastava, CMAME 2022 (arXiv:2104.08426) — ADF hard BC
- **Zhu, Chen, Deng, Bian 2025, Acta Mechanica Sinica（已發表期刊, DOI 10.1007/s10409-025-25273-x, arXiv:2503.24074）** — immersed body-fraction PINN，Eq.(12) `(1−φ)·NS + α·φ·(u−U)`，verbatim 記錄「competitive relationship」+ rear-boundary 主誤差，α=10 緩解 ★ 最對症
- Horne, Jimack, Khan, Wang (arXiv:2601.06244) — stream-function + HCP projection
- 'Project and Generate' (arXiv:2603.24500) — Leray projection div-free
- Rabeh et al. 2025 (arXiv:2503.17289) — Geometric-DeepONet (SDF)
- arXiv:2411.08122 — 純 hard BC degrade interior，改 soft 更好
- FLRONet (arXiv:2412.08009) + Voronoi-CNN (Fukami Nat. Commun. 2021)
- Manohar et al. IEEE CSM 2018 (arXiv:1701.07569) — QR-pivot placement

## Open Questions（workflow 自報，其中 #2 是我們的下一步關鍵）

1. sparse sensor 在「全集中 wake、無 upstream/body 覆蓋」下怎麼最優佈點？驗證到的 reconstruction 論文都不做 placement 最優化（只用 placement-agnostic Voronoi）。
2. **★ 結合 exact div-free 架構（stream-function/Leray）+ body-fraction/hard no-slip，是否真的消除 wake over-energy，還是同時強制 div-free 與 solid-zero-velocity 會重新引入 Zhu et al. 記錄的 competitive-term 失衡？無任何 source 測過這個組合。** ← 這正是我們若走 stream-function 要面對的核心未知。
3. 為何 production reconstruction（FLRONet/FLRNet）完全不用 physics residual + adaptive weighting？是否因 full-field MSE 已足夠；在「真正 sensors-only 無 full-field」預算下加 PDE residual + hard BC 是否有幫助？（= 我們的設定）
4. body-fraction PINN (Zhu 2503.24074) 在 cylinder sparse reconstruction 的量化精度（KE/RMSE）是多少？vs hard-output-transform 在相同稀疏度下的 head-to-head？corpus 只抓到失敗機制，沒抓到精度數字。

## Limitations（workflow 自報）
- 多個量化 claim（Geo-DeepONet +32%、DFK exactness、FLRONet superiority）single-source、自報、未獨立複現。
- 兩篇 2026 arXiv ID（2601.06244, 2603.24500）peer-review 狀態未確認，是極新、輕度審查的工作。
- stream-function 優越性在 *data-driven decoder* 設定最強；轉到 *PDE-residual-only sparse* PINN 物理合理但未被直接 benchmark。
- over-energy 失敗只在**單一 source（Zhu 2503.24074）**具體記錄；其修法（提高 body-fraction penalty）是經驗性，非一般保證。
- 1 條 claim 被推翻：HDNet Helmholtz hard-decomposition（arXiv:2406.08570）未通過驗證（1-2）。
- 統計：6 角度、28 來源 fetch、116 claims 抽取、25 驗證、24 confirmed、1 killed、合成後 10 條。
- 無一篇同時涵蓋我們三個約束（sparse wake-only + hard solid BC + PDE-only）。
