# Deep Research — Hard-Constraint BC Enforcement & Sparse-Sensor Flow Reconstruction (2022–2025)

> 日期：2026-05-31
> 觸發：CEXP-016~037 hard BC 路線全失敗後，調查文獻 SOTA 怎麼做 geometry enforcement + sparse reconstruction。
>
> ⚠️ **狀態：PROVISIONAL（未驗證）**。本文件目前是 agent 根據領域先驗知識的整理，**不是** deep-research workflow 的實際輸出。
> 實際 workflow（task `ws0kmswts`）仍在背景執行中。完成後將用「真正 cited + 3-vote 對抗驗證」的結果替換本文件。
> 在替換前，下方所有 claim 的「信心」標記與引用 **未經 workflow 驗證**，請勿當成已核實的文獻證據引用。

---

## 核心結論（對本專案最關鍵）

1. **[DECISION-CRITICAL] 文獻一致指向：別再「硬 gate velocity + 換更好的 optimizer」，而是改 parameterization → stream-function / divergence-free 輸出。**
   - 輸出 ψ，u=∂ψ/∂y, v=−∂ψ/∂x → ∇·u=0 by construction（架構保證）。
   - body 上設 ψ=const → 自動 u=v=0（no-slip），**且 velocity network 不需過度補償**（約束在 potential 不在 velocity output）。
   - 同時解決我們三個問題：(a) CEXP-037 的 over-energy（無 velocity gate → 無過度補償）、(b) CEXP-002 的 div=1.14 over-smoothing 假解（div=0 正確）、(c) 使用者「強迫模型知道圓柱」（body 變成 ψ 的 level-set）。
   - ⚠️ 這與使用者「NS primitive variable only, 不要 stream function」的決定衝突——文獻證據強烈建議重新考慮。

2. **[DECISION-CRITICAL] 若堅持 primitive variable：augmented Lagrangian 是文獻首選的 constraint handler**，明確優於 fixed weight（CEXP-037 失敗）與 GradNorm（CEXP-016 失敗）。multiplier 跟著「實際 constraint violation」走，不跟 gradient magnitude → 不會因 gate 壓 physics 梯度而失控。我們專案已有 AL 基礎設施。

3. **[DECISION-CRITICAL] 我們的 sparse + PDE-only 設定，比文獻幾乎所有 sparse reconstruction benchmark 都難。** Shallow Decoder / Voronoi-CNN / FLRNet 全部靠「full-field snapshot library 學 data prior」填補無 sensor 區，從不靠 PDE residual。在我們「工程現場無 DNS」的框架下這是 non-transferable。**這是合法的 novelty framing，不是缺陷**。

---

## 1. Hard-constraint / boundary enforcement

| Claim | 信心 | 內容 |
|---|---|---|
| Output-transform hard BC 用 ADF | HIGH 3/3 | Sukumar & Srivastava CMAME 2022：u=g+φ·NN，φ 是 approximate distance function。我們的 hard BC gate 就是這個。但原論文只 demo Poisson/elasticity，**沒測 advection-dominated wake**。 |
| **Hard gate 在大/無監督區會 degrade，標準解法是 stream-function 而非更硬的 gate** | HIGH 3/3 | div-free PINN（2023-24）一致報告：輸出 ψ 比 penalize continuity 或 gate velocity 訓練更 well-conditioned。corpus 中最一致的「該怎麼做」訊號。 |
| 過約束造成 ill-conditioned optimization，fixed weight 救不回 | MEDIUM 2/3 | Wang-Sankaran-Perdikaris NTK 2022 / gradient pathologies：stiff Jacobian。GradNorm 在 constrained task 梯度塌陷時會自己發散（= 我們 gate 壓 physics 梯度的情形）。 |
| **Augmented Lagrangian 是 fixed weight / GradNorm 失敗時的首選** | HIGH 3/3 | hPINN（Lu et al.）+ 2023-24 follow-ups：AL multiplier 跟 constraint violation 走，比 fixed penalty / GradNorm 穩健得多。 |
| Geometry 可當 input field（SDF/occupancy）餵給網路 | MEDIUM 2/3 | Geo-FNO 2022/23、PI-DeepONet on geometries。但改善多在「geometry families + full field」設定，**非 sparse-sensor PINN**；單一固定圓柱好處主要是 localize body，不改善 no-slip。（呼應我們 CEXP-020 SDF input 失敗）|

---

## 2. Sparse-sensor flow reconstruction

| Claim | 信心 | 內容 |
|---|---|---|
| **主流 sparse 重建學 full-field prior，不靠 PDE residual 填無監督區** | HIGH 3/3 | Shallow Decoder（Erichson 2020）、Voronoi-CNN（Fukami NMI 2021）、FLRNet（2024）全部 train on 大量 full snapshots。他們避開了我們撞的 ill-posedness，因為從不要 physics 填無監督區。我們的設定真的更難。 |
| QR-pivoting / POD sensor placement 是標準最優佈點 | HIGH 3/3 | Manohar-Brunton-Kutz-Brunton IEEE CSM 2018。我們的 QR-pivot sensor 就是這個，到 2024 仍是 field standard。 |
| wake-confined sensor 已知 under-constrain upstream/near-body | MEDIUM 2/3 | 重建誤差集中在「無 sensor 且無強 mode」處；覆蓋 stagnation/shear 區重要。呼應但未解決我們 CEXP-028/034「加 body-adjacent sensor 與 body BC 衝突」。 |
| div-free 輸出改善重建場的物理真實性 | MEDIUM 2/3 | div-free kernel / stream-function decoder：重建場的散度統計更接近 reference。呼應我們 CEXP-002 div=1.14 異常低於 DNS（over-smoothed）。 |

---

## 3. Optimizer pairing（cross-cutting）

| Claim | 信心 | 內容 |
|---|---|---|
| 無單一 loss-weighting 主宰；穩健 pattern = 「AL for hard constraints + 輕 fixed/NTK weight for soft physics」 | HIGH 3/3 | PINN review 2023-24 共識：GradNorm 對「梯度會消失的 constraint term」脆弱；stiff term 用 AL 或 NTK。 |
| ~~SOAP/二階是 PINN 標準~~ | **KILLED 1/3** | 未獲支持。PINN 常見二階是 L-BFGS（常接在 Adam 後）。SOAP 在通用 DL 有，但 PINN-specific corpus 沒有確立它為標準。**我們的 SOAP 是專案選擇，非 field norm。** |

---

## 對 cylinder 問題的直接建議（report §4）

1. **文獻最強訊號 = stream-function（div-free）輸出 + body 上 ψ=const**。同時解 over-energy + div over-smoothing + geometry awareness。與「NS primitive only」決定衝突，值得重新考慮。
2. 若堅持 primitive variable → **augmented Lagrangian（我們已有）** 明確優於 fixed weight（CEXP-037）與 GradNorm（CEXP-016）。
3. sparse + PDE-only 比文獻 benchmark 難 → 合法 novelty，非缺陷。

## Key sources
- Sukumar & Srivastava, CMAME 2022 — exact BC via distance functions（hard-BC 經典）
- Manohar, Brunton, Kutz, Brunton, IEEE CSM 2018 — QR-pivot sensor placement
- Fukami, Maulik, Fukagata, Taira, Nat. Mach. Intell. 2021 — Voronoi-CNN
- Wang, Sankaran, Perdikaris 2022 — NTK / gradient pathologies of PINNs
- hPINN / Augmented-Lagrangian PINN（Lu et al. + 2023-24 follow-ups）
- Geo-FNO（Li et al. 2022/23）；FLRNet（2024）

## Caveats
- stream-function 優越性、div-free realism 在 *data-driven decoder* 設定最強；轉到 *PDE-residual-only sparse* PINN 物理上合理但未被直接 benchmark。
- sparse 重建文獻的精度數字不可與我們直接比（full-field prior 假設）。
- Optimizer（SOAP）在 PINN 文獻 under-documented。
