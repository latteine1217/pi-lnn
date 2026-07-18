# 口試簡報修正清單 — 指導教授 meeting, 2026-07-18

> 來源：與林洸銓老師 meeting 的 19 條意見。
> **頁面一律以「標題」指認，不用頁碼** —— 本次修正包含大幅重排，頁碼會全部位移。
> 對應當下版本：commit `7ec82b3`，33 頁。

---

## A. 全域規則（影響所有頁，最後統一執行）

### A1. 分隔符 `·` 全面改為逗號
教授：「所有的 dot 都要改成 comma，看起來文法才對。」

- 範圍：頁面**可見文字**中作為並列分隔的 `·`（目前全簡報大量使用）。
- **不可一律取代**，以下三類要保留原樣，執行時逐一檢視：
  - 數學符號：`∇·u`（散度）、`f·u`
  - 單位與乘號：`k_max·η`、`2ν𝒵`
  - SectionTag 內的層級分隔 `§ Results · …`（屬結構標記非句子成分，待確認是否也改）
- 風險：機械取代會破壞上列符號，必須逐處確認。

### A2. 無因次化之後不再標單位
教授：「在 Kolmogorov flow 那頁定義完如何無因次化之後，後面就可以都不用單位。」

- 先決條件：**A2 依賴 C2**（先在 `Kolmogorov flow at Re = 10⁴` 頁把無因次化寫清楚）。
- 之後各頁的 `t = 5 s` → `t = 5`，其餘同理。
- ⚠️ 待確認：圖檔內的軸標籤（`time t (s)`、`RMSE (m/s)` 等）目前都帶單位，且 `thesis/CLAUDE.md` 的圖表規則要求「所有軸/表欄有單位（無量綱標 dimensionless）」。**投影片去單位會與論文圖表規則衝突**，需決定：
  - (a) 只有投影片正文去單位，圖檔維持有單位；或
  - (b) 圖檔一併重畫為無因次（工作量大，且與論文不一致）。

---

## B. 結構調整（先做，因為會改變頁序）

### B1. NavBar 增加 Literature review
教授：「最上排 Background 跟 Objective 中間要補一個 Literature review。」

- 檔案：`thesis/slide/components/NavBar.vue`（需確認實際元件路徑）
- 新分段順序：`Background → Literature review → Motivation? → Objective → Methodology → Results → Summary`
- ⚠️ 待確認：Motivation 是否也要成為 NavBar 的一個獨立分段？教授說「slide 少了 motivation」並要求把數頁歸入 motivation，但沒明說 NavBar 是否要加。**建議一併加**，否則那幾頁在 NavBar 上無處歸屬。

### B2. 新增兩頁於簡報最前（封面之後、`The sparse-sensor reconstruction problem` 之前）

| 新頁 | 內容 |
|---|---|
| **為什麼需要 reconstruction** | 真實情況只有 sensor，沒有全場資料，所以需要重建 |
| **PINNs 是什麼、如何運作** | 功能與運作機制的示意 |

- ⚠️ PINNs 示意圖：教授說「可以上網找示意圖」。**建議自己畫**（用既有的 architecture 圖工具），避免版權問題與風格不一致。若要沿用他人圖必須標註出處。

### B3. 新增一頁：Operator 介紹（FNO 與 DeepONet 兩大類）
教授：「在 neural network 跟 neural operator 介紹差異的後面，介紹 operator 的部分，以及 operator 分成兩大類：FNO 跟 DeepONet，這樣教授才看得懂什麼是 DeepONet。」

- 位置：緊接在 `Operator vs. plain PINN` 之後。
- 內容：operator learning 的概念 → 兩大家族 FNO / DeepONet → 本研究採 DeepONet 的 branch–trunk 結構。
- 依據：`thesis/contents/chapter01.tex:37-40`（Table 1.1 的 operator learning 列，含 DeepONet/FNO 引用與 branch–trunk 說明）。

### B4. 頁面重新歸類

| 頁面標題 | 現分類 | 改為 |
|---|---|---|
| `What classical inverse methods require` | Background | **Motivation**（置於 literature review 之後） |
| `Operator vs. plain PINN` | Background | **Motivation**（同上） |
| `Four gaps` | Background | **Motivation** — 作為「文獻有哪些問題」的說明 |
| `K = 100 — the sensor resolution limit` | Background | **Motivation** |
| `Same regime: sensors + PDE, no reference field` | Background | **Objective** — 說明本研究處理了文獻的哪些缺口 |

### B5. Literature review 改為「一類一張表」
教授：「literature review 一頁一張表，將 literature 全部分類，一類一張表來說明。」

- 依據：論文 Table 1.1（`chapter01.tex:23-83`）的**七條研究線**：
  1. Reduced-Order Model (ROM) & sparse identification
  2. Data assimilation
  3. Deep super-resolution / ROM
  4. Operator learning
  5. Stabilized PINNs
  6. Liquid NN / continuous-time
  7. Sparse-sensor with physics
- ⚠️ **待決策**：七類 = 七頁，加上既有頁面，簡報會大幅膨脹（目前已 33 頁／50 分鐘，對 30 分鐘時段嚴重超時）。可能的收斂方式：
  - (a) 七類合併為 3–4 張表（例如：ROM 與資料同化／學習式重建／operator 與 PINN／sparse-sensor with physics）
  - (b) 照教授指示做七頁，另從結果章節大幅刪頁
  - **需與教授確認**，因為這直接衝突於 30 分鐘限制。

### B6. Literature review 不放 ours
教授：「literature review 不要放 ours，只比較文獻。」

- 影響頁面：`What prior methods are trained against`（目前最後一列是 PI-CON）、`Same regime: sensors + PDE, no reference field`（目前含 PI-CON 列，且整頁移往 Objective）。
- 執行：literature review 各表移除 PI-CON 列；比較留到 Objective 段。

### B7. 文獻標註加上期刊名
教授：「literature review 除了作者跟年份，還要加上期刊名字。」

- 目前僅有作者與年份（如 `Mo & Magri 2025`、`Williams 2024`）。
- 資料來源：`thesis/back/references.bib`（已含 `journal` 欄位，例如 Parfenyev 2024 → JETP Letters）。
- ⚠️ 部分條目為 arXiv preprint，可能無期刊名，需逐條查證後標「arXiv」或實際期刊。

---

## C. 逐頁修正

### C1. `Kolmogorov flow at Re = 10⁴` — 標題加註
教授：「title 後面括號寫 DNS solution。」

→ `Kolmogorov flow at Re = 10⁴ (DNS solution)`

### C2. `Kolmogorov flow at Re = 10⁴` — 補無因次化推導
教授：「要寫特徵長度、特徵速度的定義，然後再定義 Re，以及如何做無因次化的。還有 injection-scale 拿掉，會不知道怎麼解釋。」

- 需呈現：特徵長度 `L★` → 特徵速度 `U★` → `Re ≡ U★L★/ν★` → 無因次化方式。
- 依據：`thesis/contents/chapter03.tex:20`（`L★ = 1 m`、`U★ = 1 m/s`、`ν★ = 10⁻⁴ m²/s`、`Re = 10⁴`，且「prescribed through ν★ rather than derived from the realised flow」）。
- **移除** injection-scale Reynolds（`Re_f ≈ 2.5×10³`）。
  - ⚠️ 副作用：`Re_f` 目前也是「這個場到底多湍流」的誠實揭露之一（外部審閱者曾據此質疑）。移除後若被問「injection scale 的 Re 是多少」，需在 speaker notes 備妥答案。

### C3. `2-D Kolmogorov benchmark — setup at a glance` — 重新排版
教授：「重新排版，現在這樣很難理解，不夠乾淨，然後寫 statistic 會讓人不懂到底做了哪些確認。」

- 現況問題：`DNS VERIFICATION` 卡片只寫 `Resolution & turbulence statistics ✓`，未說明實際做了哪些檢查。
- 應明列實際驗證項目，依據 `thesis/CLAUDE.md` 的「DNS/LES verification 5 條」與 `chapter03`：
  - 解析度 `k_max·η = 1.91 ≥ 1.5`（Pope 2000）
  - 能譜斜率
  - 時間窗充分性 `T/t_eddy = 2.51`（**未達理想 50–100，論文誠實揭露**，不可寫成通過）
  - energy budget 殘差
  - grid independence
- ⚠️ 注意：不可把未達標的項目標成 ✓。

### C4. `2-D Kolmogorov benchmark — setup at a glance` — 定義取樣參數
教授：「`Δt_s` 跟 `N_t` 要定義清楚，這是 sampling 的參數，然後還要說清楚實際模擬時的 `Δt` 是多少。」

- 需區分兩個時間步：
  - **模擬步長** `Δt = 2.5×10⁻⁴`（solver 積分步）
  - **取樣間隔** `Δt_s = 0.025`（每 100 步存一幀），`N_t = 201`
- 依據：DNS config（`dt=2.5e-4`, `save_interval=100`, `T_end=5`）。

### C5. `2-D Kolmogorov benchmark — setup at a glance` — 圖加 caption
教授：「圖要有 caption 說明這張圖在表示什麼（展示 DNS 跟 sensor 位置的展示），還要註明如何表示 sensor 取點的方式跟是固定位置的這件事。」

- 對象：`sensor_distribution_kolmogorov_K100.png`
- caption 需含：(a) 底圖是 DNS 場、(b) 點是 K = 100 個 sensor 位置、(c) 取點方式為 **QR-pivoting on the LES POD basis**、(d) 位置**訓練前選定、推論時不變**（固定測站）。

### C6. `Operator vs. plain PINN` — 下方框框移往 Objective
教授：「下方框框放到 objective 那邊再說。」

- 需確認該頁下方框的實際內容後移動。

### C7. `K = 100 — the sensor resolution limit` — 改標題
教授：「說法要像是：在 sparse sensor amount 下，討論 resolution limit 的部份。」

- 建議：`Resolution limit under a sparse sensor budget`（待定案）
- 同時歸入 Motivation（見 B4）。

### C8. `K = 100 — the sensor resolution limit` — 能量帶說明寫清楚
教授：「energy inside the band 現在太精簡了，要寫清楚一點，說明白是多少 k 最多可以重建多少 energy。」

- 需明確表述：**到 k = ? 為止，最多可重建 ? % 的能量**。
- 數據來源：`scripts/plot_nyquist_recoverability.py` 與 `chapter04` 的 `F_DNS(k_max^sensor)` 累積能量分數；`k_max^sensor = √(K/π) ≈ 5.64` 對應約 99 % 能量（**須重新核對實際數值後再寫**，不可沿用記憶）。

### C9. `Three additions to DeepONet` — 標題重想 + 圖例標注
教授：「要標注框框底色代表的是什麼，還有加入標注哪一個部分是我們這個研究新加的，標題也要重新想一下，現在寫的 three additions 的寫法不是很好。」

- 加圖例（legend）說明各底色語意（灰 = 沿用的 DeepONet backbone、藍 = 本研究新增）。
- 明確標示「本研究新增」的部分。
- 標題重擬（待定案）。

### C10. `Three additions to DeepONet` — 拆頁
教授：「下方的三個 card 移到下一頁去介紹，這一頁只放架構圖，然後 card 下方要明確說明：K = 100 用 vanilla DeepONet 是做不出來 reconstruction 的，但是加了這三項修正之後，才能夠成功訓練出來。」

- 本頁：只留架構圖。
- 新頁：三個 card（CfC / cross-attention / AL）+ 結論句。
- ⚠️ **結論句的措辭需要證據校準**：目前 2×2 ablation 的實測是 **B0 vanilla DeepONet = 8.23 %**（可訓練，並非「做不出來」），B3 = 5.71 %。
  - 「vanilla DeepONet 做不出來」與實測不符，若照字面寫會被委員以自己的 ablation 表反駁。
  - 建議措辭：**vanilla DeepONet 可訓練但達不到工程門檻／誤差高出 2.52 個百分點（p = 3×10⁻⁷）**，三項修正才把它壓進可用範圍。
  - **需向教授確認**此處是否為口誤，或有其他所指（例如早期實驗確實無法收斂）。

---

## D. 衝突點與裁決（2026-07-18 已確認）

1. ~~**時間**~~ → **由作者自行 dry run 掌握**，執行時不因時間限制刪減內容。
2. **B5 的七類表格**：是否合併為 3–4 張 → **尚未裁決**，執行到 B5 時再確認。
3. ~~**A2 去單位**與圖表規則衝突~~ → **圖檔單位不動**，只有投影片正文去單位。圖軸維持
   `time t (s)` 等現況，與論文圖表規則一致。
4. ~~**C10 「vanilla DeepONet 做不出來」**~~ → **已澄清：指的是「重建結果不好」，不是無法訓練。**
   措辭因此定為「重建品質不足以達到工程門檻」之類，**不可寫成「無法訓練／訓不出來」**
   （B0 實測 8.23 % 是可訓練的，寫成訓不出來會被自己的 ablation 表反駁）。
5. **C2 injection-scale** → **同意移出頁面，改放 speaker notes**。被問「injection scale 的 Re」
   時照 note 回答 `Re_f ≈ 2.5×10³`。

---

## E. 執行順序建議

1. **D 的四點先取得裁決**（尤其時間與七類表格，會決定整體規模）
2. B1 NavBar → B4 重新歸類 → B2/B3 新增頁（結構定案）
3. B5 / B6 / B7 literature review 改寫
4. C1–C10 逐頁修正
5. **A1 / A2 全域規則最後執行**（避免新增頁面又引入不合規寫法）
6. 全頁 overflow 複查（歷次修改最常見的回歸）
