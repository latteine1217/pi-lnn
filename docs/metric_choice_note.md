# Metric Choice Note — KE rel-err 為何不能單獨用，以及 forward-CFD baseline 的正確歸因

> **Status**: analysis note, 2026-05-27
> **Purpose**: 補強 `paper_framing_draft.md` contribution 2（"KE-as-misleading-metric"）的證據；
> 並釐清 `pi-lnn-cfd-baseline` 的 forward-CFD baseline 之「KE 低」**機制與任務**，避免論文錯誤歸因。
> **Scope**: 不推翻既有結論。論文主張「multi-metric evaluation」方向正確，本文只把證據釘死、把一個歸因坑標出來。

---

## TL;DR

- KE rel-err 對 Re=10000 Kolmogorov 這種混沌場是 **necessary-not-sufficient** 的零階統計量，單獨使用會**把優劣排反**。
- 實測：`forward_cfd_baseline`（**gappy POD** rank-40 IC + **open-loop** ETDRK4 forward）KE rel-err **3.85%**，但 u/v rel-L2 = **153%/204%**、u 空間相關 **−0.58**（場與參考反相關）。
- 同協議下 Pi-LNN B3 真實值（seed3/4 `summary.json`）u/v rel-L2 = **12.4%/22.9%**、KE ≈ **10.7%**。**用 KE 看 baseline 贏，用 field-L2 看 Pi-LNN 完勝 ~9–12×。**
- **歸因坑**：論文 contribution 2 把 classical 的低 KE 歸因為 *over-smoothing（predicting spatial mean）*。這對 trig-LSQ reconstruction 成立，但對 forward-CFD baseline **不成立**——它的低 KE 是 **chaotic phase decorrelation**（能譜解析充分，相位被混沌放大打亂），且任務是 **forecast** 而非 reconstruction。

---

## 1. Evidence — KE 排反優劣（apples-to-apples 指標、非 apples-to-apples 任務）

同樣 K=100 QR-pivot sensors、同樣 DNS Re=10000 reference、同樣 t=5：

| Metric | Pi-LNN B3 (real `summary.json`) | forward_cfd_baseline | Winner |
|---|---|---|---|
| KE rel-err | ~10.7% | **3.85%** | baseline (misleading) |
| u rel-L2 | **12.4%** | 153% | Pi-LNN ~12× |
| v rel-L2 | **22.9%** | 204% | Pi-LNN ~9× |
| ω rel-L2 | ~49% | (not computed) | — |
| forcing-mode phase err | −0.025 rad | u-corr −0.58 (相位全錯) | Pi-LNN |

來源：baseline = `pi-lnn-cfd-baseline/reports/forward_cfd_baseline_T5_rank40.{json,npz}`（值經本機 npz 獨立重算確認）；
B3 = `artifacts/kolmogorov/deeponet-cfc-re10000-exp09[78]-b3-seed[34]/deeponet-cfc-eval/summary.json`。

> 一個會把「場全錯（corr −0.58）」評為優於「場大致對（u-L2 12%）」的指標，不能當 headline 或排名指標。

---

## 2. 為何 KE rel-err 對混沌場欠定（four reasons）

1. **Dimensional collapse**：KE = ½⟨u²+v²⟩ 把 2·256² ≈ 1.3×10⁵ 自由度壓成 1 個純量；用單一純量驗證逆問題嚴重欠定。
2. **Attractor-bounded**：統計穩態下 total KE 由 forcing–dissipation 平衡鎖在窄帶；任何「沒爆掉」的場 KE 都自動接近 → 低 KE rel-err 是最低門檻（baseline 盲推去相關場都能拿 3.85%）。
3. **Phase-blind**：由 Parseval，KE 只累加能譜振幅 |û(k)|²，丟棄相位；湍流結構住在相位裡。能譜相同、相位隨機化的兩場 KE 完全相同。
4. **Non-diagnostic**：純量無法定位誤差（大/小尺度、振幅/相位、頻段），無法支撐「學到動力學」的 claim。

`summary.json` 已具備正確的多指標（`u/v/ω_rel_l2`、`ek_ratio_kf`、`band_energy_rel_err`、`div_l2`、`kf_phase/amp`）。**論文要做的是把 headline 從 KE 換成 field-L2 + spectrum，KE 降為附屬 sanity check。**

---

## 3. 歸因坑 — forward-CFD baseline 不是 over-smoothing

論文 abstract/contribution 2 的低-KE 解釋是 *classical methods optimize KE through over-smoothing*。比對 `scripts/baseline_squeeze.py` 後確認：論文既有 baseline 與 forward_cfd_baseline 分屬**三類**——前兩類是 reconstruction（每個 t 都見該 t 的 sensor），第三類是 forecast：

| | 論文 fair (RBF×3 / IDW / div-free trig-LSQ×3) | 論文 cheat (gappy POD r=100) | forward_cfd_baseline (gappy POD r=40 + open-loop) |
|---|---|---|---|
| 任務 | per-snapshot **reconstruction**（每個 t 用該 t 的 sensor） | per-snapshot **reconstruction** | **t=0-only forecast**（僅 t=0 sensor 盲推到 t=5） |
| 用 DNS basis | 否（engineering-transferable, "fair"） | 是（DNS-trained gappy POD rank-100, "cheat"） | **是**（DNS-trained gappy POD rank-40，但只用於 IC） |
| 低 KE 機制 | **over-smoothing**（往 spatial mean 收，砍小尺度能量） | 近完美（DNS basis 直接張成解空間） | **chaotic decorrelation**（譜滿、相位被 Lyapunov 打亂） |
| KE rel-err | trig-LSQ k≤5 (80 modes) **3.93%** | **0.12%** | **3.85%** |
| u rel-L2 | trig-LSQ **~28%**（= EXP-080 17% +11pp） | **0.85%** | **153%** |
| IC 不可壓 | n/a（每步重建） | n/a | **max|div|=1.7×10⁻²**（rank-40 截斷不保無散；ref 5×10⁻¹³） |

> **命名（2026-07-18 統一）**：後兩欄是**同一個重建方法**的兩種用法——都是 gappy POD（Everson & Sirovich, *JOSA A* 12:1657–1664, 1995），差別只在 rank（100 vs 40）與「每個 t 重做一次」vs「只在 t=0 做一次、之後 open-loop 自由前推」。`forward_cfd_baseline` / "forward CFD" 是本專案的**內部簡稱，不是文獻方法名**；正式描述為 **gappy-POD initialisation + open-loop (free-run) forward integration**（後者是 data assimilation 的標準 no-assimilation 對照組）。artifact / script 檔名沿用舊名不改，避免破壞既有引用。
>
> Pi-LNN（B3 u-L2 12.4% / EXP-080 17%）與前兩類同為 **reconstruction**（皆見 t=5 sensor）→ 任務對等，論文這部分比較合理。`forward_cfd_baseline` 是唯一的 **forecast**，自成一類，不屬於論文現有任何一格。
>
> **二度印證 KE 誤導**：forward_cfd_baseline 的 KE 3.85% ≈ trig-LSQ 的 3.93%（幾乎相同），但 u rel-L2 是 153% vs ~28%。只看 KE 會把「forecast 去相關」誤判為「≈ trig-LSQ reconstruction」，完全錯置。

證據：baseline 的 enstrophy-spectrum tail 18.75 decades + CFL 0.073 → **排除 under-resolution**；場去相關純由 IC 誤差（POD 截斷 5%）經混沌放大造成，**不是 over-smoothing**。

**含意**：
- 若論文要把 `forward_cfd_baseline` 列為 baseline，**不能說它 over-smooths**——它的譜是滿的，問題在相位/任務。錯誤歸因會被審稿人以「baseline 機制誤述」質疑。
- `forward_cfd_baseline` 與 Pi-LNN 任務**不對等**（forecast vs reconstruction）。要納入比較，須對齊任務：要嘛 baseline 也做同化（用 t=0–5 sensor），要嘛只在「forecast horizon / 統計量」層面討論，不要拿 t=5 pointwise 直接對 reconstruction。

---

## 4. 建議的指標階層（可直接寫進 metric-justification 段）

| 層級 | 指標 | 作用 | 狀態 |
|---|---|---|---|
| Pointwise | u/v/ω rel-L2、spatial correlation | 主指標：場對不對 | ✅ 已有 |
| Structure | full-band E(k) 比較、band-wise rel-err | 能量分布（非總和） | ✅ band_energy 已有 |
| Constraint | divergence L2/Linf | 不可壓（注意：NN 軟約束 vs spectral 投影定義不同，**勿跨方法直接比**） | ✅ 已有 |
| Dynamics | forcing-mode phase/amp（kf_phase/amp） | 相位/驅動一致性 | ✅ 已有 |
| Scalar SC | KE rel-err、ek_ratio | **附屬** sanity check（necessary-not-sufficient） | ✅ 已有 |
| Chaos-aware | correlation-decay time / Lyapunov horizon | forecast 任務專用；超過 horizon 改報統計量 | ⬜ 若做 forecast 比較才需 |

---

## 5. Action items for the paper

1. Headline / abstract 的成績以 **u/v rel-L2 + spectrum** 表述；KE 明確標為 necessary-not-sufficient。
2. 修正 `pi-lnn-cfd-baseline/dns/forward_cfd_baseline.py` `[6] Comparison`：目前**只比 KE**，會得出「baseline 贏」的反向錯誤結論——至少並列 field-L2。
3. 若保留 forward-CFD baseline：(a) 改正歸因（chaotic decorrelation，非 over-smoothing）；(b) 標明 forecast vs reconstruction 任務差異，或對齊任務後再比。
4. **`forward_cfd_baseline` 不屬於論文現有 baseline 清單的任何一格**（論文 baseline 全是 per-snapshot reconstruction；它是唯一的 t=0-only forecast，且用 DNS-POD basis）。**不可放進那張 reconstruction 比較表**——u-L2 153% 與 reconstruction 的 ~0.85%/~28% 不在同一任務上。若要保留，另立「forecast / predictability-limit baseline」一節，它回答的是「最強 cheat IC（rank-40 DNS-POD）+ 完美 solver 純前推到 t=5 會如何」（答：混沌去相關），而非 reconstruction quality。
