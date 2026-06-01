# Design — Cylinder Sensor-Conditioned FiLM (CEXP-039 / CEXP-040)

| Field | Value |
|---|---|
| Date | 2026-05-31 |
| Status | Approved |
| Owner | latteine |
| Related | cylinder_log_v2.md Finding #8, [literature survey](../../research/2026-05-31-hard-constraint-sparse-sensor-literature.md) |
| Predecessor | CEXP-020/022/023-027 (geometry embedding 全失敗), CEXP-037/038 (hard/soft body BC 全失敗) |

---

## 1. Goal & Background

**Goal**：用 FiLM（Feature-wise Linear Modulation）讓 geometry/sensor context **大域地調制 decoder 整個函數**，而非像過去那樣把幾何當 input 一維 concat。透過 ablation 分離「sensor state 是否足夠」vs「明示 geometry 是否有幫助」。

**為何過去全失敗（root cause）**：
- CEXP-020（SDF input）/ CEXP-022（geometry token）/ CEXP-023-027（graph memory）：把幾何當 **input 一維**。NN 仍 `u=f(x,y,t,[geo])` 直接輸出 velocity；body 區無 sensor → geo 是 adversarial prior → over-energy。
- CEXP-037/038（hard gate / soft penalty）：在 velocity output 強制 body=0 → 圓柱邊緣人工剪切層 → over-energy（vorticity 圖確認）。
- **共同根因（Finding #8）**：body 區無 sensor supervision，任何「告訴模型 body 在哪 + 要求 velocity 反應」都缺校正。

**FiLM 為何可能不同**：
```
過去:  u = f(x, y, t, [geo])           ← geo 是 input 一維，局部反應 → 邊緣不連續
FiLM:  u = f_cond(x, y, t)             ← cond 變調整個網路 h_layer = γ(cond)·h + β(cond)
```
FiLM 讓 conditioning **大域影響整個函數行為**（不是局部一維）。關鍵設計：**conditioning 信號從 sensor hidden state 來**（= 有 supervision 的地方），所以「body 區無校正」這個根因被**結構性回避**——校正來自 sensor 經 CfC encoder 學到的 wake 狀態。

**程式碼事實**：decoder forward 內已有 `h_branch_tokens = h_states[idx]`（per-query 的 time-gathered sensor hidden state，[N, d_model]）。這正是 FiLM 的天然 conditioning 源。既有 `use_re_film` flag（Re conditioning FiLM）提供 FiLM 機構參考。

---

## 2. Architecture

### 共享 FiLM 機構（CEXP-039/040 共用）

新增 `use_sensor_film: bool` flag。啟用時，decoder 在 trunk MLP 各 block 後套 FiLM：
```
h_layer ← γ(cond) ⊙ h_layer + β(cond)
```
- `cond` 從 per-query conditioning vector 來（見下 B3-a/b 差異）。
- `γ, β` 由小 MLP 從 cond 算：`[γ; β] = FiLM_MLP(cond)`，γ identity-init（γ=1, β=0）→ 啟用時不破壞既有行為。
- FiLM 套在 `trunk_feat` 上（line ~345 各 block 後），對 3N batch 廣播。

### B3-a (CEXP-039): Pure sensor state
- `cond = pool(h_branch_tokens)`（per-query sensor hidden，[N, d_model] → FiLM）。
- **geometry 完全不入**。模型靠 sensor 狀態隱式推斷 body 存在。
- 新 flag：`use_sensor_film=true`, `film_use_geometry=false`。

### B3-b (CEXP-040): Sensor state + geometry
- `cond = concat(pool(h_branch_tokens), geo_descriptor)`。
- `geo_descriptor` = 每個 query 的 body distance（既有 `body_distance` 已在 forward_uvp 簽名內，differentiable）。
- 明示 geometry，但與 sensor h 同進 FiLM → sensor h 提供校正錨。
- 新 flag：`use_sensor_film=true`, `film_use_geometry=true`。

### Src 改動（~50-70 行）
| File | 改動 |
|---|---|
| `config.py` | 加 `use_sensor_film: False`, `film_use_geometry: False`（預設 false 向後相容）|
| `decoder.py` | `__init__`: 加 FiLM_MLP module（cond_dim → 2·hidden, γ identity-init）；`forward_uvp` + `forward`: trunk block 後套 FiLM；B3-b 時 concat body_distance 進 cond |
| `operator.py` | `create_picon_model` 轉送新 flags |

**body velocity 完全不被強制**：FiLM 只調制 trunk feature，輸出仍是學出來的 u,v。沒有 gate、沒有 penalty、沒有 body=0 強制 → **不重演 over-energy 機制**。

---

## 3. Config

| Variable | CEXP-002 | CEXP-039 (B3-a) | CEXP-040 (B3-b) |
|---|---|---|---|
| `use_sensor_film` | (新, false) | **true** | **true** |
| `film_use_geometry` | (新, false) | **false** | **true** |
| `use_hard_body_bc` | false | false | false |
| `bc_body_n_points` | 64 | **0** | **0** |
| `use_gradnorm` | true | true | true |
| 其餘 | — | 對齊 CEXP-002 | 對齊 CEXP-002 |

- `bc_body_n_points=0`：不再用 body soft BC（已證失敗），純測 FiLM。
- GradNorm 保留（CEXP-002 baseline 用它，FiLM 不涉 hard constraint，無 CEXP-016 的 gate-GradNorm 病態）。

---

## 4. Falsifiability Gates

| 結果 | Outcome | 解讀 |
|---|---|---|
| 任一 < 10% | ✅ A | FiLM 路線成功；比較 CEXP-039 vs 040 判斷 geometry 是否有幫助 |
| 10-30% | 🟡 B | 部分有效，比 CEXP-002 的 3.54% 差但比 geometry embedding 系列（99-490%）好 |
| > 100% | ❌ C | FiLM 也不行；conditioning 機制不是解法 |

**Ablation 對比（核心價值）**：
- CEXP-039（pure sensor）< CEXP-040（+geometry）→ geometry 明示有幫助
- CEXP-039 ≈ CEXP-040 → sensor state 已足夠，geometry 冗餘
- CEXP-040 > CEXP-039（更差）→ geometry 即使在 FiLM 下仍 adversarial（強化 Finding #8）

**額外診斷**：div L2 vs DNS 8.74、ke_pred/ref ∈ [0.8,1.2]、vorticity 圖是否還有圓柱邊緣人工剪切層。

---

## 5. Workflow

- Src 改動需 smoke test（FiLM identity-init 驗證：use_sensor_film=false 時行為與現況一致）。
- 兩個 config 並行提交（r740, 8-CPU template 同節點並行）。
- 訓練後 eval → **先確認 job ID + summary.json 路徑再讀數字**（CEXP-036/037/038 教訓）→ 判讀。

### Out-of-scope
- ❌ Stream-function（2D-only，使用者否決）
- ❌ Hard gate / soft penalty（已全失敗）
- ❌ Multi-seed（成功後才做）
- ❌ B3-b 的 geometry descriptor 用 SDF 以外的複雜編碼（先用既有 body_distance）

---

## Next Steps
Spec approval → writing-plans → 執行（src + smoke test + 兩 job 並行）。
Estimated: ~1.6 hr GPU 每個，並行 → ~1.6 hr 總。
