# Design — Cylinder Zhu-style Body Penalty α-rebalancing (CEXP-038)

| Field | Value |
|---|---|
| Date | 2026-05-31 |
| Status | Approved |
| Owner | latteine |
| Related | [literature survey](../../research/2026-05-31-hard-constraint-sparse-sensor-literature.md), cylinder_log_v2.md Finding #6/#8 |
| Predecessor | CEXP-016 (hard BC+GradNorm 111%), CEXP-037 (hard BC+fixed 283%) |
| Source paper | Zhu, Chen, Deng, Bian 2025, Acta Mechanica Sinica, DOI 10.1007/s10409-025-25273-x (arXiv:2503.24074) |

---

## 1. Goal & Background

**Goal**：驗證 Zhu 2025 的歸因——cylinder hard BC 失敗的真因是 **body no-slip penalty 的權重 α 太弱**，而非 hard gate 或 GradNorm。Zhu 明確記錄：圓柱後方 wake 速度低 → body penalty `α·φ·(u−U)` 相對 fluid NS residual 太小 → 形成「competitive relationship」→ 圓柱後緣 over-energy。修法是提高 α 到 ~10。

**Background（失敗史）**：
| Exp | 機制 | body α_eff | KE | ke_pred/ref |
|---|---|---|---|---|
| CEXP-002 | soft inflow BC, no body BC | — | 3.54% | 1.01（但 div=1.14 over-smoothed 假解）|
| CEXP-016 | hard gate + GradNorm + body soft BC | 0.1 | 111.6% | 2.12 |
| CEXP-037 | hard gate + fixed weight | 0.1 | 283% | 3.83 |
| **CEXP-038** | **soft body penalty α=10（Zhu），no gate** | **10** | TBD | — |

**為何選 soft penalty 而非 hard gate**：literature survey 已驗證 (a) Zhu 用 soft body-fraction penalty（非 output gate）；(b) arXiv:2411.08122 發現純 hard BC 在複雜邊界 degrade interior，改 soft 反而更準。我們的 hard gate 路線（CEXP-016/021/022/037）已全失敗。

**程式碼事實（查證結論）**：
- collocation 撒點 `cylinder_dataset.sample_physics_points` **僅限 fluid 域**（body 已排除）→ 無法在 fluid collocation 上嵌 Zhu 的 `(1−φ)·NS + α·φ·(u−U)`（φ 恆 0）。
- 因此 Zhu-style 在本 codebase 的可行形式 = **在 body 內額外撒點施加 `α·u²` penalty**（既有 `bc_body_n_points` 機制）。
- 既有 inflow/body/slip BC 全累加進同一 `l_bc_total`，**統一乘 `bc_loss_weight`**；body 項無獨立權重 → config-only 無法只放大 body α（會連 inflow 一起放大、破壞 CEXP-002 baseline）。**需小改 src 加獨立 body weight。**

---

## 2. Architecture / Src 改動（最小，~15 行）

**File 1: `src/pi_con/config.py`** — 新增 key（預設 1.0 = 不影響既有行為）：
```python
"bc_body_weight": 1.0,   # body no-slip 項相對其他 BC 的額外乘數（Zhu α-rebalancing）；
                         # α_eff = bc_loss_weight × bc_body_weight。預設 1.0 向後相容。
```

**File 2: `src/pi_con/training.py`** line ~1289-1293（body BC 累加處）：
```python
# 改動前：
l_bc_total = l_bc_total + torch.mean((_u_body - _u_zero_norm)**2) + torch.mean((_v_body - _v_zero_norm)**2)
# 改動後：
_bc_body_w = float(args.get("bc_body_weight", 1.0))
l_bc_total = l_bc_total + _bc_body_w * (torch.mean((_u_body - _u_zero_norm)**2) + torch.mean((_v_body - _v_zero_norm)**2))
```
- inflow（line 1270）/ slip（line 1313）不變 → α 提高只作用於 body。
- 最終 body 等效權重 α_eff = `bc_loss_weight × bc_body_weight` = 0.1 × 100 = **10**（Zhu 值）。

**向後相容性**：`bc_body_weight` 預設 1.0，所有既有 config 行為不變。

---

## 3. Config (CEXP-038)

File: `configs/exp_cylinder_038_zhu_body_penalty.toml`，由 CEXP-002 派生。

| Variable | CEXP-002 | CEXP-038 | 備註 |
|---|---|---|---|
| `use_hard_body_bc` | false | false | soft penalty，不用 gate |
| `bc_body_n_points` | 64 | 64 | body 內撒點施加 no-slip |
| `bc_body_weight` | (新, 1.0) | **100** | α_eff = 0.1×100 = 10（Zhu）|
| `use_gradnorm` | true | **false** | 固定權重，純測 α 效果（排除 GradNorm 干擾）|
| `bc_loss_weight` | 0.1 | 0.1 | inflow/slip 維持 0.1 |
| 其餘 | — | 對齊 CEXP-002 | |

---

## 4. Falsifiability Gates

| KE rel-err | ke_pred/ref | Outcome | 解讀 |
|---|---|---|---|
| **< 20%** | [0.8, 1.2] | ✅ A | **Zhu 對**：body penalty α 是真因，夠強就壓住 over-energy → 可進一步調 α / multi-seed |
| 20-100% | 1.2-2.5 | 🟡 B | α 部分有效但不足；可再加大 α 或加 body collocation 點數 |
| > 100% | > 2.5 | ❌ C | **α 不是真因**：body soft penalty 本質不足（強化 Finding #8：body 區無 sensor 是更深問題）→ 回 stream-function |

**額外診斷**：
- div L2 vs DNS ref 8.74：CEXP-002 的 1.14 是 over-smoothed 假解；CEXP-038 若 div 接近 8.74 = 物理更正確。
- L_phys 訓練軌跡是否穩定（無 CEXP-036 式爆漲）。
- α-sweep 預留：若 α=10（weight 100）結果介於 🟡，下一輪試 α=5（weight 50）或 α=20（weight 200）。

---

## 5. Workflow

- Src 改動：config.py + training.py（~15 行），需 smoke test。
- Lab deploy：sed 修 arrow_shards + kolmogorov dummy。
- Submit：`scripts/slurm/submit_exp.sh cylinder_038 configs/exp_cylinder_038_zhu_body_penalty.toml`
- 訓練後 eval → **先確認 job ID + summary.json 路徑再讀數字**（CEXP-036/037 教訓）→ 判讀 → 更新 log。

### Out-of-scope
- ❌ Stream-function（A 路線；若 CEXP-038 ❌C 才考慮）
- ❌ Hard gate（已全失敗）
- ❌ α-sweep multi-run（先單點 α=10 first pass）
- ❌ Multi-seed（成功後才做）

---

## Next Steps
Spec approval → writing-plans → 執行（含 src smoke test + lab submit）。
Estimated: ~1.6 hr GPU。
