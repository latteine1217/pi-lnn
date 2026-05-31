# Design — Cylinder Hard BC + Fixed Weight (CEXP-037)

| Field | Value |
|---|---|
| Date | 2026-05-31 |
| Status | Approved |
| Owner | latteine |
| Related state | [docs/cylinder_log_v2.md](../../cylinder_log_v2.md) Finding #4, #6, #8 |
| Predecessor | CEXP-016 (hard BC + GradNorm, KE 111.6%) |

---

## 1. Goal & Background

**Goal**：以最小變更（config-only）驗證 Finding #6 的歸因——「hard BC 失敗的根因是 hard BC gate 與 GradNorm 的優化不相容」。若關掉 GradNorm 改用固定權重，hard BC 是否就不再爆炸（KE 不再 > 100%）。

**使用者目標（2026-05-31）**：物理正確性優先。CEXP-002 的 div L2 = 1.14（非 divergence-free，假解）被使用者指出「不是好結果」。使用者要求「強迫模型知道 geometry——中間有一個圓柱障礙物」。在 NS primitive-variable formulation（不用 stream-function，使用者明確要求）下，唯一的架構級 geometry enforcement 是 **hard BC output gate**（Sukumar 2022，body 內 u=v=0 機器精度）。

**Background（hard BC 失敗史）**：

| Exp | 機制 | KE | w_ns_u | 失敗原因 |
|---|---|---|---|---|
| CEXP-016 | hard BC + GradNorm + body BC 64 | 111.6% | ~2.09 | GradNorm 把 w_ns_u 推爆 |
| CEXP-017 | + 5-task GradNorm | 303.6% | 3.82 | GradNorm 更糟 |
| CEXP-021 | + SDF trunk | 174% | 1.96 | 同上 |
| CEXP-022 | + geometry tokens | 99.8% | 2.09 | 同上 |
| **CEXP-037** | **hard BC + 固定權重（no GradNorm）+ body BC 0** | **TBD** | N/A | — |

**Finding #6 核心主張**：所有 hard BC 變體 w_ns_u 最終都在 ~2，這是 gate + GradNorm 架構級不相容，非超參問題。CEXP-037 是這個歸因的**乾淨對照組**：移除 GradNorm，看 hard BC 單獨在固定權重下是否可行。

---

## 2. Architecture

**無 src 變更**。training.py 在 `use_gradnorm=false` 時走「非 GradNorm 路徑」（line 1502-1506），loss 用固定權重組合：

```
l_total = data_loss_weight · l_data
        + physics_loss_weight · l_physics      # l_physics = ns_u + ns_v + continuity_weight·cont
        + bc_loss_weight · l_bc_total          # inflow + slip（body BC=0 時無 body 項）
```

hard BC gate（decoder.py line 429-442）為 output transformation，與權重模式無關：
```
u = (φ/scale).clamp(0,1) · NN_u    # body 內 φ=0 → u=0 機器精度
v = (φ/scale).clamp(0,1) · NN_v
p = NN_p                            # p 不 gate
```

gate 由 `dataset.query_body_distance_torch(xy)` 提供（differentiable bilinear interp on SDF grid），訓練前 `set_body_bc_scale()` 注入 dataset max fluid distance。

---

## 3. Config (CEXP-037)

File: `configs/exp_cylinder_037_hardbc_fixed_weight.toml`，由 CEXP-002 派生。

| Variable | CEXP-002 | CEXP-037 | 備註 |
|---|---|---|---|
| `use_hard_body_bc` | false | **true** | Sukumar gate，架構級 geometry enforcement |
| `use_gradnorm` | true | **false** | 移除動態權重（Finding #6 元兇） |
| `bc_body_n_points` | 64 | **0** | gate 已管 body（Finding #8：恰好一個約束）|
| `data_loss_weight` | 1.0 | 1.0 | 固定 |
| `physics_loss_weight` | 0.01 | 0.01 | = CEXP-002 GradNorm init |
| `continuity_weight` | 1.0 | 1.0 | 固定 |
| `bc_loss_weight` | 0.1 | 0.1 | = CEXP-002 GradNorm init（inflow + slip）|
| 其餘 | — | 與 CEXP-002 對齊 | |

**vs CEXP-016 差異**：CEXP-016 = hard BC + GradNorm + body BC 64。CEXP-037 改 GradNorm→固定 + body BC→0。兩個變數，但 body BC=0 符合 Finding #8 且使用者明確選擇。

---

## 4. Falsifiability Gates

| KE rel-err | L_phys 軌跡 | Outcome | 解讀 |
|---|---|---|---|
| **< 20%** | 穩定不爆 | ✅ A | **Finding #6 歸因正確**：GradNorm 是 hard BC 失敗元兇 → 下一步上 augmented Lagrangian 求更佳 |
| 20-100% | 中度 | 🟡 B | GradNorm 是部分原因，固定權重改善但不足 |
| > 100% | L_phys 爆漲 | ❌ C | **Finding #6 歸因錯誤**：hard BC 失敗與 GradNorm 無關，是更深架構問題 → hard BC 路線徹底封閉 |

**額外診斷量**：
- `div L2` < 1.14（baseline）→ hard BC + physics 是否真的改善 incompressibility（使用者的核心目標）
- `body_u_max` / `body_v_max` ≈ 0 → 確認 gate 確實強制 no-slip（機器精度）
- `ke_pred/ke_ref` ∈ [0.85, 1.15] → 不 over-predict（CEXP-016 是 2.12）

---

## 5. Workflow

- **無 src 變更**，config-only。
- Lab deploy：sed 修 arrow_shards + kolmogorov dummy A/k_f（同既有 pattern）。
- Submit：`scripts/slurm/submit_exp.sh cylinder_037 configs/exp_cylinder_037_hardbc_fixed_weight.toml`
- 訓練後 eval（r740 SLURM）→ rsync summary → 依 §4 判讀 → 更新 cylinder_log_v2.md。
- **監控重點**：L_phys 是否在訓練中爆漲（CEXP-036 的 SOAP+RAR 教訓）。hard BC gate 壓 physics 梯度，固定權重下 L_phys 若仍失控，即 ❌ C。

### Out-of-scope
- ❌ Augmented Lagrangian（若 ✅ A 才進下一階段）
- ❌ Stream-function formulation（使用者明確要求保留 NS primitive variable）
- ❌ Multi-seed（single-seed first pass）

---

## Next Steps

Spec approval → `superpowers:writing-plans` → implementation plan → 執行。

Estimated: ~1.6 hr GPU（與既有 cylinder 實驗同）。
