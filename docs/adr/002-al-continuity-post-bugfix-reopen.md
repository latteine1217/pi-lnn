# ADR-002 — AL-continuity 結論的後修補重訪（Post-Bugfix Re-evaluation）

**Date:** 2026-05-07
**Status:** Draft（待 owner accept）
**Scope:** 修補 ADR-001 §7 觸發條件 #1 對 AL-continuity 的後續行動判斷。
**Supersedes:** ADR-001 §7 條件 #1 後續行動敘述（**不**取代 §1–§6）。
**Reviewers:** latteine（Owner）；TBD external review。

---

## 1. 觸發本 ADR 的事件

### 1.1 evaluator silent regression（已 CLOSED）

`d62e698` (2026-05-03) 引入 evaluator-side denormalization regression：default 對 raw output 套 `phys = raw * std + mean`，但 model raw output 本來就是 physical 量級。所有 EXP-070~074 的評估數字被 double-scaled。

修補：
- `8647e51` (2026-05-07)：evaluator default 反轉為 identity，新增 `--apply-denormalization` opt-in flag
- Round 1–7 evaluator review-fix loop：另外發現 30 項 issue（dataset 一致性、time alignment、spectrum bin 等），全數修補 + 200 tests pass

詳見 `docs/experiment_log.md` DIAGNOSTIC section（2026-05-06~07，最終結論）。

### 1.2 修補後重跑（Round 7 evaluator）

| EXP | 重跑 KE | 原紀錄 KE (bug) | DIAG 真實值 | 重跑 div L2 | DIAG div L2 |
|---|---|---|---|---|---|
| EXP-064 | 7.80% | 7.80%（無 bug） | 7.80% | 0.184 | 0.184 |
| EXP-070 (AL pure ρ=1.0) | **6.30%** | 84.29% | 6.30% | **0.682** | 0.682 |
| EXP-070b (AL pure ρ=0.2) | **7.06%** | 84% | 7.06% | **0.735** | 0.735 |
| EXP-072 (Poisson, step 5000) | 11.76% | 85% | 11.76% | 0.670 | 0.670 |
| EXP-073 (no-sensor-phys diag) | 7.98% | 85% | 8.48% | 0.676 | 0.693 |
| EXP-074 (AL option 2) | 15.65% | 86% | 15.98% | 1.870 | 1.867 |

雙向驗證：未受 bug 影響的 EXP-064 重跑 KE 7.80% byte-aligned；受 bug 影響的 5 個 EXP 重跑全部對齊 DIAGNOSTIC 真實值（最大偏差 −0.5pp 在 reproducibility ±6% 內）。

---

## 2. ADR-001 §7 條件 #1 的判斷修正

### 2.1 原條件文字（ADR-001:104）

> **AL-continuity (EXP-070..072) 都未能將 div L2 從 0.184 降到 < 0.05** → 重新考慮 stream function reparam 或 pseudo-FVM。

### 2.2 條件本身仍成立

- AL-pure (EXP-070): div L2 = 0.682（**退步 3.7×**，遠高於 0.05 閾值）
- AL-pure ρ-lite (EXP-070b): 0.735（**退步 4.0×**）
- Poisson 對照 (EXP-072 step 5000): 0.670（**退步 3.6×**，AL/Poisson 都不降 div）
- AL option 2 (EXP-074): 1.870（**退步 10.2×**）

**觸發 #1 已啟動**：5 個 AL 系列實驗均未把 div L2 從 0.184 降到 < 0.05。

### 2.3 但後續行動的「失敗描述」不準確

bug 修補前的論述：「AL KE=84% 場崩 → AL 設計本身失敗 → 應切換 stream function reparam」。

bug 修補後的真實畫面：
- **KE 維度**：AL pure (EXP-070) KE=6.30% **優於 baseline 7.80%**；AL pure ρ-lite (EXP-070b) KE=7.06% 接近 baseline；只有 AL option 2 (EXP-074) KE=15.65% 真退步
- **div L2 維度**：所有 AL 變體均**退步 3–10×**
- **NS residual**：AL pure NS-u/v RMS = 1.58 / 1.53（vs EXP-064 0.52 / 0.51）→ **AL 實際上把 NS residual 與 div constraint 同時放鬆**

正確描述：**AL 不是 failure，是把 NS-momentum + cont 的 trade-off 從 div_L2 transferred 到沒有實質改善**。

---

## 3. 新決策（取代 ADR-001 §7 #1 後續行動）

### Decision-A: 不直接跳到 stream function reparam

ADR-001 §7 的「重新考慮 stream function reparam 或 pseudo-FVM」**保持為候選**，但**不立即啟動**。理由：

1. AL 的 KE 維度成功 → 沒有「AL 整體失敗」的證據支撐結構性 reparam
2. 5 個 AL 實驗都呈現 div L2 0.6–1.9 範圍，**收斂在某個 plateau**（不是 NaN 或數值爆炸）→ 暗示是 loss landscape 結構問題，不是公式錯誤
3. stream function reparam 與現有 `forward_uvp` 架構衝突大（ADR-001 §6 凍結清單就是這個理由）

### Decision-B: 先驗證「AL + div-aware weighting」

提案 EXP-075：在 AL pure (EXP-070 recipe) 的基礎上，把 GradNorm 的 cont task weight floor 從 0.05 提升到 0.2，並加入「per-task causal weighting only on div residual」（修正 EXP-068 失敗根因）。

- 假設：AL 把 dual variable λ 推上 clip 但 GradNorm 同時把 cont weight 拉下，造成 div constraint 在 loss landscape 上「被 AL 推、被 GradNorm 拉」對銷。
- 驗證：div L2 降到 < 0.3（baseline 0.184 的 1.6×）即視為 hypothesis 成立 → 進一步調 floor / λ_clip。
- 失敗條件：div L2 仍 ≥ 0.5 → reopen Decision-A，啟動 stream function reparam spec。

### Decision-C: EXP-072 補跑 step 10000

EXP-072 (Poisson control) ckpt 只到 step 5000（experiment_log:348 已記錄）。AL 系列都跑滿 10k，Poisson 對照不公平。**新規定：**EXP-072 必須補跑 step 10000 才能納入 AL vs Poisson 對照結論。

### Decision-D: 補完 EXP-071

ADR-001 §4 規劃了 EXP-071（3-task GradNorm + AL-cont），但實際上 EXP-071 的 ckpt 在 worktree 也找不到（DIAGNOSTIC 表格也未列）。**新規定：**EXP-071 必須補跑，才能完整檢驗 ADR-001 §4 的「AL 與 GradNorm 相容性」假設。

### Decision-E: ADR-001 §1 baseline 數字保留

EXP-064 重跑 KE 7.80%、div L2 0.184 確認，ADR-001 §1 不需修訂。

---

## 4. 對 ADR-001 其他章節的影響

| ADR-001 段落 | 是否受影響 | 說明 |
|---|---|---|
| §1 背景（EXP-064 baseline） | ❌ 不變 | 重跑 byte-aligned |
| §2 Decision-1 研究線分割 | ❌ 不變 | 與 evaluator bug 無關 |
| §3 Decision-2 Filtering 定位 | ❌ 不變 | 架構決策，與數字無關 |
| §4 Decision-3 AL-continuity 規劃 | ⚠️ Decision-D 補完 EXP-071 | 規劃本身成立，僅執行未完成 |
| §5 後續研究序 | ⚠️ #1 AL 評估後續路徑修正 | 見本 ADR Decision-B |
| §6 凍結清單 | ❌ 不變 | stream function 仍凍結（候選不啟動）|
| §7 觸發條件 #1 | ⚠️ 條件啟動但行動修正 | 見本 ADR Decision-A/B |
| §7 觸發條件 #2/#3/#4 | ❌ 不變 | 與 AL 無關 |

---

## 5. 不在本 ADR 範圍

- **EXP-075 詳細 spec**（Decision-B）— 待 owner 啟動後另寫 spec
- **EXP-071 補跑 spec**（Decision-D）— 待啟動
- **stream function reparam spec**（Decision-A 失敗時觸發）— 不寫
- **ADR-001 文本修改** — 本 ADR 為 amendment，不直接 edit ADR-001（保持 decision history）

---

## 6. 編號規範更新

- ADR-001 §8 line 117 「下一個可用：EXP-070」→ 已用至 EXP-074
- 下一個可用：**EXP-075**（Decision-B 啟動時佔用）
- EXP-071 補跑（Decision-D）保留原編號

---

## 7. Acceptance criteria

本 ADR 可從 Draft → Accepted 當：
1. Owner 確認 audit 結論（§2 條件啟動但行動修正）
2. Owner 接受 §3 五項 Decisions（A 不啟動 stream function、B AL+div-aware、C EXP-072 補 step 10k、D EXP-071 補跑、E §1 baseline 保留）
3. 本 ADR 寫進 `docs/adr/` 永久紀錄

Status: **Draft**（待 latteine accept）

---

## Appendix — Round 7 evaluator 修補摘要（reference）

修補項目（共 31 項，Round 1–7）：
- denormalization 路徑（default identity，opt-in flag）
- dataset 重建以取單一真相源 stats / SDF / split
- DNS time alignment two-tier (ULP tolerance + floor)
- spectrum bin cap `n_bins = n//2`（避免超 Nyquist）
- forcing_mode_coeff_u 加 `domain_length`
- `_add_split` schema 維持 plain mean key + `_train`/`_val` suffix（compare_experiments.py 不破壞）
- `find_dns_time_idx` 抽到 `src/pi_lnn/dns_align.py` + cross-import drift safety test
- production-style f32 sensor_time test 3 個
- T<2 / grid mismatch / `--eval-stride < 1` 等 fail-fast
- monotonic axis assert（cylinder 非均勻格保護）

200 tests pass（base 196 + 4 new safety）。詳見 `docs/experiment_log.md` DIAGNOSTIC section + `MEMORY.md` 紀錄。
