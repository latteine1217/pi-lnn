# Design — Cylinder Trunk SDF Input (Stage 2 Option A)

| Field | Value |
|---|---|
| Date | 2026-05-24 |
| Status | Approved（user sign-off §1+2 sections; §3-5 compact-confirmed; awaiting spec-file review）|
| Owner | latteine |
| Related state | [docs/cylinder_log_v2.md](../../cylinder_log_v2.md) Finding #4, Open Questions Stage 2 |
| Predecessor spec | [Stage 1 enabling conditions](2026-05-23-cylinder-hard-bc-enabling-conditions-design.md) — all 3 hypotheses falsified |

---

## 1. Goal & Scope

**Goal**: 給 PI-CON trunk net 加 SDF `φ(x,y)` raw scalar input，移除 hard BC output gate，驗證 trunk-level geometry awareness 是否解決 Stage 1 揭露的 hard BC catastrophic over-predict 機制。

**Background**: Stage 1 (CEXP-016/017/018/019) 全 ❌：hard BC + standard PI-CON architecture 在 K=100 sparse cylinder Re=10031 regime 下 fundamental incompatibility。Root cause hypothesis: **trunk net 完全沒有 geometry awareness** — hard BC 是 output post-hoc gate, NN_u 不知道 boundary 在哪 → wake 區 over-compensation → GradNorm pathology。

**Scope (minimal, 1 config)**:
- CEXP-020 = CEXP-002 baseline + `use_body_distance_feature=true` + `use_hard_body_bc=false`
- Single seed=42, 1 SLURM job ~1.6 hr
- Required src patches ~30-40 lines, 3 files

**Success criteria (Moderate gates per user 2026-05-24 decision)**:
- **KE < 10%** → ✅ Option A confirmed — trunk SDF is viable alternative to hard BC; Stage 2 paper claim 「PI-CON 在 non-periodic + geometry case 下需要 trunk-level geometry awareness」
- **KE 10-30%** → 🟡 partial — trunk SDF adds value but not full recovery; consider Options B (Fourier-on-φ) or D (per-layer gate)
- **KE > 30%** → ❌ Option A broken — root cause hypothesis wrong; escalate to Options E (cross-attn geometry tokens) / F (geometry-conditioned hypernetwork) future paper material

**Non-goals**:
- 不做 multi-seed n=3/5（single-seed first-pass）
- 不做 hard BC + SDF input 並用 (user rejected dual mode)
- 不做 Options B/D/E/F（future paper material per user 2026-05-24）

---

## 2. Architecture & Src Changes

### Trunk input dimensional change

```
Current: query_in = spatial_dim + temporal_dim + d_time + 8                  # base_feat dim
New:     query_in = spatial_dim + temporal_dim + d_time + 8 + 1              # + raw φ scalar (post-Fourier)
```

`spatial_dim` 仍由 `fourier_embed_dim` 決定（x,y 維持 Fourier encoding，**不動** `FourierEmbs(input_dim=2)` 避免改 Kolmogorov 相容性）。

`φ` 以 **raw scalar** (post-Fourier concat) 進入 trunk MLP — 不對 φ 做 Fourier encoding。User decision 2026-05-24「Raw scalar concat」確認。

### Src patches (estimated ~30-40 lines, 5 patch sites)

| File | Change | Est. lines |
|---|---|---|
| `src/pi_con/config.py` (line 9 `DEFAULT_PICON_ARGS`) | Add `"use_body_distance_feature": False` line | +2 |
| `src/pi_con/operator.py` `create_picon_model` (line ~214, calls `DeepONetCfCDecoder.__init__`) | Forward flag `use_body_distance_feature=cfg.get("use_body_distance_feature", False)` to constructor | +3 |
| `src/pi_con/decoder.py` (line ~19-130) | (a) Add `use_body_distance_feature: bool = False` param to `__init__`; (b) modify `query_in` dim calc (+1 if flag true); (c) in `forward`/`forward_uvp`/`forward_uv` when flag true: compute `phi = body_distance_fn(xy).unsqueeze(-1)` and concat into `base_inputs` list before `torch.cat` | +15-25 |
| `src/pi_con/operator.py` `make_picon_model_fn{,_uvp}` (line ~286-360) | If `use_body_distance_feature=True` AND `body_distance_fn=None` → `raise ValueError` (same pattern as `use_hard_body_bc` check) | +5 |
| `src/pi_con/training.py` (where `body_distance_fn` is conditional on `use_hard_body_bc`) | Make condition: `_pass_bd_fn = use_hard_body_bc OR use_body_distance_feature` — wire differentiable SDF regardless of which mode requests it | +3-5 |

**No new code paths** — reuses existing `dataset.query_body_distance_torch` (autograd-friendly bilinear interp, same as hard BC uses since CEXP-016 work). **Autograd chain rule integrity** confirmed.

### Output transformation

**NO gate**（because `use_hard_body_bc=false`）。Trunk 直接輸出 NN_u, NN_v, NN_p。

Body 內 u/v 不會嚴格為 0 但：
- Sensor 在 wake (集中 x > 0.10)，body 點不在 sensor MSE supervision 範圍
- Soft body BC loss (`bc_body_n_points=64`, `bc_loss_weight=0.1`) 仍保留 — body 點 u=v=0 soft penalty
- Trunk 透過 φ input 學「離 body 越近, velocity magnitude 越小」inductive bias
- 純 soft constraint, 不像 hard BC 機器精度保證, 但避免 over-compensation 機制

### Ckpt incompatibility

`query_in` dim 變動 (4 → 5)，CEXP-002 ckpt 不能 resume。**Cold start** 訓練 10k iter。預估 wall time ~1.6 hr（與 CEXP-016/017/018 同量級）。

---

## 3. Config Matrix

| Config | Hard BC | SDF input | Sampling | GradNorm tasks | iter | seed | 用途 |
|---|---|---|---|---|---|---|---|
| CEXP-002 (legacy ref, no rerun) | ❌ | ❌ | random | 4-task | 10k | 42 | reference baseline KE 3.54% |
| **CEXP-020** | ❌ | **✅** | random | 4-task | 10k | 42 | Stage 2 Option A target |

**Single variable** vs CEXP-002: `use_body_distance_feature=true`. **Not** derived from CEXP-016 (we abandon hard BC path entirely).

Config filename: `configs/exp_cylinder_020_trunk_sdf_input.toml`

### Hyperparameters (對齊 CEXP-002 unless noted)

- `use_hard_body_bc = false` (對齊 CEXP-002, 不啟用 hard BC)
- **`use_body_distance_feature = true`** (新 key, 由本 spec 啟用)
- `bc_loss_weight = 0.1` (soft body/inflow/slip BC 保留)
- `bc_n_points = 64`, `bc_body_n_points = 64`, `bc_slip_n_points = 32`, (no `bc_outlet_n_points`)
- `physics_collocation_strategy = "random"`
- `gradnorm_init_weights = [1.0, 0.01, 0.01, 0.01]` (4-task: data, ns_u, ns_v, cont — BC weight fixed 0.1 outside GradNorm)
- `d_model = 256`, `iterations = 10000`, `seed = 42`, `device = "cuda"`
- `kolmogorov_A = 1e-6`, `kolmogorov_k_f = 2.0` (deployment workaround for ForcingPrior cylinder regression, applied via lab-only sed)
- `arrow_shards`: lab path applied via sed, same as CEXP-017/018/019

---

## 4. Falsifiability Gates

### Single-config decision tree

| Outcome | Threshold | Interpretation | Next Stage |
|---|---|---|---|
| ✅ **A** | KE < 10% | Option A confirmed — trunk SDF input is viable architectural alternative to hard BC | Stage 3: paper claim writing + multi-seed n=3 confirmation; consider sensor placement sweep |
| 🟡 **B** | KE 10-30% | Partial — trunk SDF helps (vs CEXP-016 catastrophic 111%) but not full baseline-level recovery | Stage 3 candidates: Option B (Fourier-on-φ multi-scale) or Option D (per-layer trunk gate). Compare against Option A as ablation |
| ❌ **C** | KE > 30% | Option A broken — root cause hypothesis (trunk geometry awareness) wrong, deeper architectural issue exists | Re-diagnose: Options E (cross-attn geometry tokens) / F (geometry-conditioned hypernetwork). Possibly outside this paper's scope. |

### Additional diagnostic metrics (per spec §1 hypothesis)

- `ke_pred / ke_ref` — should be ~1.0 if trunk geometry awareness fixes over-compensation
- `w_ns_u` final — should be < 0.5 (i.e., not GradNorm pathology like CEXP-016's 2.09 or CEXP-017's 3.82)
- `body_u_max` / `body_v_max` — without hard BC gate, body interior 不再嚴格 0；應 < 0.05 m/s (i.e., soft penalty 有效 + trunk 學到 "near-body velocity small") 才算 trunk SDF awareness 真生效

### Stop loss

訓練中**不**啟動 early stop。Stop loss 是 **spec-level**：

- 若 KE > 50% (catastrophic, 同 CEXP-016 量級) → 不繼續 follow-up，直接寫進 v2 log 並回 brainstorming
- 若 ckpt 不存或 NaN → 立即 debug (probably src patch bug)

---

## 5. Implementation Prerequisites & Workflow

### Pre-requisites

**Src-level**:
- `use_body_distance_feature` key 落到 `DEFAULT_PICON_ARGS` (CEXP-007 silent-ignore bug **fix as part of this spec's src patch**)
- `body_distance_fn` differentiable SDF pathway 已備（CEXP-016 hard BC work 已驗證 autograd 完整）
- Reuse cylinder dataset SDF preCompute (cylinder_dataset.py:243-251)

**Lab deployment** (已備，與 Stage 1 相同):
- Lab `/home/junyi/RealPDEBench/.../data-00000-of-00092.arrow` 已存在
- Lab `data/cylinder_sensors/sensors_qrpivot_K100_cylinder_Re10031_{json,npz}` 已存在
- Lab `.venv` Python 3.12 + torch 2.7.1+cu118
- SLURM r740 partition acmt20 RTX 3090 已驗證

### Workflow

```
1. 本地: src 3-file patch (~30-40 line) + commit + push
2. 本地: 寫 configs/exp_cylinder_020_trunk_sdf_input.toml + commit + push
3. Lab: git pull
4. Lab: sed deployment edits on new config (arrow_shards + kolmogorov_A/k_f dummies, same as Stage 1)
5. Lab: scripts/slurm/submit_exp.sh cylinder_020 configs/exp_cylinder_020_trunk_sdf_input.toml
6. Wait: ~1.6 hr SLURM training
7. Lab: nohup evaluate_cylinder.py → summary.json (~5 min)
8. Rsync artifact 回本地
9. 對照 §4 decision tree 判讀 outcome
10. 更新 cylinder_log_v2.md ([INDEX] CEXP-020 row + [RECORD] detail + [STATE] Surprise Findings #5 + [STATE] Open Questions Stage 3 plan)
11. Commit + push v2 log update
```

### Out-of-scope (explicit)

- ❌ Multi-seed n=3/5 (single seed first-pass)
- ❌ Hard BC + SDF input dual mode (user rejected 2026-05-24)
- ❌ Option B (Fourier-on-φ multi-scale) — only if Option A is 🟡 partial
- ❌ Option D (per-layer trunk gate) — only if Option A is 🟡 partial
- ❌ Option E (cross-attn geometry tokens) — future paper
- ❌ Option F (geometry-conditioned hypernetwork) — future paper
- ❌ CLAUDE.md READ_PROTOCOL update (per user prior decision)
- ❌ Re=1781 / cross-Re follow-ups (out of cylinder generalization section scope)

### Spec deliverables

| Item | Path |
|---|---|
| Design doc (本 spec) | `docs/superpowers/specs/2026-05-24-cylinder-trunk-sdf-input-design.md` |
| Src patches | `src/pi_con/config.py`, `src/pi_con/operator.py`, `src/pi_con/decoder.py`, `src/pi_con/training.py` |
| Config file | `configs/exp_cylinder_020_trunk_sdf_input.toml` |
| Updated state | `docs/cylinder_log_v2.md` (results 寫回 + Finding #5 added) |

### Total scope estimate

| Phase | Wall time | GPU |
|---|---|---|
| Spec write + review | 10 min | — |
| Src patches + config + commit | 30 min | — |
| Lab deployment | 5 min | — |
| SLURM train | ~1.6 hr | RTX 3090 |
| Eval + rsync + log update + commit | 30 min | — |
| **Total** | **~3 hr** | **~1.6 GPU-hr** |

---

## Next Steps

After spec approval: **invoke `superpowers:writing-plans` skill** (not other implementation skills) to draft detailed implementation plan with file-level steps, exact sed/Edit commands, falsifiability check commands, and rollback procedures.
