# Cylinder Trunk SDF Input (Stage 2 Option A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 給 PI-CON trunk net 加 SDF `φ(x,y)` raw scalar input (`query_in` dim 4→5)，移除 hard BC gate，驗證 trunk-level geometry awareness 是否解決 Stage 1 揭露的 hard BC catastrophic over-predict。

**Architecture:** 4-file src patch (`config.py` + `operator.py` + `decoder.py` + `training.py`, ~30-40 lines) 新增 `use_body_distance_feature` flag。`body_distance_fn` differentiable SDF lookup 已存在 (hard BC 已用), 重用即可 — 只需 (1) 把 flag 加入 DEFAULT_PICON_ARGS、(2) 在 trunk MLP 前 concat raw φ scalar、(3) 拓寬 `body_distance_fn` 啟用條件 (hard BC OR SDF input)。然後跑單個 SLURM job (CEXP-020 = CEXP-002 base + flag true)。

**Tech Stack:** Python 3.12, torch 2.7.1+cu118, SLURM r740 partition (acmt20 RTX 3090), uv venv.

**Spec reference:** [`docs/superpowers/specs/2026-05-24-cylinder-trunk-sdf-input-design.md`](../specs/2026-05-24-cylinder-trunk-sdf-input-design.md)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/pi_con/config.py` | Modify line ~175 | Add `"use_body_distance_feature": False` to DEFAULT_PICON_ARGS |
| `src/pi_con/operator.py` | Modify lines 19-99 (LiquidOperator.__init__) + lines 214-265 (create_picon_model) + lines 286-323 (make_picon_model_fn) + lines 326-367 (make_picon_model_fn_uvp) | Forward flag config → LiquidOperator → DeepONetCfCDecoder; update `use_bd` condition to OR with new flag |
| `src/pi_con/decoder.py` | Modify lines 26-181 (__init__) + lines 183-322 (forward_uvp) + lines 324-431 (forward) | Add flag param; query_in dim +1 if true; in forward paths concat `phi = body_distance(xy)` to base_inputs |
| `src/pi_con/training.py` | Modify lines 136-163 (body_distance_fn wiring) | Compute `_need_body_distance_fn` = use_hard_body_bc OR use_body_distance_feature; replicate condition at 7 sites where `_use_hard_body_bc` gates `body_distance_fn`|
| `configs/exp_cylinder_020_trunk_sdf_input.toml` | Create | CEXP-002 base + `use_body_distance_feature=true` + `use_hard_body_bc=false` + new artifacts_dir + header |
| `docs/cylinder_log_v2.md` | Modify | After CEXP-020 result: [INDEX] row + [RECORD] detail + [STATE] Surprise Findings #5 + Open Questions Stage 3 |

---

## Task 1: Add `use_body_distance_feature` to DEFAULT_PICON_ARGS

**Files:**
- Modify: `src/pi_con/config.py` (around line 175)

- [ ] **Step 1.1: Read current DEFAULT_PICON_ARGS around `use_hard_body_bc` to find exact context**

```bash
grep -n -A 5 'use_hard_body_bc' src/pi_con/config.py | head -10
```

- [ ] **Step 1.2: Use Edit to insert `use_body_distance_feature` key immediately AFTER `use_hard_body_bc` block**

`old_string`:
```
    "use_hard_body_bc": False,           # cylinder hard body BC（Sukumar 2022 風格）：
                                          # u = (φ/scale).clamp(0,1) · NN，物理保證 body 內 u=v=0。
                                          # 取代有 detach bug 的 distance-as-input feature。
                                          # True 會改 model architecture（query_in +1），ckpt 不相容
                                          # 僅 cylinder dataset 真正需要；kolmogorov 為 dummy 1.0
```

`new_string`:
```
    "use_hard_body_bc": False,           # cylinder hard body BC（Sukumar 2022 風格）：
                                          # u = (φ/scale).clamp(0,1) · NN，物理保證 body 內 u=v=0。
                                          # 取代有 detach bug 的 distance-as-input feature。
                                          # True 會改 model architecture（query_in +1），ckpt 不相容
                                          # 僅 cylinder dataset 真正需要；kolmogorov 為 dummy 1.0
    "use_body_distance_feature": False,  # cylinder trunk SDF input feature (Stage 2 Option A)：
                                          # query_in = [x, y, t, c, φ(x,y)] (dim 4→5)
                                          # raw scalar concat post-Fourier, no encoding on φ。
                                          # 與 use_hard_body_bc 獨立，可同時用 (但 Stage 2 spec 推薦只用此 flag, hard BC off)
                                          # 改 model architecture（query_in +1），ckpt 不相容。
                                          # 僅 cylinder dataset 有 SDF; kolmogorov 啟用會 raise。
```

- [ ] **Step 1.3: Verify**

```bash
grep -A 3 'use_body_distance_feature' src/pi_con/config.py | head -5
```

Expected: 顯示 `"use_body_distance_feature": False,` 加上 comment。

---

## Task 2: Add flag to LiquidOperator.__init__ + create_picon_model wiring

**Files:**
- Modify: `src/pi_con/operator.py` (lines 19-99 LiquidOperator.__init__) + (lines 214-265 create_picon_model)

- [ ] **Step 2.1: Add param to LiquidOperator.__init__ signature (after `use_hard_body_bc`)**

`old_string`:
```python
        use_hard_body_bc: bool = False,
        decoder_attention_heads: int = 1,
```

`new_string`:
```python
        use_hard_body_bc: bool = False,
        use_body_distance_feature: bool = False,
        decoder_attention_heads: int = 1,
```

- [ ] **Step 2.2: Store flag in `__init__` body (after `self.use_hard_body_bc` line ~56)**

`old_string`:
```python
        self.use_hard_body_bc = bool(use_hard_body_bc)
        self.spatial_encoder = SpatialSetEncoder(
```

`new_string`:
```python
        self.use_hard_body_bc = bool(use_hard_body_bc)
        # SDF-as-trunk-input (Stage 2 Option A)：query 多 1 維 raw φ scalar post-Fourier concat。
        # 與 hard BC 獨立; 兩者都需要 body_distance_fn 在 training.py 傳入 (使用差異見 decoder.forward*)。
        self.use_body_distance_feature = bool(use_body_distance_feature)
        self.spatial_encoder = SpatialSetEncoder(
```

- [ ] **Step 2.3: Forward flag to DeepONetCfCDecoder constructor (line ~95)**

`old_string`:
```python
            use_hard_body_bc=use_hard_body_bc,
            decoder_attention_heads=decoder_attention_heads,
```

`new_string`:
```python
            use_hard_body_bc=use_hard_body_bc,
            use_body_distance_feature=use_body_distance_feature,
            decoder_attention_heads=decoder_attention_heads,
```

- [ ] **Step 2.4: Forward flag from config in create_picon_model (line ~261)**

`old_string`:
```python
        use_hard_body_bc=bool(cfg.get("use_hard_body_bc", False)),
        decoder_attention_heads=int(cfg.get("decoder_attention_heads", 1)),
```

`new_string`:
```python
        use_hard_body_bc=bool(cfg.get("use_hard_body_bc", False)),
        use_body_distance_feature=bool(cfg.get("use_body_distance_feature", False)),
        decoder_attention_heads=int(cfg.get("decoder_attention_heads", 1)),
```

- [ ] **Step 2.5: Verify**

```bash
grep -n 'use_body_distance_feature' src/pi_con/operator.py | head -5
```

Expected: 3 lines (signature, store, decoder forward, create_picon_model 各一處).

---

## Task 3: Add flag to DeepONetCfCDecoder.__init__ + modify query_in dim

**Files:**
- Modify: `src/pi_con/decoder.py` (lines 26-181 __init__)

- [ ] **Step 3.1: Add param to __init__ signature (after `use_hard_body_bc: bool = False,` around line 45)**

`old_string`:
```python
        use_hard_body_bc: bool = False,
        decoder_attention_heads: int = 1,
```

`new_string`:
```python
        use_hard_body_bc: bool = False,
        use_body_distance_feature: bool = False,
        decoder_attention_heads: int = 1,
```

- [ ] **Step 3.2: Store flag in __init__ body (after `self.use_hard_body_bc = ...` around line 75)**

`old_string`:
```python
        self.use_hard_body_bc = bool(use_hard_body_bc)
        self.fourier_harmonics = int(fourier_harmonics)
```

`new_string`:
```python
        self.use_hard_body_bc = bool(use_hard_body_bc)
        # Stage 2 Option A: SDF input feature
        # query = [x, y, t, c, φ(x,y)] post-Fourier concat (raw scalar, no encoding on φ).
        # φ via dataset.query_body_distance_torch (differentiable, autograd-friendly).
        # 與 hard BC 獨立 — 兩者皆需 caller 傳入 body_distance；hard BC 在 output gate,
        # 此 flag 在 trunk input。Stage 2 spec 推薦只用此 flag, hard BC=False。
        self.use_body_distance_feature = bool(use_body_distance_feature)
        self.fourier_harmonics = int(fourier_harmonics)
```

- [ ] **Step 3.3: Modify query_in dim calc (line 104)**

`old_string`:
```python
        # query_in 不含 body_distance—— hard BC 是 output transformation，不是 input feature。
        query_in = spatial_dim + temporal_dim + d_time + 8
```

`new_string`:
```python
        # query_in 預設不含 body_distance（hard BC 是 output transformation）。
        # 但 Stage 2 Option A 啟用 use_body_distance_feature=True 時, query_in +1 (raw φ scalar)。
        query_in = spatial_dim + temporal_dim + d_time + 8 + (1 if self.use_body_distance_feature else 0)
```

- [ ] **Step 3.4: Verify**

```bash
grep -n 'use_body_distance_feature\|query_in =' src/pi_con/decoder.py | head -10
```

Expected: 顯示 (1) signature param, (2) self.use_body_distance_feature = ..., (3) query_in = ... + (1 if ...)

---

## Task 4: Modify decoder.forward_uvp — concat φ to base_inputs

**Files:**
- Modify: `src/pi_con/decoder.py` (lines 183-322 forward_uvp)

- [ ] **Step 4.1: Read current base_inputs construction in forward_uvp (around line 227-234)**

```bash
sed -n '225,240p' src/pi_con/decoder.py
```

- [ ] **Step 4.2: Modify base_inputs construction to include φ when flag true**

`old_string`:
```python
        # ── c-conditional：批次化 c=0,1,2，flatten 成 [3N, ...] ──────────
        base_inputs = [pos_enc]
        if self.use_temporal_anchor:
            base_inputs.append(temporal_phase_anchor(
                t_q.unsqueeze(-1), self.T_total, self.temporal_anchor_harmonics
            ))
        base_inputs.append(time_e)
        # NOTE: hard body BC 不在這裡 concat distance（output transformation, 在 return 前）
        base_feat = torch.cat(base_inputs, dim=-1)                              # [N, query_in - 8]
```

`new_string`:
```python
        # ── c-conditional：批次化 c=0,1,2，flatten 成 [3N, ...] ──────────
        base_inputs = [pos_enc]
        if self.use_temporal_anchor:
            base_inputs.append(temporal_phase_anchor(
                t_q.unsqueeze(-1), self.T_total, self.temporal_anchor_harmonics
            ))
        base_inputs.append(time_e)
        # Stage 2 Option A: SDF trunk input — concat raw φ scalar (post-Fourier, no encoding)
        # 與 hard BC（output gate, return 前處理）獨立, 在 trunk MLP 之前注入幾何感知。
        if self.use_body_distance_feature:
            if body_distance is None:
                raise ValueError(
                    "use_body_distance_feature=True 但 forward_uvp() body_distance=None；"
                    "請傳入 dataset.query_body_distance_torch(xy) 結果（differentiable）。"
                )
            phi_feat = body_distance.reshape(-1, 1)                              # [N, 1]
            base_inputs.append(phi_feat)
        # NOTE: hard body BC 不在這裡 concat distance（output transformation, 在 return 前）
        base_feat = torch.cat(base_inputs, dim=-1)                              # [N, query_in - 8]
```

- [ ] **Step 4.3: Verify**

```bash
grep -n 'use_body_distance_feature\|phi_feat' src/pi_con/decoder.py | head -10
```

Expected: At least 1 occurrence each of `self.use_body_distance_feature` (init) and `phi_feat` (forward_uvp).

---

## Task 5: Modify decoder.forward — concat φ to trunk_inputs

**Files:**
- Modify: `src/pi_con/decoder.py` (lines 324-431 forward)

- [ ] **Step 5.1: Read current trunk_inputs construction in forward (around lines 345-353)**

```bash
sed -n '343,355p' src/pi_con/decoder.py
```

- [ ] **Step 5.2: Modify trunk_inputs construction to include φ when flag true**

`old_string`:
```python
        emb_c = self.component_emb(c)
        trunk_inputs = [pos_enc]
        if self.use_temporal_anchor:
            trunk_inputs.append(temporal_phase_anchor(t_q.unsqueeze(-1), self.T_total, self.temporal_anchor_harmonics))
        # 注意：body_distance 在 emb_c 之前 concat，與 forward_uvp 的 base_feat
        # 結構一致（base_feat 含 body_distance，再跟 emb_c concat）。
        trunk_inputs.append(time_e)
        # NOTE: hard body BC 是 output transformation，不在這裡 concat distance
        trunk_inputs.append(emb_c)
        trunk_in_cat = torch.cat(trunk_inputs, dim=-1)
```

`new_string`:
```python
        emb_c = self.component_emb(c)
        trunk_inputs = [pos_enc]
        if self.use_temporal_anchor:
            trunk_inputs.append(temporal_phase_anchor(t_q.unsqueeze(-1), self.T_total, self.temporal_anchor_harmonics))
        # 注意：body_distance 在 emb_c 之前 concat，與 forward_uvp 的 base_feat
        # 結構一致（base_feat 含 body_distance，再跟 emb_c concat）。
        trunk_inputs.append(time_e)
        # Stage 2 Option A: SDF trunk input feature (與 forward_uvp 對稱)。
        if self.use_body_distance_feature:
            if body_distance is None:
                raise ValueError(
                    "use_body_distance_feature=True 但 forward() body_distance=None；"
                    "請傳入 dataset.query_body_distance_torch(xy) 結果（differentiable）。"
                )
            trunk_inputs.append(body_distance.reshape(-1, 1))                    # [N, 1] raw φ
        # NOTE: hard body BC 是 output transformation，不在這裡 concat distance
        trunk_inputs.append(emb_c)
        trunk_in_cat = torch.cat(trunk_inputs, dim=-1)
```

- [ ] **Step 5.3: Verify trunk_inputs in forward now has flag branch**

```bash
sed -n '343,365p' src/pi_con/decoder.py | grep -A 2 'use_body_distance_feature'
```

Expected: 找到 if-branch + raise + append.

---

## Task 6: Update operator.py make_picon_model_fn{,_uvp} use_bd condition

**Files:**
- Modify: `src/pi_con/operator.py` (lines 286-323 + 326-367)

- [ ] **Step 6.1: Update make_picon_model_fn `use_bd` condition (line ~307-321)**

`old_string`:
```python
    use_bd = bool(getattr(net, "use_hard_body_bc", False))
    if use_bd and body_distance_fn is None:
        raise ValueError(
            "model use_hard_body_bc=True 但 make_picon_model_fn() 沒收到 body_distance_fn"
        )

    def model_fn(xyt: torch.Tensor, c: int) -> torch.Tensor:
        xyt_d = xyt.to(net_device)
        xy_d = xyt_d[:, :2]
        t_q_d = xyt_d[:, 2]
        c_t = torch.full((xyt_d.shape[0],), c, dtype=torch.long, device=net_device)
        bd = body_distance_fn(xy_d) if use_bd else None
        return net.query_decoder(
            xy_d, t_q_d, c_t, h_states, s_time, sensor_pos, body_distance=bd,
        ).to(xyt.device)

    return model_fn
```

`new_string`:
```python
    # Stage 2: body_distance_fn 對 hard BC (output gate) 或 SDF input feature (trunk concat) 都需要。
    _need_bd = bool(getattr(net, "use_hard_body_bc", False)) or bool(getattr(net, "use_body_distance_feature", False))
    if _need_bd and body_distance_fn is None:
        raise ValueError(
            "model use_hard_body_bc=True 或 use_body_distance_feature=True 但 make_picon_model_fn() 沒收到 body_distance_fn"
        )

    def model_fn(xyt: torch.Tensor, c: int) -> torch.Tensor:
        xyt_d = xyt.to(net_device)
        xy_d = xyt_d[:, :2]
        t_q_d = xyt_d[:, 2]
        c_t = torch.full((xyt_d.shape[0],), c, dtype=torch.long, device=net_device)
        bd = body_distance_fn(xy_d) if _need_bd else None
        return net.query_decoder(
            xy_d, t_q_d, c_t, h_states, s_time, sensor_pos, body_distance=bd,
        ).to(xyt.device)

    return model_fn
```

- [ ] **Step 6.2: Same update for make_picon_model_fn_uvp (line ~346-367)**

`old_string`:
```python
    use_bd = bool(getattr(net, "use_hard_body_bc", False))
    if use_bd and body_distance_fn is None:
        raise ValueError(
            "model use_hard_body_bc=True 但 make_picon_model_fn_uvp() 沒收到 body_distance_fn"
        )

    def model_fn_uvp(xyt: torch.Tensor) -> torch.Tensor:
        xyt_d = xyt.to(net_device)
        xy_d = xyt_d[:, :2]
        t_q_d = xyt_d[:, 2]
        bd = body_distance_fn(xy_d) if use_bd else None
```

`new_string`:
```python
    # Stage 2: body_distance_fn 對 hard BC (output gate) 或 SDF input feature (trunk concat) 都需要。
    _need_bd = bool(getattr(net, "use_hard_body_bc", False)) or bool(getattr(net, "use_body_distance_feature", False))
    if _need_bd and body_distance_fn is None:
        raise ValueError(
            "model use_hard_body_bc=True 或 use_body_distance_feature=True 但 make_picon_model_fn_uvp() 沒收到 body_distance_fn"
        )

    def model_fn_uvp(xyt: torch.Tensor) -> torch.Tensor:
        xyt_d = xyt.to(net_device)
        xy_d = xyt_d[:, :2]
        t_q_d = xyt_d[:, 2]
        bd = body_distance_fn(xy_d) if _need_bd else None
```

- [ ] **Step 6.3: Verify**

```bash
grep -n '_need_bd' src/pi_con/operator.py | head -5
```

Expected: 4 lines (2 def + 2 condition + 2 closure = ~4-6 lines).

---

## Task 7: Update training.py body_distance_fn wiring

**Files:**
- Modify: `src/pi_con/training.py` (lines 132-163 + ~7 conditional usage sites)

- [ ] **Step 7.1: Update gate variable at line 132-141**

`old_string`:
```python
    # ── Body-distance feature（cylinder boundary layer 學習）──────────
    # use_hard_body_bc=True 時，model output 會套 (φ/scale).clamp(0,1) gate 強制 body 內 u=v=0。
    # 對 cylinder dataset 從 _detect_body 預計算的 SDF grid 用 bilinear interp 取值；
    # 對 kolmogorov 退化為常數 1.0（無 body）。
    _use_hard_body_bc = bool(args.get("use_hard_body_bc", False))
    if _use_hard_body_bc and not getattr(net, "use_hard_body_bc", False):
        raise ValueError(
            "config use_hard_body_bc=True 但 net.use_hard_body_bc=False；"
            "可能 model 用舊 ckpt resume 但 ckpt 是 hard BC 關閉時訓的。"
        )
```

`new_string`:
```python
    # ── Body-distance feature（cylinder boundary layer 學習）──────────
    # 兩種使用方式 (兩者皆需要 body_distance_fn 注入):
    # 1. use_hard_body_bc=True (Stage 1 path)：output gate u = (φ/scale).clamp(0,1) · NN。
    # 2. use_body_distance_feature=True (Stage 2 Option A)：trunk input concat query+=φ。
    # 對 cylinder dataset 從 _detect_body 預計算的 SDF grid 用 bilinear interp 取值；
    # 對 kolmogorov 沒 SDF，啟用任一 flag 都會在 _make_body_distance_fn 內 raise。
    _use_hard_body_bc = bool(args.get("use_hard_body_bc", False))
    _use_body_distance_feature = bool(args.get("use_body_distance_feature", False))
    _need_body_distance_fn = _use_hard_body_bc or _use_body_distance_feature
    if _use_hard_body_bc and not getattr(net, "use_hard_body_bc", False):
        raise ValueError(
            "config use_hard_body_bc=True 但 net.use_hard_body_bc=False；"
            "可能 model 用舊 ckpt resume 但 ckpt 是 hard BC 關閉時訓的。"
        )
    if _use_body_distance_feature and not getattr(net, "use_body_distance_feature", False):
        raise ValueError(
            "config use_body_distance_feature=True 但 net.use_body_distance_feature=False；"
            "可能 model 用舊 ckpt resume 但 ckpt 是該 flag 關閉時訓的。"
        )
```

- [ ] **Step 7.2: Update body_distance_fns construction gate (line ~160-163)**

`old_string`:
```python
    body_distance_fns: list = []
    if _use_hard_body_bc:
        for ds in datasets:
            body_distance_fns.append(_make_body_distance_fn(ds))
```

`new_string`:
```python
    body_distance_fns: list = []
    if _need_body_distance_fn:
        for ds in datasets:
            body_distance_fns.append(_make_body_distance_fn(ds))
```

- [ ] **Step 7.3: Replace ALL 7 occurrences of `_use_hard_body_bc` (as conditional gate for `body_distance_fns[...]` access) with `_need_body_distance_fn`**

`replace_all=True` on Edit:
- `old_string`: `if _use_hard_body_bc else None`
- `new_string`: `if _need_body_distance_fn else None`

Then for the other 2 usages:
- `old_string`: `body_distance_fn=body_distance_fns[_i] if _use_hard_body_bc else None,`
- `new_string`: `body_distance_fn=body_distance_fns[_i] if _need_body_distance_fn else None,`
- `replace_all=True`

And:
- `old_string`: `body_distance_fn=body_distance_fns[i] if _use_hard_body_bc else None,`
- `new_string`: `body_distance_fn=body_distance_fns[i] if _need_body_distance_fn else None,`
- `replace_all=True`

And for the RAR-pool case (line ~1059):
- `old_string`: `body_distance_fns=body_distance_fns if _use_hard_body_bc else None,`
- `new_string`: `body_distance_fns=body_distance_fns if _need_body_distance_fn else None,`

- [ ] **Step 7.4: Verify ALL _use_hard_body_bc → _need_body_distance_fn substitutions complete in conditional gate sites**

```bash
grep -n '_use_hard_body_bc\|_need_body_distance_fn\|_use_body_distance_feature' src/pi_con/training.py
```

Expected (counts):
- `_use_hard_body_bc` 應**只**剩 3 處：line 136 (declaration), line 137 (raise check), and now possibly 0 inline `if _use_hard_body_bc else None`
- `_need_body_distance_fn` 應有 ≥ 7 occurrences (1 declaration + 6+ conditional gate sites)
- `_use_body_distance_feature` 應有 2 處 (declaration + raise check)

---

## Task 8: Smoke test src patches — load CEXP-002 with flag=True, create model, forward pass

**Files:**
- Use existing: `src/pi_con/training.py` model factory + Python REPL

- [ ] **Step 8.1: Create a smoke test script (one-shot, no commit)**

Write to `/tmp/smoke_sdf_input.py`:

```python
"""Smoke test: load CEXP-002 config, set use_body_distance_feature=True, create model, forward."""
import sys
sys.path.insert(0, "/Users/latteine/Documents/coding/pi-lnn")
sys.path.insert(0, "/Users/latteine/Documents/coding/pi-lnn/src")

from pi_con.config import DEFAULT_PICON_ARGS, load_picon_config
from pi_con.operator import create_picon_model

cfg = dict(DEFAULT_PICON_ARGS)
cfg["use_periodic_domain"] = False
cfg["fourier_embed_dim"] = 128
cfg["d_model"] = 128
cfg["use_body_distance_feature"] = True
cfg["use_hard_body_bc"] = False

print("Creating model with use_body_distance_feature=True...")
model = create_picon_model(cfg)
print(f"  trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
print(f"  net.use_body_distance_feature: {model.use_body_distance_feature}")
print(f"  decoder.use_body_distance_feature: {model.query_decoder.use_body_distance_feature}")
print("OK — model created.")
```

- [ ] **Step 8.2: Run the smoke test**

```bash
cd /Users/latteine/Documents/coding/pi-lnn && .venv/bin/python /tmp/smoke_sdf_input.py 2>&1 | head -20
```

Expected output (no errors):
```
Creating model with use_body_distance_feature=True...
  trainable params: <some int>
  net.use_body_distance_feature: True
  decoder.use_body_distance_feature: True
OK — model created.
```

If errors:
- ImportError → check src patches missing
- ValueError 提到 query_in dim → check Task 3 query_in calc patched
- AttributeError → check Task 2 LiquidOperator forwarding

---

## Task 9: Write configs/exp_cylinder_020_trunk_sdf_input.toml

**Files:**
- Create: `configs/exp_cylinder_020_trunk_sdf_input.toml`

- [ ] **Step 9.1: Copy CEXP-002 baseline as starting template**

```bash
cp configs/exp_cylinder_002_k100_bc.toml configs/exp_cylinder_020_trunk_sdf_input.toml
```

- [ ] **Step 9.2: Use Edit to replace header block (top of file)**

`old_string`: 整個 CEXP-002 header (約 line 1-10, 不含 [train])

`new_string`:
```
# configs/exp_cylinder_020_trunk_sdf_input.toml
# CEXP-020 = Stage 2 Option A: trunk SDF input feature
#
# 設計目的：
#   Stage 1 (CEXP-016/017/018/019) 全 ❌：hard BC + standard PI-CON architecture
#   fundamental incompatibility (KE 111-303%). Root cause: trunk net 完全沒有
#   geometry awareness — hard BC 只是 output post-hoc gate, NN_u 不知 boundary
#   在哪 → wake 區 over-compensation → GradNorm pathology。
#
#   本實驗：trunk 級 geometry awareness — 加 raw φ scalar 到 query input
#   (query_in dim 4→5, post-Fourier concat), 移除 hard BC gate 完全。
#   trunk MLP 自學「離 body 越近 → velocity magnitude 越小」inductive bias。
#
# 與 CEXP-002 (KE 3.54% working baseline) 唯一差異：
#   use_body_distance_feature: false → true   (新增 src patch by 2026-05-24 spec)
#
# Falsifiability gates (per spec §4):
#   KE < 10%   → ✅ A: Option A confirmed; trunk SDF is viable alt to hard BC
#   KE 10-30%  → 🟡 B: partial; consider Options B (Fourier-on-φ) or D (per-layer gate)
#   KE > 30%   → ❌ C: deeper architectural issue; escalate to E/F
#
# 啟動條件與狀態追蹤：見 docs/cylinder_log_v2.md CEXP-020 entry
```

- [ ] **Step 9.3: Use Edit to add `use_body_distance_feature = true` (after `use_physics_denormalization` block)**

`old_string`:
```
use_physics_denormalization = true
```

`new_string`:
```
use_physics_denormalization = true

# ===== Stage 2 Option A 唯一變動 =====
use_body_distance_feature = true   # trunk query_in += φ raw scalar, hard BC OFF
# ====================================
```

- [ ] **Step 9.4: Use Edit to change `artifacts_dir`**

`old_string`:
```
artifacts_dir = "artifacts/cylinder/deeponet-cfc-cylinder-exp002-k100-bc"
```

`new_string`:
```
artifacts_dir = "artifacts/cylinder/deeponet-cfc-cylinder-exp020-trunk-sdf-input"
```

- [ ] **Step 9.5: Verify diff vs CEXP-002**

```bash
diff configs/exp_cylinder_002_k100_bc.toml configs/exp_cylinder_020_trunk_sdf_input.toml | head -40
```

Expected: ONLY 3 logical change blocks: (a) header rewritten, (b) new `use_body_distance_feature = true` line, (c) `artifacts_dir` path.

---

## Task 10: Commit src + config + push

**Files:**
- Modify: git index

- [ ] **Step 10.1: Stage exactly the 5 files**

```bash
git add src/pi_con/config.py \
        src/pi_con/operator.py \
        src/pi_con/decoder.py \
        src/pi_con/training.py \
        configs/exp_cylinder_020_trunk_sdf_input.toml
git status | head -15
```

Expected: 4 modified src + 1 new config staged。Untracked Kolmogorov files NOT staged.

- [ ] **Step 10.2: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(cylinder): Stage 2 Option A — trunk SDF input feature (CEXP-020)

Per docs/superpowers/specs/2026-05-24-cylinder-trunk-sdf-input-design.md.

Stage 1 (CEXP-016/017/018/019) 全 ❌: hard BC + standard PI-CON architecture
fundamental incompatibility (KE 111-303%). Root cause: trunk net no geometry
awareness — hard BC is post-hoc output gate, NN doesn't see boundary.

Stage 2 Option A redirect: add SDF φ as raw scalar input to trunk query
(query_in dim 4→5 post-Fourier concat), remove hard BC gate entirely.

Src changes (5 patch sites across 4 files):
- config.py: + use_body_distance_feature key in DEFAULT_PICON_ARGS
- operator.py: forward flag LiquidOperator → DeepONetCfCDecoder + create_picon_model
- decoder.py: query_in dim +1 if flag; concat phi in forward_uvp + forward
- training.py: _need_body_distance_fn = use_hard_body_bc OR use_body_distance_feature
- operator.py make_picon_model_fn{,_uvp}: same OR condition

Config CEXP-020 = CEXP-002 baseline + use_body_distance_feature=true,
use_hard_body_bc=false. Single variable vs baseline.

Falsifiability gates (Moderate per spec §4):
- KE < 10% → ✅ Option A confirmed; trunk SDF viable alt to hard BC
- KE 10-30% → 🟡 partial; escalate to Options B/D
- KE > 30% → ❌ broken; escalate to Options E/F (future paper)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 10.3: Push**

```bash
git push origin main 2>&1 | tail -3
```

Expected: `main -> main` 帶 commit hash.

---

## Task 11: Lab git pull + deployment sed

**Files:**
- Modify on lab (not committed): `configs/exp_cylinder_020_trunk_sdf_input.toml`

- [ ] **Step 11.1: Lab git pull**

```bash
ssh lab-server 'cd pi-lnn && git pull --ff-only 2>&1 | tail -5'
```

Expected: 4 src files + 1 config + plan/spec changes pulled.

- [ ] **Step 11.2: Apply 3 lab-only sed edits to CEXP-020 config**

```bash
ssh lab-server 'cd pi-lnn && sed -i \
    -e "s|/Users/latteine/Documents/coding/RealPDEBench|/home/junyi/RealPDEBench|g" \
    -e "s|^kolmogorov_A = 0.0$|kolmogorov_A = 1e-6|" \
    -e "s|^kolmogorov_k_f = 0.0$|kolmogorov_k_f = 2.0|" \
    configs/exp_cylinder_020_trunk_sdf_input.toml'
```

- [ ] **Step 11.3: Verify all 3 lab edits applied**

```bash
ssh lab-server 'cd pi-lnn && grep -E "^(kolmogorov_|arrow_shards)" configs/exp_cylinder_020_trunk_sdf_input.toml | head -8; grep -A 1 "arrow_shards = \[" configs/exp_cylinder_020_trunk_sdf_input.toml | head -3'
```

Expected:
- `arrow_shards` line shows `/home/junyi/RealPDEBench/...`
- `kolmogorov_A = 1e-6`
- `kolmogorov_k_f = 2.0`

- [ ] **Step 11.4: Lab smoke test (verify src + new flag work on lab)**

```bash
ssh lab-server 'cd pi-lnn && .venv/bin/python -c "
import sys; sys.path.insert(0, \"src\")
from pi_con.config import DEFAULT_PICON_ARGS
from pi_con.operator import create_picon_model
cfg = dict(DEFAULT_PICON_ARGS)
cfg[\"use_periodic_domain\"] = False
cfg[\"fourier_embed_dim\"] = 128
cfg[\"d_model\"] = 128
cfg[\"use_body_distance_feature\"] = True
m = create_picon_model(cfg)
print(f\"lab smoke OK: net.use_body_distance_feature={m.use_body_distance_feature}, decoder.use_body_distance_feature={m.query_decoder.use_body_distance_feature}\")
"' 2>&1 | tail -3
```

Expected: `lab smoke OK: net.use_body_distance_feature=True, decoder.use_body_distance_feature=True`

---

## Task 12: Submit CEXP-020 SLURM job

**Files:**
- Lab logs: `logs/exp_cylinder_020_<jobid>.{out,err}`

- [ ] **Step 12.1: Check r740 partition has free GPU slot**

```bash
ssh lab-server 'sinfo -p r740 -N -o "%N %t %G" 2>&1 | head -3; echo "---squeue---"; squeue -p r740 2>&1 | head -5'
```

Expected: acmt20 idle or mix with ≤ 1 job (GRES gpu:2 → at most 1 other concurrent).

- [ ] **Step 12.2: Submit CEXP-020 via SLURM helper**

```bash
ssh lab-server 'cd pi-lnn && scripts/slurm/submit_exp.sh cylinder_020 configs/exp_cylinder_020_trunk_sdf_input.toml 2>&1 | tail -3'
```

Expected: `Submitted batch job <jobid>`. Record jobid (e.g., 3700).

- [ ] **Step 12.3: Verify job in queue**

```bash
ssh lab-server 'squeue --me 2>&1 | head -5'
```

Expected: CEXP-020 jobid in state R (RUNNING) or PD (PENDING short queue).

---

## Task 13: Wait + eval + rsync

**Files:**
- Lab: `artifacts/cylinder/deeponet-cfc-cylinder-exp020-trunk-sdf-input/`
- Local: same path under `/Users/latteine/Documents/coding/pi-lnn/`

- [ ] **Step 13.1: Set up background wait for SLURM job completion (~1.6 hr)**

Use Bash with `run_in_background: true`:
```bash
until ssh lab-server "sacct -j <JOBID> --noheader --format=State 2>/dev/null | head -1 | grep -qvE 'PENDING|RUNNING'" 2>/dev/null; do sleep 120; done && ssh lab-server 'sacct -j <JOBID> --format=State,Elapsed,ExitCode 2>&1 | head -3; tail -15 pi-lnn/logs/exp_cylinder_020_<JOBID>.out 2>&1'
```

Replace `<JOBID>` with actual jobid from Task 12.2.

Expected after notification: `COMPLETED` with `Elapsed ~1:30:00-1:40:00`, ExitCode 0:0.

- [ ] **Step 13.2: Launch evaluator on lab head node**

```bash
ssh lab-server 'cd pi-lnn && nohup .venv/bin/python -u scripts/evaluate_cylinder.py \
  --config configs/exp_cylinder_020_trunk_sdf_input.toml \
  --checkpoint artifacts/cylinder/deeponet-cfc-cylinder-exp020-trunk-sdf-input/picon_kolmogorov_final.pt \
  > logs/eval_cylinder_020.out 2>&1 </dev/null & disown; echo "eval_020 launched"'
```

- [ ] **Step 13.3: Wait for eval summary.json (background, ~5 min)**

Use Bash with `run_in_background: true`:
```bash
until ssh lab-server '[ -f pi-lnn/artifacts/cylinder/deeponet-cfc-cylinder-exp020-trunk-sdf-input/cylinder-eval/summary.json ]' 2>/dev/null; do sleep 30; done && ssh lab-server 'jq "{ke_rel_err_mean, ke_rel_err_late, ke_pred_mean, ke_ref_mean, u_rmse_mean, v_rmse_mean, omega_rmse_mean, div_l2_mean, body_u_max, body_v_max}" pi-lnn/artifacts/cylinder/deeponet-cfc-cylinder-exp020-trunk-sdf-input/cylinder-eval/summary.json'
```

- [ ] **Step 13.4: Rsync artifact back to local**

```bash
rsync -avz lab-server:/home/junyi/pi-lnn/artifacts/cylinder/deeponet-cfc-cylinder-exp020-trunk-sdf-input/ \
  /Users/latteine/Documents/coding/pi-lnn/artifacts/cylinder/deeponet-cfc-cylinder-exp020-trunk-sdf-input/ 2>&1 | tail -3
ls artifacts/cylinder/deeponet-cfc-cylinder-exp020-trunk-sdf-input/cylinder-eval/summary.json 2>&1
```

Expected: rsync completes, summary.json present locally.

- [ ] **Step 13.5: Capture final w_ns_u from training log (for falsifiability check per spec §4 additional diagnostic)**

```bash
ssh lab-server 'grep "^10000 " pi-lnn/logs/exp_cylinder_020_<JOBID>.out 2>&1 | tail -1'
```

Expected: 訓練最終 step 行，包含 `w_ns_u <value>`。If `< 0.5` → GradNorm not pathology (good sign).

---

## Task 14: Decision tree judgment + update v2 log + commit

**Files:**
- Modify: `docs/cylinder_log_v2.md`
- Read: `artifacts/cylinder/deeponet-cfc-cylinder-exp020-trunk-sdf-input/cylinder-eval/summary.json`

- [ ] **Step 14.1: Read CEXP-020 metrics**

```bash
jq '{ke_rel_err_mean, ke_rel_err_late, ke_pred_mean, ke_ref_mean, u_rmse_mean, v_rmse_mean, omega_rmse_mean, div_l2_mean, body_u_max, body_v_max}' \
  artifacts/cylinder/deeponet-cfc-cylinder-exp020-trunk-sdf-input/cylinder-eval/summary.json
```

- [ ] **Step 14.2: 對照 spec §4 decision tree 判讀 outcome**

依以下 thresholds:

| ke_rel_err_mean | Outcome | Status label |
|---|---|---|
| < 10 % | ✅ A | `ACTIVE_REFERENCE` (Stage 2 confirmed) |
| 10-30 % | 🟡 B | `PARTIAL_RESULT` |
| > 30 % | ❌ C | `NEGATIVE_RESULT` |

Additional diagnostic checks:
- `ke_pred / ke_ref` ∈ [0.85, 1.15] (healthy, not over-predict like CEXP-016/017)
- `w_ns_u_final` < 0.5 (not GradNorm pathology)
- `body_u_max`, `body_v_max` < 0.05 (trunk SDF awareness 真生效)

- [ ] **Step 14.3: Use Edit to update cylinder_log_v2.md [INDEX] (add CEXP-020 row)**

`old_string` (locate the row 在 CEXP-019 row 之後)：
```
| **CEXP-019** | `NEGATIVE_RESULT` | Re=10031, hard BC + **bc_body 96 + bc_outlet 32** (H3) | 139.3 % ❌ | 2.39 | 12.70 | 6.13 | 10k | **❌ H3-C falsified**：dense BC 沒救（且更糟一點）|
```

`new_string` (insert CEXP-020 row immediately after):
```
| **CEXP-019** | `NEGATIVE_RESULT` | Re=10031, hard BC + **bc_body 96 + bc_outlet 32** (H3) | 139.3 % ❌ | 2.39 | 12.70 | 6.13 | 10k | **❌ H3-C falsified**：dense BC 沒救（且更糟一點）|
| **CEXP-020** | `<status>` | Re=10031, **trunk SDF input + hard BC off** (Stage 2 Option A) | <KE> % | <ratio> | <ω> | <div> | 10k | **<A/B/C outcome per §4>**: <one-sentence interpretation> |
```

Replace `<status>` etc. placeholders with Task 14.1 values + Task 14.2 outcome label.

- [ ] **Step 14.4: Use Edit to add [RECORD] detail section for CEXP-020 (after CEXP-019 detail block)**

`old_string`:
```
### CEXP-019：Hard BC + dense BC supervision (H3, KE=139.3%, mild worse)

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_019_hard_bc_dense_bc.toml` |
```
(...full CEXP-019 detail block...)
```
| 結論 | **❌ H3-C falsified**。加密 soft BC supervision **微更糟** (KE 111→139%)。Hard BC over-predict 機制不被多 BC points 打破。 |
```

`new_string`: 上面整段不變 + 新增 CEXP-020 detail:
```
| 結論 | **❌ H3-C falsified**。加密 soft BC supervision **微更糟** (KE 111→139%)。Hard BC over-predict 機制不被多 BC points 打破。 |

### CEXP-020：Stage 2 Option A — trunk SDF input + hard BC off (KE=<X.X>%, <outcome>)

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_020_trunk_sdf_input.toml` |
| Artifact | `artifacts/cylinder/deeponet-cfc-cylinder-exp020-trunk-sdf-input/` |
| Checkpoint | `picon_kolmogorov_final.pt`（step 10000） |
| KE rel-err mean / late | <X.X>% / <X.X>% |
| u / v RMSE | <X.X> / <X.X> |
| ω RMSE | <X.X> |
| div L2 | <X.X> |
| ke_pred / ke_ref | <ratio> |
| GradNorm w_ns_u final | <X.X> |
| body u/v max | <X.X> / <X.X> (no hard BC gate, soft penalty only) |
| 設計變動 (vs CEXP-002 baseline) | 唯一新增 `use_body_distance_feature=true` (trunk query_in dim 4→5, raw φ post-Fourier concat); hard BC remains OFF |
| 結論 | **<A/B/C per spec §4>**: <根據 Task 14.1 metrics 判讀的一句話> |
```

Replace `<X.X>` and `<outcome>` with actual values.

- [ ] **Step 14.5: Use Edit to add [STATE] Surprise Findings #5 (after Finding #4)**

`old_string`:
```
**Stage 2 redirect (Option A)**: 加 SDF `φ` 進 trunk input concat (`query = [x, y, t, c, φ]`), 移除 hard BC gate。Trunk 自學「near-body vs far-field」區分，不依賴 output gate 救援。詳見 [STATE] Open Questions。

---

## [RECORD] Cylinder 實驗詳細記錄
```

`new_string`:
```
**Stage 2 redirect (Option A)**: 加 SDF `φ` 進 trunk input concat (`query = [x, y, t, c, φ]`), 移除 hard BC gate。Trunk 自學「near-body vs far-field」區分，不依賴 output gate 救援。詳見 [STATE] Open Questions。

### Finding 5 — Stage 2 Option A outcome (CEXP-020, 2026-05-24)

CEXP-020 = CEXP-002 baseline + `use_body_distance_feature=true` + `use_hard_body_bc=false`. Single variable vs baseline (KE 3.54%).

| Metric | CEXP-002 baseline | CEXP-020 (Option A) | Δ |
|---|---|---|---|
| KE rel-err mean | 3.54 % | <X.X> % | <Δ pp> |
| ke_pred / ke_ref | 1.01 | <ratio> | — |
| w_ns_u final | 0.108 | <X.X> | <Δ ×> |
| ω RMSE | 2.14 | <X.X> | — |
| div L2 | 1.14 | <X.X> | — |

Outcome (per spec §4 decision tree): **<✅ A / 🟡 B / ❌ C>**

**Interpretation**: <Per outcome 的一段話，從以下 templates 選一個並填具體數字>:

- A) Trunk SDF input concat is a viable architectural alternative to hard body BC output gate. Stage 1 catastrophic 機制 (NN over-compensation in wake) avoided by giving trunk MLP geometry awareness. Paper claim 「PI-CON 在 non-periodic + geometry case 下需要 trunk-level geometry awareness」**confirmed**.

- B) Trunk SDF input 改善 vs hard BC catastrophic (CEXP-016 111% → CEXP-020 <X.X>%) but does not fully match CEXP-002 baseline. Stage 3 candidates: Option B (Fourier-on-φ multi-scale) or Option D (per-layer trunk gate).

- C) Trunk SDF input alone is insufficient. Root cause hypothesis (lack of trunk geometry awareness) is **wrong or incomplete**. Escalate to Options E (cross-attn geometry tokens) / F (geometry-conditioned hypernetwork). Possibly outside this paper scope.

**Diagnostic check**:
- `ke_pred / ke_ref = <ratio>`: <healthy if 0.85-1.15, else over-predict like Stage 1>
- `w_ns_u_final = <X.X>`: <not GradNorm pathology if < 0.5, vs CEXP-017's 3.82>
- `body_u_max = <X.X>, body_v_max = <X.X>`: <trunk learned near-body suppression if < 0.05>

---

## [RECORD] Cylinder 實驗詳細記錄
```

Replace `<X.X>` placeholders + select one of A/B/C interpretation paragraphs per actual outcome.

- [ ] **Step 14.6: Use Edit to update [STATE] Open Questions (Stage 3 plan based on outcome)**

`old_string`:
```
| **Stage 2: Option A — Trunk SDF input concat (CEXP-020)** | per [Option A redirect 2026-05-24]: `use_body_distance_feature=true` + `use_hard_body_bc=false`, raw scalar `φ` 進 trunk query input dim 4→5。需先修 src `DEFAULT_PICON_ARGS` 缺失 key（~30 line patch）| **下個 session 啟動**（spec 寫作待開工）|
```

`new_string` (one of these based on actual outcome):

If **outcome A** (KE < 10%):
```
| **Stage 2: Option A — Trunk SDF input concat (CEXP-020)** | ✅ **Confirmed** (KE <X.X>% < 10%). Trunk SDF input is viable. | ✅ Closed 2026-05-24 (Finding #5) |
| **Stage 3 (next): Multi-seed n=3 confirmation of CEXP-020** | Single-seed result needs statistical replication | **High priority** for paper-grade rigor |
```

If **outcome B** (10-30%):
```
| **Stage 2: Option A — Trunk SDF input concat (CEXP-020)** | 🟡 **Partial** (KE <X.X>%, improved vs CEXP-016 catastrophic but not full baseline-level). | 🟡 Partial 2026-05-24 (Finding #5) |
| **Stage 3 (next): Option B (Fourier-on-φ multi-scale) or D (per-layer trunk gate)** | Improve on Option A partial result | Open for next session |
```

If **outcome C** (> 30%):
```
| **Stage 2: Option A — Trunk SDF input concat (CEXP-020)** | ❌ **Broken** (KE <X.X>% > 30%). Trunk geometry awareness alone insufficient. | ❌ Closed 2026-05-24 (Finding #5) |
| **Stage 3 (escalation): Options E (cross-attn geometry tokens) / F (geometry-conditioned hypernetwork)** | Per spec §4 re-diagnose path | Long-term, likely outside current paper scope |
```

- [ ] **Step 14.7: Update 變更紀錄 (changelog) at end of v2 log**

`old_string` (last changelog entry):
```
- **2026-05-23/24 Stage 1 全 ❌**:
```
(...full entry...)
```
  - [INDEX] CEXP-016/017/018/019 entries finalized, [RECORD] 4 個 detail tables 新增, [STATE] Open Questions Stage 2 plan 寫入
```

`new_string`: above 完整保留 + 新增:
```
  - [INDEX] CEXP-016/017/018/019 entries finalized, [RECORD] 4 個 detail tables 新增, [STATE] Open Questions Stage 2 plan 寫入
- **2026-05-24 Stage 2 Option A result (CEXP-020)**:
  - Src patches landed (5 sites across 4 files, ~30-40 lines): `use_body_distance_feature` flag added to DEFAULT_PICON_ARGS + wired through LiquidOperator → DeepONetCfCDecoder; `_need_body_distance_fn = use_hard_body_bc OR use_body_distance_feature` in operator/training; CEXP-007 silent-ignore bug fixed
  - CEXP-020 KE <X.X>%, ke_pred/ke_ref <ratio>, w_ns_u_final <X.X>, outcome **<✅ A / 🟡 B / ❌ C>** per spec §4
  - Finding #5 added; Open Questions updated with Stage 3 plan based on outcome
  - Options E/F (cross-attn geometry tokens / geometry-conditioned hypernetwork) remain future paper material per user 2026-05-24 decision
```

Replace `<X.X>` placeholders.

- [ ] **Step 14.8: Stage + commit v2 log update**

```bash
git add docs/cylinder_log_v2.md
git commit -m "$(cat <<'EOF'
docs(v2): Stage 2 Option A result — CEXP-020 trunk SDF input (KE <X.X>%)

CEXP-020 = CEXP-002 baseline + use_body_distance_feature=true,
use_hard_body_bc=false. Single variable vs working baseline.

Per spec docs/superpowers/specs/2026-05-24-cylinder-trunk-sdf-input-design.md
§4 decision tree:
- KE <X.X>% / ke_pred-ke_ref <ratio> / w_ns_u_final <X.X>
- Outcome: <A / B / C>
- Interpretation: <one-line>

Finding #5 added; INDEX row + RECORD detail table inserted;
Open Questions Stage 3 plan updated per outcome.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Replace `<X.X>` `<ratio>` `<A/B/C>` placeholders with Task 14.1/14.2 actuals.

- [ ] **Step 14.9: Push**

```bash
git push origin main 2>&1 | tail -3
```

Expected: `main -> main` 帶 commit hash.

---

## Spec Coverage Self-Check

| Spec section | Plan task | OK? |
|---|---|---|
| §1 Goal: trunk SDF input feature, hard BC off | Tasks 1-7 (src patch) + Task 9 (config) | ✅ |
| §1 Success criteria: KE < 10% / 10-30% / > 30% | Task 14.2 decision tree judgment | ✅ |
| §2 Architecture: query_in dim 4→5 post-Fourier | Tasks 3-5 | ✅ |
| §2 Src patches: 4 files | Tasks 1-7 | ✅ |
| §3 Config Matrix: CEXP-020 only | Task 9 | ✅ |
| §3 Hyperparams 對齊 CEXP-002 | Task 9.1 cp from CEXP-002 | ✅ |
| §4 Falsifiability gates KE thresholds + additional diagnostic | Task 14.2-14.5 | ✅ |
| §4 Stop loss (KE > 50% catastrophic) | Implicit in Task 14.2 outcome C path | ✅ |
| §5 Prereqs (src patches + lab deployment) | Tasks 1-7 + Task 11 | ✅ |
| §5 Workflow (commit→push→lab pull→submit→eval→rsync→judge→update log) | Tasks 10-14 | ✅ |
| §5 Out-of-scope (no multi-seed, no dual mode, no E/F) | Implicit (only 1 config CEXP-020) | ✅ |

---

## Stop Loss Conditions（implementer 觀察）

訓練中**不** early stop。但若：

- Task 8 smoke test 拋 exception → 立即 debug src patches (Tasks 1-7 中某個漏掉)
- Task 11.4 lab smoke 失敗但本地 8 成功 → check lab .venv 是否要 reinstall
- Task 12.2 submit FAILED → debug logs/exp_cylinder_020_<jobid>.err (probably src import error not caught by smoke)
- Task 13.1 SLURM job FAILED → 立即 abort & check err logs
- Task 14.1 KE > 50 % (catastrophic, 同 CEXP-016 量級) → 仍照 Task 14 流程完成記錄 (作 negative finding), 但**不**啟 follow-up Stage 3 in this session
