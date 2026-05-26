# Cylinder Geometry Tokens (CEXP-022) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add body surface geometry tokens to cross-attention K-V pool (Option E), enabling the decoder to attend to zero-velocity body boundary tokens alongside sensor tokens.

**Architecture:** Three new modules in `DeepONetCfCDecoder` (`geo_key_proj`, `geo_value` parameter, `geo_token_type_bias`) encode body surface positions into the attention key space and a shared zero-velocity prior as the value. A `geometry_pos` buffer (filled at training start from `ds.body_xy`) holds body surface coordinates. In `forward_uvp` and `forward`, geometry tokens are concatenated to the K-V pool before multi-head attention, and the `relpos_bias` is extended to cover body tokens. Combined with `use_hard_body_bc=True` output gate for machine-precision boundary enforcement.

**Tech Stack:** Python 3.12, PyTorch 2.7.1+cu118, SLURM r740 partition (acmt20 RTX 3090), uv venv, SOAP + ScheduleFree optimizer.

**Spec reference:** [`docs/superpowers/specs/2026-05-27-cylinder-geometry-tokens-design.md`](../specs/2026-05-27-cylinder-geometry-tokens-design.md)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/pi_con/config.py` | Modify | Add `use_geometry_tokens: False`, `n_geometry_tokens: -1` to DEFAULT_PICON_ARGS |
| `src/pi_con/decoder.py` | Modify | (1) `__init__`: 3 new modules + `geometry_pos` buffer; (2) `forward_uvp`: concat geo tokens; (3) `forward`: concat geo tokens |
| `src/pi_con/operator.py` | Modify | Forward new flags through `LiquidOperator.__init__` + `create_picon_model`; add `set_geometry_tokens()` method |
| `src/pi_con/training.py` | Modify | After model creation, inject `ds.body_xy` into `model.set_geometry_tokens()` |
| `configs/exp_cylinder_022_geometry_tokens.toml` | Create | CEXP-022 config: geometry tokens + hard BC, no SDF trunk input |
| `docs/cylinder_log_v2.md` | Modify | [INDEX] CEXP-022 row + [RECORD] detail + [STATE] updates |

---

## Task 1: Add `use_geometry_tokens` and `n_geometry_tokens` to DEFAULT_PICON_ARGS

**Files:**
- Modify: `src/pi_con/config.py`

- [ ] **Step 1.1: Insert two new keys immediately AFTER `use_hard_body_bc` block**

Use Edit. `old_string`:
```python
    "use_hard_body_bc": False,           # cylinder hard body BC（Sukumar 2022 風格）：
                                          # u = (φ/scale).clamp(0,1) · NN，物理保證 body 內 u=v=0。
                                          # 取代有 detach bug 的 distance-as-input feature。
                                          # True 會改 model architecture（query_in +1），ckpt 不相容
                                          # 僅 cylinder dataset 真正需要；kolmogorov 為 dummy 1.0
    "use_body_distance_feature": False,  # cylinder trunk SDF input feature (Stage 2 Option A)：
```

`new_string`:
```python
    "use_hard_body_bc": False,           # cylinder hard body BC（Sukumar 2022 風格）：
                                          # u = (φ/scale).clamp(0,1) · NN，物理保證 body 內 u=v=0。
                                          # 取代有 detach bug 的 distance-as-input feature。
                                          # True 會改 model architecture（query_in +1），ckpt 不相容
                                          # 僅 cylinder dataset 真正需要；kolmogorov 為 dummy 1.0
    "use_geometry_tokens": False,        # Option E: body 表面 geometry tokens 加入 cross-attention K-V pool。
                                          # 每個 body token: key=Fourier(body_xy)+type_bias, value=geo_value 共用先驗。
                                          # body_xy 從 cylinder dataset.body_xy 在訓練開始前注入。
                                          # True 加入 3 個新 learnable module，ckpt 不相容（cold start）。
    "n_geometry_tokens": -1,            # geometry token 數量：-1 = 全部 ds.body_xy；正整數 = 取前 n 點。
    "use_body_distance_feature": False,  # cylinder trunk SDF input feature (Stage 2 Option A)：
```

- [ ] **Step 1.2: Verify**

```bash
grep -A 2 '"use_geometry_tokens"' src/pi_con/config.py | head -5
```

Expected: `"use_geometry_tokens": False,` followed by `"n_geometry_tokens": -1,`

---

## Task 2: Add `use_geometry_tokens` to `DeepONetCfCDecoder.__init__` — new modules + buffer

**Files:**
- Modify: `src/pi_con/decoder.py` (lines 44-186)

- [ ] **Step 2.1: Add `use_geometry_tokens` to `__init__` signature (after `use_body_distance_feature` at line 46)**

`old_string`:
```python
        use_body_distance_feature: bool = False,
        decoder_attention_heads: int = 1,
```

`new_string`:
```python
        use_body_distance_feature: bool = False,
        use_geometry_tokens: bool = False,
        decoder_attention_heads: int = 1,
```

- [ ] **Step 2.2: Store flag + create 3 new modules + geometry_pos buffer (after body_bc_scale buffer registration at line ~118)**

`old_string`:
```python
        self.register_buffer(
            "body_bc_scale",
            torch.tensor(1.0, dtype=torch.float32),
            persistent=False,
        )
        rank = d_model if operator_rank is None else operator_rank
```

`new_string`:
```python
        self.register_buffer(
            "body_bc_scale",
            torch.tensor(1.0, dtype=torch.float32),
            persistent=False,
        )
        # Option E: Cross-attention geometry tokens (CEXP-022+)
        # body surface xy injected by training.py into geometry_pos buffer.
        # 3 new learnable modules:
        #   geo_key_proj: Fourier(body_xy) [N_body, spatial_dim] → key space [N_body, query_mlp_hidden_dim]
        #   geo_value: shared zero-velocity prior [1, query_mlp_hidden_dim] (learned init = 0)
        #   geo_token_type_bias: key bias distinguishing body tokens from sensor tokens [query_mlp_hidden_dim]
        self.use_geometry_tokens = bool(use_geometry_tokens)
        if self.use_geometry_tokens:
            self.geo_key_proj = nn.Linear(spatial_dim, query_mlp_hidden_dim)
            self.geo_value = nn.Parameter(torch.zeros(1, query_mlp_hidden_dim))
            self.geo_token_type_bias = nn.Parameter(torch.zeros(query_mlp_hidden_dim))
        # geometry_pos: body surface normalized coordinates, shape [N_body, 2]
        # persistent=False: not saved in ckpt; filled at training start from ds.body_xy
        self.register_buffer(
            "geometry_pos",
            torch.zeros(0, 2, dtype=torch.float32),
            persistent=False,
        )
        rank = d_model if operator_rank is None else operator_rank
```

- [ ] **Step 2.3: Verify**

```bash
grep -n 'use_geometry_tokens\|geo_key_proj\|geo_value\|geo_token_type_bias\|geometry_pos' src/pi_con/decoder.py | head -15
```

Expected: 7-10 lines covering `__init__` signature + store + 3 modules + buffer.

---

## Task 3: Modify `decoder.forward_uvp` — concat geometry tokens after k_3/v_3/rel_bias_3 creation

**Files:**
- Modify: `src/pi_con/decoder.py` (lines 270-280)

- [ ] **Step 3.1: Insert geometry token concat block between k_3/v_3/rel_bias_3 lines and `if self.disable_cross_attention:`**

`old_string`:
```python
        # c-independent tensors 對齊到 [3N, ...]：等同於 [c=0 段 / c=1 段 / c=2 段]
        k_3 = k_proj.repeat(3, 1, 1)                                            # [3N, K, hidden]
        v_3 = v_proj.repeat(3, 1, 1)                                            # [3N, K, hidden]
        rel_bias_3 = rel_bias.repeat(3, 1)                                      # [3N, K]
        rel_r_3 = rel_r.repeat(3, 1, 1)                                         # [3N, K, 1]

        if self.disable_cross_attention:
```

`new_string`:
```python
        # c-independent tensors 對齊到 [3N, ...]：等同於 [c=0 段 / c=1 段 / c=2 段]
        k_3 = k_proj.repeat(3, 1, 1)                                            # [3N, K, hidden]
        v_3 = v_proj.repeat(3, 1, 1)                                            # [3N, K, hidden]
        rel_bias_3 = rel_bias.repeat(3, 1)                                      # [3N, K]
        rel_r_3 = rel_r.repeat(3, 1, 1)                                         # [3N, K, 1]

        # Option E: Append geometry tokens to K-V pool
        if self.use_geometry_tokens and self.geometry_pos.shape[0] > 0:
            N_body = self.geometry_pos.shape[0]
            geo_pos = self.geometry_pos.to(device=device, dtype=xy.dtype)
            # Fourier encode body surface positions (reuse same spatial_emb as query)
            if self.spatial_emb is not None:
                geo_enc = self.spatial_emb(geo_pos, self.domain_length)          # [N_body, spatial_dim]
            else:
                geo_enc = periodic_fourier_encode(geo_pos, self.domain_length, self.fourier_harmonics)
            # Key: position encoding → key projection + token type bias
            geo_k = self.geo_key_proj(geo_enc) + self.geo_token_type_bias       # [N_body, hidden]
            # Value: shared zero-velocity prior (all body tokens share same value)
            geo_v = self.geo_value.expand(N_body, -1)                           # [N_body, hidden]
            # Extend to [3N, N_body, hidden] for 3-channel batch
            geo_k_3 = geo_k.unsqueeze(0).expand(3 * N, -1, -1)                 # [3N, N_body, hidden]
            geo_v_3 = geo_v.unsqueeze(0).expand(3 * N, -1, -1)                 # [3N, N_body, hidden]
            k_3 = torch.cat([k_3, geo_k_3], dim=1)                             # [3N, K+N_body, hidden]
            v_3 = torch.cat([v_3, geo_v_3], dim=1)                             # [3N, K+N_body, hidden]
            # Extend relpos bias: query-to-body distances
            geo_rel = xy.unsqueeze(1) - geo_pos.unsqueeze(0)                   # [N, N_body, 2]
            geo_rel_r = torch.sqrt((geo_rel ** 2).sum(-1, keepdim=True) + 1e-8) # [N, N_body, 1]
            geo_bias = self.relpos_bias(geo_rel_r).squeeze(-1)                  # [N, N_body]
            rel_bias_3 = torch.cat([rel_bias_3, geo_bias.repeat(3, 1)], dim=1) # [3N, K+N_body]
            if self.use_locality_decay:
                rel_r_3 = torch.cat([rel_r_3, geo_rel_r.repeat(3, 1, 1)], dim=1)  # [3N, K+N_body, 1]

        if self.disable_cross_attention:
```

---

## Task 4: Modify `decoder.forward` — concat geometry tokens (single-channel path)

**Files:**
- Modify: `src/pi_con/decoder.py` (lines 394-422)

- [ ] **Step 4.1: Insert geometry token concat block between rel_bias computation and `if self.disable_cross_attention:`**

`old_string`:
```python
        rel_bias = self.relpos_bias(rel_r).squeeze(-1)
        if self.disable_cross_attention:
```

`new_string`:
```python
        rel_bias = self.relpos_bias(rel_r).squeeze(-1)

        # Option E: Append geometry tokens to K-V pool (single-channel path, symmetric to forward_uvp)
        if self.use_geometry_tokens and self.geometry_pos.shape[0] > 0:
            N_body = self.geometry_pos.shape[0]
            geo_pos = self.geometry_pos.to(device=xy.device, dtype=xy.dtype)
            if self.spatial_emb is not None:
                geo_enc = self.spatial_emb(geo_pos, self.domain_length)
            else:
                geo_enc = periodic_fourier_encode(geo_pos, self.domain_length, self.fourier_harmonics)
            geo_k = self.geo_key_proj(geo_enc) + self.geo_token_type_bias       # [N_body, hidden]
            geo_v = self.geo_value.expand(N_body, -1)                           # [N_body, hidden]
            N_q = q.shape[0]
            geo_k_q = geo_k.unsqueeze(0).expand(N_q, -1, -1)                   # [N_q, N_body, hidden]
            geo_v_q = geo_v.unsqueeze(0).expand(N_q, -1, -1)                   # [N_q, N_body, hidden]
            k = torch.cat([k, geo_k_q], dim=1)                                 # [N_q, K+N_body, hidden]
            v = torch.cat([v, geo_v_q], dim=1)                                 # [N_q, K+N_body, hidden]
            geo_rel = xy.unsqueeze(1) - geo_pos.unsqueeze(0)
            geo_rel_r = torch.sqrt((geo_rel ** 2).sum(-1, keepdim=True) + 1e-8)
            geo_bias = self.relpos_bias(geo_rel_r).squeeze(-1)
            rel_bias = torch.cat([rel_bias, geo_bias], dim=1)                  # [N_q, K+N_body]
            if self.use_locality_decay:
                rel_r = torch.cat([rel_r, geo_rel_r], dim=1)                   # [N_q, K+N_body, 1]

        if self.disable_cross_attention:
```

---

## Task 5: Update operator.py — LiquidOperator + set_geometry_tokens + create_picon_model

**Files:**
- Modify: `src/pi_con/operator.py`

- [ ] **Step 5.1: Add `use_geometry_tokens` param to LiquidOperator.__init__ signature (after `use_body_distance_feature` at line ~49)**

`old_string`:
```python
        use_body_distance_feature: bool = False,
        decoder_attention_heads: int = 1,
```

`new_string`:
```python
        use_body_distance_feature: bool = False,
        use_geometry_tokens: bool = False,
        decoder_attention_heads: int = 1,
```

- [ ] **Step 5.2: Store flag + forward to DeepONetCfCDecoder (after `self.use_body_distance_feature` line ~59)**

`old_string`:
```python
        self.use_body_distance_feature = bool(use_body_distance_feature)
        self.spatial_encoder = SpatialSetEncoder(
```

`new_string`:
```python
        self.use_body_distance_feature = bool(use_body_distance_feature)
        self.use_geometry_tokens = bool(use_geometry_tokens)
        self.spatial_encoder = SpatialSetEncoder(
```

- [ ] **Step 5.3: Forward use_geometry_tokens to DeepONetCfCDecoder in LiquidOperator.__init__ (find `use_body_distance_feature=use_body_distance_feature,` in the DeepONetCfCDecoder constructor call)**

`old_string`:
```python
            use_body_distance_feature=use_body_distance_feature,
            decoder_attention_heads=decoder_attention_heads,
```

`new_string`:
```python
            use_body_distance_feature=use_body_distance_feature,
            use_geometry_tokens=use_geometry_tokens,
            decoder_attention_heads=decoder_attention_heads,
```

- [ ] **Step 5.4: Add `set_geometry_tokens()` method to LiquidOperator class (add right after `set_body_bc_scale()` method)**

First locate the end of `set_body_bc_scale` by reading. It ends around:
```python
        if scale <= 0:
            raise ValueError(f"body_bc_scale 必須 > 0，收到 {scale}")
        self.query_decoder.body_bc_scale.fill_(float(scale))
```

Use Edit:
`old_string`:
```python
        if scale <= 0:
            raise ValueError(f"body_bc_scale 必須 > 0，收到 {scale}")
        self.query_decoder.body_bc_scale.fill_(float(scale))

    def encode(
```

`new_string`:
```python
        if scale <= 0:
            raise ValueError(f"body_bc_scale 必須 > 0，收到 {scale}")
        self.query_decoder.body_bc_scale.fill_(float(scale))

    def set_geometry_tokens(self, body_xy: torch.Tensor) -> None:
        """Inject body surface positions for geometry token cross-attention (Option E).

        Args:
            body_xy: [N_body, 2] normalized body surface coordinates from ds.body_xy.
        """
        if not self.use_geometry_tokens:
            raise ValueError(
                "set_geometry_tokens() 呼叫時 use_geometry_tokens=False；"
                "請先在 config 啟用 use_geometry_tokens=True。"
            )
        self.query_decoder.register_buffer("geometry_pos", body_xy, persistent=False)

    def encode(
```

- [ ] **Step 5.5: Forward flag in `create_picon_model` (after `use_body_distance_feature=bool(cfg.get(...))` line)**

`old_string`:
```python
        use_body_distance_feature=bool(cfg.get("use_body_distance_feature", False)),
        decoder_attention_heads=int(cfg.get("decoder_attention_heads", 1)),
```

`new_string`:
```python
        use_body_distance_feature=bool(cfg.get("use_body_distance_feature", False)),
        use_geometry_tokens=bool(cfg.get("use_geometry_tokens", False)),
        decoder_attention_heads=int(cfg.get("decoder_attention_heads", 1)),
```

- [ ] **Step 5.6: Verify operator.py changes**

```bash
grep -n 'use_geometry_tokens\|set_geometry_tokens' src/pi_con/operator.py | head -10
```

Expected: ≥ 5 occurrences (signature, store, decoder forward, create_picon_model, set_geometry_tokens method x2).

---

## Task 6: Update training.py — inject body_xy geometry positions at model init

**Files:**
- Modify: `src/pi_con/training.py` (around lines 160-164, after body_distance_fns construction)

- [ ] **Step 6.1: Add geometry token injection block (after the `body_distance_fns` for-loop, before the `# 注入 dataset-specific bc_distance_scale` section or training loop)**

First run: `grep -n '_need_body_distance_fn\|body_distance_fns\|bc_distance_scale' src/pi_con/training.py | head -10` to find the exact insertion point.

Then use Edit to insert after the body_distance_fns construction loop. Locate:
```python
    body_distance_fns: list = []
    if _need_body_distance_fn:
        for ds in datasets:
            body_distance_fns.append(_make_body_distance_fn(ds))
```

`old_string`:
```python
    body_distance_fns: list = []
    if _need_body_distance_fn:
        for ds in datasets:
            body_distance_fns.append(_make_body_distance_fn(ds))
```

`new_string`:
```python
    body_distance_fns: list = []
    if _need_body_distance_fn:
        for ds in datasets:
            body_distance_fns.append(_make_body_distance_fn(ds))

    # Option E: Geometry tokens — inject body surface coordinates from dataset
    _use_geometry_tokens = bool(args.get("use_geometry_tokens", False))
    if _use_geometry_tokens:
        if not datasets:
            raise ValueError("use_geometry_tokens=True 但 datasets 為空。")
        ds0 = datasets[0]
        if not hasattr(ds0, "body_xy"):
            raise AttributeError(
                "use_geometry_tokens=True 需要 cylinder dataset (有 body_xy)；"
                "kolmogorov dataset 不支援 geometry tokens。"
            )
        _n_geo = int(args.get("n_geometry_tokens", -1))
        body_pos = torch.tensor(ds0.body_xy, dtype=torch.float32, device=device)
        if _n_geo > 0 and _n_geo < body_pos.shape[0]:
            body_pos = body_pos[:_n_geo]
        net.set_geometry_tokens(body_pos)
        print(f"  geometry_tokens: {body_pos.shape[0]} body surface points injected.")
```

- [ ] **Step 6.2: Verify**

```bash
grep -n '_use_geometry_tokens\|set_geometry_tokens' src/pi_con/training.py | head -5
```

Expected: 2-3 lines showing declaration and method call.

---

## Task 7: Smoke test — verify all 4 src patches work together

- [ ] **Step 7.1: Create smoke test script**

Write to `/tmp/smoke_geo_tokens.py`:

```python
"""Smoke test: geometry tokens + hard BC model creation and forward."""
import sys
sys.path.insert(0, "/Users/latteine/Documents/coding/pi-lnn")
sys.path.insert(0, "/Users/latteine/Documents/coding/pi-lnn/src")
import torch
from pi_con.config import DEFAULT_PICON_ARGS
from pi_con.operator import create_picon_model

cfg = dict(DEFAULT_PICON_ARGS)
cfg.update({
    "use_periodic_domain": False,
    "fourier_embed_dim": 128,
    "d_model": 128,
    "d_time": 16,
    "query_mlp_hidden_dim": 128,
    "operator_rank": 128,
    "use_geometry_tokens": True,
    "use_hard_body_bc": True,
    "use_body_distance_feature": False,   # YAGNI
    "kolmogorov_A": 0.1,
    "kolmogorov_k_f": 4.0,
})

print("Creating model with use_geometry_tokens=True, use_hard_body_bc=True...")
model = create_picon_model(cfg)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  trainable params: {n_params:,}")
print(f"  net.use_geometry_tokens: {model.use_geometry_tokens}")
print(f"  decoder.use_geometry_tokens: {model.query_decoder.use_geometry_tokens}")
print(f"  decoder has geo_key_proj: {hasattr(model.query_decoder, 'geo_key_proj')}")
print(f"  decoder has geo_value: {hasattr(model.query_decoder, 'geo_value')}")
print(f"  decoder geometry_pos shape: {model.query_decoder.geometry_pos.shape}")

# Inject fake body points (simulate ds.body_xy)
fake_body_xy = torch.rand(50, 2) * 0.1  # 50 body surface points
model.set_geometry_tokens(fake_body_xy)
print(f"  After set_geometry_tokens: geometry_pos shape = {model.query_decoder.geometry_pos.shape}")

# Verify geometry_pos change
assert model.query_decoder.geometry_pos.shape == (50, 2), "geometry_pos injection failed"
print("OK — geometry tokens src patches verified.")
```

- [ ] **Step 7.2: Run smoke test**

```bash
cd /Users/latteine/Documents/coding/pi-lnn && .venv/bin/python /tmp/smoke_geo_tokens.py 2>&1 | head -20
```

Expected:
```
Creating model with use_geometry_tokens=True, use_hard_body_bc=True...
  trainable params: <some int>
  net.use_geometry_tokens: True
  decoder.use_geometry_tokens: True
  decoder has geo_key_proj: True
  decoder has geo_value: True
  decoder geometry_pos shape: torch.Size([0, 2])
  After set_geometry_tokens: geometry_pos shape = torch.Size([50, 2])
OK — geometry tokens src patches verified.
```

If any assertion fails, debug the specific task that introduced the issue (Tasks 2-6).

---

## Task 8: Create `configs/exp_cylinder_022_geometry_tokens.toml`

**Files:**
- Create: `configs/exp_cylinder_022_geometry_tokens.toml`

- [ ] **Step 8.1: Copy CEXP-016 (hard BC only) as baseline**

```bash
cp configs/exp_cylinder_016_hard_bc_fair.toml configs/exp_cylinder_022_geometry_tokens.toml
```

Rationale: CEXP-022 = hard BC (from CEXP-016) + geometry tokens (new). Starting from CEXP-016 means we only add `use_geometry_tokens=true` and `n_geometry_tokens=-1`, nothing else.

- [ ] **Step 8.2: Replace header with CEXP-022 description**

Read first ~30 lines of the new file to find exact old header, then use Edit:

`new_string` for header (replace everything from line 1 to the blank line before `[train]`):

```
# configs/exp_cylinder_022_geometry_tokens.toml
# CEXP-022 = Geometry tokens + hard BC gate (Option E)
#
# 設計目的：
#   CEXP-016 (hard BC only, KE 111%): trunk 不知邊界位置 → NN_u 大 → GradNorm 爆
#   CEXP-021 (SDF trunk + hard BC, KE 174%): 改善但 NS residual 仍高 (w_ns_u=1.96)
#
#   本實驗：Option E — 把 body 表面幾何作為 K-V pool 中的 geometry tokens，
#   讓 cross-attention 自然學「attend body token → output ≈ 0」。
#
#   三個新 module（geo_key_proj / geo_value parameter / geo_token_type_bias）
#   + register_buffer geometry_pos（從 ds.body_xy 在訓練開始前注入）。
#   搭配 hard BC output gate 確保邊界機器精度 = 0。
#
# 與 CEXP-016 差異：
#   use_geometry_tokens: false → true   (新增 Option E modules)
#   n_geometry_tokens:   (新增) -1      (全部 body_xy)
#
# Falsifiability gates:
#   KE < 10%   → ✅ geometry tokens + hard BC 協同成功
#   KE 10-30%  → 🟡 partial; 考慮 per-point geo_value 或更多 iter
#   KE > 30%   → ❌ 設計不夠; 考慮 sensor placement 擴展到 body 區域
#
# 啟動條件與狀態追蹤：見 docs/cylinder_log_v2.md CEXP-022 entry
```

- [ ] **Step 8.3: Add `use_geometry_tokens` and `n_geometry_tokens` keys**

Use Edit — find `# ===== 唯一變動 =====` section (from CEXP-016 header) and replace with geometry tokens addition:

`old_string`:
```
# ===== 唯一變動 =====
use_hard_body_bc = true   # ← 啟用 Sukumar 2022 output transformation
                          #    u, v ← (φ/scale).clamp(0,1) · NN
                          #    p 不 gate
                          # 物理保證 body 內 u=v=0（no-slip hard constraint）
                          # cylinder_dataset 自動 detect body 並預計算 SDF grid
# ====================
```

`new_string`:
```
# ===== CEXP-022 變動 =====
use_geometry_tokens = true   # Option E: body 表面 geometry tokens 加入 cross-attention
n_geometry_tokens = -1       # -1 = 全部 ds.body_xy；正整數 = 取前 n 點
use_hard_body_bc = true      # hard BC output gate 確保 body 內 u=v=0 精確為 0
# ========================
```

- [ ] **Step 8.4: Update `artifacts_dir`**

`old_string`:
```
artifacts_dir = "artifacts/cylinder/deeponet-cfc-cylinder-exp016-hard-bc-fair"
```

`new_string`:
```
artifacts_dir = "artifacts/cylinder/deeponet-cfc-cylinder-exp022-geometry-tokens"
```

- [ ] **Step 8.5: Verify diff vs CEXP-016**

```bash
diff configs/exp_cylinder_016_hard_bc_fair.toml configs/exp_cylinder_022_geometry_tokens.toml | head -60
```

Expected: (a) header changed, (b) use_geometry_tokens+n_geometry_tokens added+use_hard_body_bc retained, (c) artifacts_dir changed. NO other hyperparameter changes.

---

## Task 9: Commit + push src patches + CEXP-022 config

- [ ] **Step 9.1: Stage 5 files by name**

```bash
git add src/pi_con/config.py \
        src/pi_con/operator.py \
        src/pi_con/decoder.py \
        src/pi_con/training.py \
        configs/exp_cylinder_022_geometry_tokens.toml
git status | head -15
```

Expected: 4 modified src files + 1 new config staged. NO other files.

- [ ] **Step 9.2: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(cylinder): CEXP-022 Option E — cross-attention geometry tokens + hard BC

Per docs/superpowers/specs/2026-05-27-cylinder-geometry-tokens-design.md.

Cross-attention geometry tokens: body surface positions (ds.body_xy) as
additional K-V tokens in cross-attention pool. Each geometry token:
- Key: geo_key_proj(Fourier(body_xy)) + geo_token_type_bias
- Value: geo_value (shared learned zero-velocity prior, init=0)
relpos_bias extended to K+N_body tokens uniformly.

Combined with hard BC gate (use_hard_body_bc=True) for boundary enforcement.

Src changes (4 files, ~100 lines):
- config.py: + use_geometry_tokens, n_geometry_tokens keys
- decoder.py: __init__ 3 new modules + geometry_pos buffer;
  forward_uvp + forward: concat geo tokens to K-V + extend relpos_bias
- operator.py: forward flags + set_geometry_tokens() method
- training.py: inject ds.body_xy via model.set_geometry_tokens() at start

Config CEXP-022 = CEXP-016 (hard BC) + use_geometry_tokens=true.
Smoke test: model creates with geometry tokens, set_geometry_tokens works.

Falsifiability: KE < 10% ✅; > 30% ❌ escalate to sensor placement expansion.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 9.3: Push**

```bash
git push origin main 2>&1 | tail -3
```

Expected: `main -> main` with commit SHA.

---

## Task 10: Lab git pull + deployment sed + submit SLURM

- [ ] **Step 10.1: Lab git pull**

```bash
ssh lab-server 'cd pi-lnn && git pull --ff-only 2>&1 | tail -5'
```

Expected: pulls commit with 5-file change + sees `configs/exp_cylinder_022_geometry_tokens.toml`.

- [ ] **Step 10.2: Apply 3 deployment-only sed edits**

```bash
ssh lab-server 'cd pi-lnn && sed -i \
    -e "s|/Users/latteine/Documents/coding/RealPDEBench|/home/junyi/RealPDEBench|g" \
    -e "s|^kolmogorov_A = 0.0$|kolmogorov_A = 1e-6|" \
    -e "s|^kolmogorov_k_f = 0.0$|kolmogorov_k_f = 2.0|" \
    configs/exp_cylinder_022_geometry_tokens.toml && echo "sed OK"'
```

Expected: `sed OK`

- [ ] **Step 10.3: Lab smoke test**

```bash
ssh lab-server 'cd pi-lnn && .venv/bin/python -c "
import sys; sys.path.insert(0, \"src\")
from pi_con.config import DEFAULT_PICON_ARGS
from pi_con.operator import create_picon_model
import torch
cfg = dict(DEFAULT_PICON_ARGS)
cfg[\"use_periodic_domain\"] = False; cfg[\"fourier_embed_dim\"] = 128
cfg[\"d_model\"] = 128; cfg[\"use_geometry_tokens\"] = True
cfg[\"use_hard_body_bc\"] = True; cfg[\"kolmogorov_A\"] = 0.1; cfg[\"kolmogorov_k_f\"] = 4.0
m = create_picon_model(cfg)
m.set_geometry_tokens(torch.rand(50, 2))
print(f\"lab smoke OK: use_geometry_tokens={m.use_geometry_tokens}, geometry_pos={m.query_decoder.geometry_pos.shape}\")
"' 2>&1 | tail -3
```

Expected: `lab smoke OK: use_geometry_tokens=True, geometry_pos=torch.Size([50, 2])`

- [ ] **Step 10.4: Submit CEXP-022**

```bash
ssh lab-server 'cd pi-lnn && scripts/slurm/submit_exp.sh cylinder_022 configs/exp_cylinder_022_geometry_tokens.toml 2>&1 | tail -3 && squeue --me 2>&1 | head -5'
```

Expected: `Submitted batch job <jobid>` + job visible in squeue. Record jobid.

---

## Task 11: Wait + eval + rsync

- [ ] **Step 11.1: Set up background wait for SLURM (replace `<JOBID>` with actual)**

```bash
until ssh lab-server "sacct -j <JOBID> --noheader --format=State 2>/dev/null | head -1 | grep -qvE 'PENDING|RUNNING|^\s*$'" 2>/dev/null; do sleep 180; done && ssh lab-server 'sacct -j <JOBID> --format=State,Elapsed,ExitCode 2>&1 | head -3; tail -12 pi-lnn/logs/exp_cylinder_022_<JOBID>.out 2>&1'
```

Expected: `COMPLETED 01:38:xx 0:0`, final step shows L_data, w_ns_u values.

- [ ] **Step 11.2: Run evaluator**

```bash
ssh lab-server 'cd pi-lnn && .venv/bin/python -u scripts/evaluate_cylinder.py \
  --config configs/exp_cylinder_022_geometry_tokens.toml \
  --checkpoint artifacts/cylinder/deeponet-cfc-cylinder-exp022-geometry-tokens/picon_kolmogorov_final.pt \
  2>&1 | tail -15'
```

Expected: Evaluation table with KE rel-err, u/v/ω RMSE, div L2.

- [ ] **Step 11.3: Rsync artifact back to local**

```bash
rsync -avz lab-server:/home/junyi/pi-lnn/artifacts/cylinder/deeponet-cfc-cylinder-exp022-geometry-tokens/ \
  /Users/latteine/Documents/coding/pi-lnn/artifacts/cylinder/deeponet-cfc-cylinder-exp022-geometry-tokens/ \
  2>&1 | tail -3
```

- [ ] **Step 11.4: Verify summary.json locally**

```bash
jq '{ke_rel_err_mean, ke_rel_err_late, ke_pred_mean, ke_ref_mean, u_rmse_mean, v_rmse_mean, omega_rmse_mean, div_l2_mean}' \
  artifacts/cylinder/deeponet-cfc-cylinder-exp022-geometry-tokens/cylinder-eval/summary.json
```

---

## Task 12: Decision tree + update cylinder_log_v2 + commit

- [ ] **Step 12.1: Decision tree judgment**

Per spec §4 thresholds:

| ke_rel_err_mean | Outcome | Status label |
|---|---|---|
| < 10% | ✅ A | `ACTIVE_REFERENCE` |
| 10-30% | 🟡 B | `PARTIAL_RESULT` |
| > 30% | ❌ C | `NEGATIVE_RESULT` |

Additional checks:
- `w_ns_u_final` from training log `grep "^10000 " logs/exp_cylinder_022_<JOBID>.out`
- `ke_pred_mean / ke_ref_mean` ratio: healthy if ∈ [0.85, 1.15]

- [ ] **Step 12.2: Update cylinder_log_v2.md [INDEX] — add CEXP-022 row**

Edit `docs/cylinder_log_v2.md`, add after CEXP-021 row:

```markdown
| **CEXP-022** | `<status>` | Re=10031, **geometry tokens** (Option E) + hard BC | <KE>% | <ratio> | <ω> | <div> | 10k | **<outcome summary>** |
```

Replace `<status>`, `<KE>`, `<ratio>`, `<ω>`, `<div>`, `<outcome summary>` with actual values.

- [ ] **Step 12.3: Add [RECORD] detail section for CEXP-022 (after CEXP-021 detail)**

```markdown
### CEXP-022：Geometry tokens + hard BC (Option E, KE=<X.X>%, <outcome>)

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_022_geometry_tokens.toml` |
| Artifact | `artifacts/cylinder/deeponet-cfc-cylinder-exp022-geometry-tokens/` |
| KE rel-err mean / late | <X.X>% / <X.X>% |
| u / v RMSE | <X.X> / <X.X> |
| ω RMSE | <X.X> |
| div L2 | <X.X> |
| ke_pred / ke_ref | <ratio> |
| GradNorm w_ns_u final | <X.X> |
| 設計 | cross-attn K-V pool += [N_body geometry tokens] (key=geo_key_proj(Fourier(body_xy))+geo_token_type_bias, value=geo_value shared parameter) + hard BC gate |
| 結論 | **<A/B/C per spec §4>**: <one-sentence interpretation> |
```

Replace placeholders with actual values from Step 11.4 + 12.1.

- [ ] **Step 12.4: Update [STATE] Surprise Findings and Open Questions**

Add Surprise Finding #6 after Finding #5:

```markdown
### Finding 6 — Option E Geometry Tokens outcome (CEXP-022, 2026-05-27)

CEXP-022 = CEXP-016 (hard BC) + use_geometry_tokens=true. Geometry tokens:
[body surface positions → key, shared geo_value zero-velocity prior → value].

Result: KE <X.X>%, ke_pred/ke_ref <ratio>, w_ns_u_final <X.X>
Outcome: **<✅ A / 🟡 B / ❌ C>** per spec §4.

<Interpretation paragraph per actual outcome>
```

Update [STATE] Open Questions to reflect CEXP-022 result.

- [ ] **Step 12.5: Update changelog at end of v2 log**

Add entry:
```
- **2026-05-27 CEXP-022 Option E**:
  - Option E (cross-attention geometry tokens) added to src (4 files, ~100 lines)
  - CEXP-022: KE <X.X>%, outcome <A/B/C>
  - Finding #6 added
  - Open Questions updated
```

- [ ] **Step 12.6: Stage + commit + push**

```bash
git add docs/cylinder_log_v2.md
git commit -m "$(cat <<'EOF'
docs(v2): CEXP-022 Option E geometry tokens result (KE <X.X>%, outcome <A/B/C>)

CEXP-022 = CEXP-016 hard BC + use_geometry_tokens=true.
Geometry tokens: body surface points as K-V pool tokens in cross-attention.
Key = geo_key_proj(Fourier(body_xy)) + geo_token_type_bias (type disambiguation).
Value = shared geo_value parameter (zero-velocity prior, init=0, learned).

<One-sentence interpretation based on actual outcome>

[INDEX] CEXP-022 row; [RECORD] detail; Finding #6; Open Questions updated.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git push origin main 2>&1 | tail -3
```

Replace `<X.X>%` and `<A/B/C>` with actual values.

---

## Spec Coverage Self-Check

| Spec requirement | Plan task |
|---|---|
| §1 Goal: geometry tokens in K-V pool | Tasks 2-4 (decoder) + 6 (training inject) |
| §2 `geo_key_proj` module | Task 2 |
| §2 `geo_value` shared parameter | Task 2 |
| §2 `geo_token_type_bias` | Task 2 |
| §2 `geometry_pos` buffer | Task 2 |
| §2 `forward_uvp` geometry token concat | Task 3 |
| §2 `forward` geometry token concat | Task 4 |
| §2 `relpos_bias` extended to K+N_body | Tasks 3, 4 |
| §2 `set_geometry_tokens()` method | Task 5 |
| §2 `use_hard_body_bc=True` | CEXP-022 config (Task 8) — hard BC already wired |
| §3 Config `use_geometry_tokens=true`, `n_geometry_tokens=-1` | Task 8 |
| §4 Falsifiability judgment | Task 12.1 |
| §5 `training.py` body_xy injection | Task 6 |
| §5 Lab deployment | Task 10 |
| §5 Ckpt cold start (no resume) | Task 9 push after Task 7 smoke pass |
| §5 `use_body_distance_feature=False` | CEXP-022 config inherits from CEXP-016 (not set) |

---

## Stop Loss

訓練中不 early stop。但若：
- Task 7 smoke test fails → debug Tasks 2-6 before proceeding
- Task 10.3 lab smoke fails → check .venv Python path or import errors
- Task 10.4 submit fails → check err log for ForcingPrior regression (same A_init/k_f issue as earlier — check if config has kolmogorov_A/k_f properly sed'd on lab)
- Task 11.1 SLURM FAILED → check err log; likely import error in src patches
- Task 12.1 KE > 100% → write up as negative finding; consider sensor placement diversification
