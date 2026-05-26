# Design — Cylinder Cross-Attention Geometry Tokens (CEXP-022)

| Field | Value |
|---|---|
| Date | 2026-05-27 |
| Status | Approved |
| Owner | latteine |
| Related state | [docs/cylinder_log_v2.md](../../cylinder_log_v2.md) Finding #5 |
| Predecessor | CEXP-021 (KE 174%, SDF trunk + hard BC) |

---

## 1. Goal & Background

**Goal**: 把 cylinder body 表面點作為 geometry tokens 加入 cross-attention K-V pool，讓 decoder 在 attend sensor tokens 同時也能 attend「body 表面 → 速度應為 0」先驗，搭配 SDF hard BC output gate 確保邊界精確為 0。

**Background (實驗進展)**:

| Exp | 機制 | KE rel-err | 失敗原因 |
|---|---|---|---|
| CEXP-002 | Soft BC only (baseline) | **3.54%** ✅ | — |
| CEXP-016 | Hard BC only | 111% | Trunk 不知邊界位置, GradNorm 爆 |
| CEXP-020 | SDF trunk input only | 405% | 無 zero-velocity prior, adversarial signal |
| CEXP-021 | SDF trunk + hard BC | 174% | 改善但 NS residual 仍高 (w_ns_u=1.96) |
| **CEXP-022** | **Geometry tokens + hard BC** | **TBD** | — |

**Root cause synthesis**: Cross-attention 只有 K=100 wake sensor tokens（x > 0.10），body 邊界無 token 覆蓋。任何 query 靠近 body 都得到低 attention weight → 重建全靠 trunk MLP 猜 → 猜錯。

**Geometry tokens 的原理**: 直接在 K-V pool 加入「body 表面位置 → 零速度值」的 token，model 透過 attention weight 自然學「attend body token → output ≈ 0」。這是 information（有 zero-velocity prior）而非 hint（SDF input 只有距離資訊）。

---

## 2. Architecture

### 新 modules（`DeepONetCfCDecoder.__init__`）

| Module | Type | Shape | 用途 |
|---|---|---|---|
| `geo_key_proj` | `nn.Linear(fourier_embed_dim, query_mlp_hidden_dim)` | — | body_xy → Fourier encoding → attention key space |
| `geo_value` | `nn.Parameter` | `[1, query_mlp_hidden_dim]` | 所有 body token 共用的「零速度先驗」向量（learned initialization） |
| `geo_token_type_bias` | `nn.Parameter` | `[query_mlp_hidden_dim]` | 區分 sensor token vs geometry token（key 加 bias），讓 model 可 attend 不同方式 |
| `geometry_pos` | `register_buffer(persistent=False)` | `[N_body, 2]` | body surface 正規化座標（從 `ds.body_xy` 注入，不進 ckpt state_dict） |

**flag**:
- `use_geometry_tokens: bool = False` — 新 config flag，控制是否啟用所有 geometry token 邏輯
- `n_geometry_tokens: int = -1` — 負數表示全部使用 `body_xy`；正整數表示取前 n 點（未來擴展用）

### Forward path（`forward_uvp` + `forward` 兩個 path 皆需修改）

```python
# In decoder.forward_uvp (forward 同邏輯，下同)

# 1. Geometry key：Fourier encode body positions → key projection + token type bias
if self.use_geometry_tokens and self.geometry_pos.shape[0] > 0:
    N_body = self.geometry_pos.shape[0]
    body_enc = self.spatial_emb(self.geometry_pos, self.domain_length)        # [N_body, fourier_embed_dim]
    geo_k_raw = self.geo_key_proj(body_enc) + self.geo_token_type_bias       # [N_body, query_mlp_hidden_dim]
    geo_v_raw = self.geo_value.expand(N_body, -1)                            # [N_body, query_mlp_hidden_dim]
    # broadcast to [N_q, N_body, d] for attention batch
    geo_k = geo_k_raw.unsqueeze(0).expand(N_q, -1, -1)                      # [N_q, N_body, H]
    geo_v = geo_v_raw.unsqueeze(0).expand(N_q, -1, -1)                      # [N_q, N_body, H]

    # 2. Concat to sensor K-V pool
    k_all = torch.cat([k_proj, geo_k], dim=1)    # [N_q, K+N_body, H]
    v_all = torch.cat([v_proj, geo_v], dim=1)    # [N_q, K+N_body, H]

    # 3. Extend relpos bias to geometry tokens
    geo_rel = xy.unsqueeze(1) - self.geometry_pos.unsqueeze(0)              # [N_q, N_body, 2]
    geo_rel_r = torch.sqrt((geo_rel**2).sum(-1, keepdim=True) + 1e-8)      # [N_q, N_body, 1]
    geo_bias = self.relpos_bias(geo_rel_r).squeeze(-1)                      # [N_q, N_body]
    rel_bias_all = torch.cat([rel_bias, geo_bias], dim=1)                   # [N_q, K+N_body]
else:
    k_all, v_all, rel_bias_all = k_proj, v_proj, rel_bias  # fallback = unchanged

# Attention runs over K+N_body tokens (or K if geometry disabled)
```

注意：
- `self.spatial_emb` 已存在（handles non-periodic domain 的 `FourierEmbs`）— **不需要新 encoder**，直接重用
- `rel_bias` 延伸到 geometry tokens：靠近 body token 的 query 會被 relpos_bias 拉高 attention weight → model 更強地 attend 到 body token
- forward_uvp 與 forward 兩者的 3N batch logic 都需對應延伸

### Hard BC gate（同 Stage 1，不變）

```
u_final = clamp(φ(x,y) / scale, 0, 1) · NN_u(x, y, t, c)
```

Geometry tokens 提供 attention-level soft prior；hard BC gate 確保 body 內 u=v=0 machine-precision。兩者協同，前者幫 trunk 學到正確的 NN_u（near-body 自然小），後者確保輸出精確。

### 為何這次 GradNorm 不應該爆

CEXP-016 爆的機制：trunk 不知邊界 → NN_u 大 → gate 壓 → physics residual 大 → GradNorm 推 w_ns_u。

CEXP-022：
- Query 靠近 body → attend to body geometry tokens (key = body position encoding) → output NN_u 受「零速度先驗」value 拉向 0 → NN_u 在 body 附近自然較小 → gate 壓的量 = `gate · NN_u` 其中 NN_u 已較小 → physics residual 小 → GradNorm w_ns_u 不爆

### Body positions 注入（training.py）

```python
# After model creation, for cylinder dataset:
if args.get("use_geometry_tokens", False):
    for ds, net_i in zip(datasets, [...]):
        if hasattr(ds, "body_xy"):
            body_pos = torch.tensor(ds.body_xy, dtype=torch.float32, device=device)
            n_geo = args.get("n_geometry_tokens", -1)
            if n_geo > 0 and n_geo < body_pos.shape[0]:
                body_pos = body_pos[:n_geo]  # subsample if needed
            net_i.query_decoder.geometry_pos = body_pos  # direct assign (persistent=False buffer)
```

No ckpt interference: `geometry_pos` 是 `persistent=False` buffer，不進 state_dict。

---

## 3. Config (CEXP-022)

File: `configs/exp_cylinder_022_geometry_tokens.toml`

| Variable | Value | 備註 |
|---|---|---|
| `use_geometry_tokens` | **true** | 新 flag |
| `n_geometry_tokens` | **-1** | 全部 body_xy |
| `use_hard_body_bc` | **true** | 保留 Stage 1 hard BC gate |
| `use_body_distance_feature` | false | 不用（YAGNI, CEXP-020 更糟）|
| GradNorm tasks | 4-task | 對齊 CEXP-002 baseline |
| iterations | 10000 | single-seed first pass |
| seed | 42 | |
| 其餘 | 與 CEXP-002 對齊 | |

---

## 4. Falsifiability Gates

| KE rel-err | Outcome | 解讀 |
|---|---|---|
| **< 10%** | ✅ A | Geometry tokens + hard BC 協同成功；architecture-level geometry awareness 到位；可進 multi-seed 確認 |
| 10-30% | 🟡 B | 部分改善；考慮 per-point value (`nn.Embedding(N_body, H)`) 或更長 training |
| > 30% | ❌ C | 設計不夠；考慮 sensor placement 擴展（增加 body-adjacent sensors） |
| > 100% (同 Stage 1) | 🛑 Stop-loss | 回 brainstorm；問題更深 |

額外診斷量：
- `w_ns_u_final` < 0.5 → GradNorm 沒爆（最關鍵信號）
- `ke_pred / ke_ref` ∈ [0.85, 1.15] → 不 over-predict

---

## 5. Prerequisites & Workflow

### Src 改動（~80-100 lines, 4 files）

| File | 改動 |
|---|---|
| `src/pi_con/config.py` | 加 `use_geometry_tokens: False` + `n_geometry_tokens: -1` |
| `src/pi_con/decoder.py` | `__init__`: 3 新 module; `forward_uvp` + `forward`: concat geo tokens to K-V pool + extend rel_bias |
| `src/pi_con/operator.py` | `LiquidOperator.__init__` + `create_picon_model` 轉送新 flags |
| `src/pi_con/training.py` | 訓練開始前注入 `ds.body_xy` 到 `model.query_decoder.geometry_pos` |

### Out-of-scope

- ❌ `use_body_distance_feature` (SDF trunk input, dropped)
- ❌ Multi-seed (single-seed first pass)
- ❌ Per-point geometry value embedding (YAGNI)
- ❌ Option F (geometry-conditioned hypernetwork) — future paper
- ❌ `evaluate_cylinder.py` 無需修改（hard BC + body_distance wiring 已在 8a4ca86 修好）

### Lab deployment（同 Stage 2 pattern）

- Lab 已有 arrow shards + cylinder sensor files + `.venv`
- Deployment: lab in-place sed (arrow_shards + kolmogorov dummy A/k_f)
- Submit: `scripts/slurm/submit_exp.sh cylinder_022 configs/exp_cylinder_022_geometry_tokens.toml`

### Ckpt incompatibility

3 個新 parameters (`geo_key_proj`, `geo_value`, `geo_token_type_bias`) 不在舊 ckpt → **cold start** 訓練（無法 resume from CEXP-021）。

---

## Next Steps

After spec approval → `superpowers:writing-plans` → implementation plan → `superpowers:subagent-driven-development` 執行。

Estimated GPU: ~1.6 hr (same as previous cylinder experiments). Total wall: ~3 hr.
