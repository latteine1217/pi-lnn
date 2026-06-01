# CEXP-039/040 Sensor-Conditioned FiLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans。Steps 用 checkbox。

**Goal:** 用 sensor hidden state 經 FiLM 大域調制 decoder trunk feature，做 B3-a（pure sensor）vs B3-b（+geometry）ablation。

**Architecture:** decoder trunk blocks 後對 trunk_feat 套 `γ(cond)⊙feat+β(cond)`，cond 從 per-query sensor hidden（B3-a）或 sensor hidden+body_distance（B3-b）。γ identity-init（γ=1,β=0）確保 use_sensor_film=false 時行為不變。

**Tech Stack:** Python, PyTorch, SLURM r740, uv

---

## File Structure
- `src/pi_con/config.py` — 加 2 flags
- `src/pi_con/decoder.py` — FiLM module + 2 forward path 套用
- `src/pi_con/operator.py` — 轉送 flags
- `configs/exp_cylinder_039_sensor_film.toml` / `exp_cylinder_040_sensor_film_geo.toml`
- `tests/test_sensor_film_identity.py` — identity-init 不破壞既有行為

---

### Task 1: Config flags

**Files:** Modify `src/pi_con/config.py`

- [ ] **Step 1: 在 DEFAULT_PICON_ARGS 加 flags（緊接 use_re_film 後）**

找到 `"use_re_film": False,` 那行，其後加：
```python
    "use_sensor_film": False,    # B3: sensor hidden state 經 FiLM 調制 decoder trunk feature。
                                 # 預設 false 向後相容（FiLM γ identity-init，啟用不破壞 baseline）。
    "film_use_geometry": False,  # B3-b: FiLM conditioning 額外 concat body_distance（明示 geometry）。
                                 # 僅在 use_sensor_film=true 時生效。
```

- [ ] **Step 2: 驗證 import**

Run: `cd /Users/latteine/Documents/coding/pi-lnn && uv run python -c "from pi_con.config import DEFAULT_PICON_ARGS as D; print(D['use_sensor_film'], D['film_use_geometry'])"`
Expected: `False False`

---

### Task 2: FiLM module in decoder __init__

**Files:** Modify `src/pi_con/decoder.py`

- [ ] **Step 1: decoder __init__ 簽名加參數**

找到 `geometry_preserve_base_rng: bool = False,` 那行（__init__ 簽名末），其後加：
```python
        use_sensor_film: bool = False,
        film_use_geometry: bool = False,
```

- [ ] **Step 2: __init__ body 建 FiLM module**

找到 `self.use_trunk_geo_context = bool(use_trunk_geo_context)` 那行附近（__init__ body），加：
```python
        # B3: Sensor-conditioned FiLM. cond = pooled sensor hidden (+ body_distance if geometry).
        # γ identity-init (γ=1, β=0) → use_sensor_film=False 時 self.film_mlp 不建立，
        # 啟用時初始輸出 γ=1,β=0 不破壞既有 trunk_feat。
        self.use_sensor_film = bool(use_sensor_film)
        self.film_use_geometry = bool(film_use_geometry)
        if self.use_sensor_film:
            # cond_dim: d_model (pooled sensor hidden) + (1 if geometry else 0)
            _cond_dim = query_mlp_hidden_dim + (1 if self.film_use_geometry else 0)
            self.film_mlp = nn.Sequential(
                nn.Linear(_cond_dim, query_mlp_hidden_dim),
                nn.SiLU(),
                nn.Linear(query_mlp_hidden_dim, 2 * query_mlp_hidden_dim),
            )
            # identity init: 最後一層 weight=0, bias=[1...1(γ), 0...0(β)]
            nn.init.zeros_(self.film_mlp[-1].weight)
            with torch.no_grad():
                self.film_mlp[-1].bias[:query_mlp_hidden_dim].fill_(1.0)   # γ=1
                self.film_mlp[-1].bias[query_mlp_hidden_dim:].fill_(0.0)   # β=0
```

- [ ] **Step 3: 加 FiLM 套用 helper method**

在 class 內（`_apply_trunk_geo_context` method 附近）加：
```python
    def _apply_sensor_film(
        self,
        trunk_feat: torch.Tensor,      # [M, hidden]，M = N 或 3N
        h_branch_tokens: torch.Tensor, # [N, d_model] per-query sensor hidden
        body_distance: torch.Tensor | None,  # [N, 1] or None
        n_repeat: int,                 # 1 (forward) or 3 (forward_uvp)
    ) -> torch.Tensor:
        """B3: γ(cond)⊙feat + β(cond)，cond 從 sensor hidden (+geometry)。"""
        if not self.use_sensor_film:
            return trunk_feat
        cond = h_branch_tokens  # [N, hidden]（d_model == query_mlp_hidden_dim）
        if self.film_use_geometry:
            if body_distance is None:
                raise ValueError("film_use_geometry=True 但 body_distance=None")
            cond = torch.cat([cond, body_distance.reshape(-1, 1)], dim=-1)  # [N, hidden+1]
        gb = self.film_mlp(cond)                                    # [N, 2·hidden]
        hidden = gb.shape[-1] // 2
        gamma, beta = gb[:, :hidden], gb[:, hidden:]               # [N, hidden] each
        if n_repeat > 1:
            gamma = gamma.repeat(n_repeat, 1)                      # [3N, hidden]
            beta = beta.repeat(n_repeat, 1)
        return gamma * trunk_feat + beta
```
注意 `d_model == query_mlp_hidden_dim`（CEXP-002 兩者都 256），故 h_branch_tokens 直接當 cond。若不等需 raise，下步驗證。

- [ ] **Step 4: __init__ 加 d_model 一致性檢查**

在 Step 2 的 `if self.use_sensor_film:` block 開頭加：
```python
            if d_model != query_mlp_hidden_dim:
                raise ValueError(
                    f"use_sensor_film 需要 d_model({d_model}) == query_mlp_hidden_dim"
                    f"({query_mlp_hidden_dim})；否則 sensor hidden 維度與 trunk 不符。"
                )
```
（確認 __init__ 簽名有 `d_model` 參數；若無，用 `h_states` 的維度——查既有簽名）

- [ ] **Step 5: Commit（暫不套用，先確保 import 不壞）**

Run: `cd /Users/latteine/Documents/coding/pi-lnn && uv run python -c "import pi_con.decoder; print('import OK')"`
Expected: `import OK`

---

### Task 3: 套用 FiLM 到 forward_uvp（3N path）

**Files:** Modify `src/pi_con/decoder.py`

- [ ] **Step 1: forward_uvp 內 trunk blocks 後套 FiLM**

找到 forward_uvp 內 `trunk_feat = self._apply_trunk_geo_context(trunk_feat, xy_3)` 那行（line ~347），在其**前**插入：
```python
        trunk_feat = self._apply_sensor_film(trunk_feat, h_branch_tokens, body_distance, n_repeat=3)
```

- [ ] **Step 2: 確認 h_branch_tokens 在 scope 內**

forward_uvp 早段已有 `h_branch_tokens = h_states[idx]`（line ~293），在 scope 內。body_distance 也在簽名。無需改 scope。

---

### Task 4: 套用 FiLM 到 forward（N path）

**Files:** Modify `src/pi_con/decoder.py`

- [ ] **Step 1: forward 內 trunk blocks 後套 FiLM**

找到 forward 內 `trunk_feat = self._apply_trunk_geo_context(trunk_feat, xy)` 那行（line ~496），在其**前**插入：
```python
        trunk_feat = self._apply_sensor_film(trunk_feat, h_branch_tokens, body_distance, n_repeat=1)
```

- [ ] **Step 2: 確認 forward 的 h_branch_tokens / body_distance 在 scope**

forward 早段有 `h_branch_tokens = h_states[idx]`（line ~459），body_distance 在簽名。OK。

---

### Task 5: Operator 轉送 flags

**Files:** Modify `src/pi_con/operator.py`

- [ ] **Step 1: LiquidOperator __init__ 簽名加參數**

找到 `use_trunk_geo_context: bool = False,` 那行（operator __init__ 簽名），其後加：
```python
        use_sensor_film: bool = False,
        film_use_geometry: bool = False,
```

- [ ] **Step 2: DeepONetCfCDecoder(...) 呼叫加轉送**

找到 `use_trunk_geo_context=self.use_trunk_geo_context,`（line ~129，decoder 實例化），其後加：
```python
            use_sensor_film=bool(use_sensor_film),
            film_use_geometry=bool(film_use_geometry),
```

- [ ] **Step 3: create_picon_model 讀 config**

找到 `create_picon_model` 內讀 `use_trunk_geo_context` 的地方（`cfg.get("use_trunk_geo_context"...`），其附近加：
```python
        use_sensor_film=bool(cfg.get("use_sensor_film", False)),
        film_use_geometry=bool(cfg.get("film_use_geometry", False)),
```

- [ ] **Step 4: import 驗證**

Run: `cd /Users/latteine/Documents/coding/pi-lnn && uv run python -c "import pi_con.operator; print('OK')"`
Expected: `OK`

---

### Task 6: Identity-init regression test

**Files:** Create `tests/test_sensor_film_identity.py`

- [ ] **Step 1: 寫 test — FiLM 啟用但 identity-init 應與 baseline 數值接近**

```python
"""use_sensor_film=True 但 identity-init (γ=1,β=0) 時，trunk_feat 不被改變。
驗證 _apply_sensor_film 在初始狀態是 identity transform。"""
import torch
from pi_con.decoder import DeepONetCfCDecoder


def _mk(use_film, film_geo):
    torch.manual_seed(0)
    return DeepONetCfCDecoder(
        spatial_dim=32, temporal_dim=4, d_model=64, rank=64,
        query_mlp_hidden_dim=64, fourier_embed_dim=32,
        use_periodic_domain=False,
        use_sensor_film=use_film, film_use_geometry=film_geo,
    )


def test_film_module_built_only_when_enabled():
    d_off = _mk(False, False)
    assert not hasattr(d_off, "film_mlp")
    d_on = _mk(True, False)
    assert hasattr(d_on, "film_mlp")


def test_film_identity_init_is_identity():
    d = _mk(True, False)
    feat = torch.randn(5, 64)
    h = torch.randn(5, 64)          # per-query sensor hidden
    out = d._apply_sensor_film(feat, h, None, n_repeat=1)
    # identity-init: γ=1,β=0 → out == feat
    assert torch.allclose(out, feat, atol=1e-6), f"max diff {(out-feat).abs().max()}"


def test_film_geometry_requires_distance():
    d = _mk(True, True)
    feat = torch.randn(5, 64); h = torch.randn(5, 64)
    try:
        d._apply_sensor_film(feat, h, None, n_repeat=1)
        assert False, "應 raise（film_use_geometry=True 但 body_distance=None）"
    except ValueError:
        pass
```

- [ ] **Step 2: Run test**

Run: `cd /Users/latteine/Documents/coding/pi-lnn && uv run pytest tests/test_sensor_film_identity.py -v`
Expected: 3 passed。若 DeepONetCfCDecoder 簽名參數名不符（spatial_dim/temporal_dim/d_model 等），依實際簽名調整 `_mk`。

- [ ] **Step 3: Commit src + test**

```bash
cd /Users/latteine/Documents/coding/pi-lnn
git add src/pi_con/config.py src/pi_con/decoder.py src/pi_con/operator.py tests/test_sensor_film_identity.py
git commit -m "feat(cylinder): sensor-conditioned FiLM (CEXP-039/040 B3)

FiLM modulates trunk_feat by gamma(cond)*feat+beta(cond), cond from
per-query sensor hidden state (+body_distance if film_use_geometry).
gamma identity-init keeps use_sensor_film=False behavior unchanged.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Configs

**Files:** Create both config files (由 CEXP-002 派生，+flags)

- [ ] **Step 1: CEXP-039（B3-a pure sensor）**

複製 `configs/exp_cylinder_002_k100_bc.toml` → `configs/exp_cylinder_039_sensor_film.toml`，改：
- `bc_body_n_points = 0`（移除 body soft BC）
- 加 `use_sensor_film = true`、`film_use_geometry = false`
- `device = "cuda"`
- `artifacts_dir = "artifacts/cylinder/deeponet-cfc-cylinder-exp039-sensor-film"`

- [ ] **Step 2: CEXP-040（B3-b +geometry）**

複製 CEXP-039 → `configs/exp_cylinder_040_sensor_film_geo.toml`，改：
- `film_use_geometry = true`
- `artifacts_dir = "artifacts/cylinder/deeponet-cfc-cylinder-exp040-sensor-film-geo"`

- [ ] **Step 3: 驗證 diff vs CEXP-002**

Run: `cd /Users/latteine/Documents/coding/pi-lnn && diff <(grep -v "^#" configs/exp_cylinder_002_k100_bc.toml|grep -v "^$"|sed 's/  *#.*//') <(grep -v "^#" configs/exp_cylinder_039_sensor_film.toml|grep -v "^$"|sed 's/  *#.*//')`
Expected: 差異 = bc_body_n_points, use_sensor_film, film_use_geometry, device, artifacts_dir。

- [ ] **Step 4: Commit + push**

```bash
cd /Users/latteine/Documents/coding/pi-lnn
git add configs/exp_cylinder_039_sensor_film.toml configs/exp_cylinder_040_sensor_film_geo.toml
git commit -m "exp: CEXP-039/040 sensor FiLM configs (B3-a/b ablation)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

---

### Task 8: Lab deploy + submit（並行）

- [ ] **Step 1: Pull + sed 兩 config**

```bash
ssh lab-server 'cd pi-lnn && git stash 2>/dev/null; git pull; git stash drop 2>/dev/null
for c in exp_cylinder_039_sensor_film exp_cylinder_040_sensor_film_geo; do
  sed -i "s|/Users/latteine/Documents/coding/RealPDEBench|/home/junyi/RealPDEBench|g" configs/$c.toml
  sed -i "s|kolmogorov_A = 0.0|kolmogorov_A = 1e-6|" configs/$c.toml
  sed -i "s|kolmogorov_k_f = 0.0|kolmogorov_k_f = 2.0|" configs/$c.toml
done
grep -E "use_sensor_film|film_use_geometry" configs/exp_cylinder_039_sensor_film.toml configs/exp_cylinder_040_sensor_film_geo.toml'
```
Expected: 039 → film true / geo false；040 → film true / geo true。

- [ ] **Step 2: Smoke test（lab 端 1-step，確認 FiLM 不 crash）**

```bash
ssh lab-server 'cd pi-lnn && /home/junyi/.local/bin/uv run python -c "
import toml
from pi_con.operator import create_picon_model
cfg = toml.load(\"configs/exp_cylinder_040_sensor_film_geo.toml\")[\"train\"]
m = create_picon_model(cfg)
print(\"model built, use_sensor_film:\", m.query_decoder.use_sensor_film, \"geo:\", m.query_decoder.film_use_geometry)
"'
```
Expected: `model built, use_sensor_film: True geo: True`（若 create_picon_model 簽名不同，依實際調整）。

- [ ] **Step 3: Submit 兩 job**

```bash
ssh lab-server 'cd pi-lnn && bash scripts/slurm/submit_exp.sh cylinder_039 configs/exp_cylinder_039_sensor_film.toml 2>&1 | tail -2; bash scripts/slurm/submit_exp.sh cylinder_040 configs/exp_cylinder_040_sensor_film_geo.toml 2>&1 | tail -2; squeue -u junyi'
```
記下兩個 job ID（以 squeue 為準，非預期 ID）。

---

## Post-train checklist
訓練 ~1.6 hr。完成後對每個 exp：
1. Eval（r740 SLURM，checkpoint `picon_kolmogorov_final.pt`）
2. **先確認 job ID + summary.json 真實路徑（可能 nested）再讀數字**（CEXP-036/037/038 鐵律）
3. 判讀（§4 gates）+ ablation 對比（039 vs 040）+ 更新 cylinder_log_v2.md
4. 若任一成功，看 vorticity 圖確認圓柱邊緣人工剪切層是否消失
