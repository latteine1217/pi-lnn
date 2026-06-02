# 3D Channel Flow 重建 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 pi-lnn CfC-DeepONet pipeline 上新增 3D turbulent channel flow (Re_τ=1000) 稀疏重建 case，sensor + 3D NS residual + wall BC 訓練，對照 JHTDB DNS。

**Architecture:** 混合策略 — physics/decoder/config 維度泛化（2D 是 3D 特例）；channel dataset/wall-BC/evaluator 新建（follow kolmogorov/cylinder 既有 pattern）。LES (minimal 2π×π) QR-pivot 佈點 tile×(4×3) 到 DNS full domain。

**Tech Stack:** PyTorch (MPS), torch.autograd 二階導, JHTDB givernylocal/getCutout, pyvista (VTU 後處理), uv。

**測試指令前綴：** pytest 設定無 `pythonpath`，pi_con 在 `src/` 未安裝為 editable → 所有測試指令必須加 `PYTHONPATH=src`。標準前綴：`PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python -m pytest ...`

**Spec:** [docs/superpowers/specs/2026-06-02-3d-channel-flow-reconstruction-design.md](../specs/2026-06-02-3d-channel-flow-reconstruction-design.md)

**Git note:** 本專案 `GIT_POLICY` 為「不主動 git」。各 Task 的 commit step 列出供執行者參考；實際 commit 由使用者確認後執行。

---

## Sub-Plan 拆分（scope check）

此 case 涵蓋六個子系統，拆成 5 個獨立可測 sub-plan：

| Plan | 範圍 | 外部依賴 | 狀態 |
|---|---|---|---|
| **Plan 1**（本檔詳述） | `physics.py` 3D channel NS residual + manufactured-solution TDD | 無（純函數） | 詳細展開 |
| Plan 2 | `decoder.py` + `config.py` + `operator.py` 3D 泛化（4 分量 uvwp、input_dim=3、Lz/num_spatial_dims keys）+ 2D regression | Plan 1 | stub |
| Plan 3 | JHTDB DNS 抓取 script + LES VTU→grid + QR-pivot tile + sensor 生成 + 3D axis-convention test | JHTDB token、home-gpu | stub |
| Plan 4 | `channel_dataset.py` + wall BC loss (training.py) + loss 組裝 + channel config | Plan 1,2,3 | stub |
| Plan 5 | `evaluate_channel.py`（U⁺(y⁺)、Reynolds stress）+ smoke + coarse benchmark | Plan 1-4 | stub |

**執行順序**：Plan 1 → Plan 2 → Plan 4 → Plan 5；Plan 3 可與 1/2 平行（卡 token，先動程式碼）。

Plan 1 是基礎且零阻塞，本檔完整展開。Plan 2-5 在 Plan 1 完成後逐一展開（各自一份 plan 檔）。

---

## Plan 1：physics.py 3D Channel NS Residual

### 設計決策

- **新增** `channel_ns_residuals`（不改既有 2D `unsteady_ns_residuals`）→ 既有 Kolmogorov/Cylinder zero regression risk。advection/viscous 計算與 2D 版有重複，但 YAGNI：先正確跑通；future 若要 DRY 可抽 helper。
- **Forcing**：channel 用 **constant streamwise body force** `body_force_x`（取代 mean dP/dx，對應 Lethe flow-control），物理值 = `u_τ²/h`（Re_τ=1000 → 0.0499²/1.0 ≈ 2.49e-3）。非 Kolmogorov 的 `A·sin(k_f·y)`。
- **座標 chain rule**：`xyzt` normalized 到 [0,1]，model output 物理量級。`d/dx_phys = d/dx_norm / Lx`，二階除 `Lx²`。channel domain `Lx=8π, Ly=2, Lz=3π`（full）。
- **介面**：`uvwp_fn(xyzt[N,4]) -> [N,4]`（u,v,w,p）；回傳 `(mom_u, mom_v, mom_w, cont)`。

### Task 1: channel_ns_residuals 函數 + manufactured-solution TDD

**Files:**
- Modify: `src/pi_con/physics.py`（在 `unsteady_ns_residuals` 之後新增函數，import 區已有 `_grad`）
- Test: `tests/test_channel_ns_residual.py`（新建）

- [ ] **Step 1: 寫 failing tests**

建立 `tests/test_channel_ns_residual.py`：

```python
"""3D channel NS residual — method of manufactured solutions 驗證。

每個 test 餵入解析已知場給 channel_ns_residuals，比對 residual 各項。
用 float64 提高二階 autograd 精度。
"""
import math

import torch

from pi_con.physics import channel_ns_residuals


def _make_xyzt(n: int, seed: int = 0) -> torch.Tensor:
    """產生 [n,4] = (x,y,z,t) normalized 隨機座標，requires_grad 供 autograd。"""
    g = torch.Generator().manual_seed(seed)
    xyzt = torch.rand(n, 4, generator=g, dtype=torch.float64)
    xyzt.requires_grad_(True)
    return xyzt


def test_constant_field_only_forcing():
    """常數場 → 所有導數 0 → mom_u = -body_force_x, 其餘 = 0, cont = 0。"""
    xyzt = _make_xyzt(16)
    bfx = 2.49e-3

    def fn(coords):
        base = torch.tensor([0.7, -0.3, 0.2, 1.5], dtype=coords.dtype)
        return base.expand(coords.shape[0], 4)

    mu, mv, mw, co = channel_ns_residuals(
        fn, xyzt, re=1000.0, body_force_x=bfx,
        Lx=8 * math.pi, Ly=2.0, Lz=3 * math.pi,
    )
    assert torch.allclose(mu, torch.full_like(mu, -bfx), atol=1e-9)
    assert torch.allclose(mv, torch.zeros_like(mv), atol=1e-9)
    assert torch.allclose(mw, torch.zeros_like(mw), atol=1e-9)
    assert torch.allclose(co, torch.zeros_like(co), atol=1e-9)


def test_linear_field_chain_rule_continuity_advection():
    """線性場 u=a·x, v=b·y, w=c·z → 驗證 chain rule、3D continuity、advection。"""
    xyzt = _make_xyzt(16, seed=1)
    a, b, c = 0.5, 0.3, -0.2
    Lx, Ly, Lz = 8 * math.pi, 2.0, 3 * math.pi

    def fn(coords):
        x, y, z = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
        u = a * x
        v = b * y
        w = c * z
        p = torch.zeros_like(x)
        return torch.cat([u, v, w, p], dim=1)

    mu, mv, mw, co = channel_ns_residuals(
        fn, xyzt, re=1000.0, body_force_x=0.0, Lx=Lx, Ly=Ly, Lz=Lz,
    )
    # continuity = a/Lx + b/Ly + c/Lz（處處常數，二階導 0 → viscous 0）
    expected_cont = a / Lx + b / Ly + c / Lz
    assert torch.allclose(co, torch.full_like(co, expected_cont), atol=1e-8)
    # mom_u = adv_u = u·du_dx = (a·x)·(a/Lx) = a²·x/Lx（du_dt, dp_dx, viscous 皆 0）
    x = xyzt[:, 0:1].detach()
    expected_mu = (a ** 2) * x / Lx
    assert torch.allclose(mu, expected_mu, atol=1e-7)


def test_taylor_green_xy_divergence_free():
    """Taylor-Green (x-y) div-free 場 → 3D continuity ≈ 0（驗證 dw_dz 正確併入）。"""
    xyzt = _make_xyzt(64, seed=2)
    k = 2 * math.pi

    def fn(coords):
        x, y = coords[:, 0:1], coords[:, 1:2]
        u = torch.sin(k * x) * torch.cos(k * y)
        v = -torch.cos(k * x) * torch.sin(k * y)
        w = torch.zeros_like(x)
        p = torch.zeros_like(x)
        return torch.cat([u, v, w, p], dim=1)

    _, _, _, co = channel_ns_residuals(
        fn, xyzt, re=1000.0, body_force_x=0.0, Lx=1.0, Ly=1.0, Lz=1.0,
    )
    assert torch.allclose(co, torch.zeros_like(co), atol=1e-6)


def test_viscous_laplacian_and_advection_z():
    """w=sin(k·z) 場 → 驗證 z 方向二階 viscous + advection w·dw_dz + continuity。"""
    xyzt = _make_xyzt(32, seed=3)
    Lz = 3 * math.pi
    k = 2 * math.pi
    re = 100.0
    nu = 1.0 / re

    def fn(coords):
        z = coords[:, 2:3]
        w = torch.sin(k * z)
        zero = torch.zeros_like(z)
        return torch.cat([zero, zero, w, zero], dim=1)

    _, _, mw, co = channel_ns_residuals(
        fn, xyzt, re=re, body_force_x=0.0, Lx=1.0, Ly=1.0, Lz=Lz,
    )
    z = xyzt[:, 2:3].detach()
    dw_dz = k * torch.cos(k * z) / Lz
    lap_w = -(k ** 2) * torch.sin(k * z) / (Lz ** 2)
    adv_w = torch.sin(k * z) * dw_dz          # w·dw_dz
    expected_mw = adv_w - nu * lap_w           # dw_dt=0, dp_dz=0
    assert torch.allclose(mw, expected_mw, atol=1e-6)
    assert torch.allclose(co, dw_dz, atol=1e-6)  # 只有 dw_dz 非零
```

- [ ] **Step 2: Run → 確認 FAIL**

Run: `cd /Users/latteine/Documents/coding/pi-lnn && PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python -m pytest tests/test_channel_ns_residual.py -v`
Expected: FAIL — `ImportError: cannot import name 'channel_ns_residuals' from 'pi_con.physics'`

- [ ] **Step 3: 實作 channel_ns_residuals**

在 `src/pi_con/physics.py` 的 `unsteady_ns_residuals` 函數結尾（第 72 行後）插入：

```python
def channel_ns_residuals(
    uvwp_fn: Callable,
    xyzt: torch.Tensor,
    re: float,
    body_force_x: float = 0.0,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """What: 3D incompressible NS（channel flow）的 momentum(u,v,w) 與 continuity 殘差。

    Why: channel 是首個 3D + wall-bounded case。相較 2D `unsteady_ns_residuals`，
         多 w 分量、z 方向梯度、z 方向二階黏性，continuity 加 dw_dz。
         Forcing 改為 constant streamwise body force（取代 mean dP/dx，對應 Lethe
         flow-control 維持 U_b），物理值 body_force_x = u_τ²/h。
         不改既有 2D 函數 → Kolmogorov/Cylinder zero regression。

    Coordinate chain rule (Lx, Ly, Lz):
        xyzt 正規化到 [0,1]，output 為物理 m/s。
        d/dx_phys = d/dx_norm / Lx；d²/dx²_phys = d²/dx²_norm / Lx²（y,z 同理）。
        channel full domain: Lx=8π, Ly=2, Lz=3π。

    Args:
        uvwp_fn: callable，xyzt[N,4] -> [N,4]（u,v,w,p，物理量級）。
        xyzt:    [N,4] = (x,y,z,t) normalized，requires_grad=True。
        re:      Reynolds number，ν = 1/re。
        body_force_x: constant streamwise driving force（= u_τ²/h）。
        Lx,Ly,Lz: 物理域長度，chain-rule 尺度因子。

    Returns:
        (mom_u, mom_v, mom_w, cont)，各 [N,1]，物理量級。
    """
    uvwp = uvwp_fn(xyzt)                                   # [N, 4]
    u = uvwp[:, 0:1]
    v = uvwp[:, 1:2]
    w = uvwp[:, 2:3]
    p = uvwp[:, 3:4]
    u_g = _grad(u, xyzt)
    v_g = _grad(v, xyzt)
    w_g = _grad(w, xyzt)
    p_g = _grad(p, xyzt)
    # 一階 normalized（index 0=x, 1=y, 2=z, 3=t）
    du_dx_n, du_dy_n, du_dz_n, du_dt = u_g[:, 0:1], u_g[:, 1:2], u_g[:, 2:3], u_g[:, 3:4]
    dv_dx_n, dv_dy_n, dv_dz_n, dv_dt = v_g[:, 0:1], v_g[:, 1:2], v_g[:, 2:3], v_g[:, 3:4]
    dw_dx_n, dw_dy_n, dw_dz_n, dw_dt = w_g[:, 0:1], w_g[:, 1:2], w_g[:, 2:3], w_g[:, 3:4]
    dp_dx_n, dp_dy_n, dp_dz_n = p_g[:, 0:1], p_g[:, 1:2], p_g[:, 2:3]
    # 二階 normalized（沿各自方向）
    du_dx2_n = _grad(du_dx_n, xyzt)[:, 0:1]
    du_dy2_n = _grad(du_dy_n, xyzt)[:, 1:2]
    du_dz2_n = _grad(du_dz_n, xyzt)[:, 2:3]
    dv_dx2_n = _grad(dv_dx_n, xyzt)[:, 0:1]
    dv_dy2_n = _grad(dv_dy_n, xyzt)[:, 1:2]
    dv_dz2_n = _grad(dv_dz_n, xyzt)[:, 2:3]
    dw_dx2_n = _grad(dw_dx_n, xyzt)[:, 0:1]
    dw_dy2_n = _grad(dw_dy_n, xyzt)[:, 1:2]
    dw_dz2_n = _grad(dw_dz_n, xyzt)[:, 2:3]
    sx, sy, sz = float(Lx), float(Ly), float(Lz)
    # chain rule → 物理梯度
    du_dx, du_dy, du_dz = du_dx_n / sx, du_dy_n / sy, du_dz_n / sz
    dv_dx, dv_dy, dv_dz = dv_dx_n / sx, dv_dy_n / sy, dv_dz_n / sz
    dw_dx, dw_dy, dw_dz = dw_dx_n / sx, dw_dy_n / sy, dw_dz_n / sz
    dp_dx, dp_dy, dp_dz = dp_dx_n / sx, dp_dy_n / sy, dp_dz_n / sz
    du_dx2, du_dy2, du_dz2 = du_dx2_n / sx ** 2, du_dy2_n / sy ** 2, du_dz2_n / sz ** 2
    dv_dx2, dv_dy2, dv_dz2 = dv_dx2_n / sx ** 2, dv_dy2_n / sy ** 2, dv_dz2_n / sz ** 2
    dw_dx2, dw_dy2, dw_dz2 = dw_dx2_n / sx ** 2, dw_dy2_n / sy ** 2, dw_dz2_n / sz ** 2
    nu = 1.0 / float(re)
    lap_u = du_dx2 + du_dy2 + du_dz2
    lap_v = dv_dx2 + dv_dy2 + dv_dz2
    lap_w = dw_dx2 + dw_dy2 + dw_dz2
    adv_u = u * du_dx + v * du_dy + w * du_dz
    adv_v = u * dv_dx + v * dv_dy + w * dv_dz
    adv_w = u * dw_dx + v * dw_dy + w * dw_dz
    mom_u = du_dt + adv_u + dp_dx - nu * lap_u - body_force_x
    mom_v = dv_dt + adv_v + dp_dy - nu * lap_v
    mom_w = dw_dt + adv_w + dp_dz - nu * lap_w
    cont = du_dx + dv_dy + dw_dz
    return mom_u, mom_v, mom_w, cont
```

- [ ] **Step 4: Run → 確認 PASS**

Run: `cd /Users/latteine/Documents/coding/pi-lnn && PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python -m pytest tests/test_channel_ns_residual.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 2D regression — 既有 physics test 必須仍 PASS**

Run: `cd /Users/latteine/Documents/coding/pi-lnn && PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python -m pytest tests/test_ns_residual_chain_rule.py -v`
Expected: PASS（既有 2D `unsteady_ns_residuals` 未動，行為不變）

- [ ] **Step 6: Commit**（GIT_POLICY：使用者確認後執行）

```bash
git add src/pi_con/physics.py tests/test_channel_ns_residual.py
git commit -m "feat(channel): add 3D channel_ns_residuals with manufactured-solution tests"
```

---

## Plan 2-5（stub，Plan 1 完成後展開）

### Plan 2：model 3D 化（橫跨 5 檔，新增 forward path 策略）

讀完 decoder/operator/encoders/encodings 後確認 scope 橫跨 5 檔；採「新增 3D path」策略（2D path zero regression）。拆成 5 個 TDD sub-task，incremental 執行（每個 dispatch 時提供 complete code）：

**Task 2a — config 3D keys**（低風險，self-contained）
- `config.py:DEFAULT_PICON_ARGS` 新增 `num_spatial_dims=2`、`num_velocity_components=3`、`Lz=1.0`、`periodic_axes=None`（None=沿用 `use_periodic_domain` 對所有軸；list 如 `[0,2]`=x,z 週期 y 非週期）。
- 驗證：載入含新 key 的 config OK + default 正確 + 仍 reject unknown key。

**Task 2b — encodings.py per-axis 3D**（核心難點）
- `periodic_fourier_encode(z, domain_length, n_harmonics, periodic_axes=None)`：泛用 `z.shape[1]`；週期軸出 sin/cos（2/軸/harmonic），非週期軸出 raw coord（1/軸）。`periodic_axes=None` → 全週期（2D 向後相容）。
- `LearnableFourierEmb`：`input_dim` 參數化、`proj = Linear(2*len(periodic_axes), ...)` + 非週期軸 concat raw。
- 驗證：2D 輸出維度/值不變（regression）；3D periodic 軸 `encode(x)≈encode(x+Lx)`；非週期 y 軸 `encode(y)≠encode(y+Ly)`。

**Task 2c — encoders.py SpatialSetEncoder 3D**
- `__init__` 加 `spatial_dim=2`、`periodic_axes`；`FourierEmbs(input_dim=spatial_dim)`；harmonics-only `spatial_dim_out = (2*len(periodic_axes)+n_nonperiodic)*fourier_harmonics`（隨 encoding 改）。
- `encode_pos` 驗證/支援 [K,spatial_dim]；`_apply_geometry_graph` 週期修正改 per-axis（channel 不用 geometry，但別破壞）。
- 驗證：2D regression（既有 encoder test）；3D sensor_pos [K,3] forward shape 正確。

**Task 2d — decoder.py component 參數化 + forward_uvwp**（核心）
- `component_emb/trunk_out/branch_proj/component_scale/bias` 由 `num_velocity_components`(C) 參數化（2D=3, 3D=4）；`FourierEmbs input_dim`、`spatial_dim` 隨 encoding。
- 新增 `forward_uvwp(xyz, t_q, h_states, sensor_time, sensor_pos)` method：複製 `forward_uvp` 改 C comp / 3D coords，**移除** hard_body_bc/geometry_tokens/body_distance（channel 不需要），保留 cross-attention/trunk/branch/FiLM(可選)。
- 驗證：2D `forward_uvp` 輸出不變（regression）；3D `forward_uvwp` 輸出 [N,4]；對 xyz 可二階 autograd（接 `channel_ns_residuals` 不報錯）。

**Task 2e — operator.py 3D closure + physics buffer**
- `physics_output_mean/std` 泛化成可變 size（依 C）；`set_physics_normalization` 接受 shape (C,)。
- 新增 `make_picon_model_fn_uvwp`（xyzt[N,4]→uvwp[N,4] denormalized）；`create_picon_model` 傳 3D config（spatial_dim/num_velocity_components/periodic_axes）。
- 驗證：2D 不變；3D end-to-end `model_fn_uvwp` 輸出 [N,4] 且能餵 `channel_ns_residuals` 算出 4 個 residual（整合測試）。

**全程硬約束**：每個 sub-task 後既有 2D Kolmogorov/Cylinder 測試必須 PASS（Never Break Userspace）。

### Plan 3：資料管線（token 已驗證 ✓ 2026-06-02）

**工具鏈已驗證**（smoke 16³ cutout，2.3s）：
- `uv run --with givernylocal`（臨時環境，42 deps，不進主 pyproject）。
- `turb_dataset(dataset_title='channel', output_path='data/jhtdb_cache', auth_token=os.environ['JHTDB_AUTH_TOKEN'])`。
- `getCutout(ds, 'velocity', axes_ranges, strides)`：`axes_ranges=np.array([[x1,x2],[y1,y2],[z1,z2],[t1,t2]])` **1-indexed**；`strides=np.array([sx,sy,sz,st])`；回 xarray，key=`velocity_{t:04d}`。
- **軸序 INVARIANT**：`data.dims = (zcoor, ycoor, xcoor, values)` → `arr[z, y, x, c]`，c: 0=u, 1=v, 2=w。
- `getData(ds, 'velocity', time, temporal_method, 'lag8', 'field', points[N,3], option=[t_end,dt], return_times=True)`：任意物理座標 time series（`temporal_method='pchip'`）。channel 物理域 x∈[0,8π], y∈[-1,1], z∈[0,3π]。

**Task 3a — DNS cutout 抓取**（評估 GT）
- `scripts/fetch_channel_dns_jhtdb.py`：getCutout 512×128×384 (stride 4, 全 domain) × N frames（snapshot t=1,2,…）→ `.npy` `[T,Nz,Ny,Nx,3]` + metadata `{x,y,z 物理座標, time}`。
- ⚠️ channel chunk 64³，512×128×384 涵蓋全 domain → 大量 chunk 請求/frame。對策：**分 frame 抓 + 進度 log + resumable（跳過已存 frame）**。smoke 先 64³ 確認，再全量。

**Task 3b — LES 佈點 tile**（home-gpu）
- home-gpu pyvista 讀 LES VTU → node values on (2π×π) grid → QR-pivot → K sensor 佈局 → tile×(4×3) 投到 (8π×3π) → K 個物理座標 (x,y,z)。

**Task 3c — sensor 量測**（getData，免插值）
- `getData` 在 tile 後 K 個物理座標抽 velocity time series（pchip）→ `sensors_channel_*.{json,npz}`（座標 [K,3]、{u,v,w,time}）。

**Task 3d — 3D axis convention test**
- `tests/test_sensor_axis_convention.py` 擴 3D：assert `getData(sensor座標)` ≈ `getCutout[對應 z,y,x]`（依上述 (z,y,x) 軸序）。INVARIANT，所有 sensor generator 必過。

### Plan 4：channel dataset + wall BC + loss 組裝
- `src/channel_dataset.py`（follow kolmogorov_dataset，提供 Lx/Ly/Lz、re_value、sample_physics_points 3D）。
- wall BC loss（training.py，沿用 cylinder body no-slip 1234-1360）：y=±1 Dirichlet u=v=w=0；x,z periodic。config `bc_wall_n_points`、`bc_wall_weight`。
- l_physics 接 `channel_ns_residuals`（含 `l_ns_w`）；channel config（`configs/chexp_001_*.toml`）。

### Plan 5：evaluator + 驗證階梯
- `scripts/evaluate_channel.py`：U⁺(y⁺)+log-law、Reynolds stress(u'u',v'v',w'w',u'v')、KE rel-err(含 w)、div L2（沿用 LES `channel_post.py` per-y average 邏輯）。
- Smoke（256×64×192, K=100, ~500 steps）→ Coarse benchmark（512×128×384, K=100, ~15k steps）。

---

## Self-Review

**1. Spec coverage（Plan 1 對 spec §5 Phase 1 physics）：** spec §5 要求 physics 加 w/dw_dz/du_dz2、continuity+dw_dz、Lz chain-rule、forcing — Plan 1 Task 1 全涵蓋。spec 提的「smooth-norm sqrt(r²+1e-8)」屬 decoder（`decoder.py:566` relpos_bias），不在 physics.py，移至 Plan 2 處理（已修正 spec 誤植認知）。

**2. Placeholder scan：** Plan 1 所有 code step 含完整可執行 code，無 TBD/TODO。Plan 2-5 為 stub（已明確標示，待展開），非 placeholder。

**3. Type consistency：** `channel_ns_residuals` signature（`uvwp_fn`, `xyzt[N,4]`, 回 4-tuple）在函數定義、4 個 test、Plan 2 `make_picon_model_fn_uvwp`、Plan 4 loss 組裝中一致。`body_force_x` 命名一致。
