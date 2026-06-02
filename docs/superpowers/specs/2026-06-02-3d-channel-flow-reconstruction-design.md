# 3D Channel Flow 重建 — Design Spec

- Date: 2026-06-02
- Status: Draft（待 user review → writing-plans）
- Case prefix: `CHEXP-NNN`（kolmogorov=`EXP`, cylinder=`CEXP`）
- State 檔: 新建 `docs/channel_log.md`（比照 `docs/archive/cylinder_log.md`）

---

## 1. 目標與範圍

在既有 pi-lnn CfC-DeepONet sparse-reconstruction pipeline 上新增 **3D turbulent channel flow** 重建 case，作為繼 2D Kolmogorov、2D Cylinder 之後的第三個（首個 3D、首個 wall-bounded）case。

工程情境（沿用 `REAL_WORLD_PIPELINE`）：低保真 LES 佈點 → 在真實位置（以 DNS 代理）量測稀疏 sensor → sensor MSE + PDE residual 訓練 → 重建全場 → DNS offline benchmark。

**範圍邊界**：本 spec 涵蓋「coarse 跑通」到「coarse benchmark」。production-res 加密列為後續。

---

## 2. 資料來源與物理對齊

| 項目 | JHTDB DNS（ground truth） | LES 第二版（佈點先驗） |
|---|---|---|
| 來源 | JHTDB `givernylocal`/`getCutout`（需重抓） | home-gpu `~/cfd_solver/channel_les/`（已有） |
| Solver | DNS（B-spline collocation） | Lethe ILES, BDF2 + GLS |
| Domain | **8π × 2 × 3π** (full) | **2π × 2 × π** (minimal, Jiménez-Moin) |
| Grid | 2048 × 512 × 1536 | 64 × 48 × 32 (98,304 hex) = `channel_v2_98k.msh` |
| Re_τ / u_τ / ν | 1000 / 0.0499 / 5e-5 | ~1000 / (目標同) / 5e-5 |
| y 方向 | 7th-order B-spline（壁面加密） | gmsh 非均勻 |
| 時間 | 4000 frames, t≈0~25.99, Δt≈0.0065 | 49 frames, t≈0~50, 5.9 GB VTU |

**物理對齊決策（已拍板）**：
- GT 用 **full domain**（忠於原始 full-channel 統計）。
- LES 是不同 domain、不同湍流實現，**不可 pointwise 對齊** DNS。
- 利用 channel 在 x,z 的**統計齊次性**：QR-pivot 在 LES `2π×π` 算出的佈局（y 分佈 + x-z pattern），**tile ×(4×3)** 平鋪到 DNS `8π×3π`。
- [RISK] minimal-box LES 缺最大尺度結構 → 佈點 transfer 品質未知，列為 Phase 0 驗證項；fallback = DNS 自身 canonical 佈點。

**Downsample（已拍板）**：各方向 1/4 → **512 × 128 × 384**（full 的 1/64 體積）。單 frame (u,v,w) float32 ≈ 300 MB；16 frames ≈ 4.8 GB。

**Token**：個人 JHTDB API token 已取得，存於 gitignored `.env`（`JHTDB_AUTH_TOKEN`）。抓取 script 從 `os.environ` 讀，禁止 hardcode 進 git-tracked 檔。Plan 3 阻塞解除。

---

## 3. 架構：擴展策略（混合 — 已拍板）

現有 2D 假設散佈在 decoder / physics / dataset / config 四層。採**混合策略**：

- **核心泛化**（2D 是 3D 設 w=0,∂/∂z=0 的特例，數學自然泛化，用既有 2D test 護 regression）：
  - `src/pi_con/physics.py`
  - `src/pi_con/decoder.py`
  - `src/pi_con/config.py`
- **case-specific 新建**（follow kolmogorov/cylinder 各自獨立的既有 pattern，zero regression risk）：
  - `src/channel_dataset.py`（新）
  - wall BC loss（擴 `training.py`，沿用 cylinder body no-slip 邏輯）
  - `scripts/evaluate_channel.py`（新）

**Never Break Userspace 硬約束**：既有 2D Kolmogorov / Cylinder smoke + regression test 在每個 Phase 後必須仍 PASS。

---

## 4. Phase 0 — 資料準備（最重工程）

### 0a. DNS 抓取
- 重建 JHTDB 抓取 script（`scripts/fetch_channel_dns_jhtdb.py`）。
- coarse benchmark：512×128×384，連續 16 frames（覆蓋數個 eddy turnover）。
- y 用 index step（B-spline grid 自動保留近壁加密，index 0 起含 y=−1 壁面）。
- 落地 `.npy` + metadata `{x[512], y[128], z[384], time[16]}`。

### 0b. LES 佈點 tile
- 在 **home-gpu** 用 pyvista 把 LES VTU → node values（不傳 5.9 GB 回本地）。
- QR-pivot（POD modes）在 `2π×π` 算 K 個 sensor 佈局 → tile ×(4×3) 投到 `8π×3π`。
- **沿用 EXP-094 axis convention**（`u_full[:, x_idx, y_idx]`），新 generator 必須過 `tests/test_sensor_axis_convention.py`（擴 3D 版）。

### 0c. sensor 量測
- DNS 在 tile 後 (x,y,z) 位置抽 (u,v,w) 時序。
- schema 對齊既有 `sensors_*.{json,npz}`：JSON `selected_coordinates [K,3]`；NPZ `{u,v,w,time}` 各 `[K,T]`。

### 0d. 三道驗證防線（INVARIANT）
1. `div(DNS cutout)` 量級合理（downsample 後非機器精度，但應 ≪ 速度梯度尺度）。
2. sensor NPZ 值 ≈ `DNS[sensor 位置]`（axis convention test，3D 版）。
3. LES QR-pivot leading-r modes 與 DNS 同 r 的 y-spectrum / x-z pattern 統計對齊（佈點 transfer 驗證）。

---

## 5. Phase 1 — 核心泛化

### `physics.py:unsteady_ns_residuals`（現 13-72 行，2D 寫死）
- 簽名新增 `Lz: float`。
- 計 w 及梯度：`dw/dx, dw/dy, dw/dz, dw/dt`；二階 `du_dz2, dv_dz2, dw_dz2`（`_grad(·, xyt)[:, 2:3]`）。
- continuity：`cont = du_dx + dv_dy + dw_dz`。
- 三動量方程黏性項：`ν·(∂²/∂x² + ∂²/∂y² + ∂²/∂z²)`，各方向 chain-rule 乘 `1/Lx², 1/Ly², 1/Lz²`。
- **保留 smooth-norm `sqrt(r²+1e-8)`**（二階 autograd r=0 NaN 防護，KNOWN_PITFALL）。

### `decoder.py`（現 2D 寫死）
- `:196` `component_emb(3,8) → (4,8)`（u,v,w,p）。
- `:103` `FourierEmbs(input_dim=2 → 3)`（x,y,z）。
- `:238`/`:405` `trunk_out 3*rank → 4*rank`、view 對應改。
- `query_in` 加入 z 的 Fourier 編碼。
- 全部由 config 控制，2D 路徑預設不變。

### `config.py`（KNOWN_PITFALL：新 key 必須三處同步 — dict 初值 + TOML 驗證 + AL 驗證）
- 新增 `num_spatial_dims`（預設 2）、`num_velocity_components`（預設 3 = u,v,p；3D channel 設 4）、`Lz`。
- `observed_sensor_channels` 3D 設 `["u","v","w"]`。
- `use_periodic_domain`：channel 對 x,z periodic、y 非週期 → 需支援 per-axis periodicity（或 x,z periodic + y wall BC）。

### Regression
- 既有 2D Kolmogorov / Cylinder smoke test 必須 PASS。

---

## 6. Phase 2 — channel 專屬模組（新建）

### `src/channel_dataset.py`
- 讀 DNS `.npy`(3D volume, `[T, ...]`) + metadata + sensor `[K,3]`/`[u,v,w,t]`。
- 全場 DNS 不存進 dataset（比照 kolmogorov，evaluator 直接 load）。

### Wall BC loss（擴 `training.py`，參考現 1234-1360 cylinder BC）
- y=±1 **no-slip Dirichlet** `u=v=w=0`：隨機採 x∈[0,Lx], z∈[0,Lz], t，固定 y=±1 各半。
- x,z **periodic**（domain periodic，soft 或硬編 Fourier 週期）。
- 新 config：`bc_wall_n_points`、`bc_wall_weight`。
- l_physics 構成：`l_ns_u + l_ns_v + l_ns_w + continuity_weight·l_cont + bc_wall_weight·l_wall`。

### `scripts/evaluate_channel.py`（新）
- **U⁺(y⁺)** mean profile + viscous sublayer / log-law 對照（核心指標）。
- **Reynolds stresses** u'u', v'v', w'w', u'v'（xz-mean + time-mean，沿用 LES `channel_post.py` per-y average 邏輯）。
- KE rel-err（含 w）、divergence L2。
- u_τ 由 wall shear 或 force 反推，對照 DNS 0.0499。

---

## 7. Phase 3 — 驗證階梯（VALIDATION_LADDER）

| 階 | 解析度 | K | steps | 通過判定 |
|---|---|---|---|---|
| Smoke | 256×64×192 | 100 | ~500 | loss 非 NaN、wall BC 收斂、w 非零、ckpt 產出 |
| Coarse benchmark | 512×128×384 | 100 | ~15k | U⁺(y⁺) 對 log-law、KE rel-err vs DNS 合理、div L2 小 |
| 加密（後續） | ↑res / ↑K / ↑frames | — | — | — |

每階先過前一階。Smoke 失敗禁止往下。

---

## 8. 開放參數預設（[可調]）

| 參數 | 預設 | 備註 |
|---|---|---|
| K | 100 | LES QR-pivot tile；3D 自由度大，後續可能 ↑ |
| coarse cutout | 512×128×384 | smoke 256×64×192 |
| frames | 16 | CfC 時序窗 |
| coarse steps | 15k | smoke 500 |
| EXP prefix | `CHEXP-NNN` | — |
| state 檔 | `docs/channel_log.md` | — |

---

## 9. 風險與回退

- [RISK 佈點 transfer] LES minimal-box 缺最大尺度 → tile 後佈點對 full-channel 統計次優。回退：DNS 自身 canonical wall-normal 佈點。
- [RISK 記憶體] 3D 二階 autograd 記憶體爆量。緩解：coarse 階段小 batch + 限 collocation 數；必要時 gradient checkpointing。
- [RISK checkpoint] 改 component_emb/query_in 維度 → 舊 ckpt 不相容。對策：3D 一律 cold start。
- [RISK token] 個人 JHTDB key 未定位。緩解：testing token 先跑 smoke；benchmark 前需取得個人 token 或確認 testing token 配額足夠。
- [RISK domain periodicity] decoder Fourier 編碼目前全週期；channel y 非週期。需確認 per-axis periodicity 不破壞既有 2D 全週期路徑。

---

## 10. 成功判定（工程可驗證為主，物理診斷為輔）

- 主（工程可遷移）：sensor MSE + 3D physics residual 量級合理；訓練只用 sensor + PDE，無 DNS 全場 supervision。
- 次（DNS offline 診斷）：U⁺(y⁺) 落在 viscous sublayer + log-law 趨勢內；KE rel-err 與 Reynolds stress 形狀對 DNS 合理；div L2 小。
- 佈點 framing：LES→DNS tile pipeline 工程可遷移性成立（Phase 0d 第 3 道防線通過）。
