# Pi-LNN 設計規格：Physics-informed Liquid Neural Network for Kolmogorov Flow

**日期**：2026-03-23
**問題**：2D Kolmogorov flow 全場重建——給定 K 個感測器的時序讀數 `[K, T]`，預測任意 `(x, y, t)` 的 u/v/p
**主體架構**：LNN（Liquid Neural Network）；CfC 用於加速 LNN 內部的 ODE 求解

---

## 1. 為何 CfC 加速 LNN 的 ODE

LNN 的核心是 LTC（Liquid Time-Constant）ODE：
```
τ(x) · dh/dt = -h + f(x, h)
```
數值求解需要小時間步（如 RK4，dt=0.0005），推論時代價高。

**CfC** 以閉合解取代數值積分：
```
h(t+Δt) = sigmoid(-t_a · Δt + t_b) · f1(x,h) + (1 - sigmoid(...)) · f2(x,h)
```
使用真實物理 Δt（如 0.05），**一步直達**，無需 sub-stepping，這就是「加速 ODE 計算」的含義。

---

## 2. 資料格式

| 來源 | 格式 | 說明 |
|------|------|------|
| DNS | `[T=201, 512, 512]` u/v/p/ω | 真實高解析度場 |
| LES | `[T=201, 256, 256]` u/v/p/ω | 模擬低解析度場（訓練 source） |
| Sensors | `[K, T_obs, 3]` u/v/p | K 個固定位置的時序讀數 |
| 感測器位置 | JSON `indices` | K 個空間索引 → `(x_k, y_k)` |
| Re 值 | {100, 1000, 10000, 100000} | 多 Re 訓練 |
| 時間步 | Δt = 0.05（物理單位） | 感測器取樣間隔 |

---

## 3. 架構：Pi-LNN

### 3.1 整體流程

```
Sensors [K, T, 3]  +  sensor positions [K, 2]  +  Re (scalar)
      ↓
=== Spatial CfC Encoder（每個時間步 t）===
  對 K 個感測器做空間序列處理 → 空間隱藏態 s_t [d_model]

=== Temporal CfC Encoder（跨 T 個時間步）===
  使用物理 Δt，依序處理 s_0, ..., s_{T-1}
  h_enc = h_T [d_model]          ← 捕捉全部時序動態

=== Query CfC Decoder===
  Query (x_q, y_q, t_q, c) → RFF + time encoding → q [d_model]
  CfCCell(q, h_enc) → H_dec [N_q, d_model]   ← 向量化單步，無 for-loop
  Linear → [N_q, 1]
```

### 3.2 CfC Cell（no-gate 模式）

```
xh    = cat([x, h], dim=-1)
f1    = tanh(W_1 · xh + b_1)
f2    = tanh(W_2 · xh + b_2)
t_a   = W_a · xh + b_a             # 可學習衰減速率
t_b   = W_b · xh + b_b             # 可學習偏移
gate  = sigmoid(-t_a · Δt + t_b)   # Δt 為物理時間步（Temporal Encoder 用 0.05）
h_new = gate · f1 + (1 - gate) · f2
```

**Δt 來源**：
- Spatial CfC：Δt = 1.0（固定，感測器無空間時序語意）
- Temporal CfC：Δt = 實際物理時間間隔（從資料取得，約 0.05）
- Decoder CfC：Δt = 1.0（固定，單步解碼）

### 3.3 SpatialCfCEncoder（空間編碼，每時間步執行）

```
Input: sensor_vals [K, 3] (u_k, v_k, p_k at time t)
       sensor_pos  [K, 2] (x_k, y_k)
       B [2, rff_features] (shared RFF matrix)

1. token_k = Linear(cat([RFF(x_k, y_k), u_k, v_k, p_k]))  → [K, d_model]
2. h = 0
   for k = 0..K-1: h = CfCCell_spatial(token_k, h, dt=1.0)
3. s_t = h  [d_model]   ← 此時間步的空間摘要
```

### 3.4 TemporalCfCEncoder（時序編碼）

```
Input: s_0, s_1, ..., s_{T-1}  各 [d_model]
       re_norm (float)
       dt_phys (float, 物理時間步長)

1. Re token: re_t = Linear(re_norm)  [d_model]
2. h = re_t   ← 用 Re 資訊初始化隱藏態
   for t = 0..T-1: h = CfCCell_temporal(s_t, h, dt=dt_phys)
3. h_enc = h  [d_model]
```

**Re 注入方式**：作為初始隱藏態而非額外 token，避免增加序列長度，同時讓 Re 資訊滲透整個時序演化。

### 3.5 QueryCfCDecoder（解碼）

```
Input: xy [N_q, 2], t_q [N_q], c [N_q, torch.long], h_enc [d_model]

1. rff_q  = RFF(xy, B)                     [N_q, 2*rff_features]
2. time_e = Linear(t_q.unsqueeze(-1))      [N_q, d_time]   # 時間編碼
3. emb_c  = Embedding(3, 8)(c)             [N_q, 8]
4. q      = Linear(cat([rff_q, time_e, emb_c]))  [N_q, d_model]
5. h_0    = h_enc.unsqueeze(0).expand(N_q, -1).contiguous()  [N_q, d_model]
6. H_dec  = CfCCell_decoder(q, h_0, dt=1.0)   [N_q, d_model]  ← 完全向量化
7. out    = Linear(H_dec)                  [N_q, 1]
8. out    = out * component_scale[c] + component_bias[c]
```

### 3.6 CfCCell 實例分離

| 實例 | 作用域 | input_size | Δt |
|------|--------|-----------|-----|
| `cfc_spatial` | SpatialCfCEncoder | d_model | 1.0 |
| `cfc_temporal` | TemporalCfCEncoder | d_model | 物理 Δt |
| `cfc_decoder` | QueryCfCDecoder | d_model | 1.0 |

三個 CfCCell **各自獨立**，不共享權重。

---

## 4. 物理損失

使用 2D incompressible Navier-Stokes（與現行 `steady_ns_residuals` 相同結構，擴展為非定常版本）：

```
NS_x: ∂u/∂t + u·∂u/∂x + v·∂u/∂y + ∂p/∂x - ν·(∂²u/∂x² + ∂²u/∂y²) = f_x
NS_y: ∂u/∂t + u·∂v/∂x + v·∂v/∂y + ∂p/∂y - ν·(∂²v/∂x² + ∂²v/∂y²) = f_y
Cont: ∂u/∂x + ∂v/∂y = 0
```

強制力項 `f_x = A·sin(k_f·y)`（Kolmogorov forcing，`k_f=4`, `A=0.1`）。

梯度透過 `torch.autograd.grad` 計算，CfC 無不可微分算子，梯度流通無障礙。

---

## 5. 類別結構

| 類別 | 說明 |
|------|------|
| `CfCCell` | 核心 CfC 遞迴單元（no-gate） |
| `SpatialCfCEncoder` | 每時間步：K sensors → s_t |
| `TemporalCfCEncoder` | T 個 s_t → h_enc（使用物理 Δt） |
| `QueryCfCDecoder` | (x,y,t,c) + h_enc → u/v/p |
| `LiquidOperator` | 組合以上三者的主模型 |
| `create_lnn_model` | 工廠函式 |
| `make_lnn_model_fn` | closure，供物理損失使用 |

---

## 6. Config

```toml
[train]
# 資料
re_values = [100, 1000, 10000, 100000]
num_sensors = 50          # K
dt_phys = 0.05            # 物理時間步長

# 模型
rff_features = 64
rff_sigma = 5.0
d_model = 128
num_spatial_cfc_layers = 2
num_temporal_cfc_layers = 2
d_time = 16               # 時間編碼維度

# 損失權重
data_loss_weight = 1.0
physics_loss_weight = 0.1
continuity_weight = 1.0

# 訓練
iterations = 20000
learning_rate = 1e-3
weight_decay = 1e-4
```

---

## 7. 檔案計畫

| 動作 | 檔案 |
|------|------|
| 新建 | `src/pi_onet/kolmogorov_dataset.py` |
| 新建 | `src/pi_onet/lnn_kolmogorov.py`（主模型 + 訓練） |
| 保留 | `src/pi_onet/pit_ldc.py`（LDC 版本不動） |
| 更新 | `pyproject.toml`（新增 `lnn-kolmogorov-train` entry point） |
| 新建 | `tests/test_lnn_kolmogorov.py` |
| 新建 | `configs/lnn_kolmogorov.toml` |

---

## 8. 驗收條件

1. `pytest` 全數通過（CfCCell shape/grad、SpatialCfCEncoder、TemporalCfCEncoder、QueryCfCDecoder）
2. 模型不含任何 `nn.MultiheadAttention` 或 `nn.TransformerEncoder`
3. `lnn-kolmogorov-train --config configs/lnn_kolmogorov.toml` 可執行 3 steps 無 NaN
4. 物理損失對 `(x,y,t)` autograd 可流回 CfC 參數
5. 推論時支援任意 `(x, y, t)` 輸入（不限訓練解析度）

---

## 9. 不在本次範圍

- 自回歸 Rollout（時間外推）
- LDC 架構替換
- 效能 benchmark vs PiT
