# Pi-LNN Kolmogorov Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 實作 Physics-informed Liquid Neural Network（LNN/CfC）用於 Kolmogorov flow 全場重建——給定 K 個感測器的時序讀數 `[K, T]`，預測任意 `(x, y, t)` 的 u/v/p。

**Architecture:** 兩層 CfC encoder（Spatial：K sensors → s_t，Δt=1.0；Temporal：T steps → h_enc，**使用物理 Δt=dt_phys（由資料決定，= 1.0 感測器時間單位）**），加上單步向量化 CfC decoder（(x,y,t,c) + h_enc → u/v/p，無任何 Attention 機制）。CfC 以閉合解取代 LTC ODE 的數值積分。

**Tech Stack:** Python 3.11, PyTorch ≥ 2.6, NumPy ≥ 2.2, deepxde（僅用 autograd config），uv

---

## 資料規格（所有任務共用）

| 項目 | 值 |
|------|-----|
| 空間域 | `[0, 2π] × [0, 2π]`（週期邊界）|
| 感測器 K | 50（JSON `selected_coordinates`，單位 rad）|
| 感測器時間步 T | 21（t = 0, 1, 2, ..., 20，物理 Δt = 1.0）|
| LES ground truth | `[T=401, N=256, N=256]`，Δt=0.05，t ∈ [0, 20] |
| LES 空間 x/y | `np.linspace(0, 2π, 256)` |
| 物理 Re → ν | `ν = 1/Re`（re100=0.01, re1000=0.001, re10000=0.0001）|
| Kolmogorov forcing | `f_x = A · sin(k_f · y)`，A=0.1, k_f=4 |

---

## 檔案結構

```
src/pi_onet/
  pit_ldc.py              ← 不動（LDC 版本保留）
  kolmogorov_dataset.py   ← 新建：資料載入 + 訓練/驗證採樣
  lnn_kolmogorov.py       ← 新建：CfCCell + 三個 encoder/decoder + 訓練迴圈 + CLI

tests/
  test_lnn_kolmogorov.py  ← 新建：所有 TDD 測試

configs/
  lnn_kolmogorov.toml     ← 新建：訓練設定

pyproject.toml            ← 更新：新增 lnn-kolmogorov-train entry point
```

---

## Task 1：CfCCell（核心遞迴單元）

**Files:**
- Create: `src/pi_onet/lnn_kolmogorov.py`
- Test: `tests/test_lnn_kolmogorov.py`

- [ ] **Step 1：寫失敗測試**

```python
# tests/test_lnn_kolmogorov.py
import pytest
import torch
from pi_onet.lnn_kolmogorov import CfCCell

D = 32

def test_cfccell_output_shape():
    """CfCCell([B, in], [B, hid]) → [B, hid]."""
    cell = CfCCell(input_size=16, hidden_size=D)
    x = torch.randn(8, 16)
    h = torch.zeros(8, D)
    h_new = cell(x, h, dt=1.0)
    assert h_new.shape == (8, D)

def test_cfccell_dt_sensitivity():
    """不同 dt 應產生不同輸出（gate 依賴 dt）。"""
    cell = CfCCell(input_size=16, hidden_size=D)
    x = torch.randn(4, 16)
    h = torch.zeros(4, D)
    with torch.no_grad():
        out1 = cell(x, h, dt=1.0)
        out2 = cell(x, h, dt=5.0)
    assert not torch.allclose(out1, out2)

def test_cfccell_backward():
    """loss.backward() 成功，所有參數梯度非 None。"""
    cell = CfCCell(input_size=16, hidden_size=D)
    x = torch.randn(4, 16)
    h = torch.zeros(4, D)
    h_new = cell(x, h, dt=1.0)
    h_new.sum().backward()
    for name, p in cell.named_parameters():
        assert p.grad is not None, f"{name} 無梯度"
```

- [ ] **Step 2：確認測試失敗**

```bash
uv run pytest tests/test_lnn_kolmogorov.py::test_cfccell_output_shape -v
```
Expected: `ImportError: cannot import name 'CfCCell'`

- [ ] **Step 3：實作 CfCCell**

建立 `src/pi_onet/lnn_kolmogorov.py`：

```python
# src/pi_onet/lnn_kolmogorov.py
"""Pi-LNN: Physics-informed Liquid Neural Network for Kolmogorov flow.

What: 以 CfC (Closed-form Continuous-time) 取代 LTC ODE 的數值求解器，
      實現 Spatial CfC Encoder + Temporal CfC Encoder + Query CfC Decoder。
Why:  CfC 閉合解 h_new = gate·f1 + (1-gate)·f2 在一步內模擬連續時間動態，
      不需要 sub-stepping，加速推論同時保留 LNN 的時序誘導偏差。
"""
from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn as nn
from deepxde import config as dde_config

from pi_onet.pit_ldc import (
    rff_encode,
    configure_torch_runtime,
    count_parameters,
    write_json,
    _grad,
)


class CfCCell(nn.Module):
    """What: CfC（Closed-form Continuous-time）遞迴單元，no-gate 模式。

    Why: 以閉合解取代 LTC ODE 數值積分：
         h_new = sigmoid(-t_a·Δt + t_b) · f1(x,h) + (1 - sigmoid(...)) · f2(x,h)
         使用真實物理 Δt，一步直達目標態，加速 LNN 的 ODE 計算。
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        combined = input_size + hidden_size
        self.ff1 = nn.Linear(combined, hidden_size)
        self.ff2 = nn.Linear(combined, hidden_size)
        self.time_a = nn.Linear(combined, hidden_size)
        self.time_b = nn.Linear(combined, hidden_size)
        # Xavier init，確保初始 gate ≈ 0.5，避免 sigmoid 飽和
        for layer in (self.time_a, self.time_b):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        x: [..., input_size]
        h: [..., hidden_size]
        Returns: [..., hidden_size]
        """
        xh = torch.cat([x, h], dim=-1)
        f1 = torch.tanh(self.ff1(xh))
        f2 = torch.tanh(self.ff2(xh))
        t_a = self.time_a(xh)
        t_b = self.time_b(xh)
        gate = torch.sigmoid(-t_a * dt + t_b)
        return gate * f1 + (1.0 - gate) * f2
```

- [ ] **Step 4：確認測試通過**

```bash
uv run pytest tests/test_lnn_kolmogorov.py -k "cfccell" -v
```
Expected: 3 PASSED

- [ ] **Step 5：Commit**

```bash
git add src/pi_onet/lnn_kolmogorov.py tests/test_lnn_kolmogorov.py
git commit -m "feat: add CfCCell (no-gate CfC recurrent unit)"
```

---

## Task 2：SpatialCfCEncoder（K sensors → s_t）

**Files:**
- Modify: `src/pi_onet/lnn_kolmogorov.py`
- Test: `tests/test_lnn_kolmogorov.py`

- [ ] **Step 1：寫失敗測試**

```python
# 加入 tests/test_lnn_kolmogorov.py
from pi_onet.lnn_kolmogorov import SpatialCfCEncoder

RFF_F = 16
D = 32
K = 12  # num sensors

def test_spatial_encoder_output_shape():
    """SpatialCfCEncoder: sensors[K,3] + pos[K,2] → [d_model]。"""
    B = torch.randn(2, RFF_F)
    enc = SpatialCfCEncoder(rff_features=RFF_F, d_model=D, num_layers=2)
    sensor_vals = torch.randn(K, 3)
    sensor_pos = torch.rand(K, 2) * 6.28
    s_t = enc(sensor_vals, sensor_pos, B)
    assert s_t.shape == (D,)

def test_spatial_encoder_backward():
    """梯度可從 s_t 流回 sensor_vals。"""
    B = torch.randn(2, RFF_F)
    enc = SpatialCfCEncoder(rff_features=RFF_F, d_model=D, num_layers=1)
    sensor_vals = torch.randn(K, 3, requires_grad=True)
    sensor_pos = torch.rand(K, 2)
    s_t = enc(sensor_vals, sensor_pos, B)
    s_t.sum().backward()
    assert sensor_vals.grad is not None
```

- [ ] **Step 2：確認測試失敗**

```bash
uv run pytest tests/test_lnn_kolmogorov.py::test_spatial_encoder_output_shape -v
```
Expected: `ImportError: cannot import name 'SpatialCfCEncoder'`

- [ ] **Step 3：實作 SpatialCfCEncoder**

加入 `src/pi_onet/lnn_kolmogorov.py`：

```python
class SpatialCfCEncoder(nn.Module):
    """What: 在單一時間步內，以 CfC 序列處理 K 個感測器 → 空間摘要向量 s_t。

    Why: 每個感測器 token 為 [RFF(x,y), u, v, p]，CfC 序列掃描取代空間注意力；
         Δt=1.0（感測器無自然空間時序，RFF 已負責空間資訊）。
    """

    def __init__(self, rff_features: int, d_model: int, num_layers: int) -> None:
        super().__init__()
        sensor_in = 2 * rff_features + 3  # RFF(x,y) + u,v,p
        self.proj = nn.Linear(sensor_in, d_model)
        self.cells = nn.ModuleList([
            CfCCell(d_model, d_model) for _ in range(num_layers)
        ])

    def forward(
        self,
        sensor_vals: torch.Tensor,
        sensor_pos: torch.Tensor,
        B: torch.Tensor,
    ) -> torch.Tensor:
        """
        sensor_vals: [K, 3]  (u, v, p)
        sensor_pos:  [K, 2]  (x, y) in [0, 2π]
        B:           [2, rff_features]
        Returns:     [d_model]
        """
        rff = rff_encode(sensor_pos, B)                              # [K, 2*rff_features]
        seq = self.proj(torch.cat([rff, sensor_vals], dim=-1))       # [K, d_model]

        for cell in self.cells:
            h = torch.zeros(cell.hidden_size, device=seq.device, dtype=seq.dtype)
            new_seq = []
            for k in range(seq.shape[0]):
                h = cell(seq[k], h, dt=1.0)
                new_seq.append(h)
            seq = torch.stack(new_seq)   # [K, d_model] — 作為下一層輸入

        return seq[-1]   # 最終隱藏態 [d_model]
```

- [ ] **Step 4：確認測試通過**

```bash
uv run pytest tests/test_lnn_kolmogorov.py -k "spatial" -v
```
Expected: 2 PASSED

- [ ] **Step 5：Commit**

```bash
git add src/pi_onet/lnn_kolmogorov.py tests/test_lnn_kolmogorov.py
git commit -m "feat: add SpatialCfCEncoder (K sensors -> spatial summary)"
```

---

## Task 3：TemporalCfCEncoder（T steps → h_enc，物理 Δt）

**Files:**
- Modify: `src/pi_onet/lnn_kolmogorov.py`
- Test: `tests/test_lnn_kolmogorov.py`

- [ ] **Step 1：寫失敗測試**

```python
# 加入 tests/test_lnn_kolmogorov.py
from pi_onet.lnn_kolmogorov import TemporalCfCEncoder

T = 21   # 感測器時間步數

def test_temporal_encoder_output_shape():
    """TemporalCfCEncoder: spatial_states[T, d_model] → [d_model]。"""
    enc = TemporalCfCEncoder(d_model=D, num_layers=2)
    states = torch.randn(T, D)
    h_enc = enc(states, re_norm=0.0, dt_phys=1.0)
    assert h_enc.shape == (D,)

def test_temporal_encoder_dt_effect():
    """不同 dt_phys 應產生不同 h_enc（CfC gate 依賴 Δt）。"""
    enc = TemporalCfCEncoder(d_model=D, num_layers=1)
    states = torch.randn(T, D)
    with torch.no_grad():
        h1 = enc(states, re_norm=0.0, dt_phys=1.0)
        h2 = enc(states, re_norm=0.0, dt_phys=0.1)
    assert not torch.allclose(h1, h2)

def test_temporal_encoder_re_effect():
    """不同 re_norm 應影響 h_enc（Re 作為初始隱藏態注入）。"""
    enc = TemporalCfCEncoder(d_model=D, num_layers=1)
    states = torch.randn(T, D)
    with torch.no_grad():
        h1 = enc(states, re_norm=0.0, dt_phys=1.0)
        h2 = enc(states, re_norm=2.0, dt_phys=1.0)
    assert not torch.allclose(h1, h2)
```

- [ ] **Step 2：確認測試失敗**

```bash
uv run pytest tests/test_lnn_kolmogorov.py::test_temporal_encoder_output_shape -v
```
Expected: `ImportError: cannot import name 'TemporalCfCEncoder'`

- [ ] **Step 3：實作 TemporalCfCEncoder**

加入 `src/pi_onet/lnn_kolmogorov.py`：

```python
class TemporalCfCEncoder(nn.Module):
    """What: 以 CfC 序列處理 T 個空間摘要向量 → 時序編碼 h_enc。

    Why: 使用物理 Δt（= 1.0 感測器時間單位），CfC 在此扮演 LTC ODE 的
         閉合解角色：h_enc 捕捉全部 T 步的時序動態，一步等同 RK4 多步積分。
         Re 以初始隱藏態注入，讓 Re 資訊滲透整個時序演化過程。
    """

    def __init__(self, d_model: int, num_layers: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.re_proj = nn.Linear(1, d_model)
        self.cells = nn.ModuleList([
            CfCCell(d_model, d_model) for _ in range(num_layers)
        ])

    def forward(
        self,
        spatial_states: torch.Tensor,
        re_norm: float,
        dt_phys: float,
    ) -> torch.Tensor:
        """
        spatial_states: [T, d_model]
        re_norm:        float（正規化 Re 值）
        dt_phys:        float（物理時間步長，= 1.0 for sensor data）
        Returns:        [d_model]
        """
        re_t = torch.tensor(
            [[re_norm]], dtype=spatial_states.dtype, device=spatial_states.device
        )
        seq = spatial_states   # [T, d_model]

        for layer_idx, cell in enumerate(self.cells):
            # 第一層用 Re 初始化隱藏態；後續層用 zeros
            if layer_idx == 0:
                h = self.re_proj(re_t).squeeze(0)   # [d_model]
            else:
                h = torch.zeros(self.d_model, device=seq.device, dtype=seq.dtype)

            new_seq = []
            for t in range(seq.shape[0]):
                h = cell(seq[t], h, dt=dt_phys)
                new_seq.append(h)
            seq = torch.stack(new_seq)   # [T, d_model]

        return seq[-1]   # h_enc [d_model]
```

- [ ] **Step 4：確認測試通過**

```bash
uv run pytest tests/test_lnn_kolmogorov.py -k "temporal" -v
```
Expected: 3 PASSED

- [ ] **Step 5：Commit**

```bash
git add src/pi_onet/lnn_kolmogorov.py tests/test_lnn_kolmogorov.py
git commit -m "feat: add TemporalCfCEncoder (T steps -> h_enc with physical dt)"
```

---

## Task 4：QueryCfCDecoder + LiquidOperator

**Files:**
- Modify: `src/pi_onet/lnn_kolmogorov.py`
- Test: `tests/test_lnn_kolmogorov.py`

- [ ] **Step 1：寫失敗測試**

```python
# 加入 tests/test_lnn_kolmogorov.py
from pi_onet.lnn_kolmogorov import QueryCfCDecoder, LiquidOperator, create_lnn_model

N_Q = 16

def test_query_decoder_output_shape():
    """QueryCfCDecoder: (xy[N_q,2], t_q[N_q], c[N_q], h_enc[d]) → [N_q, 1]。"""
    B = torch.randn(2, RFF_F)
    dec = QueryCfCDecoder(rff_features=RFF_F, d_model=D, d_time=8)
    xy = torch.rand(N_Q, 2) * 6.28
    t_q = torch.rand(N_Q) * 20.0
    c = torch.randint(0, 3, (N_Q,))
    h_enc = torch.randn(D)
    out = dec(xy, t_q, c, h_enc, B)
    assert out.shape == (N_Q, 1)

def test_query_decoder_vectorized():
    """Decoder 不做 for-loop：N_q=1 與 N_q=512 的 forward 均可運行。"""
    B = torch.randn(2, RFF_F)
    dec = QueryCfCDecoder(rff_features=RFF_F, d_model=D, d_time=8)
    h_enc = torch.randn(D)
    for n in (1, 512):
        xy = torch.rand(n, 2)
        t_q = torch.rand(n)
        c = torch.randint(0, 3, (n,))
        assert dec(xy, t_q, c, h_enc, B).shape == (n, 1)

def test_liquid_operator_forward_shape():
    """LiquidOperator.forward → [N_q, 1]，無任何 Attention。"""
    cfg = {
        "rff_features": RFF_F, "rff_sigma": 1.0, "d_model": D, "d_time": 8,
        "num_spatial_cfc_layers": 1, "num_temporal_cfc_layers": 1,
    }
    net = create_lnn_model(cfg)
    # 確認無 Attention
    for mod in net.modules():
        assert not isinstance(mod, nn.MultiheadAttention), "發現 MultiheadAttention"
        assert not isinstance(mod, nn.TransformerEncoder), "發現 TransformerEncoder"
    sensor_vals = torch.randn(T, K, 3)
    sensor_pos = torch.rand(K, 2) * 6.28
    xy = torch.rand(N_Q, 2) * 6.28
    t_q = torch.rand(N_Q) * 20.0
    c = torch.randint(0, 3, (N_Q,))
    out = net(sensor_vals, sensor_pos, re_norm=0.0, dt_phys=1.0, xy=xy, t_q=t_q, c=c)
    assert out.shape == (N_Q, 1)

def test_liquid_operator_backward():
    """loss.backward() 成功，output_head 有梯度。"""
    cfg = {
        "rff_features": RFF_F, "rff_sigma": 1.0, "d_model": D, "d_time": 8,
        "num_spatial_cfc_layers": 1, "num_temporal_cfc_layers": 1,
    }
    net = create_lnn_model(cfg)
    sensor_vals = torch.randn(T, K, 3)
    sensor_pos = torch.rand(K, 2) * 6.28
    xy = torch.rand(N_Q, 2) * 6.28
    t_q = torch.rand(N_Q) * 20.0
    c = torch.randint(0, 3, (N_Q,))
    out = net(sensor_vals, sensor_pos, re_norm=0.0, dt_phys=1.0, xy=xy, t_q=t_q, c=c)
    out.sum().backward()
    assert net.query_decoder.output_head.weight.grad is not None
```

- [ ] **Step 2：確認測試失敗**

```bash
uv run pytest tests/test_lnn_kolmogorov.py::test_query_decoder_output_shape -v
```
Expected: `ImportError`

- [ ] **Step 3：實作 QueryCfCDecoder + LiquidOperator**

加入 `src/pi_onet/lnn_kolmogorov.py`：

```python
class QueryCfCDecoder(nn.Module):
    """What: 以單步向量化 CfC 將 query (x,y,t,c) 解碼為 u/v/p。

    Why: h_enc 廣播至所有 N_q query 點作為初始隱藏態；CfCCell 以 batch
         [N_q, d_model] 一次運行，完全向量化，等同矩陣運算，無 for-loop。
    """

    def __init__(self, rff_features: int, d_model: int, d_time: int) -> None:
        super().__init__()
        query_in = 2 * rff_features + d_time + 8   # RFF(x,y) + time_enc + comp_emb
        self.time_proj = nn.Linear(1, d_time)
        self.component_emb = nn.Embedding(3, 8)
        nn.init.normal_(self.component_emb.weight, mean=0.0, std=0.1)
        self.query_proj = nn.Linear(query_in, d_model)
        self.cell = CfCCell(d_model, d_model)
        self.output_head = nn.Linear(d_model, 1, bias=True)
        self.component_scale = nn.Parameter(torch.ones(3))
        self.component_bias = nn.Parameter(torch.zeros(3))

    def forward(
        self,
        xy: torch.Tensor,
        t_q: torch.Tensor,
        c: torch.Tensor,
        h_enc: torch.Tensor,
        B: torch.Tensor,
    ) -> torch.Tensor:
        """
        xy:    [N_q, 2]
        t_q:   [N_q]
        c:     [N_q] long
        h_enc: [d_model]
        B:     [2, rff_features]
        Returns: [N_q, 1]
        """
        rff_q = rff_encode(xy, B)                                     # [N_q, 2*rff_f]
        time_e = self.time_proj(t_q.unsqueeze(-1))                    # [N_q, d_time]
        emb_c = self.component_emb(c)                                 # [N_q, 8]
        q = self.query_proj(torch.cat([rff_q, time_e, emb_c], dim=-1))  # [N_q, d_model]

        # h_enc: [d_model] → unsqueeze → [1, d_model] → expand → [N_q, d_model]
        # 單步向量化 CfC，無 for-loop，無 nn.MultiheadAttention
        h_0 = h_enc.unsqueeze(0).expand(q.shape[0], -1).contiguous()
        H_dec = self.cell(q, h_0, dt=1.0)                            # [N_q, d_model]

        out = self.output_head(H_dec)                                 # [N_q, 1]
        out = out * self.component_scale[c].unsqueeze(1) + self.component_bias[c].unsqueeze(1)
        return out


class LiquidOperator(nn.Module):
    """What: Pi-LNN 主模型——組合 SpatialCfCEncoder + TemporalCfCEncoder + QueryCfCDecoder。

    Why: 完整的 LNN 架構，無任何 MultiheadAttention 或 TransformerEncoder。
         CfC 在 Temporal Encoder 中使用物理 Δt 加速 LTC ODE 計算。
    """

    def __init__(
        self,
        rff_features: int,
        rff_sigma: float,
        d_model: int,
        d_time: int,
        num_spatial_cfc_layers: int,
        num_temporal_cfc_layers: int,
    ) -> None:
        super().__init__()
        B = torch.randn(2, rff_features) * rff_sigma
        self.register_buffer("B", B)
        self.spatial_encoder = SpatialCfCEncoder(
            rff_features=rff_features, d_model=d_model, num_layers=num_spatial_cfc_layers
        )
        self.temporal_encoder = TemporalCfCEncoder(
            d_model=d_model, num_layers=num_temporal_cfc_layers
        )
        self.query_decoder = QueryCfCDecoder(
            rff_features=rff_features, d_model=d_model, d_time=d_time
        )

    def encode(
        self,
        sensor_vals: torch.Tensor,
        sensor_pos: torch.Tensor,
        re_norm: float,
        dt_phys: float,
    ) -> torch.Tensor:
        """
        sensor_vals: [T, K, 3]
        sensor_pos:  [K, 2]
        Returns:     h_enc [d_model]
        """
        spatial_states = torch.stack([
            self.spatial_encoder(sensor_vals[t], sensor_pos, self.B)
            for t in range(sensor_vals.shape[0])
        ])   # [T, d_model]
        return self.temporal_encoder(spatial_states, re_norm, dt_phys)

    def forward(
        self,
        sensor_vals: torch.Tensor,
        sensor_pos: torch.Tensor,
        re_norm: float,
        dt_phys: float,
        xy: torch.Tensor,
        t_q: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Returns: [N_q, 1]"""
        h_enc = self.encode(sensor_vals, sensor_pos, re_norm, dt_phys)
        return self.query_decoder(xy, t_q, c, h_enc, self.B)


def create_lnn_model(cfg: dict) -> LiquidOperator:
    """What: 從 config dict 建立 LiquidOperator。"""
    return LiquidOperator(
        rff_features=int(cfg["rff_features"]),
        rff_sigma=float(cfg["rff_sigma"]),
        d_model=int(cfg["d_model"]),
        d_time=int(cfg["d_time"]),
        num_spatial_cfc_layers=int(cfg["num_spatial_cfc_layers"]),
        num_temporal_cfc_layers=int(cfg["num_temporal_cfc_layers"]),
    )
```

- [ ] **Step 4：確認測試通過**

```bash
uv run pytest tests/test_lnn_kolmogorov.py -k "decoder or operator" -v
```
Expected: 4 PASSED

- [ ] **Step 5：Commit**

```bash
git add src/pi_onet/lnn_kolmogorov.py tests/test_lnn_kolmogorov.py
git commit -m "feat: add QueryCfCDecoder and LiquidOperator (full LNN model)"
```

---

## Task 5：KolmogorovDataset（資料載入 + 採樣）

**Files:**
- Create: `src/pi_onet/kolmogorov_dataset.py`
- Test: `tests/test_lnn_kolmogorov.py`

- [ ] **Step 1：寫失敗測試**

```python
# 加入 tests/test_lnn_kolmogorov.py
import pytest

RE1000_JSON = "data/kolmogorov_sensors/re1000/sensors_temporal_K50_N256_t0-20.json"
RE1000_NPZ  = "data/kolmogorov_sensors/re1000/sensors_temporal_K50_N256_t0-20_dns_values.npz"
RE1000_LES  = "data/kolmogorov_les/kolmogorov_les_re1000.npy"

@pytest.mark.skipif(
    not __import__("pathlib").Path(RE1000_JSON).exists(),
    reason="資料檔案不存在"
)
def test_dataset_shapes():
    """KolmogorovDataset 載入後，sensor_vals/pos 維度正確。"""
    from pi_onet.kolmogorov_dataset import KolmogorovDataset
    ds = KolmogorovDataset(
        sensor_json=RE1000_JSON,
        sensor_npz=RE1000_NPZ,
        les_path=RE1000_LES,
        re_value=1000.0,
        train_ratio=0.8,
        seed=0,
    )
    assert ds.sensor_vals.shape[0] == 50    # K
    assert ds.sensor_vals.shape[2] == 3     # u,v,p
    assert ds.sensor_pos.shape == (50, 2)
    assert ds.dt_phys == pytest.approx(1.0)

@pytest.mark.skipif(
    not __import__("pathlib").Path(RE1000_JSON).exists(),
    reason="資料檔案不存在"
)
def test_dataset_sample_train():
    """sample_train_batch 回傳正確維度。"""
    from pi_onet.kolmogorov_dataset import KolmogorovDataset
    ds = KolmogorovDataset(
        sensor_json=RE1000_JSON, sensor_npz=RE1000_NPZ,
        les_path=RE1000_LES, re_value=1000.0, train_ratio=0.8, seed=0,
    )
    rng = np.random.default_rng(42)
    xy, t_q, c, ref = ds.sample_train_batch(rng, n=64)
    assert xy.shape == (64, 2)
    assert t_q.shape == (64,)
    assert c.shape == (64,)
    assert ref.shape == (64,)
```

- [ ] **Step 2：確認測試失敗**

```bash
uv run pytest tests/test_lnn_kolmogorov.py::test_dataset_shapes -v
```
Expected: `ImportError: cannot import name 'KolmogorovDataset'`

- [ ] **Step 3：實作 KolmogorovDataset**

建立 `src/pi_onet/kolmogorov_dataset.py`：

```python
# src/pi_onet/kolmogorov_dataset.py
"""KolmogorovDataset: 載入感測器時序 + LES ground truth，提供訓練/驗證採樣。

What: 從 JSON（感測器位置）、NPZ（感測器時序）、LES NPY（全場 ground truth）
      建立訓練資料結構，支援隨機採樣 (x, y, t, c) → ref 值。
Why:  將資料邏輯與模型解耦；lnn_kolmogorov.py 不直接接觸檔案格式。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RE_MEAN: float = 5500.0    # mean of {100, 1000, 10000, 100000} — 近似中值
RE_STD:  float = 4000.0    # 近似 std，用於正規化


@dataclass
class KolmogorovDataset:
    """Holds sensor data + LES field for one Re value.

    Attributes:
        sensor_vals: [K, T, 3] float32 — u,v,p at K sensors over T time steps
        sensor_pos:  [K, 2]   float32 — (x, y) in [0, 2π]
        sensor_time: [T]      float32 — physical time values
        les_u:  [T_les, N, N] float32
        les_v:  [T_les, N, N] float32
        les_p:  [T_les, N, N] float32
        les_x:  [N]           float32
        les_y:  [N]           float32
        les_time: [T_les]     float32
        re_value:  float
        re_norm:   float — normalised (re - RE_MEAN) / RE_STD
        dt_phys:   float — sensor time step (= 1.0)
        train_t_idx: [n_train] — LES time indices for training
        val_t_idx:   [n_val]   — LES time indices for validation
    """

    sensor_vals:  np.ndarray
    sensor_pos:   np.ndarray
    sensor_time:  np.ndarray
    les_u:        np.ndarray
    les_v:        np.ndarray
    les_p:        np.ndarray
    les_x:        np.ndarray
    les_y:        np.ndarray
    les_time:     np.ndarray
    re_value:     float
    re_norm:      float
    dt_phys:      float
    train_t_idx:  np.ndarray
    val_t_idx:    np.ndarray

    def __init__(
        self,
        sensor_json: str | Path,
        sensor_npz:  str | Path,
        les_path:    str | Path,
        re_value:    float,
        train_ratio: float = 0.8,
        seed:        int   = 42,
    ) -> None:
        rng = np.random.default_rng(seed)

        # ── 感測器位置（物理座標，單位 rad）──────────────────────────
        with open(sensor_json, encoding="utf-8") as f:
            meta = json.load(f)
        coords = np.array(meta["selected_coordinates"], dtype=np.float32)   # [K, 2]
        self.sensor_pos = coords   # (x, y) in [0, 2π]

        # ── 感測器時序 ───────────────────────────────────────────────
        npz = np.load(sensor_npz, allow_pickle=True)
        # u[K, T], v[K, T], p[K, T], time[T]
        u_s = npz["u"].astype(np.float32)    # [K, T]
        v_s = npz["v"].astype(np.float32)
        p_s = npz["p"].astype(np.float32)
        self.sensor_time = npz["time"].astype(np.float32)   # [T]
        self.sensor_vals = np.stack([u_s, v_s, p_s], axis=-1)  # [K, T, 3]
        # Δt = 感測器時間步長
        self.dt_phys = float(self.sensor_time[1] - self.sensor_time[0])

        # ── LES 全場 ─────────────────────────────────────────────────
        les = np.load(les_path, allow_pickle=True).item()
        self.les_u    = les["u"].astype(np.float32)     # [T_les, N, N]
        self.les_v    = les["v"].astype(np.float32)
        self.les_p    = les["p"].astype(np.float32)
        self.les_x    = les["x"].astype(np.float32)     # [N]
        self.les_y    = les["y"].astype(np.float32)
        self.les_time = les["time"].astype(np.float32)  # [T_les]

        # ── Re 正規化 ────────────────────────────────────────────────
        self.re_value = float(re_value)
        self.re_norm  = float((re_value - RE_MEAN) / RE_STD)

        # ── 訓練/驗證切割（依 LES 時間步切割）──────────────────────
        T_les = len(self.les_time)
        idx = np.arange(T_les)
        rng.shuffle(idx)
        n_train = int(T_les * train_ratio)
        self.train_t_idx = idx[:n_train]
        self.val_t_idx   = idx[n_train:]

    def sample_train_batch(
        self, rng: np.random.Generator, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """What: 從訓練集採樣 n 個 (x, y, t, c) 查詢點及對應 LES 參考值。

        Returns:
            xy:  [n, 2]  float32 — 物理座標 (x, y) in [0, 2π]
            t_q: [n]     float32 — 物理時間
            c:   [n]     int32   — component (0=u, 1=v, 2=p)
            ref: [n]     float32 — 對應 LES 值
        """
        return self._sample_batch(rng, self.train_t_idx, n)

    def sample_val_batch(
        self, rng: np.random.Generator, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """What: 從驗證集採樣。"""
        return self._sample_batch(rng, self.val_t_idx, n)

    def _sample_batch(
        self,
        rng: np.random.Generator,
        t_pool: np.ndarray,
        n: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t_idx = rng.choice(t_pool, size=n, replace=True)   # LES time indices
        xi    = rng.integers(0, len(self.les_x), size=n)
        yi    = rng.integers(0, len(self.les_y), size=n)
        c     = rng.integers(0, 3, size=n).astype(np.int32)

        xy  = np.stack([self.les_x[xi], self.les_y[yi]], axis=1).astype(np.float32)
        t_q = self.les_time[t_idx].astype(np.float32)

        field_map = {0: self.les_u, 1: self.les_v, 2: self.les_p}
        ref = np.array([
            field_map[int(c[i])][t_idx[i], yi[i], xi[i]]
            for i in range(n)
        ], dtype=np.float32)

        return xy, t_q, c, ref

    def sample_physics_points(
        self, rng: np.random.Generator, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """What: 採樣 n 個物理殘差點 (x, y, t)，時間在整個感測器時間範圍內均勻採樣。

        Returns:
            xy: [n, 2] float32
            t:  [n]    float32
        """
        xy = rng.uniform(0.0, 2 * np.pi, (n, 2)).astype(np.float32)
        t  = rng.uniform(self.sensor_time[0], self.sensor_time[-1], n).astype(np.float32)
        return xy, t
```

- [ ] **Step 4：確認測試通過**

```bash
uv run pytest tests/test_lnn_kolmogorov.py -k "dataset" -v
```
Expected: 2 PASSED（若資料存在）或 2 SKIPPED（若無資料）

- [ ] **Step 5：Commit**

```bash
git add src/pi_onet/kolmogorov_dataset.py tests/test_lnn_kolmogorov.py
git commit -m "feat: add KolmogorovDataset (sensor + LES data loader)"
```

---

## Task 6：物理損失（非定常 NS + Kolmogorov forcing）

**Files:**
- Modify: `src/pi_onet/lnn_kolmogorov.py`
- Test: `tests/test_lnn_kolmogorov.py`

- [ ] **Step 1：寫失敗測試**

```python
# 加入 tests/test_lnn_kolmogorov.py
def test_physics_loss_exact_solution():
    """u=sin(y), v=0, p=0 是 Kolmogorov NS 的精確解（忽略 forcing）。

    NS_x: u·∂u/∂x + v·∂u/∂y + ∂p/∂x - ν·(∂²u/∂x² + ∂²u/∂y²) = f_x
    對 u=sin(y), v=0, p=0:
      ∂u/∂t = 0, u·du/dx = 0, v·du/dy = 0, dp/dx = 0
      Laplacian(u) = -sin(y)  → ν·(-sin(y)) = -ν·sin(y)
      f_x = A·sin(k_f·y) = A·sin(4y)
      若 k_f=1, A=ν: NS_x = -ν·sin(y) - (-ν·sin(y)) = 0 ✓
    """
    from pi_onet.lnn_kolmogorov import unsteady_ns_residuals

    def u_fn(xyt): return torch.sin(xyt[:, 1:2])          # u = sin(y)
    def v_fn(xyt): return torch.zeros_like(xyt[:, 0:1])   # v = 0
    def p_fn(xyt): return torch.zeros_like(xyt[:, 0:1])   # p = 0

    nu = 0.01   # ν = 1/Re
    # 用 k_f=1, A=ν，使得 forcing 恰好抵消黏性項
    xyt = torch.rand(20, 3, requires_grad=True)
    xyt.data[:, 2] *= 20.0   # t in [0, 20]
    ns_x, ns_y, cont = unsteady_ns_residuals(
        u_fn, v_fn, p_fn, xyt, re=1.0/nu, k_f=1.0, A=nu
    )
    assert ns_x.abs().max().item() < 1e-4
    assert ns_y.abs().max().item() < 1e-4
    assert cont.abs().max().item() < 1e-5
```

- [ ] **Step 2：確認測試失敗**

```bash
uv run pytest tests/test_lnn_kolmogorov.py::test_physics_loss_exact_solution -v
```
Expected: `ImportError: cannot import name 'unsteady_ns_residuals'`

- [ ] **Step 3：實作 unsteady_ns_residuals**

加入 `src/pi_onet/lnn_kolmogorov.py`：

```python
def unsteady_ns_residuals(
    u_fn: Callable,
    v_fn: Callable,
    p_fn: Callable,
    xyt: torch.Tensor,
    re: float,
    k_f: float = 4.0,
    A:   float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """What: 2D incompressible unsteady NS + continuity residuals at collocation points.

    Why: Kolmogorov flow 加入體積力 f_x = A·sin(k_f·y)（正弦強迫），
         ∂u/∂t 項是與 LDC steady-state 的關鍵差異。
    xyt: [N, 3] = (x, y, t) with requires_grad=True
    Returns: ns_x [N,1], ns_y [N,1], cont [N,1]
    """
    u, v, p = u_fn(xyt), v_fn(xyt), p_fn(xyt)
    u_xyt = _grad(u, xyt)
    v_xyt = _grad(v, xyt)
    p_xyt = _grad(p, xyt)
    du_dx, du_dy, du_dt = u_xyt[:, 0:1], u_xyt[:, 1:2], u_xyt[:, 2:3]
    dv_dx, dv_dy, dv_dt = v_xyt[:, 0:1], v_xyt[:, 1:2], v_xyt[:, 2:3]
    dp_dx, dp_dy         = p_xyt[:, 0:1], p_xyt[:, 1:2]
    du_dx2 = _grad(du_dx, xyt)[:, 0:1]
    du_dy2 = _grad(du_dy, xyt)[:, 1:2]
    dv_dx2 = _grad(dv_dx, xyt)[:, 0:1]
    dv_dy2 = _grad(dv_dy, xyt)[:, 1:2]
    nu  = 1.0 / float(re)
    f_x = A * torch.sin(k_f * xyt[:, 1:2])   # Kolmogorov forcing
    ns_x = du_dt + u * du_dx + v * du_dy + dp_dx - nu * (du_dx2 + du_dy2) - f_x
    ns_y = dv_dt + u * dv_dx + v * dv_dy + dp_dy - nu * (dv_dx2 + dv_dy2)
    cont = du_dx + dv_dy
    return ns_x, ns_y, cont


def make_lnn_model_fn(
    net: LiquidOperator,
    sensor_vals: torch.Tensor,
    sensor_pos: torch.Tensor,
    re_norm: float,
    dt_phys: float,
    device: torch.device,
) -> Callable:
    """What: 回傳 closure (xyt, c) → [N,1]，供物理損失計算使用。

    Why: 物理損失對 xyt 做 autograd；closure 捕捉 sensor 資料與 Re 條件。
         h_enc 可在計算所有 component 前 encode 一次，節省重複計算。
    """
    net_device = next(iter(net.buffers())).device

    def model_fn(xyt: torch.Tensor, c: int) -> torch.Tensor:
        xyt_d = xyt.to(net_device)
        xy_d  = xyt_d[:, :2]
        t_q_d = xyt_d[:, 2]
        # h_enc 需要隨 xyt 的計算圖計算，不能 no_grad
        h_enc = net.encode(sensor_vals, sensor_pos, re_norm, dt_phys)
        c_t   = torch.full((xyt_d.shape[0],), c, dtype=torch.long, device=net_device)
        return net.query_decoder(xy_d, t_q_d, c_t, h_enc, net.B).to(xyt.device)

    return model_fn
```

- [ ] **Step 4：確認測試通過**

```bash
uv run pytest tests/test_lnn_kolmogorov.py::test_physics_loss_exact_solution -v
```
Expected: PASSED

- [ ] **Step 5：Commit**

```bash
git add src/pi_onet/lnn_kolmogorov.py tests/test_lnn_kolmogorov.py
git commit -m "feat: add unsteady_ns_residuals with Kolmogorov forcing"
```

---

## Task 7：Config + Entry Point

**Files:**
- Create: `configs/lnn_kolmogorov.toml`
- Modify: `pyproject.toml`
- Modify: `src/pi_onet/lnn_kolmogorov.py`（DEFAULT_LNN_ARGS + load_lnn_config + main）

- [ ] **Step 1：建立 config 檔案**

```toml
# configs/lnn_kolmogorov.toml
[train]
# 資料檔案（每個 Re 一組 JSON + NPZ + LES）
sensor_jsons = [
  "data/kolmogorov_sensors/re1000/sensors_temporal_K50_N256_t0-20.json",
]
sensor_npzs = [
  "data/kolmogorov_sensors/re1000/sensors_temporal_K50_N256_t0-20_dns_values.npz",
]
les_paths = [
  "data/kolmogorov_les/kolmogorov_les_re1000.npy",
]
re_values = [1000.0]

# 模型
rff_features = 64
rff_sigma = 2.0
d_model = 128
d_time = 16
num_spatial_cfc_layers = 2
num_temporal_cfc_layers = 2

# 損失權重
data_loss_weight = 1.0
physics_loss_weight = 0.05
continuity_weight = 1.0
kolmogorov_k_f = 4.0
kolmogorov_A = 0.1

# 訓練
iterations = 10000
num_query_points = 1024
num_physics_points = 512
learning_rate = 1e-3
weight_decay = 1e-4
lr_schedule = "cosine"
min_learning_rate = 1e-6
max_grad_norm = 1.0
checkpoint_period = 2000
seed = 42
device = "auto"
artifacts_dir = "artifacts/lnn-kolmogorov"
```

- [ ] **Step 2：加入 DEFAULT_LNN_ARGS + load_lnn_config + main 至 lnn_kolmogorov.py**

加入 `src/pi_onet/lnn_kolmogorov.py` 末尾：

```python
DEFAULT_LNN_ARGS: dict[str, Any] = {
    "sensor_jsons": None,
    "sensor_npzs": None,
    "les_paths": None,
    "re_values": None,
    "rff_features": 64,
    "rff_sigma": 2.0,
    "d_model": 128,
    "d_time": 16,
    "num_spatial_cfc_layers": 2,
    "num_temporal_cfc_layers": 2,
    "data_loss_weight": 1.0,
    "physics_loss_weight": 0.05,
    "continuity_weight": 1.0,
    "kolmogorov_k_f": 4.0,
    "kolmogorov_A": 0.1,
    "iterations": 10000,
    "num_query_points": 1024,
    "num_physics_points": 512,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "lr_schedule": "cosine",
    "min_learning_rate": 1e-6,
    "max_grad_norm": 1.0,
    "checkpoint_period": 2000,
    "seed": 42,
    "device": "auto",
    "artifacts_dir": "artifacts/lnn-kolmogorov",
}

_REMOVED_KEYS = {"nhead", "dim_feedforward", "attn_dropout", "num_encoder_layers"}


def load_lnn_config(config_path: Path | None) -> dict[str, Any]:
    """What: 載入並驗證 TOML config。舊 PiT 欄位（nhead 等）會觸發明確錯誤。"""
    if config_path is None:
        return {}
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config_data = payload.get("train", payload)
    normalized = dict(config_data)
    # 舊 PiT 欄位 → 明確失敗（不靜默忽略）
    obsolete = sorted(set(normalized) & _REMOVED_KEYS)
    if obsolete:
        raise ValueError(
            f"Config 含有已移除的 PiT 欄位（請改用 num_spatial/temporal_cfc_layers）: {obsolete}"
        )
    unknown = sorted(set(normalized) - set(DEFAULT_LNN_ARGS))
    if unknown:
        raise ValueError(f"LNN config 含有不支援的欄位: {unknown}")
    # 解析相對路徑
    for list_key in ("sensor_jsons", "sensor_npzs", "les_paths"):
        if list_key in normalized:
            normalized[list_key] = [
                str((config_path.parent / Path(p)).resolve())
                for p in normalized[list_key]
            ]
    if "artifacts_dir" in normalized:
        normalized["artifacts_dir"] = str(
            (config_path.parent / Path(normalized["artifacts_dir"])).resolve()
        )
    return normalized
```

- [ ] **Step 3：更新 pyproject.toml**

將 `pyproject.toml` 中的 `[project.scripts]` 更新為：

```toml
[project.scripts]
pit-ldc-train = "pi_onet.pit_ldc:main"
lnn-kolmogorov-train = "pi_onet.lnn_kolmogorov:main"
```

- [ ] **Step 4：確認 entry point 可載入**

```bash
uv run lnn-kolmogorov-train --help
```
Expected: 顯示 usage，不報錯

- [ ] **Step 5：Commit**

```bash
git add configs/lnn_kolmogorov.toml pyproject.toml src/pi_onet/lnn_kolmogorov.py
git commit -m "feat: add lnn-kolmogorov-train entry point and config"
```

---

## Task 8：訓練迴圈 + Smoke Test

**Files:**
- Modify: `src/pi_onet/lnn_kolmogorov.py`（train_lnn_kolmogorov + main）
- Test: `tests/test_lnn_kolmogorov.py`

- [ ] **Step 1：寫 smoke test**

```python
# 加入 tests/test_lnn_kolmogorov.py
@pytest.mark.skipif(
    not __import__("pathlib").Path(RE1000_JSON).exists(),
    reason="資料檔案不存在"
)
def test_smoke_train(tmp_path):
    """3 步訓練產生 checkpoint，loss 無 NaN。"""
    import subprocess, sys
    cfg = f"""
[train]
sensor_jsons = ["{RE1000_JSON}"]
sensor_npzs  = ["{RE1000_NPZ}"]
les_paths    = ["{RE1000_LES}"]
re_values    = [1000.0]
rff_features = 8
rff_sigma    = 1.0
d_model      = 16
d_time       = 4
num_spatial_cfc_layers  = 1
num_temporal_cfc_layers = 1
iterations           = 3
num_query_points     = 8
num_physics_points   = 4
checkpoint_period    = 2
seed                 = 0
device               = "cpu"
artifacts_dir        = "{tmp_path}/artifacts"
"""
    cfg_path = tmp_path / "smoke.toml"
    cfg_path.write_text(cfg)
    result = subprocess.run(
        [sys.executable, "-m", "pi_onet.lnn_kolmogorov", "--config", str(cfg_path)],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "nan" not in result.stdout.lower(), f"NaN 出現:\n{result.stdout}"
    ckpts = list((tmp_path / "artifacts" / "checkpoints").glob("*.pt"))
    assert len(ckpts) > 0
```

- [ ] **Step 2：確認測試失敗（train 函式不存在）**

```bash
uv run pytest tests/test_lnn_kolmogorov.py::test_smoke_train -v
```
Expected: 失敗或 SKIP（資料不存在時）

- [ ] **Step 3：實作訓練迴圈**

加入 `src/pi_onet/lnn_kolmogorov.py`：

```python
def train_lnn_kolmogorov(args: dict[str, Any]) -> None:
    """What: Pi-LNN Kolmogorov flow 訓練迴圈。

    Why: 對每個 Re case encode 感測器時序 → h_enc，
         計算 data loss（LES 全場）+ physics loss（NS 殘差）並 backward。
    """
    from pi_onet.kolmogorov_dataset import KolmogorovDataset

    device = configure_torch_runtime(args["device"])
    torch.manual_seed(args["seed"])
    rng = np.random.default_rng(args["seed"])

    artifacts_dir  = Path(args["artifacts_dir"])
    checkpoints_dir = artifacts_dir / "checkpoints"
    best_dir        = artifacts_dir / "best_validation"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    # ── 資料集 ──────────────────────────────────────────────────────
    datasets = [
        KolmogorovDataset(
            sensor_json=args["sensor_jsons"][i],
            sensor_npz =args["sensor_npzs"][i],
            les_path   =args["les_paths"][i],
            re_value   =float(args["re_values"][i]),
            train_ratio=0.8,
            seed       =args["seed"],
        )
        for i in range(len(args["re_values"]))
    ]
    num_re = len(datasets)

    # 預先轉換感測器資料至 device（靜態）
    # ds.sensor_vals: [K, T, 3]（dataset 儲存格式）
    # transpose → [T, K, 3]（LiquidOperator.encode 所需格式）
    # LiquidOperator 邊界：sensor_vals 必須是 [T, K, 3]
    sensor_vals_list = [
        torch.tensor(ds.sensor_vals.transpose(1, 0, 2), dtype=torch.float32, device=device)
        for ds in datasets
    ]
    sensor_pos_list = [
        torch.tensor(ds.sensor_pos, dtype=torch.float32, device=device)
        for ds in datasets
    ]

    # ── 模型 ────────────────────────────────────────────────────────
    net = create_lnn_model(args).to(device)
    print("=== Configuration ===")
    print(f"trainable_parameters: {count_parameters(net)}")

    # ── Optimizer + LR scheduler ────────────────────────────────────
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args["iterations"], eta_min=args["min_learning_rate"]
    ) if args["lr_schedule"] == "cosine" else None

    best_val_metric = float("inf")
    k_f = float(args["kolmogorov_k_f"])
    A   = float(args["kolmogorov_A"])

    print("=== Training ===")
    print(f"{'Step':<8} {'L_data':>12} {'L_phys':>12} {'L_total':>12}")

    for step in range(1, args["iterations"] + 1):
        net.train()
        optimizer.zero_grad()

        # ── Data loss ───────────────────────────────────────────────
        l_data = torch.zeros(1, device=device)
        for i, ds in enumerate(datasets):
            xy_np, t_np, c_np, ref_np = ds.sample_train_batch(
                rng, n=args["num_query_points"]
            )
            xy  = torch.tensor(xy_np,  device=device)
            t_q = torch.tensor(t_np,   device=device)
            c   = torch.tensor(c_np,   dtype=torch.long, device=device)
            ref = torch.tensor(ref_np, device=device)
            pred = net(
                sensor_vals_list[i], sensor_pos_list[i],
                re_norm=ds.re_norm, dt_phys=ds.dt_phys,
                xy=xy, t_q=t_q, c=c,
            ).squeeze(1)
            l_data = l_data + torch.mean((pred - ref) ** 2)
        l_data = l_data / num_re

        # ── Physics loss ─────────────────────────────────────────────
        net.eval()
        l_ns_total   = torch.zeros(1, device=device)
        l_cont_total = torch.zeros(1, device=device)
        for i, ds in enumerate(datasets):
            xy_np, t_np = ds.sample_physics_points(rng, n=args["num_physics_points"])
            xyt = torch.tensor(
                np.concatenate([xy_np, t_np[:, None]], axis=1),
                device=device, requires_grad=True,
            )
            model_fn = make_lnn_model_fn(
                net, sensor_vals_list[i], sensor_pos_list[i],
                re_norm=ds.re_norm, dt_phys=ds.dt_phys, device=device,
            )
            u_fn = lambda xyt_, fn=model_fn: fn(xyt_, c=0)
            v_fn = lambda xyt_, fn=model_fn: fn(xyt_, c=1)
            p_fn = lambda xyt_, fn=model_fn: fn(xyt_, c=2)
            ns_x, ns_y, cont = unsteady_ns_residuals(
                u_fn, v_fn, p_fn, xyt, re=ds.re_value, k_f=k_f, A=A
            )
            l_ns_total   = l_ns_total   + torch.mean(ns_x**2) + torch.mean(ns_y**2)
            l_cont_total = l_cont_total + torch.mean(cont**2)
        net.train()

        l_ns_total   = l_ns_total   / num_re
        l_cont_total = l_cont_total / num_re
        l_physics    = l_ns_total + args["continuity_weight"] * l_cont_total
        l_total = (
            args["data_loss_weight"]    * l_data
            + args["physics_loss_weight"] * l_physics
        )
        l_total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), float(args["max_grad_norm"]))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % max(1, args["iterations"] // 10) == 0 or step == 1:
            print(
                f"{step:<8} {l_data.item():>12.4e}"
                f" {l_physics.item():>12.4e} {l_total.item():>12.4e}"
            )

        # ── Checkpoint ───────────────────────────────────────────────
        if args["checkpoint_period"] > 0 and step % args["checkpoint_period"] == 0:
            ckpt = checkpoints_dir / f"lnn_kolmogorov_step_{step}.pt"
            torch.save(net.state_dict(), str(ckpt))

    final = artifacts_dir / "lnn_kolmogorov_final.pt"
    torch.save(net.state_dict(), str(final))
    write_json(artifacts_dir / "experiment_manifest.json", {
        "configuration": {k: v for k, v in args.items()
                          if k not in ("sensor_jsons", "sensor_npzs", "les_paths")},
        "final_checkpoint": str(final),
    })
    print("=== Done ===")


def main() -> None:
    """What: Entry point for lnn-kolmogorov-train CLI。"""
    import argparse
    parser = argparse.ArgumentParser(
        description="Train Pi-LNN on Kolmogorov flow (full-field reconstruction)."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=None)
    cli_args = parser.parse_args()

    config = dict(DEFAULT_LNN_ARGS)
    config.update(load_lnn_config(cli_args.config))
    if cli_args.device is not None:
        config["device"] = cli_args.device

    train_lnn_kolmogorov(config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4：確認 smoke test 通過**

```bash
uv run pytest tests/test_lnn_kolmogorov.py::test_smoke_train -v -s
```
Expected: PASSED（需資料存在）；若資料不存在則 SKIPPED

- [ ] **Step 5：執行完整測試**

```bash
uv run pytest tests/test_lnn_kolmogorov.py -v
```
Expected: 所有測試 PASSED 或 SKIPPED（資料不存在的測試）

- [ ] **Step 6：Commit**

```bash
git add src/pi_onet/lnn_kolmogorov.py tests/test_lnn_kolmogorov.py
git commit -m "feat: add training loop and CLI for Pi-LNN Kolmogorov flow"
```

---

## 最終驗收

- [ ] `uv run pytest tests/ -v` — 全部 PASSED（或 SKIPPED）
- [ ] `grep -r "MultiheadAttention\|TransformerEncoder" src/pi_onet/lnn_kolmogorov.py` — 無輸出
- [ ] `uv run lnn-kolmogorov-train --config configs/lnn_kolmogorov.toml --device cpu` — 開始訓練，loss 無 NaN
