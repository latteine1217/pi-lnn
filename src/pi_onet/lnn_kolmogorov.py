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
        """
        input_size:  維度 of input x（encoder 中 = d_model）
        hidden_size: 維度 of hidden state h（= d_model）
        combined:    input_size + hidden_size，作為 ff1/ff2/time_a/time_b 的輸入維度
        """
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


class TemporalCfCEncoder(nn.Module):
    """What: 以 CfC 序列處理 T 個空間摘要向量 → 時序編碼 h_enc。

    Why: 使用物理 Δt（= 1.0 感測器時間單位），CfC 在此扮演 LTC ODE 的
         閉合解角色：h_enc 捕捉全部 T 步的時序動態，一步等同 RK4 多步積分。
         Re 以持續殘差（re_bias）方式加入每步輸入，確保 Re 資訊貫穿整個時序演化，
         避免初始隱藏態影響在長序列中被沖洗掉。
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
        re_bias = self.re_proj(re_t).squeeze(0)   # [d_model]
        # Re 以殘差方式加入每個時間步的輸入，確保 Re 資訊貫穿整個序列
        seq = spatial_states + re_bias.unsqueeze(0)   # [T, d_model]

        for cell in self.cells:
            h = torch.zeros(self.d_model, device=seq.device, dtype=seq.dtype)
            new_seq = []
            for t in range(seq.shape[0]):
                h = cell(seq[t], h, dt=dt_phys)
                new_seq.append(h)
            seq = torch.stack(new_seq)   # [T, d_model]

        return seq[-1]   # h_enc [d_model]
