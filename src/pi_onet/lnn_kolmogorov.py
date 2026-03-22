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
