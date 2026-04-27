"""Reusable nn.Module building blocks: CfC cell, residual MLP, token self-attention."""
from __future__ import annotations

import torch
import torch.nn as nn


class CfCCell(nn.Module):
    """What: Closed-form Continuous-time recurrent cell.

    Why: 以閉合解近似連續時間動態，避免 LTC 的數值 ODE 求解成本。
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        combined = input_size + hidden_size
        self.ff1 = nn.Linear(combined, hidden_size)
        self.ff2 = nn.Linear(combined, hidden_size)
        self.log_tau_a = nn.Parameter(torch.linspace(-1.0, 1.0, hidden_size))
        self.time_b = nn.Linear(combined, hidden_size)
        nn.init.xavier_uniform_(self.time_b.weight)
        nn.init.zeros_(self.time_b.bias)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        xh = torch.cat([x, h], dim=-1)
        f1 = torch.tanh(self.ff1(xh))
        f2 = torch.tanh(self.ff2(xh))
        tau_a = torch.exp(self.log_tau_a)
        t_b = self.time_b(xh)
        if isinstance(dt, torch.Tensor) and dt.dim() > 0:
            dt = dt.unsqueeze(-1)
        gate = torch.sigmoid(-tau_a * dt + t_b)
        return gate * f1 + (1.0 - gate) * f2


class ResidualMLPBlock(nn.Module):
    """What: 輕量殘差 MLP block。

    Why: 在 trunk path 上保留基本非線性表達力，同時維持局部可推理性。
    """

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        return x + y


class TokenSelfAttentionBlock(nn.Module):
    """What: 在 token 集合內做一次輕量自注意力訊息傳遞。

    Why: 讓感測器 token 在進入 temporal CfC 前先交換空間上下文，避免每個 token
         只攜帶局部量測歷史而缺少鄰域耦合資訊。
    """

    def __init__(self, d_model: int, num_heads: int = 4) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} 必須能被 num_heads={num_heads} 整除")
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        x = x + attn_out
        y = self.norm2(x)
        return x + self.ff(y)
