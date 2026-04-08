# src/pi_onet/lnn_kolmogorov.py
"""Pi-LNN: core DeepONet + CfC model and training loop for Kolmogorov flow.

What: 以 CfC 作為 branch-side temporal encoder，並以 DeepONet trunk 解碼 query。
Why:  CFC 保留連續時間時序偏置；DeepONet 提供更清晰的 operator factorization，
      方便後續 branch/trunk 解耦與架構擴充。
"""
from __future__ import annotations

import json
import math
import os
import tomllib
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_torch_device(device_preference: str) -> torch.device:
    """What: 解析使用者指定的裝置偏好並回傳可用裝置。"""
    preference = device_preference.lower()
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("指定 --device cuda，但目前環境沒有可用 CUDA。")
        return torch.device("cuda")
    if preference == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("指定 --device mps，但目前環境沒有可用 Metal (MPS)。")
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")
    raise ValueError(f"不支援的 device: {device_preference}")


def configure_torch_runtime(device_preference: str) -> torch.device:
    """What: 啟用 PyTorch 執行環境並回傳實際使用裝置。"""
    torch.set_float32_matmul_precision("high")
    device = _resolve_torch_device(device_preference)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return device


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """What: 用 autograd 計算一階偏導。

    Why: physics residual 依賴對 (x, y, t) 的偏導，若輸出與輸入無關則直接回零，
         避免在 sparse-data 主線上出現 silent None 梯度。
    """
    if y.grad_fn is None and not y.requires_grad:
        return torch.zeros_like(x)
    grad = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True, allow_unused=True)[0]
    if grad is None:
        return torch.zeros_like(x)
    return grad


def count_parameters(model: torch.nn.Module) -> int:
    """What: 計算可訓練參數總數。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_json(path: Path, data: dict) -> None:
    """What: 以格式化 JSON 寫出結構化輸出。"""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def periodic_fourier_encode(z: torch.Tensor, domain_length: float, n_harmonics: int) -> torch.Tensor:
    """What: 對 2D 座標 (x, y) 做確定性多諧波週期 Fourier 編碼。

    Why: 取代 RFF 的隨機頻率抽樣，改用整數倍 2π/L 頻率。
         1) 嚴格滿足 [0,L]^2 週期邊界條件（與 jaxpi PeriodEmbs 相同設計）。
         2) x/y 軸獨立編碼，消除 RFF 隨機角度偏差所造成的 x-stripe 偽影。
         3) 無隨機性，結果與 seed 無關。

    Returns:
        [N, 4 * n_harmonics]
        排列：[sin(2πk x/L), cos(2πk x/L), sin(2πk y/L), cos(2πk y/L)] for k=1..n_harmonics
    """
    x = z[:, 0:1]
    y = z[:, 1:2]
    feats = []
    for k in range(1, n_harmonics + 1):
        c = 2.0 * torch.pi * k / domain_length
        feats += [torch.sin(c * x), torch.cos(c * x), torch.sin(c * y), torch.cos(c * y)]
    return torch.cat(feats, dim=-1)


def temporal_phase_anchor(t: torch.Tensor, T_total: float, n_harmonics: int = 2) -> torch.Tensor:
    """What: 產生絕對時間的確定性 temporal-phase-anchor 特徵。

    Why: trunk 目前只有 dt_to_query（相對時間），在 chaotic 流場中，
         同樣的 dt 但不同絕對時間 t 的動態完全不同。
         注入 sin/cos(2π n t / T_total) 提供絕對時間定位，
         讓模型可以區分「t=0.1 的流場狀態」與「t=4.1 的流場狀態」。
         與空間的 forcing_phase_anchor 對稱設計。

    Args:
        t: [N, 1] 絕對時間
        T_total: 模擬總時長，用於正規化
        n_harmonics: 諧波數；每個諧波貢獻 sin/cos 共 2 維

    Returns:
        [N, 2 * n_harmonics]
    """
    feats = []
    for n in range(1, n_harmonics + 1):
        angle = (2.0 * torch.pi * n / T_total) * t
        feats.extend([torch.sin(angle), torch.cos(angle)])
    return torch.cat(feats, dim=-1)


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


class SpatialSetEncoder(nn.Module):
    """What: 將單一時刻的感測器集合編碼成保留局部 identity 的 sensor tokens。

    Why: 先保留每個 sensor 的 RFF(x,y)+觀測通道局部訊息，不在 spatial branch
         提前做混合，讓 decoder 負責主要的 cross-attention 與讀取。
    """

    def __init__(
        self,
        fourier_harmonics: int,
        sensor_value_dim: int,
        d_model: int,
        num_layers: int,
        domain_length: float = 1.0,
    ) -> None:
        super().__init__()
        self.domain_length = float(domain_length)
        self.fourier_harmonics = int(fourier_harmonics)
        self.sensor_value_dim = int(sensor_value_dim)
        base_in = 4 * fourier_harmonics + self.sensor_value_dim
        hidden = 2 * d_model
        depth = max(num_layers, 1)
        self.base_norm = nn.LayerNorm(base_in)
        self.token_in = nn.Sequential(
            nn.LayerNorm(base_in),
            nn.Linear(base_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(d_model=d_model, hidden_dim=2 * d_model)
            for _ in range(depth)
        ])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(
        self,
        sensor_vals: torch.Tensor,
        sensor_pos: torch.Tensor,
    ) -> torch.Tensor:
        pos_enc = periodic_fourier_encode(sensor_pos, self.domain_length, self.fourier_harmonics)
        base = torch.cat([pos_enc, sensor_vals], dim=-1)
        tokens = self.token_in(self.base_norm(base))
        for block in self.blocks:
            tokens = block(tokens)
        return self.out_proj(tokens)


class TemporalCfCEncoder(nn.Module):
    """What: 以 CfC 演化 sensor token 序列，產生 causal token states。

    Why: 保留每個 sensor token 的連續時間動態，讓 decoder 能直接讀取感測器級上下文。
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_token_attention_layers: int = 1,
        token_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.re_proj = nn.Linear(1, d_model)
        self.token_blocks = nn.ModuleList([
            TokenSelfAttentionBlock(d_model=d_model, num_heads=token_attention_heads)
            for _ in range(max(num_token_attention_layers, 0))
        ])
        self.cells = nn.ModuleList([CfCCell(d_model, d_model) for _ in range(num_layers)])

    def _re_bias(self, re_norm: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        re_t = torch.tensor([[re_norm]], dtype=dtype, device=device)
        return self.re_proj(re_t).squeeze(0)

    def forward(
        self,
        spatial_states: torch.Tensor,
        re_norm: float,
        sensor_time: torch.Tensor,
    ) -> torch.Tensor:
        dts = torch.cat([sensor_time[:1], sensor_time[1:] - sensor_time[:-1]])
        re_bias = self._re_bias(re_norm, spatial_states.device, spatial_states.dtype).view(1, 1, -1)
        seq = spatial_states
        for layer_idx, cell in enumerate(self.cells):
            h = torch.zeros(seq.shape[1], self.d_model, device=seq.device, dtype=seq.dtype)
            outputs = []
            for t in range(seq.shape[0]):
                x_t = seq[t]
                if self.token_blocks:
                    block_idx = min(len(self.token_blocks) - 1, layer_idx)
                    x_t = self.token_blocks[block_idx](x_t.unsqueeze(0)).squeeze(0)
                x_t = x_t + re_bias.squeeze(0)
                h = cell(x_t, h, dt=dts[t])
                outputs.append(h)
            # 層間殘差：第二層起加上前一層輸出，防止多層 CfC 的信號退化。
            # 單層時 layer_idx=0 跳過，不影響現有實驗。
            new_seq = torch.stack(outputs)
            seq = new_seq + seq if layer_idx > 0 else new_seq
        return seq

    def init_hidden(
        self,
        num_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        return [torch.zeros(num_tokens, self.d_model, device=device, dtype=dtype) for _ in self.cells]

    def step(
        self,
        spatial_state: torch.Tensor,
        h_list: list[torch.Tensor],
        re_norm: float,
        dt: float,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        inp = spatial_state
        re_bias = self._re_bias(
            re_norm,
            spatial_state.device,
            spatial_state.dtype,
        ).view(1, -1)
        new_h_list: list[torch.Tensor] = []
        for layer_idx, (cell, h) in enumerate(zip(self.cells, h_list)):
            if self.token_blocks:
                block_idx = min(len(self.token_blocks) - 1, layer_idx)
                inp = self.token_blocks[block_idx](inp.unsqueeze(0)).squeeze(0)
            inp = inp + re_bias
            new_h = cell(inp, h, dt=dt)
            # 層間殘差：加上前一層的輸出（與 forward() 一致）。
            new_h = new_h + new_h_list[layer_idx - 1] if layer_idx > 0 else new_h
            inp = new_h
            new_h_list.append(new_h)
        return new_h_list[-1], new_h_list


class DeepONetCfCDecoder(nn.Module):
    """What: 以 CfC token states 作 branch、以 query 作 trunk 的 DeepONet 解碼器。

    Why: query 先從 branch tokens 做 cross-attention 取回需要的時空上下文，
         再與 trunk basis 融合，比單一 hidden state 更符合 sparse-sensor operator learning。
    """

    def __init__(
        self,
        fourier_harmonics: int,
        d_model: int,
        d_time: int,
        domain_length: float = 1.0,
        use_temporal_anchor: bool = False,
        T_total: float = 5.0,
        temporal_anchor_harmonics: int = 2,
        num_query_mlp_layers: int = 0,
        query_mlp_hidden_dim: int = 256,
        output_head_gain: float = 1.0,
        operator_rank: int | None = None,
        fusion_temperature_init: float | None = None,
    ) -> None:
        super().__init__()
        self.fourier_harmonics = int(fourier_harmonics)
        self.use_temporal_anchor = bool(use_temporal_anchor)
        self.T_total = float(T_total)
        self.temporal_anchor_harmonics = int(temporal_anchor_harmonics)
        temporal_dim = 2 * self.temporal_anchor_harmonics if self.use_temporal_anchor else 0
        query_in = 4 * fourier_harmonics + temporal_dim + d_time + 8
        rank = d_model if operator_rank is None else operator_rank
        if rank <= 0:
            raise ValueError(f"operator_rank 必須 > 0，收到 {rank}")
        self.rank = rank
        self.domain_length = float(domain_length)
        self.time_proj = nn.Linear(1, d_time)
        self.component_emb = nn.Embedding(3, 8)
        nn.init.normal_(self.component_emb.weight, mean=0.0, std=0.1)
        self.trunk_in = nn.Linear(query_in, query_mlp_hidden_dim)
        self.trunk_blocks = nn.ModuleList([
            ResidualMLPBlock(d_model=query_mlp_hidden_dim, hidden_dim=query_mlp_hidden_dim)
            for _ in range(num_query_mlp_layers)
        ])
        if query_mlp_hidden_dim % 4 != 0:
            raise ValueError(
                f"query_mlp_hidden_dim 必須能被 4 整除，收到 {query_mlp_hidden_dim}"
            )
        self.branch_norm = nn.LayerNorm(query_mlp_hidden_dim)
        self.branch_token_proj = nn.Linear(d_model, query_mlp_hidden_dim)
        self.branch_query_proj = nn.Linear(query_mlp_hidden_dim, query_mlp_hidden_dim)
        self.branch_key_proj = nn.Linear(query_mlp_hidden_dim, query_mlp_hidden_dim)
        self.branch_value_proj = nn.Linear(query_mlp_hidden_dim, query_mlp_hidden_dim)
        self.relpos_bias = nn.Sequential(
            nn.LayerNorm(1),
            nn.Linear(1, query_mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(query_mlp_hidden_dim, 1),
        )
        self.branch_context = nn.Sequential(
            nn.LayerNorm(query_mlp_hidden_dim),
            nn.Linear(query_mlp_hidden_dim, query_mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(query_mlp_hidden_dim, query_mlp_hidden_dim),
        )
        self.trunk_out = nn.Linear(query_mlp_hidden_dim, 3 * rank)
        self.branch_proj = nn.Linear(query_mlp_hidden_dim, 3 * rank)
        nn.init.xavier_normal_(self.trunk_out.weight, gain=output_head_gain)
        nn.init.zeros_(self.trunk_out.bias)
        nn.init.xavier_normal_(self.branch_proj.weight, gain=output_head_gain)
        nn.init.zeros_(self.branch_proj.bias)
        temp_init = (1.0 / math.sqrt(rank)) if fusion_temperature_init is None else fusion_temperature_init
        if temp_init <= 0.0:
            raise ValueError(f"fusion_temperature_init 必須 > 0，收到 {temp_init}")
        # shape (1,) 而非 0-dim，避免 ScheduleFreeWrapper XOR swap 在 MPS 上失敗。
        self.log_fusion_temperature = nn.Parameter(torch.tensor([math.log(temp_init)], dtype=torch.float32))
        self.component_scale = nn.Parameter(torch.ones(3))
        self.component_bias = nn.Parameter(torch.zeros(3))

    def forward(
        self,
        xy: torch.Tensor,
        t_q: torch.Tensor,
        c: torch.Tensor,
        h_states: torch.Tensor,
        sensor_time: torch.Tensor,
        sensor_pos: torch.Tensor,
    ) -> torch.Tensor:
        idx = torch.searchsorted(sensor_time.contiguous(), t_q.contiguous(), right=True) - 1
        idx = idx.clamp(0, h_states.shape[0] - 1)
        h_branch_tokens = h_states[idx]
        dt_to_query = (t_q - sensor_time[idx]).clamp(min=0.0)
        pos_enc = periodic_fourier_encode(xy, self.domain_length, self.fourier_harmonics)
        time_e = self.time_proj(dt_to_query.unsqueeze(-1))
        emb_c = self.component_emb(c)
        trunk_inputs = [pos_enc]
        if self.use_temporal_anchor:
            trunk_inputs.append(temporal_phase_anchor(t_q.unsqueeze(-1), self.T_total, self.temporal_anchor_harmonics))
        trunk_inputs.extend([time_e, emb_c])
        trunk_feat = F.silu(self.trunk_in(torch.cat(trunk_inputs, dim=-1)))
        for block in self.trunk_blocks:
            trunk_feat = block(trunk_feat)

        trunk_basis = self.trunk_out(trunk_feat).view(-1, 3, self.rank)
        branch_tokens = self.branch_token_proj(h_branch_tokens)
        branch_query = self.branch_norm(trunk_feat)
        q = self.branch_query_proj(branch_query)
        k = self.branch_key_proj(branch_tokens)
        v = self.branch_value_proj(branch_tokens)

        rel = xy.unsqueeze(1) - sensor_pos.unsqueeze(0)
        rel = rel - torch.round(rel / self.domain_length) * self.domain_length
        rel_r = torch.linalg.norm(rel, dim=-1, keepdim=True)
        # 只用距離（等向量），移除方向分量 (rel_x, rel_y)。
        # Why: 感測器 x 分佈非均勻，方向向量會將 x-column 非均勻性注入 attention bias，
        #      造成 x 方向條紋偽影；距離已足夠描述近鄰感測器的貢獻強度。
        rel_bias = self.relpos_bias(rel_r).squeeze(-1)
        scores = torch.einsum("nd,nkd->nk", q, k) / math.sqrt(k.shape[-1])
        scores = scores + rel_bias
        attn = torch.softmax(scores, dim=1)
        branch_ctx = torch.einsum("nk,nkd->nd", attn, v)
        branch_ctx = branch_ctx + self.branch_context(branch_ctx)
        branch_basis = self.branch_proj(branch_ctx).view(-1, 3, self.rank)
        comp_idx = c.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.rank)
        trunk_sel = trunk_basis.gather(1, comp_idx).squeeze(1)
        branch_sel = branch_basis.gather(1, comp_idx).squeeze(1)
        fusion_temperature = torch.exp(self.log_fusion_temperature).to(trunk_sel.dtype)
        out = torch.sum(trunk_sel * branch_sel, dim=1, keepdim=True) * fusion_temperature
        return out * self.component_scale[c].unsqueeze(1) + self.component_bias[c].unsqueeze(1)


class LiquidOperator(nn.Module):
    """What: 核心 Pi-LNN 模型。

    Why: 僅保留資料 -> Spatial encoder -> Temporal CfC branch -> DeepONet trunk 的最短主線。
    """

    def __init__(
        self,
        fourier_harmonics: int,
        sensor_value_dim: int,
        d_model: int,
        d_time: int,
        num_spatial_cfc_layers: int,
        num_temporal_cfc_layers: int,
        domain_length: float = 1.0,
        use_temporal_anchor: bool = False,
        T_total: float = 5.0,
        temporal_anchor_harmonics: int = 2,
        num_token_attention_layers: int = 1,
        token_attention_heads: int = 4,
        num_query_mlp_layers: int = 0,
        query_mlp_hidden_dim: int = 256,
        num_query_cfc_layers: int = 1,
        query_gate_bias_span: float = 1.0,
        output_head_gain: float = 1.0,
        operator_rank: int | None = None,
        fusion_temperature_init: float | None = None,
    ) -> None:
        super().__init__()
        self.spatial_encoder = SpatialSetEncoder(
            fourier_harmonics,
            sensor_value_dim,
            d_model,
            num_spatial_cfc_layers,
            domain_length=domain_length,
        )
        self.temporal_encoder = TemporalCfCEncoder(
            d_model,
            num_temporal_cfc_layers,
            num_token_attention_layers=num_token_attention_layers,
            token_attention_heads=token_attention_heads,
        )
        self.query_decoder = DeepONetCfCDecoder(
            fourier_harmonics=fourier_harmonics,
            d_model=d_model,
            d_time=d_time,
            domain_length=domain_length,
            use_temporal_anchor=use_temporal_anchor,
            T_total=T_total,
            temporal_anchor_harmonics=temporal_anchor_harmonics,
            num_query_mlp_layers=num_query_mlp_layers,
            query_mlp_hidden_dim=query_mlp_hidden_dim,
            output_head_gain=output_head_gain,
            operator_rank=operator_rank,
            fusion_temperature_init=fusion_temperature_init,
        )

    def encode(
        self,
        sensor_vals: torch.Tensor,
        sensor_pos: torch.Tensor,
        re_norm: float,
        sensor_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_states = torch.stack([
            self.spatial_encoder(sensor_vals[t], sensor_pos)
            for t in range(sensor_vals.shape[0])
        ])
        return self.temporal_encoder(spatial_states, re_norm, sensor_time), sensor_time

    def update_state(
        self,
        sensor_vals_t: torch.Tensor,
        sensor_pos: torch.Tensor,
        re_norm: float,
        dt: float,
        h_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        spatial = self.spatial_encoder(sensor_vals_t, sensor_pos)
        return self.temporal_encoder.step(spatial, h_list, re_norm, dt)

    def predict(
        self,
        xy: torch.Tensor,
        t_q: torch.Tensor,
        c: torch.Tensor,
        h_out: torch.Tensor,
        t_last: float,
        sensor_pos: torch.Tensor,
    ) -> torch.Tensor:
        h_states = h_out.unsqueeze(0)
        s_time = torch.tensor([t_last], device=h_out.device, dtype=h_out.dtype)
        return self.query_decoder(xy, t_q, c, h_states, s_time, sensor_pos)


    def forward(
        self,
        sensor_vals: torch.Tensor,
        sensor_pos: torch.Tensor,
        re_norm: float,
        sensor_time: torch.Tensor,
        xy: torch.Tensor,
        t_q: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        h_states, s_time = self.encode(sensor_vals, sensor_pos, re_norm, sensor_time)
        return self.query_decoder(xy, t_q, c, h_states, s_time, sensor_pos)


def create_lnn_model(cfg: dict[str, Any]) -> LiquidOperator:
    """What: 從 config 建立核心 LiquidOperator。"""
    return LiquidOperator(
        fourier_harmonics=int(cfg.get("fourier_harmonics", 8)),
        sensor_value_dim=len(cfg.get("observed_sensor_channels", ["u", "v"])),
        d_model=int(cfg["d_model"]),
        d_time=int(cfg["d_time"]),
        num_spatial_cfc_layers=int(cfg["num_spatial_cfc_layers"]),
        num_temporal_cfc_layers=int(cfg["num_temporal_cfc_layers"]),
        domain_length=float(cfg.get("domain_length", 1.0)),
        use_temporal_anchor=bool(cfg.get("use_temporal_anchor", False)),
        T_total=float(cfg.get("T_total", 5.0)),
        temporal_anchor_harmonics=int(cfg.get("temporal_anchor_harmonics", 2)),
        num_token_attention_layers=int(cfg.get("num_token_attention_layers", 1)),
        token_attention_heads=int(cfg.get("token_attention_heads", 4)),
        num_query_mlp_layers=int(cfg.get("num_query_mlp_layers", 0)),
        query_mlp_hidden_dim=int(cfg.get("query_mlp_hidden_dim", 256)),
        num_query_cfc_layers=int(cfg.get("num_query_cfc_layers", 1)),
        query_gate_bias_span=float(cfg.get("query_gate_bias_span", 1.0)),
        output_head_gain=float(cfg.get("output_head_gain", 1.0)),
        operator_rank=(
            int(cfg["operator_rank"]) if "operator_rank" in cfg and cfg["operator_rank"] is not None else None
        ),
        fusion_temperature_init=(
            float(cfg["fusion_temperature_init"])
            if "fusion_temperature_init" in cfg and cfg["fusion_temperature_init"] is not None
            else None
        ),
    )


def unsteady_ns_residuals(
    u_fn: Callable,
    v_fn: Callable,
    p_fn: Callable,
    xyt: torch.Tensor,
    re: float,
    k_f: float = 4.0,
    A: float = 0.1,
    domain_length: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """What: 2D incompressible NS 的 primitive-variable momentum 與 continuity 殘差。

    Why: sparse-data 主線的實際觀測仍只有 u,v，但 momentum equation 需要壓力梯度。
         因此 p 回到模型的內部 physics 場，只參與 PDE 殘差，不作資料 supervision。
    """
    u, v, p = u_fn(xyt), v_fn(xyt), p_fn(xyt)
    u_xyt = _grad(u, xyt)
    v_xyt = _grad(v, xyt)
    p_xyt = _grad(p, xyt)
    du_dx, du_dy, du_dt = u_xyt[:, 0:1], u_xyt[:, 1:2], u_xyt[:, 2:3]
    dv_dx, dv_dy, dv_dt = v_xyt[:, 0:1], v_xyt[:, 1:2], v_xyt[:, 2:3]
    dp_dx, dp_dy = p_xyt[:, 0:1], p_xyt[:, 1:2]
    du_dx2 = _grad(du_dx, xyt)[:, 0:1]
    du_dy2 = _grad(du_dy, xyt)[:, 1:2]
    dv_dx2 = _grad(dv_dx, xyt)[:, 0:1]
    dv_dy2 = _grad(dv_dy, xyt)[:, 1:2]
    nu = 1.0 / float(re)
    forcing_wavenumber = (2.0 * torch.pi * float(k_f)) / float(domain_length)
    forcing_x = A * torch.sin(forcing_wavenumber * xyt[:, 1:2])
    mom_u = du_dt + u * du_dx + v * du_dy + dp_dx - nu * (du_dx2 + du_dy2) - forcing_x
    mom_v = dv_dt + u * dv_dx + v * dv_dy + dp_dy - nu * (dv_dx2 + dv_dy2)
    cont = du_dx + dv_dy
    return mom_u, mom_v, cont


def make_lnn_model_fn(
    net: LiquidOperator,
    sensor_vals: torch.Tensor,
    sensor_pos: torch.Tensor,
    re_norm: float,
    sensor_time: torch.Tensor,
    device: torch.device,
) -> Callable:
    """What: 建立物理 loss 所需的 closure。"""
    net_device = next(iter(net.parameters())).device
    h_states, s_time = net.encode(sensor_vals, sensor_pos, re_norm, sensor_time)

    def model_fn(xyt: torch.Tensor, c: int) -> torch.Tensor:
        xyt_d = xyt.to(net_device)
        xy_d = xyt_d[:, :2]
        t_q_d = xyt_d[:, 2]
        c_t = torch.full((xyt_d.shape[0],), c, dtype=torch.long, device=net_device)
        return net.query_decoder(xy_d, t_q_d, c_t, h_states, s_time, sensor_pos).to(xyt.device)

    return model_fn


def observed_channel_prediction(
    net: LiquidOperator,
    xy: torch.Tensor,
    t_q: torch.Tensor,
    c_obs: torch.Tensor,
    observed_channel_names: tuple[str, ...],
    observed_channel_mean: torch.Tensor,
    observed_channel_std: torch.Tensor,
    h_states: torch.Tensor,
    s_time: torch.Tensor,
    sensor_pos: torch.Tensor,
) -> torch.Tensor:
    """What: 依實際觀測通道名稱產生對應預測值。

    Why: sparse-data 主線目前只監督真實可量測的 u,v。p 僅保留在 physics residual
         內部使用，避免在資料項中引入不可量測通道。
    """
    preds = torch.empty_like(t_q)
    unique_obs = torch.unique(c_obs).tolist()
    for obs_idx in unique_obs:
        mask = c_obs == int(obs_idx)
        channel_name = observed_channel_names[int(obs_idx)]
        mean = observed_channel_mean[int(obs_idx)]
        std = observed_channel_std[int(obs_idx)]
        xy_sel = xy[mask]
        t_sel = t_q[mask]
        if channel_name == "u":
            comp = torch.zeros(mask.sum(), dtype=torch.long, device=xy.device)
            raw_pred = net.query_decoder(xy_sel, t_sel, comp, h_states, s_time, sensor_pos).squeeze(1)
        elif channel_name == "v":
            comp = torch.ones(mask.sum(), dtype=torch.long, device=xy.device)
            raw_pred = net.query_decoder(xy_sel, t_sel, comp, h_states, s_time, sensor_pos).squeeze(1)
        else:
            raise ValueError(f"不支援的觀測通道: {channel_name}")
        pred = (raw_pred - mean) / std
        preds[mask] = pred
    return preds


def physics_weight_at_step(
    step: int,
    final_weight: float,
    warmup_steps: int,
    ramp_steps: int,
) -> float:
    """What: 線性 physics warmup/ramp。"""
    if step < 1:
        raise ValueError(f"step 必須從 1 開始，收到 {step}")
    if final_weight < 0.0:
        raise ValueError(f"final_weight 不可為負，收到 {final_weight}")
    if warmup_steps < 0 or ramp_steps < 0:
        raise ValueError(
            f"warmup_steps / ramp_steps 不可為負，收到 {warmup_steps}, {ramp_steps}"
        )
    if final_weight == 0.0:
        return 0.0
    if step <= warmup_steps:
        return 0.0
    if ramp_steps == 0:
        return final_weight
    progress = min((step - warmup_steps) / ramp_steps, 1.0)
    return float(final_weight * progress)


DEFAULT_LNN_ARGS: dict[str, Any] = {
    "sensor_jsons": None,
    "sensor_npzs": None,
    "dns_paths": None,
    "re_values": None,
    "observed_sensor_channels": ["u", "v"],
    "fourier_harmonics": 8,
    "d_model": 64,
    "d_time": 8,
    "num_spatial_cfc_layers": 1,
    "num_temporal_cfc_layers": 1,
    "num_token_attention_layers": 1,
    "token_attention_heads": 4,
    "num_query_mlp_layers": 1,
    "query_mlp_hidden_dim": 64,
    "num_query_cfc_layers": 1,
    "query_gate_bias_span": 1.0,
    "output_head_gain": 1.0,
    "operator_rank": 64,
    "fusion_temperature_init": None,
    "data_loss_weight": 1.0,
    "physics_loss_weight": 0.01,
    "physics_loss_warmup_steps": 0,
    "physics_loss_ramp_steps": 0,
    "continuity_weight": 1.0,
    "time_marching": True,
    "time_marching_start": 0.5,
    "time_marching_warmup": 0.5,
    "domain_length": 1.0,
    "kolmogorov_k_f": 4.0,
    "kolmogorov_A": 0.1,
    "use_temporal_anchor": False,
    "resume_checkpoint": None,
    "T_total": 5.0,
    "temporal_anchor_harmonics": 2,
    "iterations": 1000,
    "num_query_points": 128,
    "num_physics_points": 32,
    "physics_collocation_strategy": "random",
    "physics_residual_normalize": False,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "lr_schedule": "none",
    "use_schedule_free": False,
    "lr_warmup_steps": 300,
    "soap_precondition_frequency": 10,
    "lr_decay_steps": 1000,
    "lr_decay_gamma": 0.9,
    "min_learning_rate": 1e-6,
    "max_grad_norm": 1.0,
    "checkpoint_period": 100,
    "seed": 42,
    "device": "mps",
    "artifacts_dir": "artifacts/deeponet-cfc-midlong-uvomega-small",
}

_REMOVED_KEYS = {
    "nhead",
    "dim_feedforward",
    "attn_dropout",
    "num_encoder_layers",
    "use_local_struct_features",
    "sensor_knn_k",
    "num_latent_tokens",
}


def load_lnn_config(config_path: Path | None) -> dict[str, Any]:
    """What: 載入並驗證核心 LNN config。"""
    if config_path is None:
        return {}
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config_data = payload.get("train", payload)
    normalized = dict(config_data)
    obsolete = sorted(set(normalized) & _REMOVED_KEYS)
    if obsolete:
        raise ValueError(
            f"Config 含有已移除的 PiT 欄位（請改用 num_spatial/temporal_cfc_layers）: {obsolete}"
        )
    unknown = sorted(set(normalized) - set(DEFAULT_LNN_ARGS))
    if unknown:
        raise ValueError(f"LNN config 含有不支援的欄位: {unknown}")
    base = Path.cwd()
    for list_key in ("sensor_jsons", "sensor_npzs", "dns_paths"):
        if list_key in normalized:
            normalized[list_key] = [str((base / Path(p)).resolve()) for p in normalized[list_key]]
    if "artifacts_dir" in normalized:
        normalized["artifacts_dir"] = str((base / Path(normalized["artifacts_dir"])).resolve())
    return normalized


def train_lnn_kolmogorov(args: dict[str, Any]) -> None:
    """What: 核心 Pi-LNN 訓練迴圈。"""
    from pi_onet.kolmogorov_dataset import KolmogorovDataset

    device = configure_torch_runtime(args["device"])
    torch.manual_seed(args["seed"])
    rng = np.random.default_rng(args["seed"])

    artifacts_dir = Path(args["artifacts_dir"])
    checkpoints_dir = artifacts_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        KolmogorovDataset(
            sensor_json=args["sensor_jsons"][i],
            sensor_npz=args["sensor_npzs"][i],
            dns_path=args["dns_paths"][i],
            re_value=float(args["re_values"][i]),
            observed_channel_names=tuple(args["observed_sensor_channels"]),
            train_ratio=0.8,
            seed=args["seed"],
        )
        for i in range(len(args["re_values"]))
    ]
    num_re = len(datasets)

    sensor_vals_list = [
        torch.tensor(ds.sensor_vals.transpose(1, 0, 2), dtype=torch.float32, device=device)
        for ds in datasets
    ]
    sensor_pos_list = [
        torch.tensor(ds.sensor_pos, dtype=torch.float32, device=device)
        for ds in datasets
    ]
    sensor_time_list = [
        torch.tensor(ds.sensor_time, dtype=torch.float32, device=device)
        for ds in datasets
    ]
    observed_mean_list = [
        torch.tensor(ds.observed_channel_mean, dtype=torch.float32, device=device)
        for ds in datasets
    ]
    observed_std_list = [
        torch.tensor(ds.observed_channel_std, dtype=torch.float32, device=device)
        for ds in datasets
    ]

    net = create_lnn_model(args).to(device)
    print("=== Configuration ===")
    print(f"trainable_parameters: {count_parameters(net)}")

    # --- Optimizer + Schedule-Free 組合 ---
    # lr_schedule 控制 base optimizer 種類與 LR 衰減策略：
    #   "soap"        → SOAP（二階前置條件），不搭配 LR scheduler
    #   "step"        → AdamW + StepLR
    #   "cosine"      → AdamW + CosineAnnealingLR
    #   "none"        → AdamW，常數 LR
    #   "schedulefree"→ 舊版相容：等同 lr_schedule="none" + use_schedule_free=true
    #
    # use_schedule_free 控制是否套用 Polyak averaging：
    #   AdamW + SF  → 使用 fused AdamWScheduleFree（效率最佳，支援 warmup）
    #   SOAP  + SF  → 使用 ScheduleFreeWrapper(SOAP)
    #   任何  + no SF → 直接使用 base optimizer
    #
    # Why fused vs wrapper: ScheduleFreeWrapper 在 step() 前先寫入 state['z']，
    # 導致 AdamW._init_group 誤判狀態已存在而跳過 exp_avg 初始化 → KeyError。
    # fused AdamWScheduleFree 無此問題。

    use_schedulefree = bool(args.get("use_schedule_free", False)) or args["lr_schedule"] == "schedulefree"

    if args["lr_schedule"] == "soap":
        import sys
        _soap_dir = str(Path(__file__).parent.parent.parent / "SOAP")
        if _soap_dir not in sys.path:
            sys.path.insert(0, _soap_dir)
        from soap import SOAP as SOAPOptimizer
        base_optimizer = SOAPOptimizer(
            net.parameters(),
            lr=args["learning_rate"],
            betas=(0.95, 0.95),
            weight_decay=args["weight_decay"],
            precondition_frequency=int(args.get("soap_precondition_frequency", 10)),
        )
        scheduler = None
        if use_schedulefree:
            import schedulefree
            optimizer = schedulefree.ScheduleFreeWrapper(base_optimizer, momentum=0.9)
        else:
            optimizer = base_optimizer

    elif use_schedulefree:
        # AdamW + Schedule-Free：使用 fused 實作，支援 warmup，無 wrapper 相容性問題。
        import schedulefree
        optimizer = schedulefree.AdamWScheduleFree(
            net.parameters(),
            lr=args["learning_rate"],
            warmup_steps=int(args.get("lr_warmup_steps", 300)),
            weight_decay=args["weight_decay"],
        )
        scheduler = None

    else:
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=args["learning_rate"],
            weight_decay=args["weight_decay"],
        )
        if args["lr_schedule"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args["iterations"],
                eta_min=args["min_learning_rate"],
            )
        elif args["lr_schedule"] == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(args["lr_decay_steps"]),
                gamma=float(args["lr_decay_gamma"]),
            )
        else:
            scheduler = None

    # Resume：從 checkpoint 恢復完整訓練狀態
    start_step = 0
    resume_path = args.get("resume_checkpoint")

    def _fix_ckpt_compat(state_dict: dict) -> dict:
        """相容舊 checkpoint：log_fusion_temperature 由 0-dim 改為 shape (1,)。"""
        key = "query_decoder.log_fusion_temperature"
        if key in state_dict and state_dict[key].dim() == 0:
            state_dict[key] = state_dict[key].unsqueeze(0)
        return state_dict

    def _fix_optimizer_state_compat(opt: torch.optim.Optimizer) -> None:
        """相容舊 checkpoint：將 optimizer state 中殘留的 0-dim tensor unsqueeze 為 (1,)。
        Why: log_fusion_temperature 由 0-dim 改為 (1,) 後，SOAP 的 exp_avg/exp_avg_sq
             仍是舊形狀，導致 broadcast 失敗。"""
        for param_state in opt.state.values():
            for k, v in param_state.items():
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    param_state[k] = v.unsqueeze(0)

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            # 完整狀態格式：model + optimizer + scheduler + step
            net.load_state_dict(_fix_ckpt_compat(ckpt["model_state_dict"]))
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            _fix_optimizer_state_compat(optimizer)
            if not use_schedulefree and scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_step = int(ckpt["step"])
        else:
            # 舊格式（只有 model weights）：恢復模型；scheduler 快進（schedulefree 則略過）
            net.load_state_dict(_fix_ckpt_compat(ckpt))
            start_step = int(Path(resume_path).stem.split("_step_")[-1]) if "_step_" in Path(resume_path).stem else 0
            if not use_schedulefree and scheduler is not None:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for _ in range(start_step):
                        scheduler.step()
        print(f"  resumed from: {resume_path} (step {start_step})")

    k_f = float(args["kolmogorov_k_f"])
    A = float(args["kolmogorov_A"])
    domain_length = float(args["domain_length"])
    base_phys_weight = float(args["physics_loss_weight"])
    phys_warmup_steps = int(args["physics_loss_warmup_steps"])
    phys_ramp_steps = int(args["physics_loss_ramp_steps"])
    use_tm = bool(args["time_marching"])
    tm_t_start = float(args["time_marching_start"])
    tm_t_end = float(datasets[0].sensor_time[-1])
    tm_warmup = int(args["time_marching_warmup"] * args["iterations"])

    print("=== Training ===")
    if use_tm:
        print(f"  time_marching: t [{tm_t_start:.1f} → {tm_t_end:.1f}]  warmup={tm_warmup} steps")
    if phys_warmup_steps > 0 or phys_ramp_steps > 0:
        print(
            "  physics_ramp:"
            f" warmup={phys_warmup_steps} steps,"
            f" ramp={phys_ramp_steps} steps,"
            f" final_weight={base_phys_weight:.4f}"
        )
    print(f"{'Step':<8} {'L_data':>12} {'L_phys':>12} {'w_phys':>10} {'L_total':>12}"
          + ("  t_max" if use_tm else ""))

    for step in range(start_step + 1, args["iterations"] + 1):
        if use_tm:
            progress = min(step / max(tm_warmup, 1), 1.0)
            t_max: float | None = tm_t_start + (tm_t_end - tm_t_start) * progress
        else:
            t_max = None

        if use_schedulefree:
            optimizer.train()
        net.train()
        optimizer.zero_grad()

        l_data = torch.zeros(1, device=device)
        for i, ds in enumerate(datasets):
            xy_np, t_np, c_np, ref_np = ds.sample_sensor_batch(
                rng, n=args["num_query_points"], t_max=t_max
            )
            h_states, s_time = net.encode(
                sensor_vals_list[i], sensor_pos_list[i], ds.re_norm, sensor_time_list[i]
            )

            xy = torch.tensor(xy_np, dtype=torch.float32, device=device)
            t_q = torch.tensor(t_np, device=device)
            c = torch.tensor(c_np, dtype=torch.long, device=device)
            ref = torch.tensor(ref_np, device=device)
            pred = observed_channel_prediction(
                net=net,
                xy=xy,
                t_q=t_q,
                c_obs=c,
                observed_channel_names=ds.observed_channel_names,
                observed_channel_mean=observed_mean_list[i],
                observed_channel_std=observed_std_list[i],
                h_states=h_states,
                s_time=s_time,
                sensor_pos=sensor_pos_list[i],
            )
            l_data = l_data + torch.mean((pred - ref) ** 2)
        l_data = l_data / num_re

        phys_weight = physics_weight_at_step(
            step=step,
            final_weight=base_phys_weight,
            warmup_steps=phys_warmup_steps,
            ramp_steps=phys_ramp_steps,
        )
        if phys_weight > 0.0 and int(args["num_physics_points"]) > 0:
            net.eval()
            l_ns_total = torch.zeros(1, device=device)
            l_cont_total = torch.zeros(1, device=device)
            phys_strategy = str(args.get("physics_collocation_strategy", "random"))
            phys_normalize = bool(args.get("physics_residual_normalize", False))
            for i, ds in enumerate(datasets):
                xy_np, t_np = ds.sample_physics_points(
                    rng, n=args["num_physics_points"], t_max=t_max, strategy=phys_strategy
                )
                xyt = torch.tensor(
                    np.concatenate([xy_np, t_np[:, None]], axis=1),
                    device=device,
                    requires_grad=True,
                )
                model_fn = make_lnn_model_fn(
                    net,
                    sensor_vals_list[i],
                    sensor_pos_list[i],
                    re_norm=ds.re_norm,
                    sensor_time=sensor_time_list[i],
                    device=device,
                )
                u_fn = lambda xyt_, fn=model_fn: fn(xyt_, c=0)
                v_fn = lambda xyt_, fn=model_fn: fn(xyt_, c=1)
                p_fn = lambda xyt_, fn=model_fn: fn(xyt_, c=2)
                mom_u, mom_v, cont = unsteady_ns_residuals(
                    u_fn, v_fn, p_fn, xyt, re=ds.re_value, k_f=k_f, A=A, domain_length=domain_length
                )
                if phys_normalize:
                    # 將每個殘差除以自身批次 RMS（detach 不參與梯度），使各項貢獻量級對齊。
                    # Why: Re=10000 的 momentum 殘差 O(10)，continuity 可能 O(0.01)；
                    #      不正規化時 continuity 梯度幾乎為零，無散度約束形同虛設。
                    def _norm_r(r: torch.Tensor) -> torch.Tensor:
                        return r / r.detach().std().clamp(min=1e-8)
                    mom_u = _norm_r(mom_u)
                    mom_v = _norm_r(mom_v)
                    cont  = _norm_r(cont)
                l_ns_total = l_ns_total + 0.5 * (torch.mean(mom_u ** 2) + torch.mean(mom_v ** 2))
                l_cont_total = l_cont_total + torch.mean(cont ** 2)
            net.train()
            l_ns_total = l_ns_total / num_re
            l_cont_total = l_cont_total / num_re
            l_physics = l_ns_total + args["continuity_weight"] * l_cont_total
        else:
            l_physics = torch.zeros(1, device=device)

        l_total = args["data_loss_weight"] * l_data + phys_weight * l_physics
        l_total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), float(args["max_grad_norm"]))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % max(1, args["iterations"] // 10) == 0 or step == 1:
            tm_str = f"  t≤{t_max:5.1f}" if use_tm and t_max is not None else ""
            print(
                f"{step:<8} {l_data.item():>12.4e}"
                f" {l_physics.item():>12.4e} {phys_weight:>10.4f}"
                f" {l_total.item():>12.4e}{tm_str}"
            )

        if args["checkpoint_period"] > 0 and step % args["checkpoint_period"] == 0:
            if use_schedulefree:
                optimizer.eval()  # 切換到 Polyak 平均權重後再儲存
            torch.save(
                {
                    "step": step,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                },
                str(checkpoints_dir / f"lnn_kolmogorov_step_{step}.pt"),
            )
            if use_schedulefree:
                optimizer.train()  # 恢復訓練模式

    if use_schedulefree:
        optimizer.eval()  # final.pt 儲存 Polyak 平均推理權重
    final = artifacts_dir / "lnn_kolmogorov_final.pt"
    torch.save(net.state_dict(), str(final))
    write_json(artifacts_dir / "experiment_manifest.json", {
        "configuration": {k: v for k, v in args.items() if k not in ("sensor_jsons", "sensor_npzs", "dns_paths")},
        "final_checkpoint": str(final),
    })
    print("=== Done ===")


def main() -> None:
    """What: CLI entry point for core LNN training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train core Pi-LNN on Kolmogorov flow."
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
