# src/pi_onet/lnn_kolmogorov.py
"""Pi-LNN: core DeepONet + CfC model and training loop for Kolmogorov flow.

What: 以 CfC 作為 branch-side temporal encoder，並以 DeepONet trunk 解碼 query。
Why:  CFC 保留連續時間時序偏置；DeepONet 提供更清晰的 operator factorization，
      方便後續 branch/trunk 解耦與架構擴充。
"""
from __future__ import annotations

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

from pi_onet.pit_ldc import (
    _grad,
    configure_torch_runtime,
    count_parameters,
    rff_encode,
    write_json,
)


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

    Why: 先保留每個 sensor 的 RFF(x,y)+u,v,p 局部訊息，不在 spatial branch
         提前做混合；同時加入局部幾何與差分特徵，讓 token 除了點值外也攜帶
         局部梯度趨勢，讓 decoder 負責主要的 cross-attention 與讀取。
    """

    def __init__(
        self,
        rff_features: int,
        d_model: int,
        num_layers: int,
        num_latent_tokens: int,
        use_local_struct_features: bool = False,
        sensor_knn_k: int = 4,
    ) -> None:
        super().__init__()
        if sensor_knn_k < 1:
            raise ValueError(f"sensor_knn_k 必須 >= 1，收到 {sensor_knn_k}")
        self.use_local_struct_features = use_local_struct_features
        self.sensor_knn_k = sensor_knn_k
        base_in = 2 * rff_features + 3
        hidden = 2 * d_model
        depth = max(num_layers, 1)
        self.base_norm = nn.LayerNorm(base_in)
        self.neighbor_embed = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.geom_proj = nn.Sequential(
            nn.LayerNorm(7),
            nn.Linear(7, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        sensor_in = base_in + (3 * d_model if use_local_struct_features else 0)
        self.token_in = nn.Sequential(
            nn.LayerNorm(sensor_in),
            nn.Linear(sensor_in, hidden),
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

    def _local_features(
        self,
        sensor_vals: torch.Tensor,
        sensor_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """What: 建立 kNN 局部差分與近似梯度特徵。

        Why: 讓 token 在進入 temporal model 前就具備局部形變趨勢，而不只是單點數值。
        """
        num_tokens = sensor_pos.shape[0]
        k = min(self.sensor_knn_k, max(num_tokens - 1, 1))
        dmat = torch.cdist(sensor_pos, sensor_pos, p=2)
        knn_idx = torch.topk(dmat, k=k + 1, largest=False).indices[:, 1:]

        nbr_pos = sensor_pos[knn_idx]
        nbr_vals = sensor_vals[knn_idx]
        pos_ctr = sensor_pos.unsqueeze(1)
        val_ctr = sensor_vals.unsqueeze(1)

        rel_pos = (nbr_pos - pos_ctr) / (2.0 * math.pi)
        rel_dist = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
        rel_vals = nbr_vals - val_ctr

        nbr_raw = torch.cat([rel_pos, rel_dist, rel_vals], dim=-1)
        nbr_emb = self.neighbor_embed(nbr_raw)
        nbr_mean = nbr_emb.mean(dim=1)
        nbr_max = nbr_emb.max(dim=1).values

        a = rel_pos
        eye = torch.eye(2, device=sensor_pos.device, dtype=sensor_pos.dtype).unsqueeze(0)
        ata = torch.matmul(a.transpose(1, 2), a) + 1.0e-4 * eye
        grads = []
        for comp in range(3):
            b = rel_vals[..., comp:comp + 1]
            atb = torch.matmul(a.transpose(1, 2), b)
            grad = torch.linalg.solve(ata, atb).squeeze(-1)
            grads.append(grad)
        grad_u, grad_v, grad_p = grads
        vort = (grad_v[:, 0] - grad_u[:, 1]).unsqueeze(-1)
        geom = torch.cat([grad_u, grad_v, grad_p, vort], dim=-1)
        geom_emb = self.geom_proj(geom)
        return nbr_mean, nbr_max, geom_emb

    def forward(
        self,
        sensor_vals: torch.Tensor,
        sensor_pos: torch.Tensor,
        B: torch.Tensor,
    ) -> torch.Tensor:
        rff = rff_encode(sensor_pos, B)
        base = torch.cat([rff, sensor_vals], dim=-1)
        pieces = [self.base_norm(base)]
        if self.use_local_struct_features:
            nbr_mean, nbr_max, geom_emb = self._local_features(sensor_vals, sensor_pos)
            pieces.extend([nbr_mean, nbr_max, geom_emb])
        tokens = self.token_in(torch.cat(pieces, dim=-1))
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
        seq = spatial_states
        for block in self.token_blocks:
            seq = block(seq)
        re_bias = self._re_bias(re_norm, spatial_states.device, spatial_states.dtype).view(1, 1, -1)
        seq = seq + re_bias
        for cell in self.cells:
            h = torch.zeros(seq.shape[1], self.d_model, device=seq.device, dtype=seq.dtype)
            outputs = []
            for t in range(seq.shape[0]):
                h = cell(seq[t], h, dt=dts[t])
                outputs.append(h)
            seq = torch.stack(outputs)
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
        inp = spatial_state.unsqueeze(0)
        for block in self.token_blocks:
            inp = block(inp)
        inp = inp.squeeze(0) + self._re_bias(
            re_norm,
            spatial_state.device,
            spatial_state.dtype,
        ).view(1, -1)
        new_h_list: list[torch.Tensor] = []
        for cell, h in zip(self.cells, h_list):
            new_h = cell(inp, h, dt=dt)
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
        rff_features: int,
        d_model: int,
        d_time: int,
        num_query_mlp_layers: int = 0,
        query_mlp_hidden_dim: int = 256,
        output_head_gain: float = 1.0,
        operator_rank: int | None = None,
        fusion_temperature_init: float | None = None,
    ) -> None:
        super().__init__()
        query_in = 2 * rff_features + d_time + 8
        rank = d_model if operator_rank is None else operator_rank
        if rank <= 0:
            raise ValueError(f"operator_rank 必須 > 0，收到 {rank}")
        self.rank = rank
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
        self.branch_token_proj = nn.Linear(d_model, query_mlp_hidden_dim)
        self.branch_norm = nn.LayerNorm(query_mlp_hidden_dim)
        self.branch_attn = nn.MultiheadAttention(
            query_mlp_hidden_dim,
            num_heads=4,
            batch_first=True,
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
        self.log_fusion_temperature = nn.Parameter(torch.tensor(math.log(temp_init), dtype=torch.float32))
        self.component_scale = nn.Parameter(torch.ones(3))
        self.component_bias = nn.Parameter(torch.zeros(3))

    def forward(
        self,
        xy: torch.Tensor,
        t_q: torch.Tensor,
        c: torch.Tensor,
        h_states: torch.Tensor,
        sensor_time: torch.Tensor,
        B: torch.Tensor,
    ) -> torch.Tensor:
        idx = torch.searchsorted(sensor_time.contiguous(), t_q.contiguous(), right=True) - 1
        idx = idx.clamp(0, h_states.shape[0] - 1)
        h_branch_tokens = h_states[idx]
        dt_to_query = (t_q - sensor_time[idx]).clamp(min=0.0)

        rff_q = rff_encode(xy, B)
        time_e = self.time_proj(dt_to_query.unsqueeze(-1))
        emb_c = self.component_emb(c)
        trunk_feat = F.silu(self.trunk_in(torch.cat([rff_q, time_e, emb_c], dim=-1)))
        for block in self.trunk_blocks:
            trunk_feat = block(trunk_feat)

        trunk_basis = self.trunk_out(trunk_feat).view(-1, 3, self.rank)
        branch_tokens = self.branch_token_proj(h_branch_tokens)
        branch_query = self.branch_norm(trunk_feat).unsqueeze(1)
        branch_ctx, _ = self.branch_attn(
            branch_query,
            branch_tokens,
            branch_tokens,
            need_weights=False,
        )
        branch_ctx = branch_ctx.squeeze(1)
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
        rff_features: int,
        rff_sigma: float,
        d_model: int,
        d_time: int,
        num_spatial_cfc_layers: int,
        num_temporal_cfc_layers: int,
        use_local_struct_features: bool = False,
        sensor_knn_k: int = 4,
        num_token_attention_layers: int = 1,
        token_attention_heads: int = 4,
        num_query_mlp_layers: int = 0,
        query_mlp_hidden_dim: int = 256,
        num_query_cfc_layers: int = 1,
        query_gate_bias_span: float = 1.0,
        output_head_gain: float = 1.0,
        operator_rank: int | None = None,
        fusion_temperature_init: float | None = None,
        num_latent_tokens: int = 8,
        rff_sigma_bands: list[tuple[int, float]] | None = None,
    ) -> None:
        super().__init__()
        if rff_sigma_bands is not None:
            total = sum(n for n, _ in rff_sigma_bands)
            if total != rff_features:
                raise ValueError(
                    f"rff_sigma_bands 的 n_freqs 總和 ({total}) != rff_features ({rff_features})"
                )
            B = torch.cat([torch.randn(2, n) * sigma for n, sigma in rff_sigma_bands], dim=1)
        else:
            B = torch.randn(2, rff_features) * rff_sigma
        self.register_buffer("B", B)
        self.spatial_encoder = SpatialSetEncoder(
            rff_features,
            d_model,
            num_spatial_cfc_layers,
            num_latent_tokens=num_latent_tokens,
            use_local_struct_features=use_local_struct_features,
            sensor_knn_k=sensor_knn_k,
        )
        self.temporal_encoder = TemporalCfCEncoder(
            d_model,
            num_temporal_cfc_layers,
            num_token_attention_layers=num_token_attention_layers,
            token_attention_heads=token_attention_heads,
        )
        self.query_decoder = DeepONetCfCDecoder(
            rff_features=rff_features,
            d_model=d_model,
            d_time=d_time,
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
            self.spatial_encoder(sensor_vals[t], sensor_pos, self.B)
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
        spatial = self.spatial_encoder(sensor_vals_t, sensor_pos, self.B)
        return self.temporal_encoder.step(spatial, h_list, re_norm, dt)

    def predict(
        self,
        xy: torch.Tensor,
        t_q: torch.Tensor,
        c: torch.Tensor,
        h_out: torch.Tensor,
        t_last: float,
    ) -> torch.Tensor:
        h_states = h_out.unsqueeze(0)
        s_time = torch.tensor([t_last], device=h_out.device, dtype=h_out.dtype)
        return self.query_decoder(xy, t_q, c, h_states, s_time, self.B)

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
        return self.query_decoder(xy, t_q, c, h_states, s_time, self.B)


def create_lnn_model(cfg: dict[str, Any]) -> LiquidOperator:
    """What: 從 config 建立核心 LiquidOperator。"""
    bands_raw = cfg.get("rff_sigma_bands")
    rff_sigma_bands = [(int(n), float(s)) for n, s in bands_raw] if bands_raw else None
    return LiquidOperator(
        rff_features=int(cfg["rff_features"]),
        rff_sigma=float(cfg.get("rff_sigma", 32.0)),
        d_model=int(cfg["d_model"]),
        d_time=int(cfg["d_time"]),
        num_spatial_cfc_layers=int(cfg["num_spatial_cfc_layers"]),
        num_temporal_cfc_layers=int(cfg["num_temporal_cfc_layers"]),
        use_local_struct_features=bool(cfg.get("use_local_struct_features", False)),
        sensor_knn_k=int(cfg.get("sensor_knn_k", 4)),
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
        num_latent_tokens=int(cfg.get("num_latent_tokens", 8)),
        rff_sigma_bands=rff_sigma_bands,
    )


def unsteady_ns_residuals(
    u_fn: Callable,
    v_fn: Callable,
    p_fn: Callable,
    xyt: torch.Tensor,
    re: float,
    k_f: float = 4.0,
    A: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """What: 2D incompressible unsteady Navier-Stokes residuals."""
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
    f_x = A * torch.sin(k_f * xyt[:, 1:2])
    ns_x = du_dt + u * du_dx + v * du_dy + dp_dx - nu * (du_dx2 + du_dy2) - f_x
    ns_y = dv_dt + u * dv_dx + v * dv_dy + dp_dy - nu * (dv_dx2 + dv_dy2)
    cont = du_dx + dv_dy
    return ns_x, ns_y, cont


def make_lnn_model_fn(
    net: LiquidOperator,
    sensor_vals: torch.Tensor,
    sensor_pos: torch.Tensor,
    re_norm: float,
    sensor_time: torch.Tensor,
    device: torch.device,
) -> Callable:
    """What: 建立物理 loss 所需的 closure。"""
    net_device = next(iter(net.buffers())).device
    h_states, s_time = net.encode(sensor_vals, sensor_pos, re_norm, sensor_time)

    def model_fn(xyt: torch.Tensor, c: int) -> torch.Tensor:
        xyt_d = xyt.to(net_device)
        xy_d = xyt_d[:, :2]
        t_q_d = xyt_d[:, 2]
        c_t = torch.full((xyt_d.shape[0],), c, dtype=torch.long, device=net_device)
        return net.query_decoder(xy_d, t_q_d, c_t, h_states, s_time, net.B).to(xyt.device)

    return model_fn


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
    "rff_features": 64,
    "rff_sigma": 32.0,
    "rff_sigma_bands": None,
    "d_model": 128,
    "d_time": 16,
    "num_spatial_cfc_layers": 2,
    "num_temporal_cfc_layers": 2,
    "use_local_struct_features": False,
    "sensor_knn_k": 4,
    "num_token_attention_layers": 1,
    "token_attention_heads": 4,
    "num_query_mlp_layers": 0,
    "query_mlp_hidden_dim": 256,
    "num_query_cfc_layers": 1,
    "query_gate_bias_span": 1.0,
    "output_head_gain": 1.0,
    "operator_rank": None,
    "fusion_temperature_init": None,
    "num_latent_tokens": 8,
    "data_loss_weight": 1.0,
    "physics_loss_weight": 0.05,
    "physics_loss_warmup_steps": 0,
    "physics_loss_ramp_steps": 0,
    "continuity_weight": 1.0,
    "time_marching": False,
    "time_marching_start": 2.0,
    "time_marching_warmup": 0.5,
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

    net = create_lnn_model(args).to(device)
    print("=== Configuration ===")
    print(f"trainable_parameters: {count_parameters(net)}")

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=args["learning_rate"],
        weight_decay=args["weight_decay"],
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args["iterations"],
            eta_min=args["min_learning_rate"],
        )
        if args["lr_schedule"] == "cosine"
        else None
    )

    k_f = float(args["kolmogorov_k_f"])
    A = float(args["kolmogorov_A"])
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

    for step in range(1, args["iterations"] + 1):
        if use_tm:
            progress = min(step / max(tm_warmup, 1), 1.0)
            t_max: float | None = tm_t_start + (tm_t_end - tm_t_start) * progress
        else:
            t_max = None

        net.train()
        optimizer.zero_grad()

        l_data = torch.zeros(1, device=device)
        for i, ds in enumerate(datasets):
            h_states, s_time = net.encode(
                sensor_vals_list[i], sensor_pos_list[i], ds.re_norm, sensor_time_list[i]
            )
            xy_np, t_np, c_np, ref_np = ds.sample_train_batch(
                rng, n=args["num_query_points"], t_max=t_max
            )
            xy = torch.tensor(xy_np, device=device)
            t_q = torch.tensor(t_np, device=device)
            c = torch.tensor(c_np, dtype=torch.long, device=device)
            ref = torch.tensor(ref_np, device=device)
            pred = net.query_decoder(xy, t_q, c, h_states, s_time, net.B).squeeze(1)
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
            for i, ds in enumerate(datasets):
                xy_np, t_np = ds.sample_physics_points(rng, n=args["num_physics_points"], t_max=t_max)
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
                ns_x, ns_y, cont = unsteady_ns_residuals(
                    u_fn, v_fn, p_fn, xyt, re=ds.re_value, k_f=k_f, A=A
                )
                l_ns_total = l_ns_total + torch.mean(ns_x ** 2) + torch.mean(ns_y ** 2)
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
            torch.save(net.state_dict(), str(checkpoints_dir / f"lnn_kolmogorov_step_{step}.pt"))

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
