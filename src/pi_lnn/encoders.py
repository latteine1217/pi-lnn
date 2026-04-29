"""Sensor-side encoders: spatial set encoder + temporal CfC encoder."""
from __future__ import annotations

import torch
import torch.nn as nn

from pi_lnn.blocks import CfCCell, ResidualMLPBlock, TokenSelfAttentionBlock
from pi_lnn.encodings import FourierEmbs, LearnableFourierEmb, periodic_fourier_encode


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
        fourier_embed_dim: int = 0,
        use_periodic_domain: bool = True,
    ) -> None:
        super().__init__()
        self.domain_length = float(domain_length)
        self.fourier_harmonics = int(fourier_harmonics)
        self.sensor_value_dim = int(sensor_value_dim)
        self.use_periodic_domain = bool(use_periodic_domain)
        if fourier_embed_dim > 0:
            # 週期：LearnableFourierEmb（PeriodEmbs + 投影），x=0≡x=L 編碼。
            # 非週期：FourierEmbs 真 RFF，無預先週期化，能區分域邊界。
            if self.use_periodic_domain:
                self.spatial_emb: nn.Module | None = LearnableFourierEmb(fourier_embed_dim)
            else:
                self.spatial_emb = FourierEmbs(fourier_embed_dim, input_dim=2)
            spatial_dim = fourier_embed_dim
        else:
            if not self.use_periodic_domain:
                raise ValueError(
                    "use_periodic_domain=False 需 fourier_embed_dim>0；"
                    "harmonics-only fallback 為週期編碼，與非週期域語義衝突。"
                )
            self.spatial_emb = None
            spatial_dim = 4 * fourier_harmonics
        base_in = spatial_dim + self.sensor_value_dim
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

    def encode_pos(self, sensor_pos: torch.Tensor) -> torch.Tensor:
        """What: 計算 sensor 位置的空間編碼。呼叫方負責在 loop 外預計算並重用。"""
        if self.spatial_emb is not None:
            return self.spatial_emb(sensor_pos, self.domain_length)
        return periodic_fourier_encode(sensor_pos, self.domain_length, self.fourier_harmonics)

    def forward(
        self,
        sensor_vals: torch.Tensor,
        pos_enc: torch.Tensor,
    ) -> torch.Tensor:
        """支援 [K, C]（單一時刻、streaming）或 [T, K, C]（向量化整段時序）。

        Why: 整個 module 對 last-dim element-wise，T 軸只是 batch，
             向量化等價於原本 Python loop over T，但消除 T×=200× kernel-launch overhead。
        """
        if sensor_vals.dim() == 3 and pos_enc.dim() == 2:
            T = sensor_vals.shape[0]
            pos_enc = pos_enc.unsqueeze(0).expand(T, -1, -1)
        base = torch.cat([pos_enc, sensor_vals], dim=-1)
        tokens = self.token_in(self.base_norm(base))
        for block in self.blocks:
            tokens = block(tokens)
        return self.out_proj(tokens)


class TemporalCfCEncoder(nn.Module):
    """What: 以 CfC 演化 sensor token 序列，產生 token states。

    Why: 保留每個 sensor token 的連續時間動態，讓 decoder 能直接讀取感測器級上下文。
         use_bidirectional=True 時加入反向掃描，使 t=0 的 hidden state 亦能看到未來觀測，
         適用於離線批量重建（所有感測器資料預先備妥）。
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_token_attention_layers: int = 1,
        token_attention_heads: int = 4,
        use_bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_bidirectional = use_bidirectional
        self.re_proj = nn.Linear(1, d_model)
        self.token_blocks = nn.ModuleList([
            TokenSelfAttentionBlock(d_model=d_model, num_heads=token_attention_heads)
            for _ in range(max(num_token_attention_layers, 0))
        ])
        self.cells = nn.ModuleList([CfCCell(d_model, d_model) for _ in range(num_layers)])
        if use_bidirectional:
            # 反向 CfC：獨立參數，從 t=T-1 掃回 t=0。
            # Why: 讓 h_states[0] 也能看到未來觀測，消除因果編碼在 t=0 的資訊不對稱。
            self.backward_cells = nn.ModuleList([CfCCell(d_model, d_model) for _ in range(num_layers)])

    def _re_bias(self, re_norm: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        re_t = torch.tensor([[re_norm]], dtype=dtype, device=device)
        return self.re_proj(re_t).squeeze(0)

    def _run_cfc_pass(
        self,
        seq: torch.Tensor,
        cells: nn.ModuleList,
        dts: torch.Tensor,
        layer_idx: int,
        reverse: bool,
    ) -> torch.Tensor:
        """What: 單方向 CfC 掃描；seq 已在 forward() 內預先做完 token attention 與 re_bias 加總。"""
        T = seq.shape[0]
        h = torch.zeros(seq.shape[1], self.d_model, device=seq.device, dtype=seq.dtype)
        outputs: list[torch.Tensor] = [torch.empty(0)] * T
        time_range = reversed(range(T)) if reverse else range(T)
        for t in time_range:
            x_t = seq[t]
            h = cells[layer_idx](x_t, h, dt=dts[t])
            outputs[t] = h
        return torch.stack(outputs)

    def forward(
        self,
        spatial_states: torch.Tensor,
        re_norm: float,
        sensor_time: torch.Tensor,
    ) -> torch.Tensor:
        """What: 多層 CfC scan，每層先做 token attention（向量化 over T）再 scan。

        Why: 原版 attention 在 per-timestep 內以 batch=1 呼叫，T*L 次 kernel launch；
             移到 scan 之外對 [T, K, d] 一次 forward，T 軸當 batch 對 attention 完全等價。
             re_bias 也由 per-t addition 改成預先加總到 attended sequence 上，
             與原版「attention → +re_bias → CfC」順序保持一致。
        """
        dts = torch.cat([sensor_time[:1], sensor_time[1:] - sensor_time[:-1]])
        re_bias = self._re_bias(re_norm, spatial_states.device, spatial_states.dtype).view(1, 1, -1)
        seq = spatial_states
        for layer_idx in range(len(self.cells)):
            if self.token_blocks:
                block_idx = min(len(self.token_blocks) - 1, layer_idx)
                attended = self.token_blocks[block_idx](seq)        # [T, K, d]，T 為 batch
            else:
                attended = seq
            attended_with_bias = attended + re_bias                 # 廣播到 [T, K, d]
            fwd = self._run_cfc_pass(attended_with_bias, self.cells, dts, layer_idx, reverse=False)
            if self.use_bidirectional:
                bwd = self._run_cfc_pass(attended_with_bias, self.backward_cells, dts, layer_idx, reverse=True)
                new_seq = fwd + bwd
            else:
                new_seq = fwd
            # 層間殘差：第二層起加上前一層輸出，防止多層 CfC 的信號退化。
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
