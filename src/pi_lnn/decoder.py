"""DeepONet-style decoder with CfC trunk and cross-attention branch."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pi_lnn.blocks import ResidualMLPBlock
from pi_lnn.encodings import (
    LearnableFourierEmb,
    periodic_fourier_encode,
    temporal_phase_anchor,
)


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
        use_locality_decay: bool = False,
        fourier_embed_dim: int = 0,
        use_periodic_domain: bool = True,
    ) -> None:
        super().__init__()
        self.use_periodic_domain = bool(use_periodic_domain)
        self.use_locality_decay = bool(use_locality_decay)
        self.fourier_harmonics = int(fourier_harmonics)
        self.use_temporal_anchor = bool(use_temporal_anchor)
        self.T_total = float(T_total)
        self.temporal_anchor_harmonics = int(temporal_anchor_harmonics)
        temporal_dim = 2 * self.temporal_anchor_harmonics if self.use_temporal_anchor else 0
        if fourier_embed_dim > 0:
            self.spatial_emb: nn.Module | None = LearnableFourierEmb(fourier_embed_dim)
            spatial_dim = fourier_embed_dim
        else:
            self.spatial_emb = None
            spatial_dim = 4 * fourier_harmonics
        query_in = spatial_dim + temporal_dim + d_time + 8
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
        if self.use_locality_decay:
            # log_locality_decay: α = exp(log_locality_decay) 為衰減率（α > 0）。
            # Why: 在 softmax 前加入 log-space 距離懲罰 score += -α * r，
            #      等效於對每個 sensor token 的 attention weight 乘以 exp(-α * r)。
            #      初始 log_locality_decay = -2.0 → α ≈ 0.135，r=0.1 時懲罰 ≈ -0.013，
            #      接近中性，讓模型自行學習是否需要近鄰優先。
            self.log_locality_decay = nn.Parameter(torch.tensor([-2.0], dtype=torch.float32))

    def forward_uvp(
        self,
        xy: torch.Tensor,
        t_q: torch.Tensor,
        h_states: torch.Tensor,
        sensor_time: torch.Tensor,
        sensor_pos: torch.Tensor,
    ) -> torch.Tensor:
        """What: 對 N 個 query 一次回傳 [N, 3]（u, v, p），與對 c=0/1/2 個別呼叫
                self.forward(...) 三次的結果在數值上等價。

        Why: physics path 同時需要 u/v/p。原本三個獨立 forward 會重複算 c-independent
             項：pos_enc / time_e / branch_token_proj / branch_key_proj / branch_value_proj
             / rel_bias 等。此處 c-independent 段只算一次；c-conditional 段（trunk MLP +
             branch query/attention/branch_proj）以 [3N, ...] batch 跑（總 flop 不變，
             但減少 Python overhead 與 3 份獨立 autograd graph，二階 backward 記憶體下降）。
        """
        N = xy.shape[0]
        device = xy.device

        # ── c-independent ─────────────────────────────────────────────────
        idx = torch.searchsorted(sensor_time.contiguous(), t_q.contiguous(), right=True) - 1
        idx = idx.clamp(0, h_states.shape[0] - 1)
        h_branch_tokens = h_states[idx]
        dt_to_query = (t_q - sensor_time[idx]).clamp(min=0.0)

        if self.spatial_emb is not None:
            pos_enc = self.spatial_emb(xy, self.domain_length)
        else:
            pos_enc = periodic_fourier_encode(xy, self.domain_length, self.fourier_harmonics)
        time_e = self.time_proj(dt_to_query.unsqueeze(-1))

        branch_tokens = self.branch_token_proj(h_branch_tokens)
        k_proj = self.branch_key_proj(branch_tokens)
        v_proj = self.branch_value_proj(branch_tokens)

        rel = xy.unsqueeze(1) - sensor_pos.unsqueeze(0)
        if self.use_periodic_domain:
            rel = rel - torch.round(rel / self.domain_length) * self.domain_length
        rel_r = torch.sqrt((rel ** 2).sum(dim=-1, keepdim=True) + 1e-8)
        rel_bias = self.relpos_bias(rel_r).squeeze(-1)

        # ── c-conditional：批次化 c=0,1,2，flatten 成 [3N, ...] ──────────
        base_inputs = [pos_enc]
        if self.use_temporal_anchor:
            base_inputs.append(temporal_phase_anchor(
                t_q.unsqueeze(-1), self.T_total, self.temporal_anchor_harmonics
            ))
        base_inputs.append(time_e)
        base_feat = torch.cat(base_inputs, dim=-1)                              # [N, query_in - 8]

        c_all = torch.arange(3, device=device, dtype=torch.long)
        emb_c_all = self.component_emb(c_all)                                   # [3, 8]
        base_feat_3 = base_feat.unsqueeze(0).expand(3, -1, -1)                  # [3, N, *]
        emb_c_3 = emb_c_all.unsqueeze(1).expand(-1, N, -1)                      # [3, N, 8]
        trunk_in_3 = torch.cat([base_feat_3, emb_c_3], dim=-1).reshape(3 * N, -1)
        trunk_feat = F.silu(self.trunk_in(trunk_in_3))
        for block in self.trunk_blocks:
            trunk_feat = block(trunk_feat)
        trunk_basis = self.trunk_out(trunk_feat).view(3 * N, 3, self.rank)

        branch_query = self.branch_norm(trunk_feat)
        q = self.branch_query_proj(branch_query)                                # [3N, hidden]

        # c-independent tensors 對齊到 [3N, ...]：等同於 [c=0 段 / c=1 段 / c=2 段]
        k_3 = k_proj.repeat(3, 1, 1)                                            # [3N, K, hidden]
        v_3 = v_proj.repeat(3, 1, 1)                                            # [3N, K, hidden]
        rel_bias_3 = rel_bias.repeat(3, 1)                                      # [3N, K]
        rel_r_3 = rel_r.repeat(3, 1, 1)                                         # [3N, K, 1]

        scores = torch.einsum("nd,nkd->nk", q, k_3) / math.sqrt(k_3.shape[-1])
        scores = scores + rel_bias_3
        if self.use_locality_decay:
            decay_rate = torch.exp(self.log_locality_decay)
            scores = scores - decay_rate * rel_r_3.squeeze(-1)
        attn = torch.softmax(scores, dim=1)
        branch_ctx = torch.einsum("nk,nkd->nd", attn, v_3)
        branch_ctx = branch_ctx + self.branch_context(branch_ctx)
        branch_basis = self.branch_proj(branch_ctx).view(3 * N, 3, self.rank)

        # gather: 第 b 列屬於哪個 component → c_flat[b] = b // N
        c_flat = torch.arange(3, device=device, dtype=torch.long).repeat_interleave(N)
        comp_idx = c_flat.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.rank)
        trunk_sel = trunk_basis.gather(1, comp_idx).squeeze(1)                  # [3N, rank]
        branch_sel = branch_basis.gather(1, comp_idx).squeeze(1)                # [3N, rank]

        fusion_temperature = torch.exp(self.log_fusion_temperature).to(trunk_sel.dtype)
        out = torch.sum(trunk_sel * branch_sel, dim=1) * fusion_temperature     # [3N]
        out = out * self.component_scale[c_flat] + self.component_bias[c_flat]  # [3N]

        # [3N] → [3, N] → [N, 3]：column 0=u, 1=v, 2=p
        return out.view(3, N).T

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

        if self.spatial_emb is not None:
            pos_enc = self.spatial_emb(xy, self.domain_length)
        else:
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
        if self.use_periodic_domain:
            rel = rel - torch.round(rel / self.domain_length) * self.domain_length
        # smooth norm 避免 r=0 時二階導數未定義（physics second-order autograd NaN 根因）。
        # linalg.norm 在 r=0 的二階導數為 0/0；sqrt(r²+ε) 的二階導數在 r=0 是有限值。
        rel_r = torch.sqrt((rel ** 2).sum(dim=-1, keepdim=True) + 1e-8)
        # 只用距離（等向量），移除方向分量 (rel_x, rel_y)。
        # Why: 感測器 x 分佈非均勻，方向向量會將 x-column 非均勻性注入 attention bias，
        #      造成 x 方向條紋偽影；距離已足夠描述近鄰感測器的貢獻強度。
        rel_bias = self.relpos_bias(rel_r).squeeze(-1)
        scores = torch.einsum("nd,nkd->nk", q, k) / math.sqrt(k.shape[-1])
        scores = scores + rel_bias
        if self.use_locality_decay:
            # log-space 距離懲罰：score += -α * r，使遠端感測器貢獻指數衰減。
            # Why: 等效於 softmax 前乘以 exp(-α * r)，保留梯度可微性與 log-space 線性疊加。
            decay_rate = torch.exp(self.log_locality_decay)  # shape (1,)，α > 0
            scores = scores - decay_rate * rel_r.squeeze(-1)
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
