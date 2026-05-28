"""Sensor-side encoders: spatial set encoder + temporal CfC encoder."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from pi_con.blocks import CfCCell, ResidualMLPBlock, TokenSelfAttentionBlock
from pi_con.encodings import FourierEmbs, LearnableFourierEmb, periodic_fourier_encode


class SpatialSetEncoder(nn.Module):
    """What: 將單一時刻的感測器集合編碼成保留局部 identity 的 sensor tokens。

    Why: 先保留每個 sensor 的 RFF(x,y)+觀測通道局部訊息，不在 spatial branch
         提前做混合，讓 decoder 負責主要的 cross-attention 與讀取。

    Re conditioning (use_re_film=True):
        - 在每個 ResidualMLPBlock 之後注入 per-block FiLM (γ_l, β_l)：
            tokens ← γ_l(re_norm) ⊙ tokens + β_l(re_norm)
        - 物理動機: 黏性項 (1/Re)·∇²u 對 feature 是 multiplicative scaling,
          加性 bias 不足以表達跨 Re 的尺度變化（multi-Re 訓練必要）。
        - Identity init: γ_proj weight=0, bias=1; β_proj weight=0, bias=0
          → 初始 γ=1, β=0, FiLM 退化為 identity, 與 use_re_film=False 同初始行為。
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
        fourier_sigma_bands: tuple[float, ...] | list[float] | None = None,
        fourier_band_dim_ratios: tuple[float, ...] | list[float] | None = None,
        use_re_film: bool = False,
        use_graph_spatial_encoder: bool = False,
        graph_k_neighbors: int = 8,
        use_graph_spatial_gate: bool = False,
        geometry_preserve_base_rng: bool = False,
    ) -> None:
        super().__init__()
        self.domain_length = float(domain_length)
        self.fourier_harmonics = int(fourier_harmonics)
        self.sensor_value_dim = int(sensor_value_dim)
        self.use_periodic_domain = bool(use_periodic_domain)
        self.use_re_film = bool(use_re_film)
        self.use_graph_spatial_encoder = bool(use_graph_spatial_encoder)
        self.use_graph_spatial_gate = bool(use_graph_spatial_gate)
        self.geometry_preserve_base_rng = bool(geometry_preserve_base_rng)
        self.graph_k_neighbors = int(graph_k_neighbors)
        if self.graph_k_neighbors <= 0:
            raise ValueError(f"graph_k_neighbors 必須 > 0，收到 {graph_k_neighbors}")
        if fourier_embed_dim > 0:
            # 週期：LearnableFourierEmb（PeriodEmbs + 投影，支援頻率分層），x=0≡x=L 編碼。
            # 非週期：FourierEmbs 真 RFF，無預先週期化，能區分域邊界。
            # 頻率分層僅用於週期路徑；FourierEmbs 已是隨機頻率。
            if self.use_periodic_domain:
                self.spatial_emb: nn.Module | None = LearnableFourierEmb(
                    fourier_embed_dim,
                    init_sigma_bands=fourier_sigma_bands,
                    band_dim_ratios=fourier_band_dim_ratios,
                )
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
        # Per-block FiLM projections (only built when use_re_film=True);
        # 每個 ResidualMLPBlock 後對應一份 (γ_l, β_l)，layer_idx 與 self.blocks 對齊。
        if self.use_re_film:
            self.re_gamma_proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(depth)])
            self.re_beta_proj  = nn.ModuleList([nn.Linear(1, d_model) for _ in range(depth)])
            for g, b in zip(self.re_gamma_proj, self.re_beta_proj):
                nn.init.zeros_(g.weight); nn.init.ones_(g.bias)
                nn.init.zeros_(b.weight); nn.init.zeros_(b.bias)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
        )
        if self.use_graph_spatial_encoder:
            _rng_state = torch.random.get_rng_state() if self.geometry_preserve_base_rng else None
            try:
                self.graph_spatial_norm = nn.LayerNorm(d_model)
                self.graph_spatial_query_proj = nn.Linear(d_model, d_model)
                self.graph_spatial_key_proj = nn.Linear(spatial_dim, d_model)
                self.graph_spatial_value_proj = nn.Linear(spatial_dim, d_model)
                self.graph_spatial_rel_bias = nn.Sequential(
                    nn.LayerNorm(1),
                    nn.Linear(1, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, 1),
                )
                self.graph_spatial_out = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, 2 * d_model),
                    nn.SiLU(),
                    nn.Linear(2 * d_model, d_model),
                )
                if self.use_graph_spatial_gate:
                    self.graph_spatial_gate = nn.Parameter(torch.zeros(1, dtype=torch.float32))
            finally:
                if _rng_state is not None:
                    torch.random.set_rng_state(_rng_state)

    def encode_pos(self, sensor_pos: torch.Tensor) -> torch.Tensor:
        """What: 計算 sensor 位置的空間編碼。呼叫方負責在 loop 外預計算並重用。"""
        if self.spatial_emb is not None:
            return self.spatial_emb(sensor_pos, self.domain_length)
        return periodic_fourier_encode(sensor_pos, self.domain_length, self.fourier_harmonics)

    def forward(
        self,
        sensor_vals: torch.Tensor,
        pos_enc: torch.Tensor,
        re_norm: float | None = None,
        sensor_pos: torch.Tensor | None = None,
        geometry_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """支援 [K, C]（單一時刻、streaming）或 [T, K, C]（向量化整段時序）。

        Why: 整個 module 對 last-dim element-wise，T 軸只是 batch，
             向量化等價於原本 Python loop over T，但消除 T×=200× kernel-launch overhead。

        re_norm: 僅 use_re_film=True 時使用，per-block FiLM modulation 輸入。
                 use_re_film=False 路徑完全忽略此參數（向後相容）。
        """
        if sensor_vals.dim() == 3 and pos_enc.dim() == 2:
            T = sensor_vals.shape[0]
            pos_enc = pos_enc.unsqueeze(0).expand(T, -1, -1)
        base = torch.cat([pos_enc, sensor_vals], dim=-1)
        tokens = self.token_in(self.base_norm(base))
        if self.use_re_film and re_norm is not None:
            re_t = torch.tensor([[re_norm]], dtype=tokens.dtype, device=tokens.device)
            # broadcast shape：[K, C] → [1, d_model]；[T, K, C] → [1, 1, d_model]
            mod_shape = (1,) * (tokens.dim() - 1) + (-1,)
            for layer_idx, block in enumerate(self.blocks):
                tokens = block(tokens)
                gamma = self.re_gamma_proj[layer_idx](re_t).view(*mod_shape)
                beta  = self.re_beta_proj[layer_idx](re_t).view(*mod_shape)
                tokens = gamma * tokens + beta
        else:
            for block in self.blocks:
                tokens = block(tokens)
        tokens = self.out_proj(tokens)
        if self.use_graph_spatial_encoder:
            tokens = self._apply_geometry_graph(tokens, sensor_pos, geometry_pos)
        return tokens

    def _apply_geometry_graph(
        self,
        tokens: torch.Tensor,
        sensor_pos: torch.Tensor | None,
        geometry_pos: torch.Tensor | None,
    ) -> torch.Tensor:
        """What: 讓 sensor tokens 從幾何 token 聚合訊息。

        Why: 複雜幾何的 boundary/body 資訊應在進入 CfC 前注入 sensor token；
             否則時間記憶只看 sensor value/history，缺少 body topology。
        """
        if sensor_pos is None:
            raise ValueError("use_graph_spatial_encoder=True 但 sensor_pos=None；無法建立 sensor-geometry graph。")
        if geometry_pos is None or geometry_pos.shape[0] == 0:
            raise ValueError("use_graph_spatial_encoder=True 但 geometry_pos 為空；請先注入幾何點。")
        if geometry_pos.dim() != 2 or geometry_pos.shape[1] != 2:
            raise ValueError(f"geometry_pos 必須是 [N_geo,2]，收到 {tuple(geometry_pos.shape)}")
        if sensor_pos.dim() != 2 or sensor_pos.shape[1] != 2:
            raise ValueError(f"sensor_pos 必須是 [K,2]，收到 {tuple(sensor_pos.shape)}")

        geo_pos = geometry_pos.to(device=tokens.device, dtype=tokens.dtype)
        sensor_pos_d = sensor_pos.to(device=tokens.device, dtype=tokens.dtype)
        if self.spatial_emb is not None:
            geo_enc = self.spatial_emb(geo_pos, self.domain_length)
        else:
            geo_enc = periodic_fourier_encode(geo_pos, self.domain_length, self.fourier_harmonics)

        q = self.graph_spatial_query_proj(self.graph_spatial_norm(tokens))
        k = self.graph_spatial_key_proj(geo_enc)
        v = self.graph_spatial_value_proj(geo_enc)

        rel = sensor_pos_d.unsqueeze(1) - geo_pos.unsqueeze(0)
        if self.use_periodic_domain:
            rel = rel - torch.round(rel / self.domain_length) * self.domain_length
        rel_r = torch.sqrt((rel ** 2).sum(dim=-1, keepdim=True) + 1e-8)  # [K, G, 1]
        rel_bias = self.graph_spatial_rel_bias(rel_r).squeeze(-1)       # [K, G]
        if self.graph_k_neighbors < rel_r.shape[1]:
            keep_idx = torch.topk(
                rel_r.squeeze(-1),
                k=self.graph_k_neighbors,
                largest=False,
                dim=1,
            ).indices
            mask = torch.full_like(rel_bias, float("-inf"))
            mask.scatter_(1, keep_idx, 0.0)
            rel_bias = rel_bias + mask

        scale = math.sqrt(q.shape[-1])
        if tokens.dim() == 3:
            scores = torch.einsum("tkd,gd->tkg", q, k) / scale
            scores = scores + rel_bias.unsqueeze(0)
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.einsum("tkg,gd->tkd", attn, v)
        else:
            scores = torch.einsum("kd,gd->kg", q, k) / scale
            scores = scores + rel_bias
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.einsum("kg,gd->kd", attn, v)
        msg = self.graph_spatial_out(ctx)
        if self.use_graph_spatial_gate:
            msg = torch.tanh(self.graph_spatial_gate).to(msg.dtype) * msg
        return tokens + msg


class TemporalCfCEncoder(nn.Module):
    """What: 以 CfC 演化 sensor token 序列，產生 token states。

    Why: 保留每個 sensor token 的連續時間動態，讓 decoder 能直接讀取感測器級上下文。
         use_bidirectional=True 時加入反向掃描，使 t=0 的 hidden state 亦能看到未來觀測，
         適用於離線批量重建（所有感測器資料預先備妥）。

    Re conditioning:
        use_re_film=False (default, legacy)
            單一 shared self.re_proj: Linear(1, d_model)，
            re_bias 加到每個 CfC layer 入口的 attended sequence（純加性 modulation）。
        use_re_film=True (multi-Re)
            per-layer (γ_l, β_l)：每個 CfC layer 各自一份投影，套用 γ_l ⊙ x + β_l。
            Identity init: γ_proj weight=0, bias=1; β_proj weight=0, bias=0
            → 初始 γ=1, β=0，使 FiLM 退化為 identity，與舊路徑「只移除 re_bias 加法」初始等價。
        物理動機: 黏性項 (1/Re)·∇²u 是 multiplicative scaling, FiLM 比加法更貼合 NS 結構。
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_token_attention_layers: int = 1,
        token_attention_heads: int = 4,
        use_bidirectional: bool = False,
        cfc_log_tau_min: float = -1.0,
        cfc_log_tau_max: float = 1.0,
        use_re_film: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_bidirectional = use_bidirectional
        self.use_re_film = bool(use_re_film)
        if self.use_re_film:
            # Per-layer FiLM projection；layer_idx 與 self.cells 對齊。
            self.re_gamma_proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_layers)])
            self.re_beta_proj  = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_layers)])
            for g, b in zip(self.re_gamma_proj, self.re_beta_proj):
                nn.init.zeros_(g.weight); nn.init.ones_(g.bias)
                nn.init.zeros_(b.weight); nn.init.zeros_(b.bias)
        else:
            self.re_proj = nn.Linear(1, d_model)
        self.token_blocks = nn.ModuleList([
            TokenSelfAttentionBlock(d_model=d_model, num_heads=token_attention_heads)
            for _ in range(max(num_token_attention_layers, 0))
        ])
        self.cells = nn.ModuleList([
            CfCCell(d_model, d_model, log_tau_min=cfc_log_tau_min, log_tau_max=cfc_log_tau_max)
            for _ in range(num_layers)
        ])
        if use_bidirectional:
            # 反向 CfC：獨立參數，從 t=T-1 掃回 t=0。
            # Why: 讓 h_states[0] 也能看到未來觀測，消除因果編碼在 t=0 的資訊不對稱。
            self.backward_cells = nn.ModuleList([
                CfCCell(d_model, d_model, log_tau_min=cfc_log_tau_min, log_tau_max=cfc_log_tau_max)
                for _ in range(num_layers)
            ])

    def _re_bias(self, re_norm: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # 舊 additive 路徑（use_re_film=False 才可呼叫）。
        re_t = torch.tensor([[re_norm]], dtype=dtype, device=device)
        return self.re_proj(re_t).squeeze(0)

    def _re_film_pair(
        self,
        layer_idx: int,
        re_norm: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-layer FiLM projection (γ_l, β_l)；僅 use_re_film=True 時可呼叫。

        Returns:
            gamma, beta: each shape [d_model]，由 caller view 成所需 broadcast 形狀。
        """
        re_t = torch.tensor([[re_norm]], dtype=dtype, device=device)
        gamma = self.re_gamma_proj[layer_idx](re_t).squeeze(0)
        beta  = self.re_beta_proj[layer_idx](re_t).squeeze(0)
        return gamma, beta

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
        device, dtype = spatial_states.device, spatial_states.dtype
        if not self.use_re_film:
            shared_bias = self._re_bias(re_norm, device, dtype).view(1, 1, -1)
        seq = spatial_states
        for layer_idx in range(len(self.cells)):
            if self.token_blocks:
                block_idx = min(len(self.token_blocks) - 1, layer_idx)
                attended = self.token_blocks[block_idx](seq)        # [T, K, d]，T 為 batch
            else:
                attended = seq
            if self.use_re_film:
                # Per-layer FiLM: γ_l ⊙ attended + β_l
                gamma, beta = self._re_film_pair(layer_idx, re_norm, device, dtype)
                attended_modulated = gamma.view(1, 1, -1) * attended + beta.view(1, 1, -1)
            else:
                attended_modulated = attended + shared_bias         # 廣播到 [T, K, d]
            fwd = self._run_cfc_pass(attended_modulated, self.cells, dts, layer_idx, reverse=False)
            if self.use_bidirectional:
                bwd = self._run_cfc_pass(attended_modulated, self.backward_cells, dts, layer_idx, reverse=True)
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
        device, dtype = spatial_state.device, spatial_state.dtype
        if not self.use_re_film:
            shared_bias = self._re_bias(re_norm, device, dtype).view(1, -1)
        new_h_list: list[torch.Tensor] = []
        for layer_idx, (cell, h) in enumerate(zip(self.cells, h_list)):
            if self.token_blocks:
                block_idx = min(len(self.token_blocks) - 1, layer_idx)
                inp = self.token_blocks[block_idx](inp.unsqueeze(0)).squeeze(0)
            if self.use_re_film:
                gamma, beta = self._re_film_pair(layer_idx, re_norm, device, dtype)
                inp = gamma.view(1, -1) * inp + beta.view(1, -1)
            else:
                inp = inp + shared_bias
            new_h = cell(inp, h, dt=dt)
            # 層間殘差：加上前一層的輸出（與 forward() 一致）。
            new_h = new_h + new_h_list[layer_idx - 1] if layer_idx > 0 else new_h
            inp = new_h
            new_h_list.append(new_h)
        return new_h_list[-1], new_h_list
