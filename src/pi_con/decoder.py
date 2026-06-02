"""DeepONet-style decoder with CfC trunk and cross-attention branch."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pi_con.blocks import ModifiedMLPBlock, ResidualMLPBlock
from pi_con.encodings import (
    FourierEmbs,
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
        fourier_sigma_bands: tuple[float, ...] | list[float] | None = None,
        fourier_band_dim_ratios: tuple[float, ...] | list[float] | None = None,
        use_hard_body_bc: bool = False,
        use_body_distance_feature: bool = False,
        use_geometry_tokens: bool = False,
        decoder_attention_heads: int = 1,
        use_modified_mlp: bool = False,
        disable_cross_attention: bool = False,
        use_trunk_geo_context: bool = False,
        geometry_preserve_base_rng: bool = False,
        use_sensor_film: bool = False,
        film_use_geometry: bool = False,
    ) -> None:
        super().__init__()
        self.use_periodic_domain = bool(use_periodic_domain)
        self.use_locality_decay = bool(use_locality_decay)
        # B1 ablation: 停用 cross-attention，改用 mean-pool over K sensor tokens。
        # 仍保留 branch_token_proj / branch_context / branch_proj 路徑（output basis）。
        self.disable_cross_attention = bool(disable_cross_attention)
        # Multi-head cross-attention：head 數須整除 query_mlp_hidden_dim。
        # H=1 即原 single-head（向後相容）；H>1 把 hidden 切成 H 個獨立 attention pattern，
        # rel_bias / locality_decay 仍 shared across heads（物理 distance penalty 與 head 無關）。
        if decoder_attention_heads <= 0:
            raise ValueError(f"decoder_attention_heads 必須 > 0，收到 {decoder_attention_heads}")
        self.decoder_attention_heads = int(decoder_attention_heads)
        # Modified MLP (Wang 2021 / PirateNet)：trunk path 用 gating between U/V，
        # 緩解 spectral bias。False = 原 ResidualMLPBlock（向後相容）。
        self.use_modified_mlp = bool(use_modified_mlp)
        # Hard body BC（Sukumar 2022 風格 output transformation）：
        #   u(x,y,t) = (φ(x,y) / scale).clamp(0,1) · NN_u(x,y,t)
        #   v(x,y,t) = (φ(x,y) / scale).clamp(0,1) · NN_v(x,y,t)
        #   p(x,y,t) = NN_p(x,y,t)                                # p 不 gate
        # 物理保證 body 內 (φ=0) → u=v=0 (no-slip)。
        # φ 是固定 SDF（不進 NN 學）；NN 是 raw output。
        # autograd 自然處理 chain rule: ∂u/∂x = ∂(φ·NN)/∂x = ∂φ/∂x·NN + φ·∂NN/∂x。
        # 因為 φ 用 PyTorch ops 實作（dataset.query_body_distance_torch），
        # autograd 走得通 → NS residual 算對 ∂u/∂x（不像 distance-as-input
        # 的 detach numpy lookup 漏 chain rule path）。
        self.use_hard_body_bc = bool(use_hard_body_bc)
        # Stage 2 Option A: SDF input feature
        # query = [x, y, t, c, φ(x,y)] post-Fourier concat (raw scalar, no encoding on φ).
        self.use_body_distance_feature = bool(use_body_distance_feature)
        self.geometry_preserve_base_rng = bool(geometry_preserve_base_rng)
        self.fourier_harmonics = int(fourier_harmonics)
        self.use_temporal_anchor = bool(use_temporal_anchor)
        self.T_total = float(T_total)
        self.temporal_anchor_harmonics = int(temporal_anchor_harmonics)
        temporal_dim = 2 * self.temporal_anchor_harmonics if self.use_temporal_anchor else 0
        if fourier_embed_dim > 0:
            # 週期：LearnableFourierEmb（支援頻率分層）；非週期：FourierEmbs 真 RFF。
            # 詳見 encodings.py 兩個類別的註解與 jaxpi 對齊說明。
            # 頻率分層（fourier_sigma_bands/band_dim_ratios）僅適用於週期路徑——
            # FourierEmbs 已是 RFF 隨機頻率，不需手動分層。
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
        # query_in 預設不含 body_distance（hard BC 是 output transformation）。
        # 但 Stage 2 Option A 啟用 use_body_distance_feature=True 時, query_in +1 (raw φ scalar)。
        query_in = spatial_dim + temporal_dim + d_time + 8 + (1 if self.use_body_distance_feature else 0)
        # Hard BC scale：phi_scale 用於把 distance 轉成 [0, 1] gate。
        # `body_bc_scale=1.0` default = identity scale；caller (training.py) 應該用
        # `set_body_bc_scale()` 注入 dataset-specific 的 max fluid distance。
        # 用 register_buffer (persistent=False) 避免污染既有 ckpt state_dict。
        self.register_buffer(
            "body_bc_scale",
            torch.tensor(1.0, dtype=torch.float32),
            persistent=False,
        )
        # Option E: Cross-attention geometry tokens (CEXP-022+)
        # body surface xy injected by training.py into geometry_pos buffer.
        # 3 new learnable modules:
        #   geo_key_proj: Fourier(body_xy) [N_body, spatial_dim] → key space [N_body, query_mlp_hidden_dim]
        #   geo_value: shared zero-velocity prior [1, query_mlp_hidden_dim] (learned init = 0)
        #   geo_token_type_bias: key bias distinguishing body tokens from sensor tokens [query_mlp_hidden_dim]
        self.use_geometry_tokens = bool(use_geometry_tokens)
        if self.use_geometry_tokens:
            self.geo_key_proj = nn.Linear(spatial_dim, query_mlp_hidden_dim)
            self.geo_value = nn.Parameter(torch.zeros(1, query_mlp_hidden_dim))
            self.geo_token_type_bias = nn.Parameter(torch.zeros(query_mlp_hidden_dim))
        # B3: Sensor-conditioned FiLM. cond = per-query sensor hidden (+ body_distance if geometry)。
        # γ identity-init (γ=1, β=0) → use_sensor_film=False 時 self.film_mlp 不建立；
        # 啟用時初始輸出 γ=1,β=0 不破壞既有 trunk_feat（向後相容）。
        self.use_sensor_film = bool(use_sensor_film)
        self.film_use_geometry = bool(film_use_geometry)
        if self.use_sensor_film:
            if d_model != query_mlp_hidden_dim:
                raise ValueError(
                    f"use_sensor_film 需要 d_model({d_model}) == query_mlp_hidden_dim"
                    f"({query_mlp_hidden_dim})；否則 sensor hidden 維度與 trunk feature 不符。"
                )
            _cond_dim = query_mlp_hidden_dim + (1 if self.film_use_geometry else 0)
            self.film_mlp = nn.Sequential(
                nn.Linear(_cond_dim, query_mlp_hidden_dim),
                nn.SiLU(),
                nn.Linear(query_mlp_hidden_dim, 2 * query_mlp_hidden_dim),
            )
            # identity init：最後一層 weight=0，bias=[1...(γ), 0...(β)] → 初始 γ=1,β=0。
            nn.init.zeros_(self.film_mlp[-1].weight)
            with torch.no_grad():
                self.film_mlp[-1].bias[:query_mlp_hidden_dim].fill_(1.0)
                self.film_mlp[-1].bias[query_mlp_hidden_dim:].fill_(0.0)

        self.use_trunk_geo_context = bool(use_trunk_geo_context)
        if self.use_trunk_geo_context:
            _rng_state = torch.random.get_rng_state() if self.geometry_preserve_base_rng else None
            try:
                self.trunk_geo_query_proj = nn.Linear(query_mlp_hidden_dim, query_mlp_hidden_dim)
                self.trunk_geo_key_proj = nn.Linear(spatial_dim, query_mlp_hidden_dim)
                self.trunk_geo_value_proj = nn.Linear(spatial_dim, query_mlp_hidden_dim)
                self.trunk_geo_rel_bias = nn.Sequential(
                    nn.LayerNorm(1),
                    nn.Linear(1, query_mlp_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(query_mlp_hidden_dim, 1),
                )
                self.trunk_geo_context = nn.Sequential(
                    nn.LayerNorm(query_mlp_hidden_dim),
                    nn.Linear(query_mlp_hidden_dim, query_mlp_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(query_mlp_hidden_dim, query_mlp_hidden_dim),
                )
                # 由 0 開始，讓新實驗初期從舊 trunk 附近啟動；只有 flag=True 時才存在。
                self.trunk_geo_gate = nn.Parameter(torch.zeros(1, dtype=torch.float32))
            finally:
                if _rng_state is not None:
                    torch.random.set_rng_state(_rng_state)
        # geometry_pos: body surface normalized coordinates, shape [N_body, 2]
        # persistent=False: not saved in ckpt; filled at training start from ds.body_xy
        self.register_buffer(
            "geometry_pos",
            torch.zeros(0, 2, dtype=torch.float32),
            persistent=False,
        )
        rank = d_model if operator_rank is None else operator_rank
        if rank <= 0:
            raise ValueError(f"operator_rank 必須 > 0，收到 {rank}")
        self.rank = rank
        self.domain_length = float(domain_length)
        self.time_proj = nn.Linear(1, d_time)
        self.component_emb = nn.Embedding(3, 8)
        nn.init.normal_(self.component_emb.weight, mean=0.0, std=0.1)
        self.trunk_in = nn.Linear(query_in, query_mlp_hidden_dim)
        # Modified MLP gating: U, V 是 trunk_in 的兩個 nonlinear embedding，
        # 每層 ModifiedMLPBlock 在 U/V 之間 gate（Wang 2021）。
        # 只在 use_modified_mlp=True 時建構，避免增加 ResidualMLP 路徑的 param。
        if self.use_modified_mlp:
            self.trunk_U_proj = nn.Linear(query_in, query_mlp_hidden_dim)
            self.trunk_V_proj = nn.Linear(query_in, query_mlp_hidden_dim)
        self.trunk_blocks = nn.ModuleList([
            ModifiedMLPBlock(d_model=query_mlp_hidden_dim, hidden_dim=query_mlp_hidden_dim)
            if self.use_modified_mlp else
            ResidualMLPBlock(d_model=query_mlp_hidden_dim, hidden_dim=query_mlp_hidden_dim)
            for _ in range(num_query_mlp_layers)
        ])
        if query_mlp_hidden_dim % 4 != 0:
            raise ValueError(
                f"query_mlp_hidden_dim 必須能被 4 整除，收到 {query_mlp_hidden_dim}"
            )
        if query_mlp_hidden_dim % self.decoder_attention_heads != 0:
            raise ValueError(
                f"query_mlp_hidden_dim={query_mlp_hidden_dim} 必須能被 "
                f"decoder_attention_heads={self.decoder_attention_heads} 整除"
            )
        self.attn_head_dim = query_mlp_hidden_dim // self.decoder_attention_heads
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

    def _geometry_positions(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.geometry_pos.shape[0] == 0:
            raise ValueError("use_trunk_geo_context=True 但 geometry_pos 為空；請先注入幾何點。")
        return self.geometry_pos.to(device=device, dtype=dtype)

    def _encode_geometry(self, geo_pos: torch.Tensor) -> torch.Tensor:
        if self.spatial_emb is not None:
            return self.spatial_emb(geo_pos, self.domain_length)
        return periodic_fourier_encode(geo_pos, self.domain_length, self.fourier_harmonics)

    def _apply_sensor_film(
        self,
        trunk_feat: torch.Tensor,        # [M, hidden]，M = N (forward) 或 3N (forward_uvp)
        h_branch_tokens: torch.Tensor,   # [N, K, d_model] 或 [N, d_model]
        body_distance: torch.Tensor | None,  # [N] or [N,1] or None
        n_repeat: int,                   # 1 (forward) 或 3 (forward_uvp)
    ) -> torch.Tensor:
        """B3: γ(cond)⊙trunk_feat + β(cond)，cond 從 per-query sensor hidden (+geometry)。

        Why: 不在 velocity output 強制 body=0（已證 over-energy），改用 sensor 狀態
             大域調制 trunk feature。conditioning 來自有 supervision 的 sensor，
             結構性回避「body 區無校正」根因（Finding #8）。

        h_branch_tokens 形狀說明：
            - 若 temporal encoder 輸出 [T, K, d_model]，則 h_states[idx] = [N, K, d_model]（3D）。
            - 3D 時對 K tokens 做 mean pooling → 得到 [N, d_model] summary。
            - 2D [N, d_model] 時直接使用（test / legacy path）。
        """
        if not self.use_sensor_film:
            return trunk_feat
        # 若 h_branch_tokens 是 [N, K, d_model]（從 h_states 直接切出），先 mean over K 維
        # Why: film_mlp 期望 1D per-query conditioning；K sensor tokens 的平均是最自然的全局摘要。
        if h_branch_tokens.dim() == 3:
            cond = h_branch_tokens.mean(dim=-2)                      # [N, d_model]
        else:
            cond = h_branch_tokens                                   # [N, d_model]（2D legacy/test）
        if self.film_use_geometry:
            if body_distance is None:
                raise ValueError(
                    "film_use_geometry=True 但 _apply_sensor_film body_distance=None；"
                    "請傳入 dataset.query_body_distance_torch(xy)。"
                )
            cond = torch.cat([cond, body_distance.reshape(-1, 1)], dim=-1)  # [N, d_model+1]
        gb = self.film_mlp(cond)                                     # [N, 2·d_model]
        hidden = gb.shape[-1] // 2
        gamma, beta = gb[..., :hidden], gb[..., hidden:]            # [N, hidden] each（... 保持維度中性）
        if n_repeat > 1:
            gamma = gamma.repeat(n_repeat, 1)                       # [3N, hidden]
            beta = beta.repeat(n_repeat, 1)
        return gamma * trunk_feat + beta

    def _apply_trunk_geo_context(
        self,
        trunk_feat: torch.Tensor,
        xy: torch.Tensor,
    ) -> torch.Tensor:
        """What: 讓 continuous query trunk 從全域 geometry memory 取回 query-local context。"""
        if not self.use_trunk_geo_context:
            return trunk_feat
        geo_pos = self._geometry_positions(device=xy.device, dtype=xy.dtype)
        geo_enc = self._encode_geometry(geo_pos)
        q = self.trunk_geo_query_proj(trunk_feat)
        k = self.trunk_geo_key_proj(geo_enc)
        v = self.trunk_geo_value_proj(geo_enc)

        rel = xy.unsqueeze(1) - geo_pos.unsqueeze(0)
        if self.use_periodic_domain:
            rel = rel - torch.round(rel / self.domain_length) * self.domain_length
        rel_r = torch.sqrt((rel ** 2).sum(dim=-1, keepdim=True) + 1e-8)
        rel_bias = self.trunk_geo_rel_bias(rel_r).squeeze(-1)
        scores = torch.matmul(q, k.T) / math.sqrt(q.shape[-1])
        scores = scores + rel_bias
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)
        return trunk_feat + torch.tanh(self.trunk_geo_gate).to(trunk_feat.dtype) * self.trunk_geo_context(ctx)

    def forward_uvp(
        self,
        xy: torch.Tensor,
        t_q: torch.Tensor,
        h_states: torch.Tensor,
        sensor_time: torch.Tensor,
        sensor_pos: torch.Tensor,
        body_distance: torch.Tensor | None = None,
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
        # Stage 2 Option A: SDF trunk input — concat raw φ scalar (post-Fourier, no encoding)
        if self.use_body_distance_feature:
            if body_distance is None:
                raise ValueError(
                    "use_body_distance_feature=True 但 forward_uvp() body_distance=None；"
                    "請傳入 dataset.query_body_distance_torch(xy) 結果（differentiable）。"
                )
            phi_feat = body_distance.reshape(-1, 1)                              # [N, 1]
            base_inputs.append(phi_feat)
        # NOTE: hard body BC 不在這裡 concat distance（output transformation, 在 return 前）
        base_feat = torch.cat(base_inputs, dim=-1)                              # [N, query_in - 8]

        c_all = torch.arange(3, device=device, dtype=torch.long)
        emb_c_all = self.component_emb(c_all)                                   # [3, 8]
        base_feat_3 = base_feat.unsqueeze(0).expand(3, -1, -1)                  # [3, N, *]
        emb_c_3 = emb_c_all.unsqueeze(1).expand(-1, N, -1)                      # [3, N, 8]
        trunk_in_3 = torch.cat([base_feat_3, emb_c_3], dim=-1).reshape(3 * N, -1)
        trunk_feat = F.silu(self.trunk_in(trunk_in_3))
        if self.use_modified_mlp:
            # mMLP: 計算 U, V embedding 一次（layer-independent），每層 block 在 U/V gate。
            U_emb = F.silu(self.trunk_U_proj(trunk_in_3))
            V_emb = F.silu(self.trunk_V_proj(trunk_in_3))
            for block in self.trunk_blocks:
                trunk_feat = block(trunk_feat, U_emb, V_emb)
        else:
            for block in self.trunk_blocks:
                trunk_feat = block(trunk_feat)
        xy_3 = xy.unsqueeze(0).expand(3, -1, -1).reshape(3 * N, 2)
        trunk_feat = self._apply_sensor_film(trunk_feat, h_branch_tokens, body_distance, n_repeat=3)
        trunk_feat = self._apply_trunk_geo_context(trunk_feat, xy_3)
        trunk_basis = self.trunk_out(trunk_feat).view(3 * N, 3, self.rank)

        branch_query = self.branch_norm(trunk_feat)
        q = self.branch_query_proj(branch_query)                                # [3N, hidden]

        # c-independent tensors 對齊到 [3N, ...]：等同於 [c=0 段 / c=1 段 / c=2 段]
        k_3 = k_proj.repeat(3, 1, 1)                                            # [3N, K, hidden]
        v_3 = v_proj.repeat(3, 1, 1)                                            # [3N, K, hidden]
        rel_bias_3 = rel_bias.repeat(3, 1)                                      # [3N, K]
        rel_r_3 = rel_r.repeat(3, 1, 1)                                         # [3N, K, 1]

        # Option E: Append geometry tokens to K-V pool
        if self.use_geometry_tokens and self.geometry_pos.shape[0] > 0:
            N_body = self.geometry_pos.shape[0]
            geo_pos = self.geometry_pos.to(device=device, dtype=xy.dtype)
            # Fourier encode body surface positions (reuse same spatial_emb as query)
            if self.spatial_emb is not None:
                geo_enc = self.spatial_emb(geo_pos, self.domain_length)          # [N_body, spatial_dim]
            else:
                geo_enc = periodic_fourier_encode(geo_pos, self.domain_length, self.fourier_harmonics)
            # Key: position encoding → key projection + token type bias
            geo_k = self.geo_key_proj(geo_enc) + self.geo_token_type_bias       # [N_body, hidden]
            # Value: shared zero-velocity prior (all body tokens share same value)
            geo_v = self.geo_value.expand(N_body, -1)                           # [N_body, hidden]
            # Extend to [3N, N_body, hidden] for 3-channel batch
            geo_k_3 = geo_k.unsqueeze(0).expand(3 * N, -1, -1)                 # [3N, N_body, hidden]
            geo_v_3 = geo_v.unsqueeze(0).expand(3 * N, -1, -1)                 # [3N, N_body, hidden]
            k_3 = torch.cat([k_3, geo_k_3], dim=1)                             # [3N, K+N_body, hidden]
            v_3 = torch.cat([v_3, geo_v_3], dim=1)                             # [3N, K+N_body, hidden]
            # Extend relpos bias: query-to-body distances
            geo_rel = xy.unsqueeze(1) - geo_pos.unsqueeze(0)                   # [N, N_body, 2]
            geo_rel_r = torch.sqrt((geo_rel ** 2).sum(-1, keepdim=True) + 1e-8) # [N, N_body, 1]
            geo_bias = self.relpos_bias(geo_rel_r).squeeze(-1)                  # [N, N_body]
            rel_bias_3 = torch.cat([rel_bias_3, geo_bias.repeat(3, 1)], dim=1) # [3N, K+N_body]
            if self.use_locality_decay:
                rel_r_3 = torch.cat([rel_r_3, geo_rel_r.repeat(3, 1, 1)], dim=1)  # [3N, K+N_body, 1]

        if self.disable_cross_attention:
            # B1 ablation: 替 cross-attention 為 mean-pool over K sensor tokens (per query)。
            # v_3 [3N, K, hidden]; pool over dim=1 (K dimension); 結果 [3N, hidden]。
            # rel_bias / locality_decay 都不適用 (mean-pool 不需 attention weights)。
            branch_ctx = v_3.mean(dim=1)
        else:
            # Multi-head cross-attention（H=1 退化為原 single-head 行為，shape 驗證見 __init__）：
            # q  [3N, hidden]      → [3N, H, D]
            # k  [3N, K, hidden]   → [3N, K, H, D]
            # v  [3N, K, hidden]   → [3N, K, H, D]
            # scores [3N, H, K] = einsum("nhd,nkhd->nhk") / sqrt(D)
            # rel_bias / locality_decay shared across heads → unsqueeze H 維度 broadcast。
            H = self.decoder_attention_heads
            D = self.attn_head_dim
            K = k_3.shape[1]
            q_h = q.view(3 * N, H, D)
            k_h = k_3.view(3 * N, K, H, D)
            v_h = v_3.view(3 * N, K, H, D)
            scores = torch.einsum("nhd,nkhd->nhk", q_h, k_h) / math.sqrt(D)
            scores = scores + rel_bias_3.unsqueeze(1)                                # [3N, 1, K]→broadcast H
            if self.use_locality_decay:
                decay_rate = torch.exp(self.log_locality_decay)
                scores = scores - decay_rate * rel_r_3.squeeze(-1).unsqueeze(1)
            attn = torch.softmax(scores, dim=-1)                                     # softmax over K
            branch_ctx_h = torch.einsum("nhk,nkhd->nhd", attn, v_h)                  # [3N, H, D]
            branch_ctx = branch_ctx_h.reshape(3 * N, H * D)                          # concat heads
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
        uvp = out.view(3, N).T

        # Hard body BC (Sukumar 2022): u, v ← gate · NN; p 不 gate
        # gate = (φ / scale).clamp(0, 1)：body 內 (φ=0) → 0 → u=v=0；遠 body 飽和 → 1
        if self.use_hard_body_bc:
            if body_distance is None:
                raise ValueError(
                    "use_hard_body_bc=True 但 forward_uvp() body_distance=None；"
                    "請傳入 dataset.query_body_distance_torch(xy) 結果（differentiable）。"
                )
            phi = body_distance.reshape(-1)                # [N]
            scale = self.body_bc_scale.to(phi.dtype)
            gate = (phi / scale).clamp(min=0.0, max=1.0)   # [N]，body=0, fluid=1
            gate = gate.unsqueeze(-1)                       # [N, 1]
            # 只 gate u, v（component 0, 1），保留 p（column 2）
            uvp = torch.cat([
                gate * uvp[:, 0:1],
                gate * uvp[:, 1:2],
                uvp[:, 2:3],
            ], dim=-1)
        return uvp

    def forward(
        self,
        xy: torch.Tensor,
        t_q: torch.Tensor,
        c: torch.Tensor,
        h_states: torch.Tensor,
        sensor_time: torch.Tensor,
        sensor_pos: torch.Tensor,
        body_distance: torch.Tensor | None = None,
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
        # 注意：body_distance 在 emb_c 之前 concat，與 forward_uvp 的 base_feat
        # 結構一致（base_feat 含 body_distance，再跟 emb_c concat）。
        trunk_inputs.append(time_e)
        # Stage 2 Option A: SDF trunk input feature (與 forward_uvp 對稱)。
        if self.use_body_distance_feature:
            if body_distance is None:
                raise ValueError(
                    "use_body_distance_feature=True 但 forward() body_distance=None；"
                    "請傳入 dataset.query_body_distance_torch(xy) 結果（differentiable）。"
                )
            trunk_inputs.append(body_distance.reshape(-1, 1))                    # [N, 1] raw φ
        # NOTE: hard body BC 是 output transformation，不在這裡 concat distance
        trunk_inputs.append(emb_c)
        trunk_in_cat = torch.cat(trunk_inputs, dim=-1)
        trunk_feat = F.silu(self.trunk_in(trunk_in_cat))
        if self.use_modified_mlp:
            # 與 forward_uvp 同邏輯：U/V 從 trunk_in_cat 計算，每層 block gate U/V。
            U_emb = F.silu(self.trunk_U_proj(trunk_in_cat))
            V_emb = F.silu(self.trunk_V_proj(trunk_in_cat))
            for block in self.trunk_blocks:
                trunk_feat = block(trunk_feat, U_emb, V_emb)
        else:
            for block in self.trunk_blocks:
                trunk_feat = block(trunk_feat)
        _n_repeat_fwd = trunk_feat.shape[0] // h_branch_tokens.shape[0]   # 動態：N→1, 2N→2, 3N→3
        trunk_feat = self._apply_sensor_film(trunk_feat, h_branch_tokens, body_distance, n_repeat=_n_repeat_fwd)
        trunk_feat = self._apply_trunk_geo_context(trunk_feat, xy)

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

        # Option E: Append geometry tokens to K-V pool (single-channel path, symmetric to forward_uvp)
        if self.use_geometry_tokens and self.geometry_pos.shape[0] > 0:
            N_body = self.geometry_pos.shape[0]
            geo_pos = self.geometry_pos.to(device=xy.device, dtype=xy.dtype)
            if self.spatial_emb is not None:
                geo_enc = self.spatial_emb(geo_pos, self.domain_length)
            else:
                geo_enc = periodic_fourier_encode(geo_pos, self.domain_length, self.fourier_harmonics)
            geo_k = self.geo_key_proj(geo_enc) + self.geo_token_type_bias       # [N_body, hidden]
            geo_v = self.geo_value.expand(N_body, -1)                           # [N_body, hidden]
            N_q = q.shape[0]
            geo_k_q = geo_k.unsqueeze(0).expand(N_q, -1, -1)                   # [N_q, N_body, hidden]
            geo_v_q = geo_v.unsqueeze(0).expand(N_q, -1, -1)                   # [N_q, N_body, hidden]
            k = torch.cat([k, geo_k_q], dim=1)                                 # [N_q, K+N_body, hidden]
            v = torch.cat([v, geo_v_q], dim=1)                                 # [N_q, K+N_body, hidden]
            geo_rel = xy.unsqueeze(1) - geo_pos.unsqueeze(0)
            geo_rel_r = torch.sqrt((geo_rel ** 2).sum(-1, keepdim=True) + 1e-8)
            geo_bias = self.relpos_bias(geo_rel_r).squeeze(-1)
            rel_bias = torch.cat([rel_bias, geo_bias], dim=1)                  # [N_q, K+N_body]
            if self.use_locality_decay:
                rel_r = torch.cat([rel_r, geo_rel_r], dim=1)                   # [N_q, K+N_body, 1]

        if self.disable_cross_attention:
            # B1 ablation: mean-pool over K sensor tokens (per query)。
            # v [N_q, K, hidden] → mean dim=1 → [N_q, hidden]。
            branch_ctx = v.mean(dim=1)
        else:
            # Multi-head cross-attention（與 forward_uvp 同邏輯）：H=1 退化為原 single-head。
            # k/v 是 [N_q, K, hidden] 而非 [K, hidden]：每個 query 對應自己的 sensor time-snapshot
            # （見 line 280-281 idx-based gather of h_states）。
            N_q = q.shape[0]
            H = self.decoder_attention_heads
            D = self.attn_head_dim
            K = k.shape[1]
            q_h = q.view(N_q, H, D)
            k_h = k.view(N_q, K, H, D)
            v_h = v.view(N_q, K, H, D)
            scores = torch.einsum("nhd,nkhd->nhk", q_h, k_h) / math.sqrt(D)            # [N_q, H, K]
            scores = scores + rel_bias.unsqueeze(1)                                     # [N_q, 1, K]→broadcast H
            if self.use_locality_decay:
                # log-space 距離懲罰：score += -α * r，使遠端感測器貢獻指數衰減。
                # Why: 等效於 softmax 前乘以 exp(-α * r)，保留梯度可微性與 log-space 線性疊加。
                decay_rate = torch.exp(self.log_locality_decay)  # shape (1,)，α > 0
                scores = scores - decay_rate * rel_r.squeeze(-1).unsqueeze(1)
            attn = torch.softmax(scores, dim=-1)                                        # softmax over K
            branch_ctx_h = torch.einsum("nhk,nkhd->nhd", attn, v_h)                     # [N_q, H, D]
            branch_ctx = branch_ctx_h.reshape(N_q, H * D)
        branch_ctx = branch_ctx + self.branch_context(branch_ctx)
        branch_basis = self.branch_proj(branch_ctx).view(-1, 3, self.rank)
        comp_idx = c.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.rank)
        trunk_sel = trunk_basis.gather(1, comp_idx).squeeze(1)
        branch_sel = branch_basis.gather(1, comp_idx).squeeze(1)
        fusion_temperature = torch.exp(self.log_fusion_temperature).to(trunk_sel.dtype)
        out = torch.sum(trunk_sel * branch_sel, dim=1, keepdim=True) * fusion_temperature
        out = out * self.component_scale[c].unsqueeze(1) + self.component_bias[c].unsqueeze(1)

        # Hard body BC：只 gate u (c=0) 與 v (c=1)，保留 p (c=2)。
        # 用 mask = (c < 2) 對 c=0/1 套 gate，c=2 不變。
        if self.use_hard_body_bc:
            if body_distance is None:
                raise ValueError(
                    "use_hard_body_bc=True 但 forward() body_distance=None；"
                    "請傳入 dataset.query_body_distance_torch(xy) 結果（differentiable）。"
                )
            phi = body_distance.reshape(-1, 1)                  # [N, 1]
            scale = self.body_bc_scale.to(phi.dtype)
            gate = (phi / scale).clamp(min=0.0, max=1.0)        # [N, 1]
            # 對 c==0/1 套 gate，c==2 保留 1.0
            apply_gate = (c.unsqueeze(1) < 2).to(out.dtype)     # [N, 1]，1 if u/v else 0
            effective_gate = apply_gate * gate + (1.0 - apply_gate) * 1.0
            out = effective_gate * out
        return out
