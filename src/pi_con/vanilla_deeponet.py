"""Vanilla DeepONet baseline for architectural ablation (B0).

What: Pure DeepONet — MLP branch + MLP trunk + inner-product readout.
Why: Strip away CfC (temporal encoder) and cross-attention to test whether
     those components are essential for our K=100 Re=10000 task.

Architecture:
  Branch (no CfC, no spatial encoding):
    sensor_at_tq [N_q, K*C] → Linear → ResidualMLPBlock × num_branch_layers → Linear
    → branch_basis [N_q, 3, rank]

  Trunk (same encoding as LiquidOperator for fair Fourier feature comparison):
    (x, y, t, c) → spatial Fourier + temporal anchor + time + component
    → Linear → ResidualMLPBlock × num_trunk_layers → Linear
    → trunk_basis [N_q, 3, rank]

  Readout:
    u_c(x,y,t) = trunk_basis[c] · branch_basis[c]   (pure inner product)

Implements same external interface as LiquidOperator so training/evaluator
pipelines work without modification:
  - encode(sensor_vals, sensor_pos, re_norm, sensor_time) → (h_states=sensor_vals passthrough, sensor_time)
  - query_decoder.forward(...) → predictions
  - query_decoder.forward_uvp(...) → [N, 3] for physics path
  - set_physics_normalization(mean, std)
  - physics_output_mean / std buffers
  - use_hard_body_bc attribute (False for B0)
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from pi_con.blocks import ResidualMLPBlock
from pi_con.encodings import (
    FourierEmbs,
    LearnableFourierEmb,
    periodic_fourier_encode,
    temporal_phase_anchor,
)


class VanillaDeepONetDecoder(nn.Module):
    """Pure DeepONet decoder: MLP branch + MLP trunk + inner product.

    Branch input is sensor measurements at the query time t_q (co-temporal
    snapshot), flattened to [K * sensor_value_dim]. No temporal recurrent
    encoding, no attention. The trunk path retains the same Fourier feature
    encoding as LiquidOperator's decoder for fair comparison on the trunk side.
    """

    def __init__(
        self,
        K_sensors: int,
        sensor_value_dim: int,
        fourier_harmonics: int,
        d_time: int,
        domain_length: float = 1.0,
        use_temporal_anchor: bool = True,
        T_total: float = 5.0,
        temporal_anchor_harmonics: int = 2,
        num_branch_layers: int = 3,
        num_trunk_layers: int = 3,
        hidden_dim: int = 256,
        operator_rank: int = 256,
        output_head_gain: float = 1.0,
        fusion_temperature_init: float | None = None,
        fourier_embed_dim: int = 0,
        use_periodic_domain: bool = True,
        fourier_sigma_bands: tuple[float, ...] | list[float] | None = None,
        fourier_band_dim_ratios: tuple[float, ...] | list[float] | None = None,
    ) -> None:
        super().__init__()
        self.K_sensors = int(K_sensors)
        self.sensor_value_dim = int(sensor_value_dim)
        self.rank = int(operator_rank)
        self.domain_length = float(domain_length)
        self.use_temporal_anchor = bool(use_temporal_anchor)
        self.T_total = float(T_total)
        self.temporal_anchor_harmonics = int(temporal_anchor_harmonics)
        self.fourier_harmonics = int(fourier_harmonics)
        self.use_periodic_domain = bool(use_periodic_domain)
        # B0 baseline: no hard-body-BC (cylinder Q2 future work)
        self.use_hard_body_bc = False

        # === Branch: sensor (at query time) → branch_basis [N, 3, rank] ===
        branch_in_dim = self.K_sensors * self.sensor_value_dim
        self.branch_in = nn.Linear(branch_in_dim, hidden_dim)
        self.branch_blocks = nn.ModuleList([
            ResidualMLPBlock(d_model=hidden_dim, hidden_dim=hidden_dim)
            for _ in range(num_branch_layers)
        ])
        self.branch_out = nn.Linear(hidden_dim, 3 * self.rank)

        # === Trunk: (x, y, t, c) → trunk_basis [N, 3, rank] ===
        if fourier_embed_dim > 0:
            if use_periodic_domain:
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
                    "use_periodic_domain=False requires fourier_embed_dim > 0"
                )
            self.spatial_emb = None
            spatial_dim = 4 * fourier_harmonics

        temporal_dim = 2 * self.temporal_anchor_harmonics if self.use_temporal_anchor else 0
        self.time_proj = nn.Linear(1, d_time)
        self.component_emb = nn.Embedding(3, 8)
        nn.init.normal_(self.component_emb.weight, mean=0.0, std=0.1)

        trunk_in_dim = spatial_dim + temporal_dim + d_time + 8
        self.trunk_in = nn.Linear(trunk_in_dim, hidden_dim)
        self.trunk_blocks = nn.ModuleList([
            ResidualMLPBlock(d_model=hidden_dim, hidden_dim=hidden_dim)
            for _ in range(num_trunk_layers)
        ])
        self.trunk_out = nn.Linear(hidden_dim, 3 * self.rank)

        # Output head init (xavier with output_head_gain，與 LiquidOperator 一致)
        nn.init.xavier_normal_(self.trunk_out.weight, gain=output_head_gain)
        nn.init.zeros_(self.trunk_out.bias)
        nn.init.xavier_normal_(self.branch_out.weight, gain=output_head_gain)
        nn.init.zeros_(self.branch_out.bias)

        # Fusion temperature + component scale/bias (與 LiquidOperator 一致)
        temp_init = (1.0 / math.sqrt(self.rank)) if fusion_temperature_init is None else fusion_temperature_init
        if temp_init <= 0.0:
            raise ValueError(f"fusion_temperature_init must be > 0, got {temp_init}")
        self.log_fusion_temperature = nn.Parameter(
            torch.tensor([math.log(temp_init)], dtype=torch.float32)
        )
        self.component_scale = nn.Parameter(torch.ones(3))
        self.component_bias = nn.Parameter(torch.zeros(3))

    # === Internal helpers ===
    def _branch_at_time(
        self,
        sensor_vals: torch.Tensor,  # [T_sensor, K, C]
        sensor_time: torch.Tensor,  # [T_sensor]
        t_q: torch.Tensor,           # [N_q]
    ) -> torch.Tensor:
        """Return sensor reading at the nearest sensor_time ≤ t_q for each query.

        Returns: [N_q, K*C] flattened branch input.
        """
        idx = torch.searchsorted(sensor_time.contiguous(), t_q.contiguous(), right=True) - 1
        idx = idx.clamp(0, sensor_vals.shape[0] - 1)
        sensor_at_tq = sensor_vals[idx]  # [N_q, K, C]
        return sensor_at_tq.reshape(sensor_at_tq.shape[0], -1)

    def _branch_encode(self, branch_input: torch.Tensor) -> torch.Tensor:
        """[N_q, K*C] → [N_q, 3, rank]."""
        x = F.silu(self.branch_in(branch_input))
        for block in self.branch_blocks:
            x = block(x)
        out = self.branch_out(x)
        return out.view(-1, 3, self.rank)

    def _build_trunk_input(
        self, xy: torch.Tensor, t_q: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        """Build trunk input feature [N_q, trunk_in_dim] (handles both single-c
        path and multi-c batch path uniformly)."""
        if self.spatial_emb is not None:
            pos_enc = self.spatial_emb(xy, self.domain_length)
        else:
            pos_enc = periodic_fourier_encode(xy, self.domain_length, self.fourier_harmonics)
        time_e = self.time_proj(t_q.unsqueeze(-1))
        emb_c = self.component_emb(c)

        parts = [pos_enc]
        if self.use_temporal_anchor:
            parts.append(temporal_phase_anchor(
                t_q.unsqueeze(-1), self.T_total, self.temporal_anchor_harmonics
            ))
        parts.append(time_e)
        parts.append(emb_c)
        return torch.cat(parts, dim=-1)

    def _trunk_encode(
        self, xy: torch.Tensor, t_q: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        """[N_q, 2], [N_q], [N_q] → [N_q, 3, rank]."""
        x = self._build_trunk_input(xy, t_q, c)
        x = F.silu(self.trunk_in(x))
        for block in self.trunk_blocks:
            x = block(x)
        out = self.trunk_out(x)
        return out.view(-1, 3, self.rank)

    # === External interface (compat with LiquidOperator.query_decoder) ===
    def forward(
        self,
        xy: torch.Tensor,
        t_q: torch.Tensor,
        c: torch.Tensor,
        h_states: torch.Tensor,        # = sensor_vals (raw passthrough from encode)
        sensor_time: torch.Tensor,
        sensor_pos: torch.Tensor,       # unused (no spatial encoding of sensor positions)
        body_distance: torch.Tensor | None = None,  # ignored (no hard body BC)
    ) -> torch.Tensor:
        """Single-component prediction. Returns [N_q, 1]."""
        N = xy.shape[0]
        branch_input = self._branch_at_time(h_states, sensor_time, t_q)
        branch_basis = self._branch_encode(branch_input)        # [N, 3, rank]
        trunk_basis = self._trunk_encode(xy, t_q, c)            # [N, 3, rank]

        comp_idx = c.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.rank)
        trunk_sel = trunk_basis.gather(1, comp_idx).squeeze(1)  # [N, rank]
        branch_sel = branch_basis.gather(1, comp_idx).squeeze(1)

        fusion_temp = torch.exp(self.log_fusion_temperature).to(trunk_sel.dtype)
        out = torch.sum(trunk_sel * branch_sel, dim=1, keepdim=True) * fusion_temp
        out = out * self.component_scale[c].unsqueeze(1) + self.component_bias[c].unsqueeze(1)
        return out  # [N, 1]

    def forward_uvp(
        self,
        xy: torch.Tensor,
        t_q: torch.Tensor,
        h_states: torch.Tensor,
        sensor_time: torch.Tensor,
        sensor_pos: torch.Tensor,
        body_distance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """All-component prediction for physics path. Returns [N_q, 3]."""
        N = xy.shape[0]
        device = xy.device

        # Branch (c-independent)
        branch_input = self._branch_at_time(h_states, sensor_time, t_q)
        branch_basis = self._branch_encode(branch_input)         # [N, 3, rank]

        # Trunk: batched over c ∈ {0, 1, 2}, expand to [3N, ...]
        xy_3 = xy.unsqueeze(0).expand(3, -1, -1).reshape(3 * N, 2)
        tq_3 = t_q.unsqueeze(0).expand(3, -1).reshape(3 * N)
        c_3 = torch.arange(3, device=device, dtype=torch.long).repeat_interleave(N)

        trunk_basis_3 = self._trunk_encode(xy_3, tq_3, c_3)      # [3N, 3, rank]
        comp_idx_3 = c_3.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.rank)
        trunk_sel_3 = trunk_basis_3.gather(1, comp_idx_3).squeeze(1)  # [3N, rank]

        # Branch repeat to [3N, 3, rank], then gather by c
        branch_basis_3 = branch_basis.unsqueeze(0).expand(3, -1, -1, -1).reshape(3 * N, 3, self.rank)
        branch_sel_3 = branch_basis_3.gather(1, comp_idx_3).squeeze(1)

        fusion_temp = torch.exp(self.log_fusion_temperature).to(trunk_sel_3.dtype)
        out = torch.sum(trunk_sel_3 * branch_sel_3, dim=1) * fusion_temp
        out = out * self.component_scale[c_3] + self.component_bias[c_3]
        return out.view(3, N).T  # [N, 3] = (u, v, p)


class VanillaDeepONetOperator(nn.Module):
    """Drop-in replacement for LiquidOperator using vanilla DeepONet decoder.

    encode() simply passes sensor_vals through (branch indexes them by query
    time). No spatial encoder, no temporal encoder, no attention.
    """

    def __init__(
        self,
        K_sensors: int,
        sensor_value_dim: int,
        fourier_harmonics: int,
        d_time: int,
        domain_length: float = 1.0,
        use_temporal_anchor: bool = True,
        T_total: float = 5.0,
        temporal_anchor_harmonics: int = 2,
        num_branch_layers: int = 3,
        num_trunk_layers: int = 3,
        hidden_dim: int = 256,
        operator_rank: int = 256,
        output_head_gain: float = 1.0,
        fusion_temperature_init: float | None = None,
        fourier_embed_dim: int = 0,
        use_periodic_domain: bool = True,
        fourier_sigma_bands: tuple[float, ...] | list[float] | None = None,
        fourier_band_dim_ratios: tuple[float, ...] | list[float] | None = None,
    ) -> None:
        super().__init__()
        # use_hard_body_bc flag for trainer compatibility (always False for B0)
        self.use_hard_body_bc = False
        self.query_decoder = VanillaDeepONetDecoder(
            K_sensors=K_sensors,
            sensor_value_dim=sensor_value_dim,
            fourier_harmonics=fourier_harmonics,
            d_time=d_time,
            domain_length=domain_length,
            use_temporal_anchor=use_temporal_anchor,
            T_total=T_total,
            temporal_anchor_harmonics=temporal_anchor_harmonics,
            num_branch_layers=num_branch_layers,
            num_trunk_layers=num_trunk_layers,
            hidden_dim=hidden_dim,
            operator_rank=operator_rank,
            output_head_gain=output_head_gain,
            fusion_temperature_init=fusion_temperature_init,
            fourier_embed_dim=fourier_embed_dim,
            use_periodic_domain=use_periodic_domain,
            fourier_sigma_bands=fourier_sigma_bands,
            fourier_band_dim_ratios=fourier_band_dim_ratios,
        )

        # Physics output denormalization buffers (compat with training pipeline)
        self.register_buffer(
            "physics_output_mean",
            torch.zeros(3, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "physics_output_std",
            torch.ones(3, dtype=torch.float32),
            persistent=False,
        )

    def set_physics_normalization(
        self,
        mean: torch.Tensor | tuple | list,
        std: torch.Tensor | tuple | list,
    ) -> None:
        """Inject physics denormalization stats. Same semantics as LiquidOperator."""
        if not torch.is_tensor(mean):
            mean = torch.as_tensor(mean, dtype=torch.float32)
        if not torch.is_tensor(std):
            std = torch.as_tensor(std, dtype=torch.float32)
        if mean.shape != (3,) or std.shape != (3,):
            raise ValueError(
                f"mean/std must be shape (3,), got {tuple(mean.shape)} / {tuple(std.shape)}"
            )
        device = self.physics_output_mean.device
        self.physics_output_mean.copy_(mean.to(device=device, dtype=torch.float32))
        self.physics_output_std.copy_(std.to(device=device, dtype=torch.float32))

    def set_body_bc_scale(self, scale: float) -> None:
        """No-op for vanilla DeepONet (no hard body BC)."""
        pass

    def encode(
        self,
        sensor_vals: torch.Tensor,
        sensor_pos: torch.Tensor,
        re_norm: float,
        sensor_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Identity encode: branch will index sensor_vals by query time."""
        return sensor_vals, sensor_time

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


def create_vanilla_deeponet_model(cfg: dict[str, Any]) -> VanillaDeepONetOperator:
    """Build vanilla DeepONet from cfg. Reuses cfg keys where applicable."""
    # K_sensors is hard requirement for this baseline
    K = int(cfg.get("vanilla_deeponet_K_sensors", 100))

    return VanillaDeepONetOperator(
        K_sensors=K,
        sensor_value_dim=int(cfg.get("sensor_value_dim", 2)),
        fourier_harmonics=int(cfg.get("fourier_harmonics", 8)),
        d_time=int(cfg.get("d_time", 16)),
        domain_length=float(cfg.get("domain_length", 1.0)),
        use_temporal_anchor=bool(cfg.get("use_temporal_anchor", True)),
        T_total=float(cfg.get("T_total", 5.0)),
        temporal_anchor_harmonics=int(cfg.get("temporal_anchor_harmonics", 2)),
        num_branch_layers=int(cfg.get("vanilla_deeponet_num_branch_layers", 3)),
        num_trunk_layers=int(cfg.get("vanilla_deeponet_num_trunk_layers", 3)),
        hidden_dim=int(cfg.get("query_mlp_hidden_dim", 256)),
        operator_rank=int(cfg.get("operator_rank", 256)),
        output_head_gain=float(cfg.get("output_head_gain", 1.0)),
        fusion_temperature_init=(
            float(cfg["fusion_temperature_init"])
            if cfg.get("fusion_temperature_init") is not None
            else None
        ),
        fourier_embed_dim=int(cfg.get("fourier_embed_dim", 0)),
        use_periodic_domain=bool(cfg.get("use_periodic_domain", True)),
        fourier_sigma_bands=cfg.get("fourier_sigma_bands"),
        fourier_band_dim_ratios=cfg.get("fourier_band_dim_ratios"),
    )
