"""Standard PINN baseline for paper-defensibility comparison.

What: Wang 2021 style single-instance PINN — (x, y, t) → MLP → (u, v, p).
Why: Demonstrate operator framework (LiquidOperator) advantage over plain PINN.
     Plain PINN is the dominant sparse-sensor PDE method; if it matches our
     operator framework on the same K=100 Re=10000 task, our architectural
     complexity is unjustified.

Architecture:
  (x, y, t) → LearnableFourierEmb + time + temporal_anchor → concat
            → Linear(input_dim, 512)
            → ResidualMLPBlock × 6 (hidden=512)
            → Linear(512, 3) → (u, v, p)

  NO operator framework (sensor enters only via loss, not model input)
  NO temporal recurrence (no CfC)
  NO cross-attention
  Single-instance: model is trained on fixed Kolmogorov trajectory

Output convention: returns (u, v, p) all at once (Wang 2021 standard).
For interface compat with LiquidOperator (which is c-conditioned), forward()
gathers c-th component; forward_uvp() returns all 3.

Param target: ~3.0-3.5M (matched to EXP-080's 3.14M for fair comparison).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from pi_lnn.blocks import ResidualMLPBlock, _build_activation
from pi_lnn.encodings import (
    FourierEmbs,
    LearnableFourierEmb,
    periodic_fourier_encode,
    temporal_phase_anchor,
)


class StandardPINNDecoder(nn.Module):
    """Single-instance PINN backbone — pure MLP on (x, y, t)."""

    def __init__(
        self,
        fourier_harmonics: int,
        d_time: int,
        domain_length: float = 1.0,
        use_temporal_anchor: bool = True,
        T_total: float = 5.0,
        temporal_anchor_harmonics: int = 2,
        num_layers: int = 6,
        hidden_dim: int = 512,
        output_head_gain: float = 1.0,
        fourier_embed_dim: int = 0,
        use_periodic_domain: bool = True,
        fourier_sigma_bands: tuple[float, ...] | list[float] | None = None,
        fourier_band_dim_ratios: tuple[float, ...] | list[float] | None = None,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.domain_length = float(domain_length)
        self.activation_name = activation
        self.use_temporal_anchor = bool(use_temporal_anchor)
        self.T_total = float(T_total)
        self.temporal_anchor_harmonics = int(temporal_anchor_harmonics)
        self.fourier_harmonics = int(fourier_harmonics)
        self.use_periodic_domain = bool(use_periodic_domain)
        # No hard body BC for PINN baseline (consistent with EXP-080)
        self.use_hard_body_bc = False

        # === Input encoding (same as LiquidOperator trunk for fair comparison) ===
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

        # Note: NO component embedding (output (u,v,p) directly, not c-conditioned)
        # This is the standard PINN convention; we wrap with gather() to match
        # LiquidOperator's c-conditioned interface.
        in_dim = spatial_dim + temporal_dim + d_time

        # === Backbone MLP ===
        # ResidualMLPBlock requires d_model == hidden_dim;
        # use a wider hidden ratio if desired in future.
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        # in_proj activation (matches block activation for consistency)
        self.in_act = _build_activation(activation)
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(d_model=hidden_dim, hidden_dim=hidden_dim, activation=activation)
            for _ in range(num_layers)
        ])
        self.trunk_out = nn.Linear(hidden_dim, 3)  # output (u, v, p) directly

        # Output head init (matched to LiquidOperator scheme)
        nn.init.xavier_normal_(self.trunk_out.weight, gain=output_head_gain)
        nn.init.zeros_(self.trunk_out.bias)

    # === Internal forward ===
    def _build_input(self, xy: torch.Tensor, t_q: torch.Tensor) -> torch.Tensor:
        """Build [N, in_dim] from query (xy, t)."""
        if self.spatial_emb is not None:
            pos_enc = self.spatial_emb(xy, self.domain_length)
        else:
            pos_enc = periodic_fourier_encode(xy, self.domain_length, self.fourier_harmonics)
        time_e = self.time_proj(t_q.unsqueeze(-1))

        parts = [pos_enc]
        if self.use_temporal_anchor:
            parts.append(temporal_phase_anchor(
                t_q.unsqueeze(-1), self.T_total, self.temporal_anchor_harmonics
            ))
        parts.append(time_e)
        return torch.cat(parts, dim=-1)

    def _forward_all(self, xy: torch.Tensor, t_q: torch.Tensor) -> torch.Tensor:
        """[N, 2], [N] → [N, 3] (u, v, p)."""
        x = self._build_input(xy, t_q)
        x = self.in_act(self.in_proj(x))   # 用 instance activation (silu / tanh / ...)
        for block in self.blocks:
            x = block(x)
        return self.trunk_out(x)

    # === External interface (compat with LiquidOperator.query_decoder) ===
    def forward(
        self,
        xy: torch.Tensor,
        t_q: torch.Tensor,
        c: torch.Tensor,
        h_states: torch.Tensor,       # unused (no sensor input)
        sensor_time: torch.Tensor,     # unused
        sensor_pos: torch.Tensor,      # unused
        body_distance: torch.Tensor | None = None,  # unused (no body BC)
    ) -> torch.Tensor:
        """Returns [N, 1] = the c-th component (for c-conditioned interface)."""
        uvp = self._forward_all(xy, t_q)  # [N, 3]
        comp_idx = c.unsqueeze(1)         # [N, 1]
        return uvp.gather(1, comp_idx)    # [N, 1]

    def forward_uvp(
        self,
        xy: torch.Tensor,
        t_q: torch.Tensor,
        h_states: torch.Tensor,
        sensor_time: torch.Tensor,
        sensor_pos: torch.Tensor,
        body_distance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns [N, 3] = (u, v, p) for physics path."""
        return self._forward_all(xy, t_q)


class StandardPINNOperator(nn.Module):
    """Drop-in replacement for LiquidOperator using a vanilla PINN backbone.

    encode() is a no-op (sensor doesn't feed into the model).
    set_physics_normalization / set_body_bc_scale / physics buffers
    all preserved for training pipeline compatibility.
    """

    def __init__(
        self,
        fourier_harmonics: int,
        d_time: int,
        domain_length: float = 1.0,
        use_temporal_anchor: bool = True,
        T_total: float = 5.0,
        temporal_anchor_harmonics: int = 2,
        num_layers: int = 6,
        hidden_dim: int = 512,
        output_head_gain: float = 1.0,
        fourier_embed_dim: int = 0,
        use_periodic_domain: bool = True,
        fourier_sigma_bands: tuple[float, ...] | list[float] | None = None,
        fourier_band_dim_ratios: tuple[float, ...] | list[float] | None = None,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.use_hard_body_bc = False
        self.query_decoder = StandardPINNDecoder(
            fourier_harmonics=fourier_harmonics,
            d_time=d_time,
            domain_length=domain_length,
            use_temporal_anchor=use_temporal_anchor,
            T_total=T_total,
            temporal_anchor_harmonics=temporal_anchor_harmonics,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            output_head_gain=output_head_gain,
            fourier_embed_dim=fourier_embed_dim,
            use_periodic_domain=use_periodic_domain,
            fourier_sigma_bands=fourier_sigma_bands,
            fourier_band_dim_ratios=fourier_band_dim_ratios,
            activation=activation,
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
        if not torch.is_tensor(mean):
            mean = torch.as_tensor(mean, dtype=torch.float32)
        if not torch.is_tensor(std):
            std = torch.as_tensor(std, dtype=torch.float32)
        if mean.shape != (3,) or std.shape != (3,):
            raise ValueError(
                f"mean/std must be (3,), got {tuple(mean.shape)} / {tuple(std.shape)}"
            )
        device = self.physics_output_mean.device
        self.physics_output_mean.copy_(mean.to(device=device, dtype=torch.float32))
        self.physics_output_std.copy_(std.to(device=device, dtype=torch.float32))

    def set_body_bc_scale(self, scale: float) -> None:
        """No-op for standard PINN (no hard body BC)."""
        pass

    def encode(
        self,
        sensor_vals: torch.Tensor,
        sensor_pos: torch.Tensor,
        re_norm: float,
        sensor_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """No-op encode: sensor doesn't feed into PINN model.

        Returns dummy (sensor_vals, sensor_time) for interface compat — the
        decoder ignores h_states / sensor_time / sensor_pos arguments.
        """
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


def create_standard_pinn_model(cfg: dict[str, Any]) -> StandardPINNOperator:
    """Build standard PINN from cfg."""
    return StandardPINNOperator(
        fourier_harmonics=int(cfg.get("fourier_harmonics", 8)),
        d_time=int(cfg.get("d_time", 16)),
        domain_length=float(cfg.get("domain_length", 1.0)),
        use_temporal_anchor=bool(cfg.get("use_temporal_anchor", True)),
        T_total=float(cfg.get("T_total", 5.0)),
        temporal_anchor_harmonics=int(cfg.get("temporal_anchor_harmonics", 2)),
        num_layers=int(cfg.get("standard_pinn_num_layers", 6)),
        hidden_dim=int(cfg.get("standard_pinn_hidden_dim", 512)),
        output_head_gain=float(cfg.get("output_head_gain", 1.0)),
        fourier_embed_dim=int(cfg.get("fourier_embed_dim", 0)),
        use_periodic_domain=bool(cfg.get("use_periodic_domain", True)),
        fourier_sigma_bands=cfg.get("fourier_sigma_bands"),
        fourier_band_dim_ratios=cfg.get("fourier_band_dim_ratios"),
        activation=str(cfg.get("standard_pinn_activation", "silu")),
    )
