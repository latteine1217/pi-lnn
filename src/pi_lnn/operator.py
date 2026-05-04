"""Pi-LNN main operator: LiquidOperator model class + factory + closure helper."""
from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from pi_lnn.decoder import DeepONetCfCDecoder
from pi_lnn.encoders import SpatialSetEncoder, TemporalCfCEncoder


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
        use_locality_decay: bool = False,
        use_bidirectional_cfc: bool = False,
        fourier_embed_dim: int = 0,
        use_periodic_domain: bool = True,
        cfc_log_tau_min: float = -1.0,
        cfc_log_tau_max: float = 1.0,
        fourier_sigma_bands: tuple[float, ...] | list[float] | None = None,
        fourier_band_dim_ratios: tuple[float, ...] | list[float] | None = None,
        use_hard_body_bc: bool = False,
    ) -> None:
        super().__init__()
        # Hard body BC 是 output transformation：u, v ← (φ/scale).clamp(0,1) · NN
        # 物理保證 body 內 u=v=0（取代 distance-as-input feature 那個會踩 detach 的方案）。
        self.use_hard_body_bc = bool(use_hard_body_bc)
        self.spatial_encoder = SpatialSetEncoder(
            fourier_harmonics,
            sensor_value_dim,
            d_model,
            num_spatial_cfc_layers,
            domain_length=domain_length,
            fourier_embed_dim=fourier_embed_dim,
            use_periodic_domain=use_periodic_domain,
            fourier_sigma_bands=fourier_sigma_bands,
            fourier_band_dim_ratios=fourier_band_dim_ratios,
        )
        self.temporal_encoder = TemporalCfCEncoder(
            d_model,
            num_temporal_cfc_layers,
            num_token_attention_layers=num_token_attention_layers,
            token_attention_heads=token_attention_heads,
            use_bidirectional=use_bidirectional_cfc,
            cfc_log_tau_min=cfc_log_tau_min,
            cfc_log_tau_max=cfc_log_tau_max,
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
            use_locality_decay=use_locality_decay,
            fourier_embed_dim=fourier_embed_dim,
            use_periodic_domain=use_periodic_domain,
            fourier_sigma_bands=fourier_sigma_bands,
            fourier_band_dim_ratios=fourier_band_dim_ratios,
            use_hard_body_bc=use_hard_body_bc,
        )

        # Physics output denormalization buffers
        # Why: query_decoder 直接學 normalized sensor target，output 量級為 unit-std。
        #      但 NS residual 用物理 ν=1/Re，混合 normalized u 與物理 ν 會讓 momentum
        #      殘差量級被壓到 ~ν，使 NG / Adam 都看不見 physics gradient signal。
        #      正確做法：physics path 把 output × std + mean 還原成物理單位再算 NS。
        # 預設 (mean=0, std=1) → identity，向後相容舊 checkpoint。
        # 訓練 / evaluate 流程必須呼叫 set_physics_normalization() 注入正確 (mean, std)。
        # persistent=False 讓 buffer 不出現在 state_dict，避免污染現有 checkpoint。
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
        mean: torch.Tensor | tuple[float, float, float] | list[float],
        std:  torch.Tensor | tuple[float, float, float] | list[float],
    ) -> None:
        """注入 physics path 的反正規化統計量。

        Args:
            mean: [3] = (mean_u, mean_v, mean_p)。p 無 supervision，建議 0.0。
            std:  [3] = (std_u,  std_v,  std_p)。p 無 supervision，建議 1.0。

        後續呼叫 forward_uvp 時 output 為：
            output_phys = output_norm * std + mean
        """
        if not torch.is_tensor(mean):
            mean = torch.as_tensor(mean, dtype=torch.float32)
        if not torch.is_tensor(std):
            std = torch.as_tensor(std, dtype=torch.float32)
        if mean.shape != (3,) or std.shape != (3,):
            raise ValueError(
                f"mean/std 必須是 shape (3,)，收到 {tuple(mean.shape)} / {tuple(std.shape)}"
            )
        device = self.physics_output_mean.device
        self.physics_output_mean.copy_(mean.to(device=device, dtype=torch.float32))
        self.physics_output_std.copy_(std.to(device=device, dtype=torch.float32))

    def set_body_bc_scale(self, scale: float) -> None:
        """注入 hard body BC 的 distance scale。

        Args:
            scale: gate = (distance / scale).clamp(0, 1)；fluid 區典型 max distance 量級。
                   例如 cylinder normalized SDF 在 fluid 內 max ≈ 0.3 → scale=0.3 →
                   wake 大部分 gate ≈ 1，boundary layer 內漸進歸零。
        """
        if scale <= 0:
            raise ValueError(f"body_bc_scale 必須 > 0，收到 {scale}")
        self.query_decoder.body_bc_scale.fill_(float(scale))

    def encode(
        self,
        sensor_vals: torch.Tensor,
        sensor_pos: torch.Tensor,
        re_norm: float,
        sensor_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """What: 對 [T, K, C] 整段 sensor_vals 一次跑完 spatial encoder。

        Why: 原本 Python loop over T 每步 T×=200 個 kernel launch；spatial encoder
             所有層對 last-dim element-wise，T 軸僅是 batch，向量化後結果等價。
        """
        pos_enc = self.spatial_encoder.encode_pos(sensor_pos)
        spatial_states = self.spatial_encoder(sensor_vals, pos_enc)  # [T, K, d_model]
        return self.temporal_encoder(spatial_states, re_norm, sensor_time), sensor_time

    def update_state(
        self,
        sensor_vals_t: torch.Tensor,
        sensor_pos: torch.Tensor,
        re_norm: float,
        dt: float,
        h_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        pos_enc = self.spatial_encoder.encode_pos(sensor_pos)
        spatial = self.spatial_encoder(sensor_vals_t, pos_enc)
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
        use_locality_decay=bool(cfg.get("use_locality_decay", False)),
        use_bidirectional_cfc=bool(cfg.get("use_bidirectional_cfc", False)),
        fourier_embed_dim=int(cfg.get("fourier_embed_dim", 0)),
        use_periodic_domain=bool(cfg.get("use_periodic_domain", True)),
        cfc_log_tau_min=float(cfg.get("cfc_log_tau_min", -1.0)),
        cfc_log_tau_max=float(cfg.get("cfc_log_tau_max", 1.0)),
        fourier_sigma_bands=cfg.get("fourier_sigma_bands"),
        fourier_band_dim_ratios=cfg.get("fourier_band_dim_ratios"),
        use_hard_body_bc=bool(cfg.get("use_hard_body_bc", False)),
    )


def make_lnn_model_fn(
    net: LiquidOperator,
    sensor_vals: torch.Tensor,
    sensor_pos: torch.Tensor,
    re_norm: float,
    sensor_time: torch.Tensor,
    device: torch.device,
    h_states: torch.Tensor | None = None,
    s_time: torch.Tensor | None = None,
    body_distance_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Callable:
    """What: 建立物理 loss 所需的 closure。

    h_states/s_time 若已由外部計算（data loss 路徑），直接傳入可避免重複 encode。
    body_distance_fn: optional callable，接受 xy [N, 2] tensor 回傳 [N] distance tensor。
                     僅 use_hard_body_bc=True 時需要；必須是 differentiable
                     （cylinder.query_body_distance_torch 而非 numpy 版）。
    """
    net_device = next(iter(net.parameters())).device
    if h_states is None or s_time is None:
        h_states, s_time = net.encode(sensor_vals, sensor_pos, re_norm, sensor_time)
    use_bd = bool(getattr(net, "use_hard_body_bc", False))
    if use_bd and body_distance_fn is None:
        raise ValueError(
            "model use_hard_body_bc=True 但 make_lnn_model_fn() 沒收到 body_distance_fn"
        )

    def model_fn(xyt: torch.Tensor, c: int) -> torch.Tensor:
        xyt_d = xyt.to(net_device)
        xy_d = xyt_d[:, :2]
        t_q_d = xyt_d[:, 2]
        c_t = torch.full((xyt_d.shape[0],), c, dtype=torch.long, device=net_device)
        bd = body_distance_fn(xy_d) if use_bd else None
        return net.query_decoder(
            xy_d, t_q_d, c_t, h_states, s_time, sensor_pos, body_distance=bd,
        ).to(xyt.device)

    return model_fn


def make_lnn_model_fn_uvp(
    net: LiquidOperator,
    sensor_vals: torch.Tensor,
    sensor_pos: torch.Tensor,
    re_norm: float,
    sensor_time: torch.Tensor,
    device: torch.device,
    h_states: torch.Tensor | None = None,
    s_time: torch.Tensor | None = None,
    body_distance_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Callable:
    """What: 建立 physics path 用的 uvp closure，回傳 [N, 3] = (u, v, p)。

    Why: 取代 (u_fn, v_fn, p_fn) 三個獨立 closure。共用 c-independent 路徑且只
         保留 1 份共享 autograd graph，二階 backward 記憶體峰值同步下降。
    body_distance_fn: 同 make_lnn_model_fn；use_hard_body_bc=True 時必須 differentiable.
    """
    net_device = next(iter(net.parameters())).device
    if h_states is None or s_time is None:
        h_states, s_time = net.encode(sensor_vals, sensor_pos, re_norm, sensor_time)
    use_bd = bool(getattr(net, "use_hard_body_bc", False))
    if use_bd and body_distance_fn is None:
        raise ValueError(
            "model use_hard_body_bc=True 但 make_lnn_model_fn_uvp() 沒收到 body_distance_fn"
        )

    def model_fn_uvp(xyt: torch.Tensor) -> torch.Tensor:
        xyt_d = xyt.to(net_device)
        xy_d = xyt_d[:, :2]
        t_q_d = xyt_d[:, 2]
        bd = body_distance_fn(xy_d) if use_bd else None
        # raw model output：normalized 量級（mean~0, std~1 per channel）
        raw = net.query_decoder.forward_uvp(
            xy_d, t_q_d, h_states, s_time, sensor_pos, body_distance=bd,
        )
        # Denormalize 回物理單位：phys = raw * std + mean。
        # buffer 預設 (mean=0, std=1) → identity；training/evaluate 注入後生效。
        # 對 NS residual 必要：du/dt + u·∇u + ∇p − ν·∇²u 各項要在同一物理量級。
        phys = raw * net.physics_output_std + net.physics_output_mean
        return phys.to(xyt.device)

    return model_fn_uvp
