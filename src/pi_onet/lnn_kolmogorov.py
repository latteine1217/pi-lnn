# src/pi_onet/lnn_kolmogorov.py
"""Pi-LNN: Physics-informed Liquid Neural Network for Kolmogorov flow.

What: 以 CfC (Closed-form Continuous-time) 取代 LTC ODE 的數值求解器，
      實現 Spatial CfC Encoder + Temporal CfC Encoder + Query CfC Decoder。
Why:  CfC 閉合解 h_new = gate·f1 + (1-gate)·f2 在一步內模擬連續時間動態，
      不需要 sub-stepping，加速推論同時保留 LNN 的時序誘導偏差。
"""
from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn as nn
from deepxde import config as dde_config

from pi_onet.pit_ldc import (
    rff_encode,
    configure_torch_runtime,
    count_parameters,
    write_json,
    _grad,
)


class CfCCell(nn.Module):
    """What: CfC（Closed-form Continuous-time）遞迴單元，no-gate 模式。

    Why: 以閉合解取代 LTC ODE 數值積分：
         h_new = sigmoid(-t_a·Δt + t_b) · f1(x,h) + (1 - sigmoid(...)) · f2(x,h)
         使用真實物理 Δt，一步直達目標態，加速 LNN 的 ODE 計算。
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        """
        input_size:  維度 of input x（encoder 中 = d_model）
        hidden_size: 維度 of hidden state h（= d_model）
        combined:    input_size + hidden_size，作為 ff1/ff2/time_a/time_b 的輸入維度
        """
        super().__init__()
        self.hidden_size = hidden_size
        combined = input_size + hidden_size
        self.ff1 = nn.Linear(combined, hidden_size)
        self.ff2 = nn.Linear(combined, hidden_size)
        self.time_a = nn.Linear(combined, hidden_size)
        self.time_b = nn.Linear(combined, hidden_size)
        # Xavier init，確保初始 gate ≈ 0.5，避免 sigmoid 飽和
        for layer in (self.time_a, self.time_b):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        x: [..., input_size]
        h: [..., hidden_size]
        Returns: [..., hidden_size]
        """
        xh = torch.cat([x, h], dim=-1)
        f1 = torch.tanh(self.ff1(xh))
        f2 = torch.tanh(self.ff2(xh))
        t_a = self.time_a(xh)
        t_b = self.time_b(xh)
        gate = torch.sigmoid(-t_a * dt + t_b)
        return gate * f1 + (1.0 - gate) * f2


class SpatialCfCEncoder(nn.Module):
    """What: 在單一時間步內，以 CfC 序列處理 K 個感測器 → 空間摘要向量 s_t。

    Why: 每個感測器 token 為 [RFF(x,y), u, v, p]，CfC 序列掃描取代空間注意力；
         Δt=1.0（感測器無自然空間時序，RFF 已負責空間資訊）。
    """

    def __init__(self, rff_features: int, d_model: int, num_layers: int) -> None:
        super().__init__()
        sensor_in = 2 * rff_features + 3  # RFF(x,y) + u,v,p
        self.proj = nn.Linear(sensor_in, d_model)
        self.cells = nn.ModuleList([
            CfCCell(d_model, d_model) for _ in range(num_layers)
        ])

    def forward(
        self,
        sensor_vals: torch.Tensor,
        sensor_pos: torch.Tensor,
        B: torch.Tensor,
    ) -> torch.Tensor:
        """
        sensor_vals: [K, 3]  (u, v, p)
        sensor_pos:  [K, 2]  (x, y) in [0, 2π]
        B:           [2, rff_features]
        Returns:     [d_model]
        """
        rff = rff_encode(sensor_pos, B)                              # [K, 2*rff_features]
        seq = self.proj(torch.cat([rff, sensor_vals], dim=-1))       # [K, d_model]

        for cell in self.cells:
            h = torch.zeros(cell.hidden_size, device=seq.device, dtype=seq.dtype)
            new_seq = []
            for k in range(seq.shape[0]):
                h = cell(seq[k], h, dt=1.0)
                new_seq.append(h)
            seq = torch.stack(new_seq)   # [K, d_model] — 作為下一層輸入

        return seq[-1]   # 最終隱藏態 [d_model]


class TemporalCfCEncoder(nn.Module):
    """What: 以 CfC 序列處理 T 個空間摘要向量 → 時序編碼 h_enc。

    Why: 使用物理 Δt（= 1.0 感測器時間單位），CfC 在此扮演 LTC ODE 的
         閉合解角色：h_enc 捕捉全部 T 步的時序動態，一步等同 RK4 多步積分。
         Re 以持續殘差（re_bias）方式加入每步輸入，確保 Re 資訊貫穿整個時序演化，
         避免初始隱藏態影響在長序列中被沖洗掉。
    """

    def __init__(self, d_model: int, num_layers: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.re_proj = nn.Linear(1, d_model)
        self.cells = nn.ModuleList([
            CfCCell(d_model, d_model) for _ in range(num_layers)
        ])

    def forward(
        self,
        spatial_states: torch.Tensor,
        re_norm: float,
        dt_phys: float,
    ) -> torch.Tensor:
        """
        spatial_states: [T, d_model]
        re_norm:        float（正規化 Re 值）
        dt_phys:        float（物理時間步長，= 1.0 for sensor data）
        Returns:        [d_model]
        """
        re_t = torch.tensor(
            [[re_norm]], dtype=spatial_states.dtype, device=spatial_states.device
        )
        re_bias = self.re_proj(re_t).squeeze(0)   # [d_model]
        # Re 以殘差方式加入每個時間步的輸入，確保 Re 資訊貫穿整個序列
        seq = spatial_states + re_bias.unsqueeze(0)   # [T, d_model]

        for cell in self.cells:
            h = torch.zeros(self.d_model, device=seq.device, dtype=seq.dtype)
            new_seq = []
            for t in range(seq.shape[0]):
                h = cell(seq[t], h, dt=dt_phys)
                new_seq.append(h)
            seq = torch.stack(new_seq)   # [T, d_model]

        return seq[-1]   # h_enc [d_model]


class QueryCfCDecoder(nn.Module):
    """What: 以單步向量化 CfC 將 query (x,y,t,c) 解碼為 u/v/p。

    Why: h_enc 廣播至所有 N_q query 點作為初始隱藏態；CfCCell 以 batch
         [N_q, d_model] 一次運行，完全向量化，等同矩陣運算，無 for-loop。
    """

    def __init__(self, rff_features: int, d_model: int, d_time: int) -> None:
        super().__init__()
        query_in = 2 * rff_features + d_time + 8   # RFF(x,y) + time_enc + comp_emb
        self.time_proj = nn.Linear(1, d_time)
        self.component_emb = nn.Embedding(3, 8)
        nn.init.normal_(self.component_emb.weight, mean=0.0, std=0.1)
        self.query_proj = nn.Linear(query_in, d_model)
        self.cell = CfCCell(d_model, d_model)
        self.output_head = nn.Linear(d_model, 1, bias=True)
        self.component_scale = nn.Parameter(torch.ones(3))
        self.component_bias = nn.Parameter(torch.zeros(3))

    def forward(
        self,
        xy: torch.Tensor,
        t_q: torch.Tensor,
        c: torch.Tensor,
        h_enc: torch.Tensor,
        B: torch.Tensor,
    ) -> torch.Tensor:
        """
        xy:    [N_q, 2]
        t_q:   [N_q]
        c:     [N_q] long
        h_enc: [d_model]
        B:     [2, rff_features]
        Returns: [N_q, 1]
        """
        rff_q = rff_encode(xy, B)                                     # [N_q, 2*rff_f]
        time_e = self.time_proj(t_q.unsqueeze(-1))                    # [N_q, d_time]
        emb_c = self.component_emb(c)                                 # [N_q, 8]
        q = self.query_proj(torch.cat([rff_q, time_e, emb_c], dim=-1))  # [N_q, d_model]

        # h_enc: [d_model] → unsqueeze → [1, d_model] → expand → [N_q, d_model]
        # 單步向量化 CfC，無 for-loop，無 nn.MultiheadAttention
        h_0 = h_enc.unsqueeze(0).expand(q.shape[0], -1).contiguous()
        H_dec = self.cell(q, h_0, dt=1.0)                            # [N_q, d_model]

        out = self.output_head(H_dec)                                 # [N_q, 1]
        out = out * self.component_scale[c].unsqueeze(1) + self.component_bias[c].unsqueeze(1)
        return out


class LiquidOperator(nn.Module):
    """What: Pi-LNN 主模型——組合 SpatialCfCEncoder + TemporalCfCEncoder + QueryCfCDecoder。

    Why: 完整的 LNN 架構，無任何 MultiheadAttention 或 TransformerEncoder。
         CfC 在 Temporal Encoder 中使用物理 Δt 加速 LTC ODE 計算。
    """

    def __init__(
        self,
        rff_features: int,
        rff_sigma: float,
        d_model: int,
        d_time: int,
        num_spatial_cfc_layers: int,
        num_temporal_cfc_layers: int,
    ) -> None:
        super().__init__()
        B = torch.randn(2, rff_features) * rff_sigma
        self.register_buffer("B", B)
        self.spatial_encoder = SpatialCfCEncoder(
            rff_features=rff_features, d_model=d_model, num_layers=num_spatial_cfc_layers
        )
        self.temporal_encoder = TemporalCfCEncoder(
            d_model=d_model, num_layers=num_temporal_cfc_layers
        )
        self.query_decoder = QueryCfCDecoder(
            rff_features=rff_features, d_model=d_model, d_time=d_time
        )

    def encode(
        self,
        sensor_vals: torch.Tensor,
        sensor_pos: torch.Tensor,
        re_norm: float,
        dt_phys: float,
    ) -> torch.Tensor:
        """
        sensor_vals: [T, K, 3]
        sensor_pos:  [K, 2]
        Returns:     h_enc [d_model]
        """
        spatial_states = torch.stack([
            self.spatial_encoder(sensor_vals[t], sensor_pos, self.B)
            for t in range(sensor_vals.shape[0])
        ])   # [T, d_model]
        return self.temporal_encoder(spatial_states, re_norm, dt_phys)

    def forward(
        self,
        sensor_vals: torch.Tensor,
        sensor_pos: torch.Tensor,
        re_norm: float,
        dt_phys: float,
        xy: torch.Tensor,
        t_q: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Returns: [N_q, 1]"""
        h_enc = self.encode(sensor_vals, sensor_pos, re_norm, dt_phys)
        return self.query_decoder(xy, t_q, c, h_enc, self.B)


def create_lnn_model(cfg: dict) -> LiquidOperator:
    """What: 從 config dict 建立 LiquidOperator。"""
    return LiquidOperator(
        rff_features=int(cfg["rff_features"]),
        rff_sigma=float(cfg["rff_sigma"]),
        d_model=int(cfg["d_model"]),
        d_time=int(cfg["d_time"]),
        num_spatial_cfc_layers=int(cfg["num_spatial_cfc_layers"]),
        num_temporal_cfc_layers=int(cfg["num_temporal_cfc_layers"]),
    )


def unsteady_ns_residuals(
    u_fn: Callable,
    v_fn: Callable,
    p_fn: Callable,
    xyt: torch.Tensor,
    re: float,
    k_f: float = 4.0,
    A:   float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """What: 2D incompressible unsteady NS + continuity residuals at collocation points.

    Why: Kolmogorov flow 加入體積力 f_x = A·sin(k_f·y)（正弦強迫），
         ∂u/∂t 項是與 LDC steady-state 的關鍵差異。
    xyt: [N, 3] = (x, y, t) with requires_grad=True
    Returns: ns_x [N,1], ns_y [N,1], cont [N,1]
    """
    u, v, p = u_fn(xyt), v_fn(xyt), p_fn(xyt)
    u_xyt = _grad(u, xyt)
    v_xyt = _grad(v, xyt)
    p_xyt = _grad(p, xyt)
    du_dx, du_dy, du_dt = u_xyt[:, 0:1], u_xyt[:, 1:2], u_xyt[:, 2:3]
    dv_dx, dv_dy, dv_dt = v_xyt[:, 0:1], v_xyt[:, 1:2], v_xyt[:, 2:3]
    dp_dx, dp_dy         = p_xyt[:, 0:1], p_xyt[:, 1:2]
    du_dx2 = _grad(du_dx, xyt)[:, 0:1]
    du_dy2 = _grad(du_dy, xyt)[:, 1:2]
    dv_dx2 = _grad(dv_dx, xyt)[:, 0:1]
    dv_dy2 = _grad(dv_dy, xyt)[:, 1:2]
    nu  = 1.0 / float(re)
    f_x = A * torch.sin(k_f * xyt[:, 1:2])   # Kolmogorov forcing
    ns_x = du_dt + u * du_dx + v * du_dy + dp_dx - nu * (du_dx2 + du_dy2) - f_x
    ns_y = dv_dt + u * dv_dx + v * dv_dy + dp_dy - nu * (dv_dx2 + dv_dy2)
    cont = du_dx + dv_dy
    return ns_x, ns_y, cont


def make_lnn_model_fn(
    net: LiquidOperator,
    sensor_vals: torch.Tensor,
    sensor_pos: torch.Tensor,
    re_norm: float,
    dt_phys: float,
    device: torch.device,
) -> Callable:
    """What: 回傳 closure (xyt, c) → [N,1]，供物理損失計算使用。

    Why: 物理損失對 xyt 做 autograd；closure 捕捉 sensor 資料與 Re 條件。
         h_enc 可在計算所有 component 前 encode 一次，節省重複計算。
    """
    net_device = next(iter(net.buffers())).device

    def model_fn(xyt: torch.Tensor, c: int) -> torch.Tensor:
        xyt_d = xyt.to(net_device)
        xy_d  = xyt_d[:, :2]
        t_q_d = xyt_d[:, 2]
        h_enc = net.encode(sensor_vals, sensor_pos, re_norm, dt_phys)
        c_t   = torch.full((xyt_d.shape[0],), c, dtype=torch.long, device=net_device)
        return net.query_decoder(xy_d, t_q_d, c_t, h_enc, net.B).to(xyt.device)

    return model_fn


DEFAULT_LNN_ARGS: dict[str, Any] = {
    "sensor_jsons": None,
    "sensor_npzs": None,
    "les_paths": None,
    "re_values": None,
    "rff_features": 64,
    "rff_sigma": 2.0,
    "d_model": 128,
    "d_time": 16,
    "num_spatial_cfc_layers": 2,
    "num_temporal_cfc_layers": 2,
    "data_loss_weight": 1.0,
    "physics_loss_weight": 0.05,
    "continuity_weight": 1.0,
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
    """What: 載入並驗證 TOML config。舊 PiT 欄位（nhead 等）會觸發明確錯誤。"""
    if config_path is None:
        return {}
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config_data = payload.get("train", payload)
    normalized = dict(config_data)
    # 舊 PiT 欄位 → 明確失敗（不靜默忽略）
    obsolete = sorted(set(normalized) & _REMOVED_KEYS)
    if obsolete:
        raise ValueError(
            f"Config 含有已移除的 PiT 欄位（請改用 num_spatial/temporal_cfc_layers）: {obsolete}"
        )
    unknown = sorted(set(normalized) - set(DEFAULT_LNN_ARGS))
    if unknown:
        raise ValueError(f"LNN config 含有不支援的欄位: {unknown}")
    # 解析相對路徑
    for list_key in ("sensor_jsons", "sensor_npzs", "les_paths"):
        if list_key in normalized:
            normalized[list_key] = [
                str((config_path.parent / Path(p)).resolve())
                for p in normalized[list_key]
            ]
    if "artifacts_dir" in normalized:
        normalized["artifacts_dir"] = str(
            (config_path.parent / Path(normalized["artifacts_dir"])).resolve()
        )
    return normalized
