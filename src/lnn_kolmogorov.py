# src/lnn_kolmogorov.py
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


class GradNormWeights(nn.Module):
    """What: GradNorm（Chen et al., 2018）的可學習 task 權重。

    Why: 直接管理 [data, ns_u, ns_v, cont] 四個 task 的權重比例。
         以 w_data = 1 為基準，physics weights 表達相對 data 的比例，
         讓 GradNorm 能真正動態調整 physics/data 平衡，而非受限於 sum=const 的固定比例。
         初始值 [1.0, 0.01, 0.01, 0.01] → physics 從 1% data 出發，由 GradNorm 自行決定是否加強。
    """

    def __init__(self, init_weights: list[float]) -> None:
        super().__init__()
        w = torch.tensor(init_weights, dtype=torch.float32)
        self.log_weights = nn.Parameter(torch.log(w))

    @property
    def weights(self) -> torch.Tensor:
        return torch.exp(self.log_weights)

    def normalize_to_data_(self) -> None:
        """固定 w_data = 1，其餘 task 表示相對 data 的比例。

        Why: 取代 sum=const 的歸一化，避免 data weight 膨脹導致 physics weights
             被壓縮到微小量級，使 GradNorm 真正能改變 physics/data 的比例關係。
        """
        with torch.no_grad():
            self.log_weights -= self.log_weights[0].clone()


def _gradnorm_step(
    gn_weights: GradNormWeights,
    losses: list[torch.Tensor],
    ref_params: list[torch.Tensor],
    ema_momentum: float = 0.5,
) -> None:
    """What: 一次 GradNorm 權重更新（直接公式 + EMA，無 optimizer）。

    Why: 各 task 的目標權重直接由梯度範數反比公式算出，再以 EMA 平滑寫回。
         不需要 create_graph=True 或獨立 optimizer，計算成本低且無 lr 調參問題。
         有效步長 = (1 - ema_momentum)，momentum=0.5 → 每次更新走 50%。

    公式：
        G_i      = ||∇_W L_i||_2          （各 task 對 ref_params 的梯度範數）
        mean_G   = mean(G_i)
        w_i_raw  = mean_G / (G_i + 1e-5 * mean_G)   （梯度範數小 → 權重大）
        w_i_norm = w_i_raw / w_i_raw[0]              （data 為基準，w_data = 1）
        w_new    = momentum * w_old + (1 - momentum) * w_i_norm

    Args:
        losses:        [l_data, l_ns_u, l_ns_v, l_cont]（retain_graph=True 保留計算圖）
        ref_params:    reference layer 參數（trunk_out.weight + bias）
        ema_momentum:  EMA 動量；有效步長 = 1 - ema_momentum
    """
    ws_old = gn_weights.weights.detach().clone()

    # 計算各 task 對 ref_params 的梯度範數（不需要 create_graph）
    G = []
    for l_i in losses:
        grads = torch.autograd.grad(
            l_i, ref_params,
            retain_graph=True, create_graph=False, allow_unused=True,
        )
        g_norms = [g.reshape(-1).norm() for g in grads if g is not None]
        G.append(torch.stack(g_norms).norm() if g_norms else torch.zeros(1, device=ws_old.device).squeeze())

    G_stack = torch.stack(G).detach()
    mean_G = G_stack.mean()

    # 目標權重：梯度範數越小 → 權重越大（讓各 task 梯度貢獻拉齊）
    w_raw = mean_G / (G_stack + 1e-5 * mean_G)
    # 以 data 為基準歸一化：w_data = 1，physics 表達相對比例
    w_computed = w_raw / w_raw[0].clamp(min=1e-8)

    # EMA：new = momentum * old + (1 - momentum) * computed
    w_new = ema_momentum * ws_old + (1.0 - ema_momentum) * w_computed
    with torch.no_grad():
        gn_weights.log_weights.copy_(torch.log(w_new.clamp(min=1e-8)))


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
        use_locality_decay: bool = False,
        use_cfc_freerun: bool = False,
    ) -> None:
        super().__init__()
        self.use_locality_decay = bool(use_locality_decay)
        self.use_cfc_freerun = bool(use_cfc_freerun)
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
        # Method B: sensor_time → t_q 之間的 CfC 自由積分。
        # Why: 原始 time_proj 只做線性映射，無法捕捉 CfC 的非線性動態。
        #      使用 CfC cell 做自由積分（x = h，無外部激勵），讓 branch token
        #      在跨 sensor 時間間距時的動態更接近真實連續演化。
        #      use_cfc_freerun=False 時跳過，確保舊 checkpoint 可無損載入。
        self.d_model = d_model
        if self.use_cfc_freerun:
            self.freerun_cell = CfCCell(d_model, d_model)
        if self.use_locality_decay:
            # log_locality_decay: α = exp(log_locality_decay) 為衰減率（α > 0）。
            # Why: 在 softmax 前加入 log-space 距離懲罰 score += -α * r，
            #      等效於對每個 sensor token 的 attention weight 乘以 exp(-α * r)。
            #      初始 log_locality_decay = -2.0 → α ≈ 0.135，r=0.1 時懲罰 ≈ -0.013，
            #      接近中性，讓模型自行學習是否需要近鄰優先。
            self.log_locality_decay = nn.Parameter(torch.tensor([-2.0], dtype=torch.float32))

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

        if self.use_cfc_freerun:
            # Method B: 以 CfC 將 h_branch_tokens 從 sensor_time[idx] 自由積分至 t_q。
            # h_branch_tokens: [N, K, D]；dt_to_query: [N]
            # 自由積分不使用外部激勵（x = h），模擬感測器觀測之間的自主動態。
            _N, _K, _D = h_branch_tokens.shape
            _h_flat = h_branch_tokens.reshape(_N * _K, _D)
            _dt_flat = dt_to_query.unsqueeze(1).expand(_N, _K).reshape(_N * _K)
            h_branch_tokens = self.freerun_cell(_h_flat, _h_flat, dt=_dt_flat).view(_N, _K, _D)

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
        use_cfc_freerun: bool = False,
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
            use_locality_decay=use_locality_decay,
            use_cfc_freerun=use_cfc_freerun,
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
        use_locality_decay=bool(cfg.get("use_locality_decay", False)),
        use_cfc_freerun=bool(cfg.get("use_cfc_freerun", False)),
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


def pressure_poisson_residual(
    u_fn: Callable,
    v_fn: Callable,
    p_fn: Callable,
    xyt: torch.Tensor,
) -> torch.Tensor:
    """What: 2D incompressible 壓力 Poisson 方程殘差。

    Why: Primitive-variable NS 中 p 沒有資料監督，模型可藉由任意調整 p 來讓
         momentum residual 歸零，即使 u, v 是錯的（壓力自由度問題）。
         Poisson 方程從 NS + ∇·u=0 推導而來：
             ∇²p = -(∂u/∂x)² - (∂v/∂y)² - 2(∂u/∂y)(∂v/∂x)
         加入此約束後，p 必須與 u, v 的二階結構一致，壓力不再能自由漂移。

    數學推導：對動量方程取散度，使用 ∇·u=0 及 Kolmogorov forcing ∇·f=0 後得到。
    不需要任何額外觀測量，僅用模型輸出的 u, v, p via autograd。
    """
    u, v, p = u_fn(xyt), v_fn(xyt), p_fn(xyt)
    u_xyt = _grad(u, xyt)
    v_xyt = _grad(v, xyt)
    p_xyt = _grad(p, xyt)
    du_dx = u_xyt[:, 0:1]
    du_dy = u_xyt[:, 1:2]
    dv_dx = v_xyt[:, 0:1]
    dv_dy = v_xyt[:, 1:2]
    dp_dx = p_xyt[:, 0:1]
    dp_dy = p_xyt[:, 1:2]
    dp_dx2 = _grad(dp_dx, xyt)[:, 0:1]   # ∂²p/∂x²
    dp_dy2 = _grad(dp_dy, xyt)[:, 1:2]   # ∂²p/∂y²
    laplacian_p = dp_dx2 + dp_dy2
    rhs = -(du_dx ** 2 + dv_dy ** 2 + 2.0 * du_dy * dv_dx)
    return laplacian_p - rhs


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


def _rar_update_pool(
    net,
    datasets,
    sensor_vals_list: list,
    sensor_pos_list: list,
    sensor_time_list: list,
    rng: np.random.Generator,
    n_select: int,
    pool_size: int,
    t_max: float | None,
    k_f: float,
    A: float,
    domain_length: float,
    device,
    exploration_ratio: float = 0.2,
) -> list[np.ndarray]:
    """What: RAR（Residual Adaptive Refinement）pool 更新。

    Why: 均勻隨機採樣可能長期錯過 t≈0 等高殘差區域；
         每隔 rar_update_freq 步，從大候選集中選 top 殘差點，
         讓 physics loss 集中在模型最難收斂的區域。
         保留 exploration_ratio 比例的隨機點，防止采样退化到固定幾個點。

    近似殘差：略去黏性項（Re=10000 時 ν=1e-4，黏性貢獻約 0.2%）；
    使用 create_graph=False，僅計算一階導數，完全避免二階 autograd 建圖。
    此近似只影響 pool 中的點排序，不影響訓練 loss 本身的精確性。

    Returns:
        list of (n_select, 3) float32 numpy arrays, one per dataset.
    """
    n_top = max(1, round(n_select * (1.0 - exploration_ratio)))
    n_rand = n_select - n_top
    kw = 2.0 * torch.pi * float(k_f) / float(domain_length)
    result = []
    net.eval()
    for i, ds in enumerate(datasets):
        xy_np, t_np = ds.sample_physics_points(rng, n=pool_size, t_max=t_max, strategy="random")
        xyt_pool = torch.tensor(
            np.concatenate([xy_np, t_np[:, None]], axis=1),
            dtype=torch.float32, device=device, requires_grad=True,
        )
        model_fn = make_lnn_model_fn(
            net, sensor_vals_list[i], sensor_pos_list[i],
            re_norm=ds.re_norm, sensor_time=sensor_time_list[i], device=device,
        )
        u = model_fn(xyt_pool, c=0)
        v = model_fn(xyt_pool, c=1)
        p = model_fn(xyt_pool, c=2)

        def _g1(y: torch.Tensor) -> torch.Tensor:
            g = torch.autograd.grad(
                y, xyt_pool, torch.ones_like(y),
                create_graph=False, allow_unused=True,
            )[0]
            return g if g is not None else torch.zeros_like(xyt_pool)

        u_xyt = _g1(u)
        v_xyt = _g1(v)
        p_xyt = _g1(p)

        du_dx = u_xyt[:, 0:1]; du_dy = u_xyt[:, 1:2]; du_dt = u_xyt[:, 2:3]
        dv_dx = v_xyt[:, 0:1]; dv_dy = v_xyt[:, 1:2]; dv_dt = v_xyt[:, 2:3]
        dp_dx = p_xyt[:, 0:1]; dp_dy = p_xyt[:, 1:2]

        forcing = float(A) * torch.sin(kw * xyt_pool[:, 1:2]).detach()
        mom_u = du_dt + u.detach() * du_dx + v.detach() * du_dy + dp_dx - forcing
        mom_v = dv_dt + u.detach() * dv_dx + v.detach() * dv_dy + dp_dy
        cont  = du_dx + dv_dy

        res_mag = (mom_u.detach() ** 2 + mom_v.detach() ** 2 + cont.detach() ** 2).squeeze(-1)
        _, top_idx = torch.topk(res_mag, n_top)
        selected = xyt_pool[top_idx].detach().cpu().numpy()
        if n_rand > 0:
            rxy, rt = ds.sample_physics_points(rng, n=n_rand, t_max=t_max, strategy="random")
            selected = np.concatenate([selected, np.concatenate([rxy, rt[:, None]], axis=1)], axis=0)
        result.append(selected.astype(np.float32))
    net.train()
    return result


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


def physics_points_at_step(
    step: int,
    start: int,
    end: int,
    ramp_steps: int,
    warmup_steps: int = 0,
) -> int:
    """What: 依訓練步數線性增加 physics collocation 點數（curriculum）。

    Why: 訓練初期 data loss 未收斂，大量 physics 點造成梯度衝突；
         先等 warmup_steps 讓模型有基本擬合，再花 ramp_steps 步逐步增加點數。

    Args:
        step:          當前步數（從 1 開始）
        start:         初始點數（warmup 期間及 ramp 起始值）
        end:           最終點數
        ramp_steps:    從 start 線性增長至 end 所需步數；0 = warmup 後立即用 end
        warmup_steps:  開始 ramp 前的等待步數（此期間固定用 start）
    Returns:
        當前步數對應的整數點數
    """
    if step <= warmup_steps:
        return start
    if ramp_steps <= 0:
        return end
    progress = min((step - warmup_steps) / ramp_steps, 1.0)
    return int(round(start + (end - start) * progress))


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
    "use_locality_decay": False,
    "use_cfc_freerun": False,
    "data_loss_weight": 1.0,
    "t_early_weight": 1.0,       # t <= t_early_threshold 的 data loss 乘數（1.0 = 無加權）
    "t_early_threshold": 0.1,    # 早期時間定義上限
    "lbfgs_max_iter": 20,        # L-BFGS 每步最大 line-search 次數
    "lbfgs_history_size": 10,    # L-BFGS curvature history buffer 大小
    "physics_loss_weight": 0.01,
    "physics_loss_warmup_steps": 0,
    "physics_loss_ramp_steps": 0,
    "continuity_weight": 1.0,
    "use_gradnorm": False,
    "gradnorm_alpha": 1.5,   # 已棄用，保留供舊 config 相容
    "gradnorm_lr": 1e-3,     # 已棄用，保留供舊 config 相容
    "gradnorm_update_freq": 10,
    "gradnorm_init_weights": [1.0, 0.01, 0.01, 0.01],
    "gradnorm_ema_momentum": 0.9,
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
    "num_query_points": 0,        # 0 = 由 sensor K 自動決定；正整數 = override
    "num_physics_points": 32,     # 最終（最大）collocation 點數
    "num_physics_points_start": 0,         # curriculum 初始點數；0 = 與 num_physics_points 相同（固定）
    "num_physics_points_warmup_steps": 0,  # ramp 開始前的等待步數
    "num_physics_points_ramp_steps": 0,    # 線性增長步數；0 = warmup 後立即使用最終值
    "physics_collocation_strategy": "random",
    "rar_update_freq": 50,         # RAR: 每幾步重新評估 residual pool
    "rar_pool_multiplier": 10,     # RAR: pool 大小 = num_physics_points × multiplier
    "rar_exploration_ratio": 0.2,  # RAR: 保留隨機點比例（防 mode collapse）
    "physics_residual_normalize": False,
    "poisson_loss_weight": 0.0,
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


def _find_project_root(start: Path) -> Path | None:
    """What: 從指定路徑向上尋找專案根目錄。"""
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return None


def _resolve_config_path_value(raw_path: str | Path, config_path: Path) -> str:
    """What: 以相容既有 workflow 的方式解析 config 內路徑。

    Why: 目前 repo 同時存在兩種寫法：
         1) 相對專案根目錄（如 `data/...`）
         2) 相對 config 檔位置（外部/臨時 config 常見）
         若只支援其中一種會直接破壞 userspace。
    """
    path = Path(raw_path)
    if path.is_absolute():
        return str(path.resolve())

    config_dir = config_path.parent
    project_root = _find_project_root(config_dir)
    candidates = [
        config_dir / path,
        *( [project_root / path] if project_root is not None else [] ),
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    # 新路徑尚未建立時，優先用 project_root；若找不到則退回 config_dir
    if project_root is not None:
        return str((project_root / path).resolve())
    return str(candidates[0].resolve())


def load_lnn_config(config_path: Path | None) -> dict[str, Any]:
    """What: 載入並驗證核心 LNN config。"""
    if config_path is None:
        return {}
    config_path = Path(config_path).resolve()
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
    for list_key in ("sensor_jsons", "sensor_npzs", "dns_paths"):
        if list_key in normalized:
            normalized[list_key] = [_resolve_config_path_value(p, config_path) for p in normalized[list_key]]
    if "artifacts_dir" in normalized:
        normalized["artifacts_dir"] = _resolve_config_path_value(normalized["artifacts_dir"], config_path)
    return normalized


def train_lnn_kolmogorov(
    args: dict[str, Any],
    log_fn: Callable[[int, dict[str, float]], None] | None = None,
) -> None:
    """What: 核心 Pi-LNN 訓練迴圈。

    Args:
        args: 訓練設定字典（見 DEFAULT_LNN_ARGS）。
        log_fn: 可選回呼，每個訓練 step 結束後以 (step, metrics_dict) 呼叫。
                metrics_dict 包含 l_data / l_physics / l_ns / l_cont / w_phys / t_max。
                Why: 保持 core module 不依賴外部日誌框架（W&B、TensorBoard 等），
                     由呼叫方（如 sweep 腳本）注入觀測邏輯。
    """
    from kolmogorov_dataset import KolmogorovDataset

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
    is_lbfgs = args["lr_schedule"] == "lbfgs"

    if args["lr_schedule"] == "soap":
        import sys
        _soap_dir = str(Path(__file__).parent.parent / "SOAP")
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

    elif is_lbfgs:
        optimizer = torch.optim.LBFGS(
            net.parameters(),
            lr=float(args.get("learning_rate", 1.0)),
            max_iter=int(args.get("lbfgs_max_iter", 20)),
            history_size=int(args.get("lbfgs_history_size", 10)),
            line_search_fn="strong_wolfe",
        )
        scheduler = None

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
            # L-BFGS 與 checkpoint 的 optimizer 類型不同，跳過 optimizer state 載入。
            if not is_lbfgs and "optimizer_state_dict" in ckpt:
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

    # GradNorm setup
    use_gradnorm = bool(args.get("use_gradnorm", False))
    gn_weights: GradNormWeights | None = None
    gn_ref_params: list[torch.Tensor] = []
    gn_update_freq = int(args.get("gradnorm_update_freq", 200))
    if use_gradnorm:
        gn_weights = GradNormWeights(
            init_weights=args.get("gradnorm_init_weights", [1.0, 0.01, 0.01, 0.01])
        ).to(device)
        gn_ref_params = list(net.query_decoder.trunk_out.parameters())

    print("=== Training ===")
    if use_tm:
        print(f"  time_marching: t [{tm_t_start:.1f} → {tm_t_end:.1f}]  warmup={tm_warmup} steps")
    if use_gradnorm:
        init_w = args.get("gradnorm_init_weights", [1.0, 0.01, 0.01, 0.01])
        print(f"  GradNorm: momentum={args.get('gradnorm_ema_momentum', 0.5):.2f}  freq={gn_update_freq}  (direct formula + EMA)")
        print(f"  GradNorm init_weights: {init_w}  (4 tasks: data, ns_u, ns_v, cont)")
    elif phys_warmup_steps > 0 or phys_ramp_steps > 0:
        print(
            "  physics_ramp:"
            f" warmup={phys_warmup_steps} steps,"
            f" ramp={phys_ramp_steps} steps,"
            f" final_weight={base_phys_weight:.4f}"
        )
    if use_gradnorm:
        print(f"{'Step':<8} {'L_data':>12} {'L_phys':>12} {'w_ns_u':>8} {'w_ns_v':>8} {'w_cont':>8} {'L_total':>12}"
              + ("  t_max" if use_tm else ""))
    else:
        print(f"{'Step':<8} {'L_data':>12} {'L_phys':>12} {'w_phys':>10} {'L_total':>12}"
              + ("  t_max" if use_tm else ""))

    _is_rar = str(args.get("physics_collocation_strategy", "random")) == "rar"
    _rar_pool_np: list[np.ndarray] | None = None
    _rar_update_freq = int(args.get("rar_update_freq", 50))
    _rar_pool_mult   = int(args.get("rar_pool_multiplier", 10))
    _rar_expl_ratio  = float(args.get("rar_exploration_ratio", 0.2))

    for step in range(start_step + 1, args["iterations"] + 1):
        if use_tm:
            progress = min(step / max(tm_warmup, 1), 1.0)
            t_max: float | None = tm_t_start + (tm_t_end - tm_t_start) * progress
        else:
            t_max = None

        if use_schedulefree:
            optimizer.train()
        net.train()

        # ── L-BFGS path ──────────────────────────────────────────────────────
        if is_lbfgs:
            # 採樣一次：closure 被 line-search 多次呼叫時重用同一批資料。
            _phys_weight = physics_weight_at_step(
                step=step,
                final_weight=base_phys_weight,
                warmup_steps=phys_warmup_steps,
                ramp_steps=phys_ramp_steps,
            )
            _n_phys_end   = int(args["num_physics_points"])
            _n_phys_start = int(args.get("num_physics_points_start", 0)) or _n_phys_end
            _n_phys_wu    = int(args.get("num_physics_points_warmup_steps", 0))
            _n_phys_ramp  = int(args.get("num_physics_points_ramp_steps", 0))
            _n_phys = physics_points_at_step(step, _n_phys_start, _n_phys_end, _n_phys_ramp, _n_phys_wu)
            _phys_gate = _phys_weight > 0.0 and _n_phys_end > 0
            _phys_strategy = str(args.get("physics_collocation_strategy", "random"))
            _phys_normalize = bool(args.get("physics_residual_normalize", False))
            _poisson_weight = float(args.get("poisson_loss_weight", 0.0))

            _fixed_data: list = []
            for i, ds in enumerate(datasets):
                n_q = int(args.get("num_query_points", 0)) or ds.sensor_pos.shape[0]
                xy_np, t_np, c_np, ref_np = ds.sample_sensor_batch(rng, n=n_q, t_max=t_max)
                _fixed_data.append((
                    torch.tensor(xy_np, dtype=torch.float32, device=device),
                    torch.tensor(t_np, device=device),
                    torch.tensor(c_np, dtype=torch.long, device=device),
                    torch.tensor(ref_np, device=device),
                ))

            _fixed_phys: list = []
            if _phys_gate:
                for i, ds in enumerate(datasets):
                    xy_np, t_np = ds.sample_physics_points(
                        rng, n=_n_phys, t_max=t_max, strategy=_phys_strategy
                    )
                    _fixed_phys.append(torch.tensor(
                        np.concatenate([xy_np, t_np[:, None]], axis=1),
                        dtype=torch.float32, device=device, requires_grad=True,
                    ))

            _lbfgs_info: dict = {}

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                net.train()
                _ld = torch.zeros(1, device=device)
                for _i, _ds in enumerate(datasets):
                    _xy, _tq, _c, _ref = _fixed_data[_i]
                    _h, _st = net.encode(
                        sensor_vals_list[_i], sensor_pos_list[_i], _ds.re_norm, sensor_time_list[_i]
                    )
                    _pred = observed_channel_prediction(
                        net=net, xy=_xy, t_q=_tq, c_obs=_c,
                        observed_channel_names=_ds.observed_channel_names,
                        observed_channel_mean=observed_mean_list[_i],
                        observed_channel_std=observed_std_list[_i],
                        h_states=_h, s_time=_st, sensor_pos=sensor_pos_list[_i],
                    )
                    _ld = _ld + ((_pred - _ref) ** 2).mean()
                _ld = _ld / num_re

                _lp = torch.zeros(1, device=device)
                _lcont = torch.zeros(1, device=device)
                if _fixed_phys:
                    net.eval()
                    for _i, _ds in enumerate(datasets):
                        _xyt = _fixed_phys[_i]
                        _mfn = make_lnn_model_fn(
                            net, sensor_vals_list[_i], sensor_pos_list[_i],
                            re_norm=_ds.re_norm, sensor_time=sensor_time_list[_i], device=device,
                        )
                        _uf = lambda x, fn=_mfn: fn(x, c=0)
                        _vf = lambda x, fn=_mfn: fn(x, c=1)
                        _pf = lambda x, fn=_mfn: fn(x, c=2)
                        _mu, _mv, _co = unsteady_ns_residuals(
                            _uf, _vf, _pf, _xyt,
                            re=_ds.re_value, k_f=k_f, A=A, domain_length=domain_length,
                        )
                        if _phys_normalize:
                            def _nr(r: torch.Tensor) -> torch.Tensor:
                                return r / r.detach().std().clamp(min=1e-8)
                            _mu, _mv, _co = _nr(_mu), _nr(_mv), _nr(_co)
                        _lp   = _lp   + torch.mean(_mu ** 2) + torch.mean(_mv ** 2)
                        _lcont = _lcont + torch.mean(_co ** 2)
                    net.train()
                    _lp   = _lp   / num_re
                    _lcont = _lcont / num_re

                _lt = args["data_loss_weight"] * _ld + _phys_weight * (_lp + args["continuity_weight"] * _lcont)
                _lt.backward()
                _lbfgs_info["l_data"]   = _ld.item()
                _lbfgs_info["l_phys"]   = (_lp + _lcont).item()
                _lbfgs_info["l_total"]  = _lt.item()
                return _lt

            optimizer.step(closure)

            l_data    = torch.tensor([_lbfgs_info.get("l_data",  0.0)], device=device)
            l_physics = torch.tensor([_lbfgs_info.get("l_phys",  0.0)], device=device)
            l_total   = torch.tensor([_lbfgs_info.get("l_total", 0.0)], device=device)
            l_ns_u_total = torch.zeros(1, device=device)
            l_ns_v_total = torch.zeros(1, device=device)
            l_cont_total = torch.zeros(1, device=device)
            phys_weight  = _phys_weight

        else:
        # ── First-order path ─────────────────────────────────────────────────
            optimizer.zero_grad()

        if not is_lbfgs:
            l_data = torch.zeros(1, device=device)
            for i, ds in enumerate(datasets):
                # num_query_points 預設由 K（sensor 數）決定，可在 config 中 override。
                n_query = int(args.get("num_query_points", 0)) or ds.sensor_pos.shape[0]
                xy_np, t_np, c_np, ref_np = ds.sample_sensor_batch(
                    rng, n=n_query, t_max=t_max
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
                per_sample_loss = (pred - ref) ** 2
                t0_w = float(args.get("t_early_weight", 1.0))
                if t0_w != 1.0:
                    t0_thresh = float(args.get("t_early_threshold", 0.1))
                    w = torch.where(t_q <= t0_thresh, torch.full_like(t_q, t0_w), torch.ones_like(t_q))
                    per_sample_loss = per_sample_loss * w
                l_data = l_data + per_sample_loss.mean()
            l_data = l_data / num_re

            phys_weight = physics_weight_at_step(
                step=step,
                final_weight=base_phys_weight,
                warmup_steps=phys_warmup_steps,
                ramp_steps=phys_ramp_steps,
            )
            poisson_weight = float(args.get("poisson_loss_weight", 0.0))
            # GradNorm 模式下不依賴 phys_weight 作為 gate，只要 num_physics_points > 0 就計算物理項。
            n_phys_end    = int(args["num_physics_points"])
            n_phys_start  = int(args.get("num_physics_points_start", 0)) or n_phys_end
            n_phys_warmup = int(args.get("num_physics_points_warmup_steps", 0))
            n_phys_ramp   = int(args.get("num_physics_points_ramp_steps", 0))
            n_phys = physics_points_at_step(step, n_phys_start, n_phys_end, n_phys_ramp, n_phys_warmup)

            phys_gate = (use_gradnorm or phys_weight > 0.0) and n_phys_end > 0

            # RAR pool update（在 net.eval() 之前，避免 eval/train 交替干擾）
            if _is_rar and phys_gate and (_rar_pool_np is None or step % _rar_update_freq == 0):
                _rar_pool_np = _rar_update_pool(
                    net, datasets, sensor_vals_list, sensor_pos_list, sensor_time_list,
                    rng=rng, n_select=n_phys,
                    pool_size=max(n_phys * _rar_pool_mult, n_phys + 1),
                    t_max=t_max, k_f=k_f, A=A, domain_length=domain_length, device=device,
                    exploration_ratio=_rar_expl_ratio,
                )

            if phys_gate:
                net.eval()
                l_ns_u_total = torch.zeros(1, device=device)
                l_ns_v_total = torch.zeros(1, device=device)
                l_ns_total = torch.zeros(1, device=device)
                l_cont_total = torch.zeros(1, device=device)
                l_poisson_total = torch.zeros(1, device=device)
                phys_strategy = str(args.get("physics_collocation_strategy", "random"))
                phys_normalize = bool(args.get("physics_residual_normalize", False))
                for i, ds in enumerate(datasets):
                    if _is_rar and _rar_pool_np is not None:
                        xyt = torch.tensor(_rar_pool_np[i], device=device, requires_grad=True)
                    else:
                        xy_np, t_np = ds.sample_physics_points(
                            rng, n=n_phys, t_max=t_max, strategy=phys_strategy
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
                    l_ns_u_total = l_ns_u_total + torch.mean(mom_u ** 2)
                    l_ns_v_total = l_ns_v_total + torch.mean(mom_v ** 2)
                    l_cont_total = l_cont_total + torch.mean(cont ** 2)
                    if poisson_weight > 0.0:
                        poisson_res = pressure_poisson_residual(u_fn, v_fn, p_fn, xyt)
                        l_poisson_total = l_poisson_total + torch.mean(poisson_res ** 2)
                net.train()
                l_ns_u_total = l_ns_u_total / num_re
                l_ns_v_total = l_ns_v_total / num_re
                l_ns_total = l_ns_u_total + l_ns_v_total
                l_cont_total = l_cont_total / num_re
                l_poisson_total = l_poisson_total / num_re
                l_physics = (
                    l_ns_total
                    + args["continuity_weight"] * l_cont_total
                    + poisson_weight * l_poisson_total
                )
            else:
                l_ns_u_total = torch.zeros(1, device=device)
                l_ns_v_total = torch.zeros(1, device=device)
                l_ns_total = torch.zeros(1, device=device)
                l_cont_total = torch.zeros(1, device=device)
                l_physics = torch.zeros(1, device=device)

            # ── Loss 組合與 backward ────────────────────────────────────────────
            if use_gradnorm and gn_weights is not None:
                # GradNorm 模式：4 個可學習權重 [data, ns_u, ns_v, cont]，直接管理各 task 比例。
                # physics_loss_weight 不再作為乘數，只有 num_physics_points > 0 才啟用物理項。
                #
                # 執行順序（關鍵）：
                #   ① GradNorm weight update（autograd.grad，retain_graph=True）
                #   ② 以更新後的 ws（detach）計算 l_total
                #   ③ l_total.backward() → optimizer.step()
                #
                # Why: optimizer.step() 就地修改 trunk_out.weight（版本號遞增），
                #      GradNorm 必須在 step() 前完成，否則 PyTorch 拋出版本衝突錯誤。
                phys_active = int(args["num_physics_points"]) > 0
                do_gn_update = phys_active and (step % gn_update_freq == 0)

                if do_gn_update:
                    _gradnorm_step(
                        gn_weights,
                        [l_data, l_ns_u_total, l_ns_v_total, l_cont_total],
                        gn_ref_params,
                        ema_momentum=float(args.get("gradnorm_ema_momentum", 0.5)),
                    )
                ws = gn_weights.weights.detach()

                if phys_active:
                    l_total = (
                        ws[0] * l_data
                        + ws[1] * l_ns_u_total
                        + ws[2] * l_ns_v_total
                        + ws[3] * l_cont_total
                    )
                else:
                    l_total = ws[0] * l_data

                l_total.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), float(args["max_grad_norm"]))
                optimizer.step()
            else:
                l_total = args["data_loss_weight"] * l_data + phys_weight * l_physics
                l_total.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), float(args["max_grad_norm"]))
                optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if log_fn is not None:
            extra: dict[str, float] = {}
            if use_gradnorm and gn_weights is not None:
                ws_vals = gn_weights.weights.detach().cpu().tolist()
                extra = {
                    "gn_w_data": ws_vals[0],
                    "gn_w_ns_u": ws_vals[1],
                    "gn_w_ns_v": ws_vals[2],
                    "gn_w_cont": ws_vals[3],
                }
            log_fn(step, {
                "l_data": l_data.item(),
                "l_physics": l_physics.item(),
                "l_ns": l_ns_total.item(),
                "l_cont": l_cont_total.item(),
                "l_total": l_total.item(),
                "w_phys": phys_weight,
                "t_max": t_max if t_max is not None else 0.0,
                **extra,
            })

        if step % max(1, args["iterations"] // 10) == 0 or step == 1:
            tm_str = f"  t≤{t_max:5.1f}" if use_tm and t_max is not None else ""
            if use_gradnorm and gn_weights is not None:
                ws_vals = gn_weights.weights.detach().cpu().tolist()
                print(
                    f"{step:<8} {l_data.item():>12.4e}"
                    f" {l_physics.item():>12.4e}"
                    f" {ws_vals[1]:>8.4f} {ws_vals[2]:>8.4f} {ws_vals[3]:>8.4f}"
                    f" {l_total.item():>12.4e}{tm_str}"
                )
            else:
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
