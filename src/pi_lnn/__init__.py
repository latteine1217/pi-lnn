"""Pi-LNN: Sparse-sensor physics-constrained operator learning for turbulent flow."""
from __future__ import annotations

# Why: PYTORCH_ENABLE_MPS_FALLBACK 必須在 import torch 之前設好；
# 任何 `import pi_lnn.X` 都會先執行本檔，所以放在所有 pi_lnn submodule
# import 之前是設定環境變數的最早時機。
import os

os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pi_lnn.blocks import CfCCell, ResidualMLPBlock, TokenSelfAttentionBlock
from pi_lnn.config import DEFAULT_LNN_ARGS, load_lnn_config
from pi_lnn.decoder import DeepONetCfCDecoder
from pi_lnn.encoders import SpatialSetEncoder, TemporalCfCEncoder
from pi_lnn.encodings import (
    FourierEmbs,
    LearnableFourierEmb,
    periodic_fourier_encode,
    temporal_phase_anchor,
)
from pi_lnn.losses import GradNormWeights, observed_channel_prediction
from pi_lnn.operator import (
    LiquidOperator,
    create_lnn_model,
    make_lnn_model_fn,
    make_lnn_model_fn_uvp,
)
from pi_lnn.physics import (
    physics_points_at_step,
    physics_weight_at_step,
    pressure_poisson_residual,
    unsteady_ns_residuals,
)
from pi_lnn.runtime import configure_torch_runtime, count_parameters, write_json
from pi_lnn.training import main, train_lnn_kolmogorov

__all__ = [
    "CfCCell",
    "DEFAULT_LNN_ARGS",
    "DeepONetCfCDecoder",
    "FourierEmbs",
    "GradNormWeights",
    "LearnableFourierEmb",
    "LiquidOperator",
    "ResidualMLPBlock",
    "SpatialSetEncoder",
    "TemporalCfCEncoder",
    "TokenSelfAttentionBlock",
    "configure_torch_runtime",
    "count_parameters",
    "create_lnn_model",
    "load_lnn_config",
    "main",
    "make_lnn_model_fn",
    "make_lnn_model_fn_uvp",
    "observed_channel_prediction",
    "periodic_fourier_encode",
    "physics_points_at_step",
    "physics_weight_at_step",
    "pressure_poisson_residual",
    "temporal_phase_anchor",
    "train_lnn_kolmogorov",
    "unsteady_ns_residuals",
    "write_json",
]
