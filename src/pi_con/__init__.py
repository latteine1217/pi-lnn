"""PI-CON: Sparse-sensor physics-constrained operator learning for turbulent flow."""
from __future__ import annotations

# Why: PYTORCH_ENABLE_MPS_FALLBACK 必須在 import torch 之前設好；
# 任何 `import pi_con.X` 都會先執行本檔，所以放在所有 pi_con submodule
# import 之前是設定環境變數的最早時機。
import os

os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pi_con.blocks import CfCCell, ModifiedMLPBlock, ResidualMLPBlock, TokenSelfAttentionBlock
from pi_con.config import DEFAULT_PICON_ARGS, _validate_al_config, load_picon_config
from pi_con.decoder import DeepONetCfCDecoder
from pi_con.dns_align import find_dns_time_idx
from pi_con.encoders import SpatialSetEncoder, TemporalCfCEncoder
from pi_con.encodings import (
    FourierEmbs,
    LearnableFourierEmb,
    periodic_fourier_encode,
    temporal_phase_anchor,
)
from pi_con.losses import (
    AugmentedLagrangianMultiplier,
    GradNormWeights,
    observed_channel_prediction,
)
from pi_con.operator import (
    LiquidOperator,
    create_picon_model,
    make_picon_model_fn,
    make_picon_model_fn_uvp,
)
from pi_con.physics import (
    physics_points_at_step,
    physics_weight_at_step,
    pressure_poisson_residual,
    unsteady_ns_residuals,
)
from pi_con.runtime import configure_torch_runtime, count_parameters, write_json
from pi_con.training import main, train_picon_kolmogorov

__all__ = [
    "AugmentedLagrangianMultiplier",
    "CfCCell",
    "DEFAULT_PICON_ARGS",
    "DeepONetCfCDecoder",
    "FourierEmbs",
    "GradNormWeights",
    "_validate_al_config",
    "LearnableFourierEmb",
    "LiquidOperator",
    "ResidualMLPBlock",
    "SpatialSetEncoder",
    "TemporalCfCEncoder",
    "TokenSelfAttentionBlock",
    "configure_torch_runtime",
    "count_parameters",
    "create_picon_model",
    "find_dns_time_idx",
    "load_picon_config",
    "main",
    "make_picon_model_fn",
    "make_picon_model_fn_uvp",
    "observed_channel_prediction",
    "periodic_fourier_encode",
    "physics_points_at_step",
    "physics_weight_at_step",
    "pressure_poisson_residual",
    "temporal_phase_anchor",
    "train_picon_kolmogorov",
    "unsteady_ns_residuals",
    "write_json",
]
