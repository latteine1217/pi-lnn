"""PI-CON optimizer extensions.

What: 收集 pi-con 額外實作的 optimizer（不在 torch.optim 內）。
Why : 既有 SOAP / ScheduleFree / LBFGS 走 torch.optim 介面，但 NG 需要 access
      per-residual Jacobian，無法套到 torch.optim.Optimizer 的 .grad 介面，
      因此另闢 module。
"""
from __future__ import annotations

from pi_con.optimizers.natural_gradient import (
    NaturalGradientOptimizer,
    compute_residual_jacobian,
    solve_ng_step,
)

__all__ = [
    "NaturalGradientOptimizer",
    "compute_residual_jacobian",
    "solve_ng_step",
]
