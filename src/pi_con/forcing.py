"""Learnable / fixed forcing prior for Kolmogorov-type external body force.

What:
    Kolmogorov flow forcing has the closed form  f = (A · sin(k_f · y_norm), 0)
    where y_norm = y / domain_length. Both A (amplitude) and k_f (wavenumber)
    are normally treated as known PDE parameters (engineering prior). This
    module makes either or both optionally learnable, so the sparse-sensor +
    physics-residual training can also identify forcing from data.

Why:
    Treating forcing as a learnable scalar turns the PDE-residual loss into a
    soft inverse problem: model output must satisfy NS *with* the inferred
    forcing. The diagnostic plot `forcing_mode_coeff_u` therefore carries
    non-trivial information about identifiability — when forcing is hardcoded
    it converges trivially.

Design choices:
    - A is parameterized via log(A) to guarantee A > 0.
    - k_f is parameterized via sigmoid(raw) · (k_max - k_min) + k_min to keep
      it inside a physically reasonable band [k_min, k_max], avoiding the
      runaway optima at k_f → 0 (DC forcing) or k_f → N/2 (Nyquist).
    - When not learning, the value is stored as a buffer (not a Parameter)
      so it travels with .to(device) but is not seen by the optimizer.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class ForcingPrior(nn.Module):
    """Forcing parameters (A, k_f) — fixed or learnable per-flag."""

    def __init__(
        self,
        A_init: float = 0.1,
        k_f_init: float = 2.0,
        learn_A: bool = False,
        learn_k_f: bool = False,
        k_f_min: float = 1.0,
        k_f_max: float = 8.0,
    ) -> None:
        super().__init__()
        if A_init <= 0:
            raise ValueError(f"A_init must be > 0 (got {A_init})")
        if not (k_f_min < k_f_init < k_f_max):
            raise ValueError(
                f"k_f_init={k_f_init} must lie strictly inside (k_f_min={k_f_min}, "
                f"k_f_max={k_f_max})"
            )

        self.learn_A = bool(learn_A)
        self.learn_k_f = bool(learn_k_f)
        self.k_f_min = float(k_f_min)
        self.k_f_max = float(k_f_max)

        # NOTE: 用 shape=(1,) 而非 0-dim — ScheduleFree optimizer 的 swap 對 0-dim
        # tensor 會 view(uint8) fail（RuntimeError: self.dim() cannot be 0）。
        # 後續所有 forcing_x = A * sin(...) 計算靠 broadcast 即可。
        if self.learn_A:
            self.log_A = nn.Parameter(torch.tensor([math.log(float(A_init))]))
        else:
            self.register_buffer("_A_const", torch.tensor([float(A_init)]))

        if self.learn_k_f:
            x = (float(k_f_init) - self.k_f_min) / (self.k_f_max - self.k_f_min)
            raw = math.log(x / (1.0 - x))
            self.raw_k_f = nn.Parameter(torch.tensor([raw]))
        else:
            self.register_buffer("_kf_const", torch.tensor([float(k_f_init)]))

    @property
    def A(self) -> torch.Tensor:
        """Forcing amplitude (1-dim tensor with shape=[1])."""
        if self.learn_A:
            return torch.exp(self.log_A)
        return self._A_const

    @property
    def k_f(self) -> torch.Tensor:
        """Forcing wavenumber (1-dim tensor bounded in [k_f_min, k_f_max])."""
        if self.learn_k_f:
            return (
                torch.sigmoid(self.raw_k_f) * (self.k_f_max - self.k_f_min)
                + self.k_f_min
            )
        return self._kf_const

    def snapshot(self) -> dict[str, float]:
        """Plain-Python view for logging / checkpoint metadata."""
        return {
            "A": float(self.A.detach().item()),
            "k_f": float(self.k_f.detach().item()),
            "learn_A": self.learn_A,
            "learn_k_f": self.learn_k_f,
        }
