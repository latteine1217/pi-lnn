"""Unit tests for ForcingPrior + NS-residual integration.

Covers:
  - Basic fixed / learnable mode for A and k_f independently
  - sigmoid bound on k_f
  - Init validation (A>0, k_f within range)
  - End-to-end gradient flow: NS residual loss → forcing params
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pi_con.forcing import ForcingPrior  # noqa: E402
from pi_con.physics import unsteady_ns_residuals  # noqa: E402


def test_fixed_mode_no_parameters():
    fp = ForcingPrior(A_init=0.1, k_f_init=2.0, learn_A=False, learn_k_f=False)
    assert list(fp.parameters()) == []
    assert float(fp.A.detach()) == pytest.approx(0.1, rel=1e-6)
    assert float(fp.k_f.detach()) == pytest.approx(2.0, rel=1e-6)


def test_learn_A_only_one_parameter():
    fp = ForcingPrior(A_init=0.05, k_f_init=2.0, learn_A=True, learn_k_f=False)
    params = list(fp.parameters())
    assert len(params) == 1
    assert params[0] is fp.log_A
    # exp(log_A) == 0.05
    assert float(fp.A.detach()) == pytest.approx(0.05, rel=1e-5)
    # k_f buffer 不可學
    assert not hasattr(fp, "raw_k_f")


def test_learn_k_f_only_sigmoid_bound():
    fp = ForcingPrior(
        A_init=0.1, k_f_init=4.0, learn_A=False, learn_k_f=True,
        k_f_min=1.0, k_f_max=8.0,
    )
    assert len(list(fp.parameters())) == 1
    assert float(fp.k_f.detach()) == pytest.approx(4.0, rel=1e-5)
    # raw 極端值仍 bounded
    with torch.no_grad():
        fp.raw_k_f.copy_(torch.tensor(1e6))
    assert float(fp.k_f.detach()) == pytest.approx(8.0, abs=1e-3)
    with torch.no_grad():
        fp.raw_k_f.copy_(torch.tensor(-1e6))
    assert float(fp.k_f.detach()) == pytest.approx(1.0, abs=1e-3)


def test_learn_both_two_parameters():
    fp = ForcingPrior(A_init=0.05, k_f_init=4.0, learn_A=True, learn_k_f=True)
    assert len(list(fp.parameters())) == 2


def test_reject_invalid_init():
    with pytest.raises(ValueError, match="A_init"):
        ForcingPrior(A_init=-0.1, learn_A=True)
    with pytest.raises(ValueError, match="A_init"):
        ForcingPrior(A_init=0.0, learn_A=True)
    with pytest.raises(ValueError, match="k_f_init"):
        ForcingPrior(k_f_init=10.0, k_f_min=1.0, k_f_max=8.0, learn_k_f=True)
    with pytest.raises(ValueError, match="k_f_init"):
        ForcingPrior(k_f_init=1.0, k_f_min=1.0, k_f_max=8.0, learn_k_f=True)


def test_state_dict_roundtrip_preserves_value():
    fp1 = ForcingPrior(A_init=0.07, k_f_init=3.5, learn_A=True, learn_k_f=True)
    sd = fp1.state_dict()
    fp2 = ForcingPrior(A_init=0.1, k_f_init=4.0, learn_A=True, learn_k_f=True)
    fp2.load_state_dict(sd)
    assert float(fp2.A.detach()) == pytest.approx(0.07, rel=1e-5)
    assert float(fp2.k_f.detach()) == pytest.approx(3.5, rel=1e-5)


def _make_uvp_fn():
    """Tiny analytic uvp_fn for gradient-flow tests (no model required)."""
    def uvp_fn(xyt):
        # u = sin(2π x), v = -cos(2π x), p = 0  (toy field, ∇·u ≠ 0)
        x = xyt[:, 0:1]
        u = torch.sin(2.0 * math.pi * x)
        v = -torch.cos(2.0 * math.pi * x)
        p = torch.zeros_like(x)
        return torch.cat([u, v, p], dim=-1)
    return uvp_fn


def test_ns_residual_accepts_tensor_k_f_A_and_backprops_to_forcing():
    """E2E: NS residual loss must propagate gradient back to log_A / raw_k_f."""
    fp = ForcingPrior(A_init=0.05, k_f_init=4.0, learn_A=True, learn_k_f=True)
    xyt = torch.rand(128, 3, requires_grad=True)
    uvp_fn = _make_uvp_fn()
    mu, mv, co = unsteady_ns_residuals(
        uvp_fn, xyt, re=10000.0, k_f=fp.k_f, A=fp.A, domain_length=1.0,
    )
    loss = (mu ** 2).mean() + (mv ** 2).mean() + (co ** 2).mean()
    loss.backward()
    assert fp.log_A.grad is not None and not torch.isnan(fp.log_A.grad).any()
    assert fp.raw_k_f.grad is not None and not torch.isnan(fp.raw_k_f.grad).any()
    assert abs(float(fp.log_A.grad)) > 0, "log_A should receive non-zero gradient"
    assert abs(float(fp.raw_k_f.grad)) > 0, "raw_k_f should receive non-zero gradient"


def test_ns_residual_with_fixed_forcing_still_works():
    """Regression: NS residual still accepts plain float k_f / A (legacy path)."""
    xyt = torch.rand(32, 3, requires_grad=True)
    uvp_fn = _make_uvp_fn()
    mu, mv, co = unsteady_ns_residuals(
        uvp_fn, xyt, re=10000.0, k_f=2.0, A=0.1, domain_length=1.0,
    )
    assert mu.shape == (32, 1)
    assert torch.isfinite(mu).all()


def test_snapshot_returns_plain_floats():
    fp = ForcingPrior(A_init=0.07, k_f_init=3.0, learn_A=True, learn_k_f=True)
    snap = fp.snapshot()
    assert isinstance(snap["A"], float)
    assert isinstance(snap["k_f"], float)
    assert snap["learn_A"] is True
    assert snap["learn_k_f"] is True
