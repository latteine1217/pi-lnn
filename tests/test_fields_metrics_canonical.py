"""Guard that the canonical kernels reproduce the implementations they replace.

The numbers in the thesis were produced by the copies inlined in
`scripts/evaluate_deeponet_cfc.py` and the baseline scripts. Consolidating those into
`pi_con.fields` and `pi_con.metrics` is only safe if the canonical version is
bit-identical on the paths that were in use, so each test below restates the old body
and asserts exact equality rather than a tolerance.

The one case where equality is *not* asserted is the second relative-L2 definition:
`whole_array` and `per_snapshot` genuinely differ on a trajectory whose magnitude
varies, and the test pins that difference so nobody silently swaps one for the other.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pi_con.fields import (  # noqa: E402
    block_avg,
    coarse_reference_grid,
    divergence_fd,
    enstrophy_fd,
    kinetic_energy,
    vorticity_fd,
)
from pi_con.metrics import ke_rel_err, rel_l2  # noqa: E402


@pytest.fixture
def fields():
    rng = np.random.default_rng(20260731)
    u = rng.normal(size=(6, 32, 32))
    v = rng.normal(size=(6, 32, 32))
    # a decaying trajectory, so mean-of-ratios and ratio-of-norms must diverge
    decay = np.exp(-np.arange(6) / 2.0)[:, None, None]
    return u * decay, v * decay


def test_block_avg_matches_evaluator(fields):
    u, _ = fields
    for f in (1, 2, 4):
        n = u.shape[-1] // f
        expected = u.reshape(*u.shape[:-2], n, f, n, f).mean(axis=(-3, -1))
        assert np.array_equal(block_avg(u, f), expected)


def test_block_avg_default_is_two_by_two(fields):
    u, _ = fields
    assert np.array_equal(block_avg(u), block_avg(u, 2))


def test_coarse_grid_matches_evaluator():
    x = np.linspace(0.0, 1.0, 32, endpoint=False)
    expected = (0.5 * (x[0::2] + x[1::2])).astype(np.float32)
    got_x, got_y = coarse_reference_grid(x, x)
    assert np.array_equal(got_x, expected)
    assert np.array_equal(got_y, expected)


def test_coarse_grid_rejects_indivisible_grid():
    x = np.linspace(0.0, 1.0, 30, endpoint=False)
    with pytest.raises(ValueError, match="divisible"):
        coarse_reference_grid(x, x, factor=4)


def test_vorticity_matches_evaluator(fields):
    u, v = fields
    dx = 1.0 / u.shape[-1]
    expected = ((np.roll(v, -1, axis=-2) - np.roll(v, 1, axis=-2)) / (2 * dx)
                - (np.roll(u, -1, axis=-1) - np.roll(u, 1, axis=-1)) / (2 * dx))
    assert np.array_equal(vorticity_fd(u, v, dx), expected)


def test_vorticity_axis_convention_is_x_on_minus_two(fields):
    """A field varying only along x must give the same omega as -d(u)/dy would not.

    This pins which axis is x. The three script-local copies in the qrpivot family
    treat axis 2 of a [T, N, N] array as x, which transposes the field and flips the
    sign of omega relative to this convention.
    """
    n, dx = 16, 1.0 / 16
    x = np.arange(n) * dx
    v = np.broadcast_to(np.sin(2 * np.pi * x)[:, None], (n, n)).copy()
    u = np.zeros((n, n))
    omega = vorticity_fd(u, v, dx)
    # omega = dv/dx here, so it must vary along axis -2 and be constant along axis -1
    assert np.allclose(omega.std(axis=-1), 0.0, atol=1e-12)
    assert omega.std(axis=-2).min() > 0.1


def test_divergence_matches_evaluator(fields):
    u, v = fields
    dx = 1.0 / u.shape[-1]
    expected = ((np.roll(u, -1, axis=-2) - np.roll(u, 1, axis=-2)) / (2 * dx)
                + (np.roll(v, -1, axis=-1) - np.roll(v, 1, axis=-1)) / (2 * dx))
    assert np.array_equal(divergence_fd(u, v, dx), expected)


def test_kinetic_energy_and_enstrophy_match_evaluator(fields):
    u, v = fields
    dx = 1.0 / u.shape[-1]
    assert kinetic_energy(u[0], v[0]) == float(0.5 * np.mean(u[0] ** 2 + v[0] ** 2))
    om = vorticity_fd(u[0], v[0], dx)
    assert enstrophy_fd(u[0], v[0], dx) == float(0.5 * np.mean(om ** 2))


def test_ke_rel_err_matches_baseline_scripts(fields):
    u, v = fields
    pu, pv = u * 1.03, v * 0.97
    ke_p = 0.5 * (pu ** 2 + pv ** 2).mean(axis=(1, 2))
    ke_t = 0.5 * (u ** 2 + v ** 2).mean(axis=(1, 2))
    expected = float(np.abs((ke_p - ke_t) / ke_t).mean())
    assert ke_rel_err(pu, pv, u, v) == pytest.approx(expected, rel=0, abs=1e-15)


def test_ke_rel_err_matches_evaluator_series_form(fields):
    u, v = fields
    pu, pv = u * 1.03, v * 0.97
    ke_p = 0.5 * np.mean(pu ** 2 + pv ** 2, axis=(1, 2))
    ke_r = 0.5 * np.mean(u ** 2 + v ** 2, axis=(1, 2))
    expected = np.abs(ke_p - ke_r) / np.maximum(ke_r, 1.0e-12)
    assert np.array_equal(ke_rel_err(pu, pv, u, v, per_snapshot=True), expected)


def test_rel_l2_per_snapshot_matches_evaluator(fields):
    u, _ = fields
    pu = u * 1.05
    expected = np.sqrt(np.sum((pu - u) ** 2, axis=(1, 2))) / np.maximum(
        np.sqrt(np.sum(u ** 2, axis=(1, 2))), 1.0e-12
    )
    assert np.array_equal(
        rel_l2(pu, u, reduction="per_snapshot_series"), expected
    )
    assert rel_l2(pu, u, reduction="per_snapshot") == pytest.approx(
        float(expected.mean()), rel=0, abs=1e-15
    )


def test_rel_l2_whole_array_matches_grid_independence_scripts(fields):
    u, _ = fields
    pu = u * 1.05
    expected = float(np.linalg.norm(pu - u) / np.linalg.norm(u))
    assert rel_l2(pu, u, reduction="whole_array") == pytest.approx(
        expected, rel=0, abs=1e-15
    )


def test_the_two_reductions_are_not_interchangeable(fields):
    """Pin the difference the repository currently has in circulation.

    On a decaying trajectory a mean of per-snapshot ratios is not the ratio of the
    stacked norms. Both appear in the repository under the name "relative L2"; this
    asserts they are distinct so the module cannot quietly default to one.
    """
    u, _ = fields
    # error concentrated in the late, low-magnitude frames
    pu = u.copy()
    pu[3:] *= 1.6
    a = rel_l2(pu, u, reduction="per_snapshot")
    b = rel_l2(pu, u, reduction="whole_array")
    assert abs(a - b) > 0.05


def test_rel_l2_requires_explicit_reduction(fields):
    u, _ = fields
    with pytest.raises(TypeError):
        rel_l2(u, u)                      # type: ignore[call-arg]
    with pytest.raises(ValueError, match="reduction must be"):
        rel_l2(u, u, reduction="mean")
