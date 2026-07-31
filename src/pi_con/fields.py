"""Canonical numpy field kernels on a periodic, uniform 2D grid.

What: vorticity, divergence, block averaging and the coarse query grid that pairs
      with it — the diagnostics every evaluator and baseline computes off a stored
      velocity field.
Why:  before this module there were nine real-space vorticity implementations in the
      repository spread over four mutually incompatible axis conventions, three of
      them transposed relative to the evaluator (so the sign of omega flipped), and
      six divergence implementations. They were duplicated not out of haste but
      because `physics.py` exposes only torch/autograd residuals: there was no numpy
      kernel to import. This module is that target.

      The same consolidation was done for the radial energy spectrum in `spectral.py`
      after duplicate implementations were found to disagree by up to 33% at k=4
      (commits 00be176 / d14bd73 / 3d89d5b). The bodies here are carried over verbatim
      from `scripts/evaluate_deeponet_cfc.py`, whose numbers are the published ones.

Axis convention (project-wide, see CLAUDE.md KNOWN_PITFALLS):
    field[..., x_idx, y_idx]  — axis -2 is x, axis -1 is y.
Every function here assumes it. The cylinder case is deliberately excluded: it lives
on a non-periodic, non-uniform grid and uses `np.gradient` on physical coordinates,
which is a different kernel rather than a variant of this one.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "block_avg",
    "coarse_reference_grid",
    "divergence_fd",
    "enstrophy_fd",
    "kinetic_energy",
    "laplacian_periodic",
    "vorticity_fd",
]


def block_avg(field: np.ndarray, factor: int = 2) -> np.ndarray:
    """f x f block average, supporting a [..., fN, fN] batch shape.

    Vectorised so a caller never loops over frames. `factor` is adjustable so DNS
    stored at different resolutions can be evaluated on one shared grid for a
    cross-Reynolds comparison; `factor=2` is bit-identical to the original behaviour.
    """
    f = int(factor)
    n_x = field.shape[-2] // f
    n_y = field.shape[-1] // f
    new_shape = (*field.shape[:-2], n_x, f, n_y, f)
    return field.reshape(new_shape).mean(axis=(-3, -1))


def coarse_reference_grid(
    x: np.ndarray, y: np.ndarray, factor: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Query grid aligned with the cells `block_avg` produces.

    A block average is the mean over a coarse cell, not the value at the fine node.
    Querying at `x[::f], y[::f]` instead leaves prediction and reference offset by
    half a cell, which contaminates RMSE, vorticity and every spectral diagnostic.
    The divisibility check is part of the contract: a grid that does not divide
    evenly has no well-defined coarse cell.
    """
    f = int(factor)
    if len(x) % f != 0 or len(y) % f != 0:
        raise ValueError(
            f"coarse_reference_grid needs a grid divisible by factor={f}; "
            f"got len(x)={len(x)}, len(y)={len(y)}"
        )
    x_coarse = x.reshape(-1, f).mean(axis=1)
    y_coarse = y.reshape(-1, f).mean(axis=1)
    return x_coarse.astype(np.float32), y_coarse.astype(np.float32)


def vorticity_fd(u: np.ndarray, v: np.ndarray, dx: float) -> np.ndarray:
    """Central-difference vorticity on a periodic grid, [..., N, N] batch shape."""
    dvdx = (np.roll(v, -1, axis=-2) - np.roll(v, 1, axis=-2)) / (2 * dx)
    dudy = (np.roll(u, -1, axis=-1) - np.roll(u, 1, axis=-1)) / (2 * dx)
    return dvdx - dudy


def divergence_fd(u: np.ndarray, v: np.ndarray, dx: float) -> np.ndarray:
    """Central-difference incompressibility residual, [..., N, N] batch shape."""
    dudx = (np.roll(u, -1, axis=-2) - np.roll(u, 1, axis=-2)) / (2 * dx)
    dvdy = (np.roll(v, -1, axis=-1) - np.roll(v, 1, axis=-1)) / (2 * dx)
    return dudx + dvdy


def laplacian_periodic(field: np.ndarray, dx: float) -> np.ndarray:
    """Five-point periodic Laplacian, [..., N, N] batch shape."""
    return (
        np.roll(field, -1, axis=-2)
        + np.roll(field, 1, axis=-2)
        + np.roll(field, -1, axis=-1)
        + np.roll(field, 1, axis=-1)
        - 4.0 * field
    ) / (dx ** 2)


def kinetic_energy(u: np.ndarray, v: np.ndarray) -> float:
    """Domain-averaged kinetic energy of a single snapshot."""
    return float(0.5 * np.mean(u ** 2 + v ** 2))


def enstrophy_fd(u: np.ndarray, v: np.ndarray, dx: float) -> float:
    """Domain-averaged enstrophy of a single snapshot, from the FD vorticity."""
    omega = vorticity_fd(u, v, dx)
    return float(0.5 * np.mean(omega ** 2))
