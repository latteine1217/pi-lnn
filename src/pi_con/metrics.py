"""Canonical reconstruction-error metrics.

What: the relative-L2 and kinetic-energy errors every evaluator and baseline reports,
      plus the band-energy split.
Why:  these formulas were previously inlined at roughly fifteen sites with no test at
      any of them, including inside the 1688-line evaluator. Writing a comparable
      baseline meant opening that script and retyping its arithmetic, which is how a
      second, non-equivalent definition entered the repository.

Two definitions of "relative L2" are in use and they are not the same number:

    per_snapshot  mean over t of ||q_pred(t) - q_ref(t)|| / ||q_ref(t)||
    whole_array   ||q_pred - q_ref|| / ||q_ref||   over the stacked array

They coincide only when the field magnitude is constant in time. On a decaying
trajectory they differ, because a mean of ratios is not a ratio of norms. The
evaluator and the baselines use `per_snapshot`; the grid-independence analysis and
several plotting scripts use `whole_array`. Rather than silently pick one, `rel_l2`
takes an explicit `reduction` and has no default — a caller must say which it means.

Axis convention: fields are [T, x, y] or [x, y]; see `pi_con.fields`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["band_energies", "ke_rel_err", "ke_series", "rel_l2"]

_EPS = 1.0e-12


def ke_series(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Domain-averaged kinetic energy per snapshot, for [T, x, y] input."""
    return 0.5 * np.mean(u ** 2 + v ** 2, axis=(-2, -1))


def ke_rel_err(
    u_pred: np.ndarray,
    v_pred: np.ndarray,
    u_ref: np.ndarray,
    v_ref: np.ndarray,
    *,
    per_snapshot: bool = False,
) -> np.ndarray | float:
    """Relative error of the kinetic energy.

    Returns the per-snapshot series when `per_snapshot` is set, otherwise its mean.
    The reference is floored at 1e-12 so a vanishing field cannot divide by zero.
    """
    ke_p = ke_series(u_pred, v_pred)
    ke_r = ke_series(u_ref, v_ref)
    series = np.abs(ke_p - ke_r) / np.maximum(ke_r, _EPS)
    return series if per_snapshot else float(np.mean(series))


def rel_l2(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    reduction: str,
) -> np.ndarray | float:
    """Relative L2 error of one field.

    `reduction` is required, because the two available definitions do not agree on a
    trajectory whose magnitude changes in time:

      "per_snapshot"       mean over t of the per-snapshot norm ratio
      "per_snapshot_series" the same, returned as the series rather than its mean
      "whole_array"        one ratio over the stacked array

    Input is [T, x, y] for the per-snapshot forms, any shape for "whole_array".
    """
    if reduction not in ("per_snapshot", "per_snapshot_series", "whole_array"):
        raise ValueError(
            f"reduction must be per_snapshot, per_snapshot_series or whole_array; "
            f"got {reduction!r}"
        )
    if reduction == "whole_array":
        return float(
            np.linalg.norm(pred - ref) / max(float(np.linalg.norm(ref)), _EPS)
        )
    series = np.sqrt(np.sum((pred - ref) ** 2, axis=(-2, -1))) / np.maximum(
        np.sqrt(np.sum(ref ** 2, axis=(-2, -1))), _EPS
    )
    return series if reduction == "per_snapshot_series" else float(np.mean(series))


def band_energies(
    u: np.ndarray,
    v: np.ndarray,
    edges: tuple[float, float],
    n_max: int | None = None,
) -> dict[str, float]:
    """Split the snapshot energy into low, mid and high wavenumber bands.

    `edges` is (k_low, k_high); the high band runs to `n_max`, defaulting to the
    Nyquist wavenumber of the grid. Shell membership follows `pi_con.spectral`, so a
    band split and a spectrum plot cannot disagree about which shell a mode is in.
    """
    from pi_con.spectral import radial_energy_spectrum

    bins, e_k = radial_energy_spectrum(u, v)
    k_low, k_high = edges
    top = float(n_max) if n_max is not None else float(bins[-1])
    return {
        "low": float(e_k[bins <= k_low].sum()),
        "mid": float(e_k[(bins > k_low) & (bins <= k_high)].sum()),
        "high": float(e_k[(bins > k_high) & (bins <= top)].sum()),
    }
