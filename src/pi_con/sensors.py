"""Canonical sensor-array handling: coordinates, indices, and value extraction.

What: convert between sensor grid indices, physical coordinates and the flat index
      used by the placement algorithms, and pull sensor time series out of a stored
      field. One module owns the axis convention.
Why:  the convention was documentation. Six generators copied it by hand and two
      copied it wrong, producing artefacts whose stored values sit at the transpose of
      the coordinates recorded alongside them. That defect cost the project a
      37-54% regression in KE on 2026-05-18, and one artefact on disk still carries it
      (`sensors_podpivot_K100_N256_t0-5_si100_les_n256_podpivot`, verified 2026-07-31).

The convention, stated once:

    field[t, x_idx, y_idx]      axis 1 is x, axis 2 is y
    flat = x_idx * N + y_idx    row-major, row = x
    coord = (x[x_idx], y[y_idx])

`np.unravel_index(flat, (N, N))` therefore returns `(x_idx, y_idx)` — the first
returned axis is x, not y. Reading a value as `field[:, y_idx, x_idx]` while writing
its coordinate as `(x[x_idx], y[y_idx])` is the transposition this module exists to
make unrepresentable: `sample_series` takes indices and never sees the coordinates,
`indices_from_coords` takes coordinates and never sees the field.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "coords_from_indices",
    "flat_to_indices",
    "indices_from_coords",
    "indices_to_flat",
    "sample_series",
]


def indices_to_flat(x_idx: np.ndarray, y_idx: np.ndarray, n: int) -> np.ndarray:
    """Row-major flat index, row = x."""
    return np.asarray(x_idx, dtype=np.int64) * int(n) + np.asarray(y_idx, dtype=np.int64)


def flat_to_indices(flat: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `indices_to_flat`; returns (x_idx, y_idx) in that order."""
    flat = np.asarray(flat, dtype=np.int64)
    return flat // int(n), flat % int(n)


def coords_from_indices(
    x_idx: np.ndarray, y_idx: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Physical (x, y) coordinates for grid indices, shape [K, 2]."""
    return np.stack(
        [np.asarray(x)[np.asarray(x_idx)], np.asarray(y)[np.asarray(y_idx)]], axis=1
    )


def indices_from_coords(
    coords: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest grid indices for physical coordinates, returned as (x_idx, y_idx).

    `coords` is [K, 2] with column 0 the x coordinate. Nearest-node lookup against the
    stored axes, rather than an analytic `round(x * N)`, so a grid that does not start
    at zero or is not unit-length still resolves correctly.
    """
    coords = np.asarray(coords, dtype=np.float64)
    x_idx = np.argmin(np.abs(coords[:, 0:1] - np.asarray(x)[None, :]), axis=1)
    y_idx = np.argmin(np.abs(coords[:, 1:2] - np.asarray(y)[None, :]), axis=1)
    return x_idx, y_idx


def sample_series(field: np.ndarray, x_idx: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    """Sensor time series from a stored field, shape [K, T].

    `field` is [T, x, y]. The transpose at the end turns the [T, K] result of the
    fancy index into the [K, T] layout the sensor NPZ files use.
    """
    return field[:, np.asarray(x_idx), np.asarray(y_idx)].T
