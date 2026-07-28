"""Fair classical-interpolation baselines for the cross-Re sensor-budget sweep.

What: reconstruct each (Re, K) cell from the same sensors with three DNS-free
      classical methods and score them with the operator evaluator's own formulas
      on the same 128^2 grid.
Why:  the sweep reported PI-CON errors with no reference point. Without a baseline,
      a large error at small K cannot be separated into "the method is weak" and
      "K carries too little information". These three methods see exactly the same
      sensor values and no DNS field, so the comparison is like-for-like.

Methods (all engineering-transferable — no DNS field is used to fit them):
  RBF       thin-plate radial basis interpolation on the periodic-tiled sensors
  IDW       Shepard inverse-distance weighting, power 2, toroidal distance
  trig-LSQ  least squares onto the Fourier modes the sensor count can support,
            |k| <= sqrt(K/pi), i.e. the sampling band edge used in the thesis

Metrics mirror scripts/evaluate_deeponet_cfc.py exactly:
  ke_rel_err = |KE_pred - KE_ref| / KE_ref            per snapshot, then mean
  rel_L2     = ||q_pred - q_ref||_2 / ||q_ref||_2     per snapshot, then mean
Reference fields go through the same f x f block average, so `--block-factor`
must match the factor used for the corresponding PI-CON evaluation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator


def block_avg(field: np.ndarray, factor: int) -> np.ndarray:
    """f x f block average; mirrors the evaluator so grids match."""
    f = int(factor)
    n_x, n_y = field.shape[-2] // f, field.shape[-1] // f
    return field.reshape(*field.shape[:-2], n_x, f, n_y, f).mean(axis=(-3, -1))


def coarse_grid(x: np.ndarray, factor: int) -> np.ndarray:
    return x.reshape(-1, int(factor)).mean(axis=1)


def vorticity(u: np.ndarray, v: np.ndarray, dx: float) -> np.ndarray:
    dvdx = (np.roll(v, -1, axis=-2) - np.roll(v, 1, axis=-2)) / (2 * dx)
    dudy = (np.roll(u, -1, axis=-1) - np.roll(u, 1, axis=-1)) / (2 * dx)
    return dvdx - dudy


def tile_periodic(pts: np.ndarray, vals: np.ndarray, L: float):
    """Replicate sensors into the 8 neighbouring images so interpolants see the wrap."""
    shifts = [(a * L, b * L) for a in (-1, 0, 1) for b in (-1, 0, 1)]
    P = np.concatenate([pts + np.array(s) for s in shifts], axis=0)
    V = np.concatenate([vals] * len(shifts), axis=0)
    return P, V


def recon_rbf(pts, vals, query, L):
    P, V = tile_periodic(pts, vals, L)
    return RBFInterpolator(P, V, kernel="thin_plate_spline", neighbors=64)(query)


def recon_idw(pts, vals, query, L, power=2.0, eps=1e-12):
    d = np.abs(query[:, None, :] - pts[None, :, :])
    d = np.minimum(d, L - d)                      # toroidal
    r = np.sqrt((d ** 2).sum(-1))
    w = 1.0 / np.maximum(r, eps) ** power
    exact = r < 1e-12
    out = (w * vals[None, :]).sum(1) / w.sum(1)
    hit = exact.any(1)
    if hit.any():                                  # query sits on a sensor
        out[hit] = vals[np.argmax(exact[hit], axis=1)]
    return out


def recon_trig_lsq(pts, vals, query, L, K):
    """Least squares onto modes inside the sampling band edge |k| <= sqrt(K/pi)."""
    n_max = int(np.floor(np.sqrt(K / np.pi)))
    modes = [(a, b) for a in range(-n_max, n_max + 1) for b in range(-n_max, n_max + 1)
             if a * a + b * b <= n_max * n_max]

    def design(p):
        cols = [np.ones(len(p))]
        for a, b in modes:
            if (a, b) == (0, 0):
                continue
            ph = 2 * np.pi * (a * p[:, 0] + b * p[:, 1]) / L
            cols += [np.cos(ph), np.sin(ph)]
        return np.stack(cols, axis=1)

    coef, *_ = np.linalg.lstsq(design(pts), vals, rcond=None)
    return design(query) @ coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dns", required=True)
    ap.add_argument("--sensor-json", required=True)
    ap.add_argument("--sensor-npz", required=True)
    ap.add_argument("--block-factor", type=int, required=True,
                    help="must match the PI-CON evaluation for this case")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--label", default="")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    d = np.load(args.dns, allow_pickle=True).item()
    x = np.asarray(d["x"], float)
    L = float(x[-1] - x[0] + (x[1] - x[0]))
    xg = coarse_grid(x.astype(np.float32), args.block_factor)
    XX, YY = np.meshgrid(xg, xg, indexing="ij")
    query = np.stack([XX.ravel(), YY.ravel()], axis=1)
    n = len(xg)
    dx = L / n

    meta = json.loads(Path(args.sensor_json).read_text())
    pts = np.asarray(meta["selected_coordinates"], float)
    K = int(meta["K"])
    npz = np.load(args.sensor_npz)
    su, sv = np.asarray(npz["u"], float), np.asarray(npz["v"], float)   # [K, T]

    tidx = np.arange(0, su.shape[1], args.stride)
    res = {m: {"ke": [], "u": [], "om": []} for m in ("rbf", "idw", "trig")}

    for ti in tidx:
        ur = block_avg(np.asarray(d["u"][ti], np.float32), args.block_factor)
        vr = block_avg(np.asarray(d["v"][ti], np.float32), args.block_factor)
        omr = vorticity(ur, vr, dx)
        ke_ref = 0.5 * np.mean(ur ** 2 + vr ** 2)
        for name, fn in (("rbf", lambda p, v: recon_rbf(pts, v, query, L)),
                         ("idw", lambda p, v: recon_idw(pts, v, query, L)),
                         ("trig", lambda p, v: recon_trig_lsq(pts, v, query, L, K))):
            up = fn(pts, su[:, ti]).reshape(n, n)
            vp = fn(pts, sv[:, ti]).reshape(n, n)
            omp = vorticity(up, vp, dx)
            ke_p = 0.5 * np.mean(up ** 2 + vp ** 2)
            res[name]["ke"].append(abs(ke_p - ke_ref) / max(ke_ref, 1e-12))
            res[name]["u"].append(np.sqrt(np.sum((up - ur) ** 2)) / max(np.sqrt(np.sum(ur ** 2)), 1e-12))
            res[name]["om"].append(np.sqrt(np.sum((omp - omr) ** 2)) / max(np.sqrt(np.sum(omr ** 2)), 1e-12))

    out = {"label": args.label, "K": K, "grid_n": n, "n_frames": len(tidx),
           "block_factor": args.block_factor}
    for m in res:
        for q in ("ke", "u", "om"):
            out[f"{m}_{q}"] = float(np.mean(res[m][q]))
    print(f"{args.label:22s} K={K:>3d} grid={n} frames={len(tidx):>3d} | " + "  ".join(
        f"{m}: KE {out[f'{m}_ke']*100:6.2f}% u {out[f'{m}_u']*100:6.2f}% om {out[f'{m}_om']*100:6.2f}%"
        for m in ("rbf", "idw", "trig")))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
