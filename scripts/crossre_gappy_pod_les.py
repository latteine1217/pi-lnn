"""Gappy POD reconstruction on an LES-derived basis (DNS-free reduced-order baseline).

What: build a POD basis from a cheap LES of the same flow, fit its coefficients to the
      K sensor readings by least squares, and score the reconstruction with the
      operator evaluator's formulas on the same grid.
Why:  the classical baselines already run (RBF, IDW, band-limited least squares) carry
      no model of the flow. A reduced-order method does, and is the standard modern
      comparison. Building the basis from DNS would make it an oracle, so the basis
      comes from the LES the placement pipeline already produces — the same information
      a deployment would have.

Truncation is chosen without touching the DNS: r = K, so the 2K velocity observations
overdetermine the r coefficients two to one. An energy criterion is unusable here — 99%
of the LES snapshot energy sits in 16 modes because the flow is condensate-dominated,
and truncating there leaves 45% projection error against the DNS. Raising r past K makes
the fit square and the reconstruction diverges (measured: 13% at r=K=100, 90% at r=200).

The run also reports the projection of the DNS onto the same basis. That quantity uses
the reference field and is therefore not a baseline; it is the diagnostic that separates
"the basis cannot represent this flow" from "the sensors cannot pin down the coefficients".

A global sign error in a basis is immaterial here: span{phi} = span{-phi}, and the
least-squares coefficient absorbs the flip, so the reconstruction is unchanged.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def block_avg(field: np.ndarray, factor: int) -> np.ndarray:
    f = int(factor)
    if f == 1:
        return field
    n_x, n_y = field.shape[-2] // f, field.shape[-1] // f
    return field.reshape(*field.shape[:-2], n_x, f, n_y, f).mean(axis=(-3, -1))


def coarse_grid(x: np.ndarray, factor: int) -> np.ndarray:
    return x if int(factor) == 1 else x.reshape(-1, int(factor)).mean(axis=1)


def vorticity(u, v, dx):
    return ((np.roll(v, -1, axis=-2) - np.roll(v, 1, axis=-2))
            - (np.roll(u, -1, axis=-1) - np.roll(u, 1, axis=-1))) / (2 * dx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--les", required=True)
    ap.add_argument("--dns", required=True)
    ap.add_argument("--sensor-json", required=True)
    ap.add_argument("--sensor-npz", required=True)
    ap.add_argument("--dns-block-factor", type=int, required=True)
    ap.add_argument("--n-basis-snapshots", type=int, default=500)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--label", default="")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    # --- reference (DNS) on the shared evaluation grid ---
    d = np.load(args.dns, allow_pickle=True).item()
    xg = coarse_grid(np.asarray(d["x"], np.float64), args.dns_block_factor)
    n = len(xg)
    L = float(xg[-1] - xg[0] + (xg[1] - xg[0]))
    dx = L / n

    # --- POD basis from LES, brought to the same grid ---
    les = np.load(args.les, allow_pickle=True).item()
    n_les = les["u"].shape[-1]
    if n_les % n:
        raise ValueError(f"LES grid {n_les} is not a multiple of the eval grid {n}")
    fac = n_les // n
    sel = np.linspace(0, les["u"].shape[0] - 1, min(args.n_basis_snapshots, les["u"].shape[0]),
                      dtype=int)
    U = np.stack([block_avg(np.asarray(les["u"][i], np.float64), fac).ravel() for i in sel], 1)
    V = np.stack([block_avg(np.asarray(les["v"][i], np.float64), fac).ravel() for i in sel], 1)
    X = np.concatenate([U, V], axis=0)                    # [2 n^2, n_snap]
    mean = X.mean(axis=1, keepdims=True)
    Phi, S, _ = np.linalg.svd(X - mean, full_matrices=False)

    meta = json.loads(Path(args.sensor_json).read_text())
    pts = np.asarray(meta["selected_coordinates"], np.float64)
    K = int(meta["K"])
    energy = np.cumsum(S**2) / np.sum(S**2)
    r99 = int(np.searchsorted(energy, 0.99) + 1)
    r = max(1, min(K, Phi.shape[1]))          # 2K observations over r modes
    Phi_r = Phi[:, :r]

    # sensor rows of the basis: flat index = x_idx * n + y_idx (project convention)
    xi = np.argmin(np.abs(pts[:, 0:1] - xg[None, :]), axis=1)
    yi = np.argmin(np.abs(pts[:, 1:2] - xg[None, :]), axis=1)
    flat = xi * n + yi
    rows = np.concatenate([flat, flat + n * n])           # u-block then v-block
    H_Phi = Phi_r[rows, :]                                 # [2K, r]
    H_mean = mean[rows, 0]

    npz = np.load(args.sensor_npz)
    su, sv = np.asarray(npz["u"], np.float64), np.asarray(npz["v"], np.float64)
    tidx = np.arange(0, su.shape[1], args.stride)

    ke_e, u_e, om_e, proj_e = [], [], [], []
    for ti in tidx:
        obs = np.concatenate([su[:, ti], sv[:, ti]]) - H_mean
        a, *_ = np.linalg.lstsq(H_Phi, obs, rcond=None)
        rec = (Phi_r @ a)[:, None] + mean
        up = rec[: n * n].reshape(n, n)
        vp = rec[n * n:].reshape(n, n)

        ur = block_avg(np.asarray(d["u"][ti], np.float64), args.dns_block_factor)
        vr = block_avg(np.asarray(d["v"][ti], np.float64), args.dns_block_factor)
        tgt = np.concatenate([ur.ravel(), vr.ravel()])[:, None]
        prj = Phi_r @ (Phi_r.T @ (tgt - mean)) + mean
        proj_e.append(float(np.linalg.norm(prj - tgt) / max(np.linalg.norm(tgt), 1e-12)))
        omr, omp = vorticity(ur, vr, dx), vorticity(up, vp, dx)
        ke_r = 0.5 * np.mean(ur**2 + vr**2)
        ke_e.append(abs(0.5 * np.mean(up**2 + vp**2) - ke_r) / max(ke_r, 1e-12))
        u_e.append(np.sqrt(np.sum((up - ur) ** 2)) / max(np.sqrt(np.sum(ur**2)), 1e-12))
        om_e.append(np.sqrt(np.sum((omp - omr) ** 2)) / max(np.sqrt(np.sum(omr**2)), 1e-12))

    out = {"label": args.label, "K": K, "r_modes": r, "r99_les": r99, "grid_n": n,
           "n_frames": len(tidx), "n_basis_snapshots": len(sel),
           "pod_ke": float(np.mean(ke_e)), "pod_u": float(np.mean(u_e)),
           "pod_om": float(np.mean(om_e)),
           "oracle_projection_rel_l2": float(np.mean(proj_e))}
    print(f"{args.label:16s} K={K:>3d} r={r:>3d} (r99_LES={r99}) grid={n} | "
          f"KE {out['pod_ke']*100:6.2f}%  u {out['pod_u']*100:6.2f}%  om {out['pod_om']*100:6.2f}%"
          f"   [basis-projection floor {out['oracle_projection_rel_l2']*100:5.2f}%]")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
