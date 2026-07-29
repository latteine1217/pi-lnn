"""Ensemble Kalman Filter reconstruction from the same sparse sensors.

What: assimilate the K sensor time series into an ensemble of forward Navier-Stokes
      integrations and score the analysis mean with the operator evaluator's formulas.
Why:  the interpolation and reduced-order baselines carry no dynamics. Ensemble
      assimilation does: it is the standard estimator when the governing equations are
      known and the observations are sparse, and it is the strongest fair competitor
      because it uses the same information the operator is given — the sensors and the
      PDE — and no reference field.

Forward model: pseudo-spectral vorticity formulation of 2D Navier-Stokes with the
Kolmogorov body force, 2/3 dealiasing, RK4 in time. `--self-test` integrates a DNS
snapshot forward and reports the drift against the DNS itself; the filter results are
only meaningful if that drift is small over one observation interval.

Filter: stochastic EnKF with perturbed observations, in the ensemble subspace so the
state covariance is never formed. Inflation and observation noise are set from the
command line, not tuned against the reference.
"""

import argparse
import json
from pathlib import Path

import numpy as np


class Spectral2D:
    """Vorticity-form 2D NS: d(omega)/dt + J(psi, omega) = nu lap(omega) + curl(f)."""

    def __init__(self, n: int, nu: float, L: float, A: float, k_f: int):
        self.n, self.nu, self.L = n, nu, L
        k = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
        self.KX, self.KY = np.meshgrid(k, k, indexing="ij")
        self.K2 = self.KX**2 + self.KY**2
        self.K2inv = 1.0 / np.where(self.K2 == 0, 1.0, self.K2)
        self.K2inv[0, 0] = 0.0
        kmax = (2.0 / 3.0) * np.max(np.abs(k))
        self.mask = (np.abs(self.KX) <= kmax) & (np.abs(self.KY) <= kmax)
        y = np.arange(n) * (L / n)
        # f = (A sin(2 pi k_f y / L), 0)  ->  curl(f) = -d f_x / dy
        fx = A * np.sin(2.0 * np.pi * k_f * y / L)
        curl_f = -np.gradient(fx, L / n)
        self.curl_f_hat = np.fft.fft2(np.broadcast_to(curl_f[None, :], (n, n)).copy())

    def uv(self, w_hat):
        psi_hat = w_hat * self.K2inv
        u = np.real(np.fft.ifft2(1j * self.KY * psi_hat))
        v = np.real(np.fft.ifft2(-1j * self.KX * psi_hat))
        return u, v

    def rhs(self, w_hat):
        u, v = self.uv(w_hat)
        wx = np.real(np.fft.ifft2(1j * self.KX * w_hat))
        wy = np.real(np.fft.ifft2(1j * self.KY * w_hat))
        adv = np.fft.fft2(u * wx + v * wy) * self.mask
        return -adv - self.nu * self.K2 * w_hat + self.curl_f_hat

    def step(self, w_hat, dt):
        k1 = self.rhs(w_hat)
        k2 = self.rhs(w_hat + 0.5 * dt * k1)
        k3 = self.rhs(w_hat + 0.5 * dt * k2)
        k4 = self.rhs(w_hat + dt * k3)
        return (w_hat + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)) * self.mask

    def advance(self, w_hat, t_span, dt):
        n_steps = max(1, int(np.ceil(t_span / dt)))
        h = t_span / n_steps
        for _ in range(n_steps):
            w_hat = self.step(w_hat, h)
        return w_hat


def vort_from_uv(u, v, solver):
    uh, vh = np.fft.fft2(u), np.fft.fft2(v)
    return 1j * solver.KX * vh - 1j * solver.KY * uh


def vorticity_fd(u, v, dx):
    return ((np.roll(v, -1, axis=-2) - np.roll(v, 1, axis=-2))
            - (np.roll(u, -1, axis=-1) - np.roll(u, 1, axis=-1))) / (2 * dx)


def block_avg(field, factor):
    f = int(factor)
    if f == 1:
        return field
    nx, ny = field.shape[-2] // f, field.shape[-1] // f
    return field.reshape(*field.shape[:-2], nx, f, ny, f).mean(axis=(-3, -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dns", required=True)
    ap.add_argument("--sensor-json", required=True)
    ap.add_argument("--sensor-npz", required=True)
    ap.add_argument("--block-factor", type=int, required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--ensemble", type=int, default=40)
    ap.add_argument("--obs-noise", type=float, default=0.01,
                    help="observation error std, as a fraction of the sensor r.m.s.")
    ap.add_argument("--inflation", type=float, default=1.05)
    ap.add_argument("--localization", type=float, default=2.0,
                    help="localization radius in units of mean sensor spacing L/sqrt(K); "
                         "0 disables. A small ensemble cannot constrain a state of this "
                         "dimension without it — the analysis is destroyed by spurious "
                         "long-range sample correlations.")
    ap.add_argument("--dt", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--self-test", action="store_true",
                    help="integrate a DNS snapshot forward and report drift; no filtering")
    ap.add_argument("--label", default="")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    d = np.load(args.dns, allow_pickle=True).item()
    cfg = d["config"]
    x_full = np.asarray(d["x"], np.float64)
    L = float(x_full[-1] - x_full[0] + (x_full[1] - x_full[0]))
    n_store = len(x_full)
    n = n_store // args.block_factor            # filter runs on the evaluation grid
    dx = L / n
    solver = Spectral2D(n, float(cfg["nu"]), L, float(cfg["A"]), int(cfg["k_f"]))
    t_all = np.asarray(d["time"], np.float64)

    if args.self_test:
        u0 = block_avg(np.asarray(d["u"][0], np.float64), args.block_factor)
        v0 = block_avg(np.asarray(d["v"][0], np.float64), args.block_factor)
        w = vort_from_uv(u0, v0, solver)
        print(f"self-test on {n}^2, nu={cfg['nu']:.1e}, dt={args.dt}")
        for j in (1, 2, 5, 10):
            if j >= len(t_all):
                break
            w_j = solver.advance(w.copy(), float(t_all[j] - t_all[0]), args.dt)
            up, vp = solver.uv(w_j)
            ur = block_avg(np.asarray(d["u"][j], np.float64), args.block_factor)
            vr = block_avg(np.asarray(d["v"][j], np.float64), args.block_factor)
            e = np.sqrt(np.sum((up - ur) ** 2 + (vp - vr) ** 2)) / np.sqrt(np.sum(ur**2 + vr**2))
            print(f"  t={t_all[0]:.3f} -> {t_all[j]:.3f} ({j} obs interval(s)): "
                  f"velocity drift {e*100:6.3f}%")
        return

    meta = json.loads(Path(args.sensor_json).read_text())
    pts = np.asarray(meta["selected_coordinates"], np.float64)
    xg = x_full if args.block_factor == 1 else x_full.reshape(-1, args.block_factor).mean(1)
    xi = np.argmin(np.abs(pts[:, 0:1] - xg[None, :]), axis=1)
    yi = np.argmin(np.abs(pts[:, 1:2] - xg[None, :]), axis=1)

    npz = np.load(args.sensor_npz)
    su, sv = np.asarray(npz["u"], np.float64), np.asarray(npz["v"], np.float64)
    tidx = np.arange(0, su.shape[1], args.stride)

    # --- covariance localization: taper the state-observation cross-covariance ---
    K_sensors = int(meta["K"])
    if args.localization > 0:
        gx = xg[:, None] * np.ones((1, n))
        gy = np.ones((n, 1)) * xg[None, :]
        gxy = np.stack([gx.ravel(), gy.ravel()], axis=1)          # [n^2, 2]
        dd = np.abs(gxy[:, None, :] - pts[None, :, :])
        dd = np.minimum(dd, L - dd)
        dist = np.sqrt((dd ** 2).sum(-1))                          # [n^2, K]
        L_loc = args.localization * L / np.sqrt(K_sensors)
        taper = np.exp(-0.5 * (dist / L_loc) ** 2)
        rho = np.concatenate([np.concatenate([taper, taper], axis=1)] * 2, axis=0)
    else:
        rho = None

    rng = np.random.default_rng(args.seed)
    N_e = args.ensemble
    # Ensemble start: band-limited random fields matched to the sensor r.m.s. at t0.
    # No reference field is used.
    target_rms = float(np.sqrt(np.mean(su[:, tidx[0]] ** 2 + sv[:, tidx[0]] ** 2)))
    ens = []
    for _ in range(N_e):
        wh = np.fft.fft2(rng.normal(size=(n, n)))
        wh *= (solver.K2 > 0) & (solver.K2 < (2 * np.pi * 8 / L) ** 2)
        u_t, v_t = solver.uv(wh)
        s = target_rms / max(np.sqrt(np.mean(u_t**2 + v_t**2)), 1e-12)
        ens.append(wh * s)
    ens = np.stack(ens)

    obs_std = args.obs_noise * target_rms
    ke_e, u_e, om_e = [], [], []
    for step_i, ti in enumerate(tidx):
        if step_i > 0:
            span = float(t_all[ti] - t_all[tidx[step_i - 1]])
            for m in range(N_e):
                ens[m] = solver.advance(ens[m], span, args.dt)

        fields = [solver.uv(ens[m]) for m in range(N_e)]
        Xu = np.stack([f[0].ravel() for f in fields], 1)
        Xv = np.stack([f[1].ravel() for f in fields], 1)
        X = np.concatenate([Xu, Xv], 0)                       # [2 n^2, N_e]
        xbar = X.mean(1, keepdims=True)
        Xa = (X - xbar) * args.inflation

        flat = xi * n + yi
        rows = np.concatenate([flat, flat + n * n])
        HX = X[rows, :]
        HXa = Xa[rows, :]
        y = np.concatenate([su[:, ti], sv[:, ti]])
        Y = y[:, None] + rng.normal(0.0, obs_std, size=(len(y), N_e))

        C = (HXa @ HXa.T) / (N_e - 1) + (obs_std**2) * np.eye(len(y))
        W = np.linalg.solve(C, Y - HX)                        # [2K, N_e]
        PHt = (Xa @ HXa.T) / (N_e - 1)                        # [2 n^2, 2K]
        if rho is not None:
            PHt = PHt * rho
        X = X + PHt @ W

        up = X[: n * n].mean(1).reshape(n, n)
        vp = X[n * n:].mean(1).reshape(n, n)
        for m in range(N_e):
            ens[m] = vort_from_uv(X[: n * n, m].reshape(n, n),
                                  X[n * n:, m].reshape(n, n), solver) * solver.mask

        ur = block_avg(np.asarray(d["u"][ti], np.float64), args.block_factor)
        vr = block_avg(np.asarray(d["v"][ti], np.float64), args.block_factor)
        omr, omp = vorticity_fd(ur, vr, dx), vorticity_fd(up, vp, dx)
        ke_r = 0.5 * np.mean(ur**2 + vr**2)
        ke_e.append(abs(0.5 * np.mean(up**2 + vp**2) - ke_r) / max(ke_r, 1e-12))
        u_e.append(np.sqrt(np.sum((up - ur) ** 2)) / max(np.sqrt(np.sum(ur**2)), 1e-12))
        om_e.append(np.sqrt(np.sum((omp - omr) ** 2)) / max(np.sqrt(np.sum(omr**2)), 1e-12))

    # A filter started from a random ensemble needs a spin-up before its error is
    # representative; averaging it in reports the cold start, not the filter.
    n_sp = max(1, int(0.5 * len(ke_e)))
    out = {"label": args.label, "K": int(meta["K"]), "grid_n": n, "n_frames": len(tidx),
           "enkf_ke_converged": float(np.mean(ke_e[n_sp:])),
           "enkf_u_converged": float(np.mean(u_e[n_sp:])),
           "enkf_om_converged": float(np.mean(om_e[n_sp:])),
           "enkf_u_final": float(u_e[-1]),
           "ensemble": N_e, "obs_noise_frac": args.obs_noise, "inflation": args.inflation,
           "localization_radii_of_spacing": args.localization,
           "enkf_ke": float(np.mean(ke_e)), "enkf_u": float(np.mean(u_e)),
           "enkf_om": float(np.mean(om_e))}
    print(f"{args.label:16s} K={out['K']:>3d} N_e={N_e} | full-window KE {out['enkf_ke']*100:6.2f}% "
          f"u {out['enkf_u']*100:6.2f}%  ||  post-spin-up KE {out['enkf_ke_converged']*100:6.2f}% "
          f"u {out['enkf_u_converged']*100:6.2f}% om {out['enkf_om_converged']*100:6.2f}% "
          f"(final u {out['enkf_u_final']*100:5.2f}%)")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
