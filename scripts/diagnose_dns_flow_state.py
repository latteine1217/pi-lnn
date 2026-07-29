"""DNS flow-state diagnostic for the cross-Reynolds sensor-budget study.

What: 對一支 Kolmogorov DNS (.npy, dict schema) 報告「這個流場到底是什麼狀態」——
      衰減程度、譜帶寬、observability wall 內的能量比例、以及是否已鬆弛到層流解。
Why:  cross-Re 的 K*(Re) 曲線只有在各 Re 的流場「同樣是非平庸湍流」時才有意義。
      低 Re 的短瞬態會衰減成層流 Kolmogorov 解，此時重建變容易是因為流場退化，
      與 sensor 資訊量無關。本腳本是花訓練成本前的 gate。

輸出純診斷，不做任何判定式宣稱；門檻判讀留給人。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pi_con.spectral import radial_energy_spectrum  # noqa: E402


def radial_spectrum(u, v, L=1.0):
    """Return (n_bins, E(n)) with n the integer wavenumber (physical k = 2*pi*n/L).

    實作見 pi_con.spectral（全 repo 單一份）。原本此處的 bin 自 n=0 起，含 mean
    mode；週期域上該 shell 不帶能量（實測佔 KE 的 3.4e-38），且下游的
    band_fraction / spectral_bandwidth 都以 `bins` 作遮罩而非拿索引當波數，故移除
    該 bin 不改變任何輸出。
    """
    return radial_energy_spectrum(u, v)


def band_fraction(bins, E, n_cut):
    """Fraction of energy at integer wavenumber n <= n_cut."""
    tot = E.sum()
    if tot <= 0:
        return float("nan")
    return float(E[bins <= n_cut].sum() / tot)


def spectral_bandwidth(bins, E, frac):
    """Smallest n whose cumulative energy reaches `frac` of the total."""
    tot = E.sum()
    if tot <= 0:
        return float("nan")
    c = np.cumsum(E) / tot
    hit = np.nonzero(c >= frac)[0]
    return int(bins[hit[0]]) if hit.size else int(bins[-1])


def laminar_reference(nu, A, k_f, L=1.0):
    """Analytic steady Kolmogorov solution u = u_amp * sin(2*pi*k_f*y/L), v = 0.

    Balance: nu * (2*pi*k_f/L)^2 * u_amp = A  ->  u_amp = A / (nu * (2*pi*k_f/L)^2).
    Domain-averaged KE = <0.5*u^2> = u_amp^2 / 4.
    """
    kphys = 2.0 * np.pi * k_f / L
    u_amp = A / (nu * kphys**2)
    return u_amp, u_amp**2 / 4.0


def forcing_mode_fraction(u, v, k_f):
    """Energy fraction sitting in the pure forcing mode (kx=0, |ky|=k_f) of u.

    A value near 1 means the field has collapsed onto the laminar Kolmogorov profile.
    Assumes the axis convention u[x, y] (axis_1 = x, axis_2 = y), per project protocol.
    """
    N = u.shape[-1]
    uh = np.fft.fft2(u) / N**2
    vh = np.fft.fft2(v) / N**2
    e = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2)
    tot = e.sum()
    if tot <= 0:
        return float("nan")
    # kx index 0 (no x-dependence), ky = +/- k_f
    sel = e[0, k_f] + e[0, -k_f]
    return float(sel / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dns", type=str, help="path to DNS .npy (dict schema)")
    ap.add_argument("--k-list", type=int, nargs="+", default=[100, 50, 10],
                    help="sensor counts K; observability wall n <= sqrt(2K/pi)")
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    d = np.load(args.dns, allow_pickle=True).item()
    cfg = d["config"]
    u, v = d["u"], d["v"]
    t = d["time"]
    nu, A, k_f, L = cfg["nu"], cfg["A"], cfg["k_f"], cfg["L"]
    Re = 1.0 / nu

    ke = np.asarray(d["diagnostics"]["kinetic_energy"], dtype=float)
    ens = np.asarray(d["diagnostics"]["enstrophy"], dtype=float)

    u_amp_lam, ke_lam = laminar_reference(nu, A, k_f, L)

    # u_rms and eddy turnover at the forcing scale (length L/k_f).
    u_rms = np.sqrt(np.mean(u**2 + v**2, axis=(1, 2)))
    tau_eddy = (L / k_f) / u_rms

    rows = []
    for i in (0, len(t) // 2, len(t) - 1):
        bins, E = radial_spectrum(u[i], v[i], L)
        rows.append({
            "t": float(t[i]),
            "KE": float(ke[i]),
            "enstrophy": float(ens[i]),
            "u_rms": float(u_rms[i]),
            "tau_eddy": float(tau_eddy[i]),
            "n_99": spectral_bandwidth(bins, E, 0.99),
            "n_999": spectral_bandwidth(bins, E, 0.999),
            "forcing_mode_frac": forcing_mode_fraction(u[i], v[i], k_f),
            "band_frac": {str(K): band_fraction(bins, E, np.sqrt(2 * K / np.pi))
                          for K in args.k_list},
        })

    out = {
        "file": str(Path(args.dns).name),
        "Re": Re, "nu": nu, "N": int(cfg["N"]), "T_end": float(cfg["T_end"]),
        "n_snapshots": int(len(t)), "A": A, "k_f": k_f,
        "KE_t0": float(ke[0]), "KE_tend": float(ke[-1]),
        "KE_decay_frac": float(1.0 - ke[-1] / ke[0]),
        "enstrophy_decay_frac": float(1.0 - ens[-1] / ens[0]),
        "laminar_u_amp": float(u_amp_lam), "laminar_KE": float(ke_lam),
        "KE_tend_over_KE_laminar": float(ke[-1] / ke_lam),
        "T_end_over_tau_eddy_tend": float(cfg["T_end"] / tau_eddy[-1]),
        "snapshots": rows,
    }

    print(json.dumps(out, indent=2))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
