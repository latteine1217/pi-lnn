"""Pin down the spectral-support scale n99 used in the K ~ pi n99^2 / 2 reading.

What: report n99(t) over the observation window of each cross-Re DNS under four
      candidate definitions, then compare each against the measured sensor budget.
Why:  the sweep quoted a single "saturated" n99, but these flows decay, so n99 falls
      through the window (Re=1e3 runs 8 -> 3). Which value enters the rule was never
      fixed, and the rule cannot be tested until it is. A definition is only useful
      here if it is computable without a DNS field at deployment time in spirit —
      all four are evaluated offline on the reference, as a diagnostic.

Definitions compared:
  n99_start   value at the first frame
  n99_end     value at the last frame
  n99_tmean   arithmetic mean of n99(t) over the window
  n99_emean   energy-weighted mean, sum_t KE(t) n99(t) / sum_t KE(t) — weights the
              frames that dominate the window-integrated error the metrics report
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pi_con.spectral import radial_energy_spectrum  # noqa: E402


def radial_spectrum(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return E(k) for k = 1 … n/2. 實作見 pi_con.spectral（全 repo 單一份）。"""
    _, e_k = radial_energy_spectrum(u, v)
    return e_k


def n_frac(E: np.ndarray, frac: float) -> int:
    """Smallest integer wavenumber whose cumulative energy reaches `frac`.

    `E` is indexed from k=1, so the array index must be shifted by one to become
    a wavenumber. The earlier local spectrum started its bins at k=0, which made
    index and wavenumber coincide; the k=0 shell carries no energy on a periodic
    domain (measured 3.4e-38 of KE), so dropping it leaves the returned n99
    unchanged once the shift is applied.
    """
    tot = E.sum()
    if tot <= 0:
        return 0
    c = np.cumsum(E) / tot
    hit = np.nonzero(c >= frac)[0]
    return int(hit[0]) + 1 if hit.size else len(E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dns", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--frac", type=float, default=0.99)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    d = np.load(args.dns, allow_pickle=True).item()
    u, v, t = d["u"], d["v"], np.asarray(d["time"], float)
    ke = np.asarray(d["diagnostics"]["kinetic_energy"], float)[: len(t)]

    n99 = np.array([n_frac(radial_spectrum(u[i], v[i]), args.frac) for i in range(len(t))],
                   dtype=float)

    defs = {
        "n99_start": float(n99[0]),
        "n99_end": float(n99[-1]),
        "n99_tmean": float(n99.mean()),
        "n99_emean": float(np.sum(ke * n99) / np.sum(ke)),
    }
    out = {"label": args.label, "n_frames": len(t),
           "n99_min": float(n99.min()), "n99_max": float(n99.max()), **defs,
           "K_pred": {k: float(np.pi * v_ * v_ / 2.0) for k, v_ in defs.items()}}

    print(f"{args.label:10s} n99: start {defs['n99_start']:.0f}  end {defs['n99_end']:.0f}  "
          f"tmean {defs['n99_tmean']:.1f}  emean {defs['n99_emean']:.1f}   "
          f"-> K* pred  start {out['K_pred']['n99_start']:6.0f}  end {out['K_pred']['n99_end']:5.0f}  "
          f"tmean {out['K_pred']['n99_tmean']:5.0f}  emean {out['K_pred']['n99_emean']:5.0f}")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
