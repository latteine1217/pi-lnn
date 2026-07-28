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
from pathlib import Path

import numpy as np


def radial_spectrum(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    n = u.shape[-1]
    uh, vh = np.fft.fft2(u) / n**2, np.fft.fft2(v) / n**2
    e = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2)
    kx = np.fft.fftfreq(n, d=1.0 / n)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    idx = np.round(np.sqrt(KX**2 + KY**2)).astype(int)
    return np.array([e[idx == k].sum() for k in range(n // 2 + 1)])


def n_frac(E: np.ndarray, frac: float) -> int:
    tot = E.sum()
    if tot <= 0:
        return 0
    c = np.cumsum(E) / tot
    hit = np.nonzero(c >= frac)[0]
    return int(hit[0]) if hit.size else len(E) - 1


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
