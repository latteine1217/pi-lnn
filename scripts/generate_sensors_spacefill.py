"""generate_sensors_spacefill.py — coverage-optimal (DNS-free) sensor placement.

Farthest-point sampling (FPS) with periodic (toroidal) distance on the grid:
greedily places each sensor in the largest existing gap, maximising the minimum
inter-sensor distance => minimal coverage gap. This is the maximally even,
alias-free, DNS-free placement (no LES needed). Sensor values come from the DNS.

Output schema matches sensors_qrpivot_*.{json,npz} so it drops into the stable
configs and passes tests/test_sensor_axis_convention.py.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def fps_periodic(coords: np.ndarray, K: int, L: float, seed: int) -> np.ndarray:
    """Farthest-point sampling with toroidal distance. Returns selected row indices."""
    rng = np.random.default_rng(seed)
    n = coords.shape[0]
    sel = [int(rng.integers(n))]
    dmin = np.full(n, np.inf)
    for _ in range(K - 1):
        diff = np.abs(coords - coords[sel[-1]])
        diff = np.minimum(diff, L - diff)          # periodic wrap
        d2 = (diff ** 2).sum(1)
        dmin = np.minimum(dmin, d2)
        sel.append(int(np.argmax(dmin)))
    return np.asarray(sel, dtype=int)


def main() -> None:
    ap = argparse.ArgumentParser(description="Coverage-optimal FPS sensor placement (DNS-free)")
    ap.add_argument("--dns", required=True, help="DNS .npy (sensor values + grid)")
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42, help="seed for FPS first point")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    raw = np.load(args.dns, allow_pickle=True).item()
    x = np.asarray(raw["x"], np.float64); y = np.asarray(raw["y"], np.float64)
    u = np.asarray(raw["u"], np.float32); v = np.asarray(raw["v"], np.float32)
    t = np.asarray(raw["time"], np.float32)
    N = len(x); L = float(x[-1] - x[0] + (x[1] - x[0]))   # periodic box length

    # candidate grid: cand[i*N + j] = (x[i], y[j])  (row-major, axis_1=x)
    X, Y = np.meshgrid(x, y, indexing="ij")
    cand = np.stack([X.ravel(), Y.ravel()], 1)
    sel_flat = fps_periodic(cand, args.K, L, args.seed)
    x_idx, y_idx = np.unravel_index(sel_flat, (N, N))      # x_idx=i, y_idx=j
    coords = np.stack([x[x_idx], y[y_idx]], 1)             # [K,2]

    # coverage diagnostic
    dd = np.abs(coords[:, None] - coords[None]); dd = np.minimum(dd, L - dd)
    nn = np.sqrt((dd ** 2).sum(-1)); np.fill_diagonal(nn, 9)
    gg = np.abs(cand[:, None] - coords[None]); gg = np.minimum(gg, L - gg)
    cover_max = np.sqrt((gg ** 2).sum(-1)).min(1).max()
    print(f"FPS K={args.K}: min inter-sensor dist={nn.min():.4f}, "
          f"max coverage gap={cover_max:.4f} (T50 ref 0.144, random 0.229)")

    # sensor values from DNS: convention u_full[t, x_idx, y_idx]
    sensor_u = u[:, x_idx, y_idx].T                        # [K, T]
    sensor_v = v[:, x_idx, y_idx].T

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    jpath = out / f"sensors_{args.tag}.json"
    npath = out / f"sensors_{args.tag}_dns_values.npz"
    meta = {
        "K": args.K, "resolution": f"{N}x{N}",
        "spatial_downsample_res": f"{N}x{N}", "spatial_downsample_stride": 1,
        "method": "farthest_point_sampling_periodic",
        "features": [], "time_stride": 1,
        "time_range": [float(t[0]), float(t[-1])], "time_steps": int(len(t)),
        "selected_coordinates": coords.tolist(),
        "source_file": str(args.dns), "seed": args.seed,
        "sensor_dt": float(t[1] - t[0]), "sensor_time_points": int(len(t)),
        "engineering_pipeline_note":
            "DNS-free coverage-optimal placement (FPS, no LES); sensor values from DNS.",
    }
    json.dump(meta, open(jpath, "w"), indent=2)
    np.savez(npath, time=t, u=sensor_u, v=sensor_v)
    print(f"Saved JSON: {jpath}\nSaved NPZ:  {npath}\nDone.")


if __name__ == "__main__":
    main()
