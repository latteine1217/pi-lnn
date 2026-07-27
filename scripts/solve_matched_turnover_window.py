"""Solve the T_end that matches a target eddy-turnover count across Reynolds numbers.

What: 給一支（夠長的）DNS probe，解出 T* 使 T* / tau_eddy(T*) = target，
      其中 tau_eddy(t) = (L/k_f) / u_rms(t)。
Why:  cross-Re 的 sensor-budget 比較必須在「相同動力學年齡」下做。固定 T=5 跨 Re
      並不等價（實測 Re=1e4 為 5.0 個 turnover，Re=1e3 只有 2.3 個），低 Re 會因為
      時間演化較少而顯得容易，汙染 K*(Re)。本腳本產生 path-A 的 matched window。

T/tau(t) 隨 t 單調上升時解唯一；否則回報所有交點，由人判讀。
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", type=str, help="long DNS probe .npy (dict schema)")
    ap.add_argument("--target", type=float, default=5.0,
                    help="target T_end/tau_eddy (default: Re=1e4 reference value)")
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    d = np.load(args.probe, allow_pickle=True).item()
    cfg = d["config"]
    u, v, t = d["u"], d["v"], np.asarray(d["time"], dtype=float)
    L, k_f, nu = cfg["L"], cfg["k_f"], cfg["nu"]

    u_rms = np.sqrt(np.mean(u**2 + v**2, axis=(1, 2)))
    tau = (L / k_f) / u_rms
    ratio = np.divide(t, tau, out=np.zeros_like(t), where=tau > 0)

    # Locate every crossing of `target`; report all, flag non-monotonicity.
    cross = []
    for i in range(1, len(t)):
        a, b = ratio[i - 1], ratio[i]
        if (a - args.target) * (b - args.target) <= 0 and a != b:
            w = (args.target - a) / (b - a)
            cross.append(float(t[i - 1] + w * (t[i] - t[i - 1])))

    monotone = bool(np.all(np.diff(ratio) >= -1e-12))

    out = {
        "probe": Path(args.probe).name,
        "Re": 1.0 / nu, "nu": nu, "N": int(cfg["N"]), "probe_T_end": float(cfg["T_end"]),
        "target_ratio": args.target,
        "ratio_monotone_in_t": monotone,
        "ratio_at_probe_end": float(ratio[-1]),
        "T_star_crossings": cross,
        "T_star": cross[0] if len(cross) == 1 else (cross[-1] if cross else None),
        "note": ("ratio never reaches target within probe horizon; extend the probe"
                 if not cross else
                 ("unique crossing" if len(cross) == 1 else "multiple crossings — inspect manually")),
        "trace": [
            {"t": float(t[i]), "u_rms": float(u_rms[i]),
             "tau_eddy": float(tau[i]), "T_over_tau": float(ratio[i])}
            for i in range(0, len(t), max(1, len(t) // 20))
        ],
    }

    print(json.dumps(out, indent=2))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
