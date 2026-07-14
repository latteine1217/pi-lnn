#!/usr/bin/env python3
"""plot_sensor_recoverability.py — per-wavenumber reconstruction recoverability
(Fourier pseudo-inverse accuracy) for K=100 / K=200 QR-pivot sensors.

取代先前誤植為「condition number κ」的 sensor_spectral_coverage 圖。實際量是
recoverability acc(k) = 1 − ||û_recon − û_true||² / ||û_true||²(孤立 k-shell),
即 K 個 sensor 能否唯一辨識該 shell 的 M_k 個模態:M_k ≤ K → 過定(acc≈1),
M_k > K → 欠定(acc 下降)。這才是高 k 崩壞的正確物理(condition number 抓不到)。

重用 generate_sensors_qrpivot.fourier_pseudoinverse_accuracy,與原圖同一計算。
輸出:向量 PDF、Times/serif、fonttype 42(順帶解 Type-3)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_sensors_qrpivot import fourier_pseudoinverse_accuracy  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "font.size": 11,
})


def acc_curve(coords, u_full, x_arr, snap_idx, N, k_eval):
    acc = np.zeros(k_eval)
    for t in snap_idx:
        fft = np.fft.fft2(u_full[t].astype(np.float64)) / (N * N)
        kk, a = fourier_pseudoinverse_accuracy(coords, fft, x_arr, k_max_eval=k_eval)
        acc += a
    return kk, acc / len(snap_idx)


def first_below(k, acc, thr):
    idx = np.where(acc < thr)[0]
    return int(k[idx[0]]) if len(idx) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dns", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sensors", nargs="+", required=True)
    ap.add_argument("--nsnap", type=int, default=8)
    ap.add_argument("--keval", type=int, default=50)
    a = ap.parse_args()

    raw = np.load(a.dns, allow_pickle=True).item()
    u = raw["u"].astype(np.float64)
    x = raw["x"].astype(np.float64)
    T, N, _ = u.shape
    snap = np.linspace(T // 10, T - 1, a.nsnap, dtype=int)
    print(f"DNS T={T} N={N}; snapshots={list(snap)}")

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    palette = {"100": "#0072B2", "200": "#D55E00", "400": "#009E73"}  # Okabe--Ito K

    for jp in a.sensors:
        m = json.load(open(jp, encoding="utf-8"))
        Kc = int(m["K"])
        coords = np.array(m["selected_coordinates"], dtype=float)
        k, acc = acc_curve(coords, u, x, snap, N, a.keval)
        c = palette.get(str(Kc), "#333333")
        ax.plot(k, acc, color=c, lw=1.8, label=f"$K={Kc}$")
        knyq = np.sqrt(Kc / np.pi)
        ax.axvline(knyq, ls=":", color=c, lw=0.9, alpha=0.7)
        print(f"K={Kc}: k_nyq=sqrt(K/pi)={knyq:.2f}; "
              f"first acc<0.8 at k={first_below(k, acc, 0.8)}; "
              f"first acc<0.5 at k={first_below(k, acc, 0.5)}")
        for kq in [5, 8, 12, 16, 20, 24, 32, 40]:
            i = np.where(k == kq)[0]
            if len(i):
                print(f"    acc(k={kq}) = {acc[i[0]]:.3f}")

    ax.axhline(0.8, ls="--", color="green", lw=0.9)
    ax.axhline(0.5, ls=":", color="orange", lw=0.9)
    ax.set_xlabel(r"Wavenumber $k$ (1/m)")
    ax.set_ylabel(r"Reconstruction recoverability")
    ax.set_xlim(1, a.keval)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
