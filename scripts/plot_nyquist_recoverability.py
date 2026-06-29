#!/usr/bin/env python3
"""
Nyquist information-theoretic ceiling — recoverability plot.

What this script answers
------------------------
Given K real-valued point sensors uniformly placed on a 2D N×N grid, what is
the maximum fraction of the DNS kinetic energy that ANY reconstruction
algorithm (linear regression, PINN, DeepONet, ...) can recover from the K
sensor measurements?

The answer is purely information-theoretic:

  K sensors  →  recoverable subspace has dimension K
              →  by isotropic Nyquist, this corresponds to all Fourier modes in
                 the disk  |k| ≤ k_Nyq(K) = √(K/π)
              →  ceiling = F_DNS(k_Nyq(K)),  where  F_DNS  is the cumulative
                 DNS energy CDF over radial wavenumber.

So the plot is "DNS energy distribution, with the K sensors' Nyquist line
drawn on top, and the ceiling read off from the intersection".

Why we don't show an algorithm-specific recovery curve
------------------------------------------------------
An earlier draft used band-limited least squares (LSQ) inside the Nyquist disk
to draw a per-shell recovery curve.  Two problems:
  1.  When  K ≈ M_disk,  the LSQ system has tiny singular values  (∝ 1/N²),
      so even moderate conditioning (κ ≈ 400) amplifies sensor signal by
      thousands and  C_rec  blows up unless an  rcond  cutoff is tuned by K.
  2.  Tuning  rcond  makes the curve algorithm-dependent and weakens the
      "information-theoretic ceiling" message.
The DNS-energy formulation here is exact, K-clean, and matches the paper's
"Nyquist ceiling" framing.

Axis convention
---------------
Following docs/CLAUDE.md KNOWN_PITFALLS:
  u_full[t, x_idx, y_idx]   (row-major: row = axis_1 = x)
  flat = x_idx * N + y_idx
(Not actually used in this script — sensor placement is no longer needed —
but kept here for cross-reference with related sensor-loading utilities.)

Outputs
-------
  docs/figures/nyquist_recoverability.png
  docs/figures/nyquist_recoverability.pdf
  docs/figures/nyquist_recoverability_data.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pi_con.plot_style import apply_journal_rcparams  # noqa: E402

# ----------------------------- Config defaults -----------------------------

DEFAULT_DNS = "data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy"
# 20 snapshots evenly spaced in T=0..5 (DNS has 201 frames @ dt=0.025)
N_SNAPSHOTS = 20

# ----------------------------- IO -----------------------------------------


def load_dns(path: str | Path) -> dict:
    data = np.load(path, allow_pickle=True).item()
    return {
        "u": np.asarray(data["u"]),
        "v": np.asarray(data["v"]),
        "x": np.asarray(data["x"]),
        "y": np.asarray(data["y"]),
        "time": np.asarray(data["time"]),
    }


# ----------------------------- Core math ----------------------------------


def make_shell_index(N: int) -> np.ndarray:
    """Return integer radial shell index for each Fourier mode (FFT order)."""
    kx = np.fft.fftfreq(N, d=1.0 / N).astype(np.int64)
    ky = np.fft.fftfreq(N, d=1.0 / N).astype(np.int64)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    return np.round(np.sqrt(KX ** 2 + KY ** 2)).astype(np.int64)


# ----------------------------- Plotting -----------------------------------

# K_COLORS — Wong colorblind-safe palette（與其他 result 圖一致）
K_COLORS = {
    100: "#0072B2",  # Wong blue
    200: "#D55E00",  # Wong vermillion
    400: "#009E73",  # Wong bluish green
}


def plot_recoverability(curves: dict[int, dict], k_axis: np.ndarray, out_png: Path, out_pdf: Path):
    """Two-panel info-theoretic figure (no algorithm dependence)."""
    from matplotlib.lines import Line2D

    apply_journal_rcparams()  # sans-serif，與其他 result 圖一致
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(5.5, 2.8), constrained_layout=True)

    # F_dns and per-shell energy are K-independent (same DNS field) → use any K
    any_K = next(iter(curves))
    F_dns = curves[any_K]["F_dns_mean"]  # cumulative
    # Recover per-shell energy from the cumulative
    E_shell = np.diff(curves[any_K]["F_dns_mean"], prepend=0.0)  # fractional per shell
    # Absolute per-shell energy (un-normalized) is more familiar; reconstruct from cumulative
    # ratio × total. Total is implicit — express in linear units relative to total energy.
    E_shell_abs = E_shell  # fraction of total energy in shell

    # ---- Panel (a): DNS energy spectrum E(k) with Nyquist cutoffs ----
    # Plot only k ≥ 1 (k=0 is mean flow, zero for zero-mean Kolmogorov)
    mask = k_axis >= 1
    axL.loglog(k_axis[mask], E_shell_abs[mask], color="black", linewidth=1.8,
               label=r"DNS $E(k)$")
    # Enstrophy-cascade k^{-3} reference (Kraichnan forward cascade for k>k_f),
    # anchored at k=5. (NOT k^{-5/3}: that is the inverse-energy cascade, k<k_f.)
    k_ref = np.array([3, 30], dtype=float)
    # anchor: at k=5 take E(5) value
    k_anchor = 5
    if k_anchor < len(E_shell_abs) and E_shell_abs[k_anchor] > 0:
        E_anchor = E_shell_abs[k_anchor]
        E_ref = E_anchor * (k_ref / k_anchor) ** (-3.0)
        axL.loglog(k_ref, E_ref, color="0.45", linestyle=":", linewidth=1.0,
                   label=r"$k^{-3}$")
    # Nyquist verticals
    for K in sorted(curves.keys()):
        color = K_COLORS[K]
        k_nyq = float(np.sqrt(K / np.pi))
        axL.axvline(k_nyq, color=color, linestyle="--", linewidth=1.1, alpha=0.85)
    # Legend: DNS line + reference slope + a single generic Nyquist-cutoff entry.
    # 各 K 的 k_Nyq 數值與配色改由 (b) 的 Nyquist-ceilings 表說明，避免兩處重複、縮短 legend。
    handles_main, labels_main = axL.get_legend_handles_labels()
    handles_main.append(Line2D([0], [0], color="0.5", linestyle="--", linewidth=1.1))
    labels_main.append(r"$k_{\mathrm{Nyq}}(K)$")
    axL.legend(handles_main, labels_main, loc="upper right", frameon=False, fontsize=6.5)
    axL.set_xlim(1, k_axis.max())
    axL.set_xlabel(r"Wavenumber  $k = \|\mathbf{k}\|$  (1/m)")
    axL.set_ylabel(r"Normalized energy in shell  $E(k)/E_{\mathrm{tot}}$")
    axL.set_title("(a)", loc="left", fontweight="bold")
    axL.grid(True, which="both", alpha=0.25, linewidth=0.5)

    # ---- Panel (b): F_dns(k) with ceiling read-offs ----
    axR.semilogx(k_axis, F_dns, color="black", linewidth=1.9,
                 label=r"$F_{\mathrm{DNS}}(k)$ = cumulative DNS energy fraction")

    # Sort K by k_nyq so annotations stack cleanly
    K_sorted = sorted(curves.keys())
    for K in K_sorted:
        color = K_COLORS[K]
        k_nyq = float(np.sqrt(K / np.pi))
        ceil_val = float(curves[K]["ceiling"])  # strict |k|<=k_nyq disk fraction (t=5)
        # Vertical drop line down from ceiling
        axR.plot([k_nyq, k_nyq], [0, ceil_val], color=color, linestyle="--", linewidth=1.2, alpha=0.85)
        # Horizontal dotted line from y-axis to intersection
        axR.plot([1, k_nyq], [ceil_val, ceil_val], color=color, linestyle=":", linewidth=1.0, alpha=0.65)
        # Tick on the y-axis showing the ceiling value
        axR.scatter([k_nyq], [ceil_val], color=color, s=50, zorder=5,
                    edgecolor="white", linewidth=1.0)

    # Stacked annotation table in the lower-right corner — same color coding
    table_x = k_axis.max() * 0.6   # x location for table
    table_y0 = 0.30                 # top of table
    dy = 0.07
    axR.text(table_x, table_y0 + dy * 1.5, "Nyquist ceilings", fontsize=7,
             ha="right", va="bottom", weight="bold", color="0.15")
    for i, K in enumerate(K_sorted):
        color = K_COLORS[K]
        k_nyq = float(np.sqrt(K / np.pi))
        ceil_val = float(curves[K]["ceiling"])
        axR.text(
            table_x,
            table_y0 - i * dy,
            f"$K{{=}}{K}$:  $k_{{\\mathrm{{Nyq}}}}={k_nyq:.2f}$,  ceiling $= {ceil_val*100:.1f}\\%$",
            fontsize=7, color=color, ha="right", va="bottom",
        )

    axR.set_xlim(1, k_axis.max())
    axR.set_ylim(0.0, 1.05)
    axR.set_xlabel(r"Wavenumber  $k$  (1/m)")
    axR.set_ylabel(r"Cumulative energy fraction  $F_{\mathrm{DNS}}(k)$")
    axR.set_title("(b)", loc="left", fontweight="bold")
    # F_DNS 黑線由 ylabel + caption 說明，markers 由下方 Nyquist-ceilings 表說明；
    # 不再加 legend，避免 (b) 文字過多遮蔽。
    axR.grid(True, which="both", alpha=0.25, linewidth=0.5)
    # suptitle 移除 — 說明移至 LaTeX caption

    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


# ----------------------------- Main ---------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dns", default=DEFAULT_DNS)
    ap.add_argument("--out-dir", default="docs/figures")
    ap.add_argument("--K-list", default="100,200,400")
    ap.add_argument("--n-snapshots", type=int, default=N_SNAPSHOTS)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    K_list = [int(x) for x in args.K_list.split(",")]

    print(f"[1/4] Loading DNS  {args.dns}")
    dns = load_dns(args.dns)
    u_full = dns["u"]  # (T, N, N)
    v_full = dns["v"]
    T_total, N, _ = u_full.shape
    print(f"      shape (T, N, N) = ({T_total}, {N}, {N})")

    # Use the t=5 final frame, matching the t=5 spectra and energy fractions reported in the text.
    snap_indices = np.array([T_total - 1])
    print(f"      using t=5 frame (index {T_total - 1})")

    # k axis (bin centers)
    shell_idx = make_shell_index(N)
    k_max = int(N // 3)  # standard dealias cutoff
    k_axis = np.arange(k_max + 1)

    # NB: this plot is purely information-theoretic — F_dns(k) and Nyquist disk size depend
    # only on DNS and on K (via k_nyq = √(K/π)), not on sensor placement. We don't load
    # sensor files. (An earlier draft computed band-limited LSQ recovery using specific
    # sensor placements but rcond-sensitivity made the comparison fragile.)
    print(f"[2/4] (sensor positions not required for this plot — K-list = {K_list})")

    print("[3/4] Computing DNS energy distribution at t=5")
    n_bins = k_max + 1
    # Continuous radial wavenumber magnitude for the strict |k|<=k_nyq disk integral.
    kfreq = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(kfreq, kfreq, indexing="ij")
    kmag = np.sqrt(KX ** 2 + KY ** 2)
    t5 = int(snap_indices[-1])
    u_hat = np.fft.fft2(u_full[t5])
    v_hat = np.fft.fft2(v_full[t5])
    e2d = np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2
    e_tot = float(e2d.sum())
    E_per_shell = np.bincount(shell_idx.ravel(), weights=e2d.ravel(), minlength=n_bins)[:n_bins]
    F_dns_mean = np.cumsum(E_per_shell) / max(E_per_shell.sum(), 1e-30)

    # Ceiling = strict continuous-disk energy fraction |k|<=k_nyq at t=5 (matches the §4.5 text).
    curves: dict[int, dict] = {}
    for K in K_list:
        k_nyq = float(np.sqrt(K / np.pi))
        ceil_cont = float(e2d[kmag <= k_nyq].sum() / e_tot)
        curves[K] = {"F_dns_mean": F_dns_mean, "ceiling": ceil_cont}
        print(f"      K={K}:  k_nyq={k_nyq:.2f}  |  ceiling (|k|<=k_nyq, t=5) = {ceil_cont*100:.2f}%")

    print(f"[4/4] Plotting → {out_dir}")
    out_png = out_dir / "nyquist_recoverability.png"
    out_pdf = out_dir / "nyquist_recoverability.pdf"
    out_npz = out_dir / "nyquist_recoverability_data.npz"
    plot_recoverability(curves, k_axis, out_png, out_pdf)

    np.savez(
        out_npz,
        k_axis=k_axis,
        F_dns_mean=F_dns_mean,
        E_per_shell=E_per_shell,
        K_list=np.array(K_list),
        k_nyq_list=np.array([np.sqrt(K / np.pi) for K in K_list]),
        ceiling_list=np.array([curves[K]["ceiling"] for K in K_list]),
    )
    print(f"  png : {out_png}")
    print(f"  pdf : {out_pdf}")
    print(f"  npz : {out_npz}")


if __name__ == "__main__":
    main()
