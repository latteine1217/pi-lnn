#!/usr/bin/env python3
"""LES vs DNS 統計量 + 流場可視化對比。

What:
    1. 載入 LES (Re=10^4, N=128, T=15) + DNS (Re=10^4, N=256, T=5)
    2. 時間統計：KE(t), Enstrophy(t), div_max(t), E(k) overlay
    3. 流場 snapshots：u, v, omega 在多個時間點對比
    4. Spectral comparison：late-time E(k) log-log overlay

Why:
    LES 是 EXP-102 sensor placement 來源。要判斷 LES 跟 DNS 在哪些方面相似/不同，
    才能診斷為何 LES-informed pipeline (KE 44%) 表現不如 random retrain (KE 37%)。

Output:
    docs/assets/les_vs_dns_stats.png       — 4-panel 統計時序對比
    docs/assets/les_vs_dns_fields_late.png — 3 fields × 2 sources late-time 場對比
    docs/assets/les_vs_dns_evolution.png   — vorticity time evolution 4 frames × 2 sources
    docs/assets/les_vs_dns_spectrum.png    — late-time E(k) log-log + slope fit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pi_con.plot_style import apply_journal_rcparams  # noqa: E402
from pi_con.spectral import radial_energy_spectrum  # noqa: E402

# 此處原本手抄了一份 JOURNAL_RCPARAMS，且漏掉 pdf.fonttype=42（會輸出部分期刊
# 拒收的 Type-3 字型）。改為呼叫單一來源。
apply_journal_rcparams()


def compute_div_max_series(u: np.ndarray, v: np.ndarray, L: float) -> np.ndarray:
    """Spectral divergence max per snapshot."""
    T, N, _ = u.shape
    k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    out = np.zeros(T)
    for i in range(T):
        d = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(u[i])
                                 + 1j * KY * np.fft.fft2(v[i])))
        out[i] = np.abs(d).max()
    return out


def compute_radial_spectrum(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single snapshot radial E(k) — 實作見 pi_con.spectral（全 repo 單一份）。"""
    return radial_energy_spectrum(u, v)


def loglog_slope(k: np.ndarray, E: np.ndarray, k_lo: float, k_hi: float) -> tuple[float, float]:
    mask = (k >= k_lo) & (k <= k_hi) & (E > 1e-30)
    if mask.sum() < 5:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(np.log10(k[mask]), np.log10(E[mask]), 1)
    return float(slope), float(intercept)


def plot_stats(les: dict, dns: dict, out_path: Path) -> None:
    """KE(t), Enstrophy(t), div_max(t), late-time E(k) overlay."""
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0), constrained_layout=True)
    ax_ke, ax_ens = axes[0]
    ax_div, ax_spec = axes[1]

    ax_ke.plot(les["t"], les["KE"], label=f"LES (Re={les['Re']:.0e}, N={les['N']})",
               color="tab:red", linestyle="-")
    ax_ke.plot(dns["t"], dns["KE"], label=f"DNS (Re={dns['Re']:.0e}, N={dns['N']})",
               color="tab:blue", linestyle="-")
    ax_ke.set_xlabel("time")
    ax_ke.set_ylabel("kinetic energy KE(t)")
    ax_ke.set_title("(a) Kinetic energy time series")
    ax_ke.legend(loc="best")

    ax_ens.plot(les["t"], les["ENS"], color="tab:red", label="LES")
    ax_ens.plot(dns["t"], dns["ENS"], color="tab:blue", label="DNS")
    ax_ens.set_xlabel("time")
    ax_ens.set_ylabel("enstrophy Z(t)")
    ax_ens.set_title("(b) Enstrophy time series")
    ax_ens.legend(loc="best")

    ax_div.semilogy(les["t"], les["DIV"], color="tab:red", label="LES")
    ax_div.semilogy(dns["t"], dns["DIV"], color="tab:blue", label="DNS")
    ax_div.set_xlabel("time")
    ax_div.set_ylabel(r"$\|\nabla\cdot u\|_\infty$")
    ax_div.set_title("(c) Incompressibility (log scale)")
    ax_div.legend(loc="best")

    k_les, E_les = les["spec"]
    k_dns, E_dns = dns["spec"]
    ax_spec.loglog(k_les[E_les > 0], E_les[E_les > 0], color="tab:red",
                   label=f"LES (t={les['t'][-1]:.1f})")
    ax_spec.loglog(k_dns[E_dns > 0], E_dns[E_dns > 0], color="tab:blue",
                   label=f"DNS (t={dns['t'][-1]:.1f})")
    s_les, _ = loglog_slope(k_les, E_les, 3, 30)
    s_dns, _ = loglog_slope(k_dns, E_dns, 3, 30)
    # reference -3 (2D enstrophy cascade)
    k_ref = np.array([3.0, 30.0])
    E_anchor = float(E_dns[np.argmin(np.abs(k_dns - 3))])
    ax_spec.loglog(k_ref, E_anchor * (k_ref / 3) ** (-3), "k--", linewidth=0.8,
                   label="k$^{-3}$ reference")
    ax_spec.set_xlabel("wavenumber k")
    ax_spec.set_ylabel("E(k)")
    ax_spec.set_title(f"(d) Energy spectrum (slope LES={s_les:.2f}, DNS={s_dns:.2f})")
    ax_spec.legend(loc="best")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  → {out_path}")


def plot_fields_late(les: dict, dns: dict, out_path: Path) -> None:
    """Late-time u, v, omega 場對比（LES 全 N=128, DNS downsampled to N=128 for fair comparison）."""
    # 取 LES last frame, DNS last frame
    u_les = les["u"][-1]; v_les = les["v"][-1]; om_les = les["omega"][-1]
    # DNS downsample to N=128 for fair comparison
    stride = dns["N"] // les["N"]
    u_dns = dns["u"][-1, ::stride, ::stride]
    v_dns = dns["v"][-1, ::stride, ::stride]
    om_dns = dns["omega"][-1, ::stride, ::stride]

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.5), constrained_layout=True)
    rows = [("LES", [u_les, v_les, om_les]), (f"DNS (↓N={les['N']})", [u_dns, v_dns, om_dns])]
    titles = ["u", "v", r"$\omega$"]
    for row_i, (label, fields) in enumerate(rows):
        for col_i, (field, title) in enumerate(zip(fields, titles)):
            ax = axes[row_i, col_i]
            vmax = float(np.abs(field).max())
            im = ax.imshow(field, origin="lower", extent=[0, 1, 0, 1],
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"{label}: {title}  (range ±{vmax:.2f})")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            cbar = plt.colorbar(im, ax=ax, shrink=0.85)
            cbar.ax.tick_params(labelsize=7)

    t_les = les["t"][-1]; t_dns = dns["t"][-1]
    fig.suptitle(f"Late-time field comparison — LES t={t_les:.1f} | DNS t={t_dns:.1f}",
                 fontsize=11)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  → {out_path}")


def plot_evolution(les: dict, dns: dict, out_path: Path) -> None:
    """Vorticity 時間演化：每個 source 取 4 個時間點。"""
    # 選 LES 時間：0, 5, 10, 15
    les_t_targets = [0.0, 5.0, 10.0, 15.0]
    les_idx = [int(np.argmin(np.abs(les["t"] - t))) for t in les_t_targets]
    # 選 DNS 時間：0, 1.67, 3.33, 5
    dns_t_targets = [0.0, 1.67, 3.33, 5.0]
    dns_idx = [int(np.argmin(np.abs(dns["t"] - t))) for t in dns_t_targets]

    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.5), constrained_layout=True)

    # 估全局 vmax (對應 source) 用於統一 colormap range
    om_max_les = float(np.abs(les["omega"]).max())
    om_max_dns = float(np.abs(dns["omega"]).max())
    om_max = max(om_max_les, om_max_dns)
    # 但 LES vorticity range 比 DNS 小很多 → 用各自獨立 vmax 更可讀
    for col, (idx, t_target) in enumerate(zip(les_idx, les_t_targets)):
        ax = axes[0, col]
        field = les["omega"][idx]
        vmax = float(np.abs(field).max())
        im = ax.imshow(field, origin="lower", extent=[0, 1, 0, 1],
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"LES  t={les['t'][idx]:.2f}  (±{vmax:.2f})")
        if col == 0:
            ax.set_ylabel("y")
        ax.set_xlabel("x")
        cbar = plt.colorbar(im, ax=ax, shrink=0.7)
        cbar.ax.tick_params(labelsize=7)

    for col, (idx, t_target) in enumerate(zip(dns_idx, dns_t_targets)):
        ax = axes[1, col]
        field = dns["omega"][idx]
        vmax = float(np.abs(field).max())
        im = ax.imshow(field, origin="lower", extent=[0, 1, 0, 1],
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"DNS  t={dns['t'][idx]:.2f}  (±{vmax:.2f})")
        if col == 0:
            ax.set_ylabel("y")
        ax.set_xlabel("x")
        cbar = plt.colorbar(im, ax=ax, shrink=0.7)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(r"Vorticity $\omega$ time evolution — top: LES (random IC), bottom: DNS (calibrated IC)",
                 fontsize=11)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  → {out_path}")


def plot_spectrum_overlay(les: dict, dns: dict, out_path: Path) -> None:
    """Late-time E(k) overlay + slope fit annotations."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    k_les, E_les = les["spec"]
    k_dns, E_dns = dns["spec"]
    ax.loglog(k_les[E_les > 0], E_les[E_les > 0], color="tab:red",
              label=f"LES (Re={les['Re']:.0e}, N={les['N']}, t={les['t'][-1]:.1f})", lw=1.6)
    ax.loglog(k_dns[E_dns > 0], E_dns[E_dns > 0], color="tab:blue",
              label=f"DNS (Re={dns['Re']:.0e}, N={dns['N']}, t={dns['t'][-1]:.1f})", lw=1.6)

    s_les, b_les = loglog_slope(k_les, E_les, 3, 30)
    s_dns, b_dns = loglog_slope(k_dns, E_dns, 3, 30)
    # fitted lines
    k_fit = np.array([3.0, 30.0])
    ax.loglog(k_fit, 10 ** (s_les * np.log10(k_fit) + b_les), "r:", lw=0.9, alpha=0.7,
              label=f"LES fit slope = {s_les:.2f}")
    ax.loglog(k_fit, 10 ** (s_dns * np.log10(k_fit) + b_dns), "b:", lw=0.9, alpha=0.7,
              label=f"DNS fit slope = {s_dns:.2f}")
    # k^-3 reference
    k_ref = np.array([3.0, 60.0])
    E_anchor = float(E_dns[np.argmin(np.abs(k_dns - 3))])
    ax.loglog(k_ref, E_anchor * (k_ref / 3) ** (-3), "k--", lw=0.8,
              label=r"$k^{-3}$ (2D enstrophy cascade)")

    ax.set_xlabel("wavenumber k")
    ax.set_ylabel("E(k)")
    ax.set_title("Late-time energy spectrum (radial)")
    ax.legend(loc="lower left")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  → {out_path}")


def load_source(path: Path, name: str, Re: float) -> dict:
    print(f"[load] {name}: {path}")
    d = np.load(path, allow_pickle=True).item()
    u = np.asarray(d["u"], dtype=np.float64)
    v = np.asarray(d["v"], dtype=np.float64)
    om = np.asarray(d["omega"], dtype=np.float64) if "omega" in d else None
    t = np.asarray(d["time"], dtype=np.float64)
    cfg = d.get("config", {})
    L = float(cfg.get("L", 1.0))
    N = u.shape[-1]
    KE = 0.5 * np.mean(u ** 2 + v ** 2, axis=(1, 2))
    if om is None:
        # compute omega
        k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
        KX, KY = np.meshgrid(k, k, indexing="ij")
        om = np.zeros_like(u)
        for i in range(u.shape[0]):
            om[i] = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(v[i])
                                          - 1j * KY * np.fft.fft2(u[i])))
    ENS = 0.5 * np.mean(om ** 2, axis=(1, 2))
    DIV = compute_div_max_series(u, v, L)
    k_bins, spec = compute_radial_spectrum(u[-1], v[-1])
    print(f"  N={N}, L={L}, T=[{t[0]:.2f}, {t[-1]:.2f}], n_frames={len(t)}")
    print(f"  KE late={KE[-1]:.3e}  ENS late={ENS[-1]:.3e}  DIV last={DIV[-1]:.2e}")
    return {
        "u": u, "v": v, "omega": om, "t": t,
        "N": N, "L": L, "Re": Re,
        "KE": KE, "ENS": ENS, "DIV": DIV,
        "spec": (k_bins, spec),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--les", default="data/les/kolmogorov_les_Re10000_N128_T15_bardina_standalone.npy")
    parser.add_argument("--dns", default="data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy")
    parser.add_argument("--out", default="docs/assets")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    les = load_source(Path(args.les), "LES", Re=1e4)
    dns = load_source(Path(args.dns), "DNS", Re=1e4)

    print("\n[plot] Statistics overview...")
    plot_stats(les, dns, out_dir / "les_vs_dns_stats.png")
    print("\n[plot] Late-time field comparison...")
    plot_fields_late(les, dns, out_dir / "les_vs_dns_fields_late.png")
    print("\n[plot] Vorticity time evolution...")
    plot_evolution(les, dns, out_dir / "les_vs_dns_evolution.png")
    print("\n[plot] Spectrum overlay...")
    plot_spectrum_overlay(les, dns, out_dir / "les_vs_dns_spectrum.png")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
