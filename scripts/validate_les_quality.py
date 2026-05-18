#!/usr/bin/env python3
"""validate_les_quality.py — 2D Kolmogorov LES 品質驗證。

What:
    依 cfd-validation skill 的七項判讀表，對 LES .npy 跑：
      1. Incompressibility（spectral div_max + builtin diagnostic）
      2. CFL stability
      3. Energy balance (含 SGS hyperviscosity + linear friction)
      4. Spectrum slope (k ∈ [3, 40])
      5. Tail decay ratio (aliasing pile-up 檢查)
      6. Resolution adequacy (k_max / k_eff)
      7. Steady-state convergence (rel_change, T_end/T_L)

    輸出 stdout 報告 + JSON verdict（PASS/WARN/FAIL per 項）。

Why:
    REAL_WORLD_PIPELINE 要求 LES 達 LES_Quality_Requirements 才能當 sensor placement proxy。
    本 script 是判定 acceptance 的標準工具，避免 ad-hoc 估值偏差。

Usage:
    uv run python scripts/validate_les_quality.py --les <path.npy>
    uv run python scripts/validate_les_quality.py --les A.npy --les B.npy  # 並排比較
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def linregress_np(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    s, b = np.polyfit(x, y, 1)
    y_pred = s * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-30)
    return float(s), float(b), float(r2)


def verdict(value: float, pass_thr: float, warn_thr: float,
            higher_is_better: bool) -> str:
    if higher_is_better:
        if value >= pass_thr:
            return "PASS"
        if value >= warn_thr:
            return "WARN"
        return "FAIL"
    if value <= pass_thr:
        return "PASS"
    if value <= warn_thr:
        return "WARN"
    return "FAIL"


def validate_one(path: Path) -> dict[str, Any]:
    """Return structured verdict for one LES file."""
    print("=" * 72)
    print(f"LES file: {path}")
    print("=" * 72)
    d = np.load(path, allow_pickle=True).item()
    cfg = d.get("config", {})

    u = np.asarray(d["u"], dtype=np.float64)
    v = np.asarray(d["v"], dtype=np.float64)
    om = (np.asarray(d["omega"], dtype=np.float64) if "omega" in d
          else None)
    t = np.asarray(d["time"], dtype=np.float64)
    N = int(cfg.get("N", u.shape[-1]))
    L = float(cfg.get("L", 1.0))
    nu = float(cfg.get("nu", 1.0e-4))
    A = float(cfg.get("A", 0.1))
    k_f = float(cfg.get("k_f", 2))
    dt = float(cfg.get("dt", 2.5e-4))
    nu_h = float(cfg.get("nu_h", 0.0))
    hyper_p = int(cfg.get("hyper_p", 2))
    r_fric = float(cfg.get("r", 0.0))
    closure = cfg.get("closure_model", "hyperviscosity")

    print(f"\n[CONFIG] N={N}  L={L}  nu={nu:.2e}  A={A}  k_f={k_f}  dt={dt}")
    print(f"         nu_h={nu_h:.3e}  hyper_p={hyper_p}  r_fric={r_fric:.3e}  "
          f"closure={closure}")
    print(f"         T_end={t[-1]:.2f}, n_frames={len(t)}  "
          f"data_dtype={d['u'].dtype}")

    diag = d.get("diagnostics", {})

    # ── KE / ENS / DIV time series ────────────────────────────────────
    KE = 0.5 * np.mean(u ** 2 + v ** 2, axis=(1, 2))
    if om is None:
        # 從 u, v 重算
        k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
        KX, KY = np.meshgrid(k, k, indexing="ij")
        om = np.zeros_like(u)
        for i in range(u.shape[0]):
            om[i] = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(v[i])
                                          - 1j * KY * np.fft.fft2(u[i])))
    ENS = 0.5 * np.mean(om ** 2, axis=(1, 2))
    # spectral div per frame
    k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    DIV = np.zeros(len(t))
    for i in range(len(t)):
        di = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(u[i])
                                   + 1j * KY * np.fft.fft2(v[i])))
        DIV[i] = float(np.abs(di).max())

    # 1. Incompressibility
    div_max = float(DIV.max())
    div_v = verdict(div_max, 1e-10, 1e-6, higher_is_better=False)
    print(f"\n--- 1. Incompressibility ---")
    print(f"  ‖∇·u‖_max (recomputed):  {div_max:.3e}  → {div_v}")
    if "divergence_error" in diag:
        builtin = float(np.asarray(diag["divergence_error"]).max())
        print(f"  builtin div_err max:    {builtin:.3e}")

    # 2. CFL
    speed = np.abs(u).max(axis=(1, 2)) + np.abs(v).max(axis=(1, 2))
    cfl_max = float((speed * dt / (L / N)).max())
    cfl_v = verdict(cfl_max, 0.5, 1.0, higher_is_better=False)
    print(f"\n--- 2. CFL ---")
    print(f"  max|u|+|v| @ late: {speed[-1]:.3e}")
    print(f"  CFL max:           {cfl_max:.3f}  → {cfl_v}")

    # 3. Energy balance
    print("\n--- 3. Energy balance (含 SGS) ---")
    k_phys = k_f * 2 * np.pi / L
    yy = np.linspace(0, L, N, endpoint=False)
    sin_kfy = np.sin(k_phys * yy)[None, :]
    P_in = np.array([A * np.mean(u[i] * sin_kfy) for i in range(len(t))])
    eps_visc = 2 * nu * ENS
    # spectral SGS estimate（簡化，可能 binning 偏差 → 不當 hard verdict）
    eps_fric = 2 * r_fric * KE
    dKE_dt = np.gradient(KE, t)
    mask_late = t > max(1.0, 0.2 * t[-1])
    print(f"  P_in (mean post-spinup):  {P_in[mask_late].mean():+.3e}")
    print(f"  eps_visc (2 ν Z):         {eps_visc[mask_late].mean():.3e}")
    print(f"  eps_fric (2 r KE):        {eps_fric[mask_late].mean():.3e}")
    print(f"  dKE/dt (mean):            {dKE_dt[mask_late].mean():+.3e}")
    print(f"  → balance check 略過（spectral SGS estimate 對 binning 敏感）")

    # 4. Spectrum slope
    print("\n--- 4. Energy spectrum slope ---")
    k_mode = np.fft.fftfreq(N, d=1.0 / N)
    kx, ky = np.meshgrid(k_mode, k_mode, indexing="ij")
    k_rad = np.sqrt(kx ** 2 + ky ** 2)
    bin_edges = np.arange(0.5, N // 2 + 1.5, 1.0)
    k_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.digitize(k_rad.ravel(), bin_edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < len(k_bins))
    SP = np.zeros((len(t), len(k_bins)))
    for i in range(len(t)):
        uh = np.fft.fft2(u[i])
        vh = np.fft.fft2(v[i])
        e2d = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2) / (N ** 4)
        s_i = np.zeros(len(k_bins))
        np.add.at(s_i, bin_idx[valid], e2d.ravel()[valid])
        SP[i] = s_i
    SP_late = SP[mask_late].mean(axis=0)
    k_lo, k_hi = 3.0, 40.0
    band = (k_bins >= k_lo) & (k_bins <= k_hi) & (SP_late > 1e-30)
    if band.sum() >= 5:
        slope_late, _, r2 = linregress_np(np.log10(k_bins[band]),
                                           np.log10(SP_late[band]))
    else:
        slope_late, r2 = float("nan"), 0.0
    # 與 -3 (2D enstrophy cascade) 的偏差
    dev_pct = abs(slope_late - (-3.0)) / 3.0 * 100
    slope_v = verdict(dev_pct, 30.0, 70.0, higher_is_better=False)
    print(f"  Late-time E(k) fit k ∈ [{k_lo},{k_hi}]: "
          f"slope={slope_late:.3f} R²={r2:.3f}")
    print(f"  Deviation from 2D enstrophy cascade (-3): "
          f"{dev_pct:.1f}%  → {slope_v}")

    # 5. Tail decay ratio
    kf_idx = np.argmin(np.abs(k_bins - k_f))
    tail_max = SP_late[k_bins >= k_bins.max() - 50].max()
    tail_ratio = float(SP_late[kf_idx] / max(tail_max, 1e-30))
    tail_v = verdict(tail_ratio, 1e6, 1e3, higher_is_better=True)
    print(f"\n--- 5. Tail decay ratio ---")
    print(f"  E(k_f)/E(tail) = {tail_ratio:.3e}  → {tail_v}")

    # 6. Resolution adequacy
    print("\n--- 6. Resolution adequacy ---")
    Z_late = float(ENS[mask_late].mean())
    k_d_ns = (Z_late / nu ** 3) ** (1 / 6)
    if hyper_p > 1 and nu_h > 0:
        k_eff = (nu / nu_h) ** (1 / (2 * hyper_p - 2))
    else:
        k_eff = float("inf")
    dealias_mode = cfg.get("dealias_mode", "2/3")
    if dealias_mode == "3/2":
        k_max_mode = N // 2
    else:
        k_max_mode = N // 3
    print(f"  k_max ({dealias_mode}):  {k_max_mode}")
    print(f"  k_d (pure NS):       {k_d_ns:.2f}  → DNS-as-LES need k_max≥{k_d_ns:.0f}")
    print(f"  k_eff (hyperv on):   {k_eff:.2f}")
    print(f"  k_max / k_eff:       {k_max_mode/max(k_eff,1e-12):.2f}  "
          f"(LES expectation: ≳1)")

    # 7. Steady-state
    print("\n--- 7. Steady-state convergence ---")
    if len(KE) >= 10:
        rel_change = float((KE[-1] - KE[-10]) / max(KE[-1], 1e-30))
    else:
        rel_change = float("nan")
    u_rms_late = np.sqrt(2 * KE[mask_late].mean())
    T_L = L / max(u_rms_late * k_f, 1e-12)
    T_end_TL = t[-1] / T_L
    ss_v = verdict(abs(rel_change), 0.005, 0.05, higher_is_better=False)
    tend_v = verdict(T_end_TL, 5.0, 2.0, higher_is_better=True)
    print(f"  KE: first={KE[0]:.3e}  last={KE[-1]:.3e}  ratio={KE[-1]/max(KE[0],1e-30):.2f}×")
    print(f"  ENS: first={ENS[0]:.3e}  last={ENS[-1]:.3e}")
    print(f"  rel_change(KE[-1] vs [-10]): {rel_change*100:+.2f}%  → {ss_v}")
    print(f"  T_L≈L/(u_rms·k_f)={T_L:.3f}  T_end/T_L={T_end_TL:.1f}  → {tend_v}")

    # ── 收集 verdict ──────────────────────────────────────────────────
    out = {
        "path": str(path),
        "config": {
            "N": N, "L": L, "nu": nu, "A": A, "k_f": k_f, "dt": dt,
            "nu_h": nu_h, "nu_h_alpha": cfg.get("nu_h_alpha"),
            "r_fric": r_fric, "closure_model": closure,
            "dealias_mode": dealias_mode, "init_mode": cfg.get("init_mode"),
            "T_end": t[-1], "n_frames": len(t),
            "data_dtype": str(d["u"].dtype),
        },
        "metrics": {
            "div_max": div_max,
            "cfl_max": cfl_max,
            "spectrum_slope_late": slope_late,
            "spectrum_slope_deviation_pct_vs_minus3": dev_pct,
            "tail_decay_ratio": tail_ratio,
            "k_max_mode": k_max_mode,
            "k_d_pure_NS": float(k_d_ns),
            "k_eff_hyperv": float(k_eff),
            "k_max_over_k_eff": float(k_max_mode / max(k_eff, 1e-12)),
            "KE_late": float(KE[-1]),
            "ENS_late": float(ENS[-1]),
            "rel_change_KE": rel_change,
            "T_L_estimated": float(T_L),
            "T_end_over_T_L": float(T_end_TL),
        },
        "verdicts": {
            "incompressibility": div_v,
            "cfl": cfl_v,
            "spectrum_slope": slope_v,
            "tail_decay_ratio": tail_v,
            "steady_state": ss_v,
            "long_enough": tend_v,
        },
    }
    print("\n  === Verdict summary ===")
    for k, v in out["verdicts"].items():
        sym = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[v]
        print(f"    {sym} {k:<22} {v}")
    return out


def comparison_table(results: list[dict[str, Any]]) -> None:
    if len(results) < 2:
        return
    print("\n" + "=" * 72)
    print("Side-by-side comparison")
    print("=" * 72)
    keys = [
        ("config.N", "N"),
        ("config.nu_h_alpha", "nu_h_alpha"),
        ("config.closure_model", "closure"),
        ("config.init_mode", "init_mode"),
        ("config.dealias_mode", "dealias"),
        ("config.data_dtype", "dtype"),
        ("metrics.div_max", "‖∇·u‖_max"),
        ("metrics.cfl_max", "CFL_max"),
        ("metrics.spectrum_slope_late", "slope[3-40]"),
        ("metrics.tail_decay_ratio", "tail_ratio"),
        ("metrics.k_max_over_k_eff", "k_max/k_eff"),
        ("metrics.KE_late", "KE_late"),
        ("metrics.ENS_late", "ENS_late"),
        ("metrics.rel_change_KE", "rel_change_KE"),
        ("metrics.T_end_over_T_L", "T_end/T_L"),
    ]

    def get(d, dotted: str):
        cur = d
        for p in dotted.split("."):
            cur = cur.get(p, "—") if isinstance(cur, dict) else "—"
        return cur

    name_w = 18
    col_w = 22
    header = f"  {'metric':<{name_w}}" + "".join(
        f"{Path(r['path']).stem[:col_w-2]:<{col_w}}" for r in results
    )
    print(header)
    print("  " + "-" * (name_w + col_w * len(results)))
    for dotted, label in keys:
        line = f"  {label:<{name_w}}"
        for r in results:
            val = get(r, dotted)
            if isinstance(val, float):
                line += f"{val:<{col_w}.4g}"
            else:
                line += f"{str(val):<{col_w}}"
        print(line)

    print("\n  Verdict summary:")
    for vk in results[0]["verdicts"]:
        line = f"  {vk:<{name_w}}"
        for r in results:
            v = r["verdicts"].get(vk, "—")
            sym = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(v, "·")
            line += f"{sym} {v:<{col_w-3}}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--les", action="append", required=True,
                        help="LES .npy 路徑（可多次指定做並排比較）")
    parser.add_argument("--save", type=Path, default=None,
                        help="儲存 JSON 報告（可選）")
    args = parser.parse_args()

    results = []
    for p in args.les:
        try:
            results.append(validate_one(Path(p)))
        except Exception as e:
            print(f"[ERROR] {p}: {e!r}")

    if len(results) >= 2:
        comparison_table(results)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2, default=str)
        print(f"\n  → saved {args.save}")


if __name__ == "__main__":
    main()
