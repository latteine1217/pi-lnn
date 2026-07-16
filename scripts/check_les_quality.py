"""check_les_quality.py — LES 品質三條門檻，逐條量測。

What: 對一支 LES .npy 量 CLAUDE.md 的 LES_Quality_Requirements，輸出 PASS/FAIL。
Why : 「看起來收斂了」不是判準。此檔把三條門檻變成可執行、可重跑、可反駁的檢查，
      並刻意避開兩個已證偽的舊 gate（見下）。
用法: uv run python scripts/check_les_quality.py <les.npy>     （exit 1 表示未過）

三條門檻（CLAUDE.md / thesis/CLAUDE.md）:
  1. 不可壓縮性 ‖∇·u‖ < 1e-10 —— 必須取 solver 內部的 fp64 診斷值。
     禁止從儲存的 float32 場重算：那會得到 ~8e-6 的儲存捨入誤差，不是 solver 精度
     （可驗算：u_rms 0.28 × float32 eps 1.2e-7 × k_max 264 ≈ 8e-6）。
  2. 無 aliasing pile-up —— 譜末端衰減比 > 1e6。
  3. 統計窗充分性 T_end/τ_int ≥ 10，並報 N_eff = T_end/(2·τ_int)。
     τ_int = KE 的整合自相關時間，丟棄 20% burn-in 後積分至首次過零。

刻意不做的兩件事:
  ✗ rel_change(KE[-1] vs KE[-10]) < 5% —— 回看窗由 save_interval 決定，結構上不可能
    失敗，曾讓兩支 transient LES 掛著 PASS 過關。
  ✗ 以 LES 能譜與 DNS 比對 —— LES 帶 linear friction、DNS 無，能量平衡本就不同；
    物理上不可能通過，吻合也不構成品質證據。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def tau_int_of(ke: np.ndarray, dt_sample: float, burn_in: float = 0.2) -> float:
    """KE 的整合自相關時間：丟 burn-in、去均值、積分自相關至首次過零。"""
    k = ke[int(burn_in * len(ke)):].astype(np.float64)
    k = k - k.mean()
    if not np.any(k):
        return float("nan")
    ac = np.correlate(k, k, "full")[len(k) - 1:]
    ac /= ac[0]
    zc = int(np.argmax(ac < 0)) if (ac < 0).any() else len(ac)
    return float(np.trapezoid(ac[:zc]) * dt_sample)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    d = np.load(path, allow_pickle=True).item()
    cfg = d.get("config", {})
    diag = d.get("diagnostics", {})
    T_end = float(cfg.get("T_end", 0.0))

    print(f"\n  LES: {path.name}")
    print(f"  N={cfg.get('N')}  nu={cfg.get('nu')}  T_end={T_end}  "
          f"r_scale={cfg.get('r_scale')}  alpha={cfg.get('nu_h_alpha')}  "
          f"closure={cfg.get('closure_model')}\n")

    verdicts = []

    # ── 1. 不可壓縮性（solver fp64 診斷）────────────────────────────────
    div = np.asarray(diag.get("divergence_error", []))
    if div.size:
        div_max = float(np.max(div))
        ok = div_max < 1e-10
        verdicts.append(ok)
        print(f"  [1] incompressibility   max‖∇·u‖ = {div_max:.2e}   "
              f"(< 1e-10)   {'PASS' if ok else 'FAIL'}")
        print(f"      ↳ solver fp64 診斷值，非從 float32 場重算")
    else:
        verdicts.append(False)
        print("  [1] incompressibility   FAIL — diagnostics 內無 divergence_error")

    # ── 2. aliasing pile-up ────────────────────────────────────────────
    spec = np.asarray(diag.get("energy_spectrum", []))
    if spec.ndim == 2 and spec.shape[0] > 1:
        late = spec[int(0.5 * spec.shape[0]):].mean(axis=0)
        peak = float(late.max())
        tail = float(late[-1])
        ratio = peak / tail if tail > 0 else np.inf
        ok = ratio > 1e6
        verdicts.append(ok)
        print(f"  [2] no aliasing pile-up decay ratio = {ratio:.2e}   "
              f"(> 1e6)   {'PASS' if ok else 'FAIL'}")
    else:
        verdicts.append(False)
        print("  [2] no aliasing pile-up FAIL — diagnostics 內無 energy_spectrum")

    # ── 3. 統計窗充分性 ─────────────────────────────────────────────────
    ke = np.asarray(diag.get("kinetic_energy", []))
    t = np.asarray(d.get("time", []))
    if ke.size > 20:
        dts = float(t[1] - t[0]) if t.size > 1 else float(cfg["dt"]) * cfg["save_interval"]
        tau = tau_int_of(ke, dts)
        ratio = T_end / tau
        n_eff = T_end / (2 * tau)
        ok = ratio >= 10
        verdicts.append(ok)
        print(f"  [3] statistical window  τ_int = {tau:.2f}   "
              f"T_end/τ_int = {ratio:.2f}   (≥ 10)   {'PASS' if ok else 'FAIL'}")
        print(f"      ↳ N_eff = {n_eff:.2f}   Δt_sample = {dts:.4f}   samples = {len(ke)}")
        if not ok:
            print(f"      ↳ 需 T_end ≳ {10 * tau:.0f} 才過")
    else:
        verdicts.append(False)
        print("  [3] statistical window  FAIL — KE 序列過短")

    passed = all(verdicts)
    print(f"\n  ⇒ {'PASS — 可用於佈點' if passed else 'FAIL — 不可用於佈點'}\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
