"""check_les_quality.py — LES 品質三條門檻，逐條量測。

What: 對一支 LES .npy 量 CLAUDE.md 的 LES_Quality_Requirements，輸出 PASS/FAIL。
Why : 「看起來收斂了」不是判準。此檔把三條門檻變成可執行、可重跑、可反駁的檢查，
      並刻意避開三個已證偽的舊 gate（見下）。
用法: uv run python scripts/check_les_quality.py <les.npy> [--tau-int SECONDS]
      （exit 1 表示未過 / 無法認證）

三條門檻（CLAUDE.md / thesis/CLAUDE.md）:
  1. 不可壓縮性 ‖∇·u‖ < 1e-10 —— 必須取 solver 內部的 fp64 診斷值。
     禁止從儲存的 float32 場重算：那會得到 ~8e-6 的儲存捨入誤差，不是 solver 精度
     （可驗算：u_rms 0.28 × float32 eps 1.2e-7 × k_max 264 ≈ 8e-6）。
  2. 無 aliasing pile-up —— 譜末端衰減比 > 1e6。
  3. 統計窗充分性 T_end/τ_int ≥ 10，並報 N_eff = T_end/(2·τ_int)。

刻意不做的三件事:
  ✗ rel_change(KE[-1] vs KE[-10]) < 5% —— 回看窗由 save_interval 決定，結構上不可能
    失敗，曾讓兩支 transient LES 掛著 PASS 過關。
  ✗ 以 LES 能譜與 DNS 比對 —— LES 帶 linear friction、DNS 無，能量平衡本就不同；
    物理上不可能通過，吻合也不構成品質證據。
  ✗ 用「積分自相關至首次過零」自我認證 τ_int —— 見下，這是本檔曾犯的錯。

────────────────────────────────────────────────────────────────────────────
為什麼 [3] 不能只看量到的 τ_int（2026-07-17 修正）

舊版把 τ_int **從受檢的那支紀錄自己**估出來，積分自相關至首次過零，然後拿
T_end/τ_int 去比 10。對 N=256 T_end=50 的參考 LES 得 τ_int = 4.28 → 11.68 → PASS。

那個 PASS 是假的。真值由論文另跑一支 T_end=400 s 的診斷量得 τ_int = 10.1 s
（chapter03:193），故 50 s 只有 T_end/τ_int = 4.9，未達。

原因是估計量的偏誤有方向：紀錄長度 T 只有約 5 個相關時間時，樣本自相關在大 lag
處被雜訊主導而提早穿越零，積分因此被截斷 → **系統性低估 τ_int**。
太短的紀錄無法自己揭露它太短，是自我實現的假通過。

修法用同一個偏誤方向來反推可信度：低估只會把 T_end/τ_int 推高，所以
  · 量到未達門檻        → 可信（真值更大 → 只會更未達）        → FAIL
  · 量到過關但餘裕不足   → 不可信（真值可能更大）              → UNCERTIFIABLE
  · 量到過關且餘裕充足   → 窗口 M ≪ T，估計自洽                → PASS
餘裕門檻取 T_end ≥ 50 τ_int（Sokal 自洽窗的常用經驗值）。

τ_int 估計改用 Sokal 自動窗（Sokal 1997; Goodman & Weare 2010 沿用）：
取最小的 M 使 M·Δt ≥ c·τ_int(M)，c = 5。它不再依賴「首次過零」這個對雜訊敏感的
截斷點，並且 M 本身就是估計是否自洽的度量。

要沿用論文做法（長跑量 τ_int、套到短跑），用 --tau-int 外部給值，本檔會直接採用
並跳過自我認證。
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Sokal 自動窗常數：取最小 M 使 M·Δt ≥ C_WINDOW · τ_int(M)。
C_WINDOW = 5.0
# 統計窗門檻（CLAUDE.md）。
RATIO_MIN = 10.0
# 自我認證 τ_int 所需的餘裕：紀錄需長於此倍數的 τ_int，估計才自洽。
RATIO_SELF_CERTIFY = 50.0


def _autocorr(x: np.ndarray) -> np.ndarray:
    """歸一化樣本自相關（lag 0 起），用 FFT 算。"""
    x = x - x.mean()
    n = 1 << (2 * len(x) - 1).bit_length()
    f = np.fft.rfft(x, n)
    ac = np.fft.irfft(f * np.conjugate(f), n)[: len(x)].real
    return ac / ac[0]


def tau_int_sokal(ke: np.ndarray, dt_sample: float, burn_in: float = 0.2
                  ) -> tuple[float, int]:
    """KE 的整合自相關時間，Sokal 自動窗。

    回傳 (τ_int [秒], M [窗口 lag 數])。採 turbulence 慣例
    τ_int = ∫₀^∞ ρ(τ) dτ，與論文 tab:les_verification 及 N_eff = T/(2τ) 一致。
    """
    k = ke[int(burn_in * len(ke)):].astype(np.float64)
    if k.size < 8 or np.allclose(k, k[0]):
        return float("nan"), 0
    ac = _autocorr(k)
    # 累積積分（梯形）：tau[m] = ∫₀^{m·Δt} ρ dτ
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (ac[1:] + ac[:-1])))) * dt_sample
    for m in range(1, len(cum)):
        if m * dt_sample >= C_WINDOW * cum[m]:
            return float(cum[m]), m
    # 窗口未閉合 → 紀錄遠短於 τ_int；回傳全長積分（仍是低估）並標示。
    return float(cum[-1]), len(cum) - 1


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="LES .npy")
    ap.add_argument("--tau-int", type=float, default=None, metavar="SECONDS",
                    help="外部量得的 τ_int（如論文的 400 s 長跑診斷）。"
                         "給了就直接採用，跳過自我認證。")
    a = ap.parse_args()

    d = np.load(a.path, allow_pickle=True).item()
    cfg = d.get("config", {})
    diag = d.get("diagnostics", {})
    T_end = float(cfg.get("T_end", 0.0))

    print(f"\n  LES: {a.path.name}")
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
    #     低估偏誤是單向的（見檔頭）：量到的 τ_int 只會偏小 → ratio 只會偏大。
    #     故「未達」可信，「剛好過」不可信，只有餘裕充足才自我認證。
    ke = np.asarray(diag.get("kinetic_energy", []))
    t = np.asarray(d.get("time", []))
    if ke.size <= 20:
        verdicts.append(False)
        print("  [3] statistical window  FAIL — KE 序列過短")
    else:
        dts = float(t[1] - t[0]) if t.size > 1 else float(cfg["dt"]) * cfg["save_interval"]

        if a.tau_int is not None:
            tau, m, src = a.tau_int, 0, "外部給定（--tau-int）"
        else:
            tau, m = tau_int_sokal(ke, dts)
            src = f"self-measured（Sokal 窗 M={m} lags = {m * dts:.1f} s，c={C_WINDOW:g}）"

        ratio = T_end / tau
        n_eff = T_end / (2 * tau)

        if a.tau_int is not None:
            ok = ratio >= RATIO_MIN
            status = "PASS" if ok else "FAIL"
        elif ratio < RATIO_MIN:
            ok, status = False, "FAIL"          # 低估仍未達 → 真值只會更差
        elif ratio >= RATIO_SELF_CERTIFY:
            ok, status = True, "PASS"           # 餘裕足夠，窗口自洽
        else:
            ok, status = False, "UNCERTIFIABLE" # 過線但餘裕不足 → 不可自我認證

        verdicts.append(ok)
        print(f"  [3] statistical window  τ_int = {tau:.2f} s   "
              f"T_end/τ_int = {ratio:.2f}   (≥ {RATIO_MIN:g})   {status}")
        print(f"      ↳ N_eff = {n_eff:.2f}   Δt_sample = {dts:.4f}   "
              f"samples = {len(ke)}   τ_int 來源：{src}")

        if status == "UNCERTIFIABLE":
            print(f"      ↳ 過了 {RATIO_MIN:g} 但未達自我認證所需的 "
                  f"{RATIO_SELF_CERTIFY:g}（需 T_end ≳ {RATIO_SELF_CERTIFY * tau:.0f} s）。")
            print(f"      ↳ 短紀錄會系統性低估 τ_int，故此 ratio 是上界而非量測值，"
                  f"不足以宣稱收斂。")
            print(f"      ↳ 解法：跑一支夠長的診斷量 τ_int，再用 "
                  f"--tau-int 餵回（論文即用 T_end=400 s 量得 10.1 s）。")
        elif status == "FAIL" and a.tau_int is None:
            print(f"      ↳ 需 T_end ≳ {RATIO_MIN * tau:.0f} s 才過；"
                  f"且此 τ_int 為低估，真實需求只會更長。")

    passed = all(verdicts)
    print(f"\n  ⇒ {'PASS — 可用於佈點' if passed else 'FAIL — 不可宣稱統計收斂'}\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
