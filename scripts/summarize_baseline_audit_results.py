"""Summarize baseline audit results (dedup Trig LSQ + RBF epsilon sweep).

Reads scripts/baseline_squeeze.py output and emits a readable comparison for
thesis update. Run AFTER baseline_squeeze.py completes.

  uv run python scripts/summarize_baseline_audit_results.py
"""
from __future__ import annotations
import json
from pathlib import Path

JSON_PATH = Path("artifacts/under_determined_proof/baseline_squeeze.json")
PICON_KE = 5.71
PICON_U_L2 = 13.65
PICON_V_L2 = 17.52
PICON_W_L2 = 41.79


def fmt(pct: float | None) -> str:
    return f"{pct*100:6.2f}%" if pct is not None and pct < 100 else (f"{pct*100:.1f}%" if pct is not None else "  —   ")


def rel_red(baseline: float, picon: float) -> str:
    """Relative reduction by PI-CON over baseline (positive = PI-CON better)."""
    return f"{(baseline - picon) / baseline * 100:+6.1f}%"


def main():
    data = json.loads(JSON_PATH.read_text())

    print("=" * 100)
    print(f"BASELINE AUDIT — DEDUP TRIG LSQ AND RBF ε SWEEP")
    print(f"(K={data['K']}, T=201, Re=10000 Kolmogorov; PI-CON multi-seed n=5)")
    print("=" * 100)

    print("\n--- Div-free Trig LSQ comparison ---")
    print(f"{'method':<55} | {'KE%':>8} | {'u_L2%':>8} | {'v_L2%':>8} | {'ω_L2%':>8} | red u")
    print("-" * 115)
    rows = []
    for k, v in data.items():
        if "trig LSQ" in k or "trig-lsq" in k.lower():
            rows.append((k, v))
    for name, m in rows:
        print(f"{name[:55]:<55} | {fmt(m['ke']):>8} | {fmt(m['u_l2']):>8} | "
              f"{fmt(m['v_l2']):>8} | {fmt(m['omega_l2']):>8} | "
              f"{rel_red(m['u_l2']*100, PICON_U_L2):>7}")

    print("\n--- RBF Gaussian ε sweep ---")
    print(f"{'method':<55} | {'KE%':>8} | {'u_L2%':>8} | {'v_L2%':>8} | {'ω_L2%':>8} | red u")
    print("-" * 115)
    for k, v in data.items():
        if "RBF Gaussian" in k:
            print(f"{k[:55]:<55} | {fmt(v['ke']):>8} | {fmt(v['u_l2']):>8} | "
                  f"{fmt(v['v_l2']):>8} | {fmt(v['omega_l2']):>8} | "
                  f"{rel_red(v['u_l2']*100, PICON_U_L2):>7}")

    print("\n--- PI-CON reference ---")
    print(f"{'PI-CON (ours, n=5)':<55} | {PICON_KE:>7.2f}% | {PICON_U_L2:>7.2f}% | "
          f"{PICON_V_L2:>7.2f}% | {PICON_W_L2:>7.2f}% | --- ")

    # ── Thesis table replacement candidates ──
    print("\n" + "=" * 100)
    print("THESIS chapter04.tex line 274-276 candidate replacement:")
    print("=" * 100)
    print("Best Trig LSQ (lowest u_L2) candidates among dedup k≤{5, 8, 12} × ridge {0, 1e-3}:")
    trig_dedup = [(k, v) for k, v in data.items() if "dedup" in k and not (v.get("u_l2", 1) > 1)]
    trig_dedup.sort(key=lambda kv: kv[1]["u_l2"])
    for k, v in trig_dedup[:3]:
        print(f"  {k}")
        print(f"    KE={v['ke']*100:.2f}%  u_L2={v['u_l2']*100:.2f}%  v_L2={v['v_l2']*100:.2f}%  ω_L2={v['omega_l2']*100:.2f}%")
    print()
    print("Best RBF Gaussian ε (lowest u_L2):")
    rbf_g = [(k, v) for k, v in data.items() if "RBF Gaussian (" in k]
    rbf_g.sort(key=lambda kv: kv[1]["u_l2"])
    for k, v in rbf_g[:5]:
        print(f"  {k}")
        print(f"    KE={v['ke']*100:.2f}%  u_L2={v['u_l2']*100:.2f}%  v_L2={v['v_l2']*100:.2f}%  ω_L2={v['omega_l2']*100:.2f}%")


if __name__ == "__main__":
    main()
