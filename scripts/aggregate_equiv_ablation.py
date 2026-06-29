"""彙總 EXP-281/282/283 對等 ablation 的 15 個 eval series.npz → per-architecture mean±std。

純 numpy（無 GPU），可在 lab head node 或本地跑（讀已產出的 series.npz）。
輸出 stdout 表 + artifacts/equiv_ablation_summary.json。
"""
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPECS = [f"281{s}" for s in ["", "b", "c", "d", "e"]] \
      + [f"282{s}" for s in ["", "b", "c", "d", "e"]] \
      + [f"283{s}" for s in ["", "b", "c", "d", "e"]]


def arch_of(eid: str) -> str:
    return ("B0 (281)" if eid.startswith("281")
            else "B1 (282)" if eid.startswith("282") else "B2 (283)")


def main() -> None:
    groups: dict[str, list] = {"B0 (281)": [], "B1 (282)": [], "B2 (283)": []}
    for eid in SPECS:
        p = ROOT / f"artifacts/kolmogorov/equal_budget/eval/eval_{eid}/series.npz"
        if not p.exists():
            print(f"[missing] eval_{eid}/series.npz")
            continue
        d = np.load(p)
        groups[arch_of(eid)].append((
            eid,
            float(d["KE_rel_err"].mean()) * 100,
            float(d["u_rel_L2"].mean()) * 100,
            float(d["v_rel_L2"].mean()) * 100,
            float(d["omega_rel_L2"].mean()) * 100,
        ))

    print("\n=== EXP-281/282/283 對等 ablation (20k × n=5, time-mean) ===")
    print(f"{'arch':<10}{'n':>3}  {'KE%':>14}  {'u-L2%':>14}  {'v-L2%':>14}  {'ω-L2%':>14}")
    summary = {}
    for k, rows in groups.items():
        if not rows:
            print(f"{k:<10}  (無資料)")
            continue
        arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
        m = arr.mean(0)
        s = arr.std(0, ddof=1) if len(rows) > 1 else np.zeros(4)
        print(f"{k:<10}{len(rows):>3}  {m[0]:>7.2f}±{s[0]:<5.2f}  {m[1]:>7.2f}±{s[1]:<5.2f}  "
              f"{m[2]:>7.2f}±{s[2]:<5.2f}  {m[3]:>7.2f}±{s[3]:<5.2f}")
        summary[k] = {
            "n": len(rows),
            "KE_mean": round(m[0], 4), "KE_std": round(s[0], 4),
            "u_mean": round(m[1], 4), "v_mean": round(m[2], 4), "omega_mean": round(m[3], 4),
            "per_seed": [{"id": r[0], "KE": round(r[1], 4), "u": round(r[2], 4),
                          "v": round(r[3], 4), "omega": round(r[4], 4)} for r in rows],
        }
    print("\n對照 B3 (EXP-245, 20k × n=5): KE 5.71 ± 0.11 %  (Table 4.3)")
    (ROOT / "artifacts/equiv_ablation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[wrote] artifacts/equiv_ablation_summary.json")


if __name__ == "__main__":
    main()
