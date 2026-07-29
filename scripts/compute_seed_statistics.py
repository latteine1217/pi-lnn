"""Multi-seed statistical comparison: B3 (Ours) vs B0 / B1 / B2 ablation cells.

對應 equal-budget ablation（EXP-281/282/283 vs EXP-245，皆為 1024-collocation
20k-iteration n=5-seed）。對每個 metric 計算：

- 每組 mean ± std（ddof=1）
- Welch's two-sample t-test（不假設等變異）+ Welch-Satterthwaite df
- Cohen's d（pooled SD）
- 95% CI for mean difference
- Bonferroni correction（預設 k = primary metrics × comparisons）
- 2x2 component decomposition（cross-attention / CfC / interaction，about B0）

輸出 markdown table + 機讀 JSON。

--- 資料來源選擇 ---
主來源是 `series.npz`，因為它是唯一涵蓋全部 20 個 run 的檔案：
EXP-283（B2）五個 seed 的 evaluator 只留下 series.npz，沒有 summary.json。
`series.npz` 的逐時序列取 mean 等價於 `summary.json` 的 `time_local.*.mean`
（實測 rel-diff ~1e-6，float32 捨入級），腳本在 summary.json 存在時會自動核對。

--- Provenance guard ---
summary.json 存在時強制檢查（見 CLAUDE.md KNOWN_PITFALLS「Evaluator 使用規則」）：
1. checkpoint 必須是 `picon_kolmogorov_final.pt`（ScheduleFree eval-mode y_t）；
   用 `step_*.pt`（train-mode x_t）評估會使 KE 偏移達 0.28pp — 2026-07-17 圖表不同源事故的根因。
2. config basename 必須與該 run 宣稱的 config 相符。
3. series.npz 與 summary.json 的 metric 必須一致。
缺 summary.json 的 run 標記為 UNVERIFIED 並在輸出中列出；`--strict` 可將其升級為錯誤。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent

# multi-seed suffix 對照（CLAUDE.md Numbering_Reference）：_a/_b/_c/_d/_e = seed 42/1/2/3/4
SEEDS = [42, 1, 2, 3, 4]

# 架構語義由 config 驗證而非命名慣例推定：
#   B0 = use_vanilla_deeponet=true（覆寫 CfC/attention）
#   B1 = CfC, no cross-attn
#   B2 = cross-attn, no CfC（num_temporal_cfc_layers=0）
#   B3 = full PI-CON
ARCH_LABELS = {
    "B0": "Vanilla DeepONet",
    "B1": "CfC, no cross-attn",
    "B2": "cross-attn, no CfC",
    "B3": "Full PI-CON (Ours)",
}

# eval 目錄 + 該 run 應對應的 config basename（provenance guard 用）
RUNS: dict[str, dict[int, tuple[str, str]]] = {
    "B3": {
        42: ("artifacts/exp245_seeds/eval_245_seeda_final", "exp_245_b3_les_T50.toml"),
        1:  ("artifacts/exp245_seeds/eval_245_seedb_final", "exp_245b_b3_les_T50_seed1.toml"),
        2:  ("artifacts/exp245_seeds/eval_245_seedc_final", "exp_245c_b3_les_T50_seed2.toml"),
        3:  ("artifacts/exp245_seeds/eval_245_seedd_final", "exp_245d_b3_les_T50_seed3.toml"),
        4:  ("artifacts/exp245_seeds/eval_245_seede_final", "exp_245e_b3_les_T50_seed4.toml"),
    },
    "B0": {
        42: ("artifacts/kolmogorov/equal_budget/eval/eval_281",  "exp_281_b0_les_T50_20k.toml"),
        1:  ("artifacts/kolmogorov/equal_budget/eval/eval_281b", "exp_281b_b0_les_T50_20k_seed1.toml"),
        2:  ("artifacts/kolmogorov/equal_budget/eval/eval_281c", "exp_281c_b0_les_T50_20k_seed2.toml"),
        3:  ("artifacts/kolmogorov/equal_budget/eval/eval_281d", "exp_281d_b0_les_T50_20k_seed3.toml"),
        4:  ("artifacts/kolmogorov/equal_budget/eval/eval_281e", "exp_281e_b0_les_T50_20k_seed4.toml"),
    },
    "B1": {
        42: ("artifacts/kolmogorov/equal_budget/eval/eval_282",  "exp_282_b1_les_T50_20k.toml"),
        1:  ("artifacts/kolmogorov/equal_budget/eval/eval_282b", "exp_282b_b1_les_T50_20k_seed1.toml"),
        2:  ("artifacts/kolmogorov/equal_budget/eval/eval_282c", "exp_282c_b1_les_T50_20k_seed2.toml"),
        3:  ("artifacts/kolmogorov/equal_budget/eval/eval_282d", "exp_282d_b1_les_T50_20k_seed3.toml"),
        4:  ("artifacts/kolmogorov/equal_budget/eval/eval_282e", "exp_282e_b1_les_T50_20k_seed4.toml"),
    },
    "B2": {
        42: ("artifacts/kolmogorov/equal_budget/eval/eval_283",  "exp_283_b2_les_T50_20k.toml"),
        1:  ("artifacts/kolmogorov/equal_budget/eval/eval_283b", "exp_283b_b2_les_T50_20k_seed1.toml"),
        2:  ("artifacts/kolmogorov/equal_budget/eval/eval_283c", "exp_283c_b2_les_T50_20k_seed2.toml"),
        3:  ("artifacts/kolmogorov/equal_budget/eval/eval_283d", "exp_283d_b2_les_T50_20k_seed3.toml"),
        4:  ("artifacts/kolmogorov/equal_budget/eval/eval_283e", "exp_283e_b2_les_T50_20k_seed4.toml"),
    },
}

REFERENCE = "B3"
COMPARISONS = ["B0", "B1", "B2"]


@dataclass(frozen=True)
class MetricSpec:
    """series.npz key -> 顯示名。scale 把 fraction 轉成論文用的 %。"""
    display: str
    series_key: str
    summary_key: str          # summary.json 的 dot-notation，供一致性核對
    scale: float
    primary: bool


METRICS: list[MetricSpec] = [
    MetricSpec("u_L2 (%)",       "u_rel_L2",     "time_local.u_rel_l2.mean",     100.0, True),
    MetricSpec("v_L2 (%)",       "v_rel_L2",     "time_local.v_rel_l2.mean",     100.0, True),
    MetricSpec("omega_L2 (%)",   "omega_rel_L2", "time_local.omega_rel_l2.mean", 100.0, True),
    MetricSpec("KE rel-err (%)", "KE_rel_err",   "time_local.ke_rel_err.mean",   100.0, True),
    MetricSpec("div_L2",         "div_l2",       "time_local.div_l2.mean",         1.0, False),
    MetricSpec("div_ratio (%)",  "div_ratio",    "time_local.div_ratio.mean",    100.0, False),
]

# NOTE: ek_ratio_kf 是 summary.json 的標量欄位，series.npz 沒有；EXP-283 缺 summary.json
#       故此 metric 無法涵蓋全部四組，暫不納入比較。


def get_nested(d: dict, dotted_key: str):
    cur = d
    for k in dotted_key.split("."):
        cur = cur[k]
    return cur


@dataclass
class RunData:
    arch: str
    seed: int
    eval_dir: Path
    values: dict[str, float]           # display name -> metric 值
    verified: bool                     # 是否通過 summary.json provenance 檢查
    notes: list[str] = field(default_factory=list)


def load_run(arch: str, seed: int, rel_dir: str, expected_config: str,
             consistency_rtol: float) -> RunData:
    """從 series.npz 取值；summary.json 存在時做 provenance + 一致性檢查。"""
    eval_dir = REPO_ROOT / rel_dir
    series_path = eval_dir / "series.npz"
    if not series_path.exists():
        raise FileNotFoundError(f"{arch} seed={seed}: 缺 series.npz → {series_path}")

    series = np.load(series_path)
    values: dict[str, float] = {}
    for spec in METRICS:
        if spec.series_key not in series:
            raise KeyError(f"{arch} seed={seed}: series.npz 缺 key '{spec.series_key}'")
        values[spec.display] = float(series[spec.series_key].mean()) * spec.scale

    notes: list[str] = []
    summary_path = eval_dir / "summary.json"
    if not summary_path.exists():
        notes.append("無 summary.json → checkpoint/config provenance 未驗證")
        return RunData(arch, seed, eval_dir, values, verified=False, notes=notes)

    with summary_path.open() as f:
        summary = json.load(f)

    # 1. checkpoint 必須是 final.pt（eval-mode y_t），不可是 step_*.pt（train-mode x_t）
    ckpt = Path(str(summary.get("checkpoint", ""))).name
    if ckpt != "picon_kolmogorov_final.pt":
        raise ValueError(
            f"{arch} seed={seed}: checkpoint 是 '{ckpt}'，必須是 picon_kolmogorov_final.pt。"
            " step_*.pt 存的是 ScheduleFree train-mode 權重，評估品質差 5-30%。"
        )

    # 2. config 對應
    cfg = Path(str(summary.get("config", ""))).name
    if cfg != expected_config:
        raise ValueError(
            f"{arch} seed={seed}: config 是 '{cfg}'，預期 '{expected_config}'。"
            " eval 目錄與 run 對應錯誤。"
        )

    # 3. series.npz 與 summary.json 一致
    for spec in METRICS:
        try:
            ref = float(get_nested(summary, spec.summary_key)) * spec.scale
        except KeyError:
            continue          # summary schema 較舊，缺該欄位就跳過核對
        got = values[spec.display]
        if abs(got - ref) > consistency_rtol * max(abs(ref), 1e-30):
            raise ValueError(
                f"{arch} seed={seed}: metric '{spec.display}' series.npz={got:.8g} 與 "
                f"summary.json={ref:.8g} 不一致（rtol={consistency_rtol:g}）。"
            )

    return RunData(arch, seed, eval_dir, values, verified=True, notes=notes)


@dataclass
class GroupStats:
    name: str
    values: np.ndarray

    @property
    def mean(self) -> float:
        return float(self.values.mean())

    @property
    def std(self) -> float:
        # ddof=1 → 樣本標準差，與 paper 慣例一致
        return float(self.values.std(ddof=1))

    @property
    def n(self) -> int:
        return len(self.values)


@dataclass
class ComparisonResult:
    metric: str
    arch: str                 # 被比較的架構（reference 恆為 B3）
    other: GroupStats
    ref: GroupStats
    diff: float               # other - ref；正值代表 B3 較好（全為 error metric）
    ci_low: float
    ci_high: float
    t_stat: float
    df_welch: float
    p_value: float
    p_bonferroni: float | None
    cohens_d: float


def welch_compare(other_vals: np.ndarray, ref_vals: np.ndarray, metric: str,
                  arch: str, is_primary: bool, bonferroni_k: int) -> ComparisonResult:
    """Welch's t-test（unequal variances）+ Cohen's d + 95% CI for difference。"""
    other = GroupStats(arch, other_vals)
    ref = GroupStats(REFERENCE, ref_vals)

    diff = other.mean - ref.mean

    # Welch standard error（不 pool 變異）
    se = math.sqrt(other.std**2 / other.n + ref.std**2 / ref.n)

    # Welch-Satterthwaite degrees of freedom
    num = (other.std**2 / other.n + ref.std**2 / ref.n) ** 2
    den = ((other.std**2 / other.n) ** 2 / (other.n - 1)
           + (ref.std**2 / ref.n) ** 2 / (ref.n - 1))
    df = num / den

    t_stat = diff / se
    p = float(2 * stats.t.sf(abs(t_stat), df))          # 雙尾；sf 比 1-cdf 在尾端穩定

    t_crit = float(stats.t.ppf(0.975, df))
    ci_low = diff - t_crit * se
    ci_high = diff + t_crit * se

    pooled_sd = math.sqrt((other.std**2 + ref.std**2) / 2)
    d = diff / pooled_sd if pooled_sd > 0 else math.inf

    p_bonf = min(p * bonferroni_k, 1.0) if is_primary else None

    return ComparisonResult(
        metric=metric, arch=arch, other=other, ref=ref,
        diff=diff, ci_low=ci_low, ci_high=ci_high,
        t_stat=t_stat, df_welch=df,
        p_value=p, p_bonferroni=p_bonf, cohens_d=d,
    )


def fmt_p(p: float) -> str:
    """論文慣例：太小不寫精確值（n=5 的尾端 p 值有效位數不可信）。"""
    if p < 1e-4:
        return "< 1e-4"
    if p < 1e-3:
        return "< 0.001"
    if p < 0.01:
        return "< 0.01"
    if p < 0.05:
        return f"{p:.3f}"
    return f"{p:.3f} (n.s.)"


def decompose_2x2(means: dict[str, float]) -> dict[str, float]:
    """2x2 component decomposition about the B0 reference cell（該處分解可加）。

    B0 = 皆無；B1 = CfC only；B2 = cross-attn only；B3 = both。
    """
    cross_attn = means["B2"] - means["B0"]
    cfc = means["B1"] - means["B0"]
    total = means["B3"] - means["B0"]
    interaction = total - cross_attn - cfc
    return {
        "cross_attention_main": cross_attn,
        "cfc_main": cfc,
        "interaction": interaction,
        "total_b3_minus_b0": total,
        "sum_check": cross_attn + cfc + interaction,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "artifacts" / "analysis" / "seed_statistics.json",
                    help="機讀 JSON 輸出路徑（注意 artifacts/ 被 .gitignore）")
    ap.add_argument("--bonferroni-k", type=int, default=None,
                    help="Bonferroni 家族大小；預設 = primary metrics × comparisons")
    ap.add_argument("--strict", action="store_true",
                    help="任一 run 缺 summary.json provenance 即視為錯誤")
    ap.add_argument("--consistency-rtol", type=float, default=1e-4,
                    help="series.npz 與 summary.json 的相對容差")
    args = ap.parse_args()

    # === 載入 ===
    runs: dict[str, dict[int, RunData]] = {}
    for arch, seed_map in RUNS.items():
        runs[arch] = {}
        for seed in SEEDS:
            rel_dir, expected_cfg = seed_map[seed]
            runs[arch][seed] = load_run(arch, seed, rel_dir, expected_cfg,
                                        args.consistency_rtol)

    unverified = [(r.arch, r.seed, r.notes)
                  for a in runs for r in runs[a].values() if not r.verified]
    if unverified and args.strict:
        raise SystemExit(
            "[STRICT] 以下 run 無 summary.json，provenance 未驗證：\n  "
            + "\n  ".join(f"{a} seed={s}: {'; '.join(n)}" for a, s, n in unverified)
        )

    n_primary = sum(1 for m in METRICS if m.primary)
    bonf_k = args.bonferroni_k if args.bonferroni_k is not None else n_primary * len(COMPARISONS)

    # === 檢定 ===
    results: list[ComparisonResult] = []
    for spec in METRICS:
        ref_vals = np.array([runs[REFERENCE][s].values[spec.display] for s in SEEDS])
        for arch in COMPARISONS:
            other_vals = np.array([runs[arch][s].values[spec.display] for s in SEEDS])
            results.append(welch_compare(other_vals, ref_vals, spec.display,
                                         arch, spec.primary, bonf_k))

    # === 輸出 ===
    print("=" * 100)
    print("Multi-Seed Statistical Comparison (equal-budget ablation: 1024 collocation, 20k iter)")
    print("=" * 100)
    print(f"Reference       : {REFERENCE} ({ARCH_LABELS[REFERENCE]})")
    print(f"Comparisons     : {', '.join(f'{a} ({ARCH_LABELS[a]})' for a in COMPARISONS)}")
    print(f"N seeds / group : {len(SEEDS)}  (seeds: {SEEDS})")
    print("Test            : Welch's two-sample t-test (unequal variances, Welch-Satterthwaite df)")
    print(f"Multiple compar.: Bonferroni k={bonf_k} "
          f"({n_primary} primary metrics x {len(COMPARISONS)} comparisons)")
    print(f"Metric source   : series.npz (time-series mean), "
          f"cross-checked against summary.json where available (rtol={args.consistency_rtol:g})")
    if unverified:
        print()
        print("[UNVERIFIED PROVENANCE] 以下 run 無 summary.json，checkpoint/config 未經核對：")
        for a, s, n in unverified:
            print(f"  - {a} seed={s}: {'; '.join(n)}")
    print()

    # 各組描述統計
    print("### Group summary (mean ± std, n=5)")
    print()
    header = "| Metric | " + " | ".join(f"{a}" for a in ["B0", "B1", "B2", "B3"]) + " |"
    print(header)
    print("|---" * 5 + "|")
    for spec in METRICS:
        cells = []
        for arch in ["B0", "B1", "B2", "B3"]:
            vals = np.array([runs[arch][s].values[spec.display] for s in SEEDS])
            cells.append(f"{vals.mean():.2f} ± {vals.std(ddof=1):.2f}")
        print(f"| {spec.display} | " + " | ".join(cells) + " |")
    print()

    # 檢定表
    print(f"### Welch t-test vs {REFERENCE}")
    print()
    print("| Metric | vs | other mean ± std | B3 mean ± std | Δ (other−B3) | 95% CI | t | df | p | p_Bonf | Cohen's d |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        p_bonf_s = fmt_p(r.p_bonferroni) if r.p_bonferroni is not None else "—"
        print(f"| {r.metric} | {r.arch} | "
              f"{r.other.mean:.2f} ± {r.other.std:.2f} | {r.ref.mean:.2f} ± {r.ref.std:.2f} | "
              f"{r.diff:+.2f} | [{r.ci_low:+.2f}, {r.ci_high:+.2f}] | "
              f"{r.t_stat:+.2f} | {r.df_welch:.1f} | {fmt_p(r.p_value)} | {p_bonf_s} | "
              f"{r.cohens_d:+.2f} |")
    print()

    # 2x2 分解
    print("### 2x2 component decomposition (about B0 reference cell)")
    print()
    decompositions: dict[str, dict[str, float]] = {}
    print("| Metric | cross-attn main | CfC main | interaction | total (B3−B0) |")
    print("|---|---|---|---|---|")
    for spec in METRICS:
        means = {a: float(np.mean([runs[a][s].values[spec.display] for s in SEEDS]))
                 for a in ["B0", "B1", "B2", "B3"]}
        dec = decompose_2x2(means)
        decompositions[spec.display] = dec
        print(f"| {spec.display} | {dec['cross_attention_main']:+.2f} | {dec['cfc_main']:+.2f} | "
              f"{dec['interaction']:+.2f} | {dec['total_b3_minus_b0']:+.2f} |")
    print()

    # === 機讀 JSON ===
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference": REFERENCE,
        "comparisons": COMPARISONS,
        "arch_labels": ARCH_LABELS,
        "seeds": SEEDS,
        "test": "Welch two-sample t-test (unequal variances, Welch-Satterthwaite df)",
        "bonferroni_k": bonf_k,
        "metric_source": "series.npz time-series mean",
        "consistency_rtol": args.consistency_rtol,
        "runs": {
            arch: {
                str(seed): {
                    "eval_dir": str(r.eval_dir.relative_to(REPO_ROOT)),
                    "verified_provenance": r.verified,
                    "notes": r.notes,
                    "values": r.values,
                }
                for seed, r in seed_map.items()
            }
            for arch, seed_map in runs.items()
        },
        "groups": {
            spec.display: {
                arch: {
                    "values": [runs[arch][s].values[spec.display] for s in SEEDS],
                    "mean": float(np.mean([runs[arch][s].values[spec.display] for s in SEEDS])),
                    "std": float(np.std([runs[arch][s].values[spec.display] for s in SEEDS], ddof=1)),
                    "n": len(SEEDS),
                }
                for arch in ["B0", "B1", "B2", "B3"]
            }
            for spec in METRICS
        },
        "comparisons_result": [
            {
                "metric": r.metric,
                "arch": r.arch,
                "diff_other_minus_ref": r.diff,
                "ci_95": [r.ci_low, r.ci_high],
                "t_statistic": r.t_stat,
                "df_welch": r.df_welch,
                "p_value": r.p_value,
                "p_bonferroni": r.p_bonferroni,
                "cohens_d": r.cohens_d,
            }
            for r in results
        ],
        "decomposition_2x2": decompositions,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Machine-readable output → {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
