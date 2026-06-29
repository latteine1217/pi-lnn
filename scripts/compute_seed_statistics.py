"""Multi-seed statistical comparison: B3 (Ours) vs B0 (Vanilla DeepONet).

對 5 個 random seeds 的 evaluator summary.json 計算：
- 每組 metric 的 mean ± std
- Welch's two-sample t-test (不假設等變異)
- Cohen's d (pooled SD effect size)
- 95% CI for mean difference
- Bonferroni correction (4 同時比較)

輸出 markdown table，可直接貼進論文。

Reproducibility: 路徑硬編在 ARTIFACT_PATHS；改 seed 集合需同步更新。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent

# 5 seeds × 2 architectures。路徑取最近一次 rerun 或 per-exp eval。
ARTIFACT_PATHS: dict[str, dict[int, str]] = {
    "B3": {  # Ours: CfC-DeepONet + cross-attention + AL-continuity
        42: "artifacts/eval-rerun-2026-05-09/exp080-al-4task-rho01/summary.json",
        1:  "artifacts/eval-rerun-2026-05-11/exp093-b3-seed1/summary.json",
        2:  "artifacts/eval-rerun-2026-05-12/exp094-b3-seed2/summary.json",
        3:  "artifacts/kolmogorov/stable/exp097-b3-seed3/deeponet-cfc-eval/summary.json",
        4:  "artifacts/kolmogorov/stable/exp098-b3-seed4/deeponet-cfc-eval/summary.json",
    },
    "B0": {  # Vanilla DeepONet (no CfC, no cross-attention, no AL)
        42: "artifacts/eval-rerun-2026-05-11/exp088-vanilla-deeponet/summary.json",
        1:  "artifacts/eval-rerun-2026-05-11/exp095-b0-seed1/summary.json",
        2:  "artifacts/eval-rerun-2026-05-12/exp096-b0-seed2/summary.json",
        3:  "artifacts/kolmogorov/stable/exp099-b0-seed3/deeponet-cfc-eval/summary.json",
        4:  "artifacts/kolmogorov/stable/exp100-b0-seed4/deeponet-cfc-eval/summary.json",
    },
}

# 報告 metric -> summary.json 路徑（dot-notation）
# 主指標用 time-averaged，因 paper claim 是「整段時間性能」
METRICS: dict[str, str] = {
    "u_L2 (%)":      "time_local.u_rel_l2.mean",
    "v_L2 (%)":      "time_local.v_rel_l2.mean",
    "omega_L2 (%)":  "time_local.omega_rel_l2.mean",
    "KE rel-err (%)":"time_local.ke_rel_err.mean",
    "div_L2":        "time_local.div_l2.mean",
    "ek_ratio_kf":   "ek_ratio_kf_last",
}

# 主賣點 4 指標套 Bonferroni；div / ek 為 secondary 不修正
PRIMARY_METRICS = {"u_L2 (%)", "v_L2 (%)", "omega_L2 (%)", "KE rel-err (%)"}


def get_nested(d: dict, dotted_key: str):
    """取巢狀 dict 值，path 用 'a.b.c' 表示。"""
    cur = d
    for k in dotted_key.split("."):
        cur = cur[k]
    return cur


def load_metric(summary_path: Path, dotted_key: str) -> float:
    with summary_path.open() as f:
        data = json.load(f)
    val = float(get_nested(data, dotted_key))
    # KE/L2 在 summary.json 是 fraction (0.10)，論文要 % (10.0)
    if dotted_key.endswith(".mean") and not dotted_key.endswith("div_l2.mean") and not dotted_key.endswith("ek_ratio_kf"):
        if any(x in dotted_key for x in ("rel_l2", "rel_err")):
            val *= 100.0
    return val


@dataclass
class GroupStats:
    name: str
    values: np.ndarray

    @property
    def mean(self) -> float:
        return float(self.values.mean())

    @property
    def std(self) -> float:
        # ddof=1 → 樣本標準差（n-1 分母），與 paper 慣例一致
        return float(self.values.std(ddof=1))

    @property
    def n(self) -> int:
        return len(self.values)


@dataclass
class ComparisonResult:
    metric: str
    b0: GroupStats
    b3: GroupStats
    diff: float           # B0 - B3
    ci_low: float         # 95% CI for diff
    ci_high: float
    t_stat: float
    df_welch: float
    p_value: float
    p_bonferroni: float | None
    cohens_d: float


def welch_compare(b0_vals: np.ndarray, b3_vals: np.ndarray, metric: str, n_compare: int) -> ComparisonResult:
    """Welch's t-test (unequal variances) + Cohen's d + 95% CI for difference."""
    b0 = GroupStats("B0", b0_vals)
    b3 = GroupStats("B3", b3_vals)

    diff = b0.mean - b3.mean

    # Welch SE
    se = math.sqrt(b0.std**2 / b0.n + b3.std**2 / b3.n)

    # Welch-Satterthwaite degrees of freedom
    num = (b0.std**2 / b0.n + b3.std**2 / b3.n) ** 2
    den = (b0.std**2 / b0.n) ** 2 / (b0.n - 1) + (b3.std**2 / b3.n) ** 2 / (b3.n - 1)
    df = num / den

    t_stat = diff / se
    # 雙尾
    p = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    # 95% CI for difference
    t_crit = stats.t.ppf(0.975, df)
    ci_low = diff - t_crit * se
    ci_high = diff + t_crit * se

    # Cohen's d (pooled SD)
    pooled_sd = math.sqrt((b0.std**2 + b3.std**2) / 2)
    d = diff / pooled_sd if pooled_sd > 0 else math.inf

    # Bonferroni 修正：只對 primary metrics
    p_bonf = min(p * n_compare, 1.0) if metric in PRIMARY_METRICS else None

    return ComparisonResult(
        metric=metric,
        b0=b0, b3=b3,
        diff=diff, ci_low=ci_low, ci_high=ci_high,
        t_stat=t_stat, df_welch=df,
        p_value=p, p_bonferroni=p_bonf,
        cohens_d=d,
    )


def fmt_p(p: float) -> str:
    """論文慣例：太小不寫精確值。"""
    if p < 1e-4:
        return "< 1e-4"
    if p < 1e-3:
        return "< 0.001"
    if p < 0.01:
        return "< 0.01"
    if p < 0.05:
        return f"{p:.3f}"
    return f"{p:.3f} (n.s.)"


def main() -> None:
    n_primary = sum(1 for m in METRICS if m in PRIMARY_METRICS)

    results: list[ComparisonResult] = []
    for metric_name, key in METRICS.items():
        b3_vals = []
        b0_vals = []
        for seed in sorted(ARTIFACT_PATHS["B3"].keys()):
            b3_vals.append(load_metric(REPO_ROOT / ARTIFACT_PATHS["B3"][seed], key))
            b0_vals.append(load_metric(REPO_ROOT / ARTIFACT_PATHS["B0"][seed], key))
        results.append(welch_compare(np.array(b0_vals), np.array(b3_vals),
                                      metric_name, n_compare=n_primary))

    # === 輸出 ===
    print("=" * 80)
    print("Multi-Seed Statistical Comparison: B0 (Vanilla DeepONet) vs B3 (Ours)")
    print("=" * 80)
    print(f"N seeds per group: 5 (seeds: {sorted(ARTIFACT_PATHS['B3'].keys())})")
    print(f"Test: Welch's two-sample t-test (unequal variances)")
    print(f"Multiple comparison: Bonferroni (k={n_primary} primary metrics)")
    print()

    # Markdown table
    print("| Metric | B0 mean ± std (n=5) | B3 mean ± std (n=5) | Δ (B0−B3) | 95% CI | t | df | p | p_Bonf | Cohen's d |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        is_pct = "%" in r.metric
        unit = "" if is_pct else ""
        b0_s = f"{r.b0.mean:.2f} ± {r.b0.std:.2f}{unit}"
        b3_s = f"{r.b3.mean:.2f} ± {r.b3.std:.2f}{unit}"
        diff_s = f"{r.diff:+.2f}{unit}"
        ci_s = f"[{r.ci_low:+.2f}, {r.ci_high:+.2f}]"
        p_bonf_s = fmt_p(r.p_bonferroni) if r.p_bonferroni is not None else "—"
        print(f"| {r.metric} | {b0_s} | {b3_s} | {diff_s} | {ci_s} | "
              f"{r.t_stat:+.2f} | {r.df_welch:.1f} | {fmt_p(r.p_value)} | {p_bonf_s} | {r.cohens_d:+.2f} |")

    # 機讀 JSON
    out_json = REPO_ROOT / "artifacts" / "seed_statistics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump([{
            "metric": r.metric,
            "b0": {"values": r.b0.values.tolist(), "mean": r.b0.mean, "std": r.b0.std, "n": r.b0.n},
            "b3": {"values": r.b3.values.tolist(), "mean": r.b3.mean, "std": r.b3.std, "n": r.b3.n},
            "diff_b0_minus_b3": r.diff,
            "ci_95": [r.ci_low, r.ci_high],
            "t_statistic": r.t_stat,
            "df_welch": r.df_welch,
            "p_value": r.p_value,
            "p_bonferroni": r.p_bonferroni,
            "cohens_d": r.cohens_d,
        } for r in results], f, indent=2)
    print()
    print(f"Machine-readable output → {out_json}")


if __name__ == "__main__":
    main()
