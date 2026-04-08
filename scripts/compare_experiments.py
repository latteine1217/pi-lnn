"""Compare multiple DeepONet + CfC evaluation runs side by side.

What: 輸入多個 eval summary.json，產生疊圖比較。
Why: 單看個別實驗的圖無法直接判斷改動的效果；疊圖讓 phase error、RMSE、
     amplitude 的差異一目了然，省去手動對照數字的步驟。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple eval runs.")
    parser.add_argument(
        "--evals",
        type=Path,
        nargs="+",
        required=True,
        help="Eval output directories (each must contain summary.json).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Display labels for each eval dir (default: dir name).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for comparison plots.",
    )
    return parser.parse_args()


def load_summary(eval_dir: Path) -> dict:
    path = eval_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"summary.json 不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def phase_error_series(steps: list[dict]) -> np.ndarray:
    """每時步的 wrapped phase error（rad）。"""
    return np.array([
        float(np.angle(np.exp(1j * (s["kf_phase_pred"] - s["kf_phase_ref"]))))
        for s in steps
    ])


def plot_phase_error(output_path: Path, time_vals: np.ndarray,
                     series: list[np.ndarray], labels: list[str]) -> None:
    """What: Phase error (wrapped) vs time 疊圖。
    Why: 直接顯示各實驗在每個時步的 phase 偏差，判斷改動是否縮小偏差。
    """
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for err, label in zip(series, labels):
        ax.plot(time_vals, err, marker="o", markersize=3, label=label)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Forcing Mode Phase Error vs Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Phase Error [rad]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_rmse(output_path: Path, time_vals: np.ndarray,
              u_series: list[np.ndarray], v_series: list[np.ndarray],
              labels: list[str]) -> None:
    """What: u / v RMSE vs time 疊圖。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for u_err, label in zip(u_series, labels):
        axes[0].plot(time_vals, u_err, marker="o", markersize=3, label=label)
    for v_err, label in zip(v_series, labels):
        axes[1].plot(time_vals, v_err, marker="o", markersize=3, label=label)
    for ax, title in zip(axes, ["u RMSE vs Time", "v RMSE vs Time"]):
        ax.set_xlabel("Time")
        ax.set_ylabel("RMSE")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_kf_amplitude(output_path: Path, time_vals: np.ndarray,
                      ref_series: np.ndarray, pred_series: list[np.ndarray],
                      labels: list[str]) -> None:
    """What: Forcing mode amplitude vs time 疊圖（含 DNS 參考）。"""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(time_vals, ref_series, color="black", linewidth=1.5,
            linestyle="--", marker="o", markersize=3, label="DNS")
    for amp, label in zip(pred_series, labels):
        ax.plot(time_vals, amp, marker="o", markersize=3, label=label)
    ax.set_title("Forcing Mode Amplitude vs Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metrics_bar(output_path: Path, summaries: list[dict],
                     labels: list[str]) -> None:
    """What: 關鍵 scalar 指標的橫向 bar chart 比較。
    Why: 讓不同實驗的整體指標差異一眼可見，補充時序圖不直觀的部分。
    """
    metric_keys = [
        ("u_rmse_mean",       "u RMSE mean"),
        ("v_rmse_mean",       "v RMSE mean"),
        ("ke_rel_err_mean",   "KE rel-err"),
        ("ens_rel_err_mean",  "Ens rel-err"),
        ("kf_amp_ratio_last", "kf amp ratio"),
    ]
    n_metrics = len(metric_keys)
    n_runs = len(summaries)
    x = np.arange(n_metrics)
    width = 0.8 / n_runs

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for i, (summary, label) in enumerate(zip(summaries, labels)):
        vals = [summary.get(k, float("nan")) for k, _ in metric_keys]
        offset = (i - (n_runs - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=label)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in metric_keys], rotation=15, ha="right")
    ax.set_title("Metrics Comparison")
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def print_summary_table(summaries: list[dict], labels: list[str]) -> None:
    metric_keys = [
        ("u_rmse_mean",        "u RMSE mean"),
        ("v_rmse_mean",        "v RMSE mean"),
        ("ke_rel_err_mean",    "KE rel-err"),
        ("ens_rel_err_mean",   "Ens rel-err"),
        ("kf_amp_ratio_last",  "kf amp ratio"),
        ("kf_phase_err_last",  "kf phase err (rad)"),
    ]
    col_w = max(len(l) for l in labels) + 2
    print("=== Metrics Comparison ===")
    header = f"{'Metric':<24}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("-" * len(header))
    for key, name in metric_keys:
        row = f"{name:<24}"
        for s in summaries:
            v = s.get(key, float("nan"))
            row += f"{v:>{col_w}.4f}"
        print(row)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = args.labels or [d.name for d in args.evals]
    if len(labels) != len(args.evals):
        raise ValueError("--labels 數量必須與 --evals 相同。")

    summaries = [load_summary(d.resolve()) for d in args.evals]

    # 時序資料
    steps0 = summaries[0]["steps"]
    time_vals = np.array([s["time"] for s in steps0])

    phase_errors = [phase_error_series(sm["steps"]) for sm in summaries]
    u_rmse_series = [np.array([s["u_rmse"] for s in sm["steps"]]) for sm in summaries]
    v_rmse_series = [np.array([s["v_rmse"] for s in sm["steps"]]) for sm in summaries]
    amp_pred_series = [np.array([s["kf_amp_pred"] for s in sm["steps"]]) for sm in summaries]
    amp_ref_series = np.array([s["kf_amp_ref"] for s in steps0])

    plot_phase_error(output_dir / "compare_phase_error_vs_time.png",
                     time_vals, phase_errors, labels)
    plot_rmse(output_dir / "compare_rmse_vs_time.png",
              time_vals, u_rmse_series, v_rmse_series, labels)
    plot_kf_amplitude(output_dir / "compare_kf_amplitude_vs_time.png",
                      time_vals, amp_ref_series, amp_pred_series, labels)
    plot_metrics_bar(output_dir / "compare_metrics_bar.png", summaries, labels)

    print_summary_table(summaries, labels)
    print()
    print(f"compare_phase_error:  {output_dir / 'compare_phase_error_vs_time.png'}")
    print(f"compare_rmse:         {output_dir / 'compare_rmse_vs_time.png'}")
    print(f"compare_kf_amplitude: {output_dir / 'compare_kf_amplitude_vs_time.png'}")
    print(f"compare_metrics_bar:  {output_dir / 'compare_metrics_bar.png'}")


if __name__ == "__main__":
    main()
