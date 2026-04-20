"""scripts/sweep_re10000.py — Re=10000 超參數掃描入口。

What: 以 Optuna TPE + W&B 對 Re=10000 DeepONet+CfC 做超參數掃描。
Why:  EXP-043（KE=27.2%）已接近容量上限；系統掃描 lr / weight_decay /
      physics_loss_weight / continuity_weight / soap_precondition_frequency /
      use_locality_decay，明確量化各超參對物理一致性指標的影響。

執行方式:
    uv run python scripts/sweep_re10000.py [--trials N] [--device mps|cpu|cuda]
    uv run python scripts/sweep_re10000.py --trials 40 --study-name re10000-v3

設計原則:
    - 每個 trial 執行 1500 steps（快速排名），top-k 手動以完整 3000 步重跑。
    - Optuna MedianPruner：若 trial 前 300 steps 的 l_data 明顯高於中位數，提前終止。
    - W&B：每 step 記錄 l_data / l_ns / l_cont 等；系統 metrics 由 W&B agent 自動收集。
    - Optuna study 存入 SQLite（artifacts/sweep/optuna.db），斷點可續跑。

Objective 設計（v3）:
    v2 的 l_phys = l_ns + cont_w * l_cont，當 cont_w < 1 時 l_cont 被縮小，
    讓高散度違反（l_cont >> 1）看似合格（EXP-045 失敗根因）。
    v3 改為「標準化物理懲罰」：分別對 l_ns 和 l_cont 設獨立門檻，
    與訓練時的 cont_w 完全無關，確保物理違反無法被參數化規避。

    objective = l_data_tail
              + 0.05 * max(0, l_ns_tail   - L_NS_THRESHOLD)
              + 0.03 * max(0, l_cont_tail - L_CONT_THRESHOLD)

    門檻依據 v1 健康 trial（cont_w≈1.0）的實測範圍：
      l_ns   健康區間 ≈ 0.4–1.0  → L_NS_THRESHOLD   = 2.0（2× 上緣）
      l_cont 健康區間 ≈ 1.2–1.5  → L_CONT_THRESHOLD  = 3.0（2× 上緣）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 確保 src/ 在 import 路徑
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import optuna
import wandb

from lnn_kolmogorov import DEFAULT_LNN_ARGS, load_lnn_config, train_lnn_kolmogorov

# ── 常數 ──────────────────────────────────────────────────────────────────────
BASE_CONFIG = Path(__file__).parent.parent / "configs" / "sweep_re10000_base.toml"
WANDB_PROJECT = "pi-lnn-re10000-sweep"
OPTUNA_DB_DIR = Path("artifacts/sweep")
# 用這幾步的 l_data 平均作為 trial 最終指標（降低尾段雜訊）
METRIC_TAIL_STEPS = 200


# ── Objective ─────────────────────────────────────────────────────────────────
def make_objective(base_cfg: dict, device: str):
    """工廠函式：產生 Optuna objective，閉包 base_cfg 與 device。"""

    def objective(trial: optuna.Trial) -> float:
        # ── 超參數建議 ──────────────────────────────────────────────────────
        lr = trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True)
        # 下界改為 5e-3：防止 Optuna 學到「關掉物理 → 低 l_data」的作弊路徑
        phys_w = trial.suggest_float("physics_loss_weight", 5e-3, 5e-2, log=True)
        cont_w = trial.suggest_float("continuity_weight", 0.7, 1.5)
        soap_freq = trial.suggest_categorical("soap_precondition_frequency", [5, 10, 20])
        use_locality = trial.suggest_categorical("use_locality_decay", [False, True])

        # ── 組合 config ─────────────────────────────────────────────────────
        cfg = {**base_cfg}
        cfg.update({
            "learning_rate": lr,
            "weight_decay": wd,
            "physics_loss_weight": phys_w,
            "continuity_weight": cont_w,
            "soap_precondition_frequency": int(soap_freq),
            "use_locality_decay": bool(use_locality),
            "device": device,
            "artifacts_dir": f"artifacts/sweep/trial_{trial.number:04d}",
            # seed 固定，避免資料抽樣差異污染超參比較
            "seed": 42,
        })

        # ── W&B 初始化 ──────────────────────────────────────────────────────
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"trial_{trial.number:04d}",
            config={
                "learning_rate": lr,
                "weight_decay": wd,
                "physics_loss_weight": phys_w,
                "continuity_weight": cont_w,
                "soap_precondition_frequency": soap_freq,
                "use_locality_decay": use_locality,
                "iterations": cfg["iterations"],
                "d_model": cfg["d_model"],
                "trial_number": trial.number,
            },
            reinit=True,
            settings=wandb.Settings(
                # 讓 W&B 自動收集系統 metrics（CPU/GPU 溫度、記憶體）
                x_stats_sampling_interval=5,
            ),
        )

        # ── 訓練中 metrics 收集 ─────────────────────────────────────────────
        tail_data: list[float] = []
        tail_ns: list[float] = []
        tail_cont: list[float] = []
        iterations = int(cfg["iterations"])

        def log_fn(step: int, metrics: dict[str, float]) -> None:
            # 每 step 上報到 W&B
            wandb.log(metrics, step=step)

            # 收集尾段 l_data / l_ns / l_cont 供 objective 使用（v3：分開追蹤）
            if step > iterations - METRIC_TAIL_STEPS:
                tail_data.append(metrics["l_data"])
                tail_ns.append(metrics["l_ns"])
                tail_cont.append(metrics["l_cont"])

            # Optuna pruner：每 100 steps 回報一次，讓 MedianPruner 判斷是否剪枝
            if step % 100 == 0:
                trial.report(metrics["l_data"], step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        # ── 執行訓練 ────────────────────────────────────────────────────────
        try:
            train_lnn_kolmogorov(cfg, log_fn=log_fn)
        except optuna.TrialPruned:
            wandb.log({"pruned": True})
            run.finish(exit_code=1)
            raise

        # ── 計算 objective（v3 標準化物理懲罰）────────────────────────────
        # Why: v2 用 l_phys = l_ns + cont_w * l_cont 作為懲罰基礎，
        #      當 cont_w < 1 時 l_cont 被縮小，讓高散度違反看似合格（EXP-045 失效根因）。
        #      v3 改為對 l_ns 和 l_cont 分別設獨立門檻，與 cont_w 完全無關。
        #
        #      門檻依據 v1 健康 trial（cont_w≈1.0）的實測範圍：
        #        l_ns   健康區間 ≈ 0.4–1.0  → L_NS_THRESHOLD   = 2.0（2× 上緣）
        #        l_cont 健康區間 ≈ 1.2–1.5  → L_CONT_THRESHOLD = 3.0（2× 上緣）
        L_NS_THRESHOLD = 2.0
        L_CONT_THRESHOLD = 3.0
        PENALTY_COEFF_NS = 0.05
        PENALTY_COEFF_CONT = 0.03

        l_data_mean = float(sum(tail_data) / len(tail_data)) if tail_data else float("inf")
        l_ns_mean = float(sum(tail_ns) / len(tail_ns)) if tail_ns else float("inf")
        l_cont_mean = float(sum(tail_cont) / len(tail_cont)) if tail_cont else float("inf")
        penalty_ns = PENALTY_COEFF_NS * max(0.0, l_ns_mean - L_NS_THRESHOLD)
        penalty_cont = PENALTY_COEFF_CONT * max(0.0, l_cont_mean - L_CONT_THRESHOLD)
        final_metric = l_data_mean + penalty_ns + penalty_cont

        wandb.log({
            "final_l_data_tail": l_data_mean,
            "final_l_ns_tail": l_ns_mean,
            "final_l_cont_tail": l_cont_mean,
            "penalty_ns": penalty_ns,
            "penalty_cont": penalty_cont,
            "final_objective": final_metric,
        })
        run.finish()
        return final_metric

    return objective


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re=10000 超參數掃描（Optuna + W&B）")
    parser.add_argument("--trials", type=int, default=40, help="Optuna trial 總數（含剪枝）")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default=None,
        help="覆蓋 config 中的 device"
    )
    parser.add_argument(
        "--study-name", default="re10000-sweep-v3",
        help="Optuna study 名稱（同名可續跑）"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="若 study 已存在則續跑，否則重建"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 載入 base config ─────────────────────────────────────────────────────
    base_cfg = dict(DEFAULT_LNN_ARGS)
    base_cfg.update(load_lnn_config(BASE_CONFIG))
    device = args.device or base_cfg.get("device", "mps")

    # ── Optuna study ─────────────────────────────────────────────────────────
    OPTUNA_DB_DIR.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{OPTUNA_DB_DIR}/optuna.db"

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=8,    # 前 8 個 trial 不剪枝（累積基準）
        n_warmup_steps=300,    # 每個 trial 前 300 steps 不剪枝
        interval_steps=100,
    )

    load_if_exists = args.resume
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=load_if_exists,
    )

    objective = make_objective(base_cfg, device)

    print("=== Re=10000 Hyperparameter Sweep ===")
    print(f"study_name : {args.study_name}")
    print(f"storage    : {storage}")
    print(f"trials     : {args.trials}")
    print(f"device     : {device}")
    print(f"wandb      : project={WANDB_PROJECT}")
    print()

    study.optimize(objective, n_trials=args.trials, catch=(Exception,))

    # ── 結果摘要 ─────────────────────────────────────────────────────────────
    print("\n=== Sweep 完成 ===")
    best = study.best_trial
    print(f"Best trial  : #{best.number}")
    print(f"Best l_data : {best.value:.4e}")
    print("Best params :")
    for k, v in best.params.items():
        print(f"  {k:<32} = {v}")

    # 輸出 top-5 供手動複核
    print("\nTop-5 trials:")
    top5 = sorted(study.trials, key=lambda t: t.value or float("inf"))[:5]
    for t in top5:
        if t.value is not None:
            print(f"  #{t.number:04d}  l_data={t.value:.4e}  {t.params}")


if __name__ == "__main__":
    main()
