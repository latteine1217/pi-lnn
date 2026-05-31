"""scripts/sweep_al_re10000.py — Augmented Lagrangian (continuity) 參數優化。

What: 以 Optuna TPE 對當前 stable 架構（EXP-245_a: B3 + GradNorm + AL + SOAP +
      ScheduleFree, LES_T50 sensor）的 **AL continuity 參數**做專注掃描。
Why:  AL 的 4 個 meta 參數從未系統優化過。GradNorm 已自動調適 task weight、
      ScheduleFree 自適應 lr，故剩下真正「自由」的 continuity-enforcement 旋鈕就是 AL。

與舊 scripts/sweep_re10000.py 的差異（為何另開檔，不覆蓋）:
  - 舊腳本搜 lr/weight_decay/physics_loss_weight/continuity_weight，是 pre-GradNorm
    (EXP-043) 架構；其 objective 門檻 (L_NS=2.0, L_CONT=3.0) 對當前 normalized loss
    量級（l_ns≈1.6e-3, l_cont≈6e-4）完全失效。保留舊 study 的 SQLite/docs 引用不動。
  - 本腳本 base = 當前 stable EXP-245_a；wandb 改為 optional（r740 離線預設關閉）。

搜尋空間（AL-focused, 4 params）:
  - al_rho            log [0.02, 1.0]   ← 最高槓桿：同時控制 ρ/2·C² 罰 + λ 累積速度
  - al_ema_momentum   [0.1, 0.9]        ← C 的 EMA 平滑（高=慢反應）
  - al_update_freq    {50,100,200,500}  ← dual update 頻率
  - al_start_step     {0,1000,2000}     ← 延後 dual update 到場有基本結構後
  剔除 al_lambda_clip：實測 λ_cont 收斂 ≈0.40 << clip=10，永不 binding（零槓桿）。

Objective（engineering-transferable，無 DNS leakage）:
  obj = l_data_tail · (1 + 0.5·relu(l_ns/L_NS_THR − 1) + 0.5·relu(l_cont/L_CONT_THR − 1))
  - 只用 sensor MSE (l_data) + physics residual (l_ns, l_cont)，符合 ENGINEERING_VISION。
  - 門檻按當前健康 run (EXP-273 tail) 2× 上緣校準：L_NS_THR=5e-3, L_CONT_THR=2e-3。
  - 單側罰：l_cont < 門檻時不獎勵更低（避免 Optuna 學到「過度強約束 → 過平滑場」）。
  - ⚠️ obj 不含 DNS KE（避免工程不可遷移）。top-3 必須用 evaluate_deeponet_cfc.py
    離線對照 DNS KE 驗證，且跑 multi-seed 排除 placement overfit 後才能採用。

執行（lab-server r740 via slurm；proxy budget 預設 8000 steps）:
    uv run python scripts/sweep_al_re10000.py --trials 30 --iterations 8000 --device cuda

本地 smoke（不送 GPU）:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/sweep_al_re10000.py \
        --trials 1 --iterations 30 --device mps --study-name smoke --no-resume
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import optuna

from picon_kolmogorov import DEFAULT_PICON_ARGS, load_picon_config, train_picon_kolmogorov

# ── 常數 ──────────────────────────────────────────────────────────────────────
BASE_CONFIG = Path(__file__).parent.parent / "configs" / "exp_245_b3_les_T50.toml"
OPTUNA_DB_DIR = Path("artifacts/sweep_al")
METRIC_TAIL_STEPS = 500          # 尾段平均步數（降雜訊）

# Objective 門檻（按 EXP-273 tail 健康量級 2× 上緣校準）
L_NS_THRESHOLD = 5.0e-3
L_CONT_THRESHOLD = 2.0e-3
PENALTY_COEFF = 0.5              # 物理違反的相對放大係數


# ── Objective ─────────────────────────────────────────────────────────────────
def make_objective(base_cfg: dict, device: str, iterations: int, use_wandb: bool):
    def objective(trial: optuna.Trial) -> float:
        # ── AL 超參數建議 ───────────────────────────────────────────────────
        al_rho = trial.suggest_float("al_rho", 0.02, 1.0, log=True)
        al_ema = trial.suggest_float("al_ema_momentum", 0.1, 0.9)
        al_freq = trial.suggest_categorical("al_update_freq", [50, 100, 200, 500])
        al_start = trial.suggest_categorical("al_start_step", [0, 1000, 2000])

        cfg = {**base_cfg}
        cfg.update({
            "al_rho": float(al_rho),
            "al_ema_momentum": float(al_ema),
            "al_update_freq": int(al_freq),
            "al_start_step": int(al_start),
            "iterations": int(iterations),
            "device": device,
            "artifacts_dir": f"artifacts/sweep_al/trial_{trial.number:04d}",
            "checkpoint_period": int(iterations),   # 只存 final，避免 sweep 撐爆磁碟
            "seed": 42,                              # 固定，隔離 AL 效應
        })

        run = None
        if use_wandb:
            import wandb
            run = wandb.init(project="pi-con-al-sweep", name=f"trial_{trial.number:04d}",
                             config=trial.params, reinit=True)

        tail_data: list[float] = []
        tail_ns: list[float] = []
        tail_cont: list[float] = []

        def log_fn(step: int, metrics: dict[str, float]) -> None:
            if use_wandb and run is not None:
                run.log(metrics, step=step)
            if step > iterations - METRIC_TAIL_STEPS:
                tail_data.append(metrics["l_data"])
                tail_ns.append(metrics.get("l_ns", metrics.get("l_physics", 0.0)))
                tail_cont.append(metrics.get("l_cont", 0.0))
            # MedianPruner：每 100 步回報 l_data
            if step % 100 == 0:
                trial.report(metrics["l_data"], step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        try:
            train_picon_kolmogorov(cfg, log_fn=log_fn)
        except optuna.TrialPruned:
            if run is not None:
                run.finish(exit_code=1)
            raise

        # ── Objective（單側物理罰，scale-relative）──────────────────────────
        def _mean(xs: list[float]) -> float:
            return float(sum(xs) / len(xs)) if xs else float("inf")

        l_data_mean = _mean(tail_data)
        l_ns_mean = _mean(tail_ns)
        l_cont_mean = _mean(tail_cont)
        pen_ns = PENALTY_COEFF * max(0.0, l_ns_mean / L_NS_THRESHOLD - 1.0)
        pen_cont = PENALTY_COEFF * max(0.0, l_cont_mean / L_CONT_THRESHOLD - 1.0)
        obj = l_data_mean * (1.0 + pen_ns + pen_cont)

        trial.set_user_attr("l_data_tail", l_data_mean)
        trial.set_user_attr("l_ns_tail", l_ns_mean)
        trial.set_user_attr("l_cont_tail", l_cont_mean)
        if run is not None:
            run.log({"final_objective": obj, "l_data_tail": l_data_mean,
                     "l_ns_tail": l_ns_mean, "l_cont_tail": l_cont_mean})
            run.finish()
        return obj

    return objective


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AL continuity 參數優化（Optuna TPE）")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--iterations", type=int, default=8000, help="每 trial proxy budget")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=None)
    p.add_argument("--study-name", default="al-re10000-v1")
    p.add_argument("--no-resume", action="store_true", help="不續跑、重建 study")
    p.add_argument("--wandb", action="store_true", help="啟用 W&B（r740 離線時勿開）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = dict(DEFAULT_PICON_ARGS)
    base_cfg.update(load_picon_config(BASE_CONFIG))
    device = args.device or base_cfg.get("device", "cuda")

    OPTUNA_DB_DIR.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{OPTUNA_DB_DIR}/optuna_al.db"

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=1000, interval_steps=100)
    study = optuna.create_study(
        study_name=args.study_name, storage=storage, direction="minimize",
        sampler=sampler, pruner=pruner, load_if_exists=not args.no_resume,
    )

    objective = make_objective(base_cfg, device, args.iterations, args.wandb)

    print("=== AL continuity Hyperparameter Sweep ===")
    print(f"study      : {args.study_name}  storage={storage}")
    print(f"trials     : {args.trials}  proxy_iters={args.iterations}  device={device}")
    print(f"objective  : l_data_tail·(1 + {PENALTY_COEFF}·relu(l_ns/{L_NS_THRESHOLD:.0e}-1)"
          f" + {PENALTY_COEFF}·relu(l_cont/{L_CONT_THRESHOLD:.0e}-1))  [no DNS]")
    print(f"search     : al_rho∈[0.02,1.0]log  al_ema∈[0.1,0.9]"
          f"  al_update_freq∈{{50,100,200,500}}  al_start_step∈{{0,1000,2000}}")
    print()

    study.optimize(objective, n_trials=args.trials, catch=(Exception,))

    print("\n=== Sweep 完成 ===")
    best = study.best_trial
    print(f"Best trial #{best.number}  objective={best.value:.4e}")
    print(f"  l_data={best.user_attrs.get('l_data_tail')}  "
          f"l_ns={best.user_attrs.get('l_ns_tail')}  l_cont={best.user_attrs.get('l_cont_tail')}")
    print("  params:")
    for k, v in best.params.items():
        print(f"    {k:<20} = {v}")
    print("\nTop-5（需後續以 full 20k + DNS KE 離線複驗 + multi-seed 排除 placement overfit）:")
    top5 = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value)[:5]
    for t in top5:
        print(f"  #{t.number:04d}  obj={t.value:.4e}  {t.params}")


if __name__ == "__main__":
    main()
