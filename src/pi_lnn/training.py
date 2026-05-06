"""Pi-LNN training loop and CLI entry point.

A1 boundary: train_lnn_kolmogorov is moved verbatim. Decomposing it is the
deferred A2 phase (see docs/superpowers/specs/2026-04-26-pi-lnn-package-refactor-design.md).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from pi_lnn.causal import causal_weighted_residual_loss
from pi_lnn.config import DEFAULT_LNN_ARGS, load_lnn_config
from pi_lnn.losses import (
    AugmentedLagrangianMultiplier,
    GradNormWeights,
    _gradnorm_step,
    observed_channel_prediction,
)
from pi_lnn.operator import (
    LiquidOperator,
    create_lnn_model,
    make_lnn_model_fn,
    make_lnn_model_fn_uvp,
)
from pi_lnn.physics import (
    _rar_update_pool,
    physics_points_at_step,
    physics_weight_at_step,
    pressure_poisson_residual,
    unsteady_ns_residuals,
)
from pi_lnn.runtime import _grad, configure_torch_runtime, count_parameters, write_json


def train_lnn_kolmogorov(
    args: dict[str, Any],
    log_fn: Callable[[int, dict[str, float]], None] | None = None,
) -> None:
    """What: 核心 Pi-LNN 訓練迴圈。

    Args:
        args: 訓練設定字典（見 DEFAULT_LNN_ARGS）。
        log_fn: 可選回呼，每個訓練 step 結束後以 (step, metrics_dict) 呼叫。
                metrics_dict 包含 l_data / l_physics / l_ns / l_cont / w_phys / t_max。
                Why: 保持 core module 不依賴外部日誌框架（W&B、TensorBoard 等），
                     由呼叫方（如 sweep 腳本）注入觀測邏輯。
    """
    from kolmogorov_dataset import KolmogorovDataset

    device = configure_torch_runtime(args["device"])
    torch.manual_seed(args["seed"])
    rng = np.random.default_rng(args["seed"])

    artifacts_dir = Path(args["artifacts_dir"])
    checkpoints_dir = artifacts_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    _dataset_type = args.get("dataset_type", "kolmogorov")
    if _dataset_type == "cylinder":
        from cylinder_dataset import CylinderDataset
        datasets = [
            CylinderDataset(
                sensor_json=args["sensor_jsons"][i],
                sensor_npz=args["sensor_npzs"][i],
                arrow_shard=args["arrow_shards"][i],
                re_value=float(args["re_values"][i]),
                observed_channel_names=tuple(args["observed_sensor_channels"]),
                train_ratio=0.8,
                seed=args["seed"],
                sensor_subsample=int(args.get("sensor_subsample", 1)),
            )
            for i in range(len(args["re_values"]))
        ]
    else:
        datasets = [
            KolmogorovDataset(
                sensor_json=args["sensor_jsons"][i],
                sensor_npz=args["sensor_npzs"][i],
                dns_path=args["dns_paths"][i],
                re_value=float(args["re_values"][i]),
                observed_channel_names=tuple(args["observed_sensor_channels"]),
                train_ratio=0.8,
                seed=args["seed"],
            )
            for i in range(len(args["re_values"]))
        ]
    num_re = len(datasets)

    sensor_vals_list = [
        torch.tensor(ds.sensor_vals.transpose(1, 0, 2), dtype=torch.float32, device=device)
        for ds in datasets
    ]
    sensor_pos_list = [
        torch.tensor(ds.sensor_pos, dtype=torch.float32, device=device)
        for ds in datasets
    ]
    sensor_time_list = [
        torch.tensor(ds.sensor_time, dtype=torch.float32, device=device)
        for ds in datasets
    ]
    observed_mean_list = [
        torch.tensor(ds.observed_channel_mean, dtype=torch.float32, device=device)
        for ds in datasets
    ]
    observed_std_list = [
        torch.tensor(ds.observed_channel_std, dtype=torch.float32, device=device)
        for ds in datasets
    ]

    # T_total mismatch 檢查：避免 config 與 dataset 不同步導致 temporal_phase_anchor
    # 相位 alias（silent 行為錯，無 runtime error）。
    if bool(args.get("use_temporal_anchor", False)):
        _T_cfg = float(args.get("T_total", 5.0))
        _T_data = float(max(float(ds.sensor_time[-1]) for ds in datasets))
        if abs(_T_cfg - _T_data) / max(_T_data, 1e-8) > 0.05:
            print(
                f"[WARN] T_total config={_T_cfg:.3f} 與 dataset 實測"
                f" max(sensor_time[-1])={_T_data:.3f} 差距 >5%；"
                f"temporal_phase_anchor 可能相位 alias，"
                f"建議將 T_total 改為 {_T_data:.3f} 或調整 sensor_subsample。"
            )

    net = create_lnn_model(args).to(device)

    # ── Body-distance feature（cylinder boundary layer 學習）──────────
    # use_hard_body_bc=True 時，model output 會套 (φ/scale).clamp(0,1) gate 強制 body 內 u=v=0。
    # 對 cylinder dataset 從 _detect_body 預計算的 SDF grid 用 bilinear interp 取值；
    # 對 kolmogorov 退化為常數 1.0（無 body）。
    _use_hard_body_bc = bool(args.get("use_hard_body_bc", False))
    if _use_hard_body_bc and not getattr(net, "use_hard_body_bc", False):
        raise ValueError(
            "config use_hard_body_bc=True 但 net.use_hard_body_bc=False；"
            "可能 model 用舊 ckpt resume 但 ckpt 是 hard BC 關閉時訓的。"
        )

    def _make_body_distance_fn(ds):
        """建立 dataset-specific 的 differentiable body distance lookup（torch ops only）.

        Why differentiable: hard BC 是 u = (φ/scale)·NN，autograd 必須能算
        ∂φ/∂xy 才能正確算 ∂u/∂x = ∂φ/∂x·NN + φ·∂NN/∂x。numpy lookup 會 detach 切斷
        graph (cylinder_007/008 的 detach bug 已驗證會讓 div_L2 翻倍)。
        """
        if not hasattr(ds, "query_body_distance_torch"):
            raise AttributeError(
                f"dataset {type(ds).__name__} 不支援 differentiable distance；"
                "use_hard_body_bc=True 需要 cylinder dataset (有 SDF grid)"
            )
        def fn(xy_tensor: torch.Tensor) -> torch.Tensor:
            # 不 detach！讓 autograd 追蹤 ∂φ/∂xy
            return ds.query_body_distance_torch(xy_tensor)
        return fn

    body_distance_fns: list = []
    if _use_hard_body_bc:
        for ds in datasets:
            body_distance_fns.append(_make_body_distance_fn(ds))
        # 注入 dataset-specific bc_distance_scale 到 model gate
        # 多 dataset 取最大 scale（保守）
        _bc_scale = max(float(getattr(ds, "bc_distance_scale", 1.0)) for ds in datasets)
        net.set_body_bc_scale(_bc_scale)
        print(f"  hard_body_bc: enabled for {len(datasets)} dataset(s), gate scale={_bc_scale:.4f}")

    # 注入 physics-path denormalization：把 query_decoder normalized output 還原成物理單位
    # (u, v) 從 dataset 的 observed_channel_mean/std 取；p 無監督，保持 (mean=0, std=1)。
    # Why: NS residual 用物理 ν=1/Re，若 u, v 是 normalized 量級 → 黏性項 ν∇²u 被壓 std⁻¹
    #      倍，physics gradient signal 進入 NG 時幾乎消失（l_phys 看似小但物理發散度大）。
    # 兩個觀測通道為 ("u", "v") 才能正確 map（其他通道組合需重新計算 mean_p）。
    #
    # Config flag `use_physics_denormalization`（2026-05-07，取代 PINN_DISABLE_PHYS_DENORM env var）：
    #   True  → 注入 dataset stats，physics path 使用物理量級（cylinder 主線）。
    #   False → buffer 留在 (mean=0, std=1) ≡ identity，physics path 用 normalized u/v
    #            （EXP-064 baseline 路徑；Kolmogorov 主線預設）。
    #   後向相容性：env var `PINN_DISABLE_PHYS_DENORM=1` 仍可作為臨時 override 強制關閉。
    _env_disable = os.environ.get(
        "PINN_DISABLE_PHYS_DENORM", ""
    ).lower() in ("1", "true", "yes")
    _enable_denorm = bool(args.get("use_physics_denormalization", False)) and not _env_disable
    _obs_names = tuple(args.get("observed_sensor_channels", ("u", "v")))
    if _enable_denorm and _obs_names == ("u", "v") and num_re == 1:
        _u_idx = _obs_names.index("u")
        _v_idx = _obs_names.index("v")
        _ch_mean = datasets[0].observed_channel_mean
        _ch_std  = datasets[0].observed_channel_std
        _mean_uvp = torch.tensor(
            [float(_ch_mean[_u_idx]), float(_ch_mean[_v_idx]), 0.0],
            dtype=torch.float32, device=device,
        )
        _std_uvp = torch.tensor(
            [float(_ch_std[_u_idx]),  float(_ch_std[_v_idx]),  1.0],
            dtype=torch.float32, device=device,
        )
        net.set_physics_normalization(_mean_uvp, _std_uvp)
        print(f"  physics_output_mean: {_mean_uvp.tolist()}", flush=True)
        print(f"  physics_output_std : {_std_uvp.tolist()}", flush=True)
    elif _enable_denorm:
        # 觸發條件不滿足（多 Re 或非 (u,v) 觀測），明確 WARN
        print(
            f"  [WARN] use_physics_denormalization=True 但條件不滿足（obs={_obs_names}, num_re={num_re}）；"
            f"physics path 將留在 identity 路徑。",
            flush=True,
        )
    elif _env_disable:
        print(
            "  [PINN_DISABLE_PHYS_DENORM=1] env var override：強制關閉 physics denorm，"
            "對齊 EXP-064 baseline 路徑。",
            flush=True,
        )
    else:
        # use_physics_denormalization=False（預設），EXP-064 baseline 路徑
        print(
            "  physics denorm: identity（use_physics_denormalization=False）— "
            "NS residual 用 normalized u/v，等同 EXP-064 baseline 路徑。",
            flush=True,
        )

    print("=== Configuration ===")
    print(f"trainable_parameters: {count_parameters(net)}")

    # --- Optimizer + Schedule-Free 組合 ---
    # lr_schedule 控制 base optimizer 種類與 LR 衰減策略：
    #   "soap"        → SOAP（二階前置條件），不搭配 LR scheduler
    #   "step"        → AdamW + StepLR
    #   "cosine"      → AdamW + CosineAnnealingLR
    #   "none"        → AdamW，常數 LR
    #   "schedulefree"→ 舊版相容：等同 lr_schedule="none" + use_schedule_free=true
    #
    # use_schedule_free 控制是否套用 Polyak averaging：
    #   AdamW + SF  → 使用 fused AdamWScheduleFree（效率最佳，支援 warmup）
    #   SOAP  + SF  → 使用 ScheduleFreeWrapper(SOAP)
    #   任何  + no SF → 直接使用 base optimizer
    #
    # Why fused vs wrapper: ScheduleFreeWrapper 在 step() 前先寫入 state['z']，
    # 導致 AdamW._init_group 誤判狀態已存在而跳過 exp_avg 初始化 → KeyError。
    # fused AdamWScheduleFree 無此問題。

    use_schedulefree = bool(args.get("use_schedule_free", False)) or args["lr_schedule"] == "schedulefree"
    is_lbfgs = args["lr_schedule"] == "lbfgs"
    is_ng = args["lr_schedule"] == "ng"

    if args["lr_schedule"] == "soap":
        import sys
        # SOAP/ 在 project root，training.py 在 src/pi_lnn/，故需要三層 .parent
        # 之前 .parent.parent 是 src/SOAP（不存在），靠 cwd 在 project root 才能 import 到
        _soap_dir = str(Path(__file__).parent.parent.parent / "SOAP")
        if _soap_dir not in sys.path:
            sys.path.insert(0, _soap_dir)
        from soap import SOAP as SOAPOptimizer
        _soap_betas = tuple(args.get("soap_betas", [0.95, 0.95]))
        base_optimizer = SOAPOptimizer(
            net.parameters(),
            lr=args["learning_rate"],
            betas=_soap_betas,
            weight_decay=args["weight_decay"],
            precondition_frequency=int(args.get("soap_precondition_frequency", 10)),
        )
        if use_schedulefree:
            import schedulefree
            optimizer = schedulefree.ScheduleFreeWrapper(base_optimizer, momentum=0.9)
        else:
            optimizer = base_optimizer
        # warmup → step decay の順に SequentialLR で繋ぐ；片方だけも可
        # ScheduleFreeWrapper は torch.optim.Optimizer を継承しないため、
        # LR scheduler は必ず base_optimizer に紐付ける。
        # ScheduleFreeWrapper は base_optimizer.param_groups['lr'] を参照するので
        # scheduler が base_optimizer の LR を更新すれば wrapper にも反映される。
        _warmup_steps = int(args.get("lr_warmup_steps", 0))
        _use_decay = bool(args.get("soap_use_step_decay", False))
        _sched_target = base_optimizer  # scheduler は常に base_optimizer に紐付ける
        if _warmup_steps > 0 and _use_decay:
            _warmup_sched = torch.optim.lr_scheduler.LinearLR(
                _sched_target, start_factor=0.01, end_factor=1.0, total_iters=_warmup_steps
            )
            _main_sched = torch.optim.lr_scheduler.StepLR(
                _sched_target, step_size=int(args["lr_decay_steps"]), gamma=float(args["lr_decay_gamma"])
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                _sched_target, schedulers=[_warmup_sched, _main_sched], milestones=[_warmup_steps]
            )
        elif _warmup_steps > 0:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                _sched_target, start_factor=0.01, end_factor=1.0, total_iters=_warmup_steps
            )
        elif _use_decay:
            scheduler = torch.optim.lr_scheduler.StepLR(
                _sched_target, step_size=int(args["lr_decay_steps"]), gamma=float(args["lr_decay_gamma"])
            )
        else:
            scheduler = None

    elif is_lbfgs:
        optimizer = torch.optim.LBFGS(
            net.parameters(),
            lr=float(args.get("learning_rate", 1.0)),
            max_iter=int(args.get("lbfgs_max_iter", 20)),
            history_size=int(args.get("lbfgs_history_size", 10)),
            line_search_fn="strong_wolfe",
        )
        scheduler = None

    elif is_ng:
        # Natural Gradient (Gauss-Newton) — kernel trick path。
        # Why: pi-lnn P~10⁵, N~200 → 必走 J^T (JJ^T + λI)^{-1} r。
        from pi_lnn.optimizers import NaturalGradientOptimizer
        optimizer = NaturalGradientOptimizer(
            net.parameters(),
            lr=float(args.get("learning_rate", 1.0)),
            damping=float(args.get("ng_damping", 1.0e-6)),
            damping_strategy=str(args.get("ng_damping_strategy", "fixed")),
            use_jacobi_scaling=bool(args.get("ng_jacobi_scaling", True)),
            solver_device=str(args.get("ng_solver_device", "cpu")),
            max_residuals=int(args.get("ng_max_residuals", 2000)),
            line_search=str(args.get("ng_line_search", "none")),
            ls_max_trials=int(args.get("ng_ls_max_trials", 5)),
            ls_alpha_init=float(args.get("ng_ls_alpha_init", 1.0)),
            ls_alpha_decay=float(args.get("ng_ls_alpha_decay", 0.5)),
            ls_armijo_c1=float(args.get("ng_ls_armijo_c1", 1.0e-4)),
        )
        if bool(args.get("ng_use_spring", False)):
            optimizer.set_spring(
                enabled=True,
                momentum=float(args.get("ng_spring_momentum", 0.9)),
            )
            print(f"  NG: SPRING momentum enabled (μ={args.get('ng_spring_momentum', 0.9)})")
        if str(args.get("ng_line_search", "none")) != "none":
            print(f"  NG: line_search={args['ng_line_search']}, "
                  f"max_trials={args.get('ng_ls_max_trials', 5)}")
        scheduler = None

    elif use_schedulefree:
        # AdamW + Schedule-Free：使用 fused 實作，支援 warmup，無 wrapper 相容性問題。
        import schedulefree
        optimizer = schedulefree.AdamWScheduleFree(
            net.parameters(),
            lr=args["learning_rate"],
            warmup_steps=int(args.get("lr_warmup_steps", 300)),
            weight_decay=args["weight_decay"],
        )
        scheduler = None

    else:
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=args["learning_rate"],
            weight_decay=args["weight_decay"],
        )
        if args["lr_schedule"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args["iterations"],
                eta_min=args["min_learning_rate"],
            )
        elif args["lr_schedule"] == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(args["lr_decay_steps"]),
                gamma=float(args["lr_decay_gamma"]),
            )
        else:
            scheduler = None

    # GradNorm setup（必須在 resume 之前，因為 resume 會嘗試 load gn_weights state）
    use_gradnorm = bool(args.get("use_gradnorm", False))
    gn_weights: GradNormWeights | None = None
    gn_ref_params: list[torch.Tensor] = []
    gn_update_freq = int(args.get("gradnorm_update_freq", 200))
    _gn_init_w = args.get("gradnorm_init_weights", [1.0, 0.01, 0.01, 0.01, 0.1])
    # task_names: 顯式 (config 提供 gradnorm_tasks 非空) 或由長度推斷
    _gn_tasks_cfg = list(args.get("gradnorm_tasks", []) or [])
    _gn_task_names: list[str] | None = _gn_tasks_cfg if _gn_tasks_cfg else None
    if _gn_task_names is None and len(_gn_init_w) not in (4, 5):
        raise ValueError(
            f"未指定 gradnorm_tasks 時 gradnorm_init_weights 長度必須 4 (data,ns_u,ns_v,cont) 或 "
            f"5 (data,ns_u,ns_v,cont,bc)，收到 {len(_gn_init_w)}: {_gn_init_w}"
        )
    if use_gradnorm:
        gn_weights = GradNormWeights(
            init_weights=_gn_init_w, task_names=_gn_task_names
        ).to(device)
        gn_ref_params = list(net.query_decoder.trunk_out.parameters())
    use_gradnorm_bc = use_gradnorm and gn_weights is not None and "bc" in gn_weights
    use_gradnorm_cont = use_gradnorm and gn_weights is not None and "cont" in gn_weights

    # AL-continuity setup（spec v4 §3 / §4）
    use_al = bool(args.get("use_augmented_lagrangian", False))
    al_cont: AugmentedLagrangianMultiplier | None = None
    al_update_freq = int(args.get("al_update_freq", 100))
    if use_al:
        # Pre-condition asserts（runtime fail-fast；config-load _validate_al_config 也會擋）
        assert args["lr_schedule"] != "ng", "AL incompatible with lr_schedule='ng'"
        assert not (args["lr_schedule"] == "lbfgs" and use_gradnorm), \
            "AL + LBFGS 不支援 use_gradnorm（closure race）"
        assert float(args.get("continuity_weight", 0.0)) == 0.0, \
            "AL active 時 continuity_weight 必須 = 0，否則 cont 被雙重 penalty"
        assert not bool(args.get("use_sensor_physics", False)), \
            "AL v1 不支援 use_sensor_physics（l_cont_total 會變成 sum-of-two-means）"
        if use_gradnorm and gn_weights is not None:
            assert "cont" not in gn_weights, "AL active 時 'cont' 必須從 gradnorm_tasks 移出"
            assert "al" not in gn_weights, \
                "v4 規定 AL term 不進 GradNorm losses 列表 — 'al' 不能出現在 gradnorm_tasks"
        al_cont = AugmentedLagrangianMultiplier(
            init_lambda=float(args.get("al_init_lambda", 0.0)),
            rho=float(args.get("al_rho", 1.0)),
            lambda_clip=float(args.get("al_lambda_clip", 10.0)),
            ema_momentum=float(args.get("al_ema_momentum", 0.5)),
        ).to(device)

    # Resume：從 checkpoint 恢復完整訓練狀態
    start_step = 0
    resume_path = args.get("resume_checkpoint")

    def _fix_ckpt_compat(state_dict: dict) -> dict:
        """相容舊 checkpoint：log_fusion_temperature 由 0-dim 改為 shape (1,)。"""
        key = "query_decoder.log_fusion_temperature"
        if key in state_dict and state_dict[key].dim() == 0:
            state_dict[key] = state_dict[key].unsqueeze(0)
        return state_dict

    def _fix_optimizer_state_compat(opt: torch.optim.Optimizer) -> None:
        """相容舊 checkpoint：將 optimizer state 中殘留的 0-dim tensor unsqueeze 為 (1,)。
        Why: log_fusion_temperature 由 0-dim 改為 (1,) 後，SOAP 的 exp_avg/exp_avg_sq
             仍是舊形狀，導致 broadcast 失敗。"""
        for param_state in opt.state.values():
            for k, v in param_state.items():
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    param_state[k] = v.unsqueeze(0)

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            # 完整狀態格式：model + optimizer + scheduler + step
            net.load_state_dict(_fix_ckpt_compat(ckpt["model_state_dict"]))
            # L-BFGS / NG 與 checkpoint 的 optimizer 類型不同，跳過 optimizer state 載入。
            if not is_lbfgs and not is_ng and "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                _fix_optimizer_state_compat(optimizer)
                # ScheduleFree resume 修補：state_dict() 不含 train_mode flag，
                # 預設載入後是 False（eval mode）。但 ckpt 是訓練中存的 → param.data
                # 是 x_t (train iterate)。第一次 optimizer.train() 會觸發 eval→train
                # swap，對已是 x_t 的 param 再 swap 一次 → 用 y_t 的反算覆蓋 → 爆炸。
                # 修法：直接 set train_mode=True，讓 .train() 不再觸發 swap。
                if use_schedulefree and hasattr(optimizer, "train_mode"):
                    optimizer.train_mode = True
                    print("  [resume] ScheduleFree train_mode=True (skip swap)")
            if not use_schedulefree and scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_step = int(ckpt["step"])
            # GradNorm state（gn_weights 是獨立 nn.Module，不在 net.state_dict 內）
            # Why: 之前 resume 後 GradNorm 重置成 init weights，導致 model + GradNorm
            #      不同步 → step ~1000 後 GradNorm 第一次 update 量級失準 → loss 爆炸。
            if (
                gn_weights is not None
                and "gradnorm_state_dict" in ckpt
                and ckpt["gradnorm_state_dict"] is not None
            ):
                _gn_state = ckpt["gradnorm_state_dict"]
                # 檢查 layout 兼容性：4-task vs 5-task
                _saved_log_w = _gn_state.get("log_weights")
                if _saved_log_w is not None and _saved_log_w.numel() == gn_weights.log_weights.numel():
                    gn_weights.load_state_dict(_gn_state)
                    print(f"  resumed GradNorm weights: {gn_weights.weights.detach().tolist()}")
                else:
                    print(
                        f"  [WARN] GradNorm layout 不相容（saved={_saved_log_w.numel() if _saved_log_w is not None else 'N/A'}"
                        f" vs current={gn_weights.log_weights.numel()}）→ 用 init weights"
                    )
            elif gn_weights is not None and ckpt.get("gradnorm_state_dict") is None:
                print(
                    "  [WARN] checkpoint 不含 GradNorm state（legacy ckpt）→ 用 init weights；"
                    "若訓練 >1000 步可能 GradNorm 重新 calibrate 時 loss 短暫上升"
                )
            # AL state restore：保 lambda_ / ema_C / _initialized；缺則 cold-start
            if al_cont is not None and ckpt.get("al_state_dict") is not None:
                al_cont.load_state_dict(ckpt["al_state_dict"])
                print(
                    f"  resumed AL state: λ={al_cont.lambda_.item():.3e} "
                    f"ema_C={al_cont.ema_C.item():.3e} initialized={bool(al_cont._initialized.item())}"
                )
            elif al_cont is not None:
                print("  [WARN] checkpoint 不含 AL state → 從 init λ/ema_C 重新累積")
        else:
            # 舊格式（只有 model weights）：恢復模型；scheduler 快進（schedulefree 則略過）
            net.load_state_dict(_fix_ckpt_compat(ckpt))
            start_step = int(Path(resume_path).stem.split("_step_")[-1]) if "_step_" in Path(resume_path).stem else 0
            if not use_schedulefree and scheduler is not None:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for _ in range(start_step):
                        scheduler.step()
        print(f"  resumed from: {resume_path} (step {start_step})")

    k_f = float(args["kolmogorov_k_f"])
    A = float(args["kolmogorov_A"])
    domain_length = float(args["domain_length"])
    base_phys_weight = float(args["physics_loss_weight"])
    phys_warmup_steps = int(args["physics_loss_warmup_steps"])
    phys_ramp_steps = int(args["physics_loss_ramp_steps"])
    use_tm = bool(args["time_marching"])
    tm_t_start = float(args["time_marching_start"])
    tm_t_end = float(datasets[0].sensor_time[-1])
    tm_warmup = int(args["time_marching_warmup"] * args["iterations"])

    # Causal weighting setup（PINN: Wang, Sankaran, Perdikaris 2022）
    use_causal = bool(args.get("use_causal_weighting", False))
    causal_eps = float(args.get("causal_eps", 1.0))
    causal_num_bins = int(args.get("causal_num_bins", 16))
    if use_causal:
        if causal_eps < 0.0:
            raise ValueError(f"causal_eps 必須 >= 0，收到 {causal_eps}")
        if causal_num_bins < 2:
            raise ValueError(f"causal_num_bins 必須 >= 2，收到 {causal_num_bins}")
        if float(args.get("t_early_weight", 1.0)) != 1.0:
            print(
                "  [WARN] use_causal_weighting=true 同時 t_early_weight != 1.0，"
                "兩者都會強化早期時間，建議將 t_early_weight 設為 1.0 避免雙重加權",
                flush=True,
            )

    print("=== Training ===", flush=True)
    if use_tm:
        print(f"  time_marching: t [{tm_t_start:.1f} → {tm_t_end:.1f}]  warmup={tm_warmup} steps", flush=True)
    if use_gradnorm and gn_weights is not None:
        _gn_layout = ", ".join(gn_weights.task_names)
        print(f"  GradNorm: momentum={args.get('gradnorm_ema_momentum', 0.5):.2f}  freq={gn_update_freq}  (direct formula + EMA)")
        print(f"  GradNorm init_weights: {_gn_init_w}  (tasks: [{_gn_layout}])")
    if use_al and al_cont is not None:
        print(
            f"  AL-continuity: λ_init={al_cont.lambda_.item():.3f} ρ={al_cont.rho:.3f} "
            f"clip=(0,{al_cont.lambda_clip:.1f}) freq={al_update_freq} ema={al_cont.ema_momentum:.2f}"
        )
    if not use_gradnorm and (phys_warmup_steps > 0 or phys_ramp_steps > 0):
        print(
            "  physics_ramp:"
            f" warmup={phys_warmup_steps} steps,"
            f" ramp={phys_ramp_steps} steps,"
            f" final_weight={base_phys_weight:.4f}"
        )
    if bool(args.get("use_sensor_physics", False)):
        _sp_K = datasets[0].sensor_pos.shape[0]
        _sp_nt = int(args.get("num_sensor_physics_time_samples", 4))
        print(f"  sensor_physics: K={_sp_K} × n_t={_sp_nt} = {_sp_K * _sp_nt} pts/step (cond≈11 for k≤16)", flush=True)
    # Header: 動態組裝
    # - GradNorm active：列出 GradNorm tasks 的 weight（跳過 data，因為永遠 = 1）
    # - AL active：append `lambda_c` 與 `C_ema`（spec §5 logging layout）
    if use_gradnorm and gn_weights is not None:
        _w_cols = " ".join(f"{f'w_{n}':>8}" for n in gn_weights.task_names if n != "data")
        _hdr = f"{'Step':<8} {'L_data':>12} {'L_phys':>12} {_w_cols} {'L_total':>12}"
    else:
        _hdr = f"{'Step':<8} {'L_data':>12} {'L_phys':>12} {'w_phys':>10} {'L_total':>12}"
    if use_al:
        _hdr += f" {'lambda_c':>10} {'C_ema':>10}"
    if use_tm:
        _hdr += "  t_max"
    print(_hdr, flush=True)

    _is_rar = str(args.get("physics_collocation_strategy", "random")) == "rar"
    _rar_pool_np: list[np.ndarray] | None = None
    _rar_update_freq = int(args.get("rar_update_freq", 50))
    _rar_pool_mult   = int(args.get("rar_pool_multiplier", 10))
    _rar_expl_ratio  = float(args.get("rar_exploration_ratio", 0.2))

    warmup_steps = int(args.get("warmup_steps", 0))

    # NG-only：固定 collocation 機制（論文 Helmholtz/Stokes 採固定批次）。
    # ng_resample_freq=0 → 永不重採；>0 → 每 freq 步重採一次。
    _ng_data_cache: list = []
    _ng_phys_cache: list = []
    _ng_sensor_cache: list = []   # sensor_physics 點（only when use_sensor_physics）
    _ng_resample_freq = int(args.get("ng_resample_freq", 50))

    for step in range(start_step + 1, args["iterations"] + 1):
        if use_tm:
            # warmup_steps 期間 t_max 固定在 tm_t_start，warm-up 結束後才開始展開。
            effective_step = max(0, step - warmup_steps)
            progress = min(effective_step / max(tm_warmup, 1), 1.0)
            t_max: float | None = tm_t_start + (tm_t_end - tm_t_start) * progress
        else:
            t_max = None

        if use_schedulefree:
            optimizer.train()
        net.train()

        # ── L-BFGS path ──────────────────────────────────────────────────────
        if is_lbfgs:
            # 採樣一次：closure 被 line-search 多次呼叫時重用同一批資料。
            _phys_weight = physics_weight_at_step(
                step=step,
                final_weight=base_phys_weight,
                warmup_steps=phys_warmup_steps,
                ramp_steps=phys_ramp_steps,
            )
            _n_phys_end   = int(args["num_physics_points"])
            _n_phys_start = int(args.get("num_physics_points_start", 0)) or _n_phys_end
            _n_phys_wu    = int(args.get("num_physics_points_warmup_steps", 0))
            _n_phys_ramp  = int(args.get("num_physics_points_ramp_steps", 0))
            _n_phys = physics_points_at_step(step, _n_phys_start, _n_phys_end, _n_phys_ramp, _n_phys_wu)
            _phys_gate = _phys_weight > 0.0 and _n_phys_end > 0
            _phys_strategy = str(args.get("physics_collocation_strategy", "random"))
            _phys_normalize = bool(args.get("physics_residual_normalize", False))
            _poisson_weight = float(args.get("poisson_loss_weight", 0.0))

            _fixed_data: list = []
            for i, ds in enumerate(datasets):
                n_q = int(args.get("num_query_points", 0)) or ds.sensor_pos.shape[0]
                xy_np, t_np, c_np, ref_np = ds.sample_sensor_batch(rng, n=n_q, t_max=t_max)
                _fixed_data.append((
                    torch.tensor(xy_np, dtype=torch.float32, device=device),
                    torch.tensor(t_np, device=device),
                    torch.tensor(c_np, dtype=torch.long, device=device),
                    torch.tensor(ref_np, device=device),
                ))

            _fixed_phys: list = []
            if _phys_gate:
                for i, ds in enumerate(datasets):
                    xy_np, t_np = ds.sample_physics_points(
                        rng, n=_n_phys, t_max=t_max, strategy=_phys_strategy
                    )
                    _fixed_phys.append(torch.tensor(
                        np.concatenate([xy_np, t_np[:, None]], axis=1),
                        dtype=torch.float32, device=device, requires_grad=True,
                    ))

            _lbfgs_info: dict = {}

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                net.train()
                _ld = torch.zeros(1, device=device)
                # 同 first-order path：跨 data → physics 共用 encoder 輸出。
                _h_cache_lbfgs: list[tuple[torch.Tensor, torch.Tensor]] = []
                for _i, _ds in enumerate(datasets):
                    _xy, _tq, _c, _ref = _fixed_data[_i]
                    _h, _st = net.encode(
                        sensor_vals_list[_i], sensor_pos_list[_i], _ds.re_norm, sensor_time_list[_i]
                    )
                    _h_cache_lbfgs.append((_h, _st))
                    _bd_d = body_distance_fns[_i](_xy) if _use_hard_body_bc else None
                    _pred = observed_channel_prediction(
                        net=net, xy=_xy, t_q=_tq, c_obs=_c,
                        observed_channel_names=_ds.observed_channel_names,
                        observed_channel_mean=observed_mean_list[_i],
                        observed_channel_std=observed_std_list[_i],
                        h_states=_h, s_time=_st, sensor_pos=sensor_pos_list[_i],
                        body_distance=_bd_d,
                    )
                    _ld = _ld + ((_pred - _ref) ** 2).mean()
                _ld = _ld / num_re

                _lp = torch.zeros(1, device=device)
                _lcont = torch.zeros(1, device=device)
                if _fixed_phys:
                    net.eval()
                    for _i, _ds in enumerate(datasets):
                        _xyt = _fixed_phys[_i]
                        _h_p, _st_p = _h_cache_lbfgs[_i]
                        _uvpfn = make_lnn_model_fn_uvp(
                            net, sensor_vals_list[_i], sensor_pos_list[_i],
                            re_norm=_ds.re_norm, sensor_time=sensor_time_list[_i], device=device,
                            h_states=_h_p, s_time=_st_p,
                            body_distance_fn=body_distance_fns[_i] if _use_hard_body_bc else None,
                        )
                        _mu, _mv, _co = unsteady_ns_residuals(
                            _uvpfn, _xyt,
                            re=_ds.re_value, k_f=k_f, A=A, domain_length=domain_length,
                            Lx=_ds.Lx, Ly=_ds.Ly,
                        )
                        if _phys_normalize:
                            def _nr(r: torch.Tensor) -> torch.Tensor:
                                return r / r.detach().std().clamp(min=1e-8)
                            _mu, _mv, _co = _nr(_mu), _nr(_mv), _nr(_co)
                        _lp   = _lp   + torch.mean(_mu ** 2) + torch.mean(_mv ** 2)
                        _lcont = _lcont + torch.mean(_co ** 2)
                    net.train()
                    _lp   = _lp   / num_re
                    _lcont = _lcont / num_re

                # AL active in LBFGS：closure 內僅讀 λ 算 al_term，不呼叫 update()。
                # cache _lcont.detach() 給 closure 外的 dual update 使用（spec v4 §4 LBFGS section）。
                if al_cont is not None:
                    _al_term = al_cont.loss_term(_lcont)
                    _lt = args["data_loss_weight"] * _ld + _phys_weight * _lp + _al_term
                    _lbfgs_info["l_cont_for_al"] = _lcont.detach().clone()
                else:
                    _lt = args["data_loss_weight"] * _ld + _phys_weight * (_lp + args["continuity_weight"] * _lcont)
                _lt.backward()
                _lbfgs_info["l_data"]   = _ld.item()
                _lbfgs_info["l_phys"]   = (_lp + _lcont).item()
                _lbfgs_info["l_cont"]   = _lcont.item()
                _lbfgs_info["l_total"]  = _lt.item()
                return _lt

            optimizer.step(closure)

            # AL dual update — 嚴格在 step(closure) 之後，用 closure 最後一次的 l_cont
            if al_cont is not None and step > 0 and step % al_update_freq == 0:
                _last_lcont = _lbfgs_info.get("l_cont_for_al")
                if _last_lcont is not None:
                    al_cont.update(_last_lcont)

            l_data    = torch.tensor([_lbfgs_info.get("l_data",  0.0)], device=device)
            l_physics = torch.tensor([_lbfgs_info.get("l_phys",  0.0)], device=device)
            l_total   = torch.tensor([_lbfgs_info.get("l_total", 0.0)], device=device)
            l_ns_u_total = torch.zeros(1, device=device)
            l_ns_v_total = torch.zeros(1, device=device)
            l_ns_total   = torch.zeros(1, device=device)
            l_cont_total = torch.tensor([_lbfgs_info.get("l_cont", 0.0)], device=device)
            l_bc_total   = torch.zeros(1, device=device)  # LBFGS path 不計 BC
            phys_weight  = _phys_weight

        # ── NG path ──────────────────────────────────────────────────────────
        # 結構與 LBFGS 類似：固定批次 + closure；差異在 closure 回傳殘差向量
        # 而非 scalar，由 NG 內部用 J^T(JJ^T + λI)^{-1} r 求步長。
        elif is_ng:
            _phys_weight = physics_weight_at_step(
                step=step,
                final_weight=base_phys_weight,
                warmup_steps=phys_warmup_steps,
                ramp_steps=phys_ramp_steps,
            )
            _n_phys_end   = int(args["num_physics_points"])
            _n_phys_start = int(args.get("num_physics_points_start", 0)) or _n_phys_end
            _n_phys_wu    = int(args.get("num_physics_points_warmup_steps", 0))
            _n_phys_ramp  = int(args.get("num_physics_points_ramp_steps", 0))
            _n_phys = physics_points_at_step(step, _n_phys_start, _n_phys_end, _n_phys_ramp, _n_phys_wu)
            _phys_gate = _phys_weight > 0.0 and _n_phys_end > 0
            _phys_strategy = str(args.get("physics_collocation_strategy", "random"))
            _phys_normalize = bool(args.get("physics_residual_normalize", False))
            _cont_w = float(args["continuity_weight"])
            _data_w = float(args["data_loss_weight"])
            _scale_re = (1.0 / num_re) ** 0.5

            # 固定 collocation：只在 step==1 或 step % freq == 1 時重採；
            # 否則重用前次採樣的批次（論文 NG 必備條件）。
            _need_resample = (
                not _ng_data_cache
                or (_ng_resample_freq > 0 and (step - 1) % _ng_resample_freq == 0)
            )
            if _need_resample:
                _ng_data_cache.clear()
                for i, ds in enumerate(datasets):
                    n_q = int(args.get("num_query_points", 0)) or ds.sensor_pos.shape[0]
                    xy_np, t_np, c_np, ref_np = ds.sample_sensor_batch(rng, n=n_q, t_max=t_max)
                    _ng_data_cache.append((
                        torch.tensor(xy_np, dtype=torch.float32, device=device),
                        torch.tensor(t_np, device=device),
                        torch.tensor(c_np, dtype=torch.long, device=device),
                        torch.tensor(ref_np, device=device),
                    ))

                _ng_phys_cache.clear()
                if _phys_gate:
                    for i, ds in enumerate(datasets):
                        xy_np, t_np = ds.sample_physics_points(
                            rng, n=_n_phys, t_max=t_max, strategy=_phys_strategy
                        )
                        _ng_phys_cache.append(torch.tensor(
                            np.concatenate([xy_np, t_np[:, None]], axis=1),
                            dtype=torch.float32, device=device, requires_grad=True,
                        ))

                # sensor_physics：在 sensor 位置額外計算 continuity 殘差（一階導，穩定）。
                # 與 first-order path 同邏輯，但搬到 NG path 並走 cache 機制。
                _ng_sensor_cache.clear()
                _use_sp = bool(args.get("use_sensor_physics", False))
                _sp_start = int(args.get("sensor_physics_start_step", 0))
                _sp_n_t = int(args.get("num_sensor_physics_time_samples", 4))
                if _use_sp and step >= _sp_start and _phys_gate:
                    for i, ds in enumerate(datasets):
                        _t_all = ds.sensor_time
                        _t_avail = _t_all[_t_all <= float(t_max if t_max is not None else _t_all[-1])]
                        if len(_t_avail) == 0:
                            _ng_sensor_cache.append(None)
                            continue
                        _n_t = min(_sp_n_t, len(_t_avail))
                        _t_idx = rng.choice(len(_t_avail), size=_n_t, replace=False)
                        _t_sp = _t_avail[_t_idx]
                        _xy_sp = np.repeat(ds.sensor_pos, _n_t, axis=0)
                        _t_sp_rep = np.tile(_t_sp, ds.sensor_pos.shape[0])[:, None]
                        _ng_sensor_cache.append(torch.tensor(
                            np.concatenate([_xy_sp, _t_sp_rep], axis=1).astype(np.float32),
                            device=device, requires_grad=True,
                        ))
                else:
                    # Pad 與 datasets 等長以便 closure 中 zip 對齊
                    _ng_sensor_cache.extend([None] * len(datasets))

            _fixed_data = _ng_data_cache  # alias，保留下方 closure 變數名
            _fixed_phys = _ng_phys_cache
            _fixed_sensor = _ng_sensor_cache

            _ng_info: dict = {}

            def ng_residual_closure() -> torch.Tensor:
                """組裝 r ∈ R^N，使 0.5 ||r||² ≈ data_w·l_data + phys_w·(l_ns + cont_w·l_cont)。

                Why scaling 寫法：
                    pi-lnn 原始 loss 用 mean((·)²) → r_i = sqrt(w / (N_i · num_re)) · raw_i
                    使 0.5 ||r||² 與 mean-loss 在尺度上對齊（相差固定 0.5 倍可由 lr 吸收）。
                """
                net.train()
                _h_cache_ng: list[tuple[torch.Tensor, torch.Tensor]] = []
                _parts: list[torch.Tensor] = []
                _l_data_acc = 0.0
                _l_ns_acc = 0.0
                _l_cont_acc = 0.0

                for _i, _ds in enumerate(datasets):
                    _xy, _tq, _c, _ref = _fixed_data[_i]
                    _h, _st = net.encode(
                        sensor_vals_list[_i], sensor_pos_list[_i], _ds.re_norm, sensor_time_list[_i]
                    )
                    _h_cache_ng.append((_h, _st))
                    _bd_d = body_distance_fns[_i](_xy) if _use_hard_body_bc else None
                    _pred = observed_channel_prediction(
                        net=net, xy=_xy, t_q=_tq, c_obs=_c,
                        observed_channel_names=_ds.observed_channel_names,
                        observed_channel_mean=observed_mean_list[_i],
                        observed_channel_std=observed_std_list[_i],
                        h_states=_h, s_time=_st, sensor_pos=sensor_pos_list[_i],
                        body_distance=_bd_d,
                    )
                    _diff = (_pred - _ref).reshape(-1)
                    _N_d = max(_diff.numel(), 1)
                    _scale_d = (_data_w / _N_d) ** 0.5 * _scale_re
                    _parts.append(_scale_d * _diff)
                    _l_data_acc += float(_diff.detach().pow(2).mean().item())
                _l_data_acc /= num_re

                if _fixed_phys:
                    net.eval()
                    for _i, _ds in enumerate(datasets):
                        _xyt = _fixed_phys[_i]
                        _h_p, _st_p = _h_cache_ng[_i]
                        _uvpfn = make_lnn_model_fn_uvp(
                            net, sensor_vals_list[_i], sensor_pos_list[_i],
                            re_norm=_ds.re_norm, sensor_time=sensor_time_list[_i], device=device,
                            h_states=_h_p, s_time=_st_p,
                            body_distance_fn=body_distance_fns[_i] if _use_hard_body_bc else None,
                        )
                        _mu, _mv, _co = unsteady_ns_residuals(
                            _uvpfn, _xyt,
                            re=_ds.re_value, k_f=k_f, A=A, domain_length=domain_length,
                            Lx=_ds.Lx, Ly=_ds.Ly,
                        )
                        if _phys_normalize:
                            def _nr(r: torch.Tensor) -> torch.Tensor:
                                return r / r.detach().std().clamp(min=1e-8)
                            _mu, _mv, _co = _nr(_mu), _nr(_mv), _nr(_co)
                        _N_p = max(_mu.numel(), 1)
                        _sm = (_phys_weight / _N_p) ** 0.5 * _scale_re
                        _sc = (_phys_weight * _cont_w / _N_p) ** 0.5 * _scale_re
                        _parts.append(_sm * _mu.reshape(-1))
                        _parts.append(_sm * _mv.reshape(-1))
                        _parts.append(_sc * _co.reshape(-1))
                        _l_ns_acc += float(
                            (_mu.detach().pow(2).mean() + _mv.detach().pow(2).mean()).item()
                        )
                        _l_cont_acc += float(_co.detach().pow(2).mean().item())

                        # sensor_physics: 在 sensor 位置加 continuity 殘差（一階導，穩定）
                        _xyt_sp = _fixed_sensor[_i] if _fixed_sensor else None
                        if _xyt_sp is not None:
                            _uvp_sp = _uvpfn(_xyt_sp)
                            _u_sp = _uvp_sp[:, 0:1]
                            _v_sp = _uvp_sp[:, 1:2]
                            _du_dx_sp = _grad(_u_sp, _xyt_sp)[:, 0:1]
                            _dv_dy_sp = _grad(_v_sp, _xyt_sp)[:, 1:2]
                            _cont_sp = (_du_dx_sp + _dv_dy_sp).reshape(-1)
                            _N_sp = max(_cont_sp.numel(), 1)
                            _sc_sp = (_phys_weight * _cont_w / _N_sp) ** 0.5 * _scale_re
                            _parts.append(_sc_sp * _cont_sp)
                            _l_cont_acc += float(_cont_sp.detach().pow(2).mean().item())
                    net.train()
                    _l_ns_acc /= num_re
                    _l_cont_acc /= num_re

                _ng_info["l_data"] = _l_data_acc
                _ng_info["l_ns"] = _l_ns_acc
                _ng_info["l_cont"] = _l_cont_acc
                _ng_info["l_phys"] = _l_ns_acc + _cont_w * _l_cont_acc
                _ng_info["l_total"] = _data_w * _l_data_acc + _phys_weight * _ng_info["l_phys"]
                return torch.cat(_parts)

            optimizer.step(ng_residual_closure)

            l_data       = torch.tensor([_ng_info.get("l_data",  0.0)], device=device)
            l_physics    = torch.tensor([_ng_info.get("l_phys",  0.0)], device=device)
            l_total      = torch.tensor([_ng_info.get("l_total", 0.0)], device=device)
            l_ns_u_total = torch.zeros(1, device=device)
            l_ns_v_total = torch.zeros(1, device=device)
            l_ns_total   = torch.tensor([_ng_info.get("l_ns",    0.0)], device=device)
            l_cont_total = torch.tensor([_ng_info.get("l_cont",  0.0)], device=device)
            l_bc_total   = torch.zeros(1, device=device)  # NG path 尚未實作 BC
            phys_weight  = _phys_weight

        else:
        # ── First-order path ─────────────────────────────────────────────────
            optimizer.zero_grad()

        if not is_lbfgs and not is_ng:
            l_data = torch.zeros(1, device=device)
            # temporal trim 後的 sensor 輸入快取，physics loop 同步使用，
            # 確保 data / physics 兩條路徑看到相同的 encoder 輸入。
            # Why: 若 data 用 trimmed 輸入但 physics 用 full 輸入，
            #      GradNorm 的梯度範數計算基準不一致，step=1000 會爆 NaN。
            _trim_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
            # h_states/s_time 跨 data → physics 路徑共用，避免在 physics 區塊
            # 重複跑一次完整 encoder（spatial × T + temporal CfC × T）。
            # 模型無 dropout/BN，data 跑 train()、physics 跑 eval()，encoder
            # 輸出對 mode 不敏感，因此 cache 安全。
            _h_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
            for i, ds in enumerate(datasets):
                # num_query_points 預設由 K（sensor 數）決定，可在 config 中 override。
                n_query = int(args.get("num_query_points", 0)) or ds.sensor_pos.shape[0]
                xy_np, t_np, c_np, ref_np = ds.sample_sensor_batch(
                    rng, n=n_query, t_max=t_max
                )
                # temporal trim — time_marching 期間只傳 t ≤ t_max 的時間步給 encoder。
                # Why: CfC 每步跑全部 T=201 時間步；warmup 時 t_max≈0.5 只需 ~20 步，
                #      其餘 ~180 步計算量是浪費。Trim 後 query decoder 仍能正確
                #      插值，因為所有 query time ≤ t_max 都在 s_time 範圍內。
                if t_max is not None:
                    _n_act = max(1, int((sensor_time_list[i] <= t_max).sum().item()))
                    _sv_enc = sensor_vals_list[i][:_n_act]
                    _st_enc = sensor_time_list[i][:_n_act]
                else:
                    _sv_enc = sensor_vals_list[i]
                    _st_enc = sensor_time_list[i]
                _trim_cache.append((_sv_enc, _st_enc))
                h_states, s_time = net.encode(
                    _sv_enc, sensor_pos_list[i], ds.re_norm, _st_enc
                )
                _h_cache.append((h_states, s_time))
                xy = torch.tensor(xy_np, dtype=torch.float32, device=device)
                t_q = torch.tensor(t_np, device=device)
                c = torch.tensor(c_np, dtype=torch.long, device=device)
                ref = torch.tensor(ref_np, device=device)
                bd_q = body_distance_fns[i](xy) if _use_hard_body_bc else None
                pred = observed_channel_prediction(
                    net=net,
                    xy=xy,
                    t_q=t_q,
                    c_obs=c,
                    observed_channel_names=ds.observed_channel_names,
                    observed_channel_mean=observed_mean_list[i],
                    observed_channel_std=observed_std_list[i],
                    h_states=h_states,
                    s_time=s_time,
                    sensor_pos=sensor_pos_list[i],
                    body_distance=bd_q,
                )
                per_sample_loss = (pred - ref) ** 2
                t0_w = float(args.get("t_early_weight", 1.0))
                if t0_w != 1.0:
                    t0_thresh = float(args.get("t_early_threshold", 0.1))
                    w = torch.where(t_q <= t0_thresh, torch.full_like(t_q, t0_w), torch.ones_like(t_q))
                    per_sample_loss = per_sample_loss * w
                l_data = l_data + per_sample_loss.mean()
            l_data = l_data / num_re

            phys_weight = physics_weight_at_step(
                step=step,
                final_weight=base_phys_weight,
                warmup_steps=phys_warmup_steps,
                ramp_steps=phys_ramp_steps,
            )
            poisson_weight = float(args.get("poisson_loss_weight", 0.0))
            # GradNorm 模式下不依賴 phys_weight 作為 gate，只要 num_physics_points > 0 就計算物理項。
            n_phys_end    = int(args["num_physics_points"])
            n_phys_start  = int(args.get("num_physics_points_start", 0)) or n_phys_end
            n_phys_warmup = int(args.get("num_physics_points_warmup_steps", 0))
            n_phys_ramp   = int(args.get("num_physics_points_ramp_steps", 0))
            n_phys = physics_points_at_step(step, n_phys_start, n_phys_end, n_phys_ramp, n_phys_warmup)

            phys_gate = (use_gradnorm or phys_weight > 0.0) and n_phys_end > 0
            _bc_weight = float(args.get("bc_loss_weight", 0.0))

            # RAR pool update（在 net.eval() 之前，避免 eval/train 交替干擾）
            if _is_rar and phys_gate and (_rar_pool_np is None or step % _rar_update_freq == 0):
                _rar_pool_np = _rar_update_pool(
                    net, datasets, sensor_vals_list, sensor_pos_list, sensor_time_list,
                    rng=rng, n_select=n_phys,
                    pool_size=max(n_phys * _rar_pool_mult, n_phys + 1),
                    t_max=t_max, k_f=k_f, A=A, domain_length=domain_length, device=device,
                    exploration_ratio=_rar_expl_ratio,
                    # Hard body BC 相容：傳 body_distance_fn 給 RAR pool 內部
                    # make_lnn_model_fn_uvp，否則 use_hard_body_bc=True 時 raise。
                    body_distance_fns=body_distance_fns if _use_hard_body_bc else None,
                )

            if phys_gate:
                net.eval()
                l_ns_u_total = torch.zeros(1, device=device)
                l_ns_v_total = torch.zeros(1, device=device)
                l_ns_total = torch.zeros(1, device=device)
                l_cont_total = torch.zeros(1, device=device)
                l_poisson_total = torch.zeros(1, device=device)
                phys_strategy = str(args.get("physics_collocation_strategy", "random"))
                phys_normalize = bool(args.get("physics_residual_normalize", False))
                _use_sensor_phys = bool(args.get("use_sensor_physics", False))
                _sp_n_t = int(args.get("num_sensor_physics_time_samples", 4))
                for i, ds in enumerate(datasets):
                    if _is_rar and _rar_pool_np is not None:
                        xyt = torch.tensor(_rar_pool_np[i], device=device, requires_grad=True)
                    else:
                        xy_np, t_np = ds.sample_physics_points(
                            rng, n=n_phys, t_max=t_max, strategy=phys_strategy
                        )
                        xyt = torch.tensor(
                            np.concatenate([xy_np, t_np[:, None]], axis=1),
                            device=device,
                            requires_grad=True,
                        )
                    # 準備 sensor physics 的 xyt（但不 concat 進隨機批次）。
                    # Why: sensor physics 只計算 continuity（一階導數），不計算 momentum（需二階）。
                    #      模型在感測器位置有精確擬合的空間梯度（data loss 所致），
                    #      ∂²u/∂x² 在這些位置可能 overflow float32 → NaN。
                    #      continuity = ∂u/∂x + ∂v/∂y 只需一階 _grad，數值穩定。
                    _xyt_sp: torch.Tensor | None = None
                    _sp_start = int(args.get("sensor_physics_start_step", 0))
                    if _use_sensor_phys and step >= _sp_start:
                        _t_all = ds.sensor_time  # numpy [T]
                        _t_avail = _t_all[_t_all <= float(t_max if t_max is not None else _t_all[-1])]
                        if len(_t_avail) > 0:
                            _n_t = min(_sp_n_t, len(_t_avail))
                            _t_idx = rng.choice(len(_t_avail), size=_n_t, replace=False)
                            _t_sp = _t_avail[_t_idx]
                            _xy_sp = np.repeat(ds.sensor_pos, _n_t, axis=0)
                            _t_sp_rep = np.tile(_t_sp, ds.sensor_pos.shape[0])[:, None]
                            _xyt_sp = torch.tensor(
                                np.concatenate([_xy_sp, _t_sp_rep], axis=1).astype(np.float32),
                                device=device, requires_grad=True,
                            )
                    _sv_phys, _st_phys = _trim_cache[i] if _trim_cache else (sensor_vals_list[i], sensor_time_list[i])
                    _h_phys, _st_phys_cached = _h_cache[i] if _h_cache else (None, None)
                    uvp_fn = make_lnn_model_fn_uvp(
                        net,
                        _sv_phys,
                        sensor_pos_list[i],
                        re_norm=ds.re_norm,
                        sensor_time=_st_phys,
                        device=device,
                        body_distance_fn=body_distance_fns[i] if _use_hard_body_bc else None,
                        h_states=_h_phys,
                        s_time=_st_phys_cached,
                    )
                    mom_u, mom_v, cont = unsteady_ns_residuals(
                        uvp_fn, xyt, re=ds.re_value, k_f=k_f, A=A,
                        domain_length=domain_length, Lx=ds.Lx, Ly=ds.Ly,
                    )
                    if phys_normalize:
                        def _norm_r(r: torch.Tensor) -> torch.Tensor:
                            return r / r.detach().std().clamp(min=1e-8)
                        mom_u = _norm_r(mom_u)
                        mom_v = _norm_r(mom_v)
                        cont  = _norm_r(cont)
                    if use_causal:
                        # 因果加權 mean(r^2)：以時間 bin 為單位，按累積殘差衰減後段權重。
                        # Why: chaotic flow 的 Lyapunov 不穩定使早期誤差指數放大；
                        #      強制 t=0 收斂前 t>0 不主導梯度。
                        weighted, _w_t = causal_weighted_residual_loss(
                            [mom_u, mom_v, cont],
                            xyt[:, 2],
                            num_bins=causal_num_bins,
                            eps=causal_eps,
                            t_max=t_max,
                        )
                        l_ns_u_total = l_ns_u_total + weighted[0]
                        l_ns_v_total = l_ns_v_total + weighted[1]
                        l_cont_total = l_cont_total + weighted[2]
                    else:
                        l_ns_u_total = l_ns_u_total + torch.mean(mom_u ** 2)
                        l_ns_v_total = l_ns_v_total + torch.mean(mom_v ** 2)
                        l_cont_total = l_cont_total + torch.mean(cont ** 2)
                    # sensor physics：僅計算 continuity（第一階導數，穩定）。
                    # 為了重用 uvp_fn，會多算一次 p_sp（不進 loss），多出的成本
                    # ≈ 1/3 個 decoder forward × K * n_t 點，相對於整體 NS 殘差可忽略。
                    if _xyt_sp is not None:
                        uvp_sp = uvp_fn(_xyt_sp)
                        u_sp = uvp_sp[:, 0:1]
                        v_sp = uvp_sp[:, 1:2]
                        du_dx_sp = _grad(u_sp, _xyt_sp)[:, 0:1]
                        dv_dy_sp = _grad(v_sp, _xyt_sp)[:, 1:2]
                        cont_sp = du_dx_sp + dv_dy_sp
                        l_cont_total = l_cont_total + torch.mean(cont_sp ** 2)
                    if poisson_weight > 0.0:
                        poisson_res = pressure_poisson_residual(uvp_fn, xyt, Lx=ds.Lx, Ly=ds.Ly)
                        l_poisson_total = l_poisson_total + torch.mean(poisson_res ** 2)
                # ── Boundary BC loss（cylinder only）─────────────────────────
                # 物理 BC（依 Lily-Pad solver setBC()，非週期 open flow）：
                #   inflow (x=0): Dirichlet u=u_inf, v=0
                #   body    : Dirichlet u=v=0 （no-slip）
                #   y=0/y=1 : Neumann (slip)；簡化為 v=0（資料 std(v_top/bot) 小）
                #   outlet (x=1): Neumann ∂u/∂x≈0（暫未實作，wake 區資料密隱式約束）
                #
                # 計算協定（修正前的 bug）：
                #   _bc_fn 回傳 normalized model output (mean~0, std~1)。
                #   舊 code 寫 ((pred_norm - u_inf_phys) / std)² 是雙重 normalize，
                #   實質要求 pred_norm = u_inf_phys + mean，違反 no-slip / inflow 物理。
                # 修正：把 physical target 轉到 normalized 空間，loss 在 normalized 空間
                # 直接做 (pred_norm - target_norm)²，與 data loss 同尺度。
                l_bc_total = torch.zeros(1, device=device)
                _bc_weight = float(args.get("bc_loss_weight", 0.0))
                if _bc_weight > 0.0 and str(args.get("dataset_type", "")) == "cylinder":
                    # bc_inflow_u 優先用 dataset 自動量測值（per-Re 自動），
                    # config override 僅當顯式設了非預設值才生效。
                    _u_inf_cfg   = args.get("bc_inflow_u", None)
                    _n_inflow    = int(args.get("bc_n_points", 32))
                    _n_body      = int(args.get("bc_body_n_points", 0))
                    _n_slip      = int(args.get("bc_slip_n_points", 0))
                    _t_max_bc = float(t_max) if t_max is not None else float(datasets[0].sensor_time[-1])
                    for i, ds in enumerate(datasets):
                        # per-dataset inlet u：dataset attribute 為主，config 為 fallback override
                        _u_inf = float(getattr(ds, "bc_inflow_u", _u_inf_cfg if _u_inf_cfg is not None else 0.33))
                        _u_idx = ds.observed_channel_names.index("u")
                        _v_idx = ds.observed_channel_names.index("v")
                        _u_mean_i = observed_mean_list[i][_u_idx]
                        _v_mean_i = observed_mean_list[i][_v_idx]
                        _u_std_i  = observed_std_list[i][_u_idx]
                        _v_std_i  = observed_std_list[i][_v_idx]
                        # Physical → normalized targets (一次算好，多 BC 共用)
                        _u_inf_norm  = (_u_inf - _u_mean_i) / _u_std_i.clamp(min=1e-8)
                        _u_zero_norm = (0.0   - _u_mean_i) / _u_std_i.clamp(min=1e-8)
                        _v_zero_norm = (0.0   - _v_mean_i) / _v_std_i.clamp(min=1e-8)
                        _t_lo, _t_hi = float(ds.sensor_time[0]), _t_max_bc

                        _sv_bc, _st_bc = _trim_cache[i]
                        _bc_fn = make_lnn_model_fn(
                            net, _sv_bc, sensor_pos_list[i],
                            re_norm=ds.re_norm, sensor_time=_st_bc, device=device,
                            body_distance_fn=body_distance_fns[i] if _use_hard_body_bc else None,
                        )

                        # 1. Inflow BC（x=0：u=u_inf, v=0）
                        _y_in = rng.uniform(0.0, 1.0, size=(_n_inflow,)).astype(np.float32)
                        _x_in = np.zeros(_n_inflow, dtype=np.float32)
                        _t_in = rng.uniform(_t_lo, _t_hi, size=(_n_inflow,)).astype(np.float32)
                        _xyt_in = torch.tensor(
                            np.stack([_x_in, _y_in, _t_in], axis=1), device=device
                        )
                        _u_in = _bc_fn(_xyt_in, c=0)
                        _v_in = _bc_fn(_xyt_in, c=1)
                        l_bc_total = (
                            l_bc_total
                            + torch.mean((_u_in - _u_inf_norm) ** 2)
                            + torch.mean((_v_in - _v_zero_norm) ** 2)
                        )

                        # 2. Body no-slip BC（cylinder 內部：u=v=0）
                        # Why: body 是 wake 起源，無此 BC 模型在 body 表面可能輸出非零速度，
                        #      造成下游 wake 結構失真。Dirichlet 內部點可代表「body 為固體」。
                        if _n_body > 0 and ds.body_xy.shape[0] > 0:
                            _bidx = rng.integers(0, ds.body_xy.shape[0], size=_n_body)
                            _xy_body = ds.body_xy[_bidx].astype(np.float32)  # [N, 2] normalized
                            _t_body = rng.uniform(_t_lo, _t_hi, size=(_n_body,)).astype(np.float32)
                            _xyt_body = torch.tensor(
                                np.concatenate([_xy_body, _t_body[:, None]], axis=1),
                                device=device,
                            )
                            _u_body = _bc_fn(_xyt_body, c=0)
                            _v_body = _bc_fn(_xyt_body, c=1)
                            l_bc_total = (
                                l_bc_total
                                + torch.mean((_u_body - _u_zero_norm) ** 2)
                                + torch.mean((_v_body - _v_zero_norm) ** 2)
                            )

                        # 3. Slip BC（y=0, y=1 簡化為 v=0）
                        # Why: solver 用 Neumann ∂u/∂y=0；資料量測 v 在 top/bottom 接近 0，
                        #      用 v=0 作為等效 Dirichlet 簡化，避免引入 gradient 計算。
                        #      不約束 u 因為 u 在 y 邊界並非常數（圓柱繞流加速）。
                        if _n_slip > 0:
                            _per_side = max(1, _n_slip // 2)
                            _x_slip = rng.uniform(0.0, 1.0, size=(2 * _per_side,)).astype(np.float32)
                            _y_slip = np.concatenate([
                                np.zeros(_per_side, dtype=np.float32),  # y=0
                                np.ones(_per_side,  dtype=np.float32),  # y=1
                            ])
                            _t_slip = rng.uniform(_t_lo, _t_hi,
                                                   size=(2 * _per_side,)).astype(np.float32)
                            _xyt_slip = torch.tensor(
                                np.stack([_x_slip, _y_slip, _t_slip], axis=1),
                                device=device,
                            )
                            _v_slip = _bc_fn(_xyt_slip, c=1)
                            l_bc_total = l_bc_total + torch.mean((_v_slip - _v_zero_norm) ** 2)

                        # 4. Outlet BC (x=1: ∂u/∂x ≈ 0, free-stream 出口)
                        # Why: 沒有 outlet BC 時 model 在 x→1 區域沒物理約束，
                        #      wake 漂移到出口時可能形成 unphysical recirculation。
                        #      Lily-Pad solver 用 Neumann ∂u/∂x=0；用 autograd 計算
                        #      gradient 並懲罰其偏離 0。
                        _n_outlet = int(args.get("bc_outlet_n_points", 0))
                        if _n_outlet > 0:
                            _y_out = rng.uniform(0.0, 1.0, size=(_n_outlet,)).astype(np.float32)
                            _x_out = np.ones(_n_outlet, dtype=np.float32)  # x=1
                            _t_out = rng.uniform(_t_lo, _t_hi, size=(_n_outlet,)).astype(np.float32)
                            _xyt_out = torch.tensor(
                                np.stack([_x_out, _y_out, _t_out], axis=1),
                                device=device, requires_grad=True,
                            )
                            # 用 forward_uvp 取 [u, v, p]，對 x 求 gradient
                            _uvp_out = make_lnn_model_fn_uvp(
                                net, _sv_bc, sensor_pos_list[i],
                                re_norm=ds.re_norm, sensor_time=_st_bc, device=device,
                                body_distance_fn=body_distance_fns[i] if _use_hard_body_bc else None,
                            )(_xyt_out)
                            _u_out = _uvp_out[:, 0:1]
                            _v_out = _uvp_out[:, 1:2]
                            _du_out = _grad(_u_out, _xyt_out)
                            _dv_out = _grad(_v_out, _xyt_out)
                            _du_dx_out = _du_out[:, 0:1]    # ∂u/∂x
                            _dv_dx_out = _dv_out[:, 0:1]    # ∂v/∂x
                            # Note: gradient 在 normalized 空間（座標 [0,1]）。
                            # u 是 physical 量級（forward_uvp 已 denorm），
                            # 故 (∂u/∂x_norm)² 量級 ≈ (∂u/∂x_phys × Lx)²，
                            # 直接 mean((·)²) 跟其他 BC normalize 空間量級不可比。
                            # 折衷：除以 std_u 讓 outlet BC 跟其他 BC 同量級。
                            _scale_u = _u_std_i.clamp(min=1e-8)
                            _scale_v = _v_std_i.clamp(min=1e-8)
                            l_bc_total = l_bc_total + torch.mean((_du_dx_out / _scale_u) ** 2)
                            l_bc_total = l_bc_total + torch.mean((_dv_dx_out / _scale_v) ** 2)
                    l_bc_total = l_bc_total / num_re

                net.train()
                l_ns_u_total = l_ns_u_total / num_re
                l_ns_v_total = l_ns_v_total / num_re
                l_ns_total = l_ns_u_total + l_ns_v_total
                l_cont_total = l_cont_total / num_re
                l_poisson_total = l_poisson_total / num_re
                l_physics = (
                    l_ns_total
                    + args["continuity_weight"] * l_cont_total
                    + poisson_weight * l_poisson_total
                )
            else:
                l_ns_u_total = torch.zeros(1, device=device)
                l_ns_v_total = torch.zeros(1, device=device)
                l_ns_total = torch.zeros(1, device=device)
                l_cont_total = torch.zeros(1, device=device)
                l_bc_total   = torch.zeros(1, device=device)
                l_physics = torch.zeros(1, device=device)

            # AL term（spec v4 §4）：在組 l_total 之前一次計算，後面分支共用
            # 注意：l_cont_total 在 AL 模式下強制為 random-collocation 單一 mean
            # （use_sensor_physics 已被 pre-condition assert 強制 false）
            al_term: torch.Tensor | None = None
            if al_cont is not None:
                al_term = al_cont.loss_term(l_cont_total)

            # ── Loss 組合與 backward ────────────────────────────────────────────
            if use_gradnorm and gn_weights is not None:
                # GradNorm 模式：可學習權重直接管理各 task 比例。
                # 動態 task layout：依 gn_weights.task_names 決定哪些 loss 進 GradNorm。
                # AL active 時：AL term 完全不進 gn_losses（spec v4 §5），避免 mean_G 污染。
                #
                # 執行順序（關鍵）：
                #   ① GradNorm weight update（autograd.grad，retain_graph=True）
                #   ② 以更新後的 ws（detach）計算 l_total
                #   ③ l_total.backward() → optimizer.step()
                #
                # Why: optimizer.step() 就地修改 trunk_out.weight（版本號遞增），
                #      GradNorm 必須在 step() 前完成，否則 PyTorch 拋出版本衝突錯誤。
                phys_active = int(args["num_physics_points"]) > 0
                do_gn_update = phys_active and (step > warmup_steps) and (step % gn_update_freq == 0)

                # 依 task_names 動態組裝 losses（與 task index 對齊）
                # "poisson" 為 EXP-072 專用 5-task layout 的可選項
                _name_to_loss = {
                    "data": l_data,
                    "ns_u": l_ns_u_total,
                    "ns_v": l_ns_v_total,
                    "cont": l_cont_total,
                    "bc": l_bc_total,
                    "poisson": l_poisson_total,
                }
                _gn_losses = [_name_to_loss[name] for name in gn_weights.task_names]

                if do_gn_update:
                    _gradnorm_step(
                        gn_weights,
                        _gn_losses,
                        gn_ref_params,
                        ema_momentum=float(args.get("gradnorm_ema_momentum", 0.5)),
                        min_weight=float(args.get("gradnorm_min_weight", 0.0)),
                    )
                ws = gn_weights.weights.detach()

                # 依 task_names 組 l_total（GradNorm-managed terms）
                if phys_active:
                    l_total = sum(ws[i] * _gn_losses[i] for i in range(len(_gn_losses)))
                else:
                    # phys 未啟用時只留 data + (BC 若有)
                    l_total = ws[gn_weights.index_of("data")] * l_data
                    if "bc" in gn_weights:
                        l_total = l_total + ws[gn_weights.index_of("bc")] * l_bc_total

                # GradNorm 不管的固定 weight 項：
                #   - BC：當 "bc" 不在 GradNorm tasks 時用固定 _bc_weight
                if "bc" not in gn_weights:
                    l_total = l_total + _bc_weight * l_bc_total
                # AL term：固定 weight = 1，AL 與 GradNorm 完全解耦（spec v4 §5）
                if al_term is not None:
                    l_total = l_total + al_term

                l_total.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), float(args["max_grad_norm"]))
                optimizer.step()
            else:
                # 非 GradNorm 路徑（包含 EXP-070 純 AL）
                if al_term is not None:
                    # AL active：cont 由 AL 接管（cont_w 已被 assert 為 0）；
                    # ns_u/ns_v 用固定 phys_weight，data 用 data_loss_weight
                    l_total = (
                        args["data_loss_weight"] * l_data
                        + phys_weight * l_ns_total
                        + al_term
                        + _bc_weight * l_bc_total
                    )
                else:
                    l_total = (
                        args["data_loss_weight"] * l_data
                        + phys_weight * l_physics
                        + _bc_weight * l_bc_total
                    )
                l_total.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), float(args["max_grad_norm"]))
                optimizer.step()

            # AL dual update（spec v4 §4）— 嚴格在 optimizer.step() 之後，且 step > 0 才動
            # 用的是 pre-step 計算的 l_cont_total（一步延遲，acceptable，spec 已說明）
            if al_cont is not None and step > 0 and step % al_update_freq == 0:
                al_cont.update(l_cont_total.detach())

        if scheduler is not None:
            scheduler.step()

        # MPS memory pool 釋放：每 200 步清空 MPS 快取的未使用記憶體。
        # Why: temporal trim 產生 ~180 種不同形狀，MPS allocator 對每種形狀
        #      保留快取但不主動釋放；定期 empty_cache 防止累積到 20GB+。
        if device.type == "mps" and step % 200 == 0:
            torch.mps.empty_cache()

        if log_fn is not None:
            extra: dict[str, float] = {}
            if use_gradnorm and gn_weights is not None:
                ws_vals = gn_weights.weights.detach().cpu().tolist()
                for name, val in zip(gn_weights.task_names, ws_vals):
                    extra[f"gn_w_{name}"] = float(val)
            if al_cont is not None:
                extra["al_lambda_c"] = float(al_cont.lambda_.item())
                extra["al_C_ema"] = float(al_cont.ema_C.item())
            log_fn(step, {
                "l_data": l_data.item(),
                "l_physics": l_physics.item(),
                "l_ns": l_ns_total.item(),
                "l_cont": l_cont_total.item(),
                "l_bc": l_bc_total.item(),
                "l_total": l_total.item(),
                "w_phys": phys_weight,
                "w_bc": float(args.get("bc_loss_weight", 0.0)),
                "t_max": t_max if t_max is not None else 0.0,
                **extra,
            })

        if step % max(1, args["iterations"] // 10) == 0 or step == 1:
            tm_str = f"  t≤{t_max:5.1f}" if use_tm and t_max is not None else ""
            # 主行：依 GradNorm task_names 顯示 weight（跳過 data，與 header 對齊）
            if use_gradnorm and gn_weights is not None:
                ws_vals = gn_weights.weights.detach().cpu().tolist()
                _w_str = " ".join(
                    f"{ws_vals[i]:>8.4f}"
                    for i, name in enumerate(gn_weights.task_names) if name != "data"
                )
                main = (
                    f"{step:<8} {l_data.item():>12.4e}"
                    f" {l_physics.item():>12.4e} {_w_str}"
                    f" {l_total.item():>12.4e}"
                )
            else:
                main = (
                    f"{step:<8} {l_data.item():>12.4e}"
                    f" {l_physics.item():>12.4e} {phys_weight:>10.4f}"
                    f" {l_total.item():>12.4e}"
                )
            if al_cont is not None:
                main += f" {al_cont.lambda_.item():>10.3e} {al_cont.ema_C.item():>10.3e}"
            print(main + tm_str, flush=True)

        if args["checkpoint_period"] > 0 and step % args["checkpoint_period"] == 0:
            # Mid-step ckpt 存 train mode (param.data = x_t)，給 resume 用。
            # Why: schedulefree 切 eval mode 會把 param.data 從 x_t (training iterate)
            #      換成 y_t (Polyak averaged inference weights)。如果存 eval mode，
            #      resume 後 wrapper 預設 train mode 會把 y_t 當 x_t 繼續 update，
            #      → x 跟 z buffer 漂走 → loss 爆炸（cylinder_007b 已驗證 bug）。
            #      所以中間 ckpt 必須存 train mode；final.pt 才存 eval mode（推理用）。
            torch.save(
                {
                    "step": step,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    # GradNorm state：避免 resume 時重置成 init weights → loss 爆炸
                    "gradnorm_state_dict": gn_weights.state_dict() if gn_weights is not None else None,
                    # AL state：保 lambda_ / ema_C / _initialized 三 buffer
                    "al_state_dict": al_cont.state_dict() if al_cont is not None else None,
                    # Mode metadata：evaluate 用此判斷 ckpt 是 inference-ready 還是 resume-only
                    #   "train" = x_t (training iterate, 給 resume，inference quality 差)
                    #   "eval"  = y_t (averaged, inference 用)
                    #   "n/a"   = 不是 schedulefree（一般 optimizer，無此區分）
                    "schedulefree_mode": "train" if use_schedulefree else "n/a",
                },
                str(checkpoints_dir / f"lnn_kolmogorov_step_{step}.pt"),
            )

    if use_schedulefree:
        optimizer.eval()  # final.pt 儲存 Polyak 平均推理權重
    final = artifacts_dir / "lnn_kolmogorov_final.pt"
    torch.save(net.state_dict(), str(final))

    # Manifest: 包含 AL 最終狀態 + log columns（spec §5 logging / §8 pathology hook）
    manifest: dict = {
        "configuration": {k: v for k, v in args.items() if k not in ("sensor_jsons", "sensor_npzs", "dns_paths")},
        "final_checkpoint": str(final),
    }
    if al_cont is not None:
        manifest["al_final_state"] = {
            "lambda_": float(al_cont.lambda_.item()),
            "ema_C": float(al_cont.ema_C.item()),
            "saturated_clip": bool(al_cont.lambda_.item() >= al_cont.lambda_clip - 1e-9),
        }
    if use_gradnorm and gn_weights is not None:
        manifest["gradnorm_final_weights"] = dict(zip(
            gn_weights.task_names,
            gn_weights.weights.detach().cpu().tolist(),
        ))
    write_json(artifacts_dir / "experiment_manifest.json", manifest)
    print("=== Done ===")


def main() -> None:
    """What: CLI entry point for core LNN training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train core Pi-LNN on Kolmogorov flow."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=None)
    cli_args = parser.parse_args()

    config = dict(DEFAULT_LNN_ARGS)
    config.update(load_lnn_config(cli_args.config))
    if cli_args.device is not None:
        config["device"] = cli_args.device
    # AL semantic validation：必須在 merge 後（spec v4 §6）
    from pi_lnn.config import _validate_al_config
    _validate_al_config(config)
    train_lnn_kolmogorov(config)

if __name__ == "__main__":
    main()
