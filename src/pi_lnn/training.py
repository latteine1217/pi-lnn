"""Pi-LNN training loop and CLI entry point.

A1 boundary: train_lnn_kolmogorov is moved verbatim. Decomposing it is the
deferred A2 phase (see docs/superpowers/specs/2026-04-26-pi-lnn-package-refactor-design.md).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from pi_lnn.causal import causal_weighted_residual_loss
from pi_lnn.config import DEFAULT_LNN_ARGS, load_lnn_config
from pi_lnn.losses import GradNormWeights, _gradnorm_step, observed_channel_prediction
from pi_lnn.operator import LiquidOperator, create_lnn_model, make_lnn_model_fn
from pi_lnn.physics import (
    _rar_update_pool,
    physics_points_at_step,
    physics_weight_at_step,
    pressure_poisson_residual,
    unsteady_ns_residuals,
)
from pi_lnn.runtime import configure_torch_runtime, count_parameters, write_json


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

    net = create_lnn_model(args).to(device)
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

    if args["lr_schedule"] == "soap":
        import sys
        _soap_dir = str(Path(__file__).parent.parent / "SOAP")
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
            # L-BFGS 與 checkpoint 的 optimizer 類型不同，跳過 optimizer state 載入。
            if not is_lbfgs and "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                _fix_optimizer_state_compat(optimizer)
            if not use_schedulefree and scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_step = int(ckpt["step"])
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

    # GradNorm setup
    use_gradnorm = bool(args.get("use_gradnorm", False))
    gn_weights: GradNormWeights | None = None
    gn_ref_params: list[torch.Tensor] = []
    gn_update_freq = int(args.get("gradnorm_update_freq", 200))
    if use_gradnorm:
        gn_weights = GradNormWeights(
            init_weights=args.get("gradnorm_init_weights", [1.0, 0.01, 0.01, 0.01])
        ).to(device)
        gn_ref_params = list(net.query_decoder.trunk_out.parameters())

    print("=== Training ===", flush=True)
    if use_tm:
        print(f"  time_marching: t [{tm_t_start:.1f} → {tm_t_end:.1f}]  warmup={tm_warmup} steps", flush=True)
    if use_gradnorm:
        init_w = args.get("gradnorm_init_weights", [1.0, 0.01, 0.01, 0.01])
        print(f"  GradNorm: momentum={args.get('gradnorm_ema_momentum', 0.5):.2f}  freq={gn_update_freq}  (direct formula + EMA)")
        print(f"  GradNorm init_weights: {init_w}  (4 tasks: data, ns_u, ns_v, cont)")
    elif phys_warmup_steps > 0 or phys_ramp_steps > 0:
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
    if use_gradnorm:
        print(f"{'Step':<8} {'L_data':>12} {'L_phys':>12} {'w_ns_u':>8} {'w_ns_v':>8} {'w_cont':>8} {'L_total':>12}"
              + ("  t_max" if use_tm else ""), flush=True)
    else:
        print(f"{'Step':<8} {'L_data':>12} {'L_phys':>12} {'w_phys':>10} {'L_total':>12}"
              + ("  t_max" if use_tm else ""), flush=True)

    _is_rar = str(args.get("physics_collocation_strategy", "random")) == "rar"
    _rar_pool_np: list[np.ndarray] | None = None
    _rar_update_freq = int(args.get("rar_update_freq", 50))
    _rar_pool_mult   = int(args.get("rar_pool_multiplier", 10))
    _rar_expl_ratio  = float(args.get("rar_exploration_ratio", 0.2))

    warmup_steps = int(args.get("warmup_steps", 0))

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
                for _i, _ds in enumerate(datasets):
                    _xy, _tq, _c, _ref = _fixed_data[_i]
                    _h, _st = net.encode(
                        sensor_vals_list[_i], sensor_pos_list[_i], _ds.re_norm, sensor_time_list[_i]
                    )
                    _pred = observed_channel_prediction(
                        net=net, xy=_xy, t_q=_tq, c_obs=_c,
                        observed_channel_names=_ds.observed_channel_names,
                        observed_channel_mean=observed_mean_list[_i],
                        observed_channel_std=observed_std_list[_i],
                        h_states=_h, s_time=_st, sensor_pos=sensor_pos_list[_i],
                    )
                    _ld = _ld + ((_pred - _ref) ** 2).mean()
                _ld = _ld / num_re

                _lp = torch.zeros(1, device=device)
                _lcont = torch.zeros(1, device=device)
                if _fixed_phys:
                    net.eval()
                    for _i, _ds in enumerate(datasets):
                        _xyt = _fixed_phys[_i]
                        _mfn = make_lnn_model_fn(
                            net, sensor_vals_list[_i], sensor_pos_list[_i],
                            re_norm=_ds.re_norm, sensor_time=sensor_time_list[_i], device=device,
                        )
                        _uf = lambda x, fn=_mfn: fn(x, c=0)
                        _vf = lambda x, fn=_mfn: fn(x, c=1)
                        _pf = lambda x, fn=_mfn: fn(x, c=2)
                        _mu, _mv, _co = unsteady_ns_residuals(
                            _uf, _vf, _pf, _xyt,
                            re=_ds.re_value, k_f=k_f, A=A, domain_length=domain_length,
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

                _lt = args["data_loss_weight"] * _ld + _phys_weight * (_lp + args["continuity_weight"] * _lcont)
                _lt.backward()
                _lbfgs_info["l_data"]   = _ld.item()
                _lbfgs_info["l_phys"]   = (_lp + _lcont).item()
                _lbfgs_info["l_total"]  = _lt.item()
                return _lt

            optimizer.step(closure)

            l_data    = torch.tensor([_lbfgs_info.get("l_data",  0.0)], device=device)
            l_physics = torch.tensor([_lbfgs_info.get("l_phys",  0.0)], device=device)
            l_total   = torch.tensor([_lbfgs_info.get("l_total", 0.0)], device=device)
            l_ns_u_total = torch.zeros(1, device=device)
            l_ns_v_total = torch.zeros(1, device=device)
            l_cont_total = torch.zeros(1, device=device)
            phys_weight  = _phys_weight

        else:
        # ── First-order path ─────────────────────────────────────────────────
            optimizer.zero_grad()

        if not is_lbfgs:
            l_data = torch.zeros(1, device=device)
            # temporal trim 後的 sensor 輸入快取，physics loop 同步使用，
            # 確保 data / physics 兩條路徑看到相同的 encoder 輸入。
            # Why: 若 data 用 trimmed 輸入但 physics 用 full 輸入，
            #      GradNorm 的梯度範數計算基準不一致，step=1000 會爆 NaN。
            _trim_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
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
                xy = torch.tensor(xy_np, dtype=torch.float32, device=device)
                t_q = torch.tensor(t_np, device=device)
                c = torch.tensor(c_np, dtype=torch.long, device=device)
                ref = torch.tensor(ref_np, device=device)
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

            # RAR pool update（在 net.eval() 之前，避免 eval/train 交替干擾）
            if _is_rar and phys_gate and (_rar_pool_np is None or step % _rar_update_freq == 0):
                _rar_pool_np = _rar_update_pool(
                    net, datasets, sensor_vals_list, sensor_pos_list, sensor_time_list,
                    rng=rng, n_select=n_phys,
                    pool_size=max(n_phys * _rar_pool_mult, n_phys + 1),
                    t_max=t_max, k_f=k_f, A=A, domain_length=domain_length, device=device,
                    exploration_ratio=_rar_expl_ratio,
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
                    model_fn = make_lnn_model_fn(
                        net,
                        _sv_phys,
                        sensor_pos_list[i],
                        re_norm=ds.re_norm,
                        sensor_time=_st_phys,
                        device=device,
                    )
                    u_fn = lambda xyt_, fn=model_fn: fn(xyt_, c=0)
                    v_fn = lambda xyt_, fn=model_fn: fn(xyt_, c=1)
                    p_fn = lambda xyt_, fn=model_fn: fn(xyt_, c=2)
                    mom_u, mom_v, cont = unsteady_ns_residuals(
                        u_fn, v_fn, p_fn, xyt, re=ds.re_value, k_f=k_f, A=A, domain_length=domain_length
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
                    if _xyt_sp is not None:
                        u_sp = u_fn(_xyt_sp)
                        v_sp = v_fn(_xyt_sp)
                        du_dx_sp = _grad(u_sp, _xyt_sp)[:, 0:1]
                        dv_dy_sp = _grad(v_sp, _xyt_sp)[:, 1:2]
                        cont_sp = du_dx_sp + dv_dy_sp
                        l_cont_total = l_cont_total + torch.mean(cont_sp ** 2)
                    if poisson_weight > 0.0:
                        poisson_res = pressure_poisson_residual(u_fn, v_fn, p_fn, xyt)
                        l_poisson_total = l_poisson_total + torch.mean(poisson_res ** 2)
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
                l_physics = torch.zeros(1, device=device)

            # ── Loss 組合與 backward ────────────────────────────────────────────
            if use_gradnorm and gn_weights is not None:
                # GradNorm 模式：4 個可學習權重 [data, ns_u, ns_v, cont]，直接管理各 task 比例。
                # physics_loss_weight 不再作為乘數，只有 num_physics_points > 0 才啟用物理項。
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

                if do_gn_update:
                    _gradnorm_step(
                        gn_weights,
                        [l_data, l_ns_u_total, l_ns_v_total, l_cont_total],
                        gn_ref_params,
                        ema_momentum=float(args.get("gradnorm_ema_momentum", 0.5)),
                    )
                ws = gn_weights.weights.detach()

                if phys_active:
                    l_total = (
                        ws[0] * l_data
                        + ws[1] * l_ns_u_total
                        + ws[2] * l_ns_v_total
                        + ws[3] * l_cont_total
                    )
                else:
                    l_total = ws[0] * l_data

                l_total.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), float(args["max_grad_norm"]))
                optimizer.step()
            else:
                l_total = args["data_loss_weight"] * l_data + phys_weight * l_physics
                l_total.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), float(args["max_grad_norm"]))
                optimizer.step()

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
                extra = {
                    "gn_w_data": ws_vals[0],
                    "gn_w_ns_u": ws_vals[1],
                    "gn_w_ns_v": ws_vals[2],
                    "gn_w_cont": ws_vals[3],
                }
            log_fn(step, {
                "l_data": l_data.item(),
                "l_physics": l_physics.item(),
                "l_ns": l_ns_total.item(),
                "l_cont": l_cont_total.item(),
                "l_total": l_total.item(),
                "w_phys": phys_weight,
                "t_max": t_max if t_max is not None else 0.0,
                **extra,
            })

        if step % max(1, args["iterations"] // 10) == 0 or step == 1:
            tm_str = f"  t≤{t_max:5.1f}" if use_tm and t_max is not None else ""
            if use_gradnorm and gn_weights is not None:
                ws_vals = gn_weights.weights.detach().cpu().tolist()
                print(
                    f"{step:<8} {l_data.item():>12.4e}"
                    f" {l_physics.item():>12.4e}"
                    f" {ws_vals[1]:>8.4f} {ws_vals[2]:>8.4f} {ws_vals[3]:>8.4f}"
                    f" {l_total.item():>12.4e}{tm_str}",
                    flush=True,
                )
            else:
                print(
                    f"{step:<8} {l_data.item():>12.4e}"
                    f" {l_physics.item():>12.4e} {phys_weight:>10.4f}"
                    f" {l_total.item():>12.4e}{tm_str}",
                    flush=True,
                )

        if args["checkpoint_period"] > 0 and step % args["checkpoint_period"] == 0:
            if use_schedulefree:
                optimizer.eval()  # 切換到 Polyak 平均權重後再儲存
            torch.save(
                {
                    "step": step,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                },
                str(checkpoints_dir / f"lnn_kolmogorov_step_{step}.pt"),
            )
            if use_schedulefree:
                optimizer.train()  # 恢復訓練模式

    if use_schedulefree:
        optimizer.eval()  # final.pt 儲存 Polyak 平均推理權重
    final = artifacts_dir / "lnn_kolmogorov_final.pt"
    torch.save(net.state_dict(), str(final))
    write_json(artifacts_dir / "experiment_manifest.json", {
        "configuration": {k: v for k, v in args.items() if k not in ("sensor_jsons", "sensor_npzs", "dns_paths")},
        "final_checkpoint": str(final),
    })
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
    train_lnn_kolmogorov(config)

if __name__ == "__main__":
    main()
