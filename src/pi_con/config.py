"""Default training arguments and TOML config loading."""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


DEFAULT_PICON_ARGS: dict[str, Any] = {
    "sensor_jsons": None,
    "sensor_npzs": None,
    "dns_paths": None,
    "re_values": None,
    "observed_sensor_channels": ["u", "v"],
    "sensor_noise_std": 0.0,          # per-channel std-relative Gaussian additive noise（0 = clean）；
                                       # e.g. 0.05 = 5% noise; injected pre-normalize, seed-locked。
    "fourier_harmonics": 8,
    "fourier_embed_dim": 0,   # 0 = 使用舊版確定性諧波；>0 = 啟用 LearnableFourierEmb
    "d_model": 64,
    "d_time": 8,
    "num_spatial_cfc_layers": 1,
    "num_temporal_cfc_layers": 1,
    "num_token_attention_layers": 1,
    "token_attention_heads": 4,
    "num_query_mlp_layers": 1,
    "query_mlp_hidden_dim": 64,
    "decoder_attention_heads": 1,    # cross-attention head 數；1=原 single-head；
                                      # >1 = multi-head reshape (param count 不變，
                                      # 純 inductive bias 細分)。需 query_mlp_hidden_dim
                                      # 能整除。rel_bias/locality_decay 仍 shared across heads。
    "use_modified_mlp": False,        # Wang 2021 modified MLP (mMLP) trunk gating。
                                      # False = 原 ResidualMLPBlock；True = ModifiedMLPBlock
                                      # (z_l = (1-H_l)U + H_l V)。緩解 PINN spectral bias，
                                      # 加 U/V projection 約 +130K params。
    "use_vanilla_deeponet": False,    # B0 ablation: 純 DeepONet (MLP branch + MLP trunk +
                                      # inner product, 無 CfC 無 cross-attention)。
                                      # False = 主架構 (LiquidOperator)；True = VanillaDeepONetOperator。
    "vanilla_deeponet_K_sensors": 100,        # B0 branch input dim = K * sensor_value_dim
    "vanilla_deeponet_num_branch_layers": 3,  # B0 branch ResidualMLPBlock 層數
    "vanilla_deeponet_num_trunk_layers": 3,   # B0 trunk ResidualMLPBlock 層數
    "disable_cross_attention": False,         # B1 ablation: 停用 decoder cross-attention，
                                              # 改用 mean-pool over K sensor tokens。仍保 CfC encoder。
    "use_standard_pinn": False,               # Standard PINN baseline (Wang 2021 style):
                                              # (x, y, t) → MLP → (u, v, p)。sensor 只入 loss。
                                              # 用於對比「operator framework」vs「plain PINN」。
    "standard_pinn_num_layers": 6,            # Standard PINN backbone layer 數 (典型 6-8)
    "standard_pinn_hidden_dim": 512,          # Standard PINN backbone width (典型 256-1024)
    "standard_pinn_activation": "silu",       # PINN activation: "silu" (= Swish-1, modern,
                                              # PirateNet 2024) 或 "tanh" (classical, Raissi 2019)。
    "num_query_cfc_layers": 1,
    "query_gate_bias_span": 1.0,
    "output_head_gain": 1.0,
    "operator_rank": 64,
    "fusion_temperature_init": None,
    "use_locality_decay": False,
    "use_bidirectional_cfc": False,
    "cfc_log_tau_min": -1.0,        # CfC 時間常數初始下界 log τ；對 turbulence 多尺度建議 -3.0
    "cfc_log_tau_max": 1.0,         # CfC 時間常數初始上界 log τ；典型 1.0~1.6（log T_total）
    "fourier_sigma_bands": None,    # LearnableFourierEmb 多頻段 σ；None=單一 σ；例 [1.0, 4.0, 12.0]
    "fourier_band_dim_ratios": None,  # 對應各 band 的通道比例；總和需 = 1.0；例 [0.5, 0.375, 0.125]
    "data_loss_weight": 1.0,
    "t_early_weight": 1.0,       # t <= t_early_threshold 的 data loss 乘數（1.0 = 無加權）
    "t_early_threshold": 0.1,    # 早期時間定義上限
    "lbfgs_max_iter": 20,        # L-BFGS 每步最大 line-search 次數
    "lbfgs_history_size": 10,    # L-BFGS curvature history buffer 大小
    # --- Natural Gradient (Gauss-Newton) optimizer ---
    # 啟用：lr_schedule="ng"。論文：Curvature-Aware Optimization for High-Accuracy PINNs。
    # 適用範圍：N (residuals) << P (params) 才划算（pi-con 典型 N~200, P~10⁵）。
    "ng_damping": 1.0e-6,          # LM 正則化強度旋鈕。注意 ng_jacobi_scaling=True
                                   # 時 effective regulariser 為 λD（row-magnitude
                                   # weighted），非 isotropic λI；λ 仍是調整旋鈕，
                                   # 但其絕對量級對應 row-relative 而非全域 isotropic。
                                   # 詳見 solve_ng_step docstring。
    "ng_damping_strategy": "fixed",# "fixed" 或 "lm"（自適應，多花一次 closure）
    "ng_jacobi_scaling": True,     # van der Sluis 對角預條件；改變 damping 結構，
                                   # 見 ng_damping 與 solve_ng_step docstring。
    "ng_max_residuals": 2000,      # N 上限保護（O(N·P) 記憶體）
    "ng_solver_device": "cpu",     # 線性求解 device；cpu 對 fp64 最穩
    "ng_resample_freq": 50,        # collocation / sensor batch 重採頻率（步）；
                                   # 0 = 永不重採（首次採樣後固定，論文 Helmholtz 風格）；
                                   # >0 = 每 N 步重新採樣（論文 Stokes 採 fixed within phase）
    # Line search（論文 Stokes case "NG with Line-search"）
    "ng_line_search": "none",      # "none" | "backtracking" | "armijo"
    "ng_ls_max_trials": 5,         # α=1, decay, decay²,... 最多試幾次
    "ng_ls_alpha_init": 1.0,       # 初始 step size（NG 預設 Newton step = 1.0）
    "ng_ls_alpha_decay": 0.5,      # backtracking 衰減係數
    "ng_ls_armijo_c1": 1.0e-4,     # Armijo sufficient decrease 條件係數
    # SPRING momentum（論文 eq.(26)，Goldshlager et al. 2024 ref [26]）
    "ng_use_spring": False,        # 啟用 SPRING-style momentum 加速
    "ng_spring_momentum": 0.9,     # μ ∈ [0, 1)；0 退化為標準 NG
    "physics_loss_weight": 0.01,
    "physics_loss_warmup_steps": 0,
    "physics_loss_ramp_steps": 0,
    "continuity_weight": 1.0,
    "use_gradnorm": False,
    "gradnorm_alpha": 1.5,   # 已棄用，保留供舊 config 相容
    "gradnorm_lr": 1e-3,     # 已棄用，保留供舊 config 相容
    "gradnorm_update_freq": 10,
    "gradnorm_min_weight": 0.0,    # Sanity floor (預設關閉，讓 GradNorm 自然動態)
                                    # 之前 cylinder_007/008 的 div_L2 大問題已查明是 distance-as-input
                                    # 的 detach chain rule bug（不是 GradNorm pathology），因此不需 floor。
                                    # 仍保留作為 opt-in（>0 啟用），給未來確認 GradNorm 真的失控時用。
    "gradnorm_max_weight": 0.0,    # Physics task weight cap (預設關閉)。
                                    # EXP-071 驗證：3-task GradNorm + AL 把 ns_u/ns_v 推到 0.30 級別 →
                                    # data 權重相對被壓 → KE rel-err 從 7.80% 退步到 14.57%。
                                    # EXP-075 設計用 cap=0.20 處理此 trade-off，期望兼顧 div 與 KE。
                                    # **僅 cap index ≥ 1（physics tasks），data (index 0) 永保 = 1**。
    "gradnorm_init_weights": [1.0, 0.01, 0.01, 0.01, 0.1],
    # 5-task layout: [data, ns_u, ns_v, cont, bc]（cylinder 主線 default）
    # 4-task layout: [data, ns_u, ns_v, cont]（kolmogorov / 無 BC 場景；自動 fallback 4）
    # Why: BC 跟 NS / cont 都是 PINN constraint 的一部分，應該被 GradNorm 統一動態
    #      平衡。pre-fix 時 bc_loss_weight=0.1 是基於 broken BC 量級 tune 的，
    #      修 C3 後 BC 量級變化 40×，固定 weight 失準 → 改用 GradNorm 自適應。
    "gradnorm_ema_momentum": 0.9,
    "warmup_steps": 0,
    "time_marching": True,
    "time_marching_start": 0.5,
    "time_marching_warmup": 0.5,
    "domain_length": 1.0,
    "use_periodic_domain": True,
    # NS residual 路徑是否反正規化 model output 到物理量級。
    # True  → 用 dataset.observed_channel_mean/std 反算，與物理 ν=1/Re commensurate；
    #         cylinder 主線需要（u_std~0.15，黏性項 ν∇²u 否則被壓掉）。
    # False → NS residual 直接用 normalized u/v，相對 ν 量級被 std⁻² 改寫，
    #         等效 Re_eff = Re·std²；Kolmogorov（u_std~0.44）的 EXP-064 在此路徑下取得 KE 7.80%。
    # 向後相容：False 預設保持 EXP-064 baseline 路徑。
    "use_physics_denormalization": False,
    "dataset_type": "kolmogorov",
    "arrow_shards": [],
    "sensor_subsample": 1,
    "kolmogorov_k_f": 4.0,
    "kolmogorov_A": 0.1,
    # ForcingPrior：把 (A, k_f) 從 PDE 已知 prior 改成 optional learnable scalar。
    # 預設 false → 行為與舊版完全一致；學時把 nn.Parameter 加入 optimizer 一起更新。
    "learn_forcing_A": False,
    "learn_forcing_k_f": False,
    "forcing_A_init": 0.05,        # 起點刻意偏離真值，避免 trivial 收斂
    "forcing_k_f_init": 4.0,
    "forcing_k_f_min": 1.0,        # sigmoid 下界，避免退化成 DC forcing
    "forcing_k_f_max": 8.0,        # 上界略高於 K=100 識別上限 k_max≈5.64，留 buffer 觀察 overshoot
    "forcing_log_every": 100,      # 每 N step 記錄 A/k_f 到 metrics.jsonl
    "use_temporal_anchor": False,
    "resume_checkpoint": None,
    "T_total": 5.0,
    "temporal_anchor_harmonics": 2,
    "iterations": 1000,
    "num_query_points": 0,        # 0 = 由 sensor K 自動決定；正整數 = override
    "num_physics_points": 32,     # 最終（最大）collocation 點數
    "num_physics_points_start": 0,         # curriculum 初始點數；0 = 與 num_physics_points 相同（固定）
    "num_physics_points_warmup_steps": 0,  # ramp 開始前的等待步數
    "num_physics_points_ramp_steps": 0,    # 線性增長步數；0 = warmup 後立即使用最終值
    "physics_collocation_strategy": "random",
    "rar_update_freq": 50,         # RAR: 每幾步重新評估 residual pool
    "rar_pool_multiplier": 10,     # RAR: pool 大小 = num_physics_points × multiplier
    "rar_exploration_ratio": 0.2,  # RAR: 保留隨機點比例（防 mode collapse）
    "physics_residual_normalize": False,
    "use_causal_weighting": False,    # PINN causal weighting (Wang 2022); 與 t_early_weight 互斥
    "causal_eps": 1.0,                # 因果嚴格度；0=均勻，∞=只訓 t=0；典型 0.5~10.0
    "causal_num_bins": 16,            # 時間 bin 數；建議 16~32（太大 cumsum 不穩）
    "use_sensor_physics": False,           # 在感測器位置額外計算 NS 殘差
    "num_sensor_physics_time_samples": 4,  # 每訓練步從 sensor_time 中採樣幾個時間步
    "sensor_physics_start_step": 0,        # 延遲啟動：前 N 步只用隨機 collocation
    "poisson_loss_weight": 0.0,
    "bc_loss_weight": 0.0,    # 整體 BC loss 權重；0 = 停用全部 BC（含 inflow/body/slip）
    "bc_inflow_u": 0.33,      # legacy：來流 u 速度（物理值，m/s）；
                              # cylinder dataset 自動從 DNS shard 量測 ds.bc_inflow_u 並覆蓋；
                              # 此 config 值僅在 dataset 沒提供時作 fallback。
    "bc_n_points": 32,        # inflow 邊界（x=0）採樣點數
    "bc_body_n_points": 0,    # cylinder body no-slip BC 採樣點數；0 = 停用
    "bc_slip_n_points": 0,    # top/bottom slip BC (y=0,y=1) 採樣點數；0 = 停用
    "bc_outlet_n_points": 0,  # cylinder outlet (x=1) ∂u/∂x≈0 BC 採樣點數；0 = 停用
    "use_hard_body_bc": False,           # cylinder hard body BC（Sukumar 2022 風格）：
                                          # u = (φ/scale).clamp(0,1) · NN，物理保證 body 內 u=v=0。
                                          # 取代有 detach bug 的 distance-as-input feature。
                                          # True 會改 model architecture（query_in +1），ckpt 不相容
                                          # 僅 cylinder dataset 真正需要；kolmogorov 為 dummy 1.0
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "lr_schedule": "none",
    "use_schedule_free": False,
    "lr_warmup_steps": 300,
    "soap_precondition_frequency": 10,
    "soap_betas": [0.95, 0.95],
    "soap_use_step_decay": False,
    "lr_decay_steps": 1000,
    "lr_decay_gamma": 0.9,
    "min_learning_rate": 1e-6,
    "max_grad_norm": 1.0,
    "checkpoint_period": 100,
    "seed": 42,
    "device": "mps",
    "artifacts_dir": "artifacts/deeponet-cfc-midlong-uvomega-small",
    # --- Augmented Lagrangian on continuity (spec v4) ---
    # 啟用：use_augmented_lagrangian=true。詳見 docs/superpowers/specs/2026-05-04-al-continuity-design.md
    # 與 lr_schedule="ng" 不相容；EXP-070/071 限用 SOAP 或 AdamW（"none"/"cosine"/"step"/"soap"）。
    "use_augmented_lagrangian": False,
    "al_init_lambda": 0.0,
    "al_rho": 1.0,
    "al_update_freq": 100,         # 每 N steps 執行一次 dual update
    "al_lambda_clip": 10.0,        # 上限；clamp(0, Λ)，C ≥ 0 故無下限負值
    "al_ema_momentum": 0.5,
    "al_allow_cont_in_gradnorm": False,    # ADR-001 §4 escape hatch — 預設禁止。
                                            # EXP-079 (2026-05-08) 啟用驗證此禁令是否過於保守。
                                            # 預期: KE 7-9% + div < 0.10 → §4 需修訂；
                                            # KE > 12% 或 div > 0.30 → §4 合理。
    # Multi-constraint AL (EXP-242 系列, 2026-05-19)
    # 對哪些 physics constraint 套 AL dual penalty；預設 ["cont"] 等價舊 single-AL 行為。
    # 可選: ["cont"], ["ns_u","ns_v"], ["ns_u","ns_v","cont"]
    # 共用 al_init_lambda/al_rho/al_lambda_clip/al_ema_momentum hyperparams (v1)。
    # Pre-condition: l_ns_u/v/cont 都已是 squared mean ≥ 0，符合 AL "C ≥ 0" 假設。
    "al_constraints": ["cont"],
    # 是否允許 ns_u/ns_v 同時 in GradNorm + AL (類似 al_allow_cont_in_gradnorm)
    # 預設 false: AL active on ns → ns 必須從 gradnorm_tasks 拿出（強制純 AL）
    "al_allow_ns_in_gradnorm": False,
    "keep_last_n_checkpoints": 2,           # 只保留最後 N 個 step_*.pt 中間 ckpt（節省磁碟）。
                                            # 預設 2 = 留最新 + 一個 fallback；resume 用最新；
                                            # 0 = 全保留（向後相容）。final.pt 不受影響。
    "gradnorm_tasks": [],          # [] = 由 init_weights 長度推斷 4/5-task；
                                   # 顯式：例如 ["data","ns_u","ns_v"] 對應 EXP-071
}

_REMOVED_KEYS = {
    "nhead",
    "dim_feedforward",
    "attn_dropout",
    "num_encoder_layers",
    "use_local_struct_features",
    "sensor_knn_k",
    "num_latent_tokens",
}


def _find_project_root(start: Path) -> Path | None:
    """What: 從指定路徑向上尋找專案根目錄。"""
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return None


def _resolve_config_path_value(raw_path: str | Path, config_path: Path) -> str:
    """What: 以相容既有 workflow 的方式解析 config 內路徑。

    Why: 目前 repo 同時存在兩種寫法：
         1) 相對專案根目錄（如 `data/...`）
         2) 相對 config 檔位置（外部/臨時 config 常見）
         若只支援其中一種會直接破壞 userspace。
    """
    path = Path(raw_path)
    if path.is_absolute():
        return str(path.resolve())

    config_dir = config_path.parent
    project_root = _find_project_root(config_dir)
    candidates = [
        config_dir / path,
        *( [project_root / path] if project_root is not None else [] ),
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    # 新路徑尚未建立時，優先用 project_root；若找不到則退回 config_dir
    if project_root is not None:
        return str((project_root / path).resolve())
    return str(candidates[0].resolve())


def load_picon_config(config_path: Path | None) -> dict[str, Any]:
    """What: 載入並驗證核心 PI-CON config。"""
    if config_path is None:
        return {}
    config_path = Path(config_path).resolve()
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config_data = payload.get("train", payload)
    normalized = dict(config_data)
    obsolete = sorted(set(normalized) & _REMOVED_KEYS)
    if obsolete:
        raise ValueError(
            f"Config 含有已移除的 PiT 欄位（請改用 num_spatial/temporal_cfc_layers）: {obsolete}"
        )
    unknown = sorted(set(normalized) - set(DEFAULT_PICON_ARGS))
    if unknown:
        raise ValueError(f"PI-CON config 含有不支援的欄位: {unknown}")
    for list_key in ("sensor_jsons", "sensor_npzs", "dns_paths"):
        if list_key in normalized:
            normalized[list_key] = [_resolve_config_path_value(p, config_path) for p in normalized[list_key]]
    if "artifacts_dir" in normalized:
        normalized["artifacts_dir"] = _resolve_config_path_value(normalized["artifacts_dir"], config_path)
    return normalized


def _validate_al_config(cfg: dict[str, Any]) -> None:
    """What: AL semantic validation — fail fast on the FULLY MERGED config.

    必須在 `DEFAULT_PICON_ARGS` 與 TOML 合併**後**呼叫（不能在 `load_picon_config` 內），
    否則 cfg 會缺少 default fallback 值。

    參考 spec: docs/superpowers/specs/2026-05-04-al-continuity-design.md §6
    """
    if not cfg.get("use_augmented_lagrangian", False):
        # AL off：仍要驗 gradnorm_tasks 不能誤刪 cont（否則 div constraint 消失）
        tasks = list(cfg.get("gradnorm_tasks", []) or [])
        if (
            cfg.get("use_gradnorm", False)
            and tasks
            and "cont" not in tasks
            and float(cfg.get("continuity_weight", 0.0)) > 0.0
        ):
            raise ValueError(
                "use_gradnorm=True + cont not in gradnorm_tasks + continuity_weight>0 → "
                "cont 既不被 GradNorm 平衡也不在固定 weight loss 中（無效設定）"
            )
        return

    # AL on:
    if cfg.get("lr_schedule") == "ng":
        raise ValueError("use_augmented_lagrangian incompatible with lr_schedule='ng'")
    if cfg.get("lr_schedule") == "lbfgs" and cfg.get("use_gradnorm", False):
        raise ValueError("AL + LBFGS 不支援 use_gradnorm（closure race）")
    if float(cfg.get("continuity_weight", 0.0)) != 0.0:
        raise ValueError(
            f"AL active 時 continuity_weight 必須 = 0，收到 {cfg['continuity_weight']}"
        )
    if cfg.get("use_sensor_physics", False):
        raise ValueError(
            "AL v1 不支援 use_sensor_physics（l_cont_total 會變成 sum-of-two-means）"
        )
    tasks = list(cfg.get("gradnorm_tasks", []) or [])
    # Multi-constraint AL (EXP-242): al_constraints 指定要套 AL 的 physics constraints。
    al_constraints = list(cfg.get("al_constraints", ["cont"]) or ["cont"])
    _ALLOWED_AL = {"cont", "ns_u", "ns_v"}
    _bad = [c for c in al_constraints if c not in _ALLOWED_AL]
    if _bad:
        raise ValueError(
            f"al_constraints 含未知 constraint {_bad}; 允許值 {_ALLOWED_AL}"
        )
    if not al_constraints:
        raise ValueError("use_augmented_lagrangian=True 但 al_constraints 為空")

    # ADR-001 §4 禁令：AL 與 GradNorm 不可同時控制同一 task。EXP-079 (2026-05-08) 開啟
    # escape hatch 驗證此禁令是否真有必要 — opt-in via `al_allow_cont_in_gradnorm`。
    _allow_cont = bool(cfg.get("al_allow_cont_in_gradnorm", False))
    _allow_ns = bool(cfg.get("al_allow_ns_in_gradnorm", False))
    if tasks and "cont" in al_constraints and "cont" in tasks and not _allow_cont:
        raise ValueError(
            "AL on 'cont' 時 'cont' 必須從 gradnorm_tasks 移出。"
            " 若要實驗性違反 ADR-001 §4 禁令，明示設 al_allow_cont_in_gradnorm = true。"
        )
    for _n in ("ns_u", "ns_v"):
        if tasks and _n in al_constraints and _n in tasks and not _allow_ns:
            raise ValueError(
                f"AL on '{_n}' 時 '{_n}' 必須從 gradnorm_tasks 移出。"
                f" 若要保留 GradNorm+AL 雙開（同 cont 模式），明示設 al_allow_ns_in_gradnorm = true。"
            )
    if tasks and "al" in tasks:
        raise ValueError(
            "v4 規定 AL term 不進 GradNorm losses 列表 — 'al' 不能出現在 gradnorm_tasks"
        )
    init_w = list(cfg.get("gradnorm_init_weights", []) or [])
    if tasks and len(init_w) != len(tasks):
        raise ValueError(
            f"gradnorm_init_weights 長度 ({len(init_w)}) 與 gradnorm_tasks ({len(tasks)}) 不符"
        )
