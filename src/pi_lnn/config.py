"""Default training arguments and TOML config loading."""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


DEFAULT_LNN_ARGS: dict[str, Any] = {
    "sensor_jsons": None,
    "sensor_npzs": None,
    "dns_paths": None,
    "re_values": None,
    "observed_sensor_channels": ["u", "v"],
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
    "num_query_cfc_layers": 1,
    "query_gate_bias_span": 1.0,
    "output_head_gain": 1.0,
    "operator_rank": 64,
    "fusion_temperature_init": None,
    "use_locality_decay": False,
    "use_bidirectional_cfc": False,
    "cfc_log_tau_min": -1.0,        # CfC 時間常數初始下界 log τ；對 turbulence 多尺度建議 -3.0
    "cfc_log_tau_max": 1.0,         # CfC 時間常數初始上界 log τ；典型 1.0~1.6（log T_total）
    "data_loss_weight": 1.0,
    "t_early_weight": 1.0,       # t <= t_early_threshold 的 data loss 乘數（1.0 = 無加權）
    "t_early_threshold": 0.1,    # 早期時間定義上限
    "lbfgs_max_iter": 20,        # L-BFGS 每步最大 line-search 次數
    "lbfgs_history_size": 10,    # L-BFGS curvature history buffer 大小
    "physics_loss_weight": 0.01,
    "physics_loss_warmup_steps": 0,
    "physics_loss_ramp_steps": 0,
    "continuity_weight": 1.0,
    "use_gradnorm": False,
    "gradnorm_alpha": 1.5,   # 已棄用，保留供舊 config 相容
    "gradnorm_lr": 1e-3,     # 已棄用，保留供舊 config 相容
    "gradnorm_update_freq": 10,
    "gradnorm_init_weights": [1.0, 0.01, 0.01, 0.01],
    "gradnorm_ema_momentum": 0.9,
    "warmup_steps": 0,
    "time_marching": True,
    "time_marching_start": 0.5,
    "time_marching_warmup": 0.5,
    "domain_length": 1.0,
    "use_periodic_domain": True,
    "dataset_type": "kolmogorov",
    "arrow_shards": [],
    "sensor_subsample": 1,
    "kolmogorov_k_f": 4.0,
    "kolmogorov_A": 0.1,
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
    "use_sensor_physics": False,           # 在感測器位置額外計算 NS 殘差
    "num_sensor_physics_time_samples": 4,  # 每訓練步從 sensor_time 中採樣幾個時間步
    "sensor_physics_start_step": 0,        # 延遲啟動：前 N 步只用隨機 collocation
    "poisson_loss_weight": 0.0,
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


def load_lnn_config(config_path: Path | None) -> dict[str, Any]:
    """What: 載入並驗證核心 LNN config。"""
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
    unknown = sorted(set(normalized) - set(DEFAULT_LNN_ARGS))
    if unknown:
        raise ValueError(f"LNN config 含有不支援的欄位: {unknown}")
    for list_key in ("sensor_jsons", "sensor_npzs", "dns_paths"):
        if list_key in normalized:
            normalized[list_key] = [_resolve_config_path_value(p, config_path) for p in normalized[list_key]]
    if "artifacts_dir" in normalized:
        normalized["artifacts_dir"] = _resolve_config_path_value(normalized["artifacts_dir"], config_path)
    return normalized
