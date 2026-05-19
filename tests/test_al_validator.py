"""tests/test_al_validator.py

Spec v4 §6 `_validate_al_config` semantic 檢查。

涵蓋：
- AL on + lr_schedule="ng" → raise（spec NG path 不支援）
- AL on + lr_schedule="lbfgs" + use_gradnorm=True → raise（closure race）
- AL on + continuity_weight != 0 → raise
- AL on + use_sensor_physics=True → raise
- AL on + "cont" in gradnorm_tasks → raise
- AL on + "al" in gradnorm_tasks → raise（v4 規定）
- AL on + tasks/init_weights 長度不符 → raise
- AL off + use_gradnorm + cont 不在 tasks + cont_w>0 → raise（無效設定守門）
- 合法配置不 raise
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pi_con import DEFAULT_PICON_ARGS, _validate_al_config


def _base_al_on(**overrides):
    cfg = {
        **DEFAULT_PICON_ARGS,
        "use_augmented_lagrangian": True,
        "continuity_weight": 0.0,
        "use_sensor_physics": False,
        "lr_schedule": "soap",
    }
    cfg.update(overrides)
    return cfg


def test_al_off_default_passes() -> None:
    _validate_al_config({**DEFAULT_PICON_ARGS, "use_augmented_lagrangian": False})


def test_al_on_minimal_valid_passes() -> None:
    _validate_al_config(_base_al_on(use_gradnorm=False))


def test_al_with_ng_raises() -> None:
    with pytest.raises(ValueError, match="lr_schedule='ng'"):
        _validate_al_config(_base_al_on(lr_schedule="ng"))


def test_al_with_lbfgs_and_gradnorm_raises() -> None:
    with pytest.raises(ValueError, match="LBFGS 不支援 use_gradnorm"):
        _validate_al_config(
            _base_al_on(
                lr_schedule="lbfgs",
                use_gradnorm=True,
                gradnorm_tasks=["data", "ns_u", "ns_v"],
                gradnorm_init_weights=[1.0, 0.01, 0.01],
            )
        )


def test_al_lbfgs_without_gradnorm_passes() -> None:
    """LBFGS + AL（無 GradNorm）合法。"""
    _validate_al_config(_base_al_on(lr_schedule="lbfgs", use_gradnorm=False))


def test_al_with_continuity_weight_nonzero_raises() -> None:
    with pytest.raises(ValueError, match="continuity_weight 必須 = 0"):
        _validate_al_config(_base_al_on(continuity_weight=1.0))


def test_al_with_sensor_physics_raises() -> None:
    with pytest.raises(ValueError, match="use_sensor_physics"):
        _validate_al_config(_base_al_on(use_sensor_physics=True))


def test_al_with_cont_in_tasks_raises() -> None:
    with pytest.raises(ValueError, match="'cont' 必須從 gradnorm_tasks 移出"):
        _validate_al_config(
            _base_al_on(
                use_gradnorm=True,
                gradnorm_tasks=["data", "ns_u", "ns_v", "cont"],
                gradnorm_init_weights=[1.0, 0.01, 0.01, 0.01],
            )
        )


def test_al_with_al_in_tasks_raises() -> None:
    """v4 規定：AL term 不進 GradNorm losses 列表。"""
    with pytest.raises(ValueError, match="'al' 不能出現在 gradnorm_tasks"):
        _validate_al_config(
            _base_al_on(
                use_gradnorm=True,
                gradnorm_tasks=["data", "ns_u", "ns_v", "al"],
                gradnorm_init_weights=[1.0, 0.01, 0.01, 0.01],
            )
        )


def test_al_with_mismatched_tasks_weights_length_raises() -> None:
    with pytest.raises(ValueError, match="長度"):
        _validate_al_config(
            _base_al_on(
                use_gradnorm=True,
                gradnorm_tasks=["data", "ns_u", "ns_v"],
                gradnorm_init_weights=[1.0, 0.01],   # 長度 2 ≠ tasks 3
            )
        )


def test_al_off_invalid_3task_no_cont_raises() -> None:
    """AL off + use_gradnorm + 3-task 缺 cont + cont_w>0 → 無效設定守門。"""
    with pytest.raises(ValueError, match="無效設定"):
        _validate_al_config({
            **DEFAULT_PICON_ARGS,
            "use_augmented_lagrangian": False,
            "use_gradnorm": True,
            "gradnorm_tasks": ["data", "ns_u", "ns_v"],
            "gradnorm_init_weights": [1.0, 0.01, 0.01],
            "continuity_weight": 1.0,
        })


def test_al_off_legacy_4task_passes() -> None:
    """AL off + 既有 4-task layout（cont 在 tasks）：完全合法，向後相容。"""
    _validate_al_config({
        **DEFAULT_PICON_ARGS,
        "use_augmented_lagrangian": False,
        "use_gradnorm": True,
        "gradnorm_tasks": ["data", "ns_u", "ns_v", "cont"],
        "gradnorm_init_weights": [1.0, 0.01, 0.01, 0.01],
    })


def test_al_on_with_3task_gradnorm_passes() -> None:
    """EXP-071 配置：AL on + 3-task GradNorm（不含 cont/al）合法。"""
    _validate_al_config(
        _base_al_on(
            use_gradnorm=True,
            gradnorm_tasks=["data", "ns_u", "ns_v"],
            gradnorm_init_weights=[1.0, 0.01, 0.01],
        )
    )
