"""tests/test_al_disabled_equivalence.py

Spec v4 §6 backward-compat 契約：`use_augmented_lagrangian=false`（預設）時，
config / training / loss 行為與既有版本 numerically equivalent。

實作策略（避免跑完整 training loop）：
- 直接驗 _validate_al_config 對 default config 不 raise
- 驗 GradNormWeights 預設 4-task layout 與舊版行為一致
- 驗 default DEFAULT_LNN_ARGS["use_augmented_lagrangian"] = False
- 驗 default config 沒有 al_* 干擾欄位
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pi_lnn import DEFAULT_LNN_ARGS, GradNormWeights, _validate_al_config


def test_default_args_have_al_off() -> None:
    assert DEFAULT_LNN_ARGS["use_augmented_lagrangian"] is False
    assert DEFAULT_LNN_ARGS["gradnorm_tasks"] == []   # 空 = 由長度推斷
    assert DEFAULT_LNN_ARGS["continuity_weight"] == 1.0  # 既有預設不變


def test_default_config_passes_validator() -> None:
    """純預設不該 raise（向後相容預設行為）。"""
    _validate_al_config(dict(DEFAULT_LNN_ARGS))


def test_legacy_4task_gradnorm_config_passes_validator() -> None:
    """既有 EXP-064 風格 config（4-task GradNorm，無 AL）：完全合法。"""
    cfg = {
        **DEFAULT_LNN_ARGS,
        "use_gradnorm": True,
        "gradnorm_init_weights": [1.0, 0.01, 0.01, 0.01],
        "use_augmented_lagrangian": False,
    }
    _validate_al_config(cfg)


def test_legacy_5task_gradnorm_config_passes_validator() -> None:
    """既有 cylinder 風格 config（5-task：含 BC，無 AL）：完全合法。"""
    cfg = {
        **DEFAULT_LNN_ARGS,
        "use_gradnorm": True,
        "gradnorm_init_weights": [1.0, 0.01, 0.01, 0.01, 0.1],
        "use_augmented_lagrangian": False,
    }
    _validate_al_config(cfg)


def test_gradnorm_weights_4task_same_layout_as_legacy() -> None:
    """`GradNormWeights([1.0, 0.01, 0.01, 0.01])` 推斷出 4-task layout，
    與既有 EXP-064 hard-coded 對應一致。
    """
    gn = GradNormWeights([1.0, 0.01, 0.01, 0.01])
    assert gn.task_names == ("data", "ns_u", "ns_v", "cont")
    # weights 與舊版相同
    ws = gn.weights.detach().tolist()
    assert abs(ws[0] - 1.0) < 1e-6
    assert all(abs(w - 0.01) < 1e-6 for w in ws[1:])


def test_no_extra_state_when_al_off() -> None:
    """AL off 時，DEFAULT_LNN_ARGS 中的 al_* 欄位全為 sane defaults，不影響行為。"""
    assert DEFAULT_LNN_ARGS["al_init_lambda"] == 0.0
    assert DEFAULT_LNN_ARGS["al_rho"] == 1.0
    assert DEFAULT_LNN_ARGS["al_update_freq"] == 100
    assert DEFAULT_LNN_ARGS["al_lambda_clip"] == 10.0
    assert DEFAULT_LNN_ARGS["al_ema_momentum"] == 0.5
