"""tests/test_gradnorm_task_names.py

GradNormWeights v4 擴充：
- task_names 顯式指定 vs 由 init_weights 長度推斷的兩條路徑等價（4-task / 5-task）
- index_of(name) / __contains__(name) API
- 「al」可出現在 task_names（class 不限制；validator 才禁），但 EXP-071 必須是 3-task
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pi_lnn import GradNormWeights


def test_default_4_task_layout_inferred() -> None:
    gn = GradNormWeights([1.0, 0.01, 0.01, 0.01])
    assert gn.task_names == ("data", "ns_u", "ns_v", "cont")


def test_default_5_task_layout_inferred() -> None:
    gn = GradNormWeights([1.0, 0.01, 0.01, 0.01, 0.1])
    assert gn.task_names == ("data", "ns_u", "ns_v", "cont", "bc")


def test_explicit_task_names_overrides_inference() -> None:
    gn = GradNormWeights(
        [1.0, 0.01, 0.01],
        task_names=["data", "ns_u", "ns_v"],
    )
    assert gn.task_names == ("data", "ns_u", "ns_v")


def test_unknown_length_without_explicit_names_raises() -> None:
    """3-element init_weights without explicit task_names → ambiguous → raise."""
    with pytest.raises(ValueError, match="task_names"):
        GradNormWeights([1.0, 0.01, 0.01])


def test_mismatched_explicit_length_raises() -> None:
    with pytest.raises(ValueError, match="長度"):
        GradNormWeights([1.0, 0.01], task_names=["data", "ns_u", "ns_v"])


def test_index_of_returns_correct_index() -> None:
    gn = GradNormWeights([1.0, 0.01, 0.01, 0.01])
    assert gn.index_of("data") == 0
    assert gn.index_of("ns_u") == 1
    assert gn.index_of("ns_v") == 2
    assert gn.index_of("cont") == 3


def test_index_of_unknown_raises_keyerror() -> None:
    gn = GradNormWeights([1.0, 0.01, 0.01, 0.01])
    with pytest.raises(KeyError):
        gn.index_of("al")


def test_contains_works() -> None:
    gn = GradNormWeights([1.0, 0.01, 0.01, 0.01])
    assert "cont" in gn
    assert "al" not in gn
    assert "bc" not in gn


def test_normalize_to_data_unchanged() -> None:
    """normalize_to_data_ 行為與既有版本一致：data 永遠 = 1。"""
    gn = GradNormWeights([2.0, 0.5, 0.3, 0.1])
    gn.normalize_to_data_()
    ws = gn.weights.detach()
    assert abs(ws[0].item() - 1.0) < 1e-6
    # 比例保留：原始 [2.0, 0.5, 0.3, 0.1] → [1.0, 0.25, 0.15, 0.05]
    assert abs(ws[1].item() - 0.25) < 1e-6
    assert abs(ws[3].item() - 0.05) < 1e-6


def test_explicit_3_task_no_cont() -> None:
    """EXP-071 layout: ["data","ns_u","ns_v"] — 不含 cont。"""
    gn = GradNormWeights([1.0, 0.01, 0.01], task_names=["data", "ns_u", "ns_v"])
    assert "cont" not in gn
    assert "al" not in gn
    assert gn.weights.detach().shape[0] == 3


def test_explicit_layout_with_al_allowed_at_class_level() -> None:
    """class 自身不禁 'al'（讓 validator 集中守門）。"""
    # 這個情境會被 _validate_al_config 拒絕，但 GradNormWeights 本身應該能 instantiate
    gn = GradNormWeights(
        [1.0, 0.01, 0.01, 0.01],
        task_names=["data", "ns_u", "ns_v", "al"],
    )
    assert "al" in gn
