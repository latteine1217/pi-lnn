"""3D channel config keys — schema 擴展驗證（Plan 2 Task 2a）。

驗證新增的 3D spatial keys 進入 DEFAULT_PICON_ARGS（2D 向後相容 default），
且 load_picon_config 接受含這些 key 的 TOML，同時不破壞既有 unknown-key 防護。
"""
import pytest

from pi_con.config import DEFAULT_PICON_ARGS, load_picon_config


def test_3d_keys_present_in_defaults():
    """新 3D keys 必須在 DEFAULT_PICON_ARGS 且 default 為 2D 向後相容值。"""
    assert DEFAULT_PICON_ARGS["num_spatial_dims"] == 2
    assert DEFAULT_PICON_ARGS["num_velocity_components"] == 3
    assert DEFAULT_PICON_ARGS["Lz"] == 1.0
    assert DEFAULT_PICON_ARGS["periodic_axes"] is None


def test_load_config_accepts_3d_keys(tmp_path):
    """含 3D keys 的 TOML 應被接受且值正確。"""
    cfg_file = tmp_path / "channel.toml"
    cfg_file.write_text(
        "[train]\n"
        "num_spatial_dims = 3\n"
        "num_velocity_components = 4\n"
        "Lz = 9.42477796\n"
        "periodic_axes = [0, 2]\n"
    )
    cfg = load_picon_config(cfg_file)
    assert cfg["num_spatial_dims"] == 3
    assert cfg["num_velocity_components"] == 4
    assert cfg["Lz"] == pytest.approx(9.42477796)
    assert cfg["periodic_axes"] == [0, 2]


def test_unknown_key_still_rejected(tmp_path):
    """新增 3D keys 不可破壞既有 unknown-key 防護。"""
    cfg_file = tmp_path / "bad.toml"
    cfg_file.write_text("[train]\nthis_key_does_not_exist = 1\n")
    with pytest.raises(ValueError, match="不支援的欄位"):
        load_picon_config(cfg_file)
