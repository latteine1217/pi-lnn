"""test_sensor_axis_convention.py — regression guard for sensor row/col swap bug.

Bug history (2026-05-18):
    `scripts/generate_sensors_qrpivot_from_les.py` 與 `generate_sensors_random_from_dns.py`
    in versions prior to commit X 使用 `u_full[:, y_idx, x_idx]` 抽 sensor value，
    但 DNS array convention 是 `u[t, axis_1=x, axis_2=y]`。
    結果 4 個 cross-source placement (EXP-101/102/103/105) 的 NPZ 值在
    swap 位置 (dns_x[y_idx], dns_y[x_idx]) 上，不是 JSON coord 標的 (dns_x[x_idx], dns_y[y_idx])。
    Model 訓練成 transposed Kolmogorov，KE rel-err 大幅退步（37–54%）。

Invariant tested:
    對 sensor file 的每個 sensor k:
        u_full[t, x_idx_k, y_idx_k] ≈ npz['u'][k, t]
        v_full[t, x_idx_k, y_idx_k] ≈ npz['v'][k, t]
    其中 x_idx_k, y_idx_k = argmin(|coord_k - dns_x|), argmin(|coord_k - dns_y|)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


DNS_PATH = Path("data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy")
SENSOR_DIR = Path("data/kolmogorov_sensors/re10000")

# 修完 bug 後，所有 sensor files 都應通過此測試
SENSOR_FILES = [
    "sensors_qrpivot_K100_N256_t0-5_si100",
    "sensors_random_K100_N256_t0-5_si100_seed42",
    "sensors_qrpivot_K100_N256_t0-5_si100_lesinformed",
    "sensors_qrpivot_K100_N256_t0-5_si100_les_n256",
    "sensors_qrpivot_K100_N256_t0-5_si100_les_n256_T50standalone",
]


@pytest.fixture(scope="module")
def dns_data():
    """載一次 DNS field + 座標。"""
    if not DNS_PATH.exists():
        pytest.skip(f"DNS file missing: {DNS_PATH}")
    d = np.load(DNS_PATH, allow_pickle=True).item()
    return {
        "u": np.asarray(d["u"], dtype=np.float64),
        "v": np.asarray(d["v"], dtype=np.float64),
        "x": np.asarray(d["x"], dtype=np.float64),
        "y": np.asarray(d["y"], dtype=np.float64),
    }


@pytest.mark.parametrize("file_stem", SENSOR_FILES)
def test_sensor_axis_convention(file_stem: str, dns_data):
    """Assert sensor NPZ value == u_full[:, x_idx, y_idx] for JSON coord (x, y)."""
    json_path = SENSOR_DIR / f"{file_stem}.json"
    # NPZ tag handling: 兩種命名方式
    candidates = [
        SENSOR_DIR / f"{file_stem}_dns_values.npz",
        SENSOR_DIR / f"{file_stem}.npz",
    ]
    npz_path = next((c for c in candidates if c.exists()), None)
    if json_path is None or not json_path.exists() or npz_path is None:
        pytest.skip(f"sensor files missing for {file_stem}")

    coords = np.asarray(json.loads(json_path.read_text())["selected_coordinates"],
                         dtype=np.float64)
    npz = np.load(npz_path, allow_pickle=True)
    sensor_u = npz["u"]  # [K, T]
    sensor_v = npz["v"]

    x_idx = np.argmin(np.abs(coords[:, 0:1] - dns_data["x"][None, :]), axis=1)
    y_idx = np.argmin(np.abs(coords[:, 1:2] - dns_data["y"][None, :]), axis=1)

    # Convention A (correct after fix): u_full[t, x_idx, y_idx]
    expected_u_A = dns_data["u"][:, x_idx, y_idx].T  # [K, T]
    expected_v_A = dns_data["v"][:, x_idx, y_idx].T

    err_u_A = np.abs(expected_u_A - sensor_u).max()
    err_v_A = np.abs(expected_v_A - sensor_v).max()

    # Convention B (buggy pre-fix): u_full[t, y_idx, x_idx] — should fail post-fix
    expected_u_B = dns_data["u"][:, y_idx, x_idx].T
    err_u_B = np.abs(expected_u_B - sensor_u).max()

    assert err_u_A < 1e-5, (
        f"{file_stem}: NPZ u value 不對齊 JSON 物理位置！"
        f"err_A (correct axis=1=x convention) = {err_u_A:.3e}, "
        f"err_B (swap convention) = {err_u_B:.3e}. "
        f"If err_B << err_A, file 是 buggy（row/col swap）；需重新生成。"
    )
    assert err_v_A < 1e-5, (
        f"{file_stem}: NPZ v value 不對齊 JSON 物理位置 (err={err_v_A:.3e})"
    )
