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
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pi_con.sensors import indices_from_coords, sample_series  # noqa: E402


RE10000_DNS = Path("data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy")
RE1000_DNS = Path("data/dns/kolmogorov_dns_fp64_etdrk4_Re1000_N128_T5_ds4.npy")
RE10000_DIR = Path("data/kolmogorov_sensors/re10000")
RE1000_DIR = Path("data/kolmogorov_sensors/re1000")

# 保留舊名，供既有 fixture / 呼叫端使用
DNS_PATH = RE10000_DNS
SENSOR_DIR = RE10000_DIR

# 修完 bug 後，所有 sensor files 都應通過此測試
SENSOR_FILES = [
    "sensors_qrpivot_K100_N256_t0-5_si100",
    "sensors_random_K100_N256_t0-5_si100_seed42",
    "sensors_qrpivot_K100_N256_t0-5_si100_lesinformed",
    "sensors_qrpivot_K100_N256_t0-5_si100_les_n256",
    "sensors_qrpivot_K100_N256_t0-5_si100_les_n256_T50standalone",
    "sensors_qrpivot_les_n256_T30dnsinit",  # EXP-296 DNS-init LES placement (research-only)
    "sensors_qrpivot_les_n256_T5dnsinit",   # EXP-297 t=5 DNS-init LES placement (research-only)
    "sensors_spacefill_K100_N256_t0-5_si100",  # EXP-298 FPS coverage-optimal placement (DNS-free)
    # EXP-300: LES_N128 T=15 with alpha=1.8 + hyperviscosity + r_scale=35, i.e. the
    # low-fidelity horizon with every other LES parameter aligned to the T=50 reference.
    "sensors_qrpivot_K100_N256_t0-5_si100_les_n128_T15_alpha1p8",
]


# (dns_path, sensor_dir, file_stem) —— 跨 Re 的 case。
# Re=1000 於 2026-07-16 納入：EXP-230 重跑為最終協定時新生的 LES-pivot 佈點。
# 舊 test 只涵蓋 re10000/，Re=1000 的 sensor 從未被此 invariant 守過。
CROSS_RE_CASES = [
    (RE1000_DNS, RE1000_DIR,
     "sensors_qrpivot_qrpivot_K100_N128_t0-5_les_n128_T100standalone"),
    (RE1000_DNS, RE1000_DIR,
     "sensors_qrpivot_K100_N128_t0-5"),  # 2026-03-29 legacy DNS-pivot
]

# Low-Re sensor-budget ladder (2026-07-25): FPS placement, K∈{100,50,10} × Re∈{1000,500,100}
# on matched-turnover DNS windows (§1.4). DNS in data/dns/cross_re/, N=128 (512→ds4).
_CROSS_RE_LADDER = {
    1000: (Path("data/dns/cross_re/kolmogorov_dns_Re1000_N128_T10p3_matched.npy"),
           Path("data/kolmogorov_sensors/re1000")),
    500:  (Path("data/dns/cross_re/kolmogorov_dns_Re500_N128_T17p28_matched.npy"),
           Path("data/kolmogorov_sensors/re500")),
    100:  (Path("data/dns/cross_re/kolmogorov_dns_Re100_N128_T55p8_matched.npy"),
           Path("data/kolmogorov_sensors/re100")),
}
for _re, (_dns, _dir) in _CROSS_RE_LADDER.items():
    for _K in (100, 50, 10):
        CROSS_RE_CASES.append(
            (_dns, _dir, f"sensors_spacefill_K{_K}_N128_Re{_re}_matched"))


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


@pytest.mark.parametrize("dns_path,sensor_dir,file_stem", CROSS_RE_CASES)
def test_sensor_axis_convention_cross_re(dns_path: Path, sensor_dir: Path, file_stem: str):
    """同一條 invariant，跨 Reynolds number。

    舊 test 的 DNS_PATH / SENSOR_DIR 寫死 Re=10000，因此 re1000/ 的 sensor 從未被守過 ——
    其中 sensors_qrpivot_K100_N128_t0-5 生成於 2026-03-29，早於 05-18 的 axis bug 修復。
    """
    if not dns_path.exists():
        pytest.skip(f"DNS file missing: {dns_path}")
    json_path = sensor_dir / f"{file_stem}.json"
    npz_path = next(
        (c for c in (sensor_dir / f"{file_stem}_dns_values.npz", sensor_dir / f"{file_stem}.npz")
         if c.exists()), None)
    if not json_path.exists() or npz_path is None:
        pytest.skip(f"sensor files missing for {file_stem}")

    d = np.load(dns_path, allow_pickle=True).item()
    u = np.asarray(d["u"], dtype=np.float64)
    v = np.asarray(d["v"], dtype=np.float64)
    xs = np.asarray(d["x"], dtype=np.float64)
    ys = np.asarray(d["y"], dtype=np.float64)

    coords = np.asarray(json.loads(json_path.read_text())["selected_coordinates"],
                        dtype=np.float64)
    npz = np.load(npz_path, allow_pickle=True)

    x_idx = np.argmin(np.abs(coords[:, 0:1] - xs[None, :]), axis=1)
    y_idx = np.argmin(np.abs(coords[:, 1:2] - ys[None, :]), axis=1)

    for name, field in (("u", u), ("v", v)):
        err_correct = np.abs(field[:, x_idx, y_idx].T - npz[name]).max()
        err_swap = np.abs(field[:, y_idx, x_idx].T - npz[name]).max()
        assert err_correct < 1e-5, (
            f"{file_stem}: NPZ {name} 不對齊 JSON 物理位置。"
            f"err_correct={err_correct:.3e}, err_swap={err_swap:.3e}. "
            f"若 err_swap << err_correct，此檔為 row/col swap，需重新生成。"
        )


# ---------------------------------------------------------------------------
# Directory sweep — the hand-maintained lists above enumerate stems, so a sensor
# file nobody remembered to add is a file nobody checks. That is not hypothetical:
# on 2026-07-31 `sensors_podpivot_K100_N256_t0-5_si100_les_n256_podpivot` was found
# stored at the transpose of its own coordinates while this suite passed 20/20,
# because its stem appeared in neither list. This sweep enumerates the directory
# instead, so enrolment is automatic and an unlisted artefact cannot hide.
# ---------------------------------------------------------------------------

_DNS_FOR_DIR = {
    RE10000_DIR: RE10000_DNS,
    RE1000_DIR: RE1000_DNS,
}


def _discover_sensor_pairs():
    """Every (dns, json, npz) triple present on disk, whatever its stem."""
    found = []
    for sensor_dir, dns_path in _DNS_FOR_DIR.items():
        if not sensor_dir.exists():
            continue
        for json_path in sorted(sensor_dir.glob("sensors_*.json")):
            stem = json_path.stem
            npz_path = next(
                (c for c in (sensor_dir / f"{stem}_dns_values.npz",
                             sensor_dir / f"{stem}.npz") if c.exists()),
                None,
            )
            if npz_path is not None:
                found.append(pytest.param(dns_path, json_path, npz_path, id=stem))
    return found


@pytest.mark.parametrize("dns_path,json_path,npz_path", _discover_sensor_pairs())
def test_every_sensor_file_on_disk_obeys_the_convention(dns_path, json_path, npz_path):
    if not dns_path.exists():
        pytest.skip(f"DNS file missing: {dns_path}")
    d = np.load(dns_path, allow_pickle=True).item()
    xs = np.asarray(d["x"], dtype=np.float64)
    ys = np.asarray(d["y"], dtype=np.float64)
    coords = np.asarray(
        json.loads(json_path.read_text())["selected_coordinates"], dtype=np.float64
    )
    npz = np.load(npz_path, allow_pickle=True)
    if npz["u"].shape[0] != coords.shape[0]:
        pytest.skip(
            f"{json_path.stem}: K mismatch json={coords.shape[0]} npz={npz['u'].shape[0]}"
        )

    x_idx, y_idx = indices_from_coords(coords, xs, ys)
    for name in ("u", "v"):
        field = np.asarray(d[name], dtype=np.float64)
        if field.shape[0] != npz[name].shape[1]:
            pytest.skip(f"{json_path.stem}: T mismatch for {name}")
        err_correct = np.abs(sample_series(field, x_idx, y_idx) - npz[name]).max()
        err_swap = np.abs(sample_series(field, y_idx, x_idx) - npz[name]).max()
        assert err_correct < 1e-5, (
            f"{json_path.stem}: NPZ {name} does not sit at the coordinates its JSON "
            f"records. err_correct={err_correct:.3e}, err_swap={err_swap:.3e}. "
            f"err_swap << err_correct means the file is stored transposed and must be "
            f"regenerated; see pi_con.sensors for the convention."
        )
