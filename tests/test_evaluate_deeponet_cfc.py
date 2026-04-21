from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import evaluate_deeponet_cfc as eval_mod
from lnn_kolmogorov import create_lnn_model, load_lnn_config


def test_energy_spectrum_matches_kinetic_energy() -> None:
    n = 128
    dx = 1.0 / n
    y = np.arange(n, dtype=np.float64) * dx
    u = np.sin(2.0 * np.pi * 2.0 * y)[None, :].repeat(n, axis=0)
    v = np.zeros_like(u)

    _, e_k = eval_mod.energy_spectrum_1d(u, v, dx)
    kinetic_energy = 0.5 * np.mean(u**2 + v**2)

    assert np.isclose(e_k.sum(), kinetic_energy, rtol=1e-6, atol=1e-8)


def test_coarse_reference_grid_uses_cell_centers() -> None:
    x = np.arange(8, dtype=np.float64) / 8.0
    y = np.arange(8, dtype=np.float64) / 8.0

    x_coarse, y_coarse = eval_mod.coarse_reference_grid(x, y)

    np.testing.assert_allclose(x_coarse, (x[0::2] + x[1::2]) / 2.0)
    np.testing.assert_allclose(y_coarse, (y[0::2] + y[1::2]) / 2.0)


def test_load_lnn_config_resolves_relative_paths_from_config_dir(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiment"
    config_dir = experiment_root / "configs"
    data_dir = experiment_root / "data"
    config_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    (data_dir / "dns.npy").write_bytes(b"stub")
    (data_dir / "sensor.json").write_text("{}", encoding="utf-8")
    (data_dir / "sensor.npz").write_bytes(b"stub")

    config_path = config_dir / "exp.toml"
    config_path.write_text(
        """
[train]
sensor_jsons = ["../data/sensor.json"]
sensor_npzs = ["../data/sensor.npz"]
dns_paths = ["../data/dns.npy"]
artifacts_dir = "../artifacts/run"
""".strip(),
        encoding="utf-8",
    )

    old_cwd = Path.cwd()
    try:
        os_dir = tmp_path / "other"
        os_dir.mkdir()
        import os

        os.chdir(os_dir)
        cfg = load_lnn_config(config_path)
    finally:
        os.chdir(old_cwd)

    assert cfg["sensor_jsons"] == [str((data_dir / "sensor.json").resolve())]
    assert cfg["sensor_npzs"] == [str((data_dir / "sensor.npz").resolve())]
    assert cfg["dns_paths"] == [str((data_dir / "dns.npy").resolve())]
    assert cfg["artifacts_dir"] == str((experiment_root / "artifacts" / "run").resolve())


def test_extract_model_state_rejects_unknown_checkpoint_dict() -> None:
    with pytest.raises(ValueError, match="不支援的 checkpoint 格式"):
        eval_mod.extract_model_state({"foo": np.array([1.0])})


def test_load_model_weights_strict_rejects_missing_keys() -> None:
    cfg = load_lnn_config(REPO_ROOT / "configs/exp_048_re10000_xlarge_10k.toml")
    model = create_lnn_model(cfg)
    state = copy.deepcopy(model.state_dict())
    state.pop(next(iter(state)))

    with pytest.raises(RuntimeError, match="checkpoint 與模型參數不一致"):
        eval_mod.load_model_weights_strict(model, state)


def test_validate_single_dataset_eval_rejects_multi_dataset_config() -> None:
    cfg = {
        "sensor_jsons": ["a.json", "b.json"],
        "sensor_npzs": ["a.npz", "b.npz"],
        "dns_paths": ["a.npy", "b.npy"],
        "re_values": [1000.0, 10000.0],
    }

    with pytest.raises(ValueError, match="只支援單一 dataset"):
        eval_mod.validate_single_dataset_eval(cfg)


def test_choose_device_auto_prefers_cuda_over_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(eval_mod.torch.backends.mps, "is_available", lambda: True)

    assert eval_mod.choose_device("auto").type == "cuda"


def test_divergence_fd_is_zero_for_incompressible_mode() -> None:
    n = 64
    dx = 1.0 / n
    x = np.arange(n, dtype=np.float64) * dx
    y = np.arange(n, dtype=np.float64) * dx
    xx, yy = np.meshgrid(x, y, indexing="ij")
    u = np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    v = -np.cos(2.0 * np.pi * xx) * np.sin(2.0 * np.pi * yy)

    div = eval_mod.divergence_fd(u, v, dx)

    assert np.max(np.abs(div)) < 1.0e-12


def test_ns_residual_fields_zero_for_steady_kolmogorov_solution() -> None:
    n = 128
    dx = 1.0 / n
    t = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    y = np.arange(n, dtype=np.float64) * dx
    k_f = 2.0
    re = 1000.0
    amplitude = 0.1
    kw = 2.0 * np.pi * k_f
    discrete_laplace_eigenvalue = 4.0 * np.sin(kw * dx / 2.0) ** 2 / dx**2
    u_profile = (amplitude / ((1.0 / re) * discrete_laplace_eigenvalue)) * np.sin(kw * y)
    u = np.broadcast_to(u_profile, (len(t), n, n)).copy()
    v = np.zeros_like(u)
    p = np.zeros_like(u)

    mom_u, mom_v, cont = eval_mod.ns_residual_fields(
        u_series=u,
        v_series=v,
        p_series=p,
        time_vals=t,
        dx=dx,
        re=re,
        k_forcing=k_f,
        forcing_amplitude=amplitude,
        domain_length=1.0,
        y_coords=y,
    )

    assert np.max(np.abs(mom_u)) < 1.0e-9
    assert np.max(np.abs(mom_v)) < 1.0e-12
    assert np.max(np.abs(cont)) < 1.0e-12


def test_summarize_time_local_metric_tracks_early_late_and_worst() -> None:
    time_vals = np.linspace(0.0, 5.0, 6, dtype=np.float64)
    values = np.array([1.0, 2.0, 3.0, 10.0, 5.0, 4.0], dtype=np.float64)

    summary = eval_mod.summarize_time_local_metric(time_vals, values)

    assert summary["early_mean"] == pytest.approx(1.5)
    assert summary["mid_mean"] == pytest.approx(6.5)
    assert summary["late_mean"] == pytest.approx(4.5)
    assert summary["worst_time"] == pytest.approx(3.0)
    assert summary["worst_value"] == pytest.approx(10.0)


def test_compute_band_energies_partitions_total_energy() -> None:
    k_vals = np.arange(1.0, 10.0, dtype=np.float64)
    e_vals = np.arange(1.0, 10.0, dtype=np.float64)

    bands = eval_mod.compute_band_energies(k_vals, e_vals)

    assert set(bands) == {"low", "mid", "high"}
    assert sum(bands.values()) == pytest.approx(float(e_vals.sum()))
