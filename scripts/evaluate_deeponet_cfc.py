"""Evaluate DeepONet + CfC smoke checkpoint on the Kolmogorov field.

What: 對指定 checkpoint 做最小場重建評估，輸出 RMSE / std / KE / Enstrophy / E(k_f)。
Why: 目前新骨架已能穩定訓練，但只看 training loss 不足以判斷場是否真的學起來。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch

from lnn_kolmogorov import create_lnn_model, load_lnn_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DeepONet + CfC checkpoint.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/deeponet_cfc_smoke.toml"),
        help="Path to model config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/deeponet-cfc-smoke/lnn_kolmogorov_final.pt"),
        help="Checkpoint to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/deeponet-cfc-eval"),
        help="Directory for summary output.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Evaluation device.",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def block_avg(field: np.ndarray) -> np.ndarray:
    n_half = field.shape[0] // 2
    return field.reshape(n_half, 2, n_half, 2).mean(axis=(1, 3))


def coarse_reference_grid(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """What: 產生與 2x2 block average 對齊的 coarse query grid。

    Why: `block_avg()` 代表的是 coarse cell 的平均值，不是原始 fine grid node。
         若仍在 `x[::2], y[::2]` 上 query，prediction 與 reference 會固定錯半格，
         系統性污染 RMSE、渦度與頻譜診斷。
    """
    if len(x) % 2 != 0 or len(y) % 2 != 0:
        raise ValueError(
            f"coarse_reference_grid 需要偶數長度 grid，收到 len(x)={len(x)}, len(y)={len(y)}"
        )
    x_coarse = 0.5 * (x[0::2] + x[1::2])
    y_coarse = 0.5 * (y[0::2] + y[1::2])
    return x_coarse.astype(np.float32), y_coarse.astype(np.float32)


def kinetic_energy(u: np.ndarray, v: np.ndarray) -> float:
    return float(0.5 * np.mean(u ** 2 + v ** 2))


def enstrophy_fd(u: np.ndarray, v: np.ndarray, dx: float) -> float:
    omega = vorticity_fd(u, v, dx)
    return float(0.5 * np.mean(omega ** 2))


def vorticity_fd(u: np.ndarray, v: np.ndarray, dx: float) -> np.ndarray:
    """What: 用中心差分近似 2D 渦度場。

    Why: 渦度是局部旋渦結構最直接的診斷量，適合保留在 evaluation 而非訓練 supervision。
    """
    dvdx = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dx)
    dudy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
    return dvdx - dudy


def divergence_fd(u: np.ndarray, v: np.ndarray, dx: float) -> np.ndarray:
    """What: 用中心差分近似 2D 不可壓縮條件殘差。"""
    dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
    dvdy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dx)
    return dudx + dvdy


def laplacian_periodic(field: np.ndarray, dx: float) -> np.ndarray:
    """What: 以 periodic stencil 計算 2D Laplacian。"""
    return (
        np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=1)
        + np.roll(field, 1, axis=1)
        - 4.0 * field
    ) / (dx**2)


def time_derivative_series(field_series: np.ndarray, time_vals: np.ndarray) -> np.ndarray:
    """What: 沿時間軸計算一階導數。"""
    edge_order = 2 if len(time_vals) >= 3 else 1
    return np.gradient(field_series, time_vals.astype(np.float64), axis=0, edge_order=edge_order)


def ns_residual_fields(
    u_series: np.ndarray,
    v_series: np.ndarray,
    p_series: np.ndarray,
    time_vals: np.ndarray,
    dx: float,
    re: float,
    k_forcing: float,
    forcing_amplitude: float,
    domain_length: float,
    y_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What: 在 evaluator coarse grid 上估計 primitive NS residual。

    Why: 現有的 RMSE / KE / spectrum 只能回答「像不像 DNS」；
         加上 NS residual 才能回答「場是否仍接近方程約束」。
    """
    du_dt = time_derivative_series(u_series, time_vals)
    dv_dt = time_derivative_series(v_series, time_vals)
    du_dx = (np.roll(u_series, -1, axis=1) - np.roll(u_series, 1, axis=1)) / (2 * dx)
    du_dy = (np.roll(u_series, -1, axis=2) - np.roll(u_series, 1, axis=2)) / (2 * dx)
    dv_dx = (np.roll(v_series, -1, axis=1) - np.roll(v_series, 1, axis=1)) / (2 * dx)
    dv_dy = (np.roll(v_series, -1, axis=2) - np.roll(v_series, 1, axis=2)) / (2 * dx)
    dp_dx = (np.roll(p_series, -1, axis=1) - np.roll(p_series, 1, axis=1)) / (2 * dx)
    dp_dy = (np.roll(p_series, -1, axis=2) - np.roll(p_series, 1, axis=2)) / (2 * dx)

    lap_u = np.stack([laplacian_periodic(frame, dx) for frame in u_series], axis=0)
    lap_v = np.stack([laplacian_periodic(frame, dx) for frame in v_series], axis=0)
    nu = 1.0 / float(re)
    forcing_wavenumber = (2.0 * np.pi * float(k_forcing)) / float(domain_length)
    forcing = float(forcing_amplitude) * np.sin(forcing_wavenumber * y_coords)[None, None, :]

    mom_u = du_dt + u_series * du_dx + v_series * du_dy + dp_dx - nu * lap_u - forcing
    mom_v = dv_dt + u_series * dv_dx + v_series * dv_dy + dp_dy - nu * lap_v
    cont = du_dx + dv_dy
    return mom_u, mom_v, cont


def energy_spectrum_1d(u: np.ndarray, v: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    n = u.shape[0]
    # ordinary wavenumber k（cycles/domain），與 k_f=2.0 的單位一致
    k1d = np.fft.fftfreq(n, d=dx)
    uh = np.fft.fft2(u) / n**2
    vh = np.fft.fft2(v) / n**2
    e2d = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2)
    kx, ky = np.meshgrid(k1d, k1d, indexing="ij")
    kk = np.sqrt(kx**2 + ky**2)
    edges = np.arange(0.5, n // 2 + 1.5, 1.0)
    e_k = np.zeros(len(edges) - 1, dtype=np.float64)
    for i in range(len(e_k)):
        mask = (kk >= edges[i]) & (kk < edges[i + 1])
        e_k[i] = np.sum(e2d[mask])
    return 0.5 * (edges[:-1] + edges[1:]), e_k


def spectrum_value_at_k(k_vals: np.ndarray, e_vals: np.ndarray, k_target: float) -> float:
    idx = int(np.argmin(np.abs(k_vals - k_target)))
    return float(e_vals[idx])


def summarize_time_local_metric(time_vals: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """What: 將時序指標壓縮成 early/mid/late 與 worst-time 摘要。"""
    if len(time_vals) != len(values):
        raise ValueError(
            f"time_vals 與 values 長度不一致：{len(time_vals)} vs {len(values)}"
        )
    idx_chunks = np.array_split(np.arange(len(values)), 3)

    def _chunk_mean(indices: np.ndarray) -> float:
        if len(indices) == 0:
            return float("nan")
        return float(np.mean(values[indices]))

    worst_idx = int(np.nanargmax(values))
    return {
        "mean": float(np.mean(values)),
        "early_mean": _chunk_mean(idx_chunks[0]),
        "mid_mean": _chunk_mean(idx_chunks[1]),
        "late_mean": _chunk_mean(idx_chunks[2]),
        "worst_time": float(time_vals[worst_idx]),
        "worst_value": float(values[worst_idx]),
    }


def compute_band_energies(k_vals: np.ndarray, e_vals: np.ndarray) -> dict[str, float]:
    """What: 將 1D spectrum 壓縮為 low/mid/high 三段 band energy。"""
    positive = k_vals > 0.0
    k_pos = k_vals[positive]
    e_pos = e_vals[positive]
    chunks = np.array_split(np.arange(len(k_pos)), 3)
    labels = ("low", "mid", "high")
    band_energies: dict[str, float] = {}
    for label, indices in zip(labels, chunks):
        band_energies[label] = float(np.sum(e_pos[indices])) if len(indices) > 0 else 0.0
    return band_energies


def validate_single_dataset_eval(cfg: dict[str, Any]) -> None:
    """What: 驗證 evaluator 僅面對單一 dataset config。

    Why: 訓練端支援多 dataset，但目前 evaluator 的輸出 schema 與圖像流程只針對單一 dataset。
         若靜默只取 index 0，會產生看似完整、實際只評第一組資料的錯誤結論。
    """
    lengths = {
        key: len(cfg.get(key, []))
        for key in ("sensor_jsons", "sensor_npzs", "dns_paths", "re_values")
    }
    unique_lengths = set(lengths.values())
    if unique_lengths != {1}:
        raise ValueError(
            "evaluate_deeponet_cfc.py 目前只支援單一 dataset；"
            f"收到 sensor_jsons={lengths['sensor_jsons']}, "
            f"sensor_npzs={lengths['sensor_npzs']}, "
            f"dns_paths={lengths['dns_paths']}, "
            f"re_values={lengths['re_values']}"
        )


def extract_model_state(checkpoint_payload: Any) -> dict[str, torch.Tensor]:
    """What: 從 checkpoint payload 萃取純模型 state_dict。

    Why: 評估腳本必須 fail fast。未知 checkpoint dict 若直接丟給 load_state_dict(strict=False)，
         可能只印 warning 就繼續產出 summary，這在評估流程中不可接受。
    """
    if not isinstance(checkpoint_payload, dict):
        raise ValueError(f"不支援的 checkpoint 格式：預期 dict，收到 {type(checkpoint_payload).__name__}")

    if "model_state_dict" in checkpoint_payload:
        state = checkpoint_payload["model_state_dict"]
    elif "model" in checkpoint_payload:
        state = checkpoint_payload["model"]
    else:
        tensor_like_values = all(torch.is_tensor(v) for v in checkpoint_payload.values())
        if tensor_like_values and checkpoint_payload:
            state = checkpoint_payload
        else:
            raise ValueError(
                "不支援的 checkpoint 格式：dict 內缺少 `model_state_dict` / `model`，"
                "且本體也不是純 state_dict。"
            )

    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint 中的模型權重為空或格式錯誤。")
    if not all(isinstance(k, str) for k in state):
        raise ValueError("checkpoint state_dict key 必須全部為字串。")
    if not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("checkpoint state_dict value 必須全部為 tensor。")
    return state


def load_model_weights_strict(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """What: 以嚴格模式載入模型權重。

    Why: 對評估腳本而言，missing/unexpected keys 不是 warning，而是直接代表結果不可相信。
    """
    lft_key = "query_decoder.log_fusion_temperature"
    if lft_key in state and state[lft_key].dim() == 0:
        state = dict(state)
        state[lft_key] = state[lft_key].unsqueeze(0)
    load_result = model.load_state_dict(state, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "checkpoint 與模型參數不一致："
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )


def forcing_mode_coeff_u(u: np.ndarray, y: np.ndarray, k_forcing: float) -> tuple[float, float]:
    """What: 擷取 x-平均後 u(y) 在 forcing mode 的複數 Fourier 係數。

    Why: 目前關鍵問題不是場完全崩潰，而是主模態是否被正確學到。
         直接量 amplitude / phase，比只看總能譜更容易判斷是沒學到還是相位錯。
    """
    u_bar = u.mean(axis=0)
    phase_arg = -2.0 * np.pi * float(k_forcing) * y
    basis = np.exp(1j * phase_arg)
    coeff = np.mean(u_bar * basis)
    return float(np.abs(coeff)), float(np.angle(coeff))


def plot_field_comparison(
    output_path: Path,
    u_ref: np.ndarray,
    u_pred: np.ndarray,
    v_ref: np.ndarray,
    v_pred: np.ndarray,
    t_val: float,
) -> None:
    """What: 輸出 DNS / LNN / Error 的場比較圖。

    Why: 直接看場結構與誤差分布，比單看 scalar 指標更能判斷是否只是振幅或相位偏移。
    """
    u_err = u_pred - u_ref
    v_err = v_pred - v_ref
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    u_lim = float(max(np.abs(u_ref).max(), np.abs(u_pred).max(), 1e-8))
    v_lim = float(max(np.abs(v_ref).max(), np.abs(v_pred).max(), 1e-8))
    ue_lim = float(max(np.abs(u_err).max(), 1e-8))
    ve_lim = float(max(np.abs(v_err).max(), 1e-8))

    panels = [
        (axes[0, 0], u_ref, "u DNS", "RdBu_r", -u_lim, u_lim),
        (axes[0, 1], u_pred, "u LNN", "RdBu_r", -u_lim, u_lim),
        (axes[0, 2], u_err, "u Error", "RdBu_r", -ue_lim, ue_lim),
        (axes[1, 0], v_ref, "v DNS", "RdBu_r", -v_lim, v_lim),
        (axes[1, 1], v_pred, "v LNN", "RdBu_r", -v_lim, v_lim),
        (axes[1, 2], v_err, "v Error", "RdBu_r", -ve_lim, ve_lim),
    ]
    for ax, field, title, cmap, vmin, vmax in panels:
        im = ax.imshow(field.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"DNS vs LNN vs Error at t={t_val:.2f}")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_vorticity_comparison(
    output_path: Path,
    omega_ref: np.ndarray,
    omega_pred: np.ndarray,
    t_val: float,
) -> None:
    """What: 輸出最後時間點的渦度 DNS / LNN / Error 圖。

    Why: 在 sparse-data 主線下，渦度應保留作診斷工具，用來判斷局部旋渦結構是否對齊。
    """
    omega_err = omega_pred - omega_ref
    om_lim = float(max(np.abs(omega_ref).max(), np.abs(omega_pred).max(), 1e-8))
    err_lim = float(max(np.abs(omega_err).max(), 1e-8))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    panels = [
        (axes[0], omega_ref, "Vorticity DNS", -om_lim, om_lim),
        (axes[1], omega_pred, "Vorticity LNN", -om_lim, om_lim),
        (axes[2], omega_err, "Vorticity Error", -err_lim, err_lim),
    ]
    for ax, field, title, vmin, vmax in panels:
        im = ax.imshow(field.T, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Vorticity Comparison at t={t_val:.2f}")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_energy_spectrum(
    output_path: Path,
    k_ref: np.ndarray,
    e_ref: np.ndarray,
    k_pred: np.ndarray,
    e_pred: np.ndarray,
    k_forcing: float,
) -> None:
    """What: 輸出最後時間點的一維能譜比較圖。

    Why: 能譜最直接反映主模態與高頻結構是否被正確重建。
    """
    mask_ref = e_ref > 0.0
    mask_pred = e_pred > 0.0
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.loglog(k_ref[mask_ref], e_ref[mask_ref], marker="o", label="DNS")
    ax.loglog(k_pred[mask_pred], e_pred[mask_pred], marker="s", label="LNN")
    ax.axvline(k_forcing, color="black", linestyle="--", linewidth=1.0, label="k_f")
    ax.set_title("Energy Spectrum")
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("Energy E(k)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_vs_time(
    output_path: Path,
    time_vals: np.ndarray,
    ref_vals: np.ndarray,
    pred_vals: np.ndarray,
    title: str,
    y_label: str,
) -> None:
    """What: 輸出隨時間變化的整體指標比較圖。

    Why: 檢查模型是否只在單一時間點看起來合理，或整段時序都維持相近偏差。
    """
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(time_vals, ref_vals, marker="o", label="DNS")
    ax.plot(time_vals, pred_vals, marker="s", label="LNN")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_series_collection(
    output_path: Path,
    time_vals: np.ndarray,
    series_map: dict[str, np.ndarray],
    title: str,
    y_label: str,
    yscale: str = "linear",
) -> None:
    """What: 將多條時序指標畫在同一張圖上。"""
    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    for label, values in series_map.items():
        ax.plot(time_vals, values, marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)
    if yscale != "linear":
        ax.set_yscale(yscale)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_uv_error_vs_time(
    output_path: Path,
    time_vals: np.ndarray,
    u_err: np.ndarray,
    v_err: np.ndarray,
) -> None:
    """What: 輸出 u / v RMSE 隨時間的變化圖。

    Why: 單看平均 RMSE 會掩蓋模型是否只在前段或後段時間失真，
         需要明確看到兩個速度分量的誤差如何隨時間演化。
    """
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(time_vals, u_err, marker="o", label="u RMSE")
    ax.plot(time_vals, v_err, marker="s", label="v RMSE")
    ax.set_title("Velocity Error vs Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("RMSE")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_mode_vs_time(
    output_path: Path,
    time_vals: np.ndarray,
    ref_vals: np.ndarray,
    pred_vals: np.ndarray,
    title: str,
    y_label: str,
) -> None:
    """What: 輸出 forcing mode amplitude / phase 的時間演化比較圖。"""
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(time_vals, ref_vals, marker="o", label="DNS")
    ax.plot(time_vals, pred_vals, marker="s", label="LNN")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_lnn_config(args.config)
    validate_single_dataset_eval(cfg)
    device = choose_device(args.device)
    model = create_lnn_model(cfg).to(device)
    checkpoint_payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = extract_model_state(checkpoint_payload)
    load_model_weights_strict(model, state)
    model.eval()

    sensor_json = json.loads(Path(cfg["sensor_jsons"][0]).read_text(encoding="utf-8"))
    sensor_npz = np.load(cfg["sensor_npzs"][0])
    dns = np.load(cfg["dns_paths"][0], allow_pickle=True).item()

    sensor_pos = np.array(sensor_json["selected_coordinates"], dtype=np.float32)
    requested = tuple(cfg.get("observed_sensor_channels", ["u", "v"]))
    observed_fields = []
    for key in requested:
        if key in sensor_npz:
            observed_fields.append(sensor_npz[key].astype(np.float32))
    if not observed_fields:
        raise ValueError(f"sensor_npz 不含指定感測器通道 {requested}。")
    sensor_vals = np.stack(observed_fields, axis=2)  # [K, T, C_obs]
    sensor_mean = sensor_vals.mean(axis=(0, 1), keepdims=True)
    sensor_std = np.maximum(sensor_vals.std(axis=(0, 1), keepdims=True), 1.0e-6)
    sensor_vals = ((sensor_vals - sensor_mean) / sensor_std).astype(np.float32)
    sensor_time = sensor_npz["time"].astype(np.float32)

    x_g, y_g = coarse_reference_grid(
        dns["x"].astype(np.float32),
        dns["y"].astype(np.float32),
    )
    xx, yy = np.meshgrid(x_g, y_g, indexing="ij")
    xy_flat = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    xy_t = torch.tensor(xy_flat, dtype=torch.float32, device=device)
    batch = 8192

    sv_t = torch.tensor(sensor_vals.transpose(1, 0, 2), dtype=torch.float32, device=device)
    sp_t = torch.tensor(sensor_pos, dtype=torch.float32, device=device)
    st_t = torch.tensor(sensor_time, dtype=torch.float32, device=device)
    re_value = float(cfg.get("re_values", [1000.0])[0])
    re_norm = float((re_value - 5500.0) / 4000.0)

    with torch.no_grad():
        h_states, s_time = model.encode(sv_t, sp_t, re_norm, st_t)

    def query_field(comp_idx: int, t_val: float) -> np.ndarray:
        parts = []
        with torch.no_grad():
            for start in range(0, xy_t.shape[0], batch):
                end = min(start + batch, xy_t.shape[0])
                xy_b = xy_t[start:end]
                bs = end - start
                t_b = torch.full((bs,), t_val, dtype=torch.float32, device=device)
                c_b = torch.full((bs,), comp_idx, dtype=torch.long, device=device)
                out = model.query_decoder(xy_b, t_b, c_b, h_states, s_time, sp_t)
                parts.append(out.squeeze(1).cpu().numpy())
        return np.concatenate(parts).reshape(len(x_g), len(y_g))

    dx = float(x_g[1] - x_g[0]) if len(x_g) > 1 else 1.0
    u_pred_series = []
    v_pred_series = []
    p_pred_series = []
    u_ref_series = []
    v_ref_series = []
    p_ref_series = []
    for t_val in sensor_time:
        dns_idx = int(np.argmin(np.abs(dns["time"].astype(np.float64) - float(t_val))))
        u_pred_series.append(query_field(0, float(t_val)).astype(np.float32))
        v_pred_series.append(query_field(1, float(t_val)).astype(np.float32))
        p_pred_series.append(query_field(2, float(t_val)).astype(np.float32))
        u_ref_series.append(block_avg(dns["u"][dns_idx].astype(np.float32)))
        v_ref_series.append(block_avg(dns["v"][dns_idx].astype(np.float32)))
        p_ref_series.append(block_avg(dns["p"][dns_idx].astype(np.float32)))

    u_pred_arr = np.stack(u_pred_series, axis=0)
    v_pred_arr = np.stack(v_pred_series, axis=0)
    p_pred_arr = np.stack(p_pred_series, axis=0)
    u_ref_arr = np.stack(u_ref_series, axis=0)
    v_ref_arr = np.stack(v_ref_series, axis=0)
    p_ref_arr = np.stack(p_ref_series, axis=0)

    u_rmse = np.sqrt(np.mean((u_pred_arr - u_ref_arr) ** 2, axis=(1, 2)))
    v_rmse = np.sqrt(np.mean((v_pred_arr - v_ref_arr) ** 2, axis=(1, 2)))
    pred_std_u = u_pred_arr.std(axis=(1, 2))
    pred_std_v = v_pred_arr.std(axis=(1, 2))
    ke_pred_series = 0.5 * np.mean(u_pred_arr**2 + v_pred_arr**2, axis=(1, 2))
    ke_ref_series = 0.5 * np.mean(u_ref_arr**2 + v_ref_arr**2, axis=(1, 2))
    ke_rel_err = np.abs(ke_pred_series - ke_ref_series) / np.maximum(ke_ref_series, 1.0e-12)

    omega_pred_arr = np.stack([vorticity_fd(u_pred_arr[i], v_pred_arr[i], dx) for i in range(len(sensor_time))], axis=0)
    omega_ref_arr = np.stack([vorticity_fd(u_ref_arr[i], v_ref_arr[i], dx) for i in range(len(sensor_time))], axis=0)
    omega_rmse = np.sqrt(np.mean((omega_pred_arr - omega_ref_arr) ** 2, axis=(1, 2)))
    ens_pred_series = 0.5 * np.mean(omega_pred_arr**2, axis=(1, 2))
    ens_ref_series = 0.5 * np.mean(omega_ref_arr**2, axis=(1, 2))
    ens_rel_err = np.abs(ens_pred_series - ens_ref_series) / np.maximum(ens_ref_series, 1.0e-12)

    div_pred_arr = np.stack([divergence_fd(u_pred_arr[i], v_pred_arr[i], dx) for i in range(len(sensor_time))], axis=0)
    div_ref_arr = np.stack([divergence_fd(u_ref_arr[i], v_ref_arr[i], dx) for i in range(len(sensor_time))], axis=0)
    div_l2_pred = np.sqrt(np.mean(div_pred_arr**2, axis=(1, 2)))
    div_linf_pred = np.max(np.abs(div_pred_arr), axis=(1, 2))
    div_l2_ref = np.sqrt(np.mean(div_ref_arr**2, axis=(1, 2)))
    div_linf_ref = np.max(np.abs(div_ref_arr), axis=(1, 2))

    mom_u_pred, mom_v_pred, cont_pred = ns_residual_fields(
        u_series=u_pred_arr,
        v_series=v_pred_arr,
        p_series=p_pred_arr,
        time_vals=sensor_time,
        dx=dx,
        re=re_value,
        k_forcing=float(cfg["kolmogorov_k_f"]),
        forcing_amplitude=float(cfg.get("kolmogorov_A", 0.1)),
        domain_length=float(cfg.get("domain_length", 1.0)),
        y_coords=y_g.astype(np.float64),
    )
    mom_u_ref, mom_v_ref, cont_ref = ns_residual_fields(
        u_series=u_ref_arr,
        v_series=v_ref_arr,
        p_series=p_ref_arr,
        time_vals=sensor_time,
        dx=dx,
        re=re_value,
        k_forcing=float(cfg["kolmogorov_k_f"]),
        forcing_amplitude=float(cfg.get("kolmogorov_A", 0.1)),
        domain_length=float(cfg.get("domain_length", 1.0)),
        y_coords=y_g.astype(np.float64),
    )
    ns_u_rms_pred = np.sqrt(np.mean(mom_u_pred**2, axis=(1, 2)))
    ns_v_rms_pred = np.sqrt(np.mean(mom_v_pred**2, axis=(1, 2)))
    ns_cont_rms_pred = np.sqrt(np.mean(cont_pred**2, axis=(1, 2)))
    ns_u_rms_ref = np.sqrt(np.mean(mom_u_ref**2, axis=(1, 2)))
    ns_v_rms_ref = np.sqrt(np.mean(mom_v_ref**2, axis=(1, 2)))
    ns_cont_rms_ref = np.sqrt(np.mean(cont_ref**2, axis=(1, 2)))

    kf_amp_ref_series = []
    kf_amp_pred_series = []
    kf_phase_ref_series = []
    kf_phase_pred_series = []
    band_rel_err_series = {"low": [], "mid": [], "high": []}
    summary_steps: list[dict[str, float]] = []
    k_ref = e_ref = k_pred = e_pred = None
    for idx, t_val in enumerate(sensor_time):
        amp_ref, phase_ref = forcing_mode_coeff_u(u_ref_arr[idx], y_g, float(cfg["kolmogorov_k_f"]))
        amp_pred, phase_pred = forcing_mode_coeff_u(u_pred_arr[idx], y_g, float(cfg["kolmogorov_k_f"]))
        k_ref_i, e_ref_i = energy_spectrum_1d(u_ref_arr[idx], v_ref_arr[idx], dx)
        k_pred_i, e_pred_i = energy_spectrum_1d(u_pred_arr[idx], v_pred_arr[idx], dx)
        bands_ref = compute_band_energies(k_ref_i, e_ref_i)
        bands_pred = compute_band_energies(k_pred_i, e_pred_i)
        for band in ("low", "mid", "high"):
            band_rel_err_series[band].append(
                abs(bands_pred[band] - bands_ref[band]) / max(bands_ref[band], 1.0e-12)
            )
        kf_amp_ref_series.append(amp_ref)
        kf_amp_pred_series.append(amp_pred)
        kf_phase_ref_series.append(phase_ref)
        kf_phase_pred_series.append(phase_pred)
        summary_steps.append(
            {
                "time": float(t_val),
                "u_rmse": float(u_rmse[idx]),
                "v_rmse": float(v_rmse[idx]),
                "omega_rmse": float(omega_rmse[idx]),
                "u_std": float(pred_std_u[idx]),
                "v_std": float(pred_std_v[idx]),
                "ke_rel_err": float(ke_rel_err[idx]),
                "ens_rel_err": float(ens_rel_err[idx]),
                "div_l2": float(div_l2_pred[idx]),
                "div_linf": float(div_linf_pred[idx]),
                "ns_u_rms": float(ns_u_rms_pred[idx]),
                "ns_v_rms": float(ns_v_rms_pred[idx]),
                "ns_cont_rms": float(ns_cont_rms_pred[idx]),
                "band_rel_err_low": float(band_rel_err_series["low"][-1]),
                "band_rel_err_mid": float(band_rel_err_series["mid"][-1]),
                "band_rel_err_high": float(band_rel_err_series["high"][-1]),
                "kf_amp_ref": amp_ref,
                "kf_amp_pred": amp_pred,
                "kf_phase_ref": phase_ref,
                "kf_phase_pred": phase_pred,
            }
        )
        k_ref, e_ref, k_pred, e_pred = k_ref_i, e_ref_i, k_pred_i, e_pred_i

    kf_amp_ref_series = np.asarray(kf_amp_ref_series)
    kf_amp_pred_series = np.asarray(kf_amp_pred_series)
    kf_phase_ref_series = np.asarray(kf_phase_ref_series)
    kf_phase_pred_series = np.asarray(kf_phase_pred_series)
    band_rel_err_series = {k: np.asarray(v) for k, v in band_rel_err_series.items()}

    assert k_ref is not None and e_ref is not None and k_pred is not None and e_pred is not None
    t_last = float(sensor_time[-1])
    u_last = u_pred_arr[-1]
    v_last = v_pred_arr[-1]
    u_ref_last = u_ref_arr[-1]
    v_ref_last = v_ref_arr[-1]
    omega_last = omega_pred_arr[-1]
    omega_ref_last = omega_ref_arr[-1]
    ek_ratio = spectrum_value_at_k(k_pred, e_pred, float(cfg["kolmogorov_k_f"])) / max(
        spectrum_value_at_k(k_ref, e_ref, float(cfg["kolmogorov_k_f"])),
        1e-12,
    )

    plot_field_comparison(
        output_dir / f"field_comparison_t{int(round(t_last))}.png",
        u_ref_last,
        u_last,
        v_ref_last,
        v_last,
        t_last,
    )
    plot_energy_spectrum(
        output_dir / "energy_spectrum.png",
        k_ref,
        e_ref,
        k_pred,
        e_pred,
        float(cfg["kolmogorov_k_f"]),
    )
    plot_vorticity_comparison(
        output_dir / f"vorticity_comparison_t{int(round(t_last))}.png",
        omega_ref_last,
        omega_last,
        t_last,
    )
    plot_metric_vs_time(
        output_dir / "kinetic_energy_vs_time.png",
        sensor_time,
        np.asarray(ke_ref_series),
        np.asarray(ke_pred_series),
        title="Kinetic Energy vs Time",
        y_label="Kinetic Energy",
    )
    plot_metric_vs_time(
        output_dir / "enstrophy_vs_time.png",
        sensor_time,
        np.asarray(ens_ref_series),
        np.asarray(ens_pred_series),
        title="Enstrophy vs Time",
        y_label="Enstrophy",
    )
    plot_uv_error_vs_time(
        output_dir / "uv_error_vs_time.png",
        sensor_time,
        u_rmse,
        v_rmse,
    )
    plot_mode_vs_time(
        output_dir / "kf_mode_amplitude_vs_time.png",
        sensor_time,
        kf_amp_ref_series,
        kf_amp_pred_series,
        title=f"Forcing Mode Amplitude (k={float(cfg['kolmogorov_k_f']):.1f})",
        y_label="Amplitude",
    )
    plot_mode_vs_time(
        output_dir / "kf_mode_phase_vs_time.png",
        sensor_time,
        np.unwrap(kf_phase_ref_series),
        np.unwrap(kf_phase_pred_series),
        title=f"Forcing Mode Phase (k={float(cfg['kolmogorov_k_f']):.1f})",
        y_label="Phase [rad]",
    )
    plot_series_collection(
        output_dir / "vorticity_error_vs_time.png",
        sensor_time,
        {"Omega RMSE": omega_rmse},
        title="Vorticity Error vs Time",
        y_label="RMSE",
    )
    plot_series_collection(
        output_dir / "divergence_vs_time.png",
        sensor_time,
        {"DNS L2": div_l2_ref, "LNN L2": div_l2_pred},
        title="Divergence Residual vs Time",
        y_label="L2",
        yscale="log",
    )
    plot_series_collection(
        output_dir / "ns_residual_vs_time.png",
        sensor_time,
        {
            "DNS NS-u": ns_u_rms_ref,
            "LNN NS-u": ns_u_rms_pred,
            "DNS NS-v": ns_v_rms_ref,
            "LNN NS-v": ns_v_rms_pred,
            "DNS Cont": ns_cont_rms_ref,
            "LNN Cont": ns_cont_rms_pred,
        },
        title="NS Residual vs Time",
        y_label="RMS",
        yscale="log",
    )
    plot_series_collection(
        output_dir / "band_energy_rel_error_vs_time.png",
        sensor_time,
        {
            "Low-k": band_rel_err_series["low"],
            "Mid-k": band_rel_err_series["mid"],
            "High-k": band_rel_err_series["high"],
        },
        title="Band Energy Relative Error vs Time",
        y_label="Relative Error",
    )

    time_local = {
        "u_rmse": summarize_time_local_metric(sensor_time, u_rmse),
        "v_rmse": summarize_time_local_metric(sensor_time, v_rmse),
        "omega_rmse": summarize_time_local_metric(sensor_time, omega_rmse),
        "ke_rel_err": summarize_time_local_metric(sensor_time, ke_rel_err),
        "div_l2": summarize_time_local_metric(sensor_time, div_l2_pred),
        "ns_u_rms": summarize_time_local_metric(sensor_time, ns_u_rms_pred),
        "ns_v_rms": summarize_time_local_metric(sensor_time, ns_v_rms_pred),
        "ns_cont_rms": summarize_time_local_metric(sensor_time, ns_cont_rms_pred),
        "band_rel_err_low": summarize_time_local_metric(sensor_time, band_rel_err_series["low"]),
        "band_rel_err_mid": summarize_time_local_metric(sensor_time, band_rel_err_series["mid"]),
        "band_rel_err_high": summarize_time_local_metric(sensor_time, band_rel_err_series["high"]),
    }

    summary = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "u_rmse_mean": float(np.mean(u_rmse)),
        "v_rmse_mean": float(np.mean(v_rmse)),
        "omega_rmse_mean": float(np.mean(omega_rmse)),
        "u_std_mean": float(np.mean(pred_std_u)),
        "v_std_mean": float(np.mean(pred_std_v)),
        "ke_rel_err_mean": float(np.mean(ke_rel_err)),
        "ens_rel_err_mean": float(np.mean(ens_rel_err)),
        "div_l2_mean": float(np.mean(div_l2_pred)),
        "div_linf_mean": float(np.mean(div_linf_pred)),
        "div_ref_l2_mean": float(np.mean(div_l2_ref)),
        "div_ref_linf_mean": float(np.mean(div_linf_ref)),
        "ns_u_rms_mean": float(np.mean(ns_u_rms_pred)),
        "ns_v_rms_mean": float(np.mean(ns_v_rms_pred)),
        "ns_cont_rms_mean": float(np.mean(ns_cont_rms_pred)),
        "ns_u_rms_ref_mean": float(np.mean(ns_u_rms_ref)),
        "ns_v_rms_ref_mean": float(np.mean(ns_v_rms_ref)),
        "ns_cont_rms_ref_mean": float(np.mean(ns_cont_rms_ref)),
        "ek_ratio_kf_last": float(ek_ratio),
        "band_energy_rel_err_mean": {
            band: float(np.mean(values)) for band, values in band_rel_err_series.items()
        },
        "band_energy_rel_err_last": {
            band: float(values[-1]) for band, values in band_rel_err_series.items()
        },
        "kf_amp_ref_last": float(kf_amp_ref_series[-1]),
        "kf_amp_pred_last": float(kf_amp_pred_series[-1]),
        "kf_amp_ratio_last": float(kf_amp_pred_series[-1] / max(kf_amp_ref_series[-1], 1e-12)),
        "kf_phase_ref_last": float(kf_phase_ref_series[-1]),
        "kf_phase_pred_last": float(kf_phase_pred_series[-1]),
        "kf_phase_err_last": float(np.angle(np.exp(1j * (kf_phase_pred_series[-1] - kf_phase_ref_series[-1])))),
        "time_local": time_local,
        "steps": summary_steps,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== DeepONet+CfC Evaluation ===")
    print(f"checkpoint: {args.checkpoint.resolve()}")
    print(f"u RMSE mean = {summary['u_rmse_mean']:.4e}")
    print(f"v RMSE mean = {summary['v_rmse_mean']:.4e}")
    print(f"u std mean  = {summary['u_std_mean']:.4e}")
    print(f"v std mean  = {summary['v_std_mean']:.4e}")
    print(f"omega RMSE mean = {summary['omega_rmse_mean']:.4e}")
    print(f"KE rel-err mean  = {summary['ke_rel_err_mean']:.4e}")
    print(f"Ens rel-err mean = {summary['ens_rel_err_mean']:.4e}")
    print(f"div L2 mean = {summary['div_l2_mean']:.4e}  (DNS {summary['div_ref_l2_mean']:.4e})")
    print(
        "NS residual RMS mean = "
        f"u {summary['ns_u_rms_mean']:.4e} / "
        f"v {summary['ns_v_rms_mean']:.4e} / "
        f"cont {summary['ns_cont_rms_mean']:.4e}"
    )
    print(f"E(k_f={float(cfg['kolmogorov_k_f']):.1f}) ratio @ last = {summary['ek_ratio_kf_last']:.4e}")
    print(f"k_f amplitude ratio @ last = {summary['kf_amp_ratio_last']:.4e}")
    print(f"k_f phase error @ last = {summary['kf_phase_err_last']:.4e} rad")
    print(f"summary_json: {output_dir / 'summary.json'}")
    print(f"field_comparison: {output_dir / f'field_comparison_t{int(round(t_last))}.png'}")
    print(f"vorticity_comparison: {output_dir / f'vorticity_comparison_t{int(round(t_last))}.png'}")
    print(f"energy_spectrum: {output_dir / 'energy_spectrum.png'}")
    print(f"kinetic_energy_plot: {output_dir / 'kinetic_energy_vs_time.png'}")
    print(f"enstrophy_plot: {output_dir / 'enstrophy_vs_time.png'}")
    print(f"uv_error_plot: {output_dir / 'uv_error_vs_time.png'}")
    print(f"vorticity_error_plot: {output_dir / 'vorticity_error_vs_time.png'}")
    print(f"divergence_plot: {output_dir / 'divergence_vs_time.png'}")
    print(f"ns_residual_plot: {output_dir / 'ns_residual_vs_time.png'}")
    print(f"band_error_plot: {output_dir / 'band_energy_rel_error_vs_time.png'}")


if __name__ == "__main__":
    main()
