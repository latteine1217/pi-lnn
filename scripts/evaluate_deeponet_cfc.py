"""Evaluate DeepONet + CfC smoke checkpoint on the Kolmogorov field.

What: 對指定 checkpoint 做最小場重建評估，輸出 RMSE / std / KE / Enstrophy / E(k_f)。
Why: 目前新骨架已能穩定訓練，但只看 training loss 不足以判斷場是否真的學起來。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pi_onet.lnn_kolmogorov import create_lnn_model, load_lnn_config


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
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def block_avg(field: np.ndarray) -> np.ndarray:
    n_half = field.shape[0] // 2
    return field.reshape(n_half, 2, n_half, 2).mean(axis=(1, 3))


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


def energy_spectrum_1d(u: np.ndarray, v: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    n = u.shape[0]
    k1d = np.fft.fftfreq(n, d=dx / (2 * np.pi))
    uh = np.fft.fft2(u) / n**2
    vh = np.fft.fft2(v) / n**2
    e2d = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2)
    kx, ky = np.meshgrid(k1d, k1d, indexing="ij")
    kk = np.sqrt(kx**2 + ky**2)
    edges = np.arange(0.5, n // 2 + 1.5, 1.0)
    e_k = np.zeros(len(edges) - 1, dtype=np.float64)
    for i in range(len(e_k)):
        mask = (kk >= edges[i]) & (kk < edges[i + 1])
        e_k[i] = np.sum(e2d[mask]) * n**2
    return 0.5 * (edges[:-1] + edges[1:]), e_k


def spectrum_value_at_k(k_vals: np.ndarray, e_vals: np.ndarray, k_target: float) -> float:
    idx = int(np.argmin(np.abs(k_vals - k_target)))
    return float(e_vals[idx])


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
    device = choose_device(args.device)
    model = create_lnn_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    # 相容舊 checkpoint：log_fusion_temperature 由 0-dim 改為 shape (1,)
    lft_key = "query_decoder.log_fusion_temperature"
    if lft_key in state and state[lft_key].dim() == 0:
        state[lft_key] = state[lft_key].unsqueeze(0)
    model.load_state_dict(state, strict=False)
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

    x_g = dns["x"][::2].astype(np.float32)
    y_g = dns["y"][::2].astype(np.float32)
    xx, yy = np.meshgrid(x_g, y_g, indexing="ij")
    xy_flat = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    xy_t = torch.tensor(xy_flat, dtype=torch.float32, device=device)
    batch = 8192

    sv_t = torch.tensor(sensor_vals.transpose(1, 0, 2), dtype=torch.float32, device=device)
    sp_t = torch.tensor(sensor_pos, dtype=torch.float32, device=device)
    st_t = torch.tensor(sensor_time, dtype=torch.float32, device=device)
    re_norm = float((1000.0 - 5500.0) / 4000.0)

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
    u_rmse = []
    v_rmse = []
    ke_rel_err = []
    ens_rel_err = []
    pred_std_u = []
    pred_std_v = []
    ke_ref_series = []
    ke_pred_series = []
    ens_ref_series = []
    ens_pred_series = []
    kf_amp_ref_series = []
    kf_amp_pred_series = []
    kf_phase_ref_series = []
    kf_phase_pred_series = []

    summary_steps: list[dict[str, float]] = []
    for t_val in sensor_time:
        dns_idx = int(np.argmin(np.abs(dns["time"].astype(np.float32) - t_val)))
        u_pred = query_field(0, float(t_val))
        v_pred = query_field(1, float(t_val))
        u_ref = block_avg(dns["u"][dns_idx].astype(np.float32))
        v_ref = block_avg(dns["v"][dns_idx].astype(np.float32))

        u_err = float(np.sqrt(np.mean((u_pred - u_ref) ** 2)))
        v_err = float(np.sqrt(np.mean((v_pred - v_ref) ** 2)))
        ke_pred = kinetic_energy(u_pred, v_pred)
        ke_ref = kinetic_energy(u_ref, v_ref)
        ens_pred = enstrophy_fd(u_pred, v_pred, dx)
        ens_ref = enstrophy_fd(u_ref, v_ref, dx)
        amp_ref, phase_ref = forcing_mode_coeff_u(u_ref, y_g, float(cfg["kolmogorov_k_f"]))
        amp_pred, phase_pred = forcing_mode_coeff_u(u_pred, y_g, float(cfg["kolmogorov_k_f"]))
        ke_err = abs(ke_pred - ke_ref) / max(ke_ref, 1e-12)
        ens_err = abs(ens_pred - ens_ref) / max(ens_ref, 1e-12)

        u_rmse.append(u_err)
        v_rmse.append(v_err)
        ke_rel_err.append(float(ke_err))
        ens_rel_err.append(float(ens_err))
        pred_std_u.append(float(u_pred.std()))
        pred_std_v.append(float(v_pred.std()))
        ke_ref_series.append(float(ke_ref))
        ke_pred_series.append(float(ke_pred))
        ens_ref_series.append(float(ens_ref))
        ens_pred_series.append(float(ens_pred))
        kf_amp_ref_series.append(amp_ref)
        kf_amp_pred_series.append(amp_pred)
        kf_phase_ref_series.append(phase_ref)
        kf_phase_pred_series.append(phase_pred)
        summary_steps.append(
            {
                "time": float(t_val),
                "u_rmse": u_err,
                "v_rmse": v_err,
                "u_std": float(u_pred.std()),
                "v_std": float(v_pred.std()),
                "ke_rel_err": float(ke_err),
                "ens_rel_err": float(ens_err),
                "kf_amp_ref": amp_ref,
                "kf_amp_pred": amp_pred,
                "kf_phase_ref": phase_ref,
                "kf_phase_pred": phase_pred,
            }
        )

    t_last = float(sensor_time[-1])
    dns_idx_last = int(np.argmin(np.abs(dns["time"].astype(np.float32) - t_last)))
    u_last = query_field(0, t_last)
    v_last = query_field(1, t_last)
    u_ref_last = block_avg(dns["u"][dns_idx_last].astype(np.float32))
    v_ref_last = block_avg(dns["v"][dns_idx_last].astype(np.float32))
    omega_last = vorticity_fd(u_last, v_last, dx)
    omega_ref_last = vorticity_fd(u_ref_last, v_ref_last, dx)
    k_pred, e_pred = energy_spectrum_1d(u_last, v_last, dx)
    k_ref, e_ref = energy_spectrum_1d(u_ref_last, v_ref_last, dx)
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
        np.asarray(u_rmse),
        np.asarray(v_rmse),
    )
    plot_mode_vs_time(
        output_dir / "kf_mode_amplitude_vs_time.png",
        sensor_time,
        np.asarray(kf_amp_ref_series),
        np.asarray(kf_amp_pred_series),
        title=f"Forcing Mode Amplitude (k={float(cfg['kolmogorov_k_f']):.1f})",
        y_label="Amplitude",
    )
    plot_mode_vs_time(
        output_dir / "kf_mode_phase_vs_time.png",
        sensor_time,
        np.unwrap(np.asarray(kf_phase_ref_series)),
        np.unwrap(np.asarray(kf_phase_pred_series)),
        title=f"Forcing Mode Phase (k={float(cfg['kolmogorov_k_f']):.1f})",
        y_label="Phase [rad]",
    )

    summary = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "u_rmse_mean": float(np.mean(u_rmse)),
        "v_rmse_mean": float(np.mean(v_rmse)),
        "u_std_mean": float(np.mean(pred_std_u)),
        "v_std_mean": float(np.mean(pred_std_v)),
        "ke_rel_err_mean": float(np.mean(ke_rel_err)),
        "ens_rel_err_mean": float(np.mean(ens_rel_err)),
        "ek_ratio_kf_last": float(ek_ratio),
        "kf_amp_ref_last": float(kf_amp_ref_series[-1]),
        "kf_amp_pred_last": float(kf_amp_pred_series[-1]),
        "kf_amp_ratio_last": float(kf_amp_pred_series[-1] / max(kf_amp_ref_series[-1], 1e-12)),
        "kf_phase_ref_last": float(kf_phase_ref_series[-1]),
        "kf_phase_pred_last": float(kf_phase_pred_series[-1]),
        "kf_phase_err_last": float(np.angle(np.exp(1j * (kf_phase_pred_series[-1] - kf_phase_ref_series[-1])))),
        "steps": summary_steps,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== DeepONet+CfC Evaluation ===")
    print(f"checkpoint: {args.checkpoint.resolve()}")
    print(f"u RMSE mean = {summary['u_rmse_mean']:.4e}")
    print(f"v RMSE mean = {summary['v_rmse_mean']:.4e}")
    print(f"u std mean  = {summary['u_std_mean']:.4e}")
    print(f"v std mean  = {summary['v_std_mean']:.4e}")
    print(f"KE rel-err mean  = {summary['ke_rel_err_mean']:.4e}")
    print(f"Ens rel-err mean = {summary['ens_rel_err_mean']:.4e}")
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


if __name__ == "__main__":
    main()
