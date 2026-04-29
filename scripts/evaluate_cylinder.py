"""evaluate_cylinder.py — Cylinder wake 流場重建品質評估。

What: 載入 checkpoint，在完整 128×256 格點上查詢模型，與 Arrow DNS 對照。
Why:  cylinder 非週期非均勻格，不能沿用 Kolmogorov evaluator 的 block_avg/periodic-FD/FFT 設計。

輸出（artifacts_dir/cylinder-eval/）：
  summary.json          — KE rel-err、u/v RMSE、divergence
  field_comparison_tXX.png  — 4 個時刻的 DNS/LNN/Error 場比較
  vorticity_tXX.png         — 渦度場比較
  ke_vs_time.png
  uv_error_vs_time.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pyarrow as pa
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pi_lnn import create_lnn_model, load_lnn_config

# ── 常數（與 CylinderDataset 一致）────────────────────────────────────────────
RE_MEAN = 7000.0
RE_STD  = 2500.0
BODY_THRESHOLD = 1e-4
QUERY_BATCH = 8192


# ── 工具函式 ─────────────────────────────────────────────────────────────────

def choose_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_arrow_fields(shard_path: str) -> dict:
    """讀取 Arrow shard，回傳 u/v/p 全場 + grid metadata。"""
    with open(shard_path, "rb") as f:
        reader = pa.ipc.open_stream(f)
        batch = reader.read_next_batch()
    row = {n: batch.column(n)[0].as_py() for n in batch.schema.names}
    T, H, W = row["shape_t"], row["shape_h"], row["shape_w"]
    xH, xW = row["x_shape_h"], row["x_shape_w"]
    t_len  = row["t_shape"]
    u = np.frombuffer(row["u"], dtype=np.float32).reshape(T, H, W)
    v = np.frombuffer(row["v"], dtype=np.float32).reshape(T, H, W)
    p = np.frombuffer(row["p"], dtype=np.float32).reshape(T, H, W)
    x = np.frombuffer(row["x"], dtype=np.float64).reshape(xH, xW)
    y = np.frombuffer(row["y"], dtype=np.float64).reshape(xH, xW)
    t = np.frombuffer(row["t"], dtype=np.float64)[:t_len]
    return dict(u=u, v=v, p=p, x=x, y=y, t=t, T=T, H=H, W=W)


def detect_body(u_all: np.ndarray) -> np.ndarray:
    """[T, H, W] → body mask [H, W] bool。"""
    speed_median = np.median(np.abs(u_all), axis=0)
    return speed_median < BODY_THRESHOLD


def vorticity_fd(u: np.ndarray, v: np.ndarray,
                 x_1d: np.ndarray, y_1d: np.ndarray) -> np.ndarray:
    """非週期 FD 渦度 ω = ∂v/∂x - ∂u/∂y。"""
    dvdx = np.gradient(v, x_1d, axis=1)
    dudy = np.gradient(u, y_1d, axis=0)
    return dvdx - dudy


def divergence_fd(u: np.ndarray, v: np.ndarray,
                  x_1d: np.ndarray, y_1d: np.ndarray) -> np.ndarray:
    """非週期 FD 散度 ∂u/∂x + ∂v/∂y。"""
    dudx = np.gradient(u, x_1d, axis=1)
    dvdy = np.gradient(v, y_1d, axis=0)
    return dudx + dvdy


def extract_model_state(payload) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(f"unsupported checkpoint type: {type(payload)}")
    if "model_state_dict" in payload:
        return payload["model_state_dict"]
    if "model" in payload:
        return payload["model"]
    if all(torch.is_tensor(v) for v in payload.values()):
        return payload
    raise ValueError("checkpoint 缺少 model_state_dict / model key")


# ── 繪圖 ────────────────────────────────────────────────────────────────────

def plot_field(output_path: Path,
               x2d: np.ndarray, y2d: np.ndarray,
               body: np.ndarray,
               u_ref: np.ndarray, u_pred: np.ndarray,
               v_ref: np.ndarray, v_pred: np.ndarray,
               t_val: float) -> None:
    def _mask(f):
        out = f.astype(float).copy()
        out[body] = np.nan
        return out
    ur, up, ue = _mask(u_ref), _mask(u_pred), _mask(u_pred - u_ref)
    vr, vp, ve = _mask(v_ref), _mask(v_pred), _mask(v_pred - v_ref)

    fig, axes = plt.subplots(2, 3, figsize=(16, 7), constrained_layout=True)
    fig.suptitle(f"DNS vs LNN at t={t_val:.2f} s")

    def _show(ax, field, title):
        vmax = np.nanpercentile(np.abs(field), 99)
        if vmax < 1e-8:
            vmax = 1e-8
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        ax.pcolormesh(x2d, y2d, field, cmap="RdBu_r", norm=norm,
                      shading="auto", rasterized=True)
        ax.set_aspect("equal"); ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")

    _show(axes[0, 0], ur, "u DNS"); _show(axes[0, 1], up, "u LNN"); _show(axes[0, 2], ue, "u Error")
    _show(axes[1, 0], vr, "v DNS"); _show(axes[1, 1], vp, "v LNN"); _show(axes[1, 2], ve, "v Error")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_vorticity(output_path: Path,
                   x2d: np.ndarray, y2d: np.ndarray,
                   body: np.ndarray,
                   omega_ref: np.ndarray, omega_pred: np.ndarray,
                   t_val: float) -> None:
    def _mask(f):
        out = f.astype(float).copy(); out[body] = np.nan; return out
    or_, op, oe = _mask(omega_ref), _mask(omega_pred), _mask(omega_pred - omega_ref)

    vmax = max(np.nanpercentile(np.abs(or_), 98), 1e-8)
    emax = max(np.nanpercentile(np.abs(oe), 98), 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    fig.suptitle(f"Vorticity at t={t_val:.2f} s")
    for ax, f, title, vm in zip(axes, [or_, op, oe],
                                 ["DNS", "LNN", "Error"],
                                 [vmax, vmax, emax]):
        norm = TwoSlopeNorm(vmin=-vm, vcenter=0, vmax=vm)
        ax.pcolormesh(x2d, y2d, f, cmap="RdBu_r", norm=norm,
                      shading="auto", rasterized=True)
        ax.set_aspect("equal"); ax.set_title(f"Vorticity {title}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_series(output_path: Path, time_vals: np.ndarray,
                series: dict[str, np.ndarray], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for label, vals in series.items():
        ax.plot(time_vals, vals, label=label)
    ax.set_title(title); ax.set_xlabel("t (s)"); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3); ax.legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── メイン ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",     type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device",     default="mps")
    p.add_argument("--eval-stride", type=int, default=5,
                   help="每隔幾個 sensor time step 評估一次（預設 5，共 ~40 幀）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_lnn_config(args.config)

    out_dir = args.output_dir or (Path(cfg["artifacts_dir"]) / "cylinder-eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"device: {device}")

    # ── 模型載入 ────────────────────────────────────────────────────────────
    model = create_lnn_model(cfg).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = extract_model_state(payload)
    lft_key = "query_decoder.log_fusion_temperature"
    if lft_key in state and state[lft_key].dim() == 0:
        state = dict(state)
        state[lft_key] = state[lft_key].unsqueeze(0)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"checkpoint 不一致: missing={result.missing_keys}, "
                           f"unexpected={result.unexpected_keys}")
    model.eval()
    print(f"checkpoint: {args.checkpoint}")

    # ── 資料載入 ─────────────────────────────────────────────────────────────
    shard = cfg["arrow_shards"][0]
    dns = load_arrow_fields(shard)
    H, W = dns["H"], dns["W"]
    x2d, y2d = dns["x"], dns["y"]
    t_all   = dns["t"]

    # 正規化座標（與 CylinderDataset 一致）
    x_lo, x_hi = float(x2d.min()), float(x2d.max())
    y_lo, y_hi = float(y2d.min()), float(y2d.max())
    x2d_n = ((x2d - x_lo) / (x_hi - x_lo)).astype(np.float32)
    y2d_n = ((y2d - y_lo) / (y_hi - y_lo)).astype(np.float32)

    x_1d_n = x2d_n[0, :]   # [W] 沿列方向
    y_1d_n = y2d_n[:, 0]   # [H] 沿行方向

    body = detect_body(dns["u"])   # [H, W]
    fluid_mask = ~body

    # sensor 資料
    sensor_json = json.loads(Path(cfg["sensor_jsons"][0]).read_text())
    sensor_npz  = np.load(cfg["sensor_npzs"][0])
    coords_raw  = np.array(sensor_json["selected_coordinates"], dtype=np.float64)
    sx_n = ((coords_raw[:, 0] - x_lo) / (x_hi - x_lo)).astype(np.float32)
    sy_n = ((coords_raw[:, 1] - y_lo) / (y_hi - y_lo)).astype(np.float32)
    sensor_pos_norm = np.stack([sx_n, sy_n], axis=1)   # [K, 2]

    # sensor time（subsampled）
    subsample = int(cfg.get("sensor_subsample", 1))
    t_idx_sub = np.arange(0, len(t_all), max(1, subsample))
    sensor_time = t_all[t_idx_sub].astype(np.float32)

    # sensor 觀測值，正規化（與 CylinderDataset 一致）
    channels = cfg.get("observed_sensor_channels", ["u", "v"])
    obs = np.stack([sensor_npz[ch].astype(np.float32) for ch in channels], axis=2)  # [K, T_full, C]
    obs_sub = obs[:, t_idx_sub, :]   # [K, T_sub, C]
    s_mean = obs_sub.mean(axis=(0, 1), keepdims=True)
    s_std  = np.maximum(obs_sub.std(axis=(0, 1), keepdims=True), 1e-6)
    obs_norm = ((obs_sub - s_mean) / s_std).astype(np.float32)   # [K, T_sub, C]

    re_value = float(cfg["re_values"][0])
    re_norm  = float((re_value - RE_MEAN) / RE_STD)
    print(f"Re={re_value:.0f}  re_norm={re_norm:.4f}  T_sub={len(sensor_time)}")

    # ── Encode ───────────────────────────────────────────────────────────────
    sv_t  = torch.tensor(obs_norm.transpose(1, 0, 2), dtype=torch.float32, device=device)  # [T, K, C]
    sp_t  = torch.tensor(sensor_pos_norm,              dtype=torch.float32, device=device)  # [K, 2]
    st_t  = torch.tensor(sensor_time,                  dtype=torch.float32, device=device)  # [T]

    with torch.no_grad():
        h_states, s_time = model.encode(sv_t, sp_t, re_norm, st_t)

    # ── 全場查詢函式 ─────────────────────────────────────────────────────────
    xy_flat = np.stack([x2d_n.ravel(), y2d_n.ravel()], axis=1).astype(np.float32)  # [H*W, 2]
    xy_t = torch.tensor(xy_flat, device=device)

    def query_field(comp_idx: int, t_val: float) -> np.ndarray:
        parts = []
        with torch.no_grad():
            for s in range(0, xy_t.shape[0], QUERY_BATCH):
                e = min(s + QUERY_BATCH, xy_t.shape[0])
                n = e - s
                xy_b = xy_t[s:e]
                t_b  = torch.full((n,), t_val, dtype=torch.float32, device=device)
                c_b  = torch.full((n,), comp_idx, dtype=torch.long,  device=device)
                out  = model.query_decoder(xy_b, t_b, c_b, h_states, s_time, sp_t)
                parts.append(out.squeeze(1).cpu().numpy())
        return np.concatenate(parts).reshape(H, W)

    # ── 評估時間步（依 eval_stride）──────────────────────────────────────────
    eval_tidx = np.arange(0, len(sensor_time), args.eval_stride)
    eval_times = sensor_time[eval_tidx]
    print(f"evaluating {len(eval_times)} time steps …")

    u_pred_list, v_pred_list = [], []
    u_ref_list,  v_ref_list  = [], []
    for i, (ti, tv) in enumerate(zip(eval_tidx, eval_times)):
        if i % 5 == 0:
            print(f"  step {i}/{len(eval_times)}  t={tv:.2f}s", flush=True)
        dns_idx = t_idx_sub[ti]
        u_pred_list.append(query_field(0, float(tv)))
        v_pred_list.append(query_field(1, float(tv)))
        u_ref_list.append(dns["u"][dns_idx])
        v_ref_list.append(dns["v"][dns_idx])

    u_pred = np.stack(u_pred_list)   # [N_eval, H, W]
    v_pred = np.stack(v_pred_list)
    u_ref  = np.stack(u_ref_list)
    v_ref  = np.stack(v_ref_list)

    # ── 指標（流體域）────────────────────────────────────────────────────────
    fm = fluid_mask  # [H, W]
    n_fluid = fm.sum()

    def fluid_rmse(pred, ref):
        return np.sqrt(np.mean((pred[:, fm] - ref[:, fm]) ** 2, axis=1))

    def fluid_ke(u_, v_):
        return 0.5 * np.mean(u_[:, fm] ** 2 + v_[:, fm] ** 2, axis=1)

    u_rmse = fluid_rmse(u_pred, u_ref)
    v_rmse = fluid_rmse(v_pred, v_ref)
    ke_pred = fluid_ke(u_pred, v_pred)
    ke_ref  = fluid_ke(u_ref,  v_ref)
    ke_rel  = np.abs(ke_pred - ke_ref) / np.maximum(ke_ref, 1e-12)

    # 渦度
    omega_pred_list, omega_ref_list = [], []
    for i in range(len(eval_times)):
        omega_pred_list.append(vorticity_fd(u_pred[i], v_pred[i], x_1d_n, y_1d_n))
        omega_ref_list.append( vorticity_fd(u_ref[i],  v_ref[i],  x_1d_n, y_1d_n))
    omega_pred_arr = np.stack(omega_pred_list)
    omega_ref_arr  = np.stack(omega_ref_list)
    omega_rmse = fluid_rmse(omega_pred_arr, omega_ref_arr)

    # divergence
    div_pred_list = []
    for i in range(len(eval_times)):
        div_pred_list.append(divergence_fd(u_pred[i], v_pred[i], x_1d_n, y_1d_n))
    div_pred = np.stack(div_pred_list)
    div_l2 = np.sqrt(np.mean(div_pred[:, fm] ** 2, axis=1))

    # ── 圖像：4 個代表時刻 ───────────────────────────────────────────────────
    target_times = [1.0, 5.0, 10.0, 18.0]
    for tt in target_times:
        ti_closest = int(np.argmin(np.abs(eval_times - tt)))
        tv = eval_times[ti_closest]
        tag = int(round(tv))

        plot_field(out_dir / f"field_comparison_t{tag:02d}.png",
                   x2d, y2d, body,
                   u_ref[ti_closest], u_pred[ti_closest],
                   v_ref[ti_closest], v_pred[ti_closest], tv)
        plot_vorticity(out_dir / f"vorticity_t{tag:02d}.png",
                       x2d, y2d, body,
                       omega_ref_arr[ti_closest], omega_pred_arr[ti_closest], tv)
        print(f"  saved figures at t={tv:.2f}s")

    # ── 時序圖 ────────────────────────────────────────────────────────────────
    plot_series(out_dir / "ke_vs_time.png", eval_times,
                {"DNS": ke_ref, "LNN": ke_pred},
                "Kinetic Energy vs Time", "KE (m²/s²)")

    plot_series(out_dir / "uv_error_vs_time.png", eval_times,
                {"u RMSE": u_rmse, "v RMSE": v_rmse},
                "Velocity RMSE vs Time", "RMSE (m/s)")

    plot_series(out_dir / "omega_error_vs_time.png", eval_times,
                {"ω RMSE": omega_rmse},
                "Vorticity RMSE vs Time", "RMSE (1/s)")

    plot_series(out_dir / "divergence_vs_time.png", eval_times,
                {"div L2": div_l2},
                "Divergence L2 vs Time", "L2")

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "config":     str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "re":         re_value,
        "n_eval_steps": int(len(eval_times)),
        "u_rmse_mean":  float(np.mean(u_rmse)),
        "v_rmse_mean":  float(np.mean(v_rmse)),
        "omega_rmse_mean": float(np.mean(omega_rmse)),
        "ke_rel_err_mean": float(np.mean(ke_rel)),
        "ke_rel_err_late": float(np.mean(ke_rel[len(ke_rel) * 2 // 3:])),
        "div_l2_mean": float(np.mean(div_l2)),
        "ke_ref_mean": float(np.mean(ke_ref)),
        "ke_pred_mean": float(np.mean(ke_pred)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Cylinder Evaluation ===")
    print(f"checkpoint: {args.checkpoint}")
    print(f"u  RMSE mean = {summary['u_rmse_mean']:.4e}")
    print(f"v  RMSE mean = {summary['v_rmse_mean']:.4e}")
    print(f"ω  RMSE mean = {summary['omega_rmse_mean']:.4e}")
    print(f"KE rel-err mean  = {summary['ke_rel_err_mean']:.4e}")
    print(f"KE rel-err late  = {summary['ke_rel_err_late']:.4e}")
    print(f"div L2 mean      = {summary['div_l2_mean']:.4e}")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
