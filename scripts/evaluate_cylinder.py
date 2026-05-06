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
from pi_lnn import create_lnn_model, find_dns_time_idx, load_lnn_config
# Why: evaluator 必須跟訓練端用 *完全相同* 的 stats / body mask / SDF / bc_scale。
#      重複實作（hardcode RE_MEAN/STD、自寫 detect_body、重算 SDF）會 silent drift。
#      改成 import dataset class，evaluator 內部重建一次拿 stats。
#      find_dns_time_idx 也從 pi_lnn 共用 module import（避免兩 evaluator 字面重複 → drift）。
from cylinder_dataset import CylinderDataset

# 訓練端 hardcoded train_ratio=0.8（pi_lnn/training.py:74）。evaluator 必須對齊。
TRAIN_RATIO_FALLBACK = 0.8
QUERY_BATCH = 8192


def validate_single_dataset_eval(cfg: dict) -> None:
    """What: 驗證 evaluator 僅面對單一 dataset config（與 deeponet evaluator 對稱）。

    Why: 訓練端支援 multi-Re，但 cylinder evaluator 的輸出 schema 只針對單一 Re。
         若 silent index 0，會產生看似完整、實際只評第一組 Re 的錯誤結論。
    """
    lengths = {
        key: len(cfg.get(key, []))
        for key in ("sensor_jsons", "sensor_npzs", "arrow_shards", "re_values")
    }
    unique = set(lengths.values())
    if unique != {1}:
        raise ValueError(
            "evaluate_cylinder.py 目前只支援單一 dataset；"
            f"收到 sensor_jsons={lengths['sensor_jsons']}, "
            f"sensor_npzs={lengths['sensor_npzs']}, "
            f"arrow_shards={lengths['arrow_shards']}, "
            f"re_values={lengths['re_values']}"
        )


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
    # CRIT-2: fail-fast 對 grid shape mismatch（u/v/p shape vs x/y shape 必須相同）。
    # Why: 之前若 (xH, xW) != (H, W)，evaluator 會在後段 SDF assert 才炸（已 reshape 過 →
    #      不可恢復狀態）。在這裡早 raise 訊息更清楚。
    if (xH, xW) != (H, W):
        raise ValueError(
            f"Arrow shard grid shape mismatch: u/v/p shape ({H},{W}) "
            f"vs x/y shape ({xH},{xW}). 同一 shard 必須一致。"
        )
    u = np.frombuffer(row["u"], dtype=np.float32).reshape(T, H, W)
    v = np.frombuffer(row["v"], dtype=np.float32).reshape(T, H, W)
    p = np.frombuffer(row["p"], dtype=np.float32).reshape(T, H, W)
    x = np.frombuffer(row["x"], dtype=np.float64).reshape(xH, xW)
    y = np.frombuffer(row["y"], dtype=np.float64).reshape(xH, xW)
    t = np.frombuffer(row["t"], dtype=np.float64)[:t_len]
    return dict(u=u, v=v, p=p, x=x, y=y, t=t, T=T, H=H, W=W)


def _body_masks_from_dataset(ds: CylinderDataset, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """從 dataset 的 _sdf_grid 取得 body / fluid mask（避免重算造成 silent drift）。

    Why: distance_transform_edt 對 body cell 給 0、fluid cell 給正距離；
         所以 body == (sdf_grid == 0) 與 dataset _detect_body 完全一致。
    """
    if not hasattr(ds, "_sdf_grid"):
        raise RuntimeError("dataset 缺 _sdf_grid（需要 CylinderDataset）")
    sdf = ds._sdf_grid
    if sdf.shape != (H, W):
        raise ValueError(f"dataset SDF shape {sdf.shape} 與 DNS grid ({H},{W}) 不符")
    body = (sdf == 0.0)
    return body, ~body


def _assert_axis_monotonic(x_1d: np.ndarray, name: str) -> None:
    """IMP-4: assert axis is strictly monotonic increasing。

    Why: np.gradient 接 nonuniform spacing 在 axis 非 monotonic 時 silent 給錯結果。
         cylinder grid 來自 RealPDEBench shard，通常 monotonic，但 evaluator 不該假設。
    """
    diffs = np.diff(np.asarray(x_1d))
    if not np.all(diffs > 0):
        raise ValueError(
            f"axis '{name}' 必須 strictly monotonic increasing；"
            f"min diff={float(diffs.min()):.4e}（非正即破壞 np.gradient 正確性）"
        )


def vorticity_fd(u: np.ndarray, v: np.ndarray,
                 x_1d: np.ndarray, y_1d: np.ndarray) -> np.ndarray:
    """非週期 FD 渦度 ω = ∂v/∂x - ∂u/∂y。

    Convention: u/v shape [H, W]; axis 0 沿 y (rows)，axis 1 沿 x (cols)。
    """
    dvdx = np.gradient(v, x_1d, axis=1)
    dudy = np.gradient(u, y_1d, axis=0)
    return dvdx - dudy


def divergence_fd(u: np.ndarray, v: np.ndarray,
                  x_1d: np.ndarray, y_1d: np.ndarray) -> np.ndarray:
    """非週期 FD 散度 ∂u/∂x + ∂v/∂y。

    Convention: u/v shape [H, W]; axis 0 沿 y (rows)，axis 1 沿 x (cols)。
    """
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
    p.add_argument(
        "--apply-denormalization", action="store_true",
        help=(
            "Apply output denormalization (raw * std + mean) before metrics."
            " 預設 False — model 學 physical raw output，不該再套 denorm（double-scale）。"
            " 此 flag 留作向後相容；正常 use case 都不該開。"
        ),
    )
    p.add_argument(
        "--legacy-checkpoint", action="store_true",
        help=(
            "[DEPRECATED] 之前指「跳過 denorm」是 opt-in；現在 default 已是跳過 denorm。"
            " 此 flag 保留為 no-op 以維持向後相容。"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_lnn_config(args.config)
    validate_single_dataset_eval(cfg)   # R4: fail-fast 對 multi-Re cfg

    out_dir = args.output_dir or (Path(cfg["artifacts_dir"]) / "cylinder-eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"device: {device}")

    # ── 模型載入 ────────────────────────────────────────────────────────────
    model = create_lnn_model(cfg).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # 偵測 schedulefree mode：mid-step ckpt 存 train mode (x_t) 不是 inference-ready。
    # Why: x_t 是 training iterate，比 y_t (Polyak averaged) noise 大，inference quality
    #      差 5-30%。final.pt 才存 y_t (eval mode)。
    if isinstance(payload, dict):
        _sf_mode = payload.get("schedulefree_mode", None)
        if _sf_mode == "train":
            print(
                f"  [WARN] checkpoint 含 schedulefree_mode='train' (x_t, 給 resume 用)；"
                f"\n         inference quality 比 final.pt (y_t, eval mode) 差 5-30%。"
                f"\n         若要 best inference 結果，請改用 .../lnn_kolmogorov_final.pt"
            )
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

    # ── Flag 互斥檢查 ────────────────────────────────────────────────────────
    if args.apply_denormalization and args.legacy_checkpoint:
        print(
            "  [WARN] --apply-denormalization 與 --legacy-checkpoint 同時設定；"
            " --apply-denormalization 強制走 raw*std+mean，"
            "--legacy-checkpoint (deprecated) 在此情況下無作用。"
        )

    # ── DNS 全場（給 evaluator 用，訓練不需要）────────────────────────────────
    shard = cfg["arrow_shards"][0]
    dns = load_arrow_fields(shard)
    H, W = dns["H"], dns["W"]
    x2d, y2d = dns["x"], dns["y"]

    # 正規化座標（與 CylinderDataset 一致）
    x_lo, x_hi = float(x2d.min()), float(x2d.max())
    y_lo, y_hi = float(y2d.min()), float(y2d.max())
    x2d_n = ((x2d - x_lo) / (x_hi - x_lo)).astype(np.float32)
    y2d_n = ((y2d - y_lo) / (y_hi - y_lo)).astype(np.float32)

    # Physical 座標：FD 算 divergence/vorticity 必須用物理單位 (m)，
    # 否則 ∂u/∂x_norm 缺 1/Lx chain rule，cylinder Lx=0.322 會放大 3.1×。
    x_1d_phys = x2d[0, :].astype(np.float32)
    y_1d_phys = y2d[:, 0].astype(np.float32)
    # IMP-4: monotonic assert — np.gradient 在 nonuniform 但 monotonic 軸正確；
    #        若 axis 非 monotonic 會 silent 算錯。Production cylinder grid 應 monotonic。
    _assert_axis_monotonic(x_1d_phys, name="x")
    _assert_axis_monotonic(y_1d_phys, name="y")

    # ── 重建 dataset 拿訓練端 stats / body mask / SDF / bc_scale ─────────────
    # Why: reviewer 報告 C1/C2 — evaluator 之前自己重算 sensor stats 與 body mask，
    #      跟 dataset (即訓練端) silent drift：
    #        * dataset 用 train-only stats（80% subset），evaluator 用 full-data stats
    #        * dataset _detect_body 用 |u|+|v|，evaluator detect_body 只看 |u|
    #      → encoder input distribution / hard BC gate scale 跟訓練不一致 → 系統性偏。
    #      改成單一真相源：直接 import CylinderDataset 重建一次，evaluator 用其 stats。
    ds_seed = int(cfg.get("seed", 42))
    ds = CylinderDataset(
        sensor_json=cfg["sensor_jsons"][0],
        sensor_npz=cfg["sensor_npzs"][0],
        arrow_shard=cfg["arrow_shards"][0],
        re_value=float(cfg["re_values"][0]),
        observed_channel_names=tuple(cfg.get("observed_sensor_channels", ["u", "v"])),
        train_ratio=TRAIN_RATIO_FALLBACK,   # 跟 pi_lnn/training.py:74 hardcode 一致
        seed=ds_seed,
        sensor_subsample=int(cfg.get("sensor_subsample", 1)),
    )

    # 從 dataset 取已 normalize 過的 sensor data + 訓練端 stats（單一真相源）
    sensor_pos_norm = ds.sensor_pos                  # [K, 2]，已 normalize 到 [0,1]²
    sensor_time = ds.sensor_time.astype(np.float32)  # [T_sub]
    obs_norm    = ds.sensor_vals.astype(np.float32)  # [K, T_sub, C]，已 train-stats normalize
    s_mean = ds.observed_channel_mean[None, None, :]  # [1, 1, C] for downstream
    s_std  = ds.observed_channel_std[None, None, :]   # [1, 1, C]

    # body mask：dataset 用 |u|+|v|，evaluator 之前只用 |u|（C2）。改用 dataset 一致的值。
    body, fluid_mask = _body_masks_from_dataset(ds, H, W)

    re_value = ds.re_value
    re_norm  = ds.re_norm
    print(
        f"Re={re_value:.0f}  re_norm={re_norm:.4f}  T_sub={len(sensor_time)}"
        f"  train_idx={len(ds.train_t_idx)}  val_idx={len(ds.val_t_idx)}"
    )

    # IMP-6: 驗證 sensor_time uniform；dataset.dt_phys 假設 uniform，nonuniform 會讓
    #        summary metadata 失真。np.gradient 對 nonuniform 仍正確，所以只 warn 不 raise。
    if len(sensor_time) >= 2:
        _diffs = np.diff(sensor_time.astype(np.float64))
        _dt_med = float(np.median(_diffs))
        if _dt_med > 0 and not np.allclose(_diffs, _dt_med, rtol=1e-3):
            print(
                f"  [WARN] sensor_time 非 uniform（max rel-diff "
                f"{float(np.max(np.abs(_diffs - _dt_med)) / _dt_med):.2e}）；"
                f"dt_phys={_dt_med:.4e} 是 median。"
            )

    # ── Encode ───────────────────────────────────────────────────────────────
    sv_t  = torch.tensor(obs_norm.transpose(1, 0, 2), dtype=torch.float32, device=device)  # [T, K, C]
    sp_t  = torch.tensor(sensor_pos_norm,              dtype=torch.float32, device=device)  # [K, 2]
    st_t  = torch.tensor(sensor_time,                  dtype=torch.float32, device=device)  # [T]

    with torch.no_grad():
        h_states, s_time = model.encode(sv_t, sp_t, re_norm, st_t)

    # ── 全場查詢函式 ─────────────────────────────────────────────────────────
    xy_flat = np.stack([x2d_n.ravel(), y2d_n.ravel()], axis=1).astype(np.float32)  # [H*W, 2]
    xy_t = torch.tensor(xy_flat, device=device)

    # Hard body BC（Sukumar 2022）：model output 套 (φ/scale).clamp(0,1) gate。
    # 改用 dataset 的 _sdf_grid + bc_distance_scale（C2 修法）：避免 evaluator 重算
    # body mask 與訓練端 silent drift（之前 detect_body 只看 |u|，dataset 看 |u|+|v|）。
    _use_hard_body_bc = bool(getattr(model, "use_hard_body_bc", False))
    _bd_full = None
    if _use_hard_body_bc:
        # 直接拿 dataset 已算好的 SDF（normalized 距離，與訓練 query_body_distance 同公式）
        _sdf_grid = ds._sdf_grid                       # [H, W]
        _H, _W = _sdf_grid.shape
        _col = np.clip(xy_flat[:, 0] * (_W - 1), 0.0, float(_W - 1))
        _row = np.clip(xy_flat[:, 1] * (_H - 1), 0.0, float(_H - 1))
        _c0 = np.minimum(_col.astype(np.int64), _W - 2); _c1 = _c0 + 1
        _r0 = np.minimum(_row.astype(np.int64), _H - 2); _r1 = _r0 + 1
        _wc = _col - _c0; _wr = _row - _r0
        _d00 = _sdf_grid[_r0, _c0]; _d01 = _sdf_grid[_r0, _c1]
        _d10 = _sdf_grid[_r1, _c0]; _d11 = _sdf_grid[_r1, _c1]
        _d0 = _d00 * (1 - _wc) + _d01 * _wc
        _d1 = _d10 * (1 - _wc) + _d11 * _wc
        _bd_full = torch.tensor(
            (_d0 * (1 - _wr) + _d1 * _wr).astype(np.float32), device=device,
        )
        # bc_distance_scale 從 dataset 取（與訓練端 set_body_bc_scale 用的值完全一致）
        _bc_scale = float(ds.bc_distance_scale)
        if hasattr(model, "set_body_bc_scale"):
            model.set_body_bc_scale(_bc_scale)
        print(f"  hard_body_bc: enabled (H={_H}, W={_W}, scale={_bc_scale:.4f})")

    # Per-component denormalization stats — 預設 identity（不套 denorm）。
    # Why: model raw output 已經是 physical 量級。
    #      losses.py:223 的 data loss 是 (raw - mean)/std vs normalized_target，
    #      最佳解 raw == physical_target。所以 raw 直接跟 physical DNS 比是對的；
    #      額外套 denorm 會 double-scale。歷史見 docs/experiment_log.md DIAGNOSTIC。
    # Note: cylinder evaluator 只 query u (c=0) 與 v (c=1)；c=2 (p) 從未被 caller 使用，
    #       故只維護 u/v 兩個 component 的 denorm（IMP-3 dead code 移除）。
    if args.apply_denormalization:
        print("  [WARN] --apply-denormalization 啟用：raw * std + mean。預期會 double-scale。")
        _denorm_mean = (
            float(s_mean[0, 0, 0]),  # u（dataset train-only stats）
            float(s_mean[0, 0, 1]),  # v
        )
        _denorm_std = (
            float(s_std[0, 0, 0]),
            float(s_std[0, 0, 1]),
        )
    else:
        if args.legacy_checkpoint:
            print("  [INFO] --legacy-checkpoint deprecated（現在 default 即 identity）")
        _denorm_mean = (0.0, 0.0)
        _denorm_std  = (1.0, 1.0)

    def query_field(comp_idx: int, t_val: float) -> np.ndarray:
        # cylinder evaluator 只支援 u (0) / v (1)；fail-fast 對誤用
        if comp_idx not in (0, 1):
            raise ValueError(f"cylinder evaluator 不支援 comp_idx={comp_idx}（僅 0=u, 1=v）")
        parts = []
        with torch.no_grad():
            for s in range(0, xy_t.shape[0], QUERY_BATCH):
                e = min(s + QUERY_BATCH, xy_t.shape[0])
                n = e - s
                xy_b = xy_t[s:e]
                t_b  = torch.full((n,), t_val, dtype=torch.float32, device=device)
                c_b  = torch.full((n,), comp_idx, dtype=torch.long,  device=device)
                bd_b = _bd_full[s:e] if _bd_full is not None else None
                out  = model.query_decoder(
                    xy_b, t_b, c_b, h_states, s_time, sp_t, body_distance=bd_b,
                )
                parts.append(out.squeeze(1).cpu().numpy())
        raw = np.concatenate(parts).reshape(H, W)
        # Denormalize: phys = raw * std + mean
        return raw * _denorm_std[comp_idx] + _denorm_mean[comp_idx]

    # ── DNS 時間對齊（含 sanity check）───────────────────────────────────────
    # CRIT-4: 對 *所有* sensor_time 預先做 dns_idx mapping + sanity check（與 Kolmogorov
    #         evaluator 對稱）。之前只在被 eval_stride 選中的子集合 check，剩下 silent。
    # Why: 不能假設 sensor_time index 對應 dns["t"] 同樣 index（dataset 可能有 stride，
    #      DNS 是原始軸）。用 find_dns_time_idx (two-tier ULP+floor) 對每個 sensor t_val 找
    #      dns index，並 assert 偏差 < 0.5·dns_dt 防 silent 全部 collapse 到 dns[0]。
    t_dns = dns["t"].astype(np.float64)
    if len(t_dns) < 2:
        raise RuntimeError(f"DNS time axis 長度 {len(t_dns)}（< 2），無法對齊")
    _dns_dt = float(np.median(np.diff(t_dns)))
    if _dns_dt <= 0:
        raise RuntimeError(f"DNS dt 非正：{_dns_dt}")
    sensor_to_dns_idx = np.empty(len(sensor_time), dtype=np.int64)
    for _i, _tv in enumerate(sensor_time):
        _ix = find_dns_time_idx(t_dns, float(_tv))
        _diff = abs(float(t_dns[_ix]) - float(_tv))
        if _diff > 0.5 * _dns_dt:
            raise RuntimeError(
                f"DNS time alignment 失敗 (sensor i={_i} t={_tv:.4f})："
                f"floor DNS t={t_dns[_ix]:.4f}, 差 {_diff:.4e} > 0.5·dt={0.5 * _dns_dt:.4e}"
            )
        sensor_to_dns_idx[_i] = _ix

    # ── 評估時間步（依 eval_stride）──────────────────────────────────────────
    # SUG-1: fail-fast 對 invalid stride（Kolmogorov 有，cylinder 對稱）
    if args.eval_stride < 1:
        raise ValueError(f"--eval-stride 必須 ≥ 1，收到 {args.eval_stride}")
    eval_tidx = np.arange(0, len(sensor_time), args.eval_stride)
    eval_times = sensor_time[eval_tidx]
    # 標記 train/val（用 dataset 的 train_t_idx / val_t_idx，C3）
    _train_set = set(int(i) for i in ds.train_t_idx.tolist())
    eval_is_train = np.array([int(ti) in _train_set for ti in eval_tidx], dtype=bool)
    eval_is_val   = ~eval_is_train
    print(
        f"evaluating {len(eval_times)} time steps "
        f"(train={int(eval_is_train.sum())}, val={int(eval_is_val.sum())}) …"
    )

    u_pred_list, v_pred_list = [], []
    u_ref_list,  v_ref_list  = [], []
    for i, (ti, tv) in enumerate(zip(eval_tidx, eval_times)):
        if i % 5 == 0:
            print(f"  step {i}/{len(eval_times)}  t={tv:.2f}s", flush=True)
        dns_idx = int(sensor_to_dns_idx[ti])   # 預先 vectorize 過的 mapping
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

    def fluid_rmse(pred, ref):
        return np.sqrt(np.mean((pred[:, fm] - ref[:, fm]) ** 2, axis=1))

    def fluid_ke(u_, v_):
        return 0.5 * np.mean(u_[:, fm] ** 2 + v_[:, fm] ** 2, axis=1)

    u_rmse = fluid_rmse(u_pred, u_ref)
    v_rmse = fluid_rmse(v_pred, v_ref)
    ke_pred = fluid_ke(u_pred, v_pred)
    ke_ref  = fluid_ke(u_ref,  v_ref)
    ke_rel  = np.abs(ke_pred - ke_ref) / np.maximum(ke_ref, 1e-12)

    # 渦度（用 physical 座標 → ω 單位為 1/s）
    omega_pred_list, omega_ref_list = [], []
    for i in range(len(eval_times)):
        omega_pred_list.append(vorticity_fd(u_pred[i], v_pred[i], x_1d_phys, y_1d_phys))
        omega_ref_list.append( vorticity_fd(u_ref[i],  v_ref[i],  x_1d_phys, y_1d_phys))
    omega_pred_arr = np.stack(omega_pred_list)
    omega_ref_arr  = np.stack(omega_ref_list)
    omega_rmse = fluid_rmse(omega_pred_arr, omega_ref_arr)

    # divergence（用 physical 座標 → div 單位為 1/s）
    # I6: 同時算 DNS 自身 divergence 作為 baseline。非均勻格 + np.gradient 的
    #     numerical scheme 在 DNS 上也有 ~1e-2 量級殘差；無 baseline 無法判斷
    #     model 預測的 div_l2 是真的不滿足連續性，還是 evaluator 自身的數值誤差。
    div_pred_list = []
    div_ref_list  = []
    for i in range(len(eval_times)):
        div_pred_list.append(divergence_fd(u_pred[i], v_pred[i], x_1d_phys, y_1d_phys))
        div_ref_list .append(divergence_fd(u_ref[i],  v_ref[i],  x_1d_phys, y_1d_phys))
    div_pred = np.stack(div_pred_list)
    div_ref  = np.stack(div_ref_list)
    div_l2     = np.sqrt(np.mean(div_pred[:, fm] ** 2, axis=1))
    div_l2_ref = np.sqrt(np.mean(div_ref [:, fm] ** 2, axis=1))

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
                {"LNN L2": div_l2, "DNS L2": div_l2_ref},
                "Divergence L2 vs Time", "L2")

    # ── Summary JSON ─────────────────────────────────────────────────────────
    # C3: train/val 切分 — 重要 caveat：訓練端 sample_sensor_batch 目前**沒有**
    #     按 dataset.train_t_idx 過濾 supervision pool（src/cylinder_dataset.py:338,
    #     src/pi_lnn/training.py:612 與 ds.sample_sensor_batch caller 群均如此），
    #     所以 evaluator 報的 `*_val` 是 dataset 內部 random partition (transductive)，
    #     不是嚴格意義上的 unseen-by-training metric。修真正 generalization 需動訓練端。
    def _add_split(out: dict, key: str, arr: np.ndarray) -> None:
        """Backward-compat schema: plain float on `key` + `key_train` + `key_val`。

        Why: 之前下游 (compare_experiments.py) 預期 plain float；新增 split 用 suffix
             避免 break。空集合的 split 不寫對應 key。
        """
        out[key] = float(np.mean(arr))
        if eval_is_train.any():
            out[f"{key}_train"] = float(np.mean(arr[eval_is_train]))
        if eval_is_val.any():
            out[f"{key}_val"] = float(np.mean(arr[eval_is_val]))

    # 每步 metric（M6: cylinder 之前沒 summary_steps，現在補上以利 reproducibility）
    summary_steps = [
        {
            "time": float(eval_times[i]),
            "split": "train" if bool(eval_is_train[i]) else "val",
            "u_rmse":     float(u_rmse[i]),
            "v_rmse":     float(v_rmse[i]),
            "omega_rmse": float(omega_rmse[i]),
            "ke_rel_err": float(ke_rel[i]),
            "div_l2":     float(div_l2[i]),
            "div_ref_l2": float(div_l2_ref[i]),   # 與 summary 鍵名 div_ref_l2_mean 一致
            "ke_pred":    float(ke_pred[i]),
            "ke_ref":     float(ke_ref[i]),
        }
        for i in range(len(eval_times))
    ]

    summary: dict = {
        "config":     str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "re":         re_value,
        "re_norm":    re_norm,
        "n_eval_steps": int(len(eval_times)),
        "n_eval_train": int(eval_is_train.sum()),
        "n_eval_val":   int(eval_is_val.sum()),
        # M5: grid metadata — 之前 summary 缺，user 必須回頭 trace cfg 才知道 grid res
        "grid_H": int(H),
        "grid_W": int(W),
        "n_fluid_points": int(fm.sum()),
        "dt_phys":        float(ds.dt_phys),
        "Lx":             float(ds.Lx),
        "Ly":             float(ds.Ly),
        "bc_distance_scale": float(ds.bc_distance_scale),
        "bc_inflow_u":       float(ds.bc_inflow_u),
        "use_hard_body_bc": bool(_use_hard_body_bc),
        "apply_denormalization": bool(args.apply_denormalization),
        # CRIT-3: 把 sensor_subsample 寫進 summary，user 後續比對 ckpt 時可驗證一致性。
        # 若 user 用不同 cfg 跑 evaluator（subsample 不同）→ stats / split 都不同 → metric drift。
        "sensor_subsample": int(cfg.get("sensor_subsample", 1)),
        "train_ratio": float(TRAIN_RATIO_FALLBACK),
        "ds_seed":     int(ds_seed),
        "eval_stride": int(args.eval_stride),   # ASYM-1: 與 Kolmogorov summary 對稱
        # late-time aggregate（保留向後相容欄位名）
        "ke_rel_err_late": float(np.mean(ke_rel[len(ke_rel) * 2 // 3:])),
    }
    # 主要 metric — 用 backward-compat schema：plain float key + _train/_val suffix
    _add_split(summary, "u_rmse_mean",     u_rmse)
    _add_split(summary, "v_rmse_mean",     v_rmse)
    _add_split(summary, "omega_rmse_mean", omega_rmse)
    _add_split(summary, "ke_rel_err_mean", ke_rel)
    _add_split(summary, "div_l2_mean",     div_l2)
    _add_split(summary, "div_ref_l2_mean", div_l2_ref)
    _add_split(summary, "ke_ref_mean",     ke_ref)
    _add_split(summary, "ke_pred_mean",    ke_pred)
    summary["steps"] = summary_steps
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    def _fmt(key: str) -> str:
        """格式化 plain mean + train/val（若 summary 中存在 _train / _val key）。"""
        parts = [f"all={summary[key]:.4e}"]
        if f"{key}_train" in summary:
            parts.append(f"train={summary[f'{key}_train']:.4e}")
        if f"{key}_val" in summary:
            parts.append(f"val={summary[f'{key}_val']:.4e}")
        return "  ".join(parts)

    print("\n=== Cylinder Evaluation ===")
    print(f"checkpoint: {args.checkpoint}")
    print(f"n_eval: {summary['n_eval_steps']} (train={summary['n_eval_train']}, val={summary['n_eval_val']})")
    print(f"u  RMSE       : {_fmt('u_rmse_mean')}")
    print(f"v  RMSE       : {_fmt('v_rmse_mean')}")
    print(f"ω  RMSE       : {_fmt('omega_rmse_mean')}")
    print(f"KE rel-err    : {_fmt('ke_rel_err_mean')}")
    print(f"KE rel-err late: {summary['ke_rel_err_late']:.4e}")
    print(f"div L2 (LNN)  : {_fmt('div_l2_mean')}")
    print(f"div L2 (DNS)  : {_fmt('div_ref_l2_mean')}   ← baseline")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
