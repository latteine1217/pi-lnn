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
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch

from picon_kolmogorov import create_picon_model, load_picon_config
# Why: evaluator 必須跟訓練端用 *完全相同* 的 stats / train-val split / RE_MEAN/STD。
#      重複 hardcode RE_MEAN/STD（line 671 之前）會 silent drift；改 import dataset class，
#      所有常數從 dataset 內部抽（dataset 自己用 RE_MEAN/RE_STD 算 re_norm，
#      evaluator 直接用 ds.re_norm）。
from kolmogorov_dataset import KolmogorovDataset
from pi_con import find_dns_time_idx   # 共用 module（避免兩 evaluator 重複實作 → drift）

# 訓練端 hardcoded train_ratio=0.8（pi_con/training.py:88）。evaluator 對齊。
TRAIN_RATIO_FALLBACK = 0.8


# 期刊風格繪圖（NeurIPS/ICLR）— 透過 shared helper 套用全域 rcParams。
# Why: 三個 evaluator script 共用同一 style 避免 figure 在同一篇 paper 中 drift。
from pi_con.plot_style import apply_journal_rcparams, DNS, PICON
apply_journal_rcparams()


def _markevery_for(n: int, target: int = 12) -> int:
    """What: 計算 markevery 步長，使一條時序線顯示約 target 個 markers。

    Why: 期刊圖 marker 太密會讓讀者分不清趨勢；自適應步長保持視覺清晰度。
    """
    return max(1, n // max(target, 1))


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
        default=Path("artifacts/deeponet-cfc-smoke/picon_kolmogorov_final.pt"),
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
    parser.add_argument(
        "--eval-block-factor", type=int, default=2,
        help=(
            "評估網格的 block-average 粗化倍率（預設 2 = 既有行為）。"
            " eval grid = stored_N / factor。跨 Re 比較時用來把不同儲存解析度的 DNS"
            " 對齊到同一張評估網格，例如 stored 512 用 factor 4、stored 256 用 factor 2"
            " 皆得 128^2。"
        ),
    )
    parser.add_argument(
        "--eval-stride", type=int, default=1,
        help=(
            "每隔幾個 sensor time step 評估一次（預設 1=全部）。"
            " IMP-2: T_sub 大時用 stride>1 加速；對 T_sub=2001 + stride=10 約 200 query × 3 component"
            " ~= cylinder 等級工作量。stride>1 會 subsample summary_steps，但 NS residual"
            " 的時間導數在 stride 後仍正確（np.gradient 接 nonuniform spacing）。"
        ),
    )
    parser.add_argument(
        "--eval-on-dns-grid",
        action="store_true",
        help=(
            "Scheme-B：評估時刻改用完整 DNS grid（而非 sensor_time）。CfC/branch context"
            " (h_states, s_time) 仍由訓練用的 sensor 序列建構、不變；只有 query 時刻換成 DNS 全格。"
            " 用於間斷 sensor（random dropout）測試：可在被丟棄的 gap 時刻 query，量 CfC 連續時間"
            " 外插 vs vanilla zero-order-hold 的差異。gap/seen 拆分於後處理用 json 的 dropped/retained 索引。"
        ),
    )
    parser.add_argument(
        "--apply-denormalization",
        action="store_true",
        help=(
            "Apply output denormalization (raw * std + mean) before metrics."
            " 預設 False — model 直接學 physical 量級 raw output（見 losses.py:223"
            " 的 (raw - mean)/std loss），evaluator 不該再套 denorm，否則 double-scale。"
            " 此 flag 留作向後相容；正常 use case 都不該開。"
        ),
    )
    parser.add_argument(
        "--legacy-checkpoint",
        action="store_true",
        help=(
            "[DEPRECATED] 之前指「跳過 denorm」是 opt-in；現在 default 已是跳過 denorm。"
            " 此 flag 保留為 no-op 以維持向後相容（舊 script 不會 break）。"
        ),
    )
    parser.add_argument(
        "--export_arrays",
        action="store_true",
        help=(
            "Dump per-time-step arrays (KE, divergence ratio, u/v error, kf amp/phase, "
            "band errors) as series.npz inside output_dir for multi-seed envelope aggregation."
        ),
    )
    parser.add_argument(
        "--export_fields",
        action="store_true",
        help=(
            "額外存全場 (u,v,omega)[t,x,y] pred+ref 與 grid 至 fields.npz，"
            "供 mean-profile / Reynolds-stress / 時間頻譜後處理（不影響 series.npz）。"
        ),
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


def block_avg(field: np.ndarray, factor: int = 2) -> np.ndarray:
    """What: f x f block average，支援 [..., fN, fN] batch shape。

    Why: 向量化避免逐 frame Python loop；既有 [2N, 2N] 用法仍兼容。
         factor 可調是為了讓不同儲存解析度的 DNS 評估在同一張網格上（cross-Re 比較），
         預設 2 與既有行為逐位元相同。
    """
    f = int(factor)
    n_x = field.shape[-2] // f
    n_y = field.shape[-1] // f
    new_shape = (*field.shape[:-2], n_x, f, n_y, f)
    return field.reshape(new_shape).mean(axis=(-3, -1))


def coarse_reference_grid(
    x: np.ndarray, y: np.ndarray, factor: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """What: 產生與 f x f block average 對齊的 coarse query grid。

    Why: `block_avg()` 代表的是 coarse cell 的平均值，不是原始 fine grid node。
         若仍在 `x[::f], y[::f]` 上 query，prediction 與 reference 會固定錯半格，
         系統性污染 RMSE、渦度與頻譜診斷。
    """
    f = int(factor)
    if len(x) % f != 0 or len(y) % f != 0:
        raise ValueError(
            f"coarse_reference_grid 需要長度可被 factor={f} 整除的 grid，"
            f"收到 len(x)={len(x)}, len(y)={len(y)}"
        )
    x_coarse = x.reshape(-1, f).mean(axis=1)
    y_coarse = y.reshape(-1, f).mean(axis=1)
    return x_coarse.astype(np.float32), y_coarse.astype(np.float32)


def kinetic_energy(u: np.ndarray, v: np.ndarray) -> float:
    return float(0.5 * np.mean(u ** 2 + v ** 2))


def enstrophy_fd(u: np.ndarray, v: np.ndarray, dx: float) -> float:
    omega = vorticity_fd(u, v, dx)
    return float(0.5 * np.mean(omega ** 2))


def vorticity_fd(u: np.ndarray, v: np.ndarray, dx: float) -> np.ndarray:
    """What: 用中心差分近似 2D 渦度場，支援 [..., N, N] batch shape。

    Why: 渦度是局部旋渦結構最直接的診斷量；批次化避免 evaluator 對 T 個 frame 各呼叫一次。
    """
    dvdx = (np.roll(v, -1, axis=-2) - np.roll(v, 1, axis=-2)) / (2 * dx)
    dudy = (np.roll(u, -1, axis=-1) - np.roll(u, 1, axis=-1)) / (2 * dx)
    return dvdx - dudy


def divergence_fd(u: np.ndarray, v: np.ndarray, dx: float) -> np.ndarray:
    """What: 用中心差分近似 2D 不可壓縮條件殘差，支援 [..., N, N] batch。"""
    dudx = (np.roll(u, -1, axis=-2) - np.roll(u, 1, axis=-2)) / (2 * dx)
    dvdy = (np.roll(v, -1, axis=-1) - np.roll(v, 1, axis=-1)) / (2 * dx)
    return dudx + dvdy


def laplacian_periodic(field: np.ndarray, dx: float) -> np.ndarray:
    """What: 以 periodic stencil 計算 2D Laplacian，支援 [..., N, N] batch。"""
    return (
        np.roll(field, -1, axis=-2)
        + np.roll(field, 1, axis=-2)
        + np.roll(field, -1, axis=-1)
        + np.roll(field, 1, axis=-1)
        - 4.0 * field
    ) / (dx**2)


def time_derivative_series(field_series: np.ndarray, time_vals: np.ndarray) -> np.ndarray:
    """What: 沿時間軸計算一階導數。

    CRIT-1: T < 2 時 fail-fast。np.gradient 在 len=1 會 IndexError；
            evaluator 在 sensor_time 過短時應立即 raise，而非到 ns_residual_fields 才炸。
    """
    if len(time_vals) < 2:
        raise RuntimeError(
            f"time_derivative_series 需要至少 2 個時間點，收到 {len(time_vals)}。"
            f" sensor_time 過短無法做 NS residual 時間導數，請增加 sensor_time 取樣。"
        )
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

    # 向量化：laplacian_periodic 已支援 [T, N, N]，無需逐 frame stack。
    lap_u = laplacian_periodic(u_series, dx)
    lap_v = laplacian_periodic(v_series, dx)
    nu = 1.0 / float(re)
    forcing_wavenumber = (2.0 * np.pi * float(k_forcing)) / float(domain_length)
    forcing = float(forcing_amplitude) * np.sin(forcing_wavenumber * y_coords)[None, None, :]

    mom_u = du_dt + u_series * du_dx + v_series * du_dy + dp_dx - nu * lap_u - forcing
    mom_v = dv_dt + u_series * dv_dx + v_series * dv_dy + dp_dy - nu * lap_v
    cont = du_dx + dv_dy
    return mom_u, mom_v, cont


def energy_spectrum_1d(u: np.ndarray, v: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """What: 計算 1D radial-averaged energy spectrum E(k)。

    Why: 替代原 Python for-loop（每 spectrum N/2 次 mask + sum），改用 np.bincount
         在 ravel 後一次 scatter-add，速度提升 ~10-50×；保留 ordinary wavenumber
         單位（cycles/domain）對齊 k_f=2.0。

    Normalization (Parseval check)：
        uh = fft2(u) / n²  →  Σ |uh|² = mean(|u|²)
        所以 Σ E(k) ≈ 0.5 (mean u² + mean v²) = KE（narrow-band 場）。`/n²` 是對的。

    Bin truncation (I2 + IMP-1 caveat)：
        kk = √(kx²+ky²) 最大可達 √2 · n/2（角落超過 Nyquist）。
        bin_idx > n//2 的 bin 物理上沒有意義（超過 Nyquist），mask 掉而非塞 last bin。
        **副作用**：對 broad-band 場（如 white noise），落在 isotropic Nyquist 與 corner
        之間的能量 (~19% 量級) 會被 mask 掉，造成 Σ E(k) < KE。
        對 Kolmogorov narrow-band 場（能量集中在 k~k_f 附近）影響微小。
        若需嚴格 Parseval consistency 在所有場上，請用 1D unwrap 而非 isotropic bin。
    """
    n = u.shape[0]
    k1d = np.fft.fftfreq(n, d=dx)
    uh = np.fft.fft2(u) / n**2
    vh = np.fft.fft2(v) / n**2
    e2d = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2)
    kx, ky = np.meshgrid(k1d, k1d, indexing="ij")
    kk = np.sqrt(kx**2 + ky**2)
    # n_bins 對應 [k=1, k=2, ..., k=n//2]（Nyquist 上限），共 n//2 個 bin。
    n_bins = n // 2
    # bin_idx ∈ [1, n//2] valid；> n//2（含對角線超 Nyquist 區）mask out。
    bin_idx = np.floor(kk + 0.5).astype(np.int64)
    valid = (bin_idx >= 1) & (bin_idx <= n_bins)
    flat_idx = np.where(valid, bin_idx - 1, 0)  # shift 到 [0, n_bins-1]
    weights = np.where(valid, e2d, 0.0).ravel()
    e_k = np.bincount(flat_idx.ravel(), weights=weights, minlength=n_bins).astype(np.float64)
    edges = np.arange(0.5, n_bins + 1.5, 1.0)  # length n_bins+1（centers = 1, 2, ..., n//2）
    return 0.5 * (edges[:-1] + edges[1:]), e_k


def spectrum_value_at_k(k_vals: np.ndarray, e_vals: np.ndarray, k_target: float) -> float:
    idx = int(np.argmin(np.abs(k_vals - k_target)))
    return float(e_vals[idx])


def summarize_time_local_metric(time_vals: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """What: 將時序指標壓縮成 early/mid/late 與 worst-time 摘要。

    Why: 用 nanmean / nanargmax 防止 MPS 偶發 NaN 整次 abort eval；early/mid/late 邊界
         也寫進 return 供下游 aggregator 知道「late」對應哪段物理時間。
    """
    if len(time_vals) != len(values):
        raise ValueError(
            f"time_vals 與 values 長度不一致：{len(time_vals)} vs {len(values)}"
        )
    idx_chunks = np.array_split(np.arange(len(values)), 3)

    def _chunk_nanmean(indices: np.ndarray) -> float:
        if len(indices) == 0 or not np.isfinite(values[indices]).any():
            return float("nan")
        return float(np.nanmean(values[indices]))

    def _bucket_times(indices: np.ndarray) -> tuple[float, float]:
        if len(indices) == 0:
            return (float("nan"), float("nan"))
        return (float(time_vals[indices[0]]), float(time_vals[indices[-1]]))

    if np.isfinite(values).any():
        worst_idx = int(np.nanargmax(values))
        worst_time = float(time_vals[worst_idx])
        worst_value = float(values[worst_idx])
    else:
        worst_time = float("nan")
        worst_value = float("nan")
    mean_val = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    return {
        "mean": mean_val,
        "early_mean": _chunk_nanmean(idx_chunks[0]),
        "mid_mean": _chunk_nanmean(idx_chunks[1]),
        "late_mean": _chunk_nanmean(idx_chunks[2]),
        "worst_time": worst_time,
        "worst_value": worst_value,
        "early_time_range": list(_bucket_times(idx_chunks[0])),
        "mid_time_range": list(_bucket_times(idx_chunks[1])),
        "late_time_range": list(_bucket_times(idx_chunks[2])),
    }


# Band edges 對應 K=100 sensor 的資訊論 Nyquist k_max ≈ √(K/π) ≈ 5.64：
#   low: k ≤ 5   → sensor 可解析的低頻（paper 主 claim 的 band）
#   mid: 5 < k ≤ 16 → 過渡帶，部分高頻仍可由 PDE residual 推回
#   high: k > 16 → 純資訊論不可解析區（高頻誤差為 K 上限的證據）
# 與 scripts/aim_diagnostic.py:134 的切法保持一致，避免兩支 script 對「low band」定義不同。
BAND_EDGES_K_LOW = 5.0
BAND_EDGES_K_HIGH = 16.0


def compute_band_energies(k_vals: np.ndarray, e_vals: np.ndarray) -> dict[str, float]:
    """What: 將 1D spectrum 壓縮為 low/mid/high 三段 band energy。

    Why: band 邊界用 fixed wavenumber cuts 而非 equal-thirds 才能與 paper claim
         「low band rel-err ≤ 10%」對齊；equal-thirds 會讓 low band 包含到 k≈22
         而 paper claim 是針對 k ≤ 5 的 sensor 可解析範圍。
    """
    positive = k_vals > 0.0
    k_pos = k_vals[positive]
    e_pos = e_vals[positive]
    masks = {
        "low":  k_pos <= BAND_EDGES_K_LOW,
        "mid":  (k_pos > BAND_EDGES_K_LOW) & (k_pos <= BAND_EDGES_K_HIGH),
        "high": k_pos > BAND_EDGES_K_HIGH,
    }
    return {
        label: float(np.sum(e_pos[mask])) if mask.any() else 0.0
        for label, mask in masks.items()
    }


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
    # forcing.* 是 ForcingPrior submodule，向後相容：舊 checkpoint 沒這些 keys 屬正常
    # （model 端會保留 config 預設值）。用 model 端實際的 forcing key 集合做 exact match，
    # 避免未來有其他 module 名也以 "forcing" 開頭時被 swallow。
    if hasattr(model, "forcing"):
        _forcing_keys = {f"forcing.{k}" for k in model.forcing.state_dict().keys()}
    else:
        _forcing_keys = set()
    missing = [k for k in load_result.missing_keys if k not in _forcing_keys]
    unexpected = [k for k in load_result.unexpected_keys if k not in _forcing_keys]
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint 與模型參數不一致："
            f"missing={missing}, unexpected={unexpected}"
        )


def forcing_mode_coeff_u(
    u: np.ndarray,
    y: np.ndarray,
    k_forcing: float,
    domain_length: float = 1.0,
) -> tuple[float, float]:
    """What: 擷取 x-平均後 u(y) 在 forcing mode 的複數 Fourier 係數。

    Why: 目前關鍵問題不是場完全崩潰，而是主模態是否被正確學到。
         直接量 amplitude / phase，比只看總能譜更容易判斷是沒學到還是相位錯。

    I4: phase_arg 必須與 ns_residual_fields 的 forcing 一致：
        forcing(y) = A·sin(2π·k_f·y / domain_length)
        所以投影 basis 也要用同樣 wavelength；之前 hardcode `2π·k·y`
        相當於假設 domain_length=1，雖然 dataset 都是 1.0，但若未來改 domain
        會 silent 偏。
    """
    u_bar = u.mean(axis=0)
    phase_arg = -2.0 * np.pi * float(k_forcing) * y / float(domain_length)
    basis = np.exp(1j * phase_arg)
    coeff = np.mean(u_bar * basis)
    return float(np.abs(coeff)), float(np.angle(coeff))


def _style_field_axes(ax) -> None:
    """What: 場域 imshow 圖的 axes 樣式：4 邊框、極簡 ticks、SI 單位 axis label。

    Why: 全域 rcParams 移除了 top/right spines（NeurIPS 時序圖風格），但 imshow
         場圖必須有完整 4 邊框才能清楚標示空間域邊界；依教授指示，所有 axis
         必須附 SI 單位（域長 L*=1 m，x/y 範圍即 [0, 1] m）。
    """
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.6)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(labelsize=7)
    ax.set_xlabel(r"$x$ [m]", fontsize=8, labelpad=2)
    ax.set_ylabel(r"$y$ [m]", fontsize=8, labelpad=2)
    ax.grid(False)


def plot_field_comparison(
    output_path: Path,
    u_ref: np.ndarray,
    u_pred: np.ndarray,
    v_ref: np.ndarray,
    v_pred: np.ndarray,
    t_val: float,
) -> None:
    """What: DNS / PI-CON / Error 場比較（期刊雙欄寬度）。"""
    u_err = u_pred - u_ref
    v_err = v_pred - v_ref
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.0), constrained_layout=True)

    u_lim = float(max(np.abs(u_ref).max(), np.abs(u_pred).max(), 1e-8))
    v_lim = float(max(np.abs(v_ref).max(), np.abs(v_pred).max(), 1e-8))
    ue_lim = float(max(np.abs(u_err).max(), 1e-8))
    ve_lim = float(max(np.abs(v_err).max(), 1e-8))

    # cbar label 用 SI 單位（u, v: m/s；無因次 NS 假設 U*=1 m/s, L*=1 m）
    panels = [
        (axes[0, 0], u_ref,  "$u$ DNS",   "RdBu_r", -u_lim,  u_lim,  r"$u$ [m/s]"),
        (axes[0, 1], u_pred, "$u$ PI-CON",   "RdBu_r", -u_lim,  u_lim,  r"$u$ [m/s]"),
        (axes[0, 2], u_err,  "$u$ Error", "RdBu_r", -ue_lim, ue_lim, r"$\Delta u$ [m/s]"),
        (axes[1, 0], v_ref,  "$v$ DNS",   "RdBu_r", -v_lim,  v_lim,  r"$v$ [m/s]"),
        (axes[1, 1], v_pred, "$v$ PI-CON",   "RdBu_r", -v_lim,  v_lim,  r"$v$ [m/s]"),
        (axes[1, 2], v_err,  "$v$ Error", "RdBu_r", -ve_lim, ve_lim, r"$\Delta v$ [m/s]"),
    ]
    for ax, field, title, cmap, vmin, vmax, cb_label in panels:
        im = ax.imshow(
            field.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
            aspect="equal", extent=(0.0, 1.0, 0.0, 1.0),
        )
        ax.set_title(title)
        _style_field_axes(ax)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cb_label, fontsize=8)
        cb.ax.tick_params(labelsize=8)
        cb.outline.set_linewidth(0.6)
    # t 值放在最左上 panel 一次（不在每個 title 重複）
    axes[0, 0].text(
        0.02, 0.98, f"$t={t_val:.2f}$",
        transform=axes[0, 0].transAxes, fontsize=9,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
    )
    fig.savefig(output_path)
    plt.close(fig)


def plot_vorticity_comparison(
    output_path: Path,
    omega_ref: np.ndarray,
    omega_pred: np.ndarray,
    t_val: float,
) -> None:
    """What: 渦度 DNS / PI-CON / Error 比較（期刊單列）。"""
    omega_err = omega_pred - omega_ref
    om_lim = float(max(np.abs(omega_ref).max(), np.abs(omega_pred).max(), 1e-8))
    err_lim = float(max(np.abs(omega_err).max(), 1e-8))

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.0), constrained_layout=True)
    panels = [
        (axes[0], omega_ref,  r"$\omega$ DNS",   -om_lim,  om_lim,  r"$\omega$ [1/s]"),
        (axes[1], omega_pred, r"$\omega$ PI-CON",   -om_lim,  om_lim,  r"$\omega$ [1/s]"),
        (axes[2], omega_err,  r"$\omega$ Error", -err_lim, err_lim, r"$\Delta\omega$ [1/s]"),
    ]
    for ax, field, title, vmin, vmax, cb_label in panels:
        im = ax.imshow(
            field.T, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax,
            aspect="equal", extent=(0.0, 1.0, 0.0, 1.0),
        )
        ax.set_title(title)
        _style_field_axes(ax)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cb_label, fontsize=8)
        cb.ax.tick_params(labelsize=8)
        cb.outline.set_linewidth(0.6)
    axes[0].text(
        0.02, 0.98, f"$t={t_val:.2f}$",
        transform=axes[0].transAxes, fontsize=9,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
    )
    fig.savefig(output_path)
    plt.close(fig)


def plot_energy_spectrum(
    output_path: Path,
    k_ref: np.ndarray,
    e_ref: np.ndarray,
    k_pred: np.ndarray,
    e_pred: np.ndarray,
    k_forcing: float,
    k_sensor_nyquist: float | None = None,
) -> None:
    """What: 一維能譜比較（期刊單欄寬度，loglog）+ k^-5/3 慣性區參考線 + sensor Nyquist。

    Why: paper claim 「中高頻 bounded by sensor information k_max ≈ √(K/π)」需要視覺證據。
         k_sensor_nyquist 不傳則跳過該線（向後相容）。
    """
    mask_ref = e_ref > 0.0
    mask_pred = e_pred > 0.0
    fig, ax = plt.subplots(figsize=(5.5, 2.8), constrained_layout=True)
    ax.loglog(k_ref[mask_ref], e_ref[mask_ref], color="#000000", linestyle="-", label="DNS")
    ax.loglog(k_pred[mask_pred], e_pred[mask_pred], color=PICON, linestyle="--", label="PI-CON")

    # k^(-3) 2D 正向 enstrophy-cascade 參考線（k>k_f；Kraichnan）。
    # 不是 k^(-5/3)：後者是 k<k_f 的 inverse energy cascade。anchor 在 k_forcing。
    k_grid_all = k_ref[mask_ref]
    if k_grid_all.size:
        anchor_idx = int(np.argmin(np.abs(k_grid_all - max(k_forcing, 1.0))))
        anchor_k = float(k_grid_all[anchor_idx])
        anchor_e = float(e_ref[mask_ref][anchor_idx])
        k_tail = k_grid_all[k_grid_all >= anchor_k]
        ax.loglog(k_tail, anchor_e * (k_tail / anchor_k) ** (-3.0),
                  color="gray", linestyle=":", linewidth=0.9, label=r"$k^{-3}$")

    ax.axvline(k_forcing, color="black", linestyle="-.", linewidth=0.8,
               label=f"$k_f={k_forcing:.0f}$")
    if k_sensor_nyquist is not None and k_sensor_nyquist > 0:
        ax.axvline(k_sensor_nyquist, color="#009E73", linestyle="--", linewidth=0.9,
                   label=fr"$k_{{\max}}\!\approx\!{k_sensor_nyquist:.2f}$")
    ax.set_xlabel(r"Wavenumber $k$ [1/m]")
    ax.set_ylabel(r"Energy $E(k)$ [m$^3$/s$^2$]")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_metric_vs_time(
    output_path: Path,
    time_vals: np.ndarray,
    ref_vals: np.ndarray,
    pred_vals: np.ndarray,
    title: str,
    y_label: str,
) -> None:
    """What: DNS vs PI-CON 時序比較（期刊單欄寬度）。"""
    me = _markevery_for(len(time_vals))
    fig, ax = plt.subplots(figsize=(3.6, 2.6), constrained_layout=True)
    ax.plot(time_vals, ref_vals, color=DNS, linestyle="-", marker="o",
            markevery=me, label="DNS")
    ax.plot(time_vals, pred_vals, color=PICON, linestyle="--", marker="o",
            markevery=me, markerfacecolor="white", markeredgecolor=PICON, label="PI-CON")
    ax.set_title(title)
    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(y_label)
    ax.legend(loc="best")
    fig.savefig(output_path)
    plt.close(fig)


def plot_forcing_param_trajectory(
    metrics_path: Path,
    output_path: Path,
    truth_A: float,
    truth_kf: float,
) -> None:
    """What: 從 metrics.jsonl 讀 forcing_A / forcing_k_f 時序，畫雙 panel 軌跡 + 真值線。

    Why: 只看 final A/k_f 看不出 identifiability —— 是直接收斂、震盪卡住、還是後期才穩定？
         真值水平線是視覺基準，越靠近代表 PDE residual 越能反推 forcing。
    """
    import json

    steps: list[int] = []
    A_vals: list[float] = []
    kf_vals: list[float] = []
    with metrics_path.open() as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                # 容忍 training 中斷導致的最後一行半寫狀態
                continue
            if "forcing_A" in d or "forcing_k_f" in d:
                steps.append(int(d["step"]))
                A_vals.append(float(d.get("forcing_A", float("nan"))))
                kf_vals.append(float(d.get("forcing_k_f", float("nan"))))
    if not steps:
        return  # 沒 learning record 就跳過（baseline / fixed forcing run）

    steps_arr = np.asarray(steps)
    A_arr = np.asarray(A_vals)
    kf_arr = np.asarray(kf_vals)

    fig, axes = plt.subplots(2, 1, figsize=(3.6, 4.4), constrained_layout=True, sharex=True)
    # A panel
    if np.isfinite(A_arr).any():
        axes[0].plot(steps_arr, A_arr, color="#d62728", marker="o", markersize=2.5,
                     markevery=max(1, len(steps_arr) // 30), linewidth=1.2, label="learned")
    axes[0].axhline(truth_A, color="#1f77b4", linestyle="--", linewidth=1.0, label=f"truth = {truth_A}")
    axes[0].set_ylabel(r"Forcing $A$")
    axes[0].set_title("Forcing parameter trajectory")
    axes[0].legend(loc="best", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # k_f panel
    if np.isfinite(kf_arr).any():
        axes[1].plot(steps_arr, kf_arr, color="#d62728", marker="o", markersize=2.5,
                     markevery=max(1, len(steps_arr) // 30), linewidth=1.2, label="learned")
    axes[1].axhline(truth_kf, color="#1f77b4", linestyle="--", linewidth=1.0, label=f"truth = {truth_kf}")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel(r"Forcing $k_f$")
    axes[1].legend(loc="best", fontsize=7)
    axes[1].grid(True, alpha=0.3)

    fig.savefig(output_path)
    plt.close(fig)


def plot_series_collection(
    output_path: Path,
    time_vals: np.ndarray,
    series_map: dict[str, np.ndarray],
    title: str,
    y_label: str,
    yscale: str = "linear",
) -> None:
    """What: 多條時序指標疊圖（期刊單欄寬度，自動色彩+線型）。"""
    me = _markevery_for(len(time_vals))
    fig, ax = plt.subplots(figsize=(3.6, 2.6), constrained_layout=True)
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    linestyles = ["-", "--", "-.", ":", "-", "--"]
    for i, (label, values) in enumerate(series_map.items()):
        color = palette[i % len(palette)]
        ls = linestyles[i % len(linestyles)]
        ax.plot(time_vals, values, color=color, linestyle=ls, marker="o",
                markevery=me, label=label)
    ax.set_title(title)
    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(y_label)
    if yscale != "linear":
        ax.set_yscale(yscale)
    ax.legend(loc="best", ncol=1 if len(series_map) <= 3 else 2)
    fig.savefig(output_path)
    plt.close(fig)


def plot_uv_error_vs_time(
    output_path: Path,
    time_vals: np.ndarray,
    u_err: np.ndarray,
    v_err: np.ndarray,
) -> None:
    """What: u / v RMSE 隨時間變化（期刊單欄寬度）。"""
    me = _markevery_for(len(time_vals))
    fig, ax = plt.subplots(figsize=(3.6, 2.6), constrained_layout=True)
    ax.plot(time_vals, u_err, color="#1f77b4", linestyle="-", marker="o",
            markevery=me, label="$u$ RMSE")
    ax.plot(time_vals, v_err, color="#d62728", linestyle="--", marker="o",
            markevery=me, markerfacecolor="white", markeredgecolor="#d62728",
            label="$v$ RMSE")
    ax.set_title("Velocity RMSE")
    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(r"RMSE [m/s]")
    ax.legend(loc="best")
    fig.savefig(output_path)
    plt.close(fig)


def plot_mode_vs_time(
    output_path: Path,
    time_vals: np.ndarray,
    ref_vals: np.ndarray,
    pred_vals: np.ndarray,
    title: str,
    y_label: str,
) -> None:
    """What: forcing mode amplitude / phase 時間演化（期刊單欄寬度）。"""
    me = _markevery_for(len(time_vals))
    fig, ax = plt.subplots(figsize=(3.6, 2.6), constrained_layout=True)
    ax.plot(time_vals, ref_vals, color=DNS, linestyle="-", marker="o",
            markevery=me, label="DNS")
    ax.plot(time_vals, pred_vals, color=PICON, linestyle="--", marker="o",
            markevery=me, markerfacecolor="white", markeredgecolor=PICON, label="PI-CON")
    ax.set_title(title)
    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(y_label)
    ax.legend(loc="best")
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_picon_config(args.config)
    validate_single_dataset_eval(cfg)
    device = choose_device(args.device)
    model = create_picon_model(cfg).to(device)
    checkpoint_payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint_payload, dict):
        _sf_mode = checkpoint_payload.get("schedulefree_mode", None)
        if _sf_mode == "train":
            print(
                "  [WARN] checkpoint 含 schedulefree_mode='train' (x_t, 給 resume 用)；"
                "\n         inference quality 比 final.pt (y_t, eval mode) 差 5-30%。"
                "\n         若要 best inference 結果，請改用 .../picon_kolmogorov_final.pt"
            )
    state = extract_model_state(checkpoint_payload)
    load_model_weights_strict(model, state)
    model.eval()

    # 印出 ForcingPrior 狀態（baseline/learn_A/learn_kf/learn_both 可一眼分辨）
    if hasattr(model, "forcing"):
        snap = model.forcing.snapshot()
        truth_A = float(cfg.get("kolmogorov_A", 0.1))
        truth_kf = float(cfg.get("kolmogorov_k_f", 2.0))
        print(
            f"  [forcing] learned A={snap['A']:.4f} (truth={truth_A}, err={abs(snap['A']-truth_A)/truth_A*100:.2f}%)"
            f"  |  learned k_f={snap['k_f']:.4f} (truth={truth_kf}, err={abs(snap['k_f']-truth_kf)/truth_kf*100:.2f}%)"
            f"  |  learn=(A:{snap['learn_A']}, k_f:{snap['learn_k_f']})"
        )
        # 軌跡圖：metrics.jsonl 通常在 checkpoint 同目錄（或父目錄，視 checkpoints/ 結構）
        if snap["learn_A"] or snap["learn_k_f"]:
            for cand in [args.checkpoint.parent / "metrics.jsonl",
                         args.checkpoint.parent.parent / "metrics.jsonl"]:
                if cand.exists():
                    plot_forcing_param_trajectory(
                        cand, output_dir / "forcing_param_vs_step.png",
                        truth_A=truth_A, truth_kf=truth_kf,
                    )
                    print(f"  [forcing] trajectory plot saved (source: {cand})")
                    break

    # Flag 互斥檢查
    if args.apply_denormalization and args.legacy_checkpoint:
        print(
            "  [WARN] --apply-denormalization 與 --legacy-checkpoint 同時設定；"
            " --apply-denormalization 強制走 raw*std+mean，"
            "--legacy-checkpoint (deprecated) 在此情況下無作用。"
        )

    # ── 重建 dataset 拿單一真相源的 stats / train_t_idx / val_t_idx ────────
    # Why: 之前 evaluator 自己重新算 sensor mean/std；雖然對 Kolmogorov 是 full-data
    #      stats（與 dataset 一致），但 RE_MEAN/STD 寫死、train_t_idx/val_t_idx
    #      無法區分 → 跟 cylinder evaluator 對稱地用 dataset import。
    ds_seed = int(cfg.get("seed", 42))
    ds = KolmogorovDataset(
        sensor_json=cfg["sensor_jsons"][0],
        sensor_npz=cfg["sensor_npzs"][0],
        dns_path=cfg["dns_paths"][0],
        re_value=float(cfg.get("re_values", [1000.0])[0]),
        observed_channel_names=tuple(cfg.get("observed_sensor_channels", ["u", "v"])),
        train_ratio=TRAIN_RATIO_FALLBACK,   # 跟 pi_con/training.py:88 hardcode 一致
        seed=ds_seed,
    )
    sensor_pos  = ds.sensor_pos.astype(np.float32)            # [K, 2]
    sensor_vals = ds.sensor_vals.astype(np.float32)           # [K, T, C]，已 normalize
    sensor_time = ds.sensor_time.astype(np.float32)
    # 仍 expose mean/std 兩個 [1,1,C] tensor 給 denorm path 用（與舊行為等價）
    sensor_mean = ds.observed_channel_mean[None, None, :].astype(np.float32)
    sensor_std  = ds.observed_channel_std [None, None, :].astype(np.float32)

    # IMP-6: 驗證 sensor_time uniform（dataset.dt_phys 假設 uniform，但無 assert）。
    #        nonuniform sensor_time 會讓 summary 的 dt_phys 失真，且訓練端某些
    #        physics scheduling 也假設 uniform。
    if len(sensor_time) >= 2:
        _diffs = np.diff(sensor_time.astype(np.float64))
        _dt_med = float(np.median(_diffs))
        if _dt_med > 0 and not np.allclose(_diffs, _dt_med, rtol=1e-3):
            print(
                f"  [WARN] sensor_time 非 uniform（max rel-diff "
                f"{float(np.max(np.abs(_diffs - _dt_med)) / _dt_med):.2e}）；"
                f"dt_phys={_dt_med:.4e} 是 median，下游 NS residual time-derivative 仍正確。"
            )

    dns = np.load(cfg["dns_paths"][0], allow_pickle=True).item()

    # Axis convention assert：CLAUDE.md KNOWN_PITFALLS 記載過 sensor file axis swap 災難
    # （EXP-101/102/103/105），這裡守住 evaluator 端：DNS array u/v 必須是 [T, N_x, N_y]
    # 與 dns["x"], dns["y"] 形狀對齊。若未來 dataset 改 (y, x) 順序會 silently 評估到
    # transposed reference，這個 assert 是最後一道防線。
    _nx_dns, _ny_dns = len(dns["x"]), len(dns["y"])
    for _ch in ("u", "v"):
        if _ch in dns:
            _shape = dns[_ch].shape
            assert _shape[-2] == _nx_dns and _shape[-1] == _ny_dns, (
                f"DNS axis convention 違反：dns['{_ch}'].shape={_shape}，"
                f"預期最後兩維 (N_x={_nx_dns}, N_y={_ny_dns})。"
                f"參考 CLAUDE.md KNOWN_PITFALLS 'Sensor file axis convention'。"
            )

    x_g, y_g = coarse_reference_grid(
        dns["x"].astype(np.float32),
        dns["y"].astype(np.float32),
        factor=args.eval_block_factor,
    )
    xx, yy = np.meshgrid(x_g, y_g, indexing="ij")
    xy_flat = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    xy_t = torch.tensor(xy_flat, dtype=torch.float32, device=device)
    # 8192 OOMs at K=400 (full-field query × K cross-attention + 2nd-order autograd
    # peaks ~17 GB); 2048 keeps the peak memory-safe on 20 GB MPS / 32 GB CPU.
    batch = 2048

    sv_t = torch.tensor(sensor_vals.transpose(1, 0, 2), dtype=torch.float32, device=device)
    sp_t = torch.tensor(sensor_pos, dtype=torch.float32, device=device)
    st_t = torch.tensor(sensor_time, dtype=torch.float32, device=device)
    re_value = ds.re_value
    re_norm  = ds.re_norm   # = (re_value - RE_MEAN) / RE_STD，dataset 內部用 import 來的常數

    with torch.no_grad():
        h_states, s_time = model.encode(sv_t, sp_t, re_norm, st_t)

    # Per-component denormalization stats — 預設 identity（不套 denorm）。
    # Why: model raw output 已經是 physical 量級。
    #      losses.py:223 的 data loss 是 (raw - mean)/std vs normalized_target，
    #      最佳解 raw == normalized_target * std + mean = physical_target。
    #      所以 evaluator 直接拿 raw 跟 physical DNS 比是對的；額外套 denorm 會 double-scale。
    # 歷史：d62e698 (2026-05-03) 誤判 model 學 normalized → evaluator default 套 denorm。
    #       這個 silent regression 把 EXP-070~074 的 KE 全部誤報成 ~84%；
    #       用 identity 後真實 KE 約 6-9%（見 docs/experiment_log.md DIAGNOSTIC section）。
    if args.apply_denormalization:
        print("  [WARN] --apply-denormalization 啟用：raw * std + mean。預期會 double-scale。")
        _denorm_mean = (
            float(sensor_mean[0, 0, 0]),  # u
            float(sensor_mean[0, 0, 1]),  # v
            0.0,                           # p（無 reference）
        )
        _denorm_std = (
            float(sensor_std[0, 0, 0]),
            float(sensor_std[0, 0, 1]),
            1.0,
        )
    else:
        # 預設路徑：identity transform，物理上正確。
        if args.legacy_checkpoint:
            print("  [INFO] --legacy-checkpoint deprecated（現在 default 即 identity）")
        _denorm_mean = (0.0, 0.0, 0.0)
        _denorm_std  = (1.0, 1.0, 1.0)

    # Kolmogorov 是 periodic flow，無 body → 不應該啟用 use_hard_body_bc。
    # 如果 model 是 hard BC 訓的（誤用），這裡 raise 而不是 silent 退化。
    if bool(getattr(model, "use_hard_body_bc", False)):
        raise ValueError(
            "Kolmogorov dataset 不支援 use_hard_body_bc=True；該機制只對有 body geometry 的"
            " cylinder 有意義。請用 use_hard_body_bc=False 訓的 ckpt。"
        )

    def query_field(comp_idx: int, t_val: float) -> np.ndarray:
        parts = []
        with torch.no_grad():
            for start in range(0, xy_t.shape[0], batch):
                end = min(start + batch, xy_t.shape[0])
                xy_b = xy_t[start:end]
                bs = end - start
                t_b = torch.full((bs,), t_val, dtype=torch.float32, device=device)
                c_b = torch.full((bs,), comp_idx, dtype=torch.long, device=device)
                out = model.query_decoder(
                    xy_b, t_b, c_b, h_states, s_time, sp_t,
                )
                parts.append(out.squeeze(1).cpu().numpy())
        raw = np.concatenate(parts).reshape(len(x_g), len(y_g))
        # Denormalize: phys = raw * std + mean，使預測場與 DNS 同物理單位。
        return raw * _denorm_std[comp_idx] + _denorm_mean[comp_idx]

    def query_at_sensor_positions(comp_idx: int, t_arr: np.ndarray) -> np.ndarray:
        """What: 在固定 sensor 位置、給定時刻陣列 query model。Returns [T_q, K] (physical units).

        Why: 與 training objective 對齊的 sanity check — sensor MSE on K observation points
             直接告訴你 data loss 收斂到哪個 level；field-level RMSE 受插值 + grid 平均
             混淆，無法分離「sensor 對齊好」與「外推差」。
        """
        K_pts = sensor_pos.shape[0]
        T_q = len(t_arr)
        sp_q = torch.tensor(sensor_pos, dtype=torch.float32, device=device)  # [K, 2]
        preds = np.empty((T_q, K_pts), dtype=np.float32)
        with torch.no_grad():
            for ti, t_val in enumerate(t_arr):
                t_b = torch.full((K_pts,), float(t_val), dtype=torch.float32, device=device)
                c_b = torch.full((K_pts,), comp_idx, dtype=torch.long, device=device)
                out = model.query_decoder(sp_q, t_b, c_b, h_states, s_time, sp_t)
                preds[ti] = out.squeeze(1).cpu().numpy()
        return preds * _denorm_std[comp_idx] + _denorm_mean[comp_idx]

    # I8: dx/dy 分開，未來若支援非正方 grid 不會 silent 算錯。
    dx = float(x_g[1] - x_g[0]) if len(x_g) > 1 else 1.0
    dy = float(y_g[1] - y_g[0]) if len(y_g) > 1 else 1.0
    if not np.isclose(dx, dy):
        raise ValueError(
            f"目前 spectrum / vorticity / divergence 假設 dx==dy（isotropic FFT），"
            f"但 coarse grid dx={dx:.6e}, dy={dy:.6e} 不一致。"
        )

    # I5: time alignment sanity — 對 sensor_time 預先做一次性 dns_idx mapping，
    #     若任一點對齊偏差 > 0.5·dns_dt 立即 raise；防 silent 全部 collapse 到 dns[0]。
    t_dns = dns["time"].astype(np.float64)
    if len(t_dns) < 2:
        raise RuntimeError(f"DNS time axis 長度 {len(t_dns)}（< 2），無法對齊")
    _dns_dt = float(np.median(np.diff(t_dns)))
    if _dns_dt <= 0:
        raise RuntimeError(f"DNS dt 非正：{_dns_dt}")
    sensor_to_dns_idx = np.empty(len(sensor_time), dtype=np.int64)
    for _i, _tv in enumerate(sensor_time):
        # find_dns_time_idx 內含 two-tier 對齊（nearest with ULP tolerance → floor），
        # 已吸收 f32 round-trip 損失，無需外層 _eps 後處理。
        _ix = find_dns_time_idx(t_dns, float(_tv))
        # Sanity: 即使 floor 也可能差太多（sensor_time 完全錯軸時）→ raise 而非 silent
        _diff = abs(float(t_dns[_ix]) - float(_tv))
        if _diff > 0.5 * _dns_dt:
            raise RuntimeError(
                f"DNS time alignment 失敗 (sensor i={_i} t={_tv:.4f})："
                f"floor DNS t={t_dns[_ix]:.4f}, 差 {_diff:.4e} > 0.5·dt={0.5 * _dns_dt:.4e}"
            )
        sensor_to_dns_idx[_i] = _ix

    # IMP-2: 套用 --eval-stride，subsample sensor_time 到 eval 用陣列。
    #         T_sub 大時可大幅加速；NS residual 的時間導數對 nonuniform spacing 仍正確
    #         （np.gradient(field, time_vals) 接 nonuniform）。stride=1 等於原行為。
    if args.eval_stride < 1:
        raise ValueError(f"--eval-stride 必須 ≥ 1，收到 {args.eval_stride}")
    if getattr(args, "eval_on_dns_grid", False):
        # Scheme-B：eval 時刻脫離 sensor grid，改用完整 DNS grid。context (h_states, s_time)
        # 已於上方 model.encode 由間斷 sensor 序列建好、此處不動；只換 query 時刻與對應 DNS 參考。
        # 使被丟棄的 gap 時刻也被 query → 可量 gap-time 重建（間斷 sensor 測試核心）。
        sensor_time = t_dns.astype(np.float32)
        sensor_to_dns_idx = np.arange(len(t_dns), dtype=np.int64)
    eval_tidx = np.arange(0, len(sensor_time), args.eval_stride, dtype=np.int64)
    sensor_time = sensor_time[eval_tidx]                      # subsample
    sensor_to_dns_idx = sensor_to_dns_idx[eval_tidx]
    if len(sensor_time) < 2:
        raise RuntimeError(
            f"--eval-stride={args.eval_stride} 太大，subsample 後只有 {len(sensor_time)} 個時間點"
            f"（NS residual 需 ≥ 2）。"
        )

    # C3: train/val 標註 — Kolmogorov dataset 的 train_t_idx / val_t_idx 是 DNS time index，
    #     evaluator loop over sensor_time → dns_idx，依 dns_idx 屬於哪邊判斷。
    #     PINN sparse-data inversion 是 transductive setting，但仍須區分以驗證 generalization。
    _train_dns_set = set(int(i) for i in ds.train_t_idx.tolist())
    sensor_is_train = np.array([int(i) in _train_dns_set for i in sensor_to_dns_idx], dtype=bool)
    sensor_is_val   = ~sensor_is_train

    u_pred_series = []
    v_pred_series = []
    p_pred_series = []
    u_ref_series = []
    v_ref_series = []
    p_ref_series = []
    for _i, t_val in enumerate(sensor_time):
        dns_idx = int(sensor_to_dns_idx[_i])
        u_pred_series.append(query_field(0, float(t_val)).astype(np.float32))
        v_pred_series.append(query_field(1, float(t_val)).astype(np.float32))
        p_pred_series.append(query_field(2, float(t_val)).astype(np.float32))
        u_ref_series.append(block_avg(dns["u"][dns_idx].astype(np.float32), factor=args.eval_block_factor))
        v_ref_series.append(block_avg(dns["v"][dns_idx].astype(np.float32), factor=args.eval_block_factor))
        p_ref_series.append(block_avg(dns["p"][dns_idx].astype(np.float32), factor=args.eval_block_factor))

    u_pred_arr = np.stack(u_pred_series, axis=0)
    v_pred_arr = np.stack(v_pred_series, axis=0)
    p_pred_arr = np.stack(p_pred_series, axis=0)
    u_ref_arr = np.stack(u_ref_series, axis=0)
    v_ref_arr = np.stack(v_ref_series, axis=0)
    p_ref_arr = np.stack(p_ref_series, axis=0)

    u_rmse = np.sqrt(np.mean((u_pred_arr - u_ref_arr) ** 2, axis=(1, 2)))
    v_rmse = np.sqrt(np.mean((v_pred_arr - v_ref_arr) ** 2, axis=(1, 2)))
    # Field-level relative L2 error: ‖pred - ref‖₂ / ‖ref‖₂ (per time-step)
    # 對齊 PINN/CFD 文獻（Wang 2022, jaxpi）的標準誤差度量。
    u_rel_l2 = np.sqrt(np.sum((u_pred_arr - u_ref_arr) ** 2, axis=(1, 2))) / np.maximum(
        np.sqrt(np.sum(u_ref_arr ** 2, axis=(1, 2))), 1.0e-12
    )
    v_rel_l2 = np.sqrt(np.sum((v_pred_arr - v_ref_arr) ** 2, axis=(1, 2))) / np.maximum(
        np.sqrt(np.sum(v_ref_arr ** 2, axis=(1, 2))), 1.0e-12
    )
    pred_std_u = u_pred_arr.std(axis=(1, 2))
    pred_std_v = v_pred_arr.std(axis=(1, 2))
    ke_pred_series = 0.5 * np.mean(u_pred_arr**2 + v_pred_arr**2, axis=(1, 2))
    ke_ref_series = 0.5 * np.mean(u_ref_arr**2 + v_ref_arr**2, axis=(1, 2))
    ke_rel_err = np.abs(ke_pred_series - ke_ref_series) / np.maximum(ke_ref_series, 1.0e-12)

    # 向量化：vorticity_fd 直接接受 [T, N, N]
    omega_pred_arr = vorticity_fd(u_pred_arr, v_pred_arr, dx)
    omega_ref_arr = vorticity_fd(u_ref_arr, v_ref_arr, dx)
    omega_rmse = np.sqrt(np.mean((omega_pred_arr - omega_ref_arr) ** 2, axis=(1, 2)))
    omega_rel_l2 = np.sqrt(np.sum((omega_pred_arr - omega_ref_arr) ** 2, axis=(1, 2))) / np.maximum(
        np.sqrt(np.sum(omega_ref_arr ** 2, axis=(1, 2))), 1.0e-12
    )
    ens_pred_series = 0.5 * np.mean(omega_pred_arr**2, axis=(1, 2))
    ens_ref_series = 0.5 * np.mean(omega_ref_arr**2, axis=(1, 2))
    ens_rel_err = np.abs(ens_pred_series - ens_ref_series) / np.maximum(ens_ref_series, 1.0e-12)

    # Energy balance diagnostic（穩態 sanity check）：
    #   dKE/dt = ⟨f·u⟩ − ε,  其中 2D incompressible periodic 域 ε = ν⟨ω²⟩ = 2νΩ
    #   穩態 ⇒ ⟨f·u⟩ ≈ 2νΩ；residual = (P_in − ε) / max(ε, eps) 應趨近 0
    #
    # 關鍵分流：DNS 滿足 NS with TRUTH forcing；PI-CON 滿足 NS with LEARNED forcing。
    #   pred-side balance：用 model.forcing (learned) — 量「PI-CON 自洽穩態」
    #   ref-side  balance：用 truth (config)        — DNS 物理穩態
    # 若兩 pattern 共用 learned，DNS residual 變成「u_DNS 對 learned mode 的投影量」，
    # 不再代表物理。（Round 2 audit B1）
    _A_truth_balance = float(cfg.get("kolmogorov_A", 0.1))
    _kf_truth_balance = float(cfg.get("kolmogorov_k_f", 2.0))
    if hasattr(model, "forcing"):
        _f_snap = model.forcing.snapshot()
        _A_pred = float(_f_snap["A"])
        _kf_pred = float(_f_snap["k_f"])
    else:
        _A_pred = _A_truth_balance
        _kf_pred = _kf_truth_balance
    _y_phys = np.asarray(y_g, dtype=np.float64)
    _L_for_forcing = float(cfg.get("domain_length", 1.0))
    _forcing_pattern_pred = _A_pred * np.sin(
        2.0 * np.pi * _kf_pred * _y_phys / _L_for_forcing
    )
    _forcing_pattern_truth = _A_truth_balance * np.sin(
        2.0 * np.pi * _kf_truth_balance * _y_phys / _L_for_forcing
    )
    # forcing 只有 x 分量 (f = A·sin(k_f·y) e_x)，與 u 內積、對 (x,y) 平均
    power_input_series = np.mean(
        u_pred_arr * _forcing_pattern_pred[None, None, :], axis=(1, 2)
    )  # [T]
    dissipation_series = (1.0 / float(re_value)) * np.mean(
        omega_pred_arr ** 2, axis=(1, 2)
    )  # ε = ν⟨ω²⟩
    power_balance_residual = (
        (power_input_series - dissipation_series)
        / np.maximum(np.abs(dissipation_series), 1.0e-12)
    )
    # DNS 穩態對照 — 用 truth forcing
    power_input_ref = np.mean(
        u_ref_arr * _forcing_pattern_truth[None, None, :], axis=(1, 2)
    )
    dissipation_ref = (1.0 / float(re_value)) * np.mean(
        omega_ref_arr ** 2, axis=(1, 2)
    )
    power_balance_residual_ref = (
        (power_input_ref - dissipation_ref)
        / np.maximum(np.abs(dissipation_ref), 1.0e-12)
    )

    # 向量化：divergence_fd 直接接受 [T, N, N]
    div_pred_arr = divergence_fd(u_pred_arr, v_pred_arr, dx)
    div_ref_arr = divergence_fd(u_ref_arr, v_ref_arr, dx)
    div_l2_pred = np.sqrt(np.mean(div_pred_arr**2, axis=(1, 2)))
    div_linf_pred = np.max(np.abs(div_pred_arr), axis=(1, 2))
    div_l2_ref = np.sqrt(np.mean(div_ref_arr**2, axis=(1, 2)))
    div_linf_ref = np.max(np.abs(div_ref_arr), axis=(1, 2))

    # Q7 pressure-gradient metric:
    #   incompressible NS 中只有 ∇p 進入 momentum equation；p 本身有 gauge freedom (p→p+C)。
    #   model 訓練只間接看到 ∇p (透過 momentum residual)，故唯一物理上有意義的比較是 ∇p。
    #   gauge-removed p 值差別本身對動力學無影響；改報 ∇p 相對 L2 誤差。
    dpdx_pred, dpdy_pred = np.gradient(p_pred_arr, dx, axis=(1, 2))
    dpdx_ref, dpdy_ref = np.gradient(p_ref_arr, dx, axis=(1, 2))
    grad_p_pred_norm_sq = dpdx_pred ** 2 + dpdy_pred ** 2
    grad_p_ref_norm_sq = dpdx_ref ** 2 + dpdy_ref ** 2
    grad_p_diff_sq = (dpdx_pred - dpdx_ref) ** 2 + (dpdy_pred - dpdy_ref) ** 2
    grad_p_rel_l2 = np.sqrt(np.sum(grad_p_diff_sq, axis=(1, 2))) / np.maximum(
        np.sqrt(np.sum(grad_p_ref_norm_sq, axis=(1, 2))), 1.0e-12
    )
    grad_p_rms_dns_per_t = np.sqrt(np.mean(grad_p_ref_norm_sq, axis=(1, 2)))
    grad_p_rms_pred_per_t = np.sqrt(np.mean(grad_p_pred_norm_sq, axis=(1, 2)))
    # 保留 gauge-removed p 值對照（次要 diagnostic，供 reviewer 看）
    p_pred_gr = p_pred_arr - p_pred_arr.mean(axis=(1, 2), keepdims=True)
    p_ref_gr = p_ref_arr - p_ref_arr.mean(axis=(1, 2), keepdims=True)
    p_rel_l2_gauge_removed = np.sqrt(np.sum((p_pred_gr - p_ref_gr) ** 2, axis=(1, 2))) / np.maximum(
        np.sqrt(np.sum(p_ref_gr ** 2, axis=(1, 2))), 1.0e-12
    )
    p_rms_dns_per_t = np.sqrt(np.mean(p_ref_gr ** 2, axis=(1, 2)))
    p_rms_pred_per_t = np.sqrt(np.mean(p_pred_gr ** 2, axis=(1, 2)))
    # ‖∇u‖_F strain-rate Frobenius norm（per-t）+ div ratio = ‖∇·u‖_2 / ‖∇u‖_F^DNS
    # 為 §subsec:dns_verification 的 div_ratio 提供 evaluator 端數值。
    dudx_ref, dudy_ref = np.gradient(u_ref_arr, dx, axis=(1, 2))
    dvdx_ref, dvdy_ref = np.gradient(v_ref_arr, dx, axis=(1, 2))
    grad_u_frob_ref = np.sqrt(np.mean(
        dudx_ref ** 2 + dudy_ref ** 2 + dvdx_ref ** 2 + dvdy_ref ** 2, axis=(1, 2)
    ))
    grad_u_frob_ref_meant = float(grad_u_frob_ref.mean())
    div_ratio_pred = div_l2_pred / max(grad_u_frob_ref_meant, 1.0e-12)
    div_ratio_ref = div_l2_ref / max(grad_u_frob_ref_meant, 1.0e-12)

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
    _domain_length = float(cfg.get("domain_length", 1.0))
    # k_f 來源分流：DNS reference 永遠用 truth (cfg["kolmogorov_k_f"])，預測場用 model
    # 學到的 k_f（learn_forcing_k_f=True 才會偏離 truth；fixed mode 等於 truth）。
    # 若 learn_k_f=True 收斂到別的 wavenumber，把預測場投到 truth k_f 上是錯的 diagnostic。
    _kf_truth = float(cfg["kolmogorov_k_f"])
    if hasattr(model, "forcing"):
        _kf_pred_proj = float(model.forcing.snapshot()["k_f"])
    else:
        _kf_pred_proj = _kf_truth
    for idx, t_val in enumerate(sensor_time):
        amp_ref, phase_ref = forcing_mode_coeff_u(
            u_ref_arr[idx], y_g, _kf_truth, domain_length=_domain_length,
        )
        amp_pred, phase_pred = forcing_mode_coeff_u(
            u_pred_arr[idx], y_g, _kf_pred_proj, domain_length=_domain_length,
        )
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
                "split": "train" if bool(sensor_is_train[idx]) else "val",
                "u_rmse": float(u_rmse[idx]),
                "v_rmse": float(v_rmse[idx]),
                "omega_rmse": float(omega_rmse[idx]),
                "u_rel_l2": float(u_rel_l2[idx]),
                "v_rel_l2": float(v_rel_l2[idx]),
                "omega_rel_l2": float(omega_rel_l2[idx]),
                "u_std": float(pred_std_u[idx]),
                "v_std": float(pred_std_v[idx]),
                "ke_rel_err": float(ke_rel_err[idx]),
                "ens_rel_err": float(ens_rel_err[idx]),
                "div_l2":     float(div_l2_pred[idx]),
                "div_ref_l2": float(div_l2_ref[idx]),     # 與 cylinder summary_steps 對稱
                "div_linf":   float(div_linf_pred[idx]),
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

    # Optional npz dump for multi-seed envelope aggregation; see
    # scripts/plot_multiseed_envelope.py.
    if args.export_arrays:
        np.savez(
            output_dir / "series.npz",
            time=sensor_time,
            KE=ke_pred_series,
            KE_dns=ke_ref_series,
            KE_rel_err=ke_rel_err,
            enstrophy=ens_pred_series,
            enstrophy_dns=ens_ref_series,
            div_l2=div_l2_pred,
            div_l2_dns=div_l2_ref,
            div_ratio=div_ratio_pred,
            div_ratio_dns=div_ratio_ref,
            u_rel_L2=u_rel_l2,
            v_rel_L2=v_rel_l2,
            omega_rel_L2=omega_rel_l2,
            u_rmse=u_rmse,
            v_rmse=v_rmse,
            omega_rmse=omega_rmse,
            kf_amp=kf_amp_pred_series,
            kf_amp_dns=kf_amp_ref_series,
            kf_phase=kf_phase_pred_series,
            kf_phase_dns=kf_phase_ref_series,
            band_low=band_rel_err_series["low"],
            band_mid=band_rel_err_series["mid"],
            band_high=band_rel_err_series["high"],
        )
        print(f"[export_arrays] wrote {output_dir / 'series.npz'}")

    # 全場時空陣列（pred + ref），供 mean-profile / Reynolds-stress / 時間頻譜後處理。
    # 與 series.npz 分檔，避免膨脹既有多 seed 聚合流程。
    if args.export_fields:
        np.savez_compressed(
            output_dir / "fields.npz",
            time=sensor_time,
            x_grid=x_g,
            y_grid=y_g,
            u_pred=u_pred_arr,
            v_pred=v_pred_arr,
            omega_pred=omega_pred_arr,
            u_ref=u_ref_arr,
            v_ref=v_ref_arr,
            omega_ref=omega_ref_arr,
        )
        print(f"[export_fields] wrote {output_dir / 'fields.npz'}")

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
        k_sensor_nyquist=float(np.sqrt(float(len(sensor_pos)) / float(np.pi))),
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
        title="Kinetic Energy",
        y_label=r"Kinetic Energy [m$^2$/s$^2$]",
    )
    plot_metric_vs_time(
        output_dir / "enstrophy_vs_time.png",
        sensor_time,
        np.asarray(ens_ref_series),
        np.asarray(ens_pred_series),
        title="Enstrophy",
        y_label=r"Enstrophy [1/s$^2$]",
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
        title=f"Forcing Mode Amplitude ($k_f={float(cfg['kolmogorov_k_f']):.0f}$)",
        y_label=r"$|\hat{u}_{k_f}|$ [m/s]",
    )
    plot_mode_vs_time(
        output_dir / "kf_mode_phase_vs_time.png",
        sensor_time,
        np.unwrap(kf_phase_ref_series),
        np.unwrap(kf_phase_pred_series),
        title=f"Forcing Mode Phase ($k_f={float(cfg['kolmogorov_k_f']):.0f}$)",
        y_label=r"$\arg(\hat{u}_{k_f})$ [rad]",
    )
    plot_series_collection(
        output_dir / "vorticity_error_vs_time.png",
        sensor_time,
        {"Omega RMSE": omega_rmse},
        title="Vorticity Error",
        y_label=r"$\omega$ RMSE [1/s]",
    )
    plot_series_collection(
        output_dir / "divergence_vs_time.png",
        sensor_time,
        {"DNS L2": div_l2_ref, "PI-CON L2": div_l2_pred},
        title="Divergence Residual",
        y_label=r"$\Vert\nabla\cdot\mathbf{u}\Vert_2$ [1/s]",
        yscale="log",
    )
    plot_series_collection(
        output_dir / "ns_residual_vs_time.png",
        sensor_time,
        {
            "DNS NS-u": ns_u_rms_ref,
            "PI-CON NS-u": ns_u_rms_pred,
            "DNS NS-v": ns_v_rms_ref,
            "PI-CON NS-v": ns_v_rms_pred,
            "DNS Cont": ns_cont_rms_ref,
            "PI-CON Cont": ns_cont_rms_pred,
        },
        title="NS Residual",
        y_label=r"NS residual r.m.s. [m/s$^2$]",
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
        title="Band Energy Relative Error",
        y_label=r"Band relative error [-]",
        yscale="log",
    )
    # Energy balance：⟨f·u⟩ vs 2νΩ — 穩態下兩線應重疊。差距代表 PI-CON 場非物理穩態。
    plot_series_collection(
        output_dir / "energy_balance_vs_time.png",
        sensor_time,
        {
            r"$\langle f\!\cdot\!u\rangle$ DNS":   power_input_ref,
            r"$\langle f\!\cdot\!u\rangle$ PI-CON": power_input_series,
            r"$\varepsilon=\nu\langle\omega^2\rangle$ DNS":   dissipation_ref,
            r"$\varepsilon=\nu\langle\omega^2\rangle$ PI-CON": dissipation_series,
        },
        title="Energy Balance (input vs dissipation)",
        y_label=r"Power [m$^2$/s$^3$]",
    )

    time_local = {
        "u_rmse": summarize_time_local_metric(sensor_time, u_rmse),
        "v_rmse": summarize_time_local_metric(sensor_time, v_rmse),
        "omega_rmse": summarize_time_local_metric(sensor_time, omega_rmse),
        "u_rel_l2": summarize_time_local_metric(sensor_time, u_rel_l2),
        "v_rel_l2": summarize_time_local_metric(sensor_time, v_rel_l2),
        "omega_rel_l2": summarize_time_local_metric(sensor_time, omega_rel_l2),
        "ke_rel_err": summarize_time_local_metric(sensor_time, ke_rel_err),
        "div_l2": summarize_time_local_metric(sensor_time, div_l2_pred),
        "ns_u_rms": summarize_time_local_metric(sensor_time, ns_u_rms_pred),
        "ns_v_rms": summarize_time_local_metric(sensor_time, ns_v_rms_pred),
        "ns_cont_rms": summarize_time_local_metric(sensor_time, ns_cont_rms_pred),
        "band_rel_err_low": summarize_time_local_metric(sensor_time, band_rel_err_series["low"]),
        "band_rel_err_mid": summarize_time_local_metric(sensor_time, band_rel_err_series["mid"]),
        "band_rel_err_high": summarize_time_local_metric(sensor_time, band_rel_err_series["high"]),
    }

    # C3: train/val split — 重要 caveat：訓練端 sample_sensor_batch 目前**沒有**
    #     按 dataset.train_t_idx 過濾 supervision pool（src/kolmogorov_dataset.py:140,
    #     src/pi_con/training.py 用 ds.sample_sensor_batch caller 群均如此），所以
    #     evaluator 報的 `*_val` 是 dataset 內部 random partition (transductive)，
    #     不是嚴格意義上的 unseen-by-training metric。
    def _add_split(out: dict, key: str, arr: np.ndarray) -> None:
        """Backward-compat schema: plain float on `key` + `key_train` + `key_val`。

        Why: 之前下游 (compare_experiments.py) 預期 plain float；新增 split 用 suffix
             避免 break。空集合的 split 不寫對應 key。
        """
        # nanmean 防 MPS 偶發 NaN 污染整個 summary；空陣列直接給 NaN 不寫
        out[key] = float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")
        if sensor_is_train.any() and np.isfinite(arr[sensor_is_train]).any():
            out[f"{key}_train"] = float(np.nanmean(arr[sensor_is_train]))
        if sensor_is_val.any() and np.isfinite(arr[sensor_is_val]).any():
            out[f"{key}_val"] = float(np.nanmean(arr[sensor_is_val]))

    summary: dict = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "re_value":      float(re_value),
        "re_norm":       float(re_norm),
        "domain_length": float(_domain_length),
        "k_forcing":     float(_kf_truth),
        "n_eval_steps":  int(len(sensor_time)),
        "n_eval_train":  int(sensor_is_train.sum()),
        "n_eval_val":    int(sensor_is_val.sum()),
        "grid_n":        int(len(x_g)),
        "dx":            float(dx),
        "dy":            float(dy),
        # B3: dt_phys 直接用 dataset 的（與 cylinder 對稱，single source of truth）
        "dt_phys":       float(ds.dt_phys),
        "apply_denormalization": bool(args.apply_denormalization),
        # CRIT-3: cfg-level metadata for reproducibility / drift detection
        "eval_stride":     int(args.eval_stride),
        "train_ratio":     float(TRAIN_RATIO_FALLBACK),
        "ds_seed":         int(ds_seed),
        "u_rel_l2_last":   float(u_rel_l2[-1]),
        "v_rel_l2_last":   float(v_rel_l2[-1]),
        "omega_rel_l2_last": float(omega_rel_l2[-1]),
        "u_std_mean":      float(np.mean(pred_std_u)),
        "v_std_mean":      float(np.mean(pred_std_v)),
        "div_linf_mean":     float(np.mean(div_linf_pred)),
        "div_ref_linf_mean": float(np.mean(div_linf_ref)),
        # Q7 pressure gradient (primary, physically meaningful) and gauge-removed p (diagnostic)
        "grad_p_rel_l2_mean":   float(np.mean(grad_p_rel_l2)),
        "grad_p_rel_l2_last":   float(grad_p_rel_l2[-1]),
        "grad_p_rms_dns_mean":  float(np.mean(grad_p_rms_dns_per_t)),
        "grad_p_rms_pred_mean": float(np.mean(grad_p_rms_pred_per_t)),
        "p_rel_l2_gauge_removed_mean": float(np.mean(p_rel_l2_gauge_removed)),
        "p_rel_l2_gauge_removed_last": float(p_rel_l2_gauge_removed[-1]),
        "p_rms_dns_mean":  float(np.mean(p_rms_dns_per_t)),
        "p_rms_pred_mean": float(np.mean(p_rms_pred_per_t)),
        # div ratio = ‖∇·u‖_2 / ‖∇u‖_F^DNS (time-mean of DNS strain rate)
        "grad_u_frob_dns_mean": grad_u_frob_ref_meant,
        "div_ratio_pred_mean":  float(np.mean(div_ratio_pred)),
        "div_ratio_ref_mean":   float(np.mean(div_ratio_ref)),
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
    # 主要 metric — 用 backward-compat schema：plain float key + _train/_val suffix
    _add_split(summary, "u_rmse_mean",       u_rmse)
    _add_split(summary, "v_rmse_mean",       v_rmse)
    _add_split(summary, "omega_rmse_mean",   omega_rmse)
    _add_split(summary, "u_rel_l2_mean",     u_rel_l2)
    _add_split(summary, "v_rel_l2_mean",     v_rel_l2)
    _add_split(summary, "omega_rel_l2_mean", omega_rel_l2)
    _add_split(summary, "ke_rel_err_mean",   ke_rel_err)
    _add_split(summary, "ens_rel_err_mean",  ens_rel_err)
    _add_split(summary, "div_l2_mean",       div_l2_pred)
    _add_split(summary, "div_ref_l2_mean",   div_l2_ref)   # 與 cylinder 鍵名統一（R3）
    _add_split(summary, "ns_u_rms_mean",     ns_u_rms_pred)
    _add_split(summary, "ns_v_rms_mean",     ns_v_rms_pred)
    _add_split(summary, "ns_cont_rms_mean",  ns_cont_rms_pred)
    _add_split(summary, "ns_u_rms_ref_mean", ns_u_rms_ref)
    _add_split(summary, "ns_v_rms_ref_mean", ns_v_rms_ref)
    _add_split(summary, "ns_cont_rms_ref_mean", ns_cont_rms_ref)

    # Band edges 元資料：讓下游 aggregator 知道 "low/mid/high" 對應哪個 wavenumber 區間
    # `high` 上限用 grid Nyquist 而非 float("inf")，後者非 strict JSON (RFC 8259)，
    # JavaScript JSON.parse 會炸。grid_n // 2 是實際可解析的最大 wavenumber。
    summary["band_edges"] = {
        "low":  [0.0, BAND_EDGES_K_LOW],
        "mid":  [BAND_EDGES_K_LOW, BAND_EDGES_K_HIGH],
        "high": [BAND_EDGES_K_HIGH, float(len(x_g) // 2)],
    }
    summary["sensor_information_k_max"] = float(
        np.sqrt(float(len(sensor_pos)) / float(np.pi))
    )

    # Sensor-MSE on K observation points：與 training data loss 同義的 sanity check。
    # 取得 model 在 sensor (x_k, y_k, t) 的 prediction，與 sensor_vals 物理值比對。
    # sensor_vals shape [K, T_full, C] (normalized)；先 denormalize 再 subsample 到 eval_tidx。
    _sensor_vals_phys = (
        sensor_vals * sensor_std + sensor_mean   # broadcast [K, T_full, C]
    )
    if getattr(args, "eval_on_dns_grid", False):
        # Scheme-B：eval 時刻是 DNS grid、不是 sensor times → 這個 sanity check 沒有對應真值，
        # 且 eval_tidx 已改為 DNS 長度，直接索引 sensor 陣列會越界。以 NaN 佔位跳過。
        _KN = (len(sensor_time), sensor_pos.shape[0])
        _u_true_K = np.full(_KN, np.nan, dtype=np.float32)
        _v_true_K = np.full(_KN, np.nan, dtype=np.float32)
    else:
        _u_true_K = _sensor_vals_phys[:, eval_tidx, 0].T.astype(np.float32)   # [T_eval, K]
        _v_true_K = _sensor_vals_phys[:, eval_tidx, 1].T.astype(np.float32)
    _u_pred_K = query_at_sensor_positions(0, sensor_time)
    _v_pred_K = query_at_sensor_positions(1, sensor_time)
    _sensor_mse_u = np.mean((_u_pred_K - _u_true_K) ** 2, axis=1)  # [T_eval]
    _sensor_mse_v = np.mean((_v_pred_K - _v_true_K) ** 2, axis=1)
    _sensor_mse_avg = 0.5 * (_sensor_mse_u + _sensor_mse_v)
    _add_split(summary, "sensor_mse_at_K_u_mean", _sensor_mse_u)
    _add_split(summary, "sensor_mse_at_K_v_mean", _sensor_mse_v)
    _add_split(summary, "sensor_mse_at_K_mean",   _sensor_mse_avg)

    # Energy balance metric：穩態下 P_in ≈ ε。residual 對 dissipation 的相對量。
    _add_split(summary, "power_input_mean",            power_input_series)
    _add_split(summary, "dissipation_mean",            dissipation_series)
    _add_split(summary, "power_balance_residual_mean", power_balance_residual)
    _add_split(summary, "power_input_ref_mean",        power_input_ref)
    _add_split(summary, "dissipation_ref_mean",        dissipation_ref)
    _add_split(summary, "power_balance_residual_ref_mean", power_balance_residual_ref)

    # Forcing learning 結果：讓 multi-seed aggregator 可以直接 jq 出來。
    # baseline (fixed forcing) 也記錄（learn_*=false），下游可一致 schema 處理。
    if hasattr(model, "forcing"):
        _fsnap = model.forcing.snapshot()
        _A_truth = float(cfg.get("kolmogorov_A", 0.1))
        _kf_truth = float(cfg.get("kolmogorov_k_f", 2.0))
        summary["forcing"] = {
            "A_learned": _fsnap["A"],
            "k_f_learned": _fsnap["k_f"],
            "A_truth": _A_truth,
            "k_f_truth": _kf_truth,
            "A_err_pct": abs(_fsnap["A"] - _A_truth) / max(abs(_A_truth), 1e-12) * 100.0,
            "k_f_err_pct": abs(_fsnap["k_f"] - _kf_truth) / max(abs(_kf_truth), 1e-12) * 100.0,
            "learn_A": bool(_fsnap["learn_A"]),
            "learn_k_f": bool(_fsnap["learn_k_f"]),
        }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _fmt(key: str) -> str:
        """格式化 plain mean + train/val（若 summary 中存在 _train / _val key）。"""
        parts = [f"all={summary[key]:.4e}"]
        if f"{key}_train" in summary:
            parts.append(f"train={summary[f'{key}_train']:.4e}")
        if f"{key}_val" in summary:
            parts.append(f"val={summary[f'{key}_val']:.4e}")
        return "  ".join(parts)

    print("=== DeepONet+CfC Evaluation ===")
    print(f"checkpoint: {args.checkpoint.resolve()}")
    print(f"n_eval: {summary['n_eval_steps']} (train={summary['n_eval_train']}, val={summary['n_eval_val']})")
    print(f"u  RMSE  : {_fmt('u_rmse_mean')}")
    print(f"v  RMSE  : {_fmt('v_rmse_mean')}")
    print(f"u  rel-L2: {_fmt('u_rel_l2_mean')}")
    print(f"v  rel-L2: {_fmt('v_rel_l2_mean')}")
    print(f"u std mean  = {summary['u_std_mean']:.4e}")
    print(f"v std mean  = {summary['v_std_mean']:.4e}")
    print(f"omega RMSE  : {_fmt('omega_rmse_mean')}")
    print(f"omega rel-L2: {_fmt('omega_rel_l2_mean')}")
    print(f"KE rel-err  : {_fmt('ke_rel_err_mean')}")
    print(f"Ens rel-err : {_fmt('ens_rel_err_mean')}")
    print(f"div L2 mean = {summary['div_l2_mean']:.4e}  (DNS {summary['div_ref_l2_mean']:.4e})")
    print(
        f"div ratio   = {summary['div_ratio_pred_mean']*100:.2f}%  "
        f"(DNS floor {summary['div_ratio_ref_mean']*100:.2f}%, "
        f"|grad u|_F DNS = {summary['grad_u_frob_dns_mean']:.3f})"
    )
    print(
        f"grad p rel-L2 (primary)  = {summary['grad_p_rel_l2_mean']*100:.2f}%  "
        f"(last {summary['grad_p_rel_l2_last']*100:.2f}%, "
        f"DNS |grad p|_rms {summary['grad_p_rms_dns_mean']:.3f}, "
        f"pred {summary['grad_p_rms_pred_mean']:.3f})"
    )
    print(
        f"p rel-L2 (gauge-rm, diag)= {summary['p_rel_l2_gauge_removed_mean']*100:.2f}%  "
        f"(DNS p_rms {summary['p_rms_dns_mean']:.3f}, pred {summary['p_rms_pred_mean']:.3f})"
    )
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
