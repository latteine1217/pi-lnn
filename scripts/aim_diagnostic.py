"""
aim_diagnostic.py — Approximate Inertial Manifold 後處理診斷

What: 對 PI-CON 輸出場做 AIM 修正，量化高頻重建的理論上限。

Why: 從資訊論觀點，K=100 sensor + NS 物理在準靜態 AIM 近似下，
     高頻模態可從低頻場解析計算，不需要額外感測器或 DNS supervision。
     本腳本驗證這個上限：若 AIM 修正後 band_mid/high 改善，代表物理先驗有效；
     改善幅度即為「完整利用 K=100 sensor 資訊」的理論天花板。

AIM 公式（zeroth-order, quasi-static）：
    û_k ≈ P_ℒ(N̂_k(û_{≤k_max})) / (νk²)   for k > k_max
    其中 P_ℒ 是 Leray 投影（確保無散度），N̂ 是從低頻場計算的非線性項。
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "src")))

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from picon_kolmogorov import create_picon_model, load_picon_config
from pi_con.plot_style import apply_journal_rcparams

# Journal style: NeurIPS/ICLR 標準（DPI≥300、Helvetica、4 邊 spines、inner ticks）
apply_journal_rcparams()


# ── AIM 核心計算 ──────────────────────────────────────────────────────────────

def aim_correct(u_pred: np.ndarray, v_pred: np.ndarray,
                Re: float, k_max_low: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Zeroth-order AIM 修正：以低頻場（k≤k_max_low）填補高頻（k>k_max_low）。

    Args:
        u_pred, v_pred: PI-CON 完整輸出場，shape (N, N)
        Re: Reynolds number
        k_max_low: 低頻邊界，AIM 修正 k > k_max_low 的模態

    Returns:
        u_aim, v_aim: AIM 修正後的完整場
    """
    N = u_pred.shape[0]
    nu = 1.0 / Re

    # ── Step 1：從 PI-CON 輸出提取低頻部分 ──
    kx_1d = np.fft.fftfreq(N) * N  # wavenumber in [0..N/2-1, -N/2..-1]
    KX, KY = np.meshgrid(kx_1d, kx_1d, indexing='ij')
    k2 = KX**2 + KY**2
    k_abs = np.sqrt(k2)

    mask_low  = (k_abs <= k_max_low)
    mask_high = (~mask_low)

    u_hat = np.fft.fft2(u_pred)
    v_hat = np.fft.fft2(v_pred)

    # 低頻場（只保留 k≤k_max_low）
    u_hat_low = u_hat * mask_low
    v_hat_low = v_hat * mask_low
    u_low = np.real(np.fft.ifft2(u_hat_low))
    v_low = np.real(np.fft.ifft2(v_hat_low))

    # ── Step 2：從低頻場計算非線性項 N = -(u·∇)u ──
    # 使用 pseudo-spectral（乘積在物理空間，微分在 Fourier 空間）
    # 空間微分：∂u/∂x = IFFT(ikx * û)
    two_pi = 2.0 * np.pi
    kx_phys = KX * two_pi  # 實際 wavenumber in rad/m (L=1)
    ky_phys = KY * two_pi

    du_dx = np.real(np.fft.ifft2(1j * kx_phys * u_hat_low))
    du_dy = np.real(np.fft.ifft2(1j * ky_phys * u_hat_low))
    dv_dx = np.real(np.fft.ifft2(1j * kx_phys * v_hat_low))
    dv_dy = np.real(np.fft.ifft2(1j * ky_phys * v_hat_low))

    Nu = -(u_low * du_dx + v_low * du_dy)
    Nv = -(u_low * dv_dx + v_low * dv_dy)

    Nu_hat = np.fft.fft2(Nu)
    Nv_hat = np.fft.fft2(Nv)

    # ── Step 3：Leray 投影（確保 AIM 結果無散度）──
    # P_uu = 1 - kx²/k²,  P_uv = P_vu = -kx ky/k²,  P_vv = 1 - ky²/k²
    k2_phys  = kx_phys**2 + ky_phys**2
    k2_safe  = np.where(k2_phys > 0, k2_phys, 1.0)

    N_proj_u = (1 - kx_phys**2 / k2_safe) * Nu_hat \
               - (kx_phys * ky_phys / k2_safe) * Nv_hat
    N_proj_v = -(kx_phys * ky_phys / k2_safe) * Nu_hat \
               + (1 - ky_phys**2 / k2_safe) * Nv_hat
    # k=0 模態設為 0（mean flow）
    N_proj_u[0, 0] = 0.0
    N_proj_v[0, 0] = 0.0

    # ── Step 4：AIM 公式 û_k = P(N̂_k) / (νk²) for k > k_max_low ──
    nu_k2 = nu * k2_phys
    nu_k2_safe = np.where(nu_k2 > 0, nu_k2, 1.0)

    u_aim_high = np.where(mask_high, N_proj_u / nu_k2_safe, 0.0)
    v_aim_high = np.where(mask_high, N_proj_v / nu_k2_safe, 0.0)

    # ── Step 5：組合低頻（sensor-driven）+ 高頻（AIM） ──
    u_aim_hat = u_hat_low + u_aim_high
    v_aim_hat = v_hat_low + v_aim_high

    return np.real(np.fft.ifft2(u_aim_hat)), np.real(np.fft.ifft2(v_aim_hat))


# ── Band energy 計算 ──────────────────────────────────────────────────────────

def energy_spectrum_1d(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """計算一維旋轉平均能量譜。"""
    N = u.shape[0]
    u_hat = np.fft.fft2(u) / N**2
    v_hat = np.fft.fft2(v) / N**2
    energy = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)

    kx_1d = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(kx_1d, kx_1d, indexing='ij')
    k_abs = np.sqrt(KX**2 + KY**2).ravel()
    e_flat = energy.ravel()

    k_int = np.round(k_abs).astype(int)
    k_max = N // 2
    k_vals = np.arange(1, k_max + 1)
    e_vals = np.zeros(k_max)
    for i, k in enumerate(k_vals):
        mask = (k_int == k)
        e_vals[i] = e_flat[mask].sum() if mask.any() else 0.0
    return k_vals, e_vals


def band_errors(k_vals, e_pred, e_ref,
                bands=((0, 5, "low"), (5, 16, "mid"), (16, 999, "high"))):
    """Band edges right-exclusive ≡ evaluate_deeponet_cfc.compute_band_energies。

    Why: 舊版用 (k_lo <= k <= k_hi) 兩端 inclusive，k=5 與 k=16 被計入 low+mid / mid+high
         double-counted；新版 right-exclusive (k_lo < k <= k_hi) — 與
         evaluate_deeponet_cfc.py:347 BAND_EDGES 對齊。
    """
    result = {}
    for k_lo, k_hi, label in bands:
        idx = np.where((k_vals > k_lo) & (k_vals <= k_hi))[0]
        ep = e_pred[idx].sum(); er = e_ref[idx].sum()
        result[label] = abs(ep - er) / max(er, 1e-12)
    return result


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",     type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/aim_diagnostic"))
    ap.add_argument("--k-max-low",  type=int, default=8,
                    help="AIM 邊界：保留 k≤k_max_low 的 sensor 重建，替換 k>k_max_low")
    ap.add_argument("--n-frames",   type=int, default=10,
                    help="分析的時間幀數（從 T=201 均勻取樣）")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = load_picon_config(args.config)
    Re  = float(cfg.get("re_values", [10000.0])[0])
    nu  = 1.0 / Re

    # ── 載入模型 ──
    model = create_picon_model(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state); model.eval()
    print(f"模型載入：{args.checkpoint.name}")

    # ── 載入 DNS 與 sensor ──
    dns_path     = cfg["dns_paths"][0]
    sensor_json  = cfg["sensor_jsons"][0]
    sensor_npz   = cfg["sensor_npzs"][0]

    dns = np.load(dns_path, allow_pickle=True).item()
    import json
    sensor_meta = json.loads(Path(sensor_json).read_text(encoding="utf-8"))
    sensor_data = np.load(sensor_npz)

    sensor_pos  = np.array(sensor_meta["selected_coordinates"], dtype=np.float32)  # (K, 2)
    sensor_time = sensor_data["time"].astype(np.float32)                            # (T,)

    requested = cfg.get("observed_sensor_channels", ["u", "v"])
    observed  = [sensor_data[ch].astype(np.float32) for ch in requested]
    sensor_vals = np.stack(observed, axis=2)  # (K, T, C)
    sensor_mean = sensor_vals.mean(axis=(0, 1), keepdims=True)
    sensor_std  = np.maximum(sensor_vals.std(axis=(0, 1), keepdims=True), 1e-6)
    sensor_vals = ((sensor_vals - sensor_mean) / sensor_std).astype(np.float32)

    N_grid = 256
    x_g = np.linspace(0, 1, N_grid, endpoint=False).astype(np.float32)
    xx, yy = np.meshgrid(x_g, x_g, indexing='ij')
    xy_flat = np.stack([xx.ravel(), yy.ravel()], axis=1)
    xy_t = torch.tensor(xy_flat, device=device)

    sv_t = torch.tensor(sensor_vals.transpose(1, 0, 2), dtype=torch.float32, device=device)
    sp_t = torch.tensor(sensor_pos, device=device)
    st_t = torch.tensor(sensor_time, device=device)
    re_norm = float((Re - 5500.0) / 4000.0)

    with torch.no_grad():
        h_states, s_time = model.encode(sv_t, sp_t, re_norm, st_t)

    def query(comp: int, t_val: float) -> np.ndarray:
        parts = []
        with torch.no_grad():
            for i in range(0, xy_t.shape[0], 8192):
                xy_b = xy_t[i:i+8192]
                bs   = xy_b.shape[0]
                t_b  = torch.full((bs,), t_val, dtype=torch.float32, device=device)
                c_b  = torch.full((bs,), comp,  dtype=torch.long,    device=device)
                out  = model.query_decoder(xy_b, t_b, c_b, h_states, s_time, sp_t)  # DeepONetCfCDecoder forward
                parts.append(out.squeeze(1).cpu().numpy())
        return np.concatenate(parts).reshape(N_grid, N_grid)

    # ── 取樣時間幀 ──
    T = len(sensor_time)
    frame_idx = np.linspace(0, T - 1, args.n_frames, dtype=int)
    t_frames  = sensor_time[frame_idx]

    # ── 各幀計算 ──
    results_raw = {"low": [], "mid": [], "high": []}
    results_aim = {"low": [], "mid": [], "high": []}

    for fi, (idx, t_val) in enumerate(zip(frame_idx, t_frames)):
        print(f"  frame {fi+1}/{args.n_frames}  t={t_val:.2f}", flush=True)

        u_pred = query(0, float(t_val))
        v_pred = query(1, float(t_val))

        # DNS reference（block-avg 到 256）
        dns_t_idx = int(np.argmin(np.abs(dns["time"] - float(t_val))))
        u_ref  = dns["u"][dns_t_idx].astype(np.float32)
        v_ref  = dns["v"][dns_t_idx].astype(np.float32)

        # AIM 修正
        u_aim, v_aim = aim_correct(u_pred, v_pred, Re, k_max_low=args.k_max_low)

        # 計算 band errors
        k_vals, ep_raw = energy_spectrum_1d(u_pred, v_pred)
        k_vals, ep_aim = energy_spectrum_1d(u_aim,  v_aim)
        k_vals, ep_ref = energy_spectrum_1d(u_ref,  v_ref)

        for label, ep in [("raw", ep_raw), ("aim", ep_aim)]:
            be = band_errors(k_vals, ep, ep_ref)
            target = results_raw if label == "raw" else results_aim
            for band in ("low", "mid", "high"):
                target[band].append(be[band])

    # ── 彙整結果 ──
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== AIM 後處理診斷結果（k_max_low={args.k_max_low}）===")
    print(f"{'Band':>6}  {'Raw (PI-CON)':>14}  {'AIM 修正後':>12}  {'改善':>10}")
    print("-" * 50)
    for band in ("low", "mid", "high"):
        r = np.mean(results_raw[band]) * 100
        a = np.mean(results_aim[band]) * 100
        delta = r - a
        print(f"  {band:>4}  {r:12.1f}%  {a:10.1f}%  {delta:+8.1f}pp")

    # ── 圖表 ──
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True)
    fig.suptitle(
        rf"AIM Post-processing: $k_{{\max,\mathrm{{low}}}}={args.k_max_low}$, $Re={int(Re)}$",
        fontsize=10,
    )

    # 圖1：Band error 比較（bar chart）
    bands_label = ["low", "mid", "high"]
    x = np.arange(3)
    r_vals = [np.mean(results_raw[b])*100 for b in bands_label]
    a_vals = [np.mean(results_aim[b])*100 for b in bands_label]
    axes[0].bar(x - 0.2, r_vals, 0.35, color="#d62728", label="PI-CON raw", alpha=0.85)
    axes[0].bar(x + 0.2, a_vals, 0.35, color="#1f77b4", label="AIM corrected", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([
        r"low ($k\!\leq\!5$)",
        r"mid ($5\!<\!k\!\leq\!16$)",
        r"high ($k\!>\!16$)",
    ])
    axes[0].set_ylabel("Band energy relative error [\\%]")
    axes[0].set_title("Band Error: Raw vs AIM")
    axes[0].legend()

    # 圖2：最後一幀的能量譜比較
    t_last = float(t_frames[-1])
    u_pred_last = query(0, t_last); v_pred_last = query(1, t_last)
    dns_idx = int(np.argmin(np.abs(dns["time"] - t_last)))
    u_ref_last  = dns["u"][dns_idx].astype(np.float32)
    v_ref_last  = dns["v"][dns_idx].astype(np.float32)
    u_aim_last, v_aim_last = aim_correct(u_pred_last, v_pred_last, Re, k_max_low=args.k_max_low)

    k_v, e_raw  = energy_spectrum_1d(u_pred_last, v_pred_last)
    k_v, e_aim  = energy_spectrum_1d(u_aim_last,  v_aim_last)
    k_v, e_ref2 = energy_spectrum_1d(u_ref_last,  v_ref_last)

    axes[1].loglog(k_v, e_ref2, color="#1f77b4", linestyle="-",  lw=1.4, label="DNS")
    axes[1].loglog(k_v, e_raw,  color="#d62728", linestyle="--", lw=1.2, label="PI-CON raw")
    axes[1].loglog(k_v, e_aim,  color="#2ca02c", linestyle="-",  lw=1.2, label="AIM corrected")
    k53 = k_v[2:40]
    axes[1].loglog(k53, 2e-3 * k53**(-5/3), color="gray", linestyle=":", lw=0.9,
                   label=r"$k^{-5/3}$")
    axes[1].axvline(args.k_max_low, color="black", linestyle="-.", lw=0.9,
                    label=rf"$k_{{\max,\mathrm{{low}}}}={args.k_max_low}$")
    axes[1].set_xlabel(r"Wavenumber $k$ [1/m]")
    axes[1].set_ylabel(r"$E(k)$ [m$^3$/s$^2$]")
    axes[1].set_title(rf"Energy Spectrum at $t={t_last:.1f}$ s")
    axes[1].legend()

    # 圖3：最後一幀渦度比較
    def vorticity(u, v, N):
        N = u.shape[0]
        kx_1d = np.fft.fftfreq(N) * N * 2 * np.pi
        KX, KY = np.meshgrid(kx_1d, kx_1d, indexing='ij')
        return np.real(np.fft.ifft2(1j*KX*np.fft.fft2(v) - 1j*KY*np.fft.fft2(u)))

    om_ref  = vorticity(u_ref_last,  v_ref_last,  256)
    om_aim  = vorticity(u_aim_last,  v_aim_last,  256)
    om_raw  = vorticity(u_pred_last, v_pred_last, 256)

    vmax = np.percentile(np.abs(om_ref), 98)
    err_raw = np.abs(om_raw - om_ref)
    err_aim = np.abs(om_aim - om_ref)
    diff = err_raw - err_aim  # positive = AIM better
    im = axes[2].imshow(diff.T, cmap="RdBu_r", origin="lower",
                        vmin=-vmax*0.3, vmax=vmax*0.3,
                        extent=(0.0, 1.0, 0.0, 1.0), aspect="equal")
    axes[2].set_title("Vorticity error improvement\n(blue: AIM better, red: AIM worse)")
    axes[2].set_xlabel(r"$x$ [m]"); axes[2].set_ylabel(r"$y$ [m]")
    cb = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cb.set_label(r"$|\omega - \omega_{\mathrm{DNS}}|$ delta [1/s]", fontsize=8)
    cb.ax.tick_params(labelsize=8)

    out_fig = args.output_dir / f"aim_diagnostic_klow{args.k_max_low}.png"
    plt.savefig(out_fig, facecolor="white")  # dpi 由 rcParams (300) 接管
    print(f"\n圖表已存：{out_fig}")


if __name__ == "__main__":
    main()
