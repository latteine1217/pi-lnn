"""Divergence smoothness control experiment.

What: 驗證「sub-DNS divergence」是否為 band-limiting (平滑) 假象。
Why: PI-CON 預測場被 K=100 Nyquist 限制在 k≲5，極平滑；FD 散度截斷誤差 ∝ 高波數含量，
     故平滑場的 FD 散度天生較低。本 control 把 DNS 場譜空間低通到 k≤K_cut（預測場所在頻段），
     再用 evaluator 同款 FD 算子算散度比，回答：
       - 若 band-limited DNS 的 div_ratio ≈ PI-CON 0.39% → sub-DNS 純屬平滑假象
       - 若仍 > 0.39% → AL 約束帶來超越平滑的額外守恆

復用 evaluator 完全相同的 block_avg / divergence_fd / strain-rate 算法（同網格、同分母）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_deeponet_cfc import block_avg, divergence_fd  # noqa: E402

DNS_PATH = "data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy"
DOMAIN_L = 1.0


def spectral_lowpass(field: np.ndarray, k_cut: float, dx: float) -> np.ndarray:
    """譜空間 isotropic 低通：保留 sqrt(kx^2+ky^2) <= k_cut 的模態。

    field: [..., N, N] real。k1d 用 fftfreq(N, d=dx)（與 evaluator energy_spectrum_1d 同單位，
    domain=1 時為整數波數 0..N/2）。
    """
    n = field.shape[-1]
    k1d = np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(k1d, k1d, indexing="ij")
    kk = np.sqrt(kx**2 + ky**2)
    mask = (kk <= k_cut).astype(np.float64)
    fh = np.fft.fft2(field, axes=(-2, -1))
    return np.real(np.fft.ifft2(fh * mask, axes=(-2, -1)))


def div_ratio_series(u: np.ndarray, v: np.ndarray, dx: float, frob_denom: float) -> np.ndarray:
    """每個 t 的 div_ratio = ||div u||_2 / frob_denom（與 evaluator 同義）。"""
    div = divergence_fd(u, v, dx)
    div_l2 = np.sqrt(np.mean(div**2, axis=(-2, -1)))
    return div_l2 / max(frob_denom, 1e-12)


def main() -> None:
    dns = np.load(DNS_PATH, allow_pickle=True).item()
    u256 = dns["u"].astype(np.float64)  # [T, 256, 256]
    v256 = dns["v"].astype(np.float64)

    # evaluator 路徑：block_avg 256 -> 128
    u = block_avg(u256)  # [T, 128, 128]
    v = block_avg(v256)
    n = u.shape[-1]
    dx = DOMAIN_L / n

    # strain-rate Frobenius norm（evaluator line 1143-1148，用 np.gradient）— 全 control 共用同一分母
    dudx, dudy = np.gradient(u, dx, axis=(-2, -1))
    dvdx, dvdy = np.gradient(v, dx, axis=(-2, -1))
    frob = np.sqrt(np.mean(dudx**2 + dudy**2 + dvdx**2 + dvdy**2, axis=(-2, -1)))
    frob_mean = float(frob.mean())

    print(f"=== Divergence Smoothness Control (Re=10000, N=256->128) ===")
    print(f"grid_n={n}  dx={dx:.6g}  strain-rate Frob (mean over t) = {frob_mean:.4f} s^-1")
    print(f"T snapshots = {u.shape[0]}")
    print()

    # baseline：full DNS（block-avg only，無低通）— 應重現 evaluator 的 ~1.04%
    r_full = div_ratio_series(u, v, dx, frob_mean)
    print(f"{'config':<32}{'div_ratio mean':>16}{'div_ratio @t=5(last)':>22}")
    print("-" * 70)
    print(f"{'DNS full (block-avg 128)':<32}{r_full.mean()*100:>14.3f}%{r_full[-1]*100:>20.3f}%")

    # control：DNS 譜空間低通到各 k_cut，再算同款 FD 散度
    for k_cut in (5.0, 8.0, 16.0):
        u_lp = spectral_lowpass(u, k_cut, dx)
        v_lp = spectral_lowpass(v, k_cut, dx)
        r_lp = div_ratio_series(u_lp, v_lp, dx, frob_mean)  # 分母仍用 full DNS strain rate
        label = f"DNS low-pass k<={int(k_cut)}"
        print(f"{label:<32}{r_lp.mean()*100:>14.3f}%{r_lp[-1]*100:>20.3f}%")

    print()
    print("Reference: PI-CON (EXP-271 DNS-pivot) div_ratio_pred mean = 0.362%, @t=5 ~ 0.36%")
    print("           PI-CON (EXP-245 LES-pivot) div_ratio_pred mean = 0.39%")
    print()
    print("判讀：若 'DNS low-pass k<=5' 的 div_ratio ≈ PI-CON 0.36-0.39% → sub-DNS 為平滑假象；")
    print("      若仍顯著 > PI-CON → AL 約束帶來超越 band-limiting 的額外守恆。")


if __name__ == "__main__":
    main()
