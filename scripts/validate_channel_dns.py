#!/usr/bin/env python3
"""驗證 JHTDB channel DNS cutout 的物理品質（Plan 3 Task 3a 後驗證）。

What: 對單一 channel DNS frame 算 channel 金標準診斷量：
      bulk velocity、U⁺(y⁺) vs log-law、Reynolds stress、divergence、上下壁對稱性。
Why:  確認抓取的 strided downsample DNS 物理正確（軸序對、流向對、近壁物理對），
      再投入後續 sensor / 訓練 pipeline。

軸序 [z,y,x,c]，c: 0=u(streamwise), 1=v(wall-normal), 2=w(spanwise)。
注意：stride-4 downsample + finite-diff，divergence 不會達 spectral 機器精度
      （truncation error 必然），重點看 U⁺(y⁺) 與 Reynolds stress 物理形狀。

資料現址: 3D channel 主線已移至 pi-lnn-jax，DNS frames 與 metadata.json 隨之搬到
      `../pi-lnn-jax/data/channel_dns/`（2026-07-29）。下方預設路徑保留原樣，因為
      fetch_channel_dns_jhtdb.py 重新抓取時仍會落在本專案的 data/channel_dns；
      要驗證既有 frames 請顯式給路徑。

用法: uv run python scripts/validate_channel_dns.py [npy_path]
      uv run python scripts/validate_channel_dns.py \\
          ../pi-lnn-jax/data/channel_dns/channel_dns_t0001.npy
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

NU = 5.0e-5
U_TAU = 0.0499           # JHTDB channel Re_τ=1000 官方
DELTA_NU = NU / U_TAU    # viscous length ≈ 1.002e-3
KAPPA, B_LOG = 0.41, 5.2  # log-law constants


def main() -> None:
    npy_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/channel_dns/channel_dns_t0001.npy")
    meta = json.loads((npy_path.parent / "metadata.json").read_text())
    arr = np.load(npy_path).astype(np.float64)   # [Nz, Ny, Nx, 3]
    y = np.asarray(meta["coord_ycoor"])          # B-spline 物理 y (128)
    x = np.asarray(meta["coord_xcoor"])
    z = np.asarray(meta["coord_zcoor"])
    Nz, Ny, Nx, _ = arr.shape
    u, v, w = arr[..., 0], arr[..., 1], arr[..., 2]

    print("=== Channel DNS 驗證 ===")
    print(f"file: {npy_path.name}  shape={arr.shape} (z,y,x,c)")
    print(f"y∈[{y[0]:.4f},{y[-1]:.4f}]  x∈[{x[0]:.3f},{x[-1]:.3f}]  z∈[{z[0]:.3f},{z[-1]:.3f}]")

    # --- 1. Bulk velocity（應 ≈ U_b=1）---
    U_b = float(u.mean())
    print(f"\n--- Bulk velocity ---")
    print(f"U_b = mean(u) = {U_b:.4f}  (目標 ≈ 1.0)  → {'✅' if abs(U_b-1) < 0.15 else '⚠️'}")
    print(f"mean(v) = {v.mean():+.4e}  mean(w) = {w.mean():+.4e}  (對稱性: 應 ≈ 0)")

    # --- 2. Mean profile U(y) + 對稱性 ---
    U_y = u.mean(axis=(0, 2))   # xz-average per y → (Ny,)
    print(f"\n--- Mean profile 對稱性 ---")
    # 上下壁對稱：U(y) ≈ U(-y)
    asym = np.abs(U_y - U_y[::-1]).max() / U_y.max()
    print(f"上下壁 U(y) 對稱性誤差 = {asym*100:.2f}%  → {'✅' if asym < 0.1 else '⚠️'}")
    print(f"U_max(center) = {U_y.max():.4f}  U(wall) = {U_y[0]:.4e}, {U_y[-1]:.4e}  (no-slip: 應 ≈ 0)")

    # --- 3. U⁺(y⁺) vs log-law（channel 金標準）---
    print(f"\n--- U⁺(y⁺) vs log-law (κ={KAPPA}, B={B_LOG}) ---")
    # 下半通道：wall distance = 1 + y
    lower = y <= 0.0
    d = 1.0 + y[lower]
    yp = d / DELTA_NU
    Up = U_y[lower] / U_TAU
    # viscous sublayer (y+<5): U+ ≈ y+
    visc_mask = (yp > 0) & (yp < 5)
    if visc_mask.sum() >= 2:
        visc_err = np.abs(Up[visc_mask] - yp[visc_mask]).mean()
        print(f"viscous sublayer (y⁺<5): mean|U⁺−y⁺| = {visc_err:.3f}  → {'✅' if visc_err < 1.0 else '⚠️'}")
    # log-law (30<y+<150): U+ ≈ (1/κ)ln(y+)+B
    log_mask = (yp > 30) & (yp < 150)
    if log_mask.sum() >= 2:
        Up_loglaw = (1 / KAPPA) * np.log(yp[log_mask]) + B_LOG
        log_err = np.abs(Up[log_mask] - Up_loglaw).mean()
        print(f"log-law (30<y⁺<150): mean|U⁺−loglaw| = {log_err:.3f}  → {'✅' if log_err < 1.5 else '⚠️'}")
    # 印幾個取樣點
    for yp_target in [1, 5, 15, 50, 100]:
        idx = np.argmin(np.abs(yp - yp_target))
        print(f"  y⁺={yp[idx]:6.1f}  U⁺={Up[idx]:6.2f}  (log-law={('%.2f'%((1/KAPPA)*np.log(yp[idx])+B_LOG)) if yp[idx]>1 else '—'})")

    # --- 4. Reynolds stress（per y，xz-average of fluctuations）---
    print(f"\n--- Reynolds stress (峰值位置與量級) ---")
    up = u - U_y[None, :, None]
    V_y = v.mean(axis=(0, 2)); W_y = w.mean(axis=(0, 2))
    vp = v - V_y[None, :, None]; wp = w - W_y[None, :, None]
    uu = (up * up).mean(axis=(0, 2)) / U_TAU**2   # 正規化成 wall units
    vv = (vp * vp).mean(axis=(0, 2)) / U_TAU**2
    ww = (wp * wp).mean(axis=(0, 2)) / U_TAU**2
    uv = (up * vp).mean(axis=(0, 2)) / U_TAU**2
    yp_full = (1.0 + y) / DELTA_NU
    i_peak = np.argmax(uu)
    print(f"u'u'⁺ peak = {uu[i_peak]:.2f} @ y⁺={yp_full[i_peak]:.1f}  (DNS: ~8 @ y⁺≈15) → {'✅' if 4 < uu[i_peak] < 12 else '⚠️'}")
    print(f"v'v'⁺ max = {vv.max():.2f}  w'w'⁺ max = {ww.max():.2f}  (應 u'u' > w'w' > v'v')")
    print(f"  各向異性: u'u'>w'w'? {'✅' if uu.max()>ww.max() else '❌'}  w'w'>v'v'? {'✅' if ww.max()>vv.max() else '❌'}")
    print(f"-u'v'⁺ max = {(-uv).max():.2f}  (應 > 0，動量輸送) → {'✅' if (-uv).max() > 0.3 else '⚠️'}")

    # --- 5. Divergence（strided finite-diff，參考用）---
    print(f"\n--- Divergence (strided finite-diff, 參考) ---")
    du_dx = np.gradient(u, x, axis=2)
    dv_dy = np.gradient(v, y, axis=1)   # 非均勻 y
    dw_dz = np.gradient(w, z, axis=0)
    div = du_dx + dv_dy + dw_dz
    grad_scale = np.abs(du_dx).mean() + np.abs(dv_dy).mean() + np.abs(dw_dz).mean()
    print(f"|div| rms = {np.sqrt((div**2).mean()):.4e}  max = {np.abs(div).max():.4e}")
    print(f"|div|_rms / typical_grad = {np.sqrt((div**2).mean())/grad_scale:.3f}  "
          f"(strided downsample + FD truncation，非 spectral 機器精度；相對量級小即合理)")

    print(f"\n=== 判讀 ===")
    print("strided downsample DNS：div 非機器精度屬正常（FD truncation）。")
    print("核心物理（U⁺(y⁺) 對 log-law、Reynolds stress 各向異性與峰值）若 ✅ → 抓取軸序/流向/近壁物理正確。")


if __name__ == "__main__":
    main()
