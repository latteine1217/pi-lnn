"""Open-loop forward-CFD baseline: POD-projected IC from K sensors + ETDRK4 integration.

What: 從 K=100 個 sensor 的 t=0 量測值，經 POD-rank-40 最小二乘投影重建初始場，
      然後用與 DNS 相同的 ETDRK4 solver 自由積分（open-loop，全程不再同化任何資料），
      量化重建誤差如何隨時間發散。
Why : 回答「為什麼不直接跑 forward CFD 就好」。這個 baseline 的初始場其實比 PI-CON 準
      （u rel-L2 約 5%），但 open-loop 積分 5 秒後 chaotic amplification 把誤差放大約
      29 倍到 150% 以上，場與參考完全去相關 —— 而統計量（KE、enstrophy、能譜）幾乎
      看不出來。這是論文 KE-as-misleading 論點的主要證據。

────────────────────────────────────────────────────────────────────────────
本檔的來歷（2026-07-18）：這是**逆向重建**的配方，不是原始腳本

`reports/forward_cfd_baseline_T5_rank40.{npz,json}` 由一支同名腳本產生，但那支腳本
**不在 repo 也不在 git 全歷史**，只留下產物。論文 appendix07 的 forward-CFD 數字
（u rel-L2 152.8%、v 203.9%、KE -3.85%）全部來自那批產物。

本檔從 json 的 `ic_reconstruction_diag` 指紋反推出原始配方並重建流程。實測比對：

    指紋                    本檔重現      json 原值     判定
    lstsq_residuals_sum     0.076187      0.076048      相符（差 0.18%）
    u rel-L2 @ t=0          5.2572%       5.2091%       相符（差 0.9%）
    alpha_l2_norm           145.18        155.70        不符（差 6.8%）
    KE @ t=0                0.160797      0.161586      不符（差 0.5%）

亦即：**配方正確（POD rank-40、不減平均、前 200 snapshots、DNS QR-pivot sensor、
lstsq 求係數），但無法位元級重現**。

殘差來源已排除兩個嫌疑，歸因如下（2026-07-18 實測）：

  · **不是 solver。** 以 DNS 自身的 t=0 場為初始條件、用本檔的 solver 設定積分，
    可重現 DNS 軌跡到機器精度：t=0.025 s 時 u rel-L2 = 4e-16，其後以混沌方式指數
    成長（t=0.075 s 為 3e-9，t=0.25 s 為 4e-5）。solver 與產生 DNS 的實作一致。
  · **不是 POD 實作。** numpy SVD、scipy SVD（gesvd driver）、method-of-snapshots
    （常規化與否）四種實作給出**完全相同**的 alpha 與重建誤差，故 alpha_l2_norm
    的 6.8% 落差不來自 POD 數值路徑。
  · 剩餘嫌疑為原始腳本中無法從產物反推的細節（snapshot 子集的取法、lstsq 的
    rcond、或觀測向量的組成）。IC 殘差約 1%，已無法再壓低。

上述混沌成長率是關鍵：誤差在 0.25 s 內成長約 11 個數量級。IC 差 1% 時，積分 5 s 後
必然完全去相關，**t=5 的數值不可能落回 152.8%**。這不是任一方有誤，而是混沌系統
對初始條件的敏感性。

因此本檔的定位是：
  · 可以：說明並復現「這個 baseline 是怎麼算的」，把配方留在版本控制裡。
  · 不可以：用本檔的輸出取代 appendix07 既有數字。IC 差約 1%，在混沌系統積分 5 秒後
    會放大成完全不同的相位，t=5 的數值必然與 152.8% 不同。若要改用本檔的數值，
    必須整批更新 appendix07 的表與內文，不可與舊值混用。

另一個不可忽略的不對稱：參考 DNS（N=256）是由 N=1024 降採樣而來（config 內
`source_N=1024`、`downsample_stride=4`），而 forward-CFD 在 N=256 上積分。兩者有效
解析度不同，這本身就是誤差來源之一。
────────────────────────────────────────────────────────────────────────────

用法:
    # 只重建 IC 並比對指紋（秒級，預設）
    uv run python scripts/forward_cfd_baseline.py

    # 另外做完整 open-loop 積分（單執行緒 numpy，約 30 分鐘）
    uv run python scripts/forward_cfd_baseline.py --integrate --out reports/forward_cfd_rerun.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "dns_generator"))

DNS_NPY = ROOT / "data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy"
SENSOR_JSON = ROOT / "data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100.json"
REF_JSON = ROOT / "reports/forward_cfd_baseline_T5_rank40.json"

POD_RANK = 40
POD_SNAPSHOTS = 200  # 前 200 幀（含 t=0）；用 [1:201] 會讓 t=0 落在基底外，誤差跳到 12%


def build_pod(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """POD basis, rank POD_RANK, from the first POD_SNAPSHOTS frames.

    不減時間平均 —— 這是從指紋反推出來的設定：減平均會讓 lstsq 奇異值與
    json 的 leading_singular_values_used 對不上。
    """
    ns = POD_SNAPSHOTS
    X = np.concatenate([u[:ns].reshape(ns, -1), v[:ns].reshape(ns, -1)], axis=1).T
    return np.linalg.svd(X, full_matrices=False)[0][:, :POD_RANK]


def reconstruct_ic(pod: np.ndarray, idx: np.ndarray, u0: np.ndarray, v0: np.ndarray,
                   n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Least-squares fit of the POD coefficients to the K sensor readings at t=0.

    觀測向量排列為 [u at K sensors, v at K sensors]，與 POD 向量 [u field, v field]
    的堆疊順序一致。POD 模態由 divergence-free 的 DNS 場張成，故線性組合自動滿足
    連續方程，不需額外投影。
    """
    A = np.concatenate([pod[idx, :], pod[n * n + idx, :]], axis=0)
    b = np.concatenate([u0.ravel()[idx], v0.ravel()[idx]])
    alpha, res, _, _ = np.linalg.lstsq(A, b, rcond=None)
    rec = pod @ alpha
    return (rec[: n * n].reshape(n, n), rec[n * n:].reshape(n, n), alpha,
            float(res.sum()) if res.size else float("nan"))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--integrate", action="store_true",
                    help="做完整 open-loop 積分（約 30 分鐘）；預設只重建 IC 並比對指紋")
    ap.add_argument("--out", type=Path, default=None, help="--integrate 時的輸出 .npz")
    ap.add_argument("--save-every", type=int, default=100,
                    help="積分時每幾步存一幀（預設 100，對應 dt=2.5e-4 的 0.025 s）")
    a = ap.parse_args()

    d = np.load(DNS_NPY, allow_pickle=True).item()
    u, v, cfg = d["u"], d["v"], d["config"]
    n = u.shape[-1]
    idx = np.array(json.loads(SENSOR_JSON.read_text())["indices"])

    print(f"  DNS  : N={n}  nu={cfg['nu']}  dt={cfg['dt']}  dealias={cfg['dealias_mode']}"
          f"  (downsampled from N={cfg.get('source_N')})")
    print(f"  POD  : rank {POD_RANK} from {POD_SNAPSHOTS} snapshots, no mean subtraction")
    print(f"  Sens : K={len(idx)} from {SENSOR_JSON.name}\n")

    pod = build_pod(u, v)
    u_ic, v_ic, alpha, resid = reconstruct_ic(pod, idx, u[0], v[0], n)

    # ── 指紋比對:對照原始產物,確認配方一致 ──────────────────────────
    ref = json.loads(REF_JSON.read_text())
    diag, m0 = ref["ic_reconstruction_diag"], ref["metrics_at_t0"]
    ke_ic = 0.5 * float(np.mean(u_ic ** 2 + v_ic ** 2))
    checks = [
        ("lstsq_residuals_sum", resid, diag["lstsq_residuals_sum"], 2e-3),
        ("u rel-L2 @ t=0", rel_l2(u_ic, u[0]), m0["u_rel_L2"], 2e-2),
        ("v rel-L2 @ t=0", rel_l2(v_ic, v[0]), m0["v_rel_L2"], 2e-2),
        ("alpha_l2_norm", float(np.linalg.norm(alpha)), diag["alpha_l2_norm"], 1e-2),
        ("KE @ t=0", ke_ic, m0["KE_pred"], 1e-2),
    ]
    print(f"  {'fingerprint':22s} {'this run':>12s} {'original':>12s}   verdict")
    n_ok = 0
    for name, got, want, rtol in checks:
        ok = abs(got - want) <= rtol * abs(want)
        n_ok += ok
        print(f"  {name:22s} {got:12.6f} {want:12.6f}   {'match' if ok else 'DIFFERS'}")
    print(f"\n  → {n_ok}/{len(checks)} fingerprints match. 配方正確但非位元級重現;"
          f"\n    請勿用本檔輸出取代 appendix07 既有數字（見檔頭）。\n")

    if not a.integrate:
        print("  (加 --integrate 做完整 open-loop 積分)")
        return 0

    # ── open-loop 積分:沿用 DNS 的 solver,只覆寫初始場 ─────────────
    from generate_kolmogorov_dns_fp64 import KolmogorovFlowDNS  # noqa: E402

    sim = KolmogorovFlowDNS(
        N=n, L=cfg["L"], nu=cfg["nu"], A=cfg["A"], k_f=cfg["k_f"], dt=cfg["dt"],
        dealias=True, dealias_mode=cfg["dealias_mode"], backend="numpy",
        integrator="etdrk4", enforce_zero_mean=cfg["enforce_zero_mean"],
    )
    # 覆寫 solver 自建的 IC 為 POD 重建場,再套用它自己的 dealias/投影/零均值處理,
    # 讓初始狀態與 DNS run 走同一條前處理路徑。
    sim.U_hat = np.fft.fft2(u_ic) * sim.dealias_mask
    sim.V_hat = np.fft.fft2(v_ic) * sim.dealias_mask
    sim.U_hat, sim.V_hat = sim._project_hat(sim.U_hat, sim.V_hat)
    if cfg["enforce_zero_mean"]:
        sim.U_hat, sim.V_hat = sim._zero_mean_hat(sim.U_hat, sim.V_hat)

    n_steps = int(round(cfg["T_end"] / cfg["dt"]))
    print(f"  integrating {n_steps} steps (dt={cfg['dt']}, T={cfg['T_end']}) ...")
    times, u_err, v_err = [0.0], [rel_l2(u_ic, u[0])], [rel_l2(v_ic, v[0])]
    frame_stride = cfg["save_interval"]  # DNS 每 100 步存一幀 → 對齊參考的時間軸
    for step in range(1, n_steps + 1):
        sim.step_etdrk4()
        if step % a.save_every == 0:
            t = step * cfg["dt"]
            uu = np.real(np.fft.ifft2(sim.U_hat))
            vv = np.real(np.fft.ifft2(sim.V_hat))
            k = step // frame_stride
            if step % frame_stride == 0 and k < u.shape[0]:
                times.append(t)
                u_err.append(rel_l2(uu, u[k]))
                v_err.append(rel_l2(vv, v[k]))
            if step % (frame_stride * 20) == 0:
                print(f"    t={t:5.2f}  u rel-L2={u_err[-1]*100:7.2f}%  "
                      f"v rel-L2={v_err[-1]*100:7.2f}%", flush=True)

    out = a.out or (ROOT / "reports/forward_cfd_rerun_T5_rank40.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, time=np.array(times), u_rel_L2=np.array(u_err),
             v_rel_L2=np.array(v_err),
             u_final=np.real(np.fft.ifft2(sim.U_hat)),
             v_final=np.real(np.fft.ifft2(sim.V_hat)))
    print(f"\n  [saved] {out}")
    print(f"  final: u rel-L2={u_err[-1]*100:.1f}%  v rel-L2={v_err[-1]*100:.1f}%")
    print("  original produced 152.8% / 203.9% — 若差異顯著,屬混沌放大的預期結果,"
          "\n  不代表任一方有誤,但也因此不可互相取代。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
