"""#4 固定探針時間追蹤 + 時間自相關（直接驗證 CfC 連續時間編碼抓到 dynamics）。

What: 用 eval fields.npz 全場，在固定空間點取 u(t)，畫 (a) 代表點 u(t) trace pred vs DNS，
      (b) 空間平均時間自相關 R(τ) pred vs DNS。
Why : PI-CON 賣點之一是 CfC 連續時間編碼，但原論文 CfC 價值僅由 ablation (B1) 間接展示。
      固定探針的時間追蹤與 decorrelation 結構直接證明重建抓到「時間動態」，而非逐幀空間內插。
資料: artifacts/eval_245_seed42_fields/fields.npz。
"""
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pi_con.plot_style import apply_journal_rcparams, DNS as DNS_C, PICON as PRED_C, LES as LES_C  # noqa: E402

FIELDS = ROOT / "artifacts/eval_245_seed42_fields/fields.npz"
OUTDIR = ROOT / "thesis/figures/results"
OUTDIR.mkdir(parents=True, exist_ok=True)
# DNS_C / PRED_C / LES_C imported from pi_con.plot_style (Okabe--Ito semantic palette)


def autocorr_mean(arr: np.ndarray, n_probe: int = 512) -> np.ndarray:
    """空間 subsample 後逐點 per-pixel 自相關（減時間 mean、歸一），再平均。arr: [t,x,y]."""
    Nt = arr.shape[0]
    a = arr.reshape(Nt, -1)
    a = a - a.mean(axis=0, keepdims=True)
    idx = np.linspace(0, a.shape[1] - 1, n_probe).astype(int)
    a = a[:, idx]
    R = np.zeros(Nt)
    for j in range(a.shape[1]):
        s = a[:, j]
        full = np.correlate(s, s, mode="full")[Nt - 1:]
        R += full / (full[0] + 1e-12)
    return R / a.shape[1]


def main() -> None:
    f = np.load(FIELDS)
    t = f["time"]
    up, ur = f["u_pred"], f["u_ref"]  # [t, x, y]
    Nt, Nx, Ny = up.shape

    # 代表點（避開域邊界）
    ix, iy = Nx // 2, Ny // 2
    u_dns_pt = ur[:, ix, iy]
    u_pred_pt = up[:, ix, iy]
    trace_relL2 = np.linalg.norm(u_pred_pt - u_dns_pt) / np.linalg.norm(u_dns_pt)

    tau = t - t[0]
    R_dns = autocorr_mean(ur)
    R_pred = autocorr_mean(up)

    # decorrelation time（R 首次跌破 1/e）
    def decorr_time(R):
        below = np.where(R < np.exp(-1))[0]
        return float(tau[below[0]]) if len(below) else float("nan")

    apply_journal_rcparams()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ax1.plot(t, u_dns_pt, color=DNS_C, lw=1.4, label="DNS")
    ax1.plot(t, u_pred_pt, color=PRED_C, ls="--", lw=1.4, label="PI-CON")
    # same-IC forward LES at the same probe (diagnostic reference, not part of the
    # reconstruction): tracks the DNS for ~1 eddy turnover then diverges.
    les_probe = ROOT / "diagnostics/les_probe_center.npz"
    if les_probe.exists():
        d_les = np.load(les_probe)
        u_les_pt = np.interp(t, d_les["t"], d_les["u_probe"])
        ax1.plot(t, u_les_pt, color=LES_C, ls=":", lw=1.2, label="LES (same IC)")
    ax1.set_xlabel(r"$t$ [s]")
    ax1.set_ylabel(r"$u(\mathbf{x}_0, t)$ [m/s]")
    ax1.set_title(rf"(a) Probe trace at $\mathbf{{x}}_0=(0.5,0.5)$", fontsize=9)
    ax1.set_xlim(0, 5)
    ax1.legend(frameon=True, fontsize=8)

    ax2.plot(tau, R_dns, color=DNS_C, lw=1.4, label="DNS")
    ax2.plot(tau, R_pred, color=PRED_C, ls="--", lw=1.4, label="PI-CON")
    ax2.axhline(np.exp(-1), color="grey", lw=0.6, ls=":")
    ax2.set_xlabel(r"time lag $\tau$ [s]")
    ax2.set_ylabel(r"temporal autocorrelation $R_{uu}(\tau)$")
    ax2.set_title("(b) Velocity time autocorrelation", fontsize=9)
    ax2.set_xlim(0, tau[-1])
    ax2.legend(frameon=True, fontsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"temporal_consistency.{ext}")
    plt.close(fig)

    print(f"[#4] wrote {OUTDIR/'temporal_consistency.pdf'} (+png)")
    print(f"[#4] center-probe u(t) trace rel-L2 = {trace_relL2*100:.2f}%")
    print(f"[#4] decorrelation time τ_1/e: DNS={decorr_time(R_dns):.3f}s, PI-CON={decorr_time(R_pred):.3f}s")


if __name__ == "__main__":
    main()
