"""#3 mean-flow profile ⟨u⟩(y) + Reynolds stress ⟨u'v'⟩(y)（低階統計驗證）。

What: 用 eval fields.npz 全場，沿 x 與穩態時段(t>=1) 平均，畫 mean streamwise profile
      與 turbulent momentum flux profile，對照 DNS。
Why : Kolmogorov forcing f_x=A sin(2πk_f y) 在 y 方向產生 mean shear ⟨u⟩(y)。低階統計是
      CFD reviewer 最先檢查、K=100 必抓得到、最能支撐「mean-flow monitoring 適用」的指標，
      原論文缺席。fields 軸序 [t,x,y]（axis1=x, axis2=y），沿 x=axis1 平均得 y-profile。
資料: artifacts/eval_245_seed42_fields/fields.npz。
"""
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pi_con.plot_style import apply_journal_rcparams, DNS as DNS_C, PICON as PRED_C  # noqa: E402

FIELDS = ROOT / "artifacts/eval_245_seed42_fields/fields.npz"
OUTDIR = ROOT / "thesis/figures/results"
OUTDIR.mkdir(parents=True, exist_ok=True)
# DNS_C / PRED_C imported from pi_con.plot_style (Okabe--Ito semantic palette)


def main() -> None:
    f = np.load(FIELDS)
    t, y = f["time"], f["y_grid"]
    up, ur = f["u_pred"], f["u_ref"]  # [t, x, y]
    vp, vr = f["v_pred"], f["v_ref"]

    # 軸 sanity：forcing 在 y → 沿 x(axis1) 平均後 profile 應隨 y 有 k_f=2 結構。
    # 用 DNS 驗證 axis2 變異 >> axis1 殘留。
    prof_y_std = ur.mean(axis=(0, 1)).std()   # 沿 (t,x) 平均，留 y
    prof_x_std = ur.mean(axis=(0, 2)).std()   # 沿 (t,y) 平均，留 x
    assert prof_y_std > 5 * prof_x_std, (
        f"軸序異常：y-profile std={prof_y_std:.4f} 未顯著大於 x-profile std={prof_x_std:.4f}"
    )

    m = t >= 1.0  # 穩態窗
    U_dns = ur[m].mean(axis=(0, 1))
    U_pred = up[m].mean(axis=(0, 1))

    def reynolds_uv(u, v):
        # 瞬時 x-mean 為基準的擾動，沿 (t>=1, x) 平均 → ⟨u'v'⟩(y)
        u_fluc = u - u.mean(axis=1, keepdims=True)
        v_fluc = v - v.mean(axis=1, keepdims=True)
        return (u_fluc * v_fluc)[m].mean(axis=(0, 1))

    R_dns = reynolds_uv(ur, vr)
    R_pred = reynolds_uv(up, vp)

    u_prof_relL2 = np.linalg.norm(U_pred - U_dns) / np.linalg.norm(U_dns)
    r_prof_relL2 = np.linalg.norm(R_pred - R_dns) / np.linalg.norm(R_dns)

    apply_journal_rcparams()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ax1.plot(y, U_dns, color=DNS_C, lw=1.4, label="DNS")
    ax1.plot(y, U_pred, color=PRED_C, ls="--", lw=1.4, label="PI-CON")
    ax1.set_xlabel(r"$y$ [m]")
    ax1.set_ylabel(r"$\langle u\rangle_{x,t}(y)$ [m/s]")
    ax1.set_title("(a) Mean streamwise profile", fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.legend(frameon=True, fontsize=8)

    ax2.plot(y, R_dns, color=DNS_C, lw=1.4, label="DNS")
    ax2.plot(y, R_pred, color=PRED_C, ls="--", lw=1.4, label="PI-CON")
    ax2.set_xlabel(r"$y$ [m]")
    ax2.set_ylabel(r"$\langle u'v'\rangle_{x,t}(y)$ [m$^2$/s$^2$]")
    ax2.set_title("(b) Reynolds shear stress", fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.legend(frameon=True, fontsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"mean_profile_reynolds.{ext}")
    plt.close(fig)

    print(f"[#3] wrote {OUTDIR/'mean_profile_reynolds.pdf'} (+png)")
    print(f"[#3] mean-profile ⟨u⟩(y) rel-L2 = {u_prof_relL2*100:.2f}%")
    print(f"[#3] Reynolds-stress ⟨u'v'⟩(y) rel-L2 = {r_prof_relL2*100:.2f}%")
    print(f"[#3] axis sanity: y-prof std={prof_y_std:.4f} >> x-prof std={prof_x_std:.5f}")


if __name__ == "__main__":
    main()
