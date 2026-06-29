"""Split Conformal Prediction（後處理 UQ）for DeepONet+CfC Kolmogorov reconstruction.

What:
  對已訓練好的 checkpoint（不重訓、不改架構）做 split conformal prediction，
  在 K=100 稀疏感測器 + PDE 重建場上給出帶覆蓋率保證的預測區間。

Why（對齊 AGENTS.md ENGINEERING_VISION / REAL_WORLD_PIPELINE）:
  - Path A（工程可遷移，論文 headline）：calibration 取「均勻隨機 held-out 位置」
    （不在訓練 100 sensor 內），只用點量測值 → 現場可複現。
  - Path B（oracle，僅研究用，工程不可遷移）：calibration 取 DNS 全場密集隨機點。

Adaptive σ（physics-informed，不重訓；取代 Gemini 建議的 CQR）:
  比較三個 nonconformity 尺度函數 σ(x,t)，皆現場可算（工程可遷移）:
    - dist      : 到最近訓練 sensor 的 periodic 距離
    - tempered  : √dist（假設誤差成長比距離緩）
    - residual  : 逐點 NS PDE residual 大小（physics-informed）
  E = |y - ŷ| / σ。任何正 σ 皆保 marginal coverage；conditional coverage 與平均
  寬度才取決於 σ 是否真的追蹤誤差。以「到 sensor 距離」分 bin 比較三者。

Sanity gate:
  本地 multiseed checkpoint 歸屬無法從 experiment_log 確認，先驗 KE rel-err 是否
  落在 EXP-245 預期範圍（~5.7%），抓 (checkpoint, sensor) 配錯的 silent bug。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from picon_kolmogorov import create_picon_model, load_picon_config  # noqa: E402
from kolmogorov_dataset import KolmogorovDataset  # noqa: E402
from pi_con.operator import make_picon_model_fn_uvp  # noqa: E402
from pi_con.physics import unsteady_ns_residuals  # noqa: E402
from evaluate_deeponet_cfc import (  # noqa: E402
    choose_device,
    extract_model_state,
    load_model_weights_strict,
)
from pi_con.plot_style import apply_journal_rcparams  # noqa: E402

apply_journal_rcparams()

EXP245_KE_REL_ERR = 0.0571
GATE_KE_REL_ERR_MAX = 0.12
SIGMA_NAMES = ("dist", "tempered", "residual")


# ──────────────────────────────────────────────────────────────────────────
# 模型載入 + 一次性 encode
# ──────────────────────────────────────────────────────────────────────────
def load_model_and_encode(cfg, ckpt_path, device):
    """建模型、load strict、重建 dataset、一次性 encode sensor → h_states。"""
    model = create_picon_model(cfg).to(device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    load_model_weights_strict(model, extract_model_state(payload))  # 架構不符 fail loud
    model.eval()

    ds = KolmogorovDataset(
        sensor_json=cfg["sensor_jsons"][0],
        sensor_npz=cfg["sensor_npzs"][0],
        dns_path=cfg["dns_paths"][0],
        re_value=float(cfg.get("re_values", [1000.0])[0]),
        observed_channel_names=tuple(cfg.get("observed_sensor_channels", ["u", "v"])),
        train_ratio=0.8,
        seed=int(cfg.get("seed", 42)),
    )
    sv = torch.tensor(ds.sensor_vals.transpose(1, 0, 2), dtype=torch.float32, device=device)
    sp = torch.tensor(ds.sensor_pos, dtype=torch.float32, device=device)
    st = torch.tensor(ds.sensor_time, dtype=torch.float32, device=device)
    with torch.no_grad():
        h_states, s_time = model.encode(sv, sp, ds.re_norm, st)
    return model, ds, (h_states, s_time, sp)


@torch.no_grad()
def predict(model, enc, coords_xy, t_phys, comp_idx, device, batch=8192):
    """在任意 (x,y,t) 點 query 模型，回 physical 預測值 [N]（denorm 為 identity）。"""
    h_states, s_time, sp = enc
    xy_t = torch.tensor(coords_xy, dtype=torch.float32, device=device)
    t_t = torch.tensor(t_phys, dtype=torch.float32, device=device)
    out = []
    for start in range(0, xy_t.shape[0], batch):
        end = min(start + batch, xy_t.shape[0])
        c_b = torch.full((end - start,), comp_idx, dtype=torch.long, device=device)
        o = model.query_decoder(xy_t[start:end], t_t[start:end], c_b, h_states, s_time, sp)
        out.append(o.squeeze(-1).cpu().numpy())
    return np.concatenate(out)


def residual_sigma(model, ds, enc, coords, t_phys, domain_length, device, batch=512):
    """What: 逐點 NS PDE residual 大小 √(mom_u²+mom_v²+cont²) [N]，physics-informed σ。

    Why: 重用訓練端 unsteady_ns_residuals（二階 autograd），denorm=identity（raw 即
         physical，與 evaluator 一致）。需 create_graph → 分 batch 控記憶體。
         conformal 只看 σ 的相對空間 pattern，故絕對量級不影響保證。
    """
    h_states, s_time, sp = enc
    snap = model.forcing.snapshot()
    kf, amp = float(snap["k_f"]), float(snap["A"])
    Lx, Ly = float(getattr(ds, "Lx", 1.0)), float(getattr(ds, "Ly", 1.0))
    uvp_fn = make_picon_model_fn_uvp(
        model, None, sp, ds.re_norm, s_time, device, h_states=h_states, s_time=s_time
    )
    out = np.empty(len(coords), dtype=np.float32)
    for start in range(0, len(coords), batch):
        end = min(start + batch, len(coords))
        xyt = torch.tensor(
            np.concatenate([coords[start:end], t_phys[start:end, None]], axis=1),
            dtype=torch.float32, device=device,
        ).requires_grad_(True)
        mu, mv, co = unsteady_ns_residuals(
            uvp_fn, xyt, re=ds.re_value, k_f=kf, A=amp,
            domain_length=domain_length, Lx=Lx, Ly=Ly,
        )
        mag = torch.sqrt(mu**2 + mv**2 + co**2 + 1e-12).squeeze(-1)
        out[start:end] = mag.detach().cpu().numpy()
    return out


# ──────────────────────────────────────────────────────────────────────────
# Sanity gate
# ──────────────────────────────────────────────────────────────────────────
def sanity_gate_ke(model, enc, dns, device, n_frames=10):
    """coarse-grid 估 KE(t) rel-err vs DNS（抓 checkpoint/sensor 配錯）。"""
    x, y = dns["x"].astype(np.float32), dns["y"].astype(np.float32)
    xg, yg = x[::2], y[::2]
    xx, yy = np.meshgrid(xg, yg, indexing="ij")
    flat = np.stack([xx.ravel(), yy.ravel()], axis=1)
    t_all = dns["time"]
    t_sel = np.linspace(0, len(t_all) - 1, n_frames).round().astype(int)
    ke_p, ke_t = [], []
    for ti in t_sel:
        tp = np.full(flat.shape[0], float(t_all[ti]), dtype=np.float32)
        u_p = predict(model, enc, flat, tp, 0, device)
        v_p = predict(model, enc, flat, tp, 1, device)
        ke_p.append(0.5 * float(np.mean(u_p**2 + v_p**2)))
        ke_t.append(0.5 * float(np.mean(dns["u"][ti] ** 2 + dns["v"][ti] ** 2)))
    return float(abs(np.mean(ke_p) - np.mean(ke_t)) / np.mean(ke_t))


# ──────────────────────────────────────────────────────────────────────────
# 取點 + σ
# ──────────────────────────────────────────────────────────────────────────
def sample_points(rng, dns, n, exclude_flat, t_idx_pool):
    """i.i.d. 抽 n 個 (grid location, time) pair，回 physical 座標 + DNS 真值。

    flat = x_idx*N + y_idx（row-major, row=x；對齊 CLAUDE.md sensor axis convention）。
    """
    x, y = dns["x"].astype(np.float32), dns["y"].astype(np.float32)
    N = len(x)
    allowed = (np.setdiff1d(np.arange(N * N), exclude_flat)
               if exclude_flat is not None else np.arange(N * N))
    flat = rng.choice(allowed, size=n, replace=True)
    xi, yj = flat // N, flat % N
    t_idx = rng.choice(t_idx_pool, size=n, replace=True)
    return {
        "coords": np.stack([x[xi], y[yj]], axis=1),
        "t_phys": dns["time"][t_idx].astype(np.float32),
        "u": dns["u"][t_idx, xi, yj].astype(np.float32),
        "v": dns["v"][t_idx, xi, yj].astype(np.float32),
    }


def periodic_nearest_sensor_dist(coords, sensor_pos, L):
    """每個點到最近訓練 sensor 的 periodic 歐式距離 [N]。"""
    d = np.abs(coords[:, None, :] - sensor_pos[None, :, :])
    d = np.minimum(d, L - d)
    return np.sqrt((d**2).sum(axis=-1)).min(axis=1)


# ──────────────────────────────────────────────────────────────────────────
# Split conformal（multi-draw, multi-σ）
# ──────────────────────────────────────────────────────────────────────────
def conformal_quantile(scores_cal, alpha):
    """split conformal 分位數，含有限樣本修正 ⌈(n+1)(1-α)⌉/n。"""
    n = len(scores_cal)
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")
    return float(np.quantile(scores_cal, k / n, method="higher"))


def run_conformal_component(pool, comp, alphas, n_cal, n_draws, rng):
    """multi-draw split conformal：fixed + 每個 σ 的 adaptive。

    conditional coverage 一律以「到 sensor 距離」(sigmas['dist']) 分 bin，使各方法可比。
    """
    e_all = np.abs(pool[f"{comp}_pred"] - pool[comp])
    sigmas = pool["sigmas"]
    dist_all = sigmas["dist"]
    n_total = len(e_all)
    methods = ["fixed"] + [f"adaptive_{s}" for s in SIGMA_NAMES]

    acc = {f"{a:.3f}": {m: {"cov": [], "hw": [], "q": [], "strat": []} for m in methods}
           for a in alphas}
    for _ in range(n_draws):
        perm = rng.permutation(n_total)
        ci, ti = perm[:n_cal], perm[n_cal:]
        e_test = e_all[ti]
        # 固定以距離四分位 bin（near→far），所有方法共用
        edges = np.quantile(dist_all[ti], [0, 0.25, 0.5, 0.75, 1.0])
        masks = [(dist_all[ti] >= lo) & (dist_all[ti] <= hi)
                 for lo, hi in zip(edges[:-1], edges[1:])]

        def record(key, m, cov_pointwise, hw_pointwise, q):
            acc[key][m]["cov"].append(float(np.mean(cov_pointwise)))
            acc[key][m]["hw"].append(float(np.mean(hw_pointwise)))
            acc[key][m]["q"].append(float(q))
            acc[key][m]["strat"].append(
                [float(np.mean(cov_pointwise[mask])) if mask.any() else float("nan")
                 for mask in masks]
            )

        for a in alphas:
            key = f"{a:.3f}"
            qf = conformal_quantile(e_all[ci], a)
            record(key, "fixed", e_test <= qf, np.full(len(ti), qf), qf)
            for s in SIGMA_NAMES:
                sig = sigmas[s]
                qa = conformal_quantile((e_all / sig)[ci], a)
                hw = qa * sig[ti]
                record(key, f"adaptive_{s}", e_test <= hw, hw, qa)

    res = {"n_cal": int(n_cal), "n_test": int(n_total - n_cal),
           "n_draws": int(n_draws), "alphas": {}}
    for a in alphas:
        key = f"{a:.3f}"
        res["alphas"][key] = {}
        for m in methods:
            d = acc[key][m]
            strat = np.nanmean(np.array(d["strat"]), axis=0)
            res["alphas"][key][m] = {
                "coverage_mean": float(np.mean(d["cov"])),
                "coverage_std": float(np.std(d["cov"])),
                "mean_halfwidth": float(np.mean(d["hw"])),
                "qhat_mean": float(np.mean(d["q"])),
                "stratified_coverage": strat.tolist(),
                "coverage_spread": float(np.nanmax(strat) - np.nanmin(strat)),
            }
    return res


# ──────────────────────────────────────────────────────────────────────────
# 繪圖
# ──────────────────────────────────────────────────────────────────────────
def plot_calibration_curve(summary, alphas, out):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    targets = [1 - a for a in alphas]
    for ax, comp in zip(axes, ["u", "v"]):
        ax.plot([0.7, 1.0], [0.7, 1.0], "k--", lw=0.8, alpha=0.6, label="ideal")
        for path, mk in [("A_transferable", "o"), ("B_oracle", "s")]:
            if path not in summary:
                continue
            cov = [summary[path][comp]["alphas"][f"{a:.3f}"]["fixed"]["coverage_mean"] for a in alphas]
            err = [summary[path][comp]["alphas"][f"{a:.3f}"]["fixed"]["coverage_std"] for a in alphas]
            ax.errorbar(targets, cov, yerr=err, marker=mk, capsize=2, label=path)
        ax.set_xlabel("Target coverage (1-alpha)")
        ax.set_ylabel("Empirical coverage")
        ax.set_title(f"Component {comp}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_stratified_coverage(summary, alpha, out):
    """Conditional coverage by distance-to-sensor quartile：fixed vs 各 adaptive σ。"""
    key = f"{alpha:.3f}"
    bins = ["Q1\n(near)", "Q2", "Q3", "Q4\n(far)"]
    labels = {"fixed": "fixed-width", "adaptive_dist": "adaptive: dist",
              "adaptive_tempered": "adaptive: sqrt(dist)", "adaptive_residual": "adaptive: PDE residual"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    xs = np.arange(len(bins))
    for ax, path in zip(axes, ["A_transferable", "B_oracle"]):
        if path not in summary:
            continue
        ax.axhline(1 - alpha, color="k", ls="--", lw=0.8, alpha=0.6, label=f"target {1-alpha:.2f}")
        for m, lab in labels.items():
            s = summary[path]["u"]["alphas"][key][m]["stratified_coverage"]
            ax.plot(xs, s, "o-", ms=4, label=lab)
        ax.set_xticks(xs)
        ax.set_xticklabels(bins)
        ax.set_ylabel("Conditional coverage (u)")
        ax.set_title(path)
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_residual_width_field(model, ds, enc, dns, device, qa, L, t_val, out):
    """Physics-informed (residual-σ) interval-width 場圖（單一時刻）。"""
    x, y = dns["x"].astype(np.float32), dns["y"].astype(np.float32)
    xg, yg = x[::4], y[::4]
    xx, yy = np.meshgrid(xg, yg, indexing="ij")
    flat = np.stack([xx.ravel(), yy.ravel()], axis=1)
    tp = np.full(flat.shape[0], float(t_val), dtype=np.float32)
    sig = residual_sigma(model, ds, enc, flat, tp, L, device)
    width = (2.0 * qa * sig).reshape(xx.shape)
    sensor_pos = ds.sensor_pos
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    pc = ax.pcolormesh(xx, yy, width, shading="auto", cmap="viridis")
    ax.scatter(sensor_pos[:, 0], sensor_pos[:, 1], s=6, c="red", marker="x",
               linewidths=0.6, label="sensors")
    fig.colorbar(pc, ax=ax, label="Interval width (u)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Physics-informed (residual) 90% width @ t={t_val:.2f}")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_time_extrapolation(pool, comp, alphas, t_splits, n_draws, rng):
    """What: temporal-split conformal — 早期時間校準、晚期時間測試（外推）。

    Why: 全部 marginal coverage 都假設 (loc,t) 對全窗 i.i.d.（交換性）。temporal split
         直接壓測「誤差統計是否時間平穩」；若外推 coverage ≈ i.i.d. → 窗內交換性近似成立。
         比 fixed vs √dist(tempered)。重用 pool（含 t_phys / preds / σ），不重算推理。
    """
    t = pool["t_phys"]
    e = np.abs(pool[f"{comp}_pred"] - pool[comp])
    sig = np.sqrt(pool["sigmas"]["dist"])  # tempered = √dist（前一輪贏家）
    out = {}
    for a in alphas:
        rows = []
        for ts in t_splits:
            cal_m, test_m = t <= ts, t > ts
            if cal_m.sum() < 50 or test_m.sum() < 50:
                continue
            # forward extrapolation（calibrate 早 t≤ts → test 晚 t>ts）
            qf = conformal_quantile(e[cal_m], a)
            qa = conformal_quantile((e / sig)[cal_m], a)
            cov_f = float(np.mean(e[test_m] <= qf))
            cov_a = float(np.mean(e[test_m] <= qa * sig[test_m]))
            # reverse extrapolation（calibrate 晚 t>ts → test 早 t≤ts；anti-conservative 風險方向）
            qf_r = conformal_quantile(e[test_m], a)
            cov_f_r = float(np.mean(e[cal_m] <= qf_r))
            # i.i.d. reference：同 n_cal 隨機切，multi-draw 平均
            n_cal = int(cal_m.sum())
            iid = []
            for _ in range(n_draws):
                perm = rng.permutation(len(e))
                qf2 = conformal_quantile(e[perm[:n_cal]], a)
                iid.append(float(np.mean(e[perm[n_cal:]] <= qf2)))
            rows.append({
                "t_split": float(ts), "n_cal": n_cal, "n_test": int(test_m.sum()),
                "fixed_fwd": cov_f, "tempered_fwd": cov_a, "fixed_reverse": cov_f_r,
                "iid_mean": float(np.mean(iid)), "iid_std": float(np.std(iid)),
            })
        out[f"{a:.3f}"] = rows
    return out


def error_distance_powerlaw(pool, comp):
    """What: fit |error| ~ C·dist^p（log-log 線性回歸），回 exponent p。

    Why: 檢驗 √dist 的 p≈0.5 假設是否成立 — 避免把 calibration 經驗律 overclaim 成
         diffusion 機制（Code-As-Hypothesis）。p≈0.5 → √dist 有 scaling 依據；否則僅 empirical。
    """
    e = np.abs(pool[f"{comp}_pred"] - pool[comp])
    d = pool["sigmas"]["dist"]
    mask = (e > 1e-8) & (d > 1e-8)
    lx, ly = np.log(d[mask]), np.log(e[mask])
    p, logC = np.polyfit(lx, ly, 1)
    # 以 dist 分 20 bin 看 median |error| 的 scaling（log-log 相關係數）
    bins = np.quantile(d, np.linspace(0, 1, 21))
    bc, be = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() > 5:
            bc.append(np.median(d[m]))
            be.append(np.median(e[m]))
    bc, be = np.array(bc), np.array(be)
    p_bin = float(np.polyfit(np.log(bc), np.log(be), 1)[0]) if len(bc) > 3 else float("nan")
    r = float(np.corrcoef(lx, ly)[0, 1])
    return {"exponent_pointwise": float(p), "exponent_binned_median": p_bin,
            "loglog_corr": r}


def plot_error_vs_time(model, enc, dns, device, out, n_pts=600, t_stride=4, seed=0):
    """What: 重建誤差隨時間演化（mean|error| 與相對誤差%）。

    Why: 佐證 temporal extrapolation 的非平穩根因 — 誤差在早期高、晚期低，而真實場
         RMS 全程穩定 → 隔離出「模型 CfC temporal context 累積」而非 flow transient。
    """
    x, y, t = dns["x"].astype(np.float32), dns["y"].astype(np.float32), dns["time"]
    N = len(x)
    rng = np.random.default_rng(seed)
    flat = rng.choice(N * N, size=n_pts, replace=False)
    xi, yj = flat // N, flat % N
    coords = np.stack([x[xi], y[yj]], axis=1)
    t_idx = np.arange(0, len(t), t_stride)
    ts, eu, ev, rel_u, rms = [], [], [], [], []
    for ti in t_idx:
        tp = np.full(n_pts, float(t[ti]), dtype=np.float32)
        up = predict(model, enc, coords, tp, 0, device)
        vp = predict(model, enc, coords, tp, 1, device)
        ut, vt = dns["u"][ti, xi, yj], dns["v"][ti, xi, yj]
        rms_u = float(np.sqrt(np.mean(ut**2)))
        ts.append(float(t[ti]))
        eu.append(float(np.mean(np.abs(up - ut))))
        ev.append(float(np.mean(np.abs(vp - vt))))
        rms.append(rms_u)
        rel_u.append(float(np.mean(np.abs(up - ut))) / rms_u * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.4))
    ax1.plot(ts, eu, "-", label="mean abs. error u")
    ax1.plot(ts, ev, "-", label="mean abs. error v")
    ax1.plot(ts, rms, "--", color="gray", alpha=0.7, label="true u RMS (stationary)")
    ax1.set_xlabel("Time t")
    ax1.set_ylabel("Magnitude")
    ax1.set_title("Reconstruction error vs time")
    ax1.legend(fontsize=7)
    ax2.plot(ts, rel_u, "-", color="C3")
    ax2.set_xlabel("Time t")
    ax2.set_ylabel("Relative error (u) [%]")
    ax2.set_title("Error decays as CfC context accumulates")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_time_extrapolation(te, alpha, out):
    """coverage vs t_split：temporal extrapolation（fixed/√dist）vs i.i.d. reference。"""
    rows = te[f"{alpha:.3f}"]
    ts = [r["t_split"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.axhline(1 - alpha, color="k", ls="--", lw=0.8, alpha=0.6, label=f"target {1-alpha:.2f}")
    ax.errorbar(ts, [r["iid_mean"] for r in rows], yerr=[r["iid_std"] for r in rows],
                marker="o", capsize=2, label="i.i.d. calib (reference)")
    ax.plot(ts, [r["fixed_fwd"] for r in rows], "s-", label="fwd extrap (early->late): fixed")
    ax.plot(ts, [r["tempered_fwd"] for r in rows], "^-", label="fwd extrap: sqrt(dist)")
    ax.plot(ts, [r["fixed_reverse"] for r in rows], "v-", label="reverse (late->early): fixed")
    ax.set_xlabel("Calibration cutoff time $t_{split}$ (test on $t>t_{split}$)")
    ax.set_ylabel("Test marginal coverage (u)")
    ax.set_title("Temporal extrapolation vs exchangeable calibration")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Split Conformal Prediction for DeepONet+CfC.")
    p.add_argument("--config", type=Path, default=Path("configs/exp_245_b3_les_T50.toml"))
    p.add_argument("--checkpoint", type=Path,
                   default=Path("artifacts/lab/multiseed/seeda/picon_kolmogorov_final.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/conformal_exp245"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    p.add_argument("--n-cal-a", type=int, default=200)
    p.add_argument("--n-test-a", type=int, default=4000)
    p.add_argument("--n-cal-b", type=int, default=5000)
    p.add_argument("--n-test-b", type=int, default=5000)
    p.add_argument("--n-draws", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-gate", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print(f"=== Conformal Prediction ===\ndevice={device}\nckpt={args.checkpoint}", flush=True)

    cfg = load_picon_config(args.config)
    L = float(cfg.get("domain_length", 1.0))
    model, ds, enc = load_model_and_encode(cfg, args.checkpoint, device)
    dns = np.load(cfg["dns_paths"][0], allow_pickle=True).item()
    sensor_pos = ds.sensor_pos.astype(np.float32)
    print(f"loaded: K={sensor_pos.shape[0]} sensors, DNS u{dns['u'].shape}", flush=True)

    ke_rel = None
    if not args.skip_gate:
        ke_rel = sanity_gate_ke(model, enc, dns, device)
        status = "PASS" if ke_rel <= GATE_KE_REL_ERR_MAX else "FAIL"
        print(f"[gate] KE rel-err = {ke_rel*100:.2f}% "
              f"(EXP-245 expect ~{EXP245_KE_REL_ERR*100:.1f}%) → {status}", flush=True)
        if status == "FAIL":
            raise SystemExit("[gate] KE rel-err 超出範圍 → (checkpoint, sensor) 可能配錯。abort。")

    # 訓練 sensor flat index（Path A held-out pool 排除之）
    N = len(dns["x"])
    dx, dy = float(dns["x"][1] - dns["x"][0]), float(dns["y"][1] - dns["y"][0])
    train_flat = ((np.round(sensor_pos[:, 0] / dx).astype(int) % N) * N
                  + (np.round(sensor_pos[:, 1] / dy).astype(int) % N))
    t_pool = np.arange(len(dns["time"]))
    rng = np.random.default_rng(args.seed)
    sigma_floor = L / N

    def build_set(n, exclude, label):
        print(f"  build {label}: {n} points (+ residual autograd)", flush=True)
        pts = sample_points(rng, dns, n, exclude, t_pool)
        pts["u_pred"] = predict(model, enc, pts["coords"], pts["t_phys"], 0, device)
        pts["v_pred"] = predict(model, enc, pts["coords"], pts["t_phys"], 1, device)
        dist = np.maximum(periodic_nearest_sensor_dist(pts["coords"], sensor_pos, L), sigma_floor)
        resid = np.maximum(
            residual_sigma(model, ds, enc, pts["coords"], pts["t_phys"], L, device), 1e-6
        )
        pts["sigmas"] = {"dist": dist, "tempered": np.sqrt(dist), "residual": resid}
        return pts

    summary: dict[str, Any] = {
        "_meta": {
            "checkpoint": str(args.checkpoint), "config": str(args.config),
            "ke_rel_err_gate": ke_rel, "alphas": args.alphas, "sigma_candidates": list(SIGMA_NAMES),
            "note_path_B": "Path B DNS 全場 calibration → 工程不可遷移，僅研究用 oracle 上限",
        }
    }

    print("[Path A] transferable: held-out random sensor locations", flush=True)
    pool_a = build_set(args.n_cal_a + args.n_test_a, train_flat, "Path A")
    summary["A_transferable"] = {
        c: run_conformal_component(pool_a, c, args.alphas, args.n_cal_a, args.n_draws, rng)
        for c in ("u", "v")
    }
    print("[Path B] oracle (research-only): DNS full-field calibration", flush=True)
    pool_b = build_set(args.n_cal_b + args.n_test_b, None, "Path B")
    summary["B_oracle"] = {
        c: run_conformal_component(pool_b, c, args.alphas, args.n_cal_b, args.n_draws, rng)
        for c in ("u", "v")
    }

    (out_dir / "conformal_summary.json").write_text(json.dumps(summary, indent=2))
    a_plot = min(args.alphas, key=lambda a: abs(a - 0.1))
    plot_calibration_curve(summary, args.alphas, out_dir / "calibration_curve.png")
    plot_stratified_coverage(summary, a_plot, out_dir / "stratified_coverage.png")
    qa_res = summary["A_transferable"]["u"]["alphas"][f"{a_plot:.3f}"]["adaptive_residual"]["qhat_mean"]
    plot_residual_width_field(model, ds, enc, dns, device, qa_res, L,
                              float(dns["time"][len(dns["time"]) // 2]),
                              out_dir / "residual_width_field.png")

    # ── Time extrapolation（壓測交換性）+ power-law fit（檢驗 √dist 機制）─────
    print("[time-extrap] temporal-split coverage vs i.i.d. reference", flush=True)
    te = run_time_extrapolation(pool_a, "u", args.alphas, [2.5, 3.0, 3.5, 4.0], args.n_draws, rng)
    pl_fit = {c: error_distance_powerlaw(pool_a, c) for c in ("u", "v")}
    summary["time_extrapolation"] = te
    summary["error_distance_powerlaw"] = pl_fit
    (out_dir / "conformal_summary.json").write_text(json.dumps(summary, indent=2))
    plot_time_extrapolation(te, a_plot, out_dir / "time_extrapolation.png")
    plot_error_vs_time(model, enc, dns, device, out_dir / "error_vs_time.png")

    # ── 摘要表（α 最接近 0.1）──────────────────────────────────────────────
    print(f"\n=== α={a_plot:.2f} (target {1-a_plot:.2f}) coverage / mean-halfwidth / spread ===",
          flush=True)
    key = f"{a_plot:.3f}"
    for path in ("A_transferable", "B_oracle"):
        print(f"--- {path} (n_cal={summary[path]['u']['n_cal']}) ---", flush=True)
        for m in ["fixed"] + [f"adaptive_{s}" for s in SIGMA_NAMES]:
            e = summary[path]["u"]["alphas"][key][m]
            print(f"  {m:18s}: cov={e['coverage_mean']:.3f}±{e['coverage_std']:.3f}  "
                  f"hw={e['mean_halfwidth']:.4f}  spread={e['coverage_spread']:.3f}", flush=True)
    print(f"\n=== Time extrapolation (α={a_plot:.2f}, u) — test coverage on t>t_split ===", flush=True)
    for r in te[f"{a_plot:.3f}"]:
        print(f"  t_split={r['t_split']:.1f} (n_test={r['n_test']:4d}): "
              f"iid={r['iid_mean']:.3f}±{r['iid_std']:.3f} | "
              f"fwd fixed={r['fixed_fwd']:.3f} √dist={r['tempered_fwd']:.3f} | "
              f"reverse fixed={r['fixed_reverse']:.3f}", flush=True)
    print("=== |error| ~ dist^p power-law (√dist 假設檢驗：p≈0.5?) ===", flush=True)
    for c in ("u", "v"):
        f = pl_fit[c]
        print(f"  {c}: p_pointwise={f['exponent_pointwise']:.3f}  "
              f"p_binned_median={f['exponent_binned_median']:.3f}  loglog_corr={f['loglog_corr']:.3f}",
              flush=True)
    print(f"\nartifacts → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
