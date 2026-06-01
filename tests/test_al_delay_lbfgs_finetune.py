"""EXP-274 測試：AL delayed-start gate + Phase2 L-BFGS finetune。

驗證兩個新訓練策略：
- al_start_step: step < al_start_step 時 λ 凍結在 init 值（僅留 ρ 二次罰），>= 後才 dual update。
- lbfgs_finetune_steps: 主 phase 後同進程切 L-BFGS finetune，λ 凍結、無 GradNorm。

使用既有 re1000 smoke 資料（K=100, N=128），iterations 設極小以快速驗證控制流。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest

from pi_con.config import DEFAULT_PICON_ARGS
from pi_con.training import train_picon_kolmogorov


_SENSOR_JSON = ROOT / "data/kolmogorov_sensors/re1000/sensors_qrpivot_K100_N128_t0-5.json"
_SENSOR_NPZ = ROOT / "data/kolmogorov_sensors/re1000/sensors_qrpivot_K100_N128_t0-5_dns_values.npz"
_DNS = ROOT / "data/dns/kolmogorov_dns_fp64_etdrk4_Re1000_N128_T5_ds4.npy"

_HAVE_DATA = _SENSOR_JSON.exists() and _SENSOR_NPZ.exists() and _DNS.exists()
pytestmark = pytest.mark.skipif(not _HAVE_DATA, reason="re1000 smoke 資料缺失")


def _base_cfg(tmp_path):
    """最小可跑配置（沿用 test_smoke_lbfgs 的 tiny 架構，AL on）。"""
    return {
        **DEFAULT_PICON_ARGS,
        "sensor_jsons": [str(_SENSOR_JSON)],
        "sensor_npzs": [str(_SENSOR_NPZ)],
        "dns_paths": [str(_DNS)],
        "re_values": [1000.0],
        "observed_sensor_channels": ["u", "v"],
        "d_model": 32,
        "d_time": 8,
        "num_query_mlp_layers": 1,
        "query_mlp_hidden_dim": 32,
        "operator_rank": 32,
        "num_query_points": 32,
        "num_physics_points": 8,
        "checkpoint_period": 0,
        "lr_schedule": "none",
        "use_schedule_free": False,
        "use_gradnorm": False,
        "time_marching": False,
        "kolmogorov_k_f": 2.0,
        "device": "cpu",
        "artifacts_dir": str(tmp_path / "art"),
        # AL on（cont），continuity_weight 必須 0
        "use_augmented_lagrangian": True,
        "al_constraints": ["cont"],
        "al_init_lambda": 0.0,
        "al_rho": 1.0,
        "al_update_freq": 1,    # 每步嘗試 update（被 gate 擋下時才不動）
        "continuity_weight": 0.0,
    }


def test_al_start_step_freezes_lambda(tmp_path):
    """al_start_step 之前 λ 必須凍結在 init(0)，之後才隨 dual update 上升。"""
    cfg = _base_cfg(tmp_path)
    cfg.update({"iterations": 6, "al_start_step": 4})
    lam_by_step: dict[int, float] = {}

    def _log(step, metrics):
        if "al_lambda_cont" in metrics:
            lam_by_step[step] = metrics["al_lambda_cont"]

    train_picon_kolmogorov(cfg, log_fn=_log)

    # step 1~3：dual update 被 gate 擋下 → λ 仍為 init 0
    for s in (1, 2, 3):
        assert lam_by_step.get(s, 0.0) == pytest.approx(0.0), f"step {s} λ 應凍結在 0"
    # step >= al_start_step：cont 殘差 > 0 → λ 上升
    assert lam_by_step[6] > 0.0, "step >= al_start_step 後 λ 應隨 dual update 上升"


def test_al_start_step_zero_is_legacy(tmp_path):
    """al_start_step=0（預設）：λ 從 step 1 起就能 update（向後相容）。"""
    cfg = _base_cfg(tmp_path)
    cfg.update({"iterations": 3, "al_start_step": 0})
    lam_by_step: dict[int, float] = {}

    def _log(step, metrics):
        if "al_lambda_cont" in metrics:
            lam_by_step[step] = metrics["al_lambda_cont"]

    train_picon_kolmogorov(cfg, log_fn=_log)
    assert lam_by_step[3] > 0.0, "al_start_step=0 時 λ 應從頭累積"


def test_lbfgs_finetune_phase_runs_and_freezes_lambda(tmp_path):
    """Phase2 L-BFGS finetune 能跑完、final.pt 存在、step 接續、λ 凍結、loss 有限。"""
    cfg = _base_cfg(tmp_path)
    cfg.update({
        "iterations": 4,
        "al_start_step": 0,
        "lbfgs_finetune_steps": 3,
        "lbfgs_max_iter": 2,
    })
    records: list[tuple[int, str | None, float]] = []

    def _log(step, metrics):
        records.append((
            step,
            metrics.get("phase"),
            metrics.get("al_lambda_cont", float("nan")),
        ))

    train_picon_kolmogorov(cfg, log_fn=_log)

    # final.pt 存在
    assert (tmp_path / "art" / "picon_kolmogorov_final.pt").exists()

    # phase2 step 編號接續主 phase（iterations+1 ..），標記 lbfgs_finetune
    ft = [r for r in records if r[1] == "lbfgs_finetune"]
    assert [r[0] for r in ft] == [5, 6, 7], "phase2 step 編號應接續主 phase"

    # phase2 期間 λ 凍結 == 主 phase 結束值
    main_records = [r for r in records if r[1] != "lbfgs_finetune"]
    lam_end_phase1 = main_records[-1][2]
    for _, _, lam in ft:
        assert lam == pytest.approx(lam_end_phase1), "phase2 λ 必須凍結"
        assert np.isfinite(lam)


def test_lbfgs_finetune_disabled_is_legacy(tmp_path):
    """lbfgs_finetune_steps=0（預設）：不產生 phase2 紀錄（向後相容）。"""
    cfg = _base_cfg(tmp_path)
    cfg.update({"iterations": 3, "al_start_step": 0, "lbfgs_finetune_steps": 0})
    phases: list[str | None] = []

    def _log(step, metrics):
        phases.append(metrics.get("phase"))

    train_picon_kolmogorov(cfg, log_fn=_log)
    assert "lbfgs_finetune" not in phases, "未啟用時不應有 phase2"
    assert (tmp_path / "art" / "picon_kolmogorov_final.pt").exists()


def test_lbfgs_finetune_fixed_batch_runs(tmp_path):
    """EXP-275: lbfgs_finetune_fixed_batch=true 能跑完、phase2 接續、λ 凍結、loss 有限。

    驗證控制流；curvature-history 是否真正生效屬訓練動態，由 EXP-275 完整實驗判定。
    """
    cfg = _base_cfg(tmp_path)
    cfg.update({
        "iterations": 4,
        "al_start_step": 0,
        "lbfgs_finetune_steps": 3,
        "lbfgs_max_iter": 2,
        "lbfgs_finetune_fixed_batch": True,
    })
    records: list[tuple[int, str | None, float]] = []

    def _log(step, metrics):
        records.append((step, metrics.get("phase"), metrics.get("l_data", float("nan"))))

    train_picon_kolmogorov(cfg, log_fn=_log)

    assert (tmp_path / "art" / "picon_kolmogorov_final.pt").exists()
    ft = [r for r in records if r[1] == "lbfgs_finetune"]
    assert [r[0] for r in ft] == [5, 6, 7], "phase2 step 編號應接續主 phase"
    for _, _, ld in ft:
        assert np.isfinite(ld), "fixed_batch phase2 l_data 必須有限"


def _capture_phase2_lbfgs_lr(monkeypatch, cfg):
    """攔截 phase2 建立的 torch.optim.LBFGS，回傳其 lr。"""
    import torch

    captured: dict[str, float] = {}
    _orig = torch.optim.LBFGS.__init__

    def _spy(self, params, lr=1.0, *a, **k):
        captured["lr"] = float(lr)
        return _orig(self, params, lr, *a, **k)

    monkeypatch.setattr(torch.optim.LBFGS, "__init__", _spy)
    train_picon_kolmogorov(cfg, log_fn=None)
    return captured.get("lr")


def test_phase2_lbfgs_lr_decoupled_from_learning_rate(tmp_path, monkeypatch):
    """EXP-275 根因回歸：phase2 L-BFGS 的 lr 必須用 lbfgs_finetune_lr（預設 1.0），

    不可沿用一階 learning_rate（=1e-3）。L-BFGS+strong_wolfe 是 Newton step，lr=1.0；
    誤用 1e-3 在深收斂點會使 step 退化成零權重更新 → loss 凍結（job 3766 根因）。
    """
    cfg = _base_cfg(tmp_path)
    cfg.update({
        "iterations": 3,
        "al_start_step": 0,
        "lbfgs_finetune_steps": 1,
        "lbfgs_max_iter": 2,
        "learning_rate": 1e-3,   # 一階 LR；phase2 不該用這個
    })
    lr = _capture_phase2_lbfgs_lr(monkeypatch, cfg)
    assert lr == pytest.approx(1.0), (
        f"phase2 L-BFGS 預設 lr 應為 1.0（Newton step），實得 {lr}；"
        "誤用一階 learning_rate 會在深收斂點凍結"
    )


def test_phase2_lbfgs_lr_configurable(tmp_path, monkeypatch):
    """lbfgs_finetune_lr 可由 config 覆寫。"""
    cfg = _base_cfg(tmp_path)
    cfg.update({
        "iterations": 3,
        "al_start_step": 0,
        "lbfgs_finetune_steps": 1,
        "lbfgs_max_iter": 2,
        "lbfgs_finetune_lr": 0.5,
    })
    lr = _capture_phase2_lbfgs_lr(monkeypatch, cfg)
    assert lr == pytest.approx(0.5), f"lbfgs_finetune_lr 未生效，實得 {lr}"
