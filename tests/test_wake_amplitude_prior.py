"""WakeAmplitudePrior 回歸測試（CEXP-046）。

鎖定不變量：
1. 單側性：low-energy 場 → cap_loss ≈ 0（不獎勵低能量，無 trivial-collapse 誘因）。
2. over-energy 場 → cap_loss > 0 且梯度可回傳到 params。
3. envelope 從「物理單位」算（反正規化 sensor_vals），與 uvp_fn 物理輸出可比。
4. 取樣點落在 [0,1] domain。
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pi_con.losses import WakeAmplitudePrior  # noqa: E402


def _mk_prior() -> WakeAmplitudePrior:
    torch.manual_seed(0)
    K, T = 100, 200
    sv = torch.randn(K, T, 2)            # normalized sensor vals [K,T,C]
    pos = torch.rand(K, 2)
    return WakeAmplitudePrior(
        sv, pos, u_idx=0, v_idx=1,
        u_mean=0.33, u_std=0.15, v_mean=0.0, v_std=0.12,
        percentile=0.95, gamma=1.5, radius_scale=2.0, sigma_scale=1.0,
        device=torch.device("cpu"),
    )


def test_envelope_finite_physical_scale():
    wp = _mk_prior()
    assert wp.E.shape == (100,)
    assert torch.isfinite(wp.E).all()
    # u_mean=0.33 → 物理能量 ~ O(0.1)；若誤用 normalized（mean=0）會偏小
    assert wp.E.min() > 0.0
    assert wp.sigma > 0.0 and wp.radius > wp.sigma


def test_sample_points_in_domain():
    wp = _mk_prior()
    rng = np.random.default_rng(0)
    xyt = wp.sample_points(rng, 256, 0.0, 20.0, torch.device("cpu"))
    assert xyt.shape == (256, 3)
    assert (xyt[:, :2] >= 0).all() and (xyt[:, :2] <= 1).all()
    assert (xyt[:, 2] >= 0).all() and (xyt[:, 2] <= 20.0).all()


def test_one_sided_low_energy_not_penalized():
    """單側上界：low-energy 場 cap_loss ≈ 0（核心防 trivial-collapse 性質）。"""
    wp = _mk_prior()
    rng = np.random.default_rng(0)
    xyt = wp.sample_points(rng, 256, 0.0, 20.0, torch.device("cpu"))

    def uvp_low(x):
        return torch.zeros(x.shape[0], 3)  # e_θ = 0

    loss = wp.cap_loss(uvp_low, xyt)
    assert loss.item() == 0.0, f"low-energy 場不應被懲罰，得 {loss.item()}"


def test_over_energy_penalized_with_gradient():
    """over-energy 場 cap_loss > 0 且梯度可回傳。"""
    wp = _mk_prior()
    rng = np.random.default_rng(0)
    xyt = wp.sample_points(rng, 256, 0.0, 20.0, torch.device("cpu"))

    lin = torch.nn.Linear(3, 3)

    def uvp_over(x):
        return lin(x) + torch.tensor([3.0, 0.0, 0.0])  # u 過高 → e_θ 爆

    loss = wp.cap_loss(uvp_over, xyt)
    assert loss.item() > 0.0
    loss.backward()
    g = lin.weight.grad
    assert g is not None and torch.isfinite(g).all()
    assert g.norm() > 0.0


def test_envelope_uses_physical_not_normalized():
    """envelope 必須反正規化：u_mean 改變應改變 envelope（證明用物理單位）。"""
    torch.manual_seed(0)
    sv = torch.randn(50, 100, 2)
    pos = torch.rand(50, 2)
    kw = dict(sensor_pos=pos, u_idx=0, v_idx=1, v_mean=0.0, v_std=0.1,
              percentile=0.95, gamma=1.5, device=torch.device("cpu"))
    e_lo = WakeAmplitudePrior(sv, u_mean=0.0, u_std=0.1, **kw).E
    e_hi = WakeAmplitudePrior(sv, u_mean=2.0, u_std=0.1, **kw).E
    # u_mean 從 0 → 2.0 應顯著抬高能量包絡
    assert (e_hi.mean() > e_lo.mean()), "envelope 未反映 u_mean 變化 → 可能誤用 normalized"
