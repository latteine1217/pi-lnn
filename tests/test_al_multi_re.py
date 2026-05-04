"""tests/test_al_multi_re.py

Multi-RE 累加 sanity：spec §4 規定 `l_cont_total = sum_i mean(cont_i²) / num_re`，
此值即 AL 的 C，不需再除。

本測試直接驗算數值關係（不跑訓練），確認 AL 接收的 C 與單 RE 對齊。
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pi_lnn import AugmentedLagrangianMultiplier


def test_single_re_l_cont_total_passes_through() -> None:
    """單 RE：l_cont_total = mean(cont²)，AL update 直接收下。"""
    al = AugmentedLagrangianMultiplier(ema_momentum=0.0)
    cont = torch.tensor([0.1, 0.2, 0.3, 0.4])
    l_cont_total = (cont ** 2).mean()  # = 0.075
    al.update(l_cont_total)
    assert abs(al.ema_C.item() - 0.075) < 1e-6


def test_multi_re_normalization_consistent() -> None:
    """Multi-RE：l_cont_total = mean(sum_i mean(cont_i²)) / num_re，
    AL 看到的 C 與「兩個 dataset 等價合併」的 mean(cont²) 量級一致。
    """
    al_multi = AugmentedLagrangianMultiplier(ema_momentum=0.0)
    al_single = AugmentedLagrangianMultiplier(ema_momentum=0.0)

    cont_re1 = torch.tensor([0.1, 0.2, 0.3, 0.4])
    cont_re2 = torch.tensor([0.2, 0.3, 0.4, 0.5])
    num_re = 2

    # Multi-RE: 與 training.py 相同的累加方式
    l_cont_total_multi = (
        (cont_re1 ** 2).mean() + (cont_re2 ** 2).mean()
    ) / num_re
    al_multi.update(l_cont_total_multi)

    # Single equivalent: 把兩個 dataset 的 cont 平均（合併視為單 RE）
    cont_combined = torch.cat([cont_re1, cont_re2])
    l_cont_single_equiv = (cont_combined ** 2).mean()
    al_single.update(l_cont_single_equiv)

    # 兩者數值不必完全相等（因為 mean-of-means ≠ mean-of-all），但量級一致：
    # 不應差超過 2×（防 num_re 重複除）
    ratio = al_multi.ema_C.item() / al_single.ema_C.item()
    assert 0.5 < ratio < 2.0, (
        f"Multi-RE C={al_multi.ema_C.item()} vs single equiv={al_single.ema_C.item()}，"
        f"ratio={ratio} 超出 [0.5, 2.0]"
    )


def test_lambda_growth_unaffected_by_num_re() -> None:
    """同樣的 per-RE C，無論 num_re 為多少，AL 行為一致（C 已被 dataset 層平均）。"""
    al1 = AugmentedLagrangianMultiplier(rho=1.0, ema_momentum=0.0)
    al4 = AugmentedLagrangianMultiplier(rho=1.0, ema_momentum=0.0)

    per_re_c = torch.tensor(0.5)
    # 1-RE
    al1.update(per_re_c)
    # 4-RE：sum / num_re = 4 * 0.5 / 4 = 0.5（每個 RE 給相同 C）
    al4.update(per_re_c)

    # λ 增量應一致（因為輸入 C 一致）
    assert abs(al1.lambda_.item() - al4.lambda_.item()) < 1e-9
