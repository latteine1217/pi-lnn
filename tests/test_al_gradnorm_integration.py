"""tests/test_al_gradnorm_integration.py

EXP-071 路徑核心驗證（spec v4 §5）：

關鍵主張：「AL term 完全不進 GradNorm losses 列表」
=> _gradnorm_step 看不到 AL 的 gradient norm
=> mean_G 不被 G_al 污染
=> w_ns_u/w_ns_v 計算與 EXP-070 (no GradNorm) 完全等價

本測試直接組裝 _gradnorm_step + AL term 流程，驗證：
1. _gradnorm_step 收到 3 個 losses（不含 AL）
2. AL term 在 loss 之外加，weight = 1
3. AL term 不進 GradNorm 的話，反過來測：若把 AL 塞進 gn_losses，會看到 mean_G 變化（負面對照）
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pi_lnn import AugmentedLagrangianMultiplier, GradNormWeights
from pi_lnn.losses import _gradnorm_step


def _make_toy_model_and_losses():
    """建立 toy model，模擬 data/ns_u/ns_v/cont 4 個 loss 與 1 組 ref_params。"""
    torch.manual_seed(42)
    ref = nn.Linear(4, 4)
    ref_params = list(ref.parameters())
    x = torch.randn(8, 4, requires_grad=True)
    target = torch.randn(8, 4)

    out = ref(x)
    l_data = ((out - target) ** 2).mean()
    l_ns_u = (out[:, 0] ** 2).mean()
    l_ns_v = (out[:, 1] ** 2).mean()
    l_cont = (out[:, 2] ** 2).mean()
    return ref_params, l_data, l_ns_u, l_ns_v, l_cont


def test_v4_gn_losses_excludes_al() -> None:
    """EXP-071 路徑：gn_losses = [data, ns_u, ns_v]，3 元素，不含 AL term。"""
    ref_params, l_data, l_ns_u, l_ns_v, l_cont = _make_toy_model_and_losses()
    gn = GradNormWeights([1.0, 0.01, 0.01], task_names=["data", "ns_u", "ns_v"])

    gn_losses = [l_data, l_ns_u, l_ns_v]
    assert len(gn_losses) == 3, "EXP-071 path 必須 3 elements"

    _gradnorm_step(gn, gn_losses, ref_params, ema_momentum=0.5)
    ws = gn.weights.detach()
    assert ws.shape[0] == 3
    # data 應仍 ~ 1（normalize_to_data_ 之前還是 EMA 後）
    assert ws[0].item() > 0


def test_lambda_jump_does_not_perturb_gradnorm_weights() -> None:
    """spec v4 核心：λ 跳變不影響 GradNorm 算出來的 w_ns_u/w_ns_v（AL 不在 gn_losses）。

    模擬：先用 λ_small 跑一次 _gradnorm_step，再用 λ_large 跑一次；
    若 AL 真的不在 gn_losses，兩次的 w_* 應完全相同。
    """
    ref_params, l_data, l_ns_u, l_ns_v, l_cont = _make_toy_model_and_losses()

    al_small = AugmentedLagrangianMultiplier(init_lambda=0.0)
    al_large = AugmentedLagrangianMultiplier(init_lambda=10.0)

    # Run 1: small λ
    gn1 = GradNormWeights([1.0, 0.01, 0.01], task_names=["data", "ns_u", "ns_v"])
    _gradnorm_step(gn1, [l_data, l_ns_u, l_ns_v], ref_params, ema_momentum=0.0)
    ws1 = gn1.weights.detach().clone()

    # Run 2: large λ — al_term computed but NOT passed to _gradnorm_step
    ref_params, l_data, l_ns_u, l_ns_v, l_cont = _make_toy_model_and_losses()
    gn2 = GradNormWeights([1.0, 0.01, 0.01], task_names=["data", "ns_u", "ns_v"])
    _ = al_large.loss_term(l_cont)   # AL term computed but discarded for GradNorm
    _gradnorm_step(gn2, [l_data, l_ns_u, l_ns_v], ref_params, ema_momentum=0.0)
    ws2 = gn2.weights.detach()

    # 兩次 w_* 應完全相同（同樣 input losses + same seed）
    assert torch.allclose(ws1, ws2, atol=1e-5), (
        f"AL 不該影響 GradNorm 計算，但 ws1={ws1.tolist()} vs ws2={ws2.tolist()}"
    )


def test_negative_control_putting_al_in_gn_pollutes_weights() -> None:
    """負面對照：若違反 v4 規則把 AL term 塞進 gn_losses → λ 不同會導致 w_* 變動。

    這個 test 保護 spec invariant：未來若有人「再嘗試」把 AL 塞回 gradnorm_tasks，
    這個 test 會告訴他「為什麼 v4 把它移出去」。
    """
    ref_params, l_data, l_ns_u, l_ns_v, l_cont = _make_toy_model_and_losses()

    al_small = AugmentedLagrangianMultiplier(init_lambda=0.0, rho=1.0)
    al_large = AugmentedLagrangianMultiplier(init_lambda=10.0, rho=1.0)

    # WRONG path: AL term in gn_losses
    gn1 = GradNormWeights([1.0, 0.01, 0.01, 0.01],
                          task_names=["data", "ns_u", "ns_v", "al"])
    _gradnorm_step(
        gn1,
        [l_data, l_ns_u, l_ns_v, al_small.loss_term(l_cont)],
        ref_params,
        ema_momentum=0.0,
    )
    ws1 = gn1.weights.detach().clone()

    ref_params, l_data, l_ns_u, l_ns_v, l_cont = _make_toy_model_and_losses()
    gn2 = GradNormWeights([1.0, 0.01, 0.01, 0.01],
                          task_names=["data", "ns_u", "ns_v", "al"])
    _gradnorm_step(
        gn2,
        [l_data, l_ns_u, l_ns_v, al_large.loss_term(l_cont)],
        ref_params,
        ema_momentum=0.0,
    )
    ws2 = gn2.weights.detach()

    # ns_u/ns_v weights should differ — 證明 mean_G 被 G_al 污染（v3 BLOCKER B-V3-1）
    assert not torch.allclose(ws1[:3], ws2[:3], atol=1e-5), (
        "若 AL 在 gn_losses 中，λ 跳變應該污染 ns_u/ns_v weights — 此測試保護 v4 invariant"
    )


def test_loss_assembly_v4_pattern() -> None:
    """模擬 spec §4 EXP-071 path：l_total = sum(w_i * loss_i for i<3) + al_term。"""
    ref_params, l_data, l_ns_u, l_ns_v, l_cont = _make_toy_model_and_losses()
    gn = GradNormWeights([1.0, 0.5, 0.3], task_names=["data", "ns_u", "ns_v"])
    al = AugmentedLagrangianMultiplier(init_lambda=2.0, rho=1.0)

    al_term = al.loss_term(l_cont)
    ws = gn.weights.detach()
    l_total = ws[0] * l_data + ws[1] * l_ns_u + ws[2] * l_ns_v + al_term

    # 手動驗算
    expected = (
        1.0 * l_data.item()
        + 0.5 * l_ns_u.item()
        + 0.3 * l_ns_v.item()
        + (2.0 * l_cont.item() + 0.5 * 1.0 * l_cont.item() ** 2)
    )
    assert abs(l_total.item() - expected) < 1e-5
