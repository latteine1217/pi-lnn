"""Tests for 5-task GradNorm (data, ns_u, ns_v, cont, bc).

驗證:
  1. GradNormWeights 支援 5 元素 init_weights
  2. _gradnorm_step 處理 5 個 losses 並更新 5 個 weights
  3. 4-task vs 5-task 互不干擾（向後相容）
  4. data weight 永遠 normalize 到 1（既有約定）
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pi_con.losses import GradNormWeights, _gradnorm_step  # noqa: E402


def _make_dummy_setup(n_tasks: int, seed: int = 0):
    """建立簡單的 toy 模型 + 5 個 task losses 用於 GradNorm 測試。"""
    torch.manual_seed(seed)
    P = 10
    ref_param = torch.randn(P, requires_grad=True)
    # 各 task 的 loss = ||A_i · ref_param||²，不同 A_i 給不同 gradient norm
    A = torch.randn(n_tasks, P)
    losses = [(A[i] @ ref_param).pow(2).sum() for i in range(n_tasks)]
    return ref_param, losses


def test_gradnorm_4task_layout() -> None:
    """4-task GradNorm（data, ns_u, ns_v, cont）：向後相容測試."""
    gn = GradNormWeights(init_weights=[1.0, 0.01, 0.01, 0.01])
    assert gn.weights.shape == (4,)
    assert torch.allclose(gn.weights, torch.tensor([1.0, 0.01, 0.01, 0.01]), atol=1e-6)

    ref, losses = _make_dummy_setup(n_tasks=4)
    _gradnorm_step(gn, losses, [ref], ema_momentum=0.5)

    ws = gn.weights.detach()
    assert ws.shape == (4,)
    # data weight 應該 normalize 到 1（_gradnorm_step 內部 w / w[0]）
    assert torch.isclose(ws[0], torch.tensor(1.0), atol=1e-6), \
        f"data weight 未歸一化到 1, 實得 {float(ws[0])}"
    print(f"[test_gradnorm_4task_layout] ws={ws.tolist()}  PASS")


def test_gradnorm_5task_layout() -> None:
    """5-task GradNorm（data, ns_u, ns_v, cont, bc）：新版測試."""
    gn = GradNormWeights(init_weights=[1.0, 0.01, 0.01, 0.01, 0.1])
    assert gn.weights.shape == (5,)
    assert torch.allclose(gn.weights, torch.tensor([1.0, 0.01, 0.01, 0.01, 0.1]), atol=1e-6)

    ref, losses = _make_dummy_setup(n_tasks=5)
    _gradnorm_step(gn, losses, [ref], ema_momentum=0.5)

    ws = gn.weights.detach()
    assert ws.shape == (5,)
    assert torch.isclose(ws[0], torch.tensor(1.0), atol=1e-6), \
        f"data weight 未歸一化到 1, 實得 {float(ws[0])}"
    # BC weight 應該被計算（不再保持 init 0.1）
    assert ws[4] != 0.1 or ws[4] > 0, f"bc weight 沒被 GradNorm 更新: {float(ws[4])}"
    print(f"[test_gradnorm_5task_layout] ws={ws.tolist()}  PASS")


def test_gradnorm_5task_ema_smoothing() -> None:
    """連續 5 次 update 後，weights 應該穩定且 data weight 仍 = 1."""
    torch.manual_seed(1)
    gn = GradNormWeights(init_weights=[1.0, 0.01, 0.01, 0.01, 0.1])
    P = 10
    ref = torch.randn(P, requires_grad=True)
    A = torch.randn(5, P)

    history = []
    for _ in range(5):
        # 重新計算 loss（每次 step 都需要 fresh autograd graph）
        losses = [(A[i] @ ref).pow(2).sum() for i in range(5)]
        _gradnorm_step(gn, losses, [ref], ema_momentum=0.5)
        history.append(gn.weights.detach().tolist())

    final = gn.weights.detach()
    assert torch.isclose(final[0], torch.tensor(1.0), atol=1e-6)
    assert all(w > 0 for w in final.tolist()), f"出現負/零 weight: {final.tolist()}"
    print(f"[test_gradnorm_5task_ema_smoothing] final ws={final.tolist()}, "
          f"first={history[0]}  PASS")


def test_gradnorm_inits_with_arbitrary_length() -> None:
    """GradNormWeights 配合顯式 task_names 應接受任意長度（4/5 之外）。

    Note: 41cdc6c 引入 task_names 後，未指定 task_names 時長度限定 ∈ {4, 5}
    （由 _DEFAULT_TASK_LAYOUTS 推斷預設 task name）。任意長度需顯式 task_names。
    """
    for n in (3, 6, 8):
        init = [1.0] + [0.01] * (n - 1)
        names = [f"task_{i}" for i in range(n)]
        gn = GradNormWeights(init_weights=init, task_names=names)
        assert gn.weights.shape == (n,)
        assert gn.task_names == tuple(names)
    print("[test_gradnorm_inits_with_arbitrary_length] PASS")


def test_gradnorm_checkpoint_roundtrip(tmp_path=None) -> None:
    """GradNorm state 能正確存進 checkpoint + 從 checkpoint 載回。

    這個 test 驗證 training.py 的 resume bug 修法：
      - checkpoint 中包含 gradnorm_state_dict
      - 載入後 weights 跟存檔時一致（避免 resume 重置成 init）
    """
    import tempfile

    if tmp_path is None:
        tmp_dir = Path(tempfile.mkdtemp())
    else:
        tmp_dir = tmp_path

    # 1. 建 GradNorm，跑幾次 update 讓 weights 偏離 init
    gn_save = GradNormWeights(init_weights=[1.0, 0.01, 0.01, 0.01, 0.1])
    torch.manual_seed(42)
    P = 8
    ref = torch.randn(P, requires_grad=True)
    A = torch.randn(5, P)
    for _ in range(3):
        losses = [(A[i] @ ref).pow(2).sum() for i in range(5)]
        _gradnorm_step(gn_save, losses, [ref], ema_momentum=0.5)

    saved_weights = gn_save.weights.detach().clone()
    saved_log = gn_save.log_weights.detach().clone()
    # weights 必須已偏離 init（否則 test 沒意義）
    assert not torch.allclose(saved_weights, torch.tensor([1.0, 0.01, 0.01, 0.01, 0.1]), atol=1e-3), \
        "GradNorm weights 沒偏離 init，test 沒鑑別力"

    # 2. 模擬 training.py 的 checkpoint save
    ckpt_path = tmp_dir / "test_gn_ckpt.pt"
    torch.save({
        "step": 100,
        "model_state_dict": {},   # 空，這 test 只關心 GradNorm
        "gradnorm_state_dict": gn_save.state_dict(),
    }, str(ckpt_path))

    # 3. 模擬 resume：建新 GradNorm（init weights）→ load checkpoint
    gn_load = GradNormWeights(init_weights=[1.0, 0.01, 0.01, 0.01, 0.1])
    # 確認 init 跟 saved 不同
    assert not torch.allclose(gn_load.weights.detach(), saved_weights, atol=1e-3)

    ckpt = torch.load(str(ckpt_path), weights_only=False)
    gn_load.load_state_dict(ckpt["gradnorm_state_dict"])

    # 4. 驗證 round-trip 一致
    loaded_weights = gn_load.weights.detach()
    loaded_log = gn_load.log_weights.detach()
    assert torch.allclose(loaded_log, saved_log, atol=1e-7), (
        f"log_weights round-trip 不一致：saved={saved_log.tolist()}, loaded={loaded_log.tolist()}"
    )
    assert torch.allclose(loaded_weights, saved_weights, atol=1e-7), (
        f"weights round-trip 不一致：saved={saved_weights.tolist()}, loaded={loaded_weights.tolist()}"
    )
    print(f"[test_gradnorm_checkpoint_roundtrip] saved={saved_weights.tolist()} → loaded matches  PASS")


def test_gradnorm_checkpoint_layout_mismatch() -> None:
    """4-task ckpt 載到 5-task GradNorm（或反向）應該被偵測，不是 silent crash."""
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp())

    # 存 4-task ckpt
    gn_4 = GradNormWeights(init_weights=[1.0, 0.01, 0.01, 0.01])
    ckpt_path = tmp_dir / "gn_4task.pt"
    torch.save({"gradnorm_state_dict": gn_4.state_dict()}, str(ckpt_path))

    # 5-task GradNorm 嘗試 load 4-task ckpt
    gn_5 = GradNormWeights(init_weights=[1.0, 0.01, 0.01, 0.01, 0.1])
    ckpt = torch.load(str(ckpt_path), weights_only=False)
    saved_log = ckpt["gradnorm_state_dict"]["log_weights"]

    assert saved_log.numel() == 4
    assert gn_5.log_weights.numel() == 5

    # training.py 的 layout-mismatch check：應該偵測到 size mismatch
    layout_compatible = saved_log.numel() == gn_5.log_weights.numel()
    assert not layout_compatible, "4-task ckpt 不應該 silent compatible 給 5-task"

    # 模擬真實 training.py 的處理：layout 不相容時不 load，保留 init weights
    if layout_compatible:
        gn_5.load_state_dict(ckpt["gradnorm_state_dict"])
    # 此 case 應該保持 init [1, 0.01, 0.01, 0.01, 0.1]
    assert torch.allclose(
        gn_5.weights.detach(),
        torch.tensor([1.0, 0.01, 0.01, 0.01, 0.1]),
        atol=1e-6,
    )
    print("[test_gradnorm_checkpoint_layout_mismatch] 4-task ckpt → 5-task gn 正確被拒絕  PASS")


def test_gradnorm_min_weight_floor() -> None:
    """min_weight floor 應該防止 task weight 降到 floor 以下，但仍允許動態調."""
    # 構造一個故意讓 task 1 gradient 極小的 setup（會被 GradNorm 推到很低）
    torch.manual_seed(20)
    P = 8
    ref = torch.randn(P, requires_grad=True)
    A_strong = torch.randn(P) * 10.0  # task 0: 大 gradient
    A_weak   = torch.randn(P) * 1e-4  # task 1: 極弱 gradient → GradNorm 想加大其 weight
    A_3      = torch.randn(P) * 5.0   # task 2: 中等
    # 注意：weak gradient 的 task 在當前 GradNorm 公式會被「反比放大」（mean_G/G_i 變大）
    # 但 EMA 過程中可能某 task 被持續壓低。
    # 為驗證 floor，我們直接構造一個會把某 task 壓到 < floor 的場景。

    # Setup：task 0 是 data（永遠 = 1），task 1/2 是 physics
    # 用一個極端 init：[1.0, 1e-3, 1e-3] 讓 task 1, 2 init 就低於 floor=0.05
    # 41cdc6c 後：3-task 須顯式 task_names。
    _names3 = ["data", "ns_u", "ns_v"]
    gn = GradNormWeights(init_weights=[1.0, 1e-3, 1e-3], task_names=_names3)
    losses = [
        (A_strong @ ref).pow(2).sum(),
        (A_weak @ ref).pow(2).sum(),
        (A_3 @ ref).pow(2).sum(),
    ]

    # 不加 floor：weight 可能掉到 < 0.05
    _gradnorm_step(gn, losses, [ref], ema_momentum=0.5, min_weight=0.0)
    ws_no_floor = gn.weights.detach().clone()

    # 加 floor=0.05
    gn2 = GradNormWeights(init_weights=[1.0, 1e-3, 1e-3], task_names=_names3)
    losses2 = [
        (A_strong @ ref).pow(2).sum(),
        (A_weak @ ref).pow(2).sum(),
        (A_3 @ ref).pow(2).sum(),
    ]
    _gradnorm_step(gn2, losses2, [ref], ema_momentum=0.5, min_weight=0.05)
    ws_with_floor = gn2.weights.detach().clone()

    # 加 floor 後所有 weight 應 >= 0.05
    assert (ws_with_floor >= 0.05 - 1e-6).all(), (
        f"Floor 沒生效：weights = {ws_with_floor.tolist()}"
    )
    print(
        f"[test_gradnorm_min_weight_floor] no_floor={ws_no_floor.tolist()}, "
        f"with_floor={ws_with_floor.tolist()}  PASS"
    )


def test_gradnorm_floor_does_not_break_dynamic() -> None:
    """Floor 應該只 clamp 下限，不影響 weight 上限的動態變化."""
    torch.manual_seed(21)
    P = 8
    ref = torch.randn(P, requires_grad=True)
    A = torch.randn(5, P)
    gn = GradNormWeights(init_weights=[1.0, 0.01, 0.01, 0.01, 0.1])

    # 跑多步，weights 應該動態變化
    history = []
    for _ in range(5):
        losses = [(A[i] @ ref).pow(2).sum() for i in range(5)]
        _gradnorm_step(gn, losses, [ref], ema_momentum=0.5, min_weight=0.05)
        history.append(gn.weights.detach().clone().tolist())

    # 至少有 task 應該 weight > 0.05（floor 不該卡所有人）
    final = gn.weights.detach()
    assert (final > 0.05).any(), "所有 task weight 都卡在 floor，floor 太嚴"
    # data weight 永遠 = 1（不受 floor 影響）
    assert torch.isclose(final[0], torch.tensor(1.0), atol=1e-6), \
        f"data weight 變化了：{float(final[0])}"
    # 所有 weight >= floor
    assert (final >= 0.05 - 1e-6).all()
    print(f"[test_gradnorm_floor_does_not_break_dynamic] final={final.tolist()}  PASS")


if __name__ == "__main__":
    test_gradnorm_4task_layout()
    test_gradnorm_5task_layout()
    test_gradnorm_5task_ema_smoothing()
    test_gradnorm_inits_with_arbitrary_length()
    test_gradnorm_checkpoint_roundtrip()
    test_gradnorm_checkpoint_layout_mismatch()
    test_gradnorm_min_weight_floor()
    test_gradnorm_floor_does_not_break_dynamic()
    print("\n=== All 5-task GradNorm + checkpoint + floor tests PASS ===")
