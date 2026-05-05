"""Test ScheduleFree checkpoint save/resume mode consistency.

驗證:
  1. mid-step ckpt 存 train mode (param.data = x_t)
  2. 含 schedulefree_mode metadata
  3. resume 後 wrapper state 與 param.data 一致（不會踩 mode mismatch）
  4. 模擬 cylinder_007b resume bug 場景：load 後 step 再 step，loss 應穩定
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _make_dummy() -> tuple[nn.Module, "schedulefree.AdamWScheduleFree"]:
    """建一個極小 net + ScheduleFree optimizer 用於 test."""
    import schedulefree
    torch.manual_seed(42)
    net = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 1))
    opt = schedulefree.AdamWScheduleFree(net.parameters(), lr=1e-3, warmup_steps=0)
    return net, opt


def _train_steps(net: nn.Module, opt, n_steps: int) -> float:
    """跑 n 步 fake training，回傳 final loss."""
    opt.train()
    net.train()
    torch.manual_seed(0)
    x = torch.randn(32, 8)
    y = torch.randn(32, 1)
    for _ in range(n_steps):
        opt.zero_grad()
        loss = ((net(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    return float(loss.item())


def test_schedulefree_param_change_between_train_eval() -> None:
    """確認 ScheduleFree.train() vs .eval() 真的改 param.data（非 no-op）."""
    try:
        import schedulefree
    except ImportError:
        print("[test_schedulefree_param_change] SKIP (schedulefree not installed)")
        return

    net, opt = _make_dummy()
    _train_steps(net, opt, 50)   # 累積一些 z buffer

    p_train = list(net.parameters())[0].data.clone()  # x_t
    opt.eval()
    p_eval = list(net.parameters())[0].data.clone()   # y_t
    opt.train()

    diff = float((p_train - p_eval).abs().max().item())
    assert diff > 1e-6, (
        f"train/eval mode 切換沒改 param.data（diff={diff}）。"
        " ScheduleFree 設計上應該 swap weights，不該是 no-op。"
    )
    print(f"[test_schedulefree_param_change] |x_t − y_t|_∞ = {diff:.4e}  PASS")


def test_resume_train_mode_preserves_loss_dynamics() -> None:
    """模擬 cylinder_007b 場景：mid-step save train mode → resume → continue training。
    Loss 應該平滑（沒爆炸）。"""
    try:
        import schedulefree
    except ImportError:
        print("[test_resume_train_mode] SKIP")
        return

    tmp_dir = Path(tempfile.mkdtemp())
    ckpt_path = tmp_dir / "test_train_mode.pt"

    # ── Phase A: 跑 100 步，然後 save train mode ckpt ─────────────────
    net_a, opt_a = _make_dummy()
    _train_steps(net_a, opt_a, 100)
    loss_a_before_save = _train_steps(net_a, opt_a, 1)  # 跑完第 101 步

    # 模擬「修法後」save：不切 eval mode，直接 save train state
    torch.save({
        "step": 101,
        "model_state_dict": net_a.state_dict(),       # = x_t
        "optimizer_state_dict": opt_a.state_dict(),
        "schedulefree_mode": "train",
    }, str(ckpt_path))

    # 繼續 phase A 跑 50 步作為 reference
    loss_a_continued = _train_steps(net_a, opt_a, 50)

    # ── Phase B: 從 ckpt resume，跑同樣的 50 步 ────────────────────────
    net_b, opt_b = _make_dummy()  # fresh init（同 seed）
    ckpt = torch.load(str(ckpt_path), weights_only=False)
    net_b.load_state_dict(ckpt["model_state_dict"])
    opt_b.load_state_dict(ckpt["optimizer_state_dict"])
    # opt_b 預設 train mode；ckpt model_state 是 x_t（train mode 的 weights）→ 一致

    # 跑 50 步
    loss_b = _train_steps(net_b, opt_b, 50)

    # ── 驗證 ─────────────────────────────────────────────────────────
    # 修法成功 → resume 後繼續 train，loss 不該爆炸
    print(
        f"[test_resume_train_mode] phase A continued loss={loss_a_continued:.4e}, "
        f"phase B resumed loss={loss_b:.4e}"
    )
    # Loss 不能爆超過 100×（爆炸場景如 cylinder_007b 看到的 loss × 1000+）
    assert loss_b < loss_a_continued * 100, (
        f"Resume 後 loss 爆炸：phase A={loss_a_continued:.3e}, phase B={loss_b:.3e}"
        f"（修法可能未生效）"
    )
    print("[test_resume_train_mode] PASS（loss 沒爆，修法有效）")


def test_resume_eval_mode_ckpt_warns_or_fails() -> None:
    """模擬「修法前」場景：save 在 eval mode → resume 應該 unstable.
    這 test 確認 bug 確實存在（控制組）。"""
    try:
        import schedulefree
    except ImportError:
        print("[test_resume_eval_mode_unstable] SKIP")
        return

    tmp_dir = Path(tempfile.mkdtemp())
    ckpt_path = tmp_dir / "test_eval_mode.pt"

    # Phase A: save 在 eval mode（模擬修法前 buggy 行為）
    net_a, opt_a = _make_dummy()
    _train_steps(net_a, opt_a, 100)
    opt_a.eval()    # ← buggy: 切 eval mode 後 save
    torch.save({
        "step": 100,
        "model_state_dict": net_a.state_dict(),       # = y_t (averaged)
        "optimizer_state_dict": opt_a.state_dict(),
        "schedulefree_mode": "eval",   # 標記是 buggy 格式
    }, str(ckpt_path))
    opt_a.train()
    loss_a_continued = _train_steps(net_a, opt_a, 50)

    # Phase B: resume 從 eval-mode ckpt
    net_b, opt_b = _make_dummy()
    ckpt = torch.load(str(ckpt_path), weights_only=False)
    net_b.load_state_dict(ckpt["model_state_dict"])  # param.data = y_t
    opt_b.load_state_dict(ckpt["optimizer_state_dict"])
    # opt_b 預設 train mode，但 param.data = y_t → mismatch!
    loss_b = _train_steps(net_b, opt_b, 50)

    # 此 test 期望 phase B 表現比 phase A 顯著差（因為 mode mismatch）
    # 我們不 strict assert（toy net 不一定爆炸到 1000×），但記錄差距
    ratio = loss_b / max(loss_a_continued, 1e-12)
    print(
        f"[test_resume_eval_mode_unstable] eval-mode resume: "
        f"phase A continued={loss_a_continued:.4e}, phase B={loss_b:.4e}, ratio={ratio:.2f}"
    )
    # 此 test 是 documentation：證明 eval mode resume 不可靠


def test_wrapper_resume_train_mode_flag_fix() -> None:
    """專測 ScheduleFreeWrapper（非 AdamWScheduleFree）的 resume bug fix。

    Why this test exists:
      cylinder_010 用 SOAP + ScheduleFreeWrapper。Wrapper.state_dict() 不含
      `train_mode` flag → load_state_dict 後 wrapper.train_mode=False（預設）。
      但 ckpt param.data = x_t（train mode）→ 第一次 .train() 觸發 swap → 爆炸。

    Fix（training.py:386-388）：load 後立刻 `optimizer.train_mode = True`。

    這 test 用最簡 SGD+Wrapper 重現修法行為，不依賴 SOAP（避免 fp64/MPS 雜訊）。
    """
    try:
        import schedulefree
    except ImportError:
        print("[test_wrapper_resume_train_mode] SKIP (schedulefree not installed)")
        return

    tmp_dir = Path(tempfile.mkdtemp())
    ckpt_path = tmp_dir / "test_wrapper.pt"

    def _make_wrapper_pair() -> tuple[nn.Module, "schedulefree.ScheduleFreeWrapper"]:
        torch.manual_seed(42)
        net = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 1))
        base = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.0)
        opt = schedulefree.ScheduleFreeWrapper(base, momentum=0.9, weight_decay_at_y=0.0)
        return net, opt

    # === Phase A：跑 100 步，save train-mode ckpt ===
    net_a, opt_a = _make_wrapper_pair()
    opt_a.train()
    assert opt_a.train_mode is True
    torch.manual_seed(0)
    x = torch.randn(32, 8)
    y = torch.randn(32, 1)
    for _ in range(100):
        opt_a.zero_grad()
        ((net_a(x) - y) ** 2).mean().backward()
        opt_a.step()

    # 此時 param.data = x_t（沒切 eval）
    sd_save = opt_a.state_dict()
    assert "train_mode" not in sd_save, (
        "若 ScheduleFreeWrapper 改成 state_dict 也存 train_mode，"
        "本 fix 變成 dead code（仍正確，但可移除）。"
    )
    torch.save({
        "model_state_dict": net_a.state_dict(),
        "optimizer_state_dict": sd_save,
    }, str(ckpt_path))

    # 繼續跑 50 步當 reference
    for _ in range(50):
        opt_a.zero_grad()
        ((net_a(x) - y) ** 2).mean().backward()
        opt_a.step()
    loss_a = ((net_a(x) - y) ** 2).mean().item()

    # === Phase B（buggy）：load 後不修，第一個 .train() 觸發 swap ===
    net_b, opt_b = _make_wrapper_pair()
    ckpt = torch.load(str(ckpt_path), weights_only=False)
    net_b.load_state_dict(ckpt["model_state_dict"])
    opt_b.load_state_dict(ckpt["optimizer_state_dict"])
    assert opt_b.train_mode is False, "預期 wrapper 載入後 train_mode=False（bug 來源）"
    opt_b.train()  # ← 觸發 eval→train swap，對已是 x_t 的 param 再 swap
    for _ in range(50):
        opt_b.zero_grad()
        ((net_b(x) - y) ** 2).mean().backward()
        opt_b.step()
    loss_b_buggy = ((net_b(x) - y) ** 2).mean().item()

    # === Phase C（fixed）：load 後 set train_mode=True ===
    net_c, opt_c = _make_wrapper_pair()
    ckpt = torch.load(str(ckpt_path), weights_only=False)
    net_c.load_state_dict(ckpt["model_state_dict"])
    opt_c.load_state_dict(ckpt["optimizer_state_dict"])
    opt_c.train_mode = True   # ← 修法（training.py:386-388）
    opt_c.train()  # 現在 .train() 是 no-op（train_mode 已 True）
    for _ in range(50):
        opt_c.zero_grad()
        ((net_c(x) - y) ** 2).mean().backward()
        opt_c.step()
    loss_c = ((net_c(x) - y) ** 2).mean().item()

    # === 驗證 ===
    print(
        f"[test_wrapper_resume_train_mode_flag_fix] "
        f"reference loss_a={loss_a:.4e}, buggy loss_b={loss_b_buggy:.4e}, "
        f"fixed loss_c={loss_c:.4e}"
    )
    # Fixed 路徑必須 close to reference（同一條軌跡）
    assert abs(loss_c - loss_a) / max(loss_a, 1e-12) < 0.05, (
        f"修法應該讓 phase C ≈ phase A（同軌跡），實際 |{loss_c:.3e} - {loss_a:.3e}| 太大"
    )
    print("[test_wrapper_resume_train_mode_flag_fix] PASS")


if __name__ == "__main__":
    test_schedulefree_param_change_between_train_eval()
    test_resume_train_mode_preserves_loss_dynamics()
    test_resume_eval_mode_ckpt_warns_or_fails()
    test_wrapper_resume_train_mode_flag_fix()
    print("\n=== ScheduleFree resume tests done ===")
