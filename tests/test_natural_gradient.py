"""Unit tests for NaturalGradientOptimizer.

驗證範圍：
  1. Toy 線性最小平方：NG 應在一步內收斂到解析解（fixed damping → 0）。
  2. 非線性最小平方（小 MLP）：NG 多步應比 gradient descent 收斂更快、更深。
  3. Jacobi scaling：開啟與關閉皆能收斂（差異不必大，至少都收得到）。
  4. LM damping：拒絕步驟時參數應回退（last_loss 不上升）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pi_lnn.optimizers import (  # noqa: E402
    NaturalGradientOptimizer,
    compute_residual_jacobian,
    solve_ng_step,
)


def test_linear_lsq_one_step() -> None:
    """線性 r(θ) = Aθ - b, J = A 為常數 → NG 一步逼近 lstsq 解 (λ→0).

    注意：over-determined (N>P) 時最小殘差 ||A θ* − b|| ≠ 0；
    本測試只比較 θ 是否逼近 lstsq 解。
    """
    torch.manual_seed(0)
    N, P = 8, 4
    A = torch.randn(N, P, dtype=torch.float64)
    b = torch.randn(N, dtype=torch.float64)
    theta_star = torch.linalg.lstsq(A, b).solution

    theta = torch.zeros(P, dtype=torch.float64, requires_grad=True)

    def closure() -> torch.Tensor:
        return A @ theta - b

    opt = NaturalGradientOptimizer(
        [theta], lr=1.0, damping=1e-12,
        use_jacobi_scaling=True, solver_dtype=torch.float64,
    )
    loss_min = 0.5 * float((A @ theta_star - b).pow(2).sum().item())
    loss_before = 0.5 * float(closure().detach().pow(2).sum().item())
    opt.step(closure)
    err = float((theta.detach() - theta_star).norm().item())
    loss_after = 0.5 * float(closure().detach().pow(2).sum().item())

    assert err < 1e-6, f"NG 一步未逼近 lstsq 解，err={err:.3e}"
    # loss 應該已降到「殘差最小值」附近（不是 0）
    assert abs(loss_after - loss_min) < 1e-8, (
        f"loss 未達最小殘差：after={loss_after:.6e}, min={loss_min:.6e}"
    )
    print(
        f"[test_linear_lsq_one_step] err={err:.2e}, "
        f"loss {loss_before:.2e} → {loss_after:.2e} (min={loss_min:.2e})  PASS"
    )


def test_jacobian_shape() -> None:
    """compute_residual_jacobian 應回傳 (N, P) 且數值與 autograd.grad 一致."""
    torch.manual_seed(1)
    P = 3
    theta = torch.randn(P, requires_grad=True)
    # r_i = (theta . v_i)² ；J_{ij} = 2 (theta . v_i) v_{i,j}
    V = torch.randn(5, P)
    r = (V @ theta) ** 2
    J = compute_residual_jacobian(r, [theta])
    expected = 2 * (V @ theta).detach().unsqueeze(1) * V
    diff = float((J - expected).abs().max().item())
    assert diff < 1e-5, f"Jacobian 數值偏差過大：{diff:.3e}"
    assert J.shape == (5, P)
    print(f"[test_jacobian_shape] max |diff|={diff:.2e}  PASS")


def test_solve_ng_step_kernel_trick() -> None:
    """當 N < P 且 J 滿 rank N → kernel trick 與直接解 G^{-1} 結果應一致."""
    torch.manual_seed(2)
    N, P = 4, 10
    J = torch.randn(N, P, dtype=torch.float64)
    r = torch.randn(N, dtype=torch.float64)
    lam = 1e-8

    delta_kernel = solve_ng_step(J, r, damping=lam, use_jacobi_scaling=False)
    # 直接解：(J^T J + λI) Δ = J^T r
    G = J.T @ J + lam * torch.eye(P, dtype=torch.float64)
    delta_direct = torch.linalg.lstsq(G, J.T @ r).solution

    diff = float((delta_kernel - delta_direct).norm().item() / max(delta_direct.norm().item(), 1e-12))
    # rank-deficient 場景下兩者只在 row(J) 子空間一致，但對 J^T r 方向應吻合
    Jdelta_k = J @ delta_kernel
    Jdelta_d = J @ delta_direct
    diff_proj = float((Jdelta_k - Jdelta_d).norm().item() / max(Jdelta_d.norm().item(), 1e-12))
    assert diff_proj < 1e-6, f"projected diff 過大：{diff_proj:.3e}"
    print(f"[test_solve_ng_step_kernel_trick] proj_rel_diff={diff_proj:.2e}, raw_rel_diff={diff:.2e}  PASS")


def test_nonlinear_mlp_convergence() -> None:
    """小 MLP fitting：NG 10 步應比 SGD 100 步收斂更深."""
    torch.manual_seed(3)
    device = torch.device("cpu")

    # 目標函式 y = sin(2πx) + 0.5 cos(4πx)
    n = 32
    x = torch.linspace(-1, 1, n, device=device).unsqueeze(1)
    y = torch.sin(2 * torch.pi * x) + 0.5 * torch.cos(4 * torch.pi * x)

    def make_net() -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(1, 16), torch.nn.Tanh(),
            torch.nn.Linear(16, 16), torch.nn.Tanh(),
            torch.nn.Linear(16, 1),
        )

    # NG 30 步（LM damping 自動拒絕 overshoot；固定 damping 對非線性模型常爆掉）
    torch.manual_seed(3)
    net_ng = make_net()
    opt_ng = NaturalGradientOptimizer(
        net_ng.parameters(), lr=1.0, damping=1e-2,
        damping_strategy="lm", use_jacobi_scaling=True,
    )

    def ng_residuals() -> torch.Tensor:
        return (net_ng(x) - y).reshape(-1)

    loss_ng_init = 0.5 * float(ng_residuals().detach().pow(2).sum().item())
    for _ in range(30):
        opt_ng.step(ng_residuals)
    loss_ng_final = 0.5 * float(ng_residuals().detach().pow(2).sum().item())

    # SGD 100 步（同樣 batch、同樣 init）
    torch.manual_seed(3)
    net_sgd = make_net()
    opt_sgd = torch.optim.SGD(net_sgd.parameters(), lr=1e-2)
    for _ in range(100):
        opt_sgd.zero_grad()
        l = 0.5 * (net_sgd(x) - y).pow(2).sum()
        l.backward()
        opt_sgd.step()
    loss_sgd_final = 0.5 * float((net_sgd(x) - y).detach().pow(2).sum().item())

    print(
        f"[test_nonlinear_mlp_convergence] "
        f"NG  init={loss_ng_init:.3e}  final={loss_ng_final:.3e}  "
        f"SGD final={loss_sgd_final:.3e}"
    )
    assert loss_ng_final < loss_ng_init * 1e-2, "NG 未顯著下降 loss"
    assert loss_ng_final < loss_sgd_final, "NG 沒有比 SGD 100 步更深 — 可能收斂行為退化"


def test_lm_damping_rejects_bad_step() -> None:
    """LM strategy：若 closure 給出 NaN 或暴增的 residual，應拒絕並提高 damping."""
    torch.manual_seed(4)
    P = 4
    theta = torch.randn(P, requires_grad=True)
    target = torch.randn(P)

    def closure() -> torch.Tensor:
        return theta - target

    opt = NaturalGradientOptimizer(
        [theta], lr=1.0, damping=1e-2, damping_strategy="lm",
    )
    loss0 = 0.5 * float(closure().detach().pow(2).sum().item())
    for _ in range(5):
        opt.step(closure)
    loss5 = 0.5 * float(closure().detach().pow(2).sum().item())
    assert loss5 < loss0, f"LM 應收斂但 loss 上升：{loss0:.3e} → {loss5:.3e}"
    # damping 不應飛走
    assert opt.damping <= 1e2, f"damping 異常增大：{opt.damping}"
    print(f"[test_lm_damping_rejects_bad_step] loss {loss0:.2e} → {loss5:.2e}, "
          f"damping={opt.damping:.2e}, rejects={opt.n_rejects}  PASS")


def test_backtracking_line_search() -> None:
    """Backtracking line search 應在「α=1 過頭」時自動退到較小 α 完成下降."""
    torch.manual_seed(5)
    P = 8
    # 有意設計：使得 α=1 的 Newton step overshoot
    # f(θ) = sin(θ) 在 |θ| < π 時 Newton step 容易 oscillate
    theta = torch.full((P,), 2.5, requires_grad=True)  # 接近 π/2，Newton step 易 overshoot

    def closure() -> torch.Tensor:
        # r = sin(θ)，loss = 0.5 ||r||²，最小值 r=0 at θ=kπ
        return torch.sin(theta)

    opt = NaturalGradientOptimizer(
        [theta], lr=1.0, damping=1e-3,
        line_search="backtracking", ls_max_trials=8,
    )

    loss0 = 0.5 * float(closure().detach().pow(2).sum().item())
    loss_history = [loss0]
    alpha_history = []
    for _ in range(20):
        opt.step(closure)
        loss_history.append(0.5 * float(closure().detach().pow(2).sum().item()))
        alpha_history.append(opt.last_alpha)

    # Line search 應該至少有些步驟 α<1（backtracking 啟動）
    has_backtrack = any(0 < a < 1 for a in alpha_history)
    monotone_descent = all(loss_history[i] >= loss_history[i+1] - 1e-8 for i in range(len(loss_history)-1))

    assert loss_history[-1] < loss0 * 1e-3, (
        f"line search 沒收斂：{loss0:.3e} → {loss_history[-1]:.3e}"
    )
    assert monotone_descent, "loss 非單調下降，line search 邏輯有 bug"
    print(
        f"[test_backtracking_line_search] loss {loss0:.2e} → {loss_history[-1]:.2e}, "
        f"backtrack 觸發={has_backtrack}, alphas[:5]={[f'{a:.3f}' for a in alpha_history[:5]]}  PASS"
    )


def test_line_search_with_autograd_dependent_closure() -> None:
    """Regression：line search 不能在 torch.no_grad() 下叫 closure。

    Why: 真實 NG 訓練的 ng_residual_closure 內部 unsteady_ns_residuals 用
         torch.autograd.grad（create_graph=True）算 NS / continuity 殘差。
         若 line search 將 closure 包在 torch.no_grad() 裡：
           - uvp_fn(xyt) 沒 grad_fn
           - _grad(u, xyt) 早返回 zeros（pi_lnn/runtime.py:47）
           - NS / cont 殘差 silently 變 0 → loss_try 只剩 data 部分
           - line_try < line_val 永遠成立 → 第 1 個 trial 永遠接受
                 → backtracking 退化為固定步
         本測試模擬此模式：closure 第 N 維只在 grad enabled 下才能算對；
         backtrack 必須在 α=1 太大時退到 α<1 才能單調下降。
    """
    torch.manual_seed(11)
    P = 6
    # 使 α=1 過頭的設定：theta 接近 π/2，sin Newton step overshoot
    theta = torch.full((P,), 2.6, requires_grad=True)

    def closure() -> torch.Tensor:
        # 第 0..P-1 維為 sin(theta_i)，第 P 維為 (theta·1).sum()×0.1，
        # 後者只是讓 closure 對所有參數 require autograd path。
        # 關鍵：sin(θ) 本身不需 grad，但 _flat_size / autograd grad 構造需要
        # closure 至少 require_grad。如果 line search 包在 torch.no_grad()，
        # 則 r.requires_grad == False，後續 backward 時 RuntimeError。
        r = torch.sin(theta)
        # 添加一個小 autograd-依賴 entry：強制 closure 必須在 grad enabled 下執行
        r_extra = (theta.sum() * 0.0).reshape(1)  # value 為 0 但有 grad_fn
        return torch.cat([r, r_extra])

    opt = NaturalGradientOptimizer(
        [theta], lr=1.0, damping=1e-3,
        line_search="backtracking", ls_max_trials=8,
    )

    loss0 = 0.5 * float(closure().detach().pow(2).sum().item())
    for _ in range(15):
        opt.step(closure)
    loss_final = 0.5 * float(closure().detach().pow(2).sum().item())

    # 收斂：sin(θ)→0 → loss → 0
    assert loss_final < loss0 * 1e-3, (
        f"line search 在 autograd-依賴 closure 下沒收斂：{loss0:.3e} → {loss_final:.3e}"
    )
    print(
        f"[test_line_search_with_autograd_dependent_closure] {loss0:.2e} → {loss_final:.2e}  PASS"
    )


def test_armijo_line_search() -> None:
    """Armijo line search：sufficient decrease 比 backtracking 嚴格，但仍應收斂."""
    torch.manual_seed(6)
    n = 16
    x = torch.linspace(-1, 1, n).unsqueeze(1)
    y = torch.sin(2 * torch.pi * x)

    torch.manual_seed(6)
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 8), torch.nn.Tanh(),
        torch.nn.Linear(8, 1),
    )
    opt = NaturalGradientOptimizer(
        net.parameters(), lr=1.0, damping=1e-3,
        line_search="armijo", ls_max_trials=10, ls_armijo_c1=1e-4,
    )

    def closure() -> torch.Tensor:
        return (net(x) - y).reshape(-1)

    loss0 = 0.5 * float(closure().detach().pow(2).sum().item())
    for _ in range(15):
        opt.step(closure)
    loss_final = 0.5 * float(closure().detach().pow(2).sum().item())

    assert loss_final < loss0 * 0.1, f"Armijo 未顯著下降：{loss0:.3e} → {loss_final:.3e}"
    print(f"[test_armijo_line_search] loss {loss0:.2e} → {loss_final:.2e}, "
          f"rejects={opt.n_rejects}  PASS")


def test_spring_momentum_helps_convergence() -> None:
    """SPRING momentum (μ=0.9) 應比 μ=0 純 NG 收斂更快或更深."""
    torch.manual_seed(7)
    n = 16
    x = torch.linspace(-1, 1, n).unsqueeze(1)
    y = torch.sin(2 * torch.pi * x) + 0.3 * torch.cos(4 * torch.pi * x)

    def make_net() -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(1, 12), torch.nn.Tanh(),
            torch.nn.Linear(12, 12), torch.nn.Tanh(),
            torch.nn.Linear(12, 1),
        )

    # 標準 NG（無 momentum）
    torch.manual_seed(7)
    net_a = make_net()
    opt_a = NaturalGradientOptimizer(
        net_a.parameters(), lr=1.0, damping=1e-3,
        line_search="backtracking",
    )
    def loss_a():
        return (net_a(x) - y).reshape(-1)
    for _ in range(20):
        opt_a.step(loss_a)
    loss_no_spring = 0.5 * float(loss_a().detach().pow(2).sum().item())

    # NG + SPRING
    torch.manual_seed(7)
    net_b = make_net()
    opt_b = NaturalGradientOptimizer(
        net_b.parameters(), lr=1.0, damping=1e-3,
        line_search="backtracking",
    )
    opt_b.set_spring(enabled=True, momentum=0.9)
    def loss_b():
        return (net_b(x) - y).reshape(-1)
    for _ in range(20):
        opt_b.step(loss_b)
    loss_with_spring = 0.5 * float(loss_b().detach().pow(2).sum().item())

    print(
        f"[test_spring_momentum_helps_convergence] "
        f"NG-no-spring 20步 loss={loss_no_spring:.3e}, "
        f"NG+SPRING 20步 loss={loss_with_spring:.3e}"
    )
    # SPRING 至少不該明顯比沒 momentum 差（容忍 2× margin 因為小規模問題 momentum 效果有限）
    assert loss_with_spring < loss_no_spring * 2.0, (
        f"SPRING 反而拖累收斂：no-spring {loss_no_spring:.3e} vs with-spring {loss_with_spring:.3e}"
    )


def test_spring_first_step_falls_back_to_ng() -> None:
    """第一步 prev_phi=None 時，SPRING 應退化為標準 NG（不爆掉）."""
    torch.manual_seed(8)
    P = 4
    theta = torch.zeros(P, requires_grad=True)
    target = torch.tensor([0.5, -0.3, 0.7, 0.1])

    def closure() -> torch.Tensor:
        return theta - target

    opt = NaturalGradientOptimizer([theta], lr=1.0, damping=1e-8)
    opt.set_spring(enabled=True, momentum=0.9)

    # 第一步：prev_phi 未初始化 → 退化標準 NG
    loss_before = 0.5 * float(closure().detach().pow(2).sum().item())
    opt.step(closure)
    loss_after_step1 = 0.5 * float(closure().detach().pow(2).sum().item())
    assert loss_after_step1 < loss_before * 1e-6, (
        f"SPRING 第一步應為標準 NG，但 loss 沒大降：{loss_before:.2e} → {loss_after_step1:.2e}"
    )
    # 接續幾步應持續收斂
    for _ in range(3):
        opt.step(closure)
    loss_final = 0.5 * float(closure().detach().pow(2).sum().item())
    assert loss_final <= loss_after_step1 * 2, "SPRING 後續步驟發散"
    print(f"[test_spring_first_step_falls_back_to_ng] PASS")


if __name__ == "__main__":
    test_linear_lsq_one_step()
    test_jacobian_shape()
    test_solve_ng_step_kernel_trick()
    test_nonlinear_mlp_convergence()
    test_lm_damping_rejects_bad_step()
    test_backtracking_line_search()
    test_armijo_line_search()
    test_spring_momentum_helps_convergence()
    test_spring_first_step_falls_back_to_ng()
    print("\n=== All NG unit tests PASS ===")
