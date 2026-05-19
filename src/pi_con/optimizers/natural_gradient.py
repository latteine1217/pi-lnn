"""Natural Gradient (Gauss-Newton) optimizer for PINN training.

論文：Curvature-Aware Optimization for High-Accuracy PINNs
       (Jnini et al., arXiv 2604.05230, 2026)

核心更新（Gauss-Newton）：
    L(θ)   = (1/2) ||r(θ)||²
    G(θ)   = J(θ)^T J(θ)             # Gauss-Newton 矩陣
    Δθ     = -α [G + λI]^{-1} J^T r

Kernel trick (適用 N ≤ P)：
    [J^T J + λI]^{-1} J^T = J^T [J J^T + λI]^{-1}
    → 解 N×N 系統，cost O(N²P)，遠優於 O(P³)

Jacobi scaling (van der Sluis, 1969；論文 §2.3)：
    K = J J^T,  D = diag(K)
    K̃ = D^{-1/2} K D^{-1/2}  → diag(K̃) = 1
    對 SPD 矩陣接近最小化 condition number。

設計取捨（pi-con 場景）
    - P ~ 10⁵, N ~ 100~500 → 強制走 kernel trick；
    - LinAlg 切到 fp64 + CPU：避開 MPS / fp32 ill-conditioning；
    - Jacobian 以 N 次 backward 逐 row 構造（torch.func.jacrev 對 CfC
      可能 vmap 失敗，採用通用且穩定的 fallback）。
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch import Tensor


# ---- 工具：parameter flatten / unflatten ---------------------------------
def _flat_size(params: list[Tensor]) -> int:
    return sum(p.numel() for p in params)


# ---- Jacobian / 線性求解 -------------------------------------------------
def compute_residual_jacobian(
    residuals: Tensor,
    params: list[Tensor],
) -> Tensor:
    """逐 row 計算 J = ∂r/∂θ ∈ R^{N × P}。

    成本：N 次 backward。建議 N < 500（pi-con 典型 N~200 OK）。
    Why 不用 torch.func.jacrev：
        CfC 內含時間迴圈與 in-place state 更新，functional_call + vmap
        在多 dataset / 多 Re 情境下穩定性差，逐 row autograd.grad 雖慢但通用。
    """
    if residuals.dim() != 1:
        residuals = residuals.reshape(-1)
    N = residuals.numel()
    P = _flat_size(params)
    J = residuals.new_zeros((N, P))
    for i in range(N):
        grads = torch.autograd.grad(
            residuals[i],
            params,
            retain_graph=(i < N - 1),
            create_graph=False,
            allow_unused=True,
        )
        offset = 0
        for g, p in zip(grads, params):
            n = p.numel()
            if g is not None:
                J[i, offset:offset + n] = g.detach().reshape(-1)
            offset += n
    return J


def solve_ng_step(
    J: Tensor,
    r: Tensor,
    damping: float = 1e-6,
    use_jacobi_scaling: bool = True,
    solver_dtype: torch.dtype = torch.float64,
    solver_device: str | torch.device = "cpu",
    mode: str = "auto",
) -> Tensor:
    """求 NG step。

    Damping 語義（重要 — 兩種模式的數學不同）：
        use_jacobi_scaling=False（純 LM）：解 (G + λI) Δθ = J^T r，G = J^T J。
        use_jacobi_scaling=True（預設）：等價於解 (G + λD) Δθ = J^T r，
            其中 D = diag(G)。實作上是先做 D^{-1/2} G D^{-1/2} 條件數均衡
            再加 λI，回代後 effective regulariser 為 λD（row-magnitude weighted），
            而非 isotropic λI。對 ill-conditioned PINN 殘差通常更穩，但 λ 的
            「絕對量級」不再對應 isotropic LM；λ 仍是「正則化強度旋鈕」，
            數值含義改為「相對於各 row 自身能量的比例」。

    自動切換兩種等價解法：
        kernel trick (N ≤ P)：解 N×N 系統 (J J^T + ...)。
        normal eq    (N >  P)：解 P×P 系統 (J^T J + ...)。
    pi-con 場景 P~10⁵ >> N~200 → 永遠走 kernel trick。

    Args:
        J: Jacobian, shape (N, P)。
        r: residual, shape (N,)；應為已乘權重平方根後的殘差，
           使 0.5 ||r||² = 訓練 loss。
        damping: Levenberg-Marquardt λ。注意：use_jacobi_scaling=True 時
                 effective regulariser 為 λD（diag scaling），參見上方語義說明。
        use_jacobi_scaling: 是否套用 D^{-1/2} K D^{-1/2} 預條件
                            （此選項會改變 damping 結構，見上方）。
        solver_dtype: 線性求解所用 dtype（建議 fp64）。
        solver_device: 線性求解所在 device（MPS 不支援 linalg.solve fp64，
                       預設搬到 CPU）。
        mode: "auto" | "kernel" | "normal"。
              "auto" 依 (N, P) 自動選擇；"kernel" 強制 kernel trick；
              "normal" 強制 normal equation。

    Returns:
        Δθ ∈ R^P，dtype/device 對齊輸入 J。
    """
    if J.dim() != 2:
        raise ValueError(f"J 必須是 2D，實得 shape={tuple(J.shape)}")
    N, P = J.shape
    if r.numel() != N:
        raise ValueError(f"r 大小 {r.numel()} 與 J row 數 {N} 不符")
    if mode not in ("auto", "kernel", "normal"):
        raise ValueError(f"mode 必須是 'auto' / 'kernel' / 'normal'，收到 {mode!r}")

    orig_dtype = J.dtype
    orig_device = J.device

    # 兩段式 cast：先換 device 再換 dtype。
    # Why: PyTorch `.to(dtype=fp64, device=cpu)` 在 MPS tensor 會先嘗試 cast fp64 in-place
    #      （MPS 不支援 fp64），導致 TypeError。先搬到 CPU 再 cast dtype 才安全。
    J64 = J.detach().to(device=solver_device).to(dtype=solver_dtype)
    r64 = r.detach().to(device=solver_device).to(dtype=solver_dtype).reshape(-1)

    use_kernel = (mode == "kernel") or (mode == "auto" and N <= P)

    if use_kernel:
        # Kernel trick：解 N×N 系統
        K = J64 @ J64.T  # (N, N)
        eye_N = torch.eye(N, dtype=solver_dtype, device=solver_device)

        if use_jacobi_scaling:
            d = K.diag().clamp(min=1e-12)
            d_inv_sqrt = d.rsqrt()
            K_tilde = K * d_inv_sqrt.view(-1, 1) * d_inv_sqrt.view(1, -1)
            rhs = d_inv_sqrt * r64
            try:
                y = torch.linalg.solve(K_tilde + damping * eye_N, rhs)
            except (torch.linalg.LinAlgError, RuntimeError):
                y = torch.linalg.lstsq(K_tilde + damping * eye_N, rhs).solution
            z = d_inv_sqrt * y
        else:
            try:
                z = torch.linalg.solve(K + damping * eye_N, r64)
            except (torch.linalg.LinAlgError, RuntimeError):
                z = torch.linalg.lstsq(K + damping * eye_N, r64).solution
        delta = J64.T @ z  # (P,)
    else:
        # Normal equation：解 P×P 系統，當 N > P 較穩
        G = J64.T @ J64  # (P, P)
        eye_P = torch.eye(P, dtype=solver_dtype, device=solver_device)
        rhs = J64.T @ r64  # (P,)

        if use_jacobi_scaling:
            d = G.diag().clamp(min=1e-12)
            d_inv_sqrt = d.rsqrt()
            G_tilde = G * d_inv_sqrt.view(-1, 1) * d_inv_sqrt.view(1, -1)
            rhs_s = d_inv_sqrt * rhs
            try:
                y = torch.linalg.solve(G_tilde + damping * eye_P, rhs_s)
            except (torch.linalg.LinAlgError, RuntimeError):
                y = torch.linalg.lstsq(G_tilde + damping * eye_P, rhs_s).solution
            delta = d_inv_sqrt * y
        else:
            try:
                delta = torch.linalg.solve(G + damping * eye_P, rhs)
            except (torch.linalg.LinAlgError, RuntimeError):
                delta = torch.linalg.lstsq(G + damping * eye_P, rhs).solution

    return delta.to(dtype=orig_dtype, device=orig_device)


# ---- Optimizer ----------------------------------------------------------
class NaturalGradientOptimizer:
    """Gauss-Newton (Natural Gradient) optimizer with kernel trick.

    使用範例：
        opt = NaturalGradientOptimizer(net.parameters(), lr=1.0)

        def residual_closure() -> torch.Tensor:
            r = compute_residuals(net, batch)   # shape (N,)，含計算圖
            return r

        loss = opt.step(residual_closure)

    Args:
        params: 一般 net.parameters()；只取 requires_grad=True 的參數。
        lr: 學習率（NG step 通常 lr=1.0 即可）。
        damping: 初始 LM λ。建議 1e-6 起。
        damping_min/max: damping_strategy="lm" 時的可調整上下界。
        damping_strategy:
            "fixed" — 固定 damping（最便宜）
            "lm"    — Levenberg-Marquardt 風格自適應：
                      接受步驟則 λ /= 2，否則 λ *= 10 並回退（多花一次 closure 評估）。
        use_jacobi_scaling: 是否套用對角預條件（推薦 True）。
        solver_dtype: 線性求解 dtype，預設 fp64。
        solver_device: 線性求解 device，預設 "cpu"（避 MPS fp64 限制）。
        max_residuals: residual 向量大小上限；超過會 raise（O(N²P) 記憶體保護）。
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1.0,
        damping: float = 1e-6,
        damping_min: float = 1e-12,
        damping_max: float = 1e2,
        damping_strategy: str = "fixed",
        use_jacobi_scaling: bool = True,
        solver_dtype: torch.dtype = torch.float64,
        solver_device: str | torch.device = "cpu",
        max_residuals: int = 2000,
        # Line search params
        line_search: str = "none",
        ls_max_trials: int = 5,
        ls_alpha_init: float = 1.0,
        ls_alpha_decay: float = 0.5,
        ls_armijo_c1: float = 1.0e-4,
    ) -> None:
        self.params: list[Tensor] = [p for p in params if p.requires_grad]
        if not self.params:
            raise ValueError("NaturalGradientOptimizer 需要至少一個 requires_grad 參數")
        if damping_strategy not in ("fixed", "lm"):
            raise ValueError(f"damping_strategy 必須是 'fixed' 或 'lm'，收到 {damping_strategy!r}")
        if line_search not in ("none", "backtracking", "armijo"):
            raise ValueError(
                f"line_search 必須是 'none' / 'backtracking' / 'armijo'，收到 {line_search!r}"
            )

        self.lr = float(lr)
        self.damping = float(damping)
        self.damping_min = float(damping_min)
        self.damping_max = float(damping_max)
        self.damping_strategy = damping_strategy
        self.use_jacobi_scaling = bool(use_jacobi_scaling)
        self.solver_dtype = solver_dtype
        self.solver_device = solver_device
        self.max_residuals = int(max_residuals)

        # Line search
        # "none"         — 不做 line search（純 LM accept/reject 或 fixed）
        # "backtracking" — α=1, ls_alpha_decay, decay²,... 直到 loss_new < loss_old
        # "armijo"       — backtracking + Armijo sufficient decrease 條件
        # Why: 論文 Stokes case 用 line search（"NG with Line-search"），
        #      而 LM accept/reject 只能「全收/全退」，無法走「半步」，固定 batch
        #      容易讓 NG 過擬合並卡 fixed point。Line search 給 step size 連續控制。
        self.line_search = line_search
        self.ls_max_trials = int(ls_max_trials)
        self.ls_alpha_init = float(ls_alpha_init)
        self.ls_alpha_decay = float(ls_alpha_decay)
        self.ls_armijo_c1 = float(ls_armijo_c1)

        # SPRING momentum（paper eq.(26)，Goldshlager 2024 ref [26]）
        #   ϕ_k = μ·ϕ_{k-1} + J^T [JJ^T + λI]^{-1} (r - μ·J·ϕ_{k-1})
        #   θ_{k+1} = θ_k - α·ϕ_k
        # 物理意義：把前一步的「未走完方向」累積進當前 step，
        #          類似 Nesterov momentum，但作用在 GN-preconditioned 空間。
        self.use_spring = False  # 由 set_spring() 動態 enable
        self.spring_momentum = 0.0
        self._prev_phi: Tensor | None = None  # 上一步的 ϕ（flat P-dim）

        # 統計
        self._step_count = 0
        self._n_rejects = 0
        self._last_loss: float | None = None
        self._last_alpha: float = 1.0  # line search 找到的 step size（供 logging）

    def set_spring(self, enabled: bool, momentum: float = 0.9) -> None:
        """啟用 / 停用 SPRING momentum。

        Args:
            enabled: True 啟用 ϕ_k = μ·ϕ_{k-1} + ... 機制
            momentum: μ ∈ [0, 1)；0.9 是論文常用值。0 退化為標準 NG。
        """
        if enabled and not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum 必須 ∈ [0, 1)，收到 {momentum}")
        self.use_spring = bool(enabled)
        self.spring_momentum = float(momentum)
        if not enabled:
            self._prev_phi = None  # 清除 momentum state

    # ---- torch.optim 風格相容介面 ----------------------------------
    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def _apply_step(self, delta: Tensor, alpha: float = 1.0) -> None:
        """Apply θ ← θ − lr * alpha * delta。

        Args:
            delta: flattened update direction, shape (P,)
            alpha: line search step size scaling（1.0 = full Newton step）
        """
        scale = self.lr * alpha
        offset = 0
        for p in self.params:
            n = p.numel()
            chunk = delta[offset:offset + n].reshape(p.shape)
            chunk = chunk.to(dtype=p.dtype, device=p.device)
            p.add_(chunk, alpha=-scale)
            offset += n

    @torch.no_grad()
    def _restore_params(self, snapshot: list[Tensor]) -> None:
        for p, old in zip(self.params, snapshot):
            p.copy_(old)

    def step(self, residual_closure: Callable[[], Tensor]) -> float:
        """Run one NG step.

        Args:
            residual_closure: 回傳 1D residual tensor r ∈ R^N，
                              含計算圖（不可 detach），使 loss = 0.5 ||r||²。
        Returns:
            current loss = 0.5 ||r||²（fixed strategy 為 step 前的 loss；
            lm strategy 為接受步驟後的 loss，拒絕則仍為步前 loss）。
        """
        self.zero_grad()
        r = residual_closure()
        if not torch.is_tensor(r):
            raise TypeError(f"residual_closure 必須回傳 Tensor，收到 {type(r)!r}")
        r_flat = r.reshape(-1)
        if r_flat.numel() > self.max_residuals:
            raise ValueError(
                f"殘差向量大小 N={r_flat.numel()} > max_residuals={self.max_residuals}；"
                "請降低 num_query_points/num_physics_points，或上調 max_residuals"
                "（注意 J 的記憶體成本為 O(N · P)）"
            )

        loss_val = 0.5 * float(r_flat.detach().pow(2).sum().item())

        # J = ∂r/∂θ
        J = compute_residual_jacobian(r_flat, self.params)

        # SPRING momentum (paper eq.(26))：
        #   ϕ_k = μ·ϕ_{k-1} + J^T [JJ^T + λI]^{-1} (r - μ·J·ϕ_{k-1})
        # 第一步沒 prev_phi → μ_eff=0 退化為標準 NG。
        if self.use_spring and self._prev_phi is not None:
            # 把 prev_phi 對齊 J 的 device/dtype（J 在 model device）
            prev_phi = self._prev_phi.to(device=J.device, dtype=J.dtype)
            # r_eff = r - μ·J·ϕ_{k-1}
            r_eff = r_flat.detach() - self.spring_momentum * (J @ prev_phi)
            # 解 z = [JJ^T + λI]^{-1} r_eff，回傳 J^T z
            delta_kernel = solve_ng_step(
                J=J,
                r=r_eff,
                damping=self.damping,
                use_jacobi_scaling=self.use_jacobi_scaling,
                solver_dtype=self.solver_dtype,
                solver_device=self.solver_device,
            )
            # ϕ_k = μ·ϕ_{k-1} + J^T z
            delta = self.spring_momentum * prev_phi + delta_kernel
        else:
            # 標準 NG：Δθ = J^T (JJ^T + λI)^{-1} r
            delta = solve_ng_step(
                J=J,
                r=r_flat,
                damping=self.damping,
                use_jacobi_scaling=self.use_jacobi_scaling,
                solver_dtype=self.solver_dtype,
                solver_device=self.solver_device,
            )

        # ── Step acceptance strategies ──────────────────────────────────
        # 三種策略可組合：
        #   line_search="none" + damping_strategy="fixed" → 直接 α=1 走（最簡單）
        #   line_search="none" + damping_strategy="lm"    → α=1 試走，全收/全退（舊行為）
        #   line_search="backtracking"                    → α=1, 0.5, 0.25,... 找下降方向
        #   line_search="armijo"                          → backtracking + sufficient decrease
        if self.line_search in ("backtracking", "armijo"):
            # Line search：保留 LM 邏輯處理「所有 trial 都失敗」的情況
            # 對 backtracking：只要 loss_new < loss_val 就接受
            # 對 armijo：要求 loss_new ≤ loss_val − c1·α·||r||² (sufficient decrease)
            #     Why: 對 GN 而言 g^T·delta ≈ ||r||²（r 是 J^T r 的源頭），這是 paper 慣用近似
            snapshot = [p.detach().clone() for p in self.params]
            alpha = self.ls_alpha_init
            accepted = False
            loss_new = loss_val
            r_norm_sq = 2.0 * loss_val  # ||r||²
            for trial in range(self.ls_max_trials):
                self._restore_params(snapshot)  # 每次 trial 都從原點 try
                self._apply_step(delta, alpha=alpha)
                # 不能用 torch.no_grad()：residual_closure 內部 unsteady_ns_residuals
                # 透過 torch.autograd.grad 算 NS/continuity 殘差，no_grad 下會 silently
                # 把 NS / cont 退化為 0（_grad 早返回 zeros），導致 line search 永遠
                # 在 trial=0 接受（loss_try ≈ data_only_loss < loss_val 含 phys）。
                r_try = residual_closure()
                loss_try = 0.5 * float(r_try.detach().reshape(-1).pow(2).sum().item())

                if self.line_search == "armijo":
                    # Sufficient decrease: loss_new ≤ loss_old − c1·α·||r||²
                    threshold = loss_val - self.ls_armijo_c1 * alpha * r_norm_sq
                    if loss_try <= threshold:
                        loss_new = loss_try
                        accepted = True
                        break
                else:  # "backtracking"
                    if loss_try < loss_val:
                        loss_new = loss_try
                        accepted = True
                        break
                alpha *= self.ls_alpha_decay

            if accepted:
                self._last_alpha = alpha
                # 接受：縮小 damping（信任 NG 方向）
                if self.damping_strategy == "lm":
                    self.damping = max(self.damping_min, self.damping * 0.5)
                loss_val = loss_new
                # SPRING：接受步驟後更新 prev_phi（用實際走的 α·delta）
                if self.use_spring:
                    self._prev_phi = (alpha * delta).detach().clone()
            else:
                # 全部 trial 失敗：退回原點 + 加大 damping
                self._restore_params(snapshot)
                self._last_alpha = 0.0
                self._n_rejects += 1
                if self.damping_strategy == "lm":
                    self.damping = min(self.damping_max, self.damping * 10.0)
                # SPRING：拒絕步驟不更新 prev_phi（保留前次 momentum）

        elif self.damping_strategy == "lm":
            # LM 自適應：α=1 試走，全收/全退（舊行為）
            snapshot = [p.detach().clone() for p in self.params]
            self._apply_step(delta)
            # 同 line search：closure 內含 autograd-based NS 殘差，不能 no_grad。
            r_new = residual_closure()
            loss_new = 0.5 * float(r_new.detach().reshape(-1).pow(2).sum().item())
            if loss_new < loss_val:
                self.damping = max(self.damping_min, self.damping * 0.5)
                loss_val = loss_new
                self._last_alpha = 1.0
                if self.use_spring:
                    self._prev_phi = delta.detach().clone()
            else:
                self._restore_params(snapshot)
                self.damping = min(self.damping_max, self.damping * 10.0)
                self._n_rejects += 1
                self._last_alpha = 0.0
        else:
            # Fixed damping + 無 line search：直接 α=1 走
            self._apply_step(delta)
            self._last_alpha = 1.0
            if self.use_spring:
                self._prev_phi = delta.detach().clone()

        self._step_count += 1
        self._last_loss = loss_val
        return loss_val

    # ---- state_dict（簡化版；參數本身由外部 net.state_dict 管理）---
    def state_dict(self) -> dict:
        return {
            "step": self._step_count,
            "n_rejects": self._n_rejects,
            "damping": self.damping,
            "lr": self.lr,
            "last_loss": self._last_loss,
        }

    def load_state_dict(self, state: dict) -> None:
        self._step_count = int(state.get("step", 0))
        self._n_rejects = int(state.get("n_rejects", 0))
        self.damping = float(state.get("damping", self.damping))
        self.lr = float(state.get("lr", self.lr))
        self._last_loss = state.get("last_loss", None)

    # ---- 可觀察的訓練統計 ---------------------------------------------
    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def n_rejects(self) -> int:
        return self._n_rejects

    @property
    def last_alpha(self) -> float:
        """Line search 找到的最後 step size（0.0 = 全部 trial 失敗，已 reject）。"""
        return self._last_alpha
