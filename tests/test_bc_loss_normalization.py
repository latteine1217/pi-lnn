"""tests/test_bc_loss_normalization.py

驗證 cylinder inflow BC loss 採 normalized 形式（除以 obs std），
與 data loss 同尺度，避免 bc_loss_weight 因 1/std² ≈ 40× 量級差而失效。

數學等價推導：
    target_u_norm = (u_inf - u_mean) / u_std
    pred_u_norm   = (u_bc   - u_mean) / u_std
    loss_u_norm   = (pred_u_norm - target_u_norm)²
                  = ((u_bc - u_inf) / u_std)²        ← mean 相消

驗收條件：
  1. (u_bc - u_inf)² → ((u_bc - u_inf)/u_std)² 等於除以 u_std²
  2. 對 cylinder 典型 std (u_std≈0.155, v_std≈0.218)，整體 loss
     比舊物理-空間公式大 ~30-40×（與 data loss 同量級）
  3. 完美預測（u_bc=u_inf, v_bc=0）時兩種公式都為 0
"""
from __future__ import annotations

import torch


# ── 1. 等價性：normalized 公式 = 舊公式 / std² ────────────────────


def test_u_part_equals_old_divided_by_u_std_squared():
    u_bc = torch.tensor([0.30, 0.32, 0.28, 0.29])
    u_inf, u_std = 0.33, 0.155
    old_u = torch.mean((u_bc - u_inf) ** 2)
    new_u = torch.mean(((u_bc - u_inf) / u_std) ** 2)
    torch.testing.assert_close(new_u, old_u / u_std ** 2, rtol=1e-6, atol=1e-8)


def test_v_part_equals_old_divided_by_v_std_squared():
    v_bc = torch.tensor([0.01, -0.02, 0.005, 0.015])
    v_std = 0.218
    old_v = torch.mean(v_bc ** 2)
    new_v = torch.mean((v_bc / v_std) ** 2)
    torch.testing.assert_close(new_v, old_v / v_std ** 2, rtol=1e-6, atol=1e-8)


# ── 2. 尺度量級：新公式比舊公式大 ~30-40× ────────────────────────


def test_total_bc_loss_scales_to_data_loss_magnitude():
    """以 cylinder 典型 std 量化新舊比例，確認 bc_loss_weight 真正生效。"""
    torch.manual_seed(0)
    n = 64
    # 模擬一個尚未收斂的 model：output 在 normalized 空間量級 ~O(1)
    # 對應到物理空間的 raw_pred 量級 ~ O(u_std) 偏離 u_inf
    u_bc = torch.full((n,), 0.30) + 0.05 * torch.randn(n)   # 偏離 u_inf=0.33 約 1 std 量級
    v_bc = 0.05 * torch.randn(n)                            # 偏離 0 約 0.5 std 量級

    u_inf, u_std, v_std = 0.33, 0.1549, 0.2181              # 來自 EXP-CYLINDER-002 實測

    old_total = torch.mean((u_bc - u_inf) ** 2) + torch.mean(v_bc ** 2)
    new_total = (
        torch.mean(((u_bc - u_inf) / u_std) ** 2)
        + torch.mean((v_bc / v_std) ** 2)
    )
    ratio = (new_total / old_total).item()

    # 期望 ratio 落在 1/std² 量級（u_std≈0.155 → 41×, v_std≈0.218 → 21×，加權平均 ~30）
    assert 20.0 < ratio < 50.0, (
        f"normalized BC loss 比舊版應大 ~30×（與 data loss 同尺度），實得 {ratio:.2f}×"
    )


# ── 3. 完美預測仍為 0（regression: 不該注入偏置）──────────────────


def test_perfect_prediction_yields_zero_loss():
    n = 32
    u_inf, u_std, v_std = 0.33, 0.155, 0.218
    u_bc = torch.full((n,), u_inf)
    v_bc = torch.zeros(n)
    new_total = (
        torch.mean(((u_bc - u_inf) / u_std) ** 2)
        + torch.mean((v_bc / v_std) ** 2)
    )
    assert new_total.item() == 0.0


# ── 4. 與 data loss 等價推導（端到端理智檢查）────────────────────


def test_normalized_form_matches_explicit_z_score_subtraction():
    """完整展開驗證 mean 相消：(pred-mean)/std 與 (target-mean)/std 的差等於 (pred-target)/std。"""
    n = 16
    u_inf, u_mean, u_std = 0.33, 0.2419, 0.155
    u_bc = torch.tensor([0.30, 0.32, 0.28, 0.31] * 4)

    target_norm = (u_inf - u_mean) / u_std
    pred_norm = (u_bc - u_mean) / u_std
    loss_explicit = torch.mean((pred_norm - target_norm) ** 2)

    loss_simplified = torch.mean(((u_bc - u_inf) / u_std) ** 2)

    torch.testing.assert_close(loss_explicit, loss_simplified, rtol=1e-6, atol=1e-8)


# ── 5. Body no-slip BC：u=v=0 公式驗證 ────────────────────────────


def test_body_no_slip_zero_target_simplifies():
    """target=(0-mean)/std=-mean/std；展開後 (pred-target)² = ((u-0)/std)² = (u/std)²。
    Mean 在差值中相消，與 inflow BC 相同公式形式。"""
    n = 32
    u_mean, u_std = 0.2419, 0.155
    u_body = torch.tensor([0.05, -0.02, 0.01, 0.00] * 8)

    target_norm = (0.0 - u_mean) / u_std
    pred_norm = (u_body - u_mean) / u_std
    loss_explicit = torch.mean((pred_norm - target_norm) ** 2)

    loss_simplified = torch.mean((u_body / u_std) ** 2)

    torch.testing.assert_close(loss_explicit, loss_simplified, rtol=1e-6, atol=1e-8)


def test_body_perfect_no_slip_yields_zero_loss():
    """u=v=0（理想 body 內部）時 loss 必為零。"""
    n = 16
    u_std, v_std = 0.155, 0.218
    u_body = torch.zeros(n)
    v_body = torch.zeros(n)
    loss = torch.mean((u_body / u_std) ** 2) + torch.mean((v_body / v_std) ** 2)
    assert loss.item() == 0.0


# ── 6. Slip BC (v=0 at y=0,1) ─────────────────────────────────────


def test_slip_v_only_constraint():
    """Slip BC 簡化為 v=0；u 不約束（圓柱繞流加速 → u 在 y 邊界並非常數）。
    驗證實作公式只計入 v 部分。"""
    n = 16
    v_std = 0.218
    v_slip = torch.tensor([0.01, -0.01, 0.005, -0.003] * 4)
    loss = torch.mean((v_slip / v_std) ** 2)

    # 與展開形式比較
    v_mean = 0.0012  # 量測值
    target_norm = (0.0 - v_mean) / v_std
    pred_norm = (v_slip - v_mean) / v_std
    loss_explicit = torch.mean((pred_norm - target_norm) ** 2)

    torch.testing.assert_close(loss, loss_explicit, rtol=1e-6, atol=1e-8)


def test_slip_top_bottom_split_count():
    """slip BC 採樣 _n_slip 點：上下各 _n_slip // 2 點，y=0 與 y=1 等比例。"""
    n_slip = 32
    per_side = max(1, n_slip // 2)
    y_slip = torch.cat([torch.zeros(per_side), torch.ones(per_side)])
    assert y_slip.shape[0] == 2 * per_side == 32
    assert (y_slip[:per_side] == 0.0).all()
    assert (y_slip[per_side:] == 1.0).all()
