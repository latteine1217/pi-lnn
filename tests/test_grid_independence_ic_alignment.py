"""Grid Independence Test for Kolmogorov DNS — IC alignment self-check.

What:
    驗證 modified generator 的兩個關鍵不變式：

    1. ic_mode='spectral_seeded' 對不同 N 在共同 grid points 給出 bit-exact 一致的 IC
       （Grid Independence Test 的物理一致性必要條件 —
        Gemini 2026-05-24 brainstorm 警告：「不同 N 呼叫 rng.standard_normal((N,N)) 不是 GI test」）

    2. ic_mode='band_limited_random' 內部 alignment loop 行為沒被破壞
       （Backward compat — 確保既有 EXP-200~268 訓練資料生成流程不受 generator 修改影響）

Why:
    GI test 需要 ~28 hr GPU time。若 IC 對齊 broken，GPU time 全浪費。本 pytest 是 **hard gate**。

Run:
    uv run pytest tests/test_grid_independence_ic_alignment.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


# 確保 vendored generator 可被 import
_REPO_ROOT = Path(__file__).resolve().parents[1]
_GEN_PATH = _REPO_ROOT / 'tools' / 'dns_generator' / 'generate_kolmogorov_dns_fp64.py'
_spec = importlib.util.spec_from_file_location('dns_generator_module', _GEN_PATH)
gen_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_mod)


# ── Common configuration for all tests ────────────────────────────────
COMMON_PARAMS = dict(
    L=1.0,
    nu=1e-4,
    A=0.1,
    k_f=2,
    dt=2.5e-4,
    backend='numpy',
    seed=42,
    integrator='etdrk4',
)


# ── Test 1: spectral_seeded IC 跨 N bit-exact 一致 ────────────────────
@pytest.mark.parametrize(
    "N_small,N_large",
    [(64, 128), (64, 256), (64, 512), (128, 256), (128, 512), (256, 512)],
)
def test_spectral_ic_n_invariance(N_small: int, N_large: int):
    """
    spectral_seeded IC: ifft(U_hat) at common grid points 對不同 N 必須一致.

    Tolerance: 1e-12 (機器 ε 等級，允許 FFT round-trip 浮點誤差)
    """
    dns_s = gen_mod.KolmogorovFlowDNS(
        N=N_small, ic_mode='spectral_seeded', ic_k_cutoff=8.0, **COMMON_PARAMS
    )
    dns_l = gen_mod.KolmogorovFlowDNS(
        N=N_large, ic_mode='spectral_seeded', ic_k_cutoff=8.0, **COMMON_PARAMS
    )

    u_s = np.real(np.fft.ifft2(dns_s.U_hat))
    v_s = np.real(np.fft.ifft2(dns_s.V_hat))
    u_l = np.real(np.fft.ifft2(dns_l.U_hat))
    v_l = np.real(np.fft.ifft2(dns_l.V_hat))

    stride = N_large // N_small
    u_l_sampled = u_l[::stride, ::stride]
    v_l_sampled = v_l[::stride, ::stride]

    np.testing.assert_allclose(
        u_s, u_l_sampled,
        atol=1e-12, rtol=0,
        err_msg=f"u N-invariance failed at (N_small={N_small}, N_large={N_large})",
    )
    np.testing.assert_allclose(
        v_s, v_l_sampled,
        atol=1e-12, rtol=0,
        err_msg=f"v N-invariance failed at (N_small={N_small}, N_large={N_large})",
    )


# ── Test 2: spectral_seeded post-rescale KE 一致到 target ──────────────
@pytest.mark.parametrize("N", [64, 128, 256, 512])
def test_spectral_ic_ke_target_match(N: int):
    """spectral_seeded mode 經 single multiplicative rescale 後 KE = target_initial_ke."""
    dns = gen_mod.KolmogorovFlowDNS(
        N=N, ic_mode='spectral_seeded', ic_k_cutoff=8.0, **COMMON_PARAMS
    )
    assert abs(dns.initial_ke - dns.target_initial_ke) / dns.target_initial_ke < 1e-10, (
        f"KE rescale failed at N={N}: got {dns.initial_ke:.6e}, "
        f"target {dns.target_initial_ke:.6e}"
    )


# ── Test 3: band_limited_random backward compat ───────────────────────
def test_backward_compat_band_limited_alignment():
    """
    ic_mode='band_limited_random' + seed=42 + N=256 跑完 iterative alignment 後:
    KE 與 forcing_coeff 必須 match 既有 target（即 JAXPI transient values）

    這是 surrogate 驗證「alignment 內部行為沒被破壞」.
    若 fail 表示既有 EXP-200~268 訓練資料的生成 pipeline 可能受影響.
    """
    dns = gen_mod.KolmogorovFlowDNS(
        N=256, ic_mode='band_limited_random', **COMMON_PARAMS
    )
    # alignment 必須對齊 target
    ke_err = abs(dns.initial_ke - dns.target_initial_ke)
    coeff_err = abs(dns.initial_forcing_coeff - dns.target_initial_forcing_coeff)
    assert ke_err < 1e-5, (
        f"band_limited alignment KE diverged: got {dns.initial_ke:.6e}, "
        f"target {dns.target_initial_ke:.6e}, diff {ke_err:.3e}"
    )
    assert coeff_err < 1e-5, (
        f"band_limited alignment forcing_coeff diverged: got {dns.initial_forcing_coeff:.6e}, "
        f"target {dns.target_initial_forcing_coeff:.6e}, diff {coeff_err:.3e}"
    )


# ── Test 4: spectral_seeded 對 ic_k_cutoff 太大時 raises ─────────────
def test_spectral_ic_raises_when_N_too_small():
    """N < 2*ceil(k_cutoff)+1 時必須 raise ValueError（防呆）."""
    with pytest.raises(ValueError, match="too small to contain k_cutoff"):
        gen_mod.KolmogorovFlowDNS(
            N=16, ic_mode='spectral_seeded', ic_k_cutoff=8.0, **COMMON_PARAMS
        )
