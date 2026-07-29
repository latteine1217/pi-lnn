"""Canonical spectral diagnostics — 單一實作，對齊論文 §3 的譜定義。

What:
    `radial_energy_spectrum` 是全專案唯一的徑向能譜實作，供 DNS、LES 與重建場
    共用。論文 thesis/contents/chapter03.tex (\\S subsec:spectrum_definition) 明文
    宣稱「A single implementation evaluates it for DNS, LES, and reconstructed
    fields」——本模組就是那個實作。

Why:
    此模組建立前，repo 內有 9 份各自獨立的徑向能譜實作。稽核（2026-07-29）確認
    其中 8 份數值等價，但 `baseline_comparison_full.py` 用 `.astype(int)` 做 bin
    索引（截斷而非四捨五入），shell 定義偏移半個 bin，逐點誤差最大達 33%
    (k=4)。該錯誤逃過既有的兩道檢查：Parseval 仍守恆（能量只是被分到錯的 bin，
    總和不變），log-log tail slope 只偏 0.0013（正負誤差交錯抵消）。收斂成單一
    實作是唯一能結構性排除此類分歧的方式。

論文定義（eq:fft_norm / eq:radial_spectrum / eq:parseval）:
    1. 變異數保持的變換    û_k = (1/n²) Σ_x u(x) e^{-2πi k·x}
    2. Cyclic（非 angular）波數，單位盒上取整數值——與 forcing k_f = 2、
       與 sensor-count scale √(K/π) 同一個量
    3. Shell 為半開環帶  k - Δk/2 ≤ |k| < k + Δk/2，Δk = 1
    4. E(k) = Σ_shell ½(|û_k|² + |v̂_k|²)，雙分量
    5. 只保留至 isotropic Nyquist k = n/2；|k| > n/2 的 corner mode 捨棄而非
       折回末個 shell；k = 0 的 mean mode 排除
    6. 不做 pair-count 補償——Durran et al. 的補償會破壞 Parseval，而
       eq:parseval 在本研究是 load-bearing（band energy 是 E(k)Δk 的部分和，
       需與 KE 交叉核對）
"""
from __future__ import annotations

import numpy as np

__all__ = ["radial_energy_spectrum", "parseval_residual"]


def radial_energy_spectrum(
    u: np.ndarray, v: np.ndarray, dx: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (k, E(k)) for one snapshot on the unit box.

    Args:
        u, v: velocity components, shape (n, n).
        dx: optional grid spacing. The paper's convention fixes the domain to
            the unit box, so the only consistent value is 1/n. It is accepted
            for call-site compatibility and validated rather than used — a
            wrong dx would silently rescale every wavenumber, so it must fail
            loud instead.

    Returns:
        k: integer cyclic wavenumbers 1 … n/2, shape (n/2,).
        E: shell-summed energy, same shape. Because Δk = 1, E(k) is numerically
           the energy in shell k while carrying spectral-density units.
    """
    if u.shape != v.shape:
        raise ValueError(f"u and v must share shape; got {u.shape} vs {v.shape}")
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError(f"expected a square 2D snapshot; got shape {u.shape}")

    n = u.shape[0]
    if dx is not None and not np.isclose(dx, 1.0 / n, rtol=1e-12):
        raise ValueError(
            f"dx={dx!r} is inconsistent with the unit-box convention (expected "
            f"{1.0 / n!r} for n={n}). A mismatched dx rescales every wavenumber "
            "and would silently invalidate k_f and sqrt(K/pi) comparisons."
        )

    # eq:fft_norm — 1/n² 前綴使 Parseval 取 eq:parseval 的形式。常見 FFT 函式庫
    # 把該因子掛在逆變換上，會得到差 n² 的譜密度。
    uh = np.fft.fft2(u) / n**2
    vh = np.fft.fft2(v) / n**2
    e2d = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2)

    # Cyclic 整數波數：fftfreq(n) 回傳 cycles/sample，乘 n 得 cycles/box。
    k1d = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(k1d, k1d, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2)

    # 半開環帶 k-0.5 ≤ |k| < k+0.5。floor(|k|+0.5) 而非 astype(int)：後者是截斷，
    # 會把 shell 錯位成 k ≤ |k| < k+1。
    n_bins = n // 2
    bin_idx = np.floor(kmag + 0.5).astype(np.int64)
    # bin_idx == 0 排除 mean mode；> n_bins 捨棄 corner mode（不折回末個 shell）。
    valid = (bin_idx >= 1) & (bin_idx <= n_bins)
    flat_idx = np.where(valid, bin_idx - 1, 0).ravel()
    weights = np.where(valid, e2d, 0.0).ravel()
    e_k = np.bincount(flat_idx, weights=weights, minlength=n_bins).astype(np.float64)

    return np.arange(1, n_bins + 1, dtype=np.float64), e_k


def parseval_residual(u: np.ndarray, v: np.ndarray) -> float:
    """Return 1 - ΣE(k)/KE — the fraction of energy the shell sum fails to capture.

    Non-zero only through the discarded corner modes, so it stays negligible for
    the energy-concentrated fields treated here and grows for broadband ones.
    Use it to check that eq:parseval holds before trusting band-aggregated
    quantities derived from E(k).
    """
    _, e_k = radial_energy_spectrum(u, v)
    ke = 0.5 * float(np.mean(u**2 + v**2))
    if ke <= 0.0:
        return float("nan")
    return 1.0 - float(e_k.sum()) / ke
