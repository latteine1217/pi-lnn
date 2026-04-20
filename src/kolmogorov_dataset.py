# src/kolmogorov_dataset.py
"""KolmogorovDataset: 載入感測器時序 + DNS ground truth，提供 sparse-data 採樣。

What: 從 JSON（感測器位置）、NPZ（感測器時序）、DNS NPY（全場 ground truth）
      建立訓練資料結構，支援感測器觀測 supervision 與 physics points 採樣。
Why:  sparse-data 主線只能監督真實可量測的感測器通道；即使檔案中附帶
      omega 等衍生量，也不應默認視為感測器量測值。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RE_MEAN: float = 5500.0    # mean of {100, 1000, 10000, 100000} — 近似中值
RE_STD:  float = 4000.0    # 近似 std，用於正規化


@dataclass
class KolmogorovDataset:
    """Holds sensor data + DNS field for one Re value.

    Attributes:
        sensor_vals: [K, T, C] float32 — normalized observed channels at K sensors over T time steps
        observed_channel_names: tuple[str, ...] — e.g. ("u","v")
        observed_channel_mean/std: [C] float32 — per-channel normalization stats
        sensor_pos:  [K, 2]   float32 — (x, y) in physical domain
        sensor_time: [T]      float32 — physical time values
        dns_u:  [T_dns, N, N] float32
        dns_v:  [T_dns, N, N] float32
        dns_p:  [T_dns, N, N] float32
        dns_x:  [N]           float32
        dns_y:  [N]           float32
        dns_time: [T_dns]     float32
        re_value:  float
        re_norm:   float — normalised (re - RE_MEAN) / RE_STD
        dt_phys:   float — sensor time step (= 1.0)
        train_t_idx: [n_train] — DNS time indices for training
        val_t_idx:   [n_val]   — DNS time indices for validation
    """

    sensor_vals:  np.ndarray
    observed_channel_names: tuple[str, ...]
    observed_channel_mean: np.ndarray
    observed_channel_std: np.ndarray
    sensor_pos:   np.ndarray
    sensor_time:  np.ndarray
    dns_u:        np.ndarray
    dns_v:        np.ndarray
    dns_p:        np.ndarray
    dns_x:        np.ndarray
    dns_y:        np.ndarray
    dns_time:     np.ndarray
    re_value:     float
    re_norm:      float
    dt_phys:      float
    train_t_idx:  np.ndarray
    val_t_idx:    np.ndarray

    def __init__(
        self,
        sensor_json: str | Path,
        sensor_npz:  str | Path,
        dns_path:    str | Path,
        re_value:    float,
        observed_channel_names: tuple[str, ...] | list[str] | None = None,
        train_ratio: float = 0.8,
        seed:        int   = 42,
    ) -> None:
        rng = np.random.default_rng(seed)

        # ── 感測器位置（物理座標）────────────────────────────────────
        with open(sensor_json, encoding="utf-8") as f:
            meta = json.load(f)
        coords = np.array(meta["selected_coordinates"], dtype=np.float32)   # [K, 2]
        self.sensor_pos = coords

        # ── 感測器時序 ───────────────────────────────────────────────
        npz = np.load(sensor_npz, allow_pickle=True)
        requested_names = tuple(observed_channel_names) if observed_channel_names is not None else ("u", "v")
        observed_names: list[str] = []
        observed_fields: list[np.ndarray] = []
        for key in requested_names:
            if key in npz:
                observed_names.append(key)
                observed_fields.append(npz[key].astype(np.float32))
        if not observed_fields:
            raise ValueError(
                f"感測器檔 {sensor_npz} 不含指定觀測通道 {requested_names}。"
            )
        self.sensor_time = npz["time"].astype(np.float32)   # [T]
        self.observed_channel_names = tuple(observed_names)
        raw_sensor_vals = np.stack(observed_fields, axis=-1)  # [K, T, C]
        self.observed_channel_mean = raw_sensor_vals.mean(axis=(0, 1)).astype(np.float32)
        self.observed_channel_std = raw_sensor_vals.std(axis=(0, 1)).astype(np.float32)
        self.observed_channel_std = np.maximum(self.observed_channel_std, 1.0e-6).astype(np.float32)
        self.sensor_vals = (
            (raw_sensor_vals - self.observed_channel_mean[None, None, :])
            / self.observed_channel_std[None, None, :]
        ).astype(np.float32)
        # Δt = 感測器時間步長
        self.dt_phys = float(self.sensor_time[1] - self.sensor_time[0])

        # ── DNS 全場 ─────────────────────────────────────────────────
        dns = np.load(dns_path, allow_pickle=True).item()
        self.dns_u    = dns["u"].astype(np.float32)     # [T_dns, N, N]
        self.dns_v    = dns["v"].astype(np.float32)
        self.dns_p    = dns["p"].astype(np.float32)
        self.dns_x    = dns["x"].astype(np.float32)     # [N]
        self.dns_y    = dns["y"].astype(np.float32)
        self.dns_time = dns["time"].astype(np.float32)  # [T_dns]
        # ── Re 正規化 ────────────────────────────────────────────────
        self.re_value = float(re_value)
        self.re_norm  = float((re_value - RE_MEAN) / RE_STD)

        # ── 訓練/驗證切割（依 DNS 時間步切割）──────────────────────
        T_dns = len(self.dns_time)
        idx = np.arange(T_dns)
        rng.shuffle(idx)
        n_train = int(T_dns * train_ratio)
        self.train_t_idx = idx[:n_train]
        self.val_t_idx   = idx[n_train:]

    def sample_sensor_batch(
        self, rng: np.random.Generator, n: int, t_max: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """What: 從感測器觀測採樣 n 個 (x, y, t, c_obs) 及對應實際量測值。

        t_max: 若給定，只從 sensor_time ≤ t_max 的時間步採樣（time-marching 用）
        """
        t_pool = np.arange(len(self.sensor_time), dtype=np.int32)
        if t_max is not None:
            t_pool = t_pool[self.sensor_time[t_pool] <= t_max]
            if len(t_pool) == 0:
                t_pool = np.array([0], dtype=np.int32)

        t_idx = rng.choice(t_pool, size=n, replace=True)
        sensor_idx = rng.integers(0, self.sensor_pos.shape[0], size=n)
        c_obs = rng.integers(0, len(self.observed_channel_names), size=n).astype(np.int32)

        xy = self.sensor_pos[sensor_idx].astype(np.float32)
        t_q = self.sensor_time[t_idx].astype(np.float32)
        ref = self.sensor_vals[sensor_idx, t_idx, c_obs].astype(np.float32)
        return xy, t_q, c_obs, ref

    def sample_physics_points(
        self,
        rng: np.random.Generator,
        n: int,
        t_max: float | None = None,
        strategy: str = "random",
    ) -> tuple[np.ndarray, np.ndarray]:
        """What: 採樣物理殘差點 (x, y, t)。

        Why: 不同採樣策略對物理 loss 能約束的空間頻率範圍有質的差異。
             "random" 的 n=32 點只能隨機覆蓋，期望約束到 k~5；
             "chebyshev" 的 n_dim×n_dim 格點系統性覆蓋，可約束至 k~n_dim。

        Args:
            n: 目標點數。"random" 回傳恰好 n 點；
               "chebyshev" 回傳 n_dim² 點，其中 n_dim = round(sqrt(n))。
            strategy:
                "random"    — 均勻隨機（原有行為，backward-compatible）。
                "chebyshev" — Gauss-Chebyshev 節點的 2D tensor-product grid；
                              時間軸仍均勻隨機（物理場在 t 方向非週期，隨機採樣合理）。
        """
        dx = float(self.dns_x[1] - self.dns_x[0]) if len(self.dns_x) > 1 else 1.0
        dy = float(self.dns_y[1] - self.dns_y[0]) if len(self.dns_y) > 1 else 1.0
        x_lo, x_hi = float(self.dns_x[0]), float(self.dns_x[-1] + dx)
        y_lo, y_hi = float(self.dns_y[0]), float(self.dns_y[-1] + dy)
        t_end = float(self.sensor_time[-1]) if t_max is None else min(float(t_max), float(self.sensor_time[-1]))
        t_start = float(self.sensor_time[0])

        if strategy == "chebyshev":
            # Gauss-Chebyshev Type-I 節點映射至物理域：
            # x_k = x_lo + 0.5*(1 - cos(π(2k+1)/(2n_dim))) * (x_hi - x_lo)
            # 這些節點在邊界附近密集，有效壓制 Runge 現象，覆蓋 Nyquist k ≈ n_dim。
            n_dim = max(2, round(n ** 0.5))
            k_idx = np.arange(n_dim)
            nodes_unit = 0.5 * (1.0 - np.cos(np.pi * (2 * k_idx + 1) / (2 * n_dim)))
            nodes_x = (x_lo + nodes_unit * (x_hi - x_lo)).astype(np.float32)
            nodes_y = (y_lo + nodes_unit * (y_hi - y_lo)).astype(np.float32)
            xx, yy = np.meshgrid(nodes_x, nodes_y)
            xy = np.stack([xx.ravel(), yy.ravel()], axis=1)  # [n_dim², 2]
            n_actual = len(xy)
            t = rng.uniform(t_start, t_end, n_actual).astype(np.float32)
            return xy, t

        # "random" — 原有行為
        xy = np.empty((n, 2), dtype=np.float32)
        xy[:, 0] = rng.uniform(x_lo, x_hi, n).astype(np.float32)
        xy[:, 1] = rng.uniform(y_lo, y_hi, n).astype(np.float32)
        t = rng.uniform(t_start, t_end, n).astype(np.float32)
        return xy, t
