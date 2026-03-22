# src/pi_onet/kolmogorov_dataset.py
"""KolmogorovDataset: 載入感測器時序 + LES ground truth，提供訓練/驗證採樣。

What: 從 JSON（感測器位置）、NPZ（感測器時序）、LES NPY（全場 ground truth）
      建立訓練資料結構，支援隨機採樣 (x, y, t, c) → ref 值。
Why:  將資料邏輯與模型解耦；lnn_kolmogorov.py 不直接接觸檔案格式。
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
    """Holds sensor data + LES field for one Re value.

    Attributes:
        sensor_vals: [K, T, 3] float32 — u,v,p at K sensors over T time steps
        sensor_pos:  [K, 2]   float32 — (x, y) in [0, 2π]
        sensor_time: [T]      float32 — physical time values
        les_u:  [T_les, N, N] float32
        les_v:  [T_les, N, N] float32
        les_p:  [T_les, N, N] float32
        les_x:  [N]           float32
        les_y:  [N]           float32
        les_time: [T_les]     float32
        re_value:  float
        re_norm:   float — normalised (re - RE_MEAN) / RE_STD
        dt_phys:   float — sensor time step (= 1.0)
        train_t_idx: [n_train] — LES time indices for training
        val_t_idx:   [n_val]   — LES time indices for validation
    """

    sensor_vals:  np.ndarray
    sensor_pos:   np.ndarray
    sensor_time:  np.ndarray
    les_u:        np.ndarray
    les_v:        np.ndarray
    les_p:        np.ndarray
    les_x:        np.ndarray
    les_y:        np.ndarray
    les_time:     np.ndarray
    re_value:     float
    re_norm:      float
    dt_phys:      float
    train_t_idx:  np.ndarray
    val_t_idx:    np.ndarray

    def __init__(
        self,
        sensor_json: str | Path,
        sensor_npz:  str | Path,
        les_path:    str | Path,
        re_value:    float,
        train_ratio: float = 0.8,
        seed:        int   = 42,
    ) -> None:
        rng = np.random.default_rng(seed)

        # ── 感測器位置（物理座標，單位 rad）──────────────────────────
        with open(sensor_json, encoding="utf-8") as f:
            meta = json.load(f)
        coords = np.array(meta["selected_coordinates"], dtype=np.float32)   # [K, 2]
        self.sensor_pos = coords   # (x, y) in [0, 2π]

        # ── 感測器時序 ───────────────────────────────────────────────
        npz = np.load(sensor_npz, allow_pickle=True)
        # u[K, T], v[K, T], p[K, T], time[T]
        u_s = npz["u"].astype(np.float32)    # [K, T]
        v_s = npz["v"].astype(np.float32)
        p_s = npz["p"].astype(np.float32)
        self.sensor_time = npz["time"].astype(np.float32)   # [T]
        self.sensor_vals = np.stack([u_s, v_s, p_s], axis=-1)  # [K, T, 3]
        # Δt = 感測器時間步長
        self.dt_phys = float(self.sensor_time[1] - self.sensor_time[0])

        # ── LES 全場 ─────────────────────────────────────────────────
        les = np.load(les_path, allow_pickle=True).item()
        self.les_u    = les["u"].astype(np.float32)     # [T_les, N, N]
        self.les_v    = les["v"].astype(np.float32)
        self.les_p    = les["p"].astype(np.float32)
        self.les_x    = les["x"].astype(np.float32)     # [N]
        self.les_y    = les["y"].astype(np.float32)
        self.les_time = les["time"].astype(np.float32)  # [T_les]

        # ── Re 正規化 ────────────────────────────────────────────────
        self.re_value = float(re_value)
        self.re_norm  = float((re_value - RE_MEAN) / RE_STD)

        # ── 訓練/驗證切割（依 LES 時間步切割）──────────────────────
        T_les = len(self.les_time)
        idx = np.arange(T_les)
        rng.shuffle(idx)
        n_train = int(T_les * train_ratio)
        self.train_t_idx = idx[:n_train]
        self.val_t_idx   = idx[n_train:]

    def sample_train_batch(
        self, rng: np.random.Generator, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """What: 從訓練集採樣 n 個 (x, y, t, c) 查詢點及對應 LES 參考值。

        Returns:
            xy:  [n, 2]  float32 — 物理座標 (x, y) in [0, 2π]
            t_q: [n]     float32 — 物理時間
            c:   [n]     int32   — component (0=u, 1=v, 2=p)
            ref: [n]     float32 — 對應 LES 值
        """
        return self._sample_batch(rng, self.train_t_idx, n)

    def sample_val_batch(
        self, rng: np.random.Generator, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """What: 從驗證集採樣。"""
        return self._sample_batch(rng, self.val_t_idx, n)

    def _sample_batch(
        self,
        rng: np.random.Generator,
        t_pool: np.ndarray,
        n: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t_idx = rng.choice(t_pool, size=n, replace=True)   # LES time indices
        xi    = rng.integers(0, len(self.les_x), size=n)
        yi    = rng.integers(0, len(self.les_y), size=n)
        c     = rng.integers(0, 3, size=n).astype(np.int32)

        xy  = np.stack([self.les_x[xi], self.les_y[yi]], axis=1).astype(np.float32)
        t_q = self.les_time[t_idx].astype(np.float32)

        field_map = {0: self.les_u, 1: self.les_v, 2: self.les_p}
        ref = np.array([
            field_map[int(c[i])][t_idx[i], yi[i], xi[i]]
            for i in range(n)
        ], dtype=np.float32)

        return xy, t_q, c, ref

    def sample_physics_points(
        self, rng: np.random.Generator, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """What: 採樣 n 個物理殘差點 (x, y, t)，時間在整個感測器時間範圍內均勻採樣。

        Returns:
            xy: [n, 2] float32
            t:  [n]    float32
        """
        xy = rng.uniform(0.0, 2 * np.pi, (n, 2)).astype(np.float32)
        t  = rng.uniform(self.sensor_time[0], self.sensor_time[-1], n).astype(np.float32)
        return xy, t
