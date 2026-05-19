# Cylinder Wake 實驗紀錄

本文件是 **RealPDEBench Cylinder Wake**（非週期域）主線的 state 檔，從 [`docs/experiment_log.md`](experiment_log.md) 拆出（2026-05-15 拆檔）。

Kolmogorov（週期域）主線見 [`docs/experiment_log.md`](experiment_log.md)；歷史紀錄見 [`docs/experiment_archive_kolmogorov.md`](experiment_archive_kolmogorov.md) 與 [`docs/experiment_archive_kolmogorov_post_k100.md`](experiment_archive_kolmogorov_post_k100.md)。

---

## [INDEX] Cylinder Active

| ID | Status | 主題 | 一句結論 |
|---|---|---|---|
| `CEXP-002` | `ACTIVE_BASELINE` | Cylinder, K=100, **inflow BC loss**（bc_w=0.1, u_inf=0.33） | **KE 3.5%（Cylinder baseline）**；BC loss 從根本解決來流 collapse；振盪幅值略大（+10%）；div L2=1.14 仍有改善空間 |
| `CEXP-001` | `NEGATIVE_RESULT` | Cylinder, K=100, 無 BC loss | KE 51%（[PHYSICAL_FAILURE]）；無 inflow BC 導致來流區 u→0；已被 CEXP-002 取代 |

---

## [STATE] Cylinder Wake — 新主線建立（2026-04-27）

### 背景與目標

完成 Kolmogorov 稀疏重建研究後，轉向 RealPDEBench Cylinder Wake 案例：
- 非週期非均勻格（domain: [0, 0.325] × [0, 0.178]，含 cylinder body）
- K=100 QR-pivot sensor（Re=10031，T=3990 frames，dt=0.005s）
- 目標：驗證 PI-CON 能否在非週期域建立 baseline，為與 FLRNet / Energy Transformer 比較做準備

### 資料設定

- Arrow shard: `RealPDEBench/data/realpdebench/cylinder/hf_dataset/numerical/data-00000-of-00092.arrow`
- Re=10031（sim_id=10031.h5），T=3990, H=128, W=256（非均勻格）
- sensor 生成：`scripts/generate_sensors_qrpivot_cylinder.py`，`data/cylinder_sensors/`
- `sensor_subsample=20`：T=3990 → T=200（dt=0.1s），對齊 Kolmogorov 計算量
- 座標正規化至 [0,1]²（domain_length=1.0）

### 主線固定假設

- **非週期域必須加 inflow BC loss**：CEXP-001（無 BC）KE 51%，根因感測器 100% 集中尾跡，來流區無 supervision；加 `bc_loss_weight=0.1, bc_inflow_u=0.33 m/s, bc_n_points=64` 後 CEXP-002 KE 降至 3.5%（**14.5× 改善**）。Kolmogorov（週期域）不需要 BC。
- `use_physics_denormalization = true`（17 個 cylinder configs 主動 ON，保留 d62e698 的 fix 對 cylinder 的真實效益；Kolmogorov 主線預設 OFF）
- BC loss 採 `bc_inflow_u` 從 DNS Arrow shard 量測（`u[100:, :, 0].mean()`），cylinder Re=10031 量得 0.33 m/s

---

## [RECORD] Cylinder 實驗結果

### CEXP-001：無 BC baseline（KE=51%，失敗）

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_001_k100.toml` |
| Artifact | `artifacts/deeponet-cfc-cylinder-exp001-k100-warmup` |
| Checkpoint | `checkpoints/picon_kolmogorov_step_10000.pt` |
| KE rel-err mean | **51.0%** |
| u RMSE mean | 2.47e-1 |
| v RMSE mean | 9.99e-2 |
| div L2 mean | 1.13 |
| 結論 | [RESULT: PHYSICAL_FAILURE]：感測器全部集中尾跡（x>0.10），無 inflow BC 約束，模型在來流區輸出 u≈0 而非 u≈0.33 m/s，導致 KE 系統性低估 50%。 |

### CEXP-002：Inflow BC Loss（KE=3.5%，成功）

| 項目 | 值 |
|---|---|
| Config | `configs/exp_cylinder_002_k100_bc.toml` |
| Artifact | `artifacts/deeponet-cfc-cylinder-exp002-k100-bc` |
| Checkpoint | `checkpoints/picon_kolmogorov_step_10000.pt` |
| KE rel-err mean | **3.5%** |
| KE rel-err late | 3.9% |
| u RMSE mean | 1.03e-1 |
| v RMSE mean | 1.06e-1 |
| div L2 mean | 1.14 |
| 修改內容 | `bc_loss_weight=0.1`，`bc_inflow_u=0.33 m/s`，`bc_n_points=64`（x=0 均勻採樣） |
| 結論 | **BC loss 錨定來流速度後，KE 從 51% 降至 3.5%（14.5× 改善）**，與 Kolmogorov EXP-064（7.8%）相當。KE(t) 振盪幅值略大（峰值比 DNS 高 ~10%），渦街結構可識別。div L2=1.14 仍高，Kármán 渦核位置有偏移，但整體可視為 cylinder 稀疏重建 **baseline 建立**。 |

### 訓練紀錄摘要（CEXP-002）

| Step | L_data | L_phys | w_ns_u | w_cont | t_max |
|---|---|---|---|---|---|
| 1 | 6.676e+0 | 2.64e-1 | 0.010 | 0.010 | 0.5 |
| 1000 | 6.74e-3 | 9.99e-1 | 0.016 | 0.012 | 7.0 |
| 3000 | 1.46e-3 | 3.30e-1 | 0.024 | 0.016 | 20.0 |
| 6000 | 9.49e-4 | 1.01e-1 | 0.074 | 0.023 | 20.0 |
| 10000 | 1.15e-3 | 3.25e-2 | 0.108 | 0.038 | 20.0 |

---

## [DIAGNOSTIC] NaN 根因診斷（CEXP-001 早期，已修復）

**症狀**：CEXP-001 早期 step_500 checkpoint 有 83/95 個參數是 NaN。

**診斷流程**：

1. Physics OFF → 訓練穩定（L_data 在 step 400 降至 0.088）→ NaN 來自 physics
2. 物理殘差分解 → second derivatives（du_dx2, du_dy2）有 NaN，first derivatives 正常
3. NaN 點的 nearest_sensor_distance = 0 → collocation point 落在 sensor 位置

**根本原因**：`torch.linalg.norm(rel, dim=-1)` 在 `rel=0`（query = sensor）時，second-order autograd 計算 `∂²|r|/∂r² = (|r|²I - rr^T)/|r|³`，在 r=0 為 0/0 = NaN。

**修法**（[`src/pi_con/decoder.py`](../src/pi_con/decoder.py) `DeepONetCfCDecoder.forward`）：

```python
# 舊：rel_r = torch.linalg.norm(rel, dim=-1, keepdim=True)
# 新：
rel_r = torch.sqrt((rel**2).sum(dim=-1, keepdim=True) + 1e-8)
```

20 trials 驗證，NaN rate 0/20。

**為何 Kolmogorov 不會觸發此 bug**：K=100 / 256² = 0.15% 機率；Cylinder fluid grid 密集，sensor 也在 grid cells 上，觸發率 ~20%/batch。詳見 main log 的 KNOWN_PITFALLS section。

---

## [STATE] Cylinder Open Questions

| 問題 | 現況 | 狀態 |
|---|---|---|
| div L2=1.14 仍偏高（vs Kolmogorov EXP-064 0.184） | Kármán 渦核位置仍有偏移，physics constraint 在非均勻格收斂較慢 | 開放（CEXP-003 候選研究）|
| KE(t) 振盪幅值高 10% | 是 sensor 集中尾跡 + BC 錨定的不對稱 supervision 副作用 | 開放（需 sensor 多元化或 BC 加強）|
| 與 FLRNet / Energy Transformer 比較 | baseline 已建立，未實際比較 | 待規劃 |

---

## 引用與相互參照

- 與 Kolmogorov 共用的 NaN bug 修法見 [`CLAUDE.md`](../CLAUDE.md) 的 `<KNOWN_PITFALLS>` section（Physics Second-Order Autograd NaN）
- `use_physics_denormalization` flag 對 Cylinder 是必要的（不同於 Kolmogorov 主線），詳見 [`docs/diagnostics_log.md`](diagnostics_log.md) Physics Output Denormalization section
