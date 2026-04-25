# pi-o-net

以 **DeepONet + CfC** 實作的 sparse-data 物理資訊算子學習模型，目標是用少量感測器時序資料重建 2D Kolmogorov flow 場。

目前正式主線架構：

- **觀測通道：`u, v`**（不使用 vorticity / spectrum / enstrophy 作為 supervision）
- **空間 domain：`[0, 1]^2` periodic**
- **空間編碼：`periodic_fourier_encode`**（確定性 Fourier 編碼，嚴格週期邊界，取代早期 RFF）
- **等向 relpos_bias**（純距離 `|rel|`，無方向分量，避免條紋偽影）
- **temporal_anchor**（`sin/cos(2π n t/T)` 絕對時間座標，n_harmonics=2）
- **token self-attention + temporal CfC encoder**
- **DeepONet-style query decoder with cross-attention**
- **訓練只依賴真實觀測（u, v）與 PDE 殘差（momentum + continuity）**

## 架構資料流

```text
sensor_obs[t, k, {u,v}] + sensor_pos[k, {x,y}]
    -> periodic_fourier_encode(x,y) + token_in
    -> thin sensor tokens
    -> token self-attention
    -> temporal CfC over time
    -> causal branch tokens h_states[t, k, d]
    -> query trunk (x, y, t, component, temporal_anchor)
       + relpos cross-attention(query → branch tokens)
    -> branch/trunk operator fusion
    -> latent field {u, v, p}
```

補充：

- `p` 為模型內部 latent 場量，只由 PDE residual 約束，無資料監督
- `omega`, KE, Enstrophy, energy spectrum 只作 evaluation 診斷，不進 training

對應實作在：

- [`src/pi_onet/lnn_kolmogorov.py`](src/pi_onet/lnn_kolmogorov.py)
- [`src/pi_onet/kolmogorov_dataset.py`](src/pi_onet/kolmogorov_dataset.py)

## 目前最佳結果

### Re=1000（主基準，EXP-030）

| 指標 | 數值 |
|---|---|
| KE rel-err | **9.61%** |
| u RMSE | 5.68e-2 |
| amp ratio (k_f=2) | 1.027 |
| Enstrophy rel-err | 11.85% |

- config: [`configs/deeponet_cfc_midlong_uvomega_small_soap_sf_5000.toml`](configs/deeponet_cfc_midlong_uvomega_small_soap_sf_5000.toml)
- artifact: `artifacts/deeponet-cfc-re1000-soap-sf-5000`
- 架構：`d_model=64`, `fourier_harmonics=8`, 1-layer attn, SOAP+SF, 5000 steps

### Re=10000（最佳，EXP-064）

| 指標 | 數值 |
|---|---|
| KE rel-err | **7.80%** |
| div L2 | 0.184 |
| u / v RMSE | 0.0689 / 0.0621 |
| amp ratio (k_f=2) | 0.9615 |
| phase error | -0.0228 rad |

- config: `configs/exp_064_re10000_xlarge_sensor_physics.toml`
- artifact: `artifacts/deeponet-cfc-re10000-exp064-sensor-physics`
- 架構：`d_model=256`, `LearnableFourierEmb(embed_dim=128)`, GradNorm, sensor continuity physics, SOAP+SF, 10000 steps

## 安裝

```bash
uv sync
```

驗證可解譯：

```bash
uv run python -m py_compile \
  src/pi_onet/lnn_kolmogorov.py \
  src/pi_onet/kolmogorov_dataset.py \
  scripts/train_deeponet_cfc.py \
  scripts/evaluate_deeponet_cfc.py
```

## 訓練

```bash
uv run python scripts/run_experiment.py \
  --config configs/deeponet_cfc_midlong_uvomega_small_soap_sf_5000.toml \
  --device mps
```

或只跑訓練（不自動 eval）：

```bash
uv run python scripts/train_deeponet_cfc.py \
  --config configs/deeponet_cfc_midlong_uvomega_small_soap_sf_5000.toml \
  --device mps
```

## Evaluation

```bash
uv run python scripts/evaluate_deeponet_cfc.py \
  --config configs/deeponet_cfc_midlong_uvomega_small_soap_sf_5000.toml \
  --checkpoint artifacts/deeponet-cfc-re1000-soap-sf-5000/lnn_kolmogorov_final.pt \
  --output-dir artifacts/deeponet-cfc-re1000-soap-sf-5000-eval
```

輸出：

- `field_comparison_tXX.png`
- `vorticity_comparison_tXX.png`
- `energy_spectrum.png`
- `kinetic_energy_vs_time.png`
- `enstrophy_vs_time.png`
- `uv_error_vs_time.png`
- `summary.json`

## 主要 Config 說明

| Config | 用途 |
|---|---|
| `deeponet_cfc_smoke_uvomega.toml` | 快速 smoke test（少步數） |
| `deeponet_cfc_midlong_uvomega_small_soap_sf_5000.toml` | **Re=1000 主線基準（EXP-030）** |
| `deeponet_cfc_re10000_wide_v2.toml` | Re=10000 基準（EXP-031, d=128） |
| `deeponet_cfc_re10000_xlarge.toml` | Re=10000 最佳（EXP-033, d=256） |
| `deeponet_cfc_re10000_transfer_wide.toml` | Re=10000 transfer from Re=1000（EXP-042） |

## 專案結構

```text
configs/
  deeponet_cfc_smoke_uvomega.toml
  deeponet_cfc_midlong_uvomega_small_soap_sf_5000.toml   # Re=1000 主線
  deeponet_cfc_re10000_wide_v2.toml                      # Re=10000 基準
  deeponet_cfc_re10000_xlarge.toml                       # Re=10000 最佳
  deeponet_cfc_re10000_transfer_wide.toml                # transfer learning

docs/
  experiment_log.md     # 實驗 state（主要歷史紀錄）
  lnn_architecture.html

scripts/
  train_deeponet_cfc.py
  evaluate_deeponet_cfc.py
  run_experiment.py     # train + eval 一體化 pipeline
  compare_experiments.py

src/pi_onet/
  kolmogorov_dataset.py
  lnn_kolmogorov.py
```

## 核心設計原則

- `u, v` 觀測作為唯一 data supervision；`omega`, KE 等只作診斷
- `LearnableFourierEmb(embed_dim=128)` 取代確定性諧波；Re=1000 仍用 `periodic_fourier_encode`
- `relpos_bias` 使用純距離輸入：保持等向性，避免感測器 x 非均勻分佈注入偏差
- `temporal_anchor` 提供絕對時間座標，改善 forcing mode 重建
- `time_marching` 改善 causal branch token 學習
- physics loss（momentum + continuity）以 GradNorm 自動均衡 task 梯度比例（Re=10000）
- sensor 位置額外施加 continuity 約束（use_sensor_physics=true）
- 優化器主線：`SOAP + Schedule-Free`（Re=10000: lr=1e-3, betas=(0.9,0.999), precond_freq=2）

詳細實驗歷史與決策依據見 [`docs/experiment_log.md`](docs/experiment_log.md)。
