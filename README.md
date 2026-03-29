# pi-o-net

以 **DeepONet + CfC** 實作的 sparse-data 物理資訊模型，目標是用少量感測器時序資料重建 2D Kolmogorov flow 場。

目前正式主線已切到新資料格式：

- **觀測通道：`u, v, omega`**
- **空間 domain：`[0, 1]^2` periodic**
- **薄 token + token self-attention**
- **temporal CfC encoder**
- **DeepONet-style query decoder with cross-attention**
- **訓練只依賴真實觀測與 PDE**

也就是：`vorticity / spectrum / kinetic energy / enstrophy` 全部保留作診斷工具，不進 training supervision。

## 目前架構

資料流如下：

```text
sensor_obs[t, k, {u,v,omega}] + sensor_pos[k, {x,y}]
    -> RFF(x,y) + token_in
    -> thin sensor tokens
    -> token self-attention
    -> temporal CfC over time
    -> causal branch tokens h_states[t, k, d]
    -> query trunk (x, y, t, component)
    -> cross-attention(query, branch tokens)
    -> branch/trunk fusion
    -> latent field {u, v, p}
```

補充：

- 真實感測器不提供 `p`
- `L_data` 只監督真實觀測通道
- 若觀測包含 `omega`，則由模型輸出的 `u, v` 導數現算
- `p` 只作模型內部 latent 場量，交由 PDE residual 約束

對應實作在：

- [`src/pi_onet/lnn_kolmogorov.py`](src/pi_onet/lnn_kolmogorov.py)
- [`src/pi_onet/kolmogorov_dataset.py`](src/pi_onet/kolmogorov_dataset.py)

互動式架構文件在：

- [`docs/lnn_architecture.html`](docs/lnn_architecture.html)

## Sparse-data 訓練原則

主線只保留這些 loss：

- `L_data`: 真實量到的資料點值誤差
- `L_momentum`: Navier-Stokes momentum residual
- `L_continuity`: incompressibility residual

保留但只作診斷、不作訓練 supervision 的量：

- vorticity
- energy spectrum
- kinetic energy
- enstrophy

這是刻意的設計選擇。若場景只有少量感測器，這些統計量或 dense derivative target 通常不是可直接取得的真實 supervision。

## 目前建議配置

目前 `uvomega` 主線基準是：

- `observed channels = u, v, omega`
- `domain_length = 1.0`
- `rff_sigma = 32.0`
- `use_local_struct_features = false`
- `num_token_attention_layers = 1`
- `physics_loss_weight = 0.01`
- `time_marching = true`
- `device = "mps"`

參考 config：

- [`configs/deeponet_cfc_smoke_uvomega.toml`](configs/deeponet_cfc_smoke_uvomega.toml)
- [`configs/deeponet_cfc_midlong_uvomega.toml`](configs/deeponet_cfc_midlong_uvomega.toml)

舊的 `deeponet_cfc_smoke.toml`、`deeponet_cfc_midlong_sigma4_tokenattn.toml` 仍留在 repo，主要用於過去實驗參考，不再是目前正式基線。

## 安裝

```bash
uv sync
```

如果只想確認 Python 檔案可被解譯：

```bash
uv run python -m py_compile \
  src/pi_onet/lnn_kolmogorov.py \
  src/pi_onet/kolmogorov_dataset.py \
  scripts/train_deeponet_cfc.py \
  scripts/evaluate_deeponet_cfc.py
```

## 訓練

Smoke：

```bash
uv run python scripts/train_deeponet_cfc.py \
  --config configs/deeponet_cfc_smoke_uvomega.toml
```

中長訓練：

```bash
uv run python scripts/train_deeponet_cfc.py \
  --config configs/deeponet_cfc_midlong_uvomega.toml
```

## Evaluation / 診斷

```bash
uv run python scripts/evaluate_deeponet_cfc.py \
  --config configs/deeponet_cfc_midlong_uvomega.toml \
  --checkpoint artifacts/deeponet-cfc-midlong-uvomega/lnn_kolmogorov_final.pt \
  --output-dir artifacts/deeponet-cfc-eval-midlong-uvomega
```

目前 evaluation 會輸出：

- `field_comparison_tXX.png`
- `vorticity_comparison_tXX.png`
- `energy_spectrum.png`
- `kinetic_energy_vs_time.png`
- `enstrophy_vs_time.png`
- `summary.json`

## 專案結構

```text
configs/
  deeponet_cfc_smoke_uvomega.toml
  deeponet_cfc_midlong_uvomega.toml
  deeponet_cfc_smoke.toml
  deeponet_cfc_midlong_sigma4_tokenattn.toml
  deeponet_cfc_long.toml
  lnn_kolmogorov.toml
  lnn_kolmogorov_quick.toml

docs/
  lnn_architecture.html

scripts/
  train_deeponet_cfc.py
  evaluate_deeponet_cfc.py

src/pi_onet/
  kolmogorov_dataset.py
  lnn_kolmogorov.py
  pit_ldc.py
```

## 現行 uvomega 基線

目前已重建出一套乾淨可跑的 `uvomega` 基線：

- smoke artifact:
  [`artifacts/deeponet-cfc-smoke-uvomega`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-smoke-uvomega)
- midlong artifact:
  [`artifacts/deeponet-cfc-midlong-uvomega`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-midlong-uvomega)
- checkpoint sweep:
  [`artifacts/deeponet-cfc-eval-midlong-uvomega-sweep`](/Users/latteine/Documents/coding/pi-lnn/artifacts/deeponet-cfc-eval-midlong-uvomega-sweep)

目前觀察：

- 這條主線已可穩定訓練，不再是資料格式不相容狀態
- `step_250 ~ 1000` 沒有掉回 near-zero collapse
- 但後期仍會往低能量解收縮，`E(k_f=4)` 幾乎為 0
- 因此它目前是「乾淨可重跑的基線」，不是「已成功收斂的模型」

## 目前研究結論

- 單一 global hidden state 容易 collapse
- 保留 sensor-level local identity 比提早 pooling 更合理
- token self-attention 能改善訓練穩定性
- `u, v, omega` 觀測需要 per-channel normalization，否則 `omega` 會主導 `L_data`
- sparse-data 主線不應依賴 dense derivative / spectrum supervision

因此目前 repo 已收斂到「**先把 sparse-data 主線做乾淨，再談更高階結構對齊**」的方向。
