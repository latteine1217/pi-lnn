# pi-o-net

以 **DeepONet + CfC** 實作的 sparse-data 物理資訊模型，目標是用少量感測器時序資料重建 2D Kolmogorov flow 場。

目前主線不是舊的 PiT LDC，而是：

- **thin sensor tokens**
- **token self-attention**
- **temporal CfC encoder**
- **DeepONet-style query decoder with cross-attention**
- **sparse-data training rule**

也就是：訓練只依賴真實觀測與 PDE，本身不把 vorticity / spectrum / energy / enstrophy 當 supervision。

## 目前架構

資料流如下：

```text
sensor_vals[t, k, {u,v,p}] + sensor_pos[k, {x,y}]
    -> RFF(x,y) + token_in
    -> thin sensor tokens
    -> token self-attention
    -> temporal CfC over time
    -> causal branch tokens h_states[t, k, d]
    -> query trunk (x, y, t, component)
    -> cross-attention(query, branch tokens)
    -> branch/trunk fusion
    -> u / v / p
```

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

這是刻意的設計選擇。若場景只有少量感測器，這些統計量或導數量通常不是可直接取得的真實 supervision。

## 目前建議配置

目前主線基準是：

- `rff_sigma = 4.0`
- `use_local_struct_features = false`
- `num_token_attention_layers = 1`
- `physics_loss_weight = 0.01`
- `physics_loss_warmup_steps = 500`
- `physics_loss_ramp_steps = 1500`
- `time_marching = true`

參考 config：

- [`configs/deeponet_cfc_smoke.toml`](configs/deeponet_cfc_smoke.toml)
- [`configs/deeponet_cfc_midlong_sigma4_tokenattn.toml`](configs/deeponet_cfc_midlong_sigma4_tokenattn.toml)
- [`configs/deeponet_cfc_long.toml`](configs/deeponet_cfc_long.toml)

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
  --config configs/deeponet_cfc_smoke.toml
```

中長訓練：

```bash
uv run python scripts/train_deeponet_cfc.py \
  --config configs/deeponet_cfc_midlong_sigma4_tokenattn.toml
```

## Evaluation / 診斷

```bash
uv run python scripts/evaluate_deeponet_cfc.py \
  --config configs/deeponet_cfc_midlong_sigma4_tokenattn.toml \
  --checkpoint artifacts/deeponet-cfc-midlong-sigma4-tokenattn/lnn_kolmogorov_final.pt \
  --output-dir artifacts/deeponet-cfc-eval-midlong-sigma4-tokenattn
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

## 現況

目前最重要的研究結論是：

- 單一 global hidden state 容易 collapse
- 保留 sensor-level local identity 比提早 pooling 更合理
- token self-attention 能改善訓練穩定性
- sparse-data 主線不應依賴 dense derivative / spectrum supervision

因此目前 repo 已收斂到「**先把 sparse-data 主線做乾淨，再談更高階結構對齊**」的方向。
