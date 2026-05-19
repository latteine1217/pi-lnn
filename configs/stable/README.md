# configs/stable/ — Stable Phase Config Aliases

本目錄為 **2026-05-19 起 stable phase** 使用的 config 別名（symlink）入口。

## 為什麼存在

`docs/experiment_log_v2.md` 將 stable phase 主線實驗重新編號為 `EXP-200` 起。為了讓 agent 與使用者直接以新 ID 跑訓練（不必每次查 legacy 對照表），本目錄為每個 stable ID 建立 symlink → legacy config。

## 設計原則

- **零 duplication**: 全為 symlink，不複製 config 內容；legacy config 更動會自動反映
- **零 artifact drift**: 訓練後 artifact 仍存在 legacy 路徑（`artifacts_dir` 設定於 legacy config 內）；不更動既有實驗 artifact
- **單向 alias**: 只支援 stable ID → legacy；legacy ID 仍以 `configs/exp_NNN_*.toml` 為 source of truth
- **完整對照表**: 雙向對照見 [`docs/experiment_log_v2.md`](../../docs/experiment_log_v2.md) `[INDEX] Legacy ↔ Stable ID 雙向對照`

## 使用方式

```bash
# 用 stable ID 跑訓練
uv run python lnn_kolmogorov.py --config configs/stable/exp_200_a.toml

# 等價於跑 legacy ID
uv run python lnn_kolmogorov.py --config configs/exp_080_re10000_al_4task_rho01.toml
```

## 編號規則

| 範圍 | 群組 |
|---|---|
| `exp_200_a` ~ `exp_200_e` | B3 multi-seed (5 seeds, AL ρ=0.1, 主線) |
| `exp_201_a` ~ `exp_201_e` | B0 multi-seed (5 seeds, vanilla DeepONet) |
| `exp_202` | B1 single (CfC, no cross-attn) |
| `exp_203` | B2 single (cross-attn, no CfC) |
| `exp_204` | Standard PINN (SiLU) |
| `exp_205` | Standard PINN (tanh) |
| `exp_220` ~ `exp_225` | Sensor placement ablation (B3, seed=2, axis-fix v2) |
| `exp_230` | Re=1000 baseline |
| `exp_240+` | 預留給後續 stable phase 新實驗 |

## 新增規則

新 stable phase config 加入流程：

1. 在 `configs/` 建立原始 config（檔名: `exp_NNN_描述.toml` 或更直接用 `exp_NNN.toml`）
2. 在 `configs/stable/` 建立 symlink: `ln -sf ../exp_NNN_描述.toml exp_NNN[_X].toml`
3. 更新 `docs/experiment_log_v2.md` `[INDEX] Legacy ↔ Stable ID 雙向對照`

## 注意

- **不要** 直接編輯 stable/ 內的 symlink 目標；改 legacy config 即可
- **不要** 將 stable/ 內 config 用於 `artifacts_dir` 路徑（仍應走 legacy path 維持 reproducibility）
- 若 legacy config 需要重命名或棄用，先確認 stable/ 內無 symlink 指向它
