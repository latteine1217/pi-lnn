# Thesis Figure Regeneration Workflow

舊 thesis figure（field/vorticity/spectrum/band_err/kf_amp/kf_phase + KE/div/uv-error trajectory）的 PNG mtime 是 2026-05-14，**對應 single-seed 10k iter 之前的 visualisation**，比論文主表 EXP-245 multi-seed 20k 數據早。

三個補救步驟（A → B → C 依嚴謹度遞增）：

| 步驟 | 做什麼 | 結果 |
|---|---|---|
| **(A)** Caption fix | 已完成於 `thesis/main.tex`：6 個舊 figure 與 1 個新 trajectory triple-panel 全部加註「single-seed visualisation (seed 42); multi-seed metric in Table 4.1」 | 圖檔不變，學術合規 |
| **(B)** Single-seed refresh | 用 EXP-245_a (seed=42, 20k iter) checkpoint 跑 evaluator 重生 single-seed PNG | 圖檔對齊 20k baseline |
| **(C)** Multi-seed envelope | 5 個 seed 跑 evaluator → 收集 npz → 畫 mean ± 1σ envelope 圖 | 圖直接呈現 multi-seed 統計 |

## (B) Single-seed refresh — `regenerate_thesis_figures_single_seed.sh`

### 前置：rsync EXP-245_a checkpoint

```bash
mkdir -p artifacts/_lab_rsync/deeponet-cfc-re10000-exp245-b3-les-T50/seed42
rsync -avzP <user>@<lab-gpu-host>:/path/to/exp_245_seed42/step_20000.pt \
            artifacts/_lab_rsync/deeponet-cfc-re10000-exp245-b3-les-T50/seed42/
```

### 重生 figure

```bash
bash scripts/regenerate_thesis_figures_single_seed.sh
# 或指定 ckpt 路徑：
bash scripts/regenerate_thesis_figures_single_seed.sh path/to/step_20000.pt
```

執行後 `thesis/figures/results/` 內的所有 PNG 會被 EXP-245_a 的 visualisation 覆蓋。完成後重新編譯：

```bash
cd thesis && pdflatex main && pdflatex main
```

## (C) Multi-seed envelope — `plot_multiseed_envelope.py`

### 步驟 1: Patch evaluator 加 `--export_arrays` flag

在 `scripts/evaluate_deeponet_cfc.py` 接近 `series_map` 完成計算的位置（約 line 690 之後、第一個 `plot_metric_series` 呼叫之前）加：

```python
if args.export_arrays:
    np.savez(
        output_dir / "series.npz",
        time=t_vals,
        KE=ke_series,
        KE_dns=ke_dns,
        div_ratio=div_ratio_series,
        u_rel_L2=u_rel_L2_series,
        v_rel_L2=v_rel_L2_series,
        kf_amp=kf_amp_series,
        kf_amp_dns=kf_amp_dns_series,
        kf_phase=kf_phase_series,
        kf_phase_dns=kf_phase_dns_series,
    )
```

並在 argparse block 加：

```python
parser.add_argument("--export_arrays", action="store_true",
                    help="Dump series.npz for multi-seed aggregation.")
```

### 步驟 2: 跑 5 個 seed 的 evaluator

```bash
for SEED_TAG in a b c d e; do
    SUFFIX=$( [ "$SEED_TAG" = "a" ] && echo "" || echo "_$SEED_TAG" )
    CFG="configs/stable/exp_245${SUFFIX}.toml"
    CKPT="artifacts/_lab_rsync/.../exp_245_seed${SEED_TAG}/step_20000.pt"
    OUT="artifacts/eval_245_seed${SEED_TAG}"

    PYTORCH_ENABLE_MPS_FALLBACK=1 \
        uv run python scripts/evaluate_deeponet_cfc.py \
            --config "$CFG" --checkpoint "$CKPT" \
            --output_dir "$OUT" --device mps --export_arrays
done
```

### 步驟 3: 合併與繪製 envelope

```bash
uv run python scripts/plot_multiseed_envelope.py \
    --seed_dirs artifacts/eval_245_seed{a,b,c,d,e} \
    --output_dir thesis/figures/results
```

執行後 5 個 trajectory figure（KE / div / uv-error / kf_amp / kf_phase）會被 mean ± 1σ envelope 版本覆蓋。Snapshot 圖（field/vorticity/spectrum）仍是 (B) 的 single-seed visualisation——multi-seed mean snapshot 在物理上不合理。

### 步驟 4: 更新 caption 標明 envelope

完成 (C) 後，把 main.tex 中相關 figure 的 caption 從「single-seed visualisation (seed 42)」改為「mean ± 1σ envelope over n=5 seeds」。我可以協助這部分 Edit。

## 完整 deliverable 對應

| 檔案 | 步驟 | 狀態 |
|---|---|---|
| `thesis/main.tex` (6 figure caption + fig:main_trajectories) | (A) | ✅ 已 patch |
| `scripts/regenerate_thesis_figures_single_seed.sh` | (B) wrapper | ✅ 已建立 |
| `scripts/plot_multiseed_envelope.py` | (C) plotter | ✅ 已建立 |
| `scripts/evaluate_deeponet_cfc.py` patch (`--export_arrays`) | (C) prerequisite | ⏳ 需手動加 |
| Multi-seed PNG 覆蓋 | (C) execution | ⏳ 需 rsync ckpt + 跑 5 seed |
