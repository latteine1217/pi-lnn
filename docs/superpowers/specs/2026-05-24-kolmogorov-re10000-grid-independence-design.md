# Design — Kolmogorov Re=10⁴ DNS Grid Independence Validation

| Field | Value |
|---|---|
| Date | 2026-05-24 |
| Status | Draft（user sign-off pending）|
| Owner | latteine |
| Related state | [docs/experiment_log_v2.md](../../experiment_log_v2.md) |
| Related skill | superpowers:brainstorming → writing-plans (next) |
| Reference | Gemini grid-independence test guidance（user prompt 附）|

---

## 1. Goal & Scope

**Goal**: 對 `data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy`（EXP-200~268 全線訓練用 DNS baseline）做 **post-hoc grid independence validation**，產出可寫入論文 §Methods 的證據（log-log convergence + 統計量收斂表）。

**Background**: EXP-200~268 全部訓練資料以此 N=256 DNS 為 ground truth supervision 與 offline benchmark。當前 §Methods **沒有** N=256 grid convergence 的直接證據，reviewer 可合理質疑：
- N=256 是否已解析到 Re=10⁴ Kolmogorov 的 dissipation cutoff？
- 訓練模型學到的物理會不會是 N=256 特定的數值耗散特性？

**Methodology choice**:
- 採 Gemini 建議的「multi-resolution comparison against high-N reference」標準做法
- 對齊「同物理 IC」是必要前提（Gemini 警告的致命陷阱），既有 DNS generator IC 模式不滿足，必須擴增
- 多時刻評估（短 t 看 spectral convergence；長 t 看統計量 convergence，避開 chaos amplification）

**Success criteria**:
- 短 t（t=0.5）pointwise `rel_L2(N=256, ref=2048)` < 1 %
- 長 t（t=5）統計量 `KE(t)` 與 `E(k)` 在共同解析範圍內 N≥256 全部重合（rel diff < 2 %）
- 結論可寫入 §Methods 一句話

**Non-goals**:
- ❌ 不為 Re=10⁶ 做 grid sizing（後續另案）
- ❌ 不驗證 algorithmic order of accuracy（spectral method 對 turbulent IC 收斂率沒有 well-defined p）
- ❌ 不修改既有 N=256 baseline `.npy` 資料（新跑的 .npy 物理上是「不同 IC」，與 baseline 並存而非取代）
- ❌ 不引入新 solver / library

---

## 2. Constraints & Risk Drivers

### Constraint A — IC 同一性是 grid independence 的必要條件

Gemini 警告：「在 $64^2$ 的網格上呼叫隨機函數生成初始場，然後在 $128^2$ 的網格上又呼叫一次，這**不是**網格獨立性測試。」

既有 DNS generator (`~/pi-lnn-cfd-baseline/dns/generate_kolmogorov_dns_fp64.py`) 的 IC 路徑：

```python
# Line 424
field = amplitude * self.rng.standard_normal((self.N, self.N))
field_hat = np.fft.fft2(field) * spectral_filter
```

`rng.standard_normal((N, N))` 對 `seed=42` 在不同 N 給出**物理上不同**的隨機 sequence。後續 `_match_initial_low_k_spectrum` 只 rescale energy per shell，**phase 仍 N-dependent**。

**Implication**: 必須擴增 generator，加 spectral-space deterministic IC mode（per-mode seeded sub-RNG），保證 `hat(k, N1) ≡ hat(k, N2)` for `k ≤ k_cutoff`。

### Constraint B — 混沌系統的誤差競爭

對 Re=10⁴ Kolmogorov，估算：
- `u_rms ≈ 0.3-0.5`（baseline DNS 後段觀測值）
- `T_L = L / (u_rms · k_f) ≈ 1 / (0.4 · 2) ≈ 1.25`
- `T_end / T_L ≈ 5 / 1.25 ≈ 4` (≈ 4 Lyapunov times)

t=5 已接近 chaos-decoupled regime。截斷誤差積累 vs 混沌指數放大 (∝ e^{λt}) 在 t≥3 可能同量級。

**Implication**: 短 t pointwise L2 評估 + 長 t 統計量評估雙軌策略。

### Constraint C — 計算預算

home-gpu RTX 3090 fp64 ETDRK4 PyTorch CUDA 估時：

| N | 預估 wall | 累計 |
|---|---|---|
| 128 | ~3 min | 3 min |
| 256 | ~10 min | 13 min |
| 512 | ~40 min | 53 min |
| 1024 | ~3 hr | ~4 hr |
| 2048 | ~16-24 hr | ~20-28 hr |

**Risks**: N=2048 wall time / OOM / CFL marginal — 三項都需 dry-run 驗證後才能 commit 全跑。

---

## 3. Architecture — 5 個 Unit

```
┌─────────────────────────────────────────────────────────────────────┐
│ Unit A: Modified DNS generator (home-gpu)                           │
│   ~/pi-lnn-cfd-baseline/dns/generate_kolmogorov_dns_fp64.py         │
│   新增 --ic_mode spectral_seeded（mode-indexed SeedSequence IC）    │
│   保留 backward compat (--ic_mode band_limited_random 為 default)   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ rsync vendored copy
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Unit C: IC alignment self-check (pi-lnn local pytest)               │
│   tests/test_grid_independence_ic_alignment.py                      │
│   驗證 hat(N1) ≡ spectral_truncate(hat(N2)) for IC + post-alignment │
│   GATE: 必須 PASS 才能進 Unit B                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Unit B: GI orchestrator (home-gpu)                                  │
│   ~/gi_test_re10000/run.sh                                          │
│   Sequential N ∈ {128, 256, 512, 1024, 2048}, T=5, seed=42          │
│   先 dry-run N=2048 驗 wall/memory/CFL，再 commit 全跑               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ pi-lnn 端 pull rsync
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Unit D: GI analysis (pi-lnn local)                                  │
│   scripts/analyze_grid_independence.py                              │
│   多時刻 rel_L2 + E(k) overlay + KE/Ens/div 時間軌跡                │
│   輸出 JSON report + 6 figures（journal style）                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Unit E: Documentation (pi-lnn)                                      │
│   docs/grid_independence_re10000.md（新檔）                         │
│   docs/experiment_log_v2.md 加 [STATE] Grid Independence Validation │
└─────────────────────────────────────────────────────────────────────┘
```

**Data flow**: home-gpu DNS runs → `~/gi_test_re10000/*.npy` → pi-lnn `data/dns/gi_test_re10000/` → analysis → figures + JSON → docs

**Generator source-of-truth 決定**: home-gpu `pi-lnn-cfd-baseline` 為 single source；pi-lnn 在 `tools/dns_generator/` 保 **vendored copy**（header 標 `# AUTO-VENDORED FROM home-gpu, DO NOT EDIT LOCALLY`），供 pytest 本地 import 驗證。

---

## 4. Unit A — Modified DNS generator (IC 擴增)

### 4.1 新方法 `_make_spectral_ic`

```
[function]   _make_spectral_ic(seed, k_cutoff) -> (hat_u, hat_v) in (N, N) complex128

[behavior]   對每個 integer Fourier mode (kx_int, ky_int) 在 k_mag ≤ k_cutoff 圓盤內：
             1. 用 SeedSequence(seed, spawn_key=(kx_int + 10000, ky_int + 10000)) 派生 sub-RNG
             2. 從 sub-RNG 抽 4 個 standard_normal → (re_u, im_u, re_v, im_v)
             3. 套上跟舊版同形 spectral filter: exp(-(k_mag / k_cutoff)^8)
             4. 填到 hat_u[kx_int % N, ky_int % N] 與 hat_v[同位置]
             5. Hermitian conjugate 填到 hat_u[(-kx_int) % N, (-ky_int) % N]
             6. (0, 0) mode 強制 0（zero-mean enforcement）
             7. Nyquist self-dual mode 特殊處理（不重複填）
             8. 經 _project_hat 強制 ∇·u = 0

[invariant]  對任意 N1, N2 ≥ 2·⌈k_cutoff⌉ + 1：
             hat_u(k, N1) ≡ spectral_truncate(hat_u(k, N2)) for k ≤ k_cutoff
             差為 0（bit-exact，不是 ε）

[後續]       仍走原本 _align_initial_statistics 的 5 次 KE/forcing-coeff matching。
             對 zero-padded high-k modes alignment 是 no-op，不破壞 N-invariance。
```

### 4.2 CLI flag

```
--ic_mode {band_limited_random, spectral_seeded}   # default: band_limited_random
--ic_k_cutoff 8.0                                   # default: max(8.0, 4*k_f)
```

`band_limited_random` 為現行行為 default → **backward compat 完整保留**。

### 4.3 修改 footprint

| 位置 | 修改 | 行數 |
|---|---|---|
| `__init__` | 接 `ic_mode`, `ic_k_cutoff`；分支選 IC 路徑 | +15 |
| 新方法 `_make_spectral_ic` | 上述合約實作 | ~70 |
| `_align_initial_statistics` | 加 `skip_low_k_match` 旗標供 spectral mode 可選跳過 | +5 |
| `main()` CLI | 加兩個 flag | +6 |
| Config dict in `run` 輸出 | 記錄 `ic_mode`, `ic_k_cutoff` | +2 |

合計 ~98 行新增，**0 行刪除**。

### 4.4 Acceptance（home-gpu 上 unit test）

1. `ic_mode=band_limited_random + seed=42 + N=256` 跑出來的 t=0 frame 與既有 baseline `.npy` 的 t=0 pointwise diff < 1e-12（保 backward compat）
2. `ic_mode=spectral_seeded + seed=42 + N ∈ {128, 256, 512, 1024}` 跑完 alignment 後：spectral truncate(IC_N1024) → N=128/256/512 與直接生 IC_N=128/256/512 pointwise diff < 1e-13（保 N-invariance）

---

## 5. Unit B — GI orchestrator

### 5.1 檔案 `~/gi_test_re10000/run.sh`（執行於 home-gpu）

**結構規格**:
```
set -euo pipefail
OUT_DIR=~/gi_test_re10000
mkdir -p $OUT_DIR

COMMON_FLAGS=(
  --L 1.0 --nu 1e-4 --A 0.1 --k_f 2
  --dt 2.5e-4 --T_end 5.0 --save_interval 100
  --integrator etdrk4 --dealias --dealias_mode 2/3
  --ic_mode spectral_seeded --ic_k_cutoff 8.0
  --seed 42
)

for N in 128 256 512 1024 2048; do
  out="$OUT_DIR/kolmogorov_dns_fp64_etdrk4_Re10000_N${N}_T5_dt2p5e4_si100_seed42_icspectral.npy"
  uv run python ~/pi-lnn-cfd-baseline/dns/generate_kolmogorov_dns_fp64.py \
      --N "$N" "${COMMON_FLAGS[@]}" --output_file "$out" \
      2>&1 | tee "$OUT_DIR/N${N}.log"
done
```

**Naming convention**: `_icspectral` 後綴與既有 baseline `_ds4` 區隔；兩者物理上是**不同 IC**，不能互相替換。

### 5.2 N=2048 pre-flight dry-run

執行 `run.sh` **之前**先單獨跑：
```
N=2048 --T_end 0.05  (約 200 steps)
```
- 量真實 wall time → 推估 full T=5 wall（若 > 30 hr 則改 N=1536 或放棄 second-ref）
- 量 PyTorch CUDA memory peak → 若接近 24 GB 上限，啟動 N=2048 失敗 fallback
- 量 `max|u| + |v|` 推算 CFL → 若 CFL > 0.5，全部 N 統一改 `dt=1e-4` 重跑（保持 dt-invariance）

### 5.3 Pull/sync 流程

pi-lnn 端 `scripts/gi_test/pull_results.sh`:
```
rsync -avzP home-gpu:~/gi_test_re10000/ data/dns/gi_test_re10000/
```
- 手動觸發（不用 cron，job ~28 hr 不值得自動化）
- home-gpu side 用 `nohup` 或 tmux detached 跑 `run.sh`

---

## 6. Unit C — IC alignment self-check (gating test)

### 6.1 檔案 `tests/test_grid_independence_ic_alignment.py`

**前置**: pi-lnn `tools/dns_generator/generate_kolmogorov_dns_fp64.py` (vendored from home-gpu) 必須存在。

### 6.2 Test cases

```
test_1_spectral_ic_n_invariance (parametrized N_pair ∈ [(128,256), (128,512),
                                                          (128,1024), (256,1024),
                                                          (512,1024)])
    Given: ic_mode=spectral_seeded + seed=42 + ic_k_cutoff=8.0
    When:  build IC at N_small and N_large independently (numpy backend)
    Then:  spectral_truncate(hat_u(N_large)) → N_small ≡ hat_u(N_small)
           pairwise complex pointwise diff < 1e-13
           （同樣比 hat_v）

test_2_backward_compat_band_limited_random
    Given: ic_mode=band_limited_random + seed=42 + N=256
           （= 既有 baseline 用的 IC mode）
    When:  build IC + 跑完整 _align_initial_statistics
    Then:  與既有 baseline .npy 的 t=0 frame (u, v) pointwise diff < 1e-12
           → 確保 ic_mode default 行為與既有 EXP-200~268 訓練資料 byte-aligned

test_3_alignment_invariant
    Given: spectral_seeded IC 經過 5 次 _align_initial_statistics 後
    When:  比較 N=256 alignment 結果 vs N=1024 alignment 結果再 spectral truncate 到 N=256
    Then:  pointwise diff < 1e-12
           → 驗證 alignment 不破壞 N-invariance（最關鍵 invariant）
```

### 6.3 Gating

`test_1` + `test_2` + `test_3` 全部 PASS → 才可以執行 Unit B（不然 GI test 物理基礎不成立，浪費 28 hr GPU time）。

---

## 7. Unit D — GI analysis script

### 7.1 檔案 `scripts/analyze_grid_independence.py` (~400 行)

### 7.2 Pipeline

```
Input:
  data/dns/gi_test_re10000/kolmogorov_dns_fp64_etdrk4_Re10000_N{N}_T5_*_icspectral.npy
  for N ∈ {128, 256, 512, 1024, 2048}

Steps:
  1. 載入全部 .npy，校驗 config (seed, dt, ic_mode, ic_k_cutoff, k_f, nu, A) 一致
     若不一致 → fail-fast，raise ValueError
  2. 對每個 N，spectral_interpolate(u, v) 到 ref_N (=2048)
     - 用 zero-padding in spectral 空間 → inverse FFT
     - 因為 IC 已保證 k > k_cutoff=8 部分 = 0，且解析過程 modes 不會憑空增生超出該 N 的 dealias 範圍，
       interpolation 在 spectral 意義是 lossless
  3. 對每個 t ∈ {0.5, 1.0, 2.0, 5.0}：
     - Frame index 直接計算：t / (dt * save_interval) = t / 0.025
       → t=0.5 → frame 20，t=1.0 → 40，t=2.0 → 80，t=5.0 → 200（全為整數）
     - n_snapshots = T_end/dt/si + 1 = 20000/100 + 1 = 201 frames total
     - 計算 rel_L2(u_N_interp, u_ref), rel_L2(v_N_interp, v_ref), rel_L2(ω_N_interp, ω_ref)
  4. KE(t), Enstrophy(t), max|∇·u|(t) 對所有 N 用 native grid 計算
  5. E(k) at t=5 對所有 N 用 native grid 計算 + 共同 k_bin 對齊
  6. Order of convergence: log-log linregress(log N, log rel_L2) per t per field
  7. 寫 JSON + 6 figures
```

### 7.3 Output

```
data/dns/gi_test_re10000/gi_analysis_report.json
docs/figures/grid_independence/
  01_rel_L2_vs_N_loglog.png       # 主結論：rel_L2(u, v, ω) vs N for each t, log-log
  02_spectrum_E(k)_at_t5.png       # E(k) overlay 全部 N at t=5
  03_KE_time_series.png            # KE(t) overlay 全部 N
  04_enstrophy_time_series.png     # Ens(t) overlay 全部 N
  05_divergence_time_series.png    # max|∇·u|(t) overlay (semi-log y)
  06_spectrum_E(k)_at_t0.png       # IC sanity check（證明所有 N IC 對齊）
```

### 7.4 JSON schema

```json
{
  "config": {
    "ref_N": 2048,
    "test_N_list": [128, 256, 512, 1024],
    "times_evaluated": [0.5, 1.0, 2.0, 5.0],
    "common_dns_params": {"dt": 2.5e-4, "nu": 1e-4, "k_f": 2, "A": 0.1, ...}
  },
  "metrics": {
    "rel_L2_u": {"t=0.5": {"N=128": float, ...}, ...},
    "rel_L2_v": {...},
    "rel_L2_omega": {...},
    "order_of_convergence_slope": {"t=0.5": {"u": slope, ...}, ...},
    "KE_t_series": {"N=128": {"t": [...], "KE": [...]}, ...},
    "Enstrophy_t_series": {...},
    "max_div_t_series": {...},
    "spectrum_E_k_at_t5": {"N=128": {"k": [...], "E": [...]}, ...},
    "spectrum_slope_late": {"N=128": float, ...}
  },
  "verdict": {
    "short_t_pointwise": "PASS|WARN|FAIL",
    "long_t_statistical": "PASS|WARN|FAIL",
    "incompressibility": "PASS|WARN|FAIL",
    "overall": "PASS|WARN|FAIL",
    "summary_sentence_for_paper": "..."
  }
}
```

### 7.5 Acceptance criteria（寫進 verdict）

| 指標 | PASS | WARN | FAIL |
|---|---|---|---|
| `rel_L2(N=256, ref=2048) at t=0.5` | < 1 % | 1-5 % | > 5 % |
| `rel_L2(N=256, ref=2048) at t=2` | < 10 % | 10-25 % | > 25 % |
| `rel_L2(N=256, ref=2048) at t=5` | < 30 % | 30-60 % | > 60 % |
| `rel_L2(N=1024, ref=2048) at t=5` | < 50 % of `rel_L2(N=256)` | 50-80 % | > 80 % |
| `KE_N256(t) vs KE_N2048(t)` 全 t 範圍 max rel diff | < 2 % | 2-5 % | > 5 % |
| `E(k)_N256 at k ≤ 32` 對 ref rel diff | < 5 % | 5-15 % | > 15 % |
| `max\|∇·u\|_N256` for t ∈ [0, 5] | < 1e-10 | 1e-10 to 1e-6 | > 1e-6 |

**Overall verdict 邏輯**:
- 全 PASS → "N=256 grid-converged for paper §Methods"
- 短 t PASS + 中/長 t WARN + 統計量 PASS → "N=256 spectrally converged at short t, statistically converged at long t (chaos-affected pointwise)"
- 任一 FAIL → 不寫進 paper，需 investigation

### 7.6 Plot style（per `memory/feedback_journal_plot_style.md`）

- NeurIPS/ICLR style
- 字型 sans-serif (Arial)，size ≥ 9pt
- DPI ≥ 300 for final
- 避免方塊 marker，用 `o`/`s`/`^`/`D` 區分 series
- spines 只保留 left + bottom
- legend frameless
- log-log plot 用 minor ticks

Helper `setup_plot_style()` 統一 apply。

---

## 8. Unit E — Documentation

### 8.1 新檔 `docs/grid_independence_re10000.md` (~150 行)

結構：
```
# Re=10^4 Kolmogorov Grid Independence Validation

## Why
（簡述 paper §Methods 需求）

## Setup
- Solver: ETDRK4 spectral fp64
- IC mode: spectral_seeded（mode-indexed SeedSequence for cross-N invariance）
- Grids: N ∈ {128, 256, 512, 1024, 2048}
- Reference: N=2048（with N=1024 second-ref）
- T_end=5, dt=2.5e-4, save_interval=100, seed=42

## Methodology
- IC alignment 不變式（含 reference 到 Unit C self-check）
- Multi-time evaluation rationale

## Results
[Figure 1: rel_L2 vs N log-log]
[Table 1: 主指標 PASS/WARN/FAIL]
[Figure 2: E(k) at t=5 overlay]
[Figure 3: KE(t) all N overlay]

## Discussion
- 短 t convergence rate 觀察
- 長 t statistical convergence
- N=256 baseline 選擇的 grid-independent 支持

## Caveats
- N > 256/3 ≈ 85 的 fine scales 解析力
- t=5 chaos transition 對 pointwise L2 的影響

## Reproducibility
[run hash, .npy paths, command lines]
```

### 8.2 修改 `docs/experiment_log_v2.md` 加 section

```markdown
## [STATE] Grid Independence Validation (Re=10^4 baseline data)

Date: 2026-05-XX
Status: PASS（或 WARN）
Report: docs/grid_independence_re10000.md
Data: data/dns/gi_test_re10000/
Verdict: N=256 grid-converged in {spectral, statistical} sense for T ≤ 5

| Metric | Value |
|---|---|
| rel_L2(N=256, ref=2048) @ t=0.5 | X.XX % |
| rel_L2(N=256, ref=2048) @ t=5 | XX.X % |
| KE rel diff (N=256 vs N=2048) max over t | X.XX % |
```

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| N=2048 wall time > 24 hr | 阻塞 second-ref 驗證 | Dry-run T=0.05 量速度；若超 → 改 N=1536 或放棄 second-ref，spec 標 caveat |
| N=2048 fp64 OOM (RTX 3090 24 GB) | N=2048 不能跑 | Dry-run 1 step 量 memory；若不行 → mixed-precision 或 N=1536 |
| dt=2.5e-4 對 N=2048 CFL marginal | 數值不穩定 | Dry-run 200 steps 量 max\|u\|+\|v\|；若 CFL > 0.5 → **全部 N 統一改 dt=1e-4 重跑**（保 dt-invariance） |
| `_align_initial_statistics` 在 N=2048 收斂慢 | IC 設定時間爆 | n_iter=5 應該 OK，加 timing log；超過 5 分鐘 → 進入 investigate |
| IC alignment 不保持 N-invariance | 整個 GI test 破功 | **Unit C test_3 必須 PASS 才繼續**（hard gate）|
| 既有 baseline IC mode 被誤改 | 訓練資料污染 | **Unit C test_2 必須 PASS**（hard gate）|
| 修改後 home-gpu generator 與 pi-lnn vendored copy 漂移 | pytest 假 PASS | rsync 流程在 `scripts/gi_test/sync_generator.sh` 寫死，每次 home-gpu 跑前先同步 |
| ref=2048 仍不 converged（unlikely but possible）| second-ref 證明失敗 | `rel_L2(N=1024, ref=2048)` 也計算，若 ≈ `rel_L2(N=512, ref=2048)` → ref 自己沒收斂，需 N=4096（成本爆，可能放棄並標 caveat）|

---

## 10. Overall Acceptance Criteria

整個 GI validation **PASS** 條件（可寫入論文 §Methods）：
1. Unit C 三個 test 全部 PASS
2. Unit D analysis JSON `verdict.overall == "PASS"` 或 `"WARN"`
3. 主結論句可寫入 §Methods：
   > "We verified grid convergence at N=256 by comparing against high-resolution references (N=1024, 2048) using a deterministic spectral-seeded initial condition. Pointwise relative L2 error of u at t=0.5 is below 1%, and statistical quantities (KE(t), spectrum) agree within 2% across all grids in t ∈ [0, 5]."

整個 GI validation **WARN** 條件（仍可寫進 paper 但需 caveat）：
- 長 t pointwise L2 > 30% 但統計量仍 < 5%
- 或 N=2048 跑不出來，只能用 N=1024 當 ref

整個 GI validation **FAIL** 條件（不寫進 paper，需 investigation）：
- 短 t pointwise L2 > 5%
- 或統計量在中 t 範圍偏差 > 10%
- 或 incompressibility 在任一 N 失守 (> 1e-6)

---

## 11. Implementation phases（粗排）

| Phase | 工作 | 估時 | Location |
|---|---|---|---|
| 1 | Unit A 改 generator + Unit C 寫 pytest + rsync vendored copy + pytest local PASS | 1-2 hr | home-gpu + pi-lnn local |
| 2 | Unit B 寫 orchestrator + N=2048 dry-run + 啟動 full sweep | 1 hr + ~28 hr GPU wait | home-gpu |
| 3 | Unit D 寫 + 跑分析 + figures + JSON | 2-3 hr | pi-lnn local |
| 4 | Unit E 寫 doc + experiment_log_v2 更新 | 1 hr | pi-lnn local |

**Critical path**: Phase 1 → Phase 2（gated by Unit C PASS）→ Phase 2 GPU wait → Phase 3 → Phase 4

---

## 12. References

- home-gpu: `~/pi-lnn-cfd-baseline/dns/generate_kolmogorov_dns_fp64.py`（DNS solver source-of-truth）
- home-gpu: `~/les-gen/`（LES generator，本 design 不涉及）
- pi-lnn: `docs/experiment_log_v2.md`（[STATE] section 將被新增）
- pi-lnn: `docs/paper_framing_draft.md`（§Methods 引用本 design 結果）
- pi-lnn: `scripts/validate_les_quality.py`（spectrum slope 計算 helper 來源）
- pi-lnn: `CLAUDE.md` § KNOWN_PITFALLS（dt / MPS / config 同步規範）
- Reference doc: Gemini grid independence test guidance（user 提供於 brainstorming 階段）
