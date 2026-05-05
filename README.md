# Pi-LNN

**Sparse-sensor physics-constrained neural operator for turbulent flow reconstruction.**

We reconstruct Re=10000 Kolmogorov flow from 100 velocity sensors without full-field supervision using a DeepONet-style query decoder, CfC temporal encoder, cross-attention over sensor tokens, and Navier–Stokes residual constraints.

> **Live demo page** → [latteine1217.github.io/pi-lnn/lnn_architecture.html](https://latteine1217.github.io/pi-lnn/lnn_architecture.html)
> **Full experiment history** → [`docs/experiment_log.md`](docs/experiment_log.md)

---

## Architecture

```
sensor_obs [T, K, {u,v}]  +  sensor_pos [K, {x,y}]
    ↓  LearnableFourierEmb + residual MLP
    →  sensor tokens [K, d]
    ↓  token self-attention  (2 layers)
    →  TemporalCfCEncoder    →  branch states  h [T, K, d]

query (x, y, t, c)
    ↓  LearnableFourierEmb + temporal anchor + dt_to_query
    →  trunk feature [N_q, d]
    ↓  causal cross-attention on h  (relpos bias, isotropic |r|)
    →  branch context [N_q, d]

branch basis ⊙ trunk basis  →  u / v / p   [N_q, 1]
```

- **Branch path:** sensor tokens are spatially encoded then evolved in continuous time by a CfC cell along the sensor sampling clock; output is per-token memory `h_k`.
- **Trunk path:** any query `(x, y, t, c)` is encoded by a learnable Fourier embedding plus an absolute-time anchor `(sin/cos 2πnt/T, n=1..2)`.
- **Causal readout:** `searchsorted(sensor_time, t_q)` selects the latest valid memory frame; cross-attention pools branch tokens, isotropic `|r|` relative bias removes directional artifacts.
- **Operator fusion:** dot-product between branch basis and trunk basis yields `u, v, p`.
- `p` is model-internal; constrained only by the PDE residual, never by data supervision.
- `ω`, KE, Enstrophy, E(k) are evaluation diagnostics — never enter training.
- **Training signal:** sensor MSE on `u, v` + NS-momentum residual + continuity, all GradNorm-balanced.

A full annotated walkthrough (with cards, time grid, decoder zoom-in) lives on the [live demo page](https://latteine1217.github.io/pi-lnn/lnn_architecture.html).

---

## Results

### Re=10000 — EXP-064 *(active baseline)*

| Metric | Value | Note |
|---|---|---|
| **KE rel-err** | **7.80%** | Re=10000 historical best |
| u rel-L2 | 17.0% | u_rmse 6.89e-2 |
| v rel-L2 | 20.2% | v_rmse 6.21e-2 |
| ω rel-L2 | 45.4% | k² high-freq amplification |
| Enstrophy rel-err | 29.1% | |
| Divergence L2 | 0.184 | DNS reference 0.092 |
| k_f amp ratio (t=5) | 0.962 | forcing mode amplitude |
| k_f phase err (t=5) | −0.023 rad | forcing mode phase |
| Band-low rel-err (k ≤ 8, t=5) | **3.62%** | 94.4% of total energy |
| Band-mid rel-err (k ~ 8..16, t=5) | ~100% | CS limit (see §Uncertainty) |
| Band-high rel-err (k > 16, t=5) | ~100% | CS limit |

`configs/exp_064_re10000_xlarge_sensor_physics.toml` · 10 000 steps · `d_model=256` · `LearnableFourierEmb (embed_dim=128, σ=2)` · GradNorm (4 tasks) · sensor-position physics

### Re=1000 — EXP-030 *(early validation)*

| Metric | Value |
|---|---|
| **KE rel-err** | **9.61%** |
| u RMSE | 5.68e-2 |
| k_f amp ratio | 1.027 |

`configs/exp_030_re1000_soap_sf_5k.toml` · 5 000 steps · `d_model=64` · SOAP + Schedule-Free

---

## Field / Vorticity Comparison

![Velocity field comparison at t equals 5, EXP-064](docs/assets/exp064/field_comparison_t5.png)

*Re=10000 velocity field at t = 5. Left: DNS reference. Middle: prediction. Right: error. Main coherent structures recovered; residual concentrates in high-shear regions.*

![Vorticity comparison at t equals 5, EXP-064](docs/assets/exp064/vorticity_comparison_t5.png)

*Vorticity ω = ∂v/∂x − ∂u/∂y at t = 5. Dominant vortices recovered; small-scale eddies are smoothed — this is the K=100 information bound, not a training failure (see §Uncertainty).*

---

## Energy Spectrum

![Radial energy spectrum, EXP-064](docs/assets/exp064/energy_spectrum.png)

*Radial energy spectrum E(k). Low-k band (k ≤ 8) carrying 94.4% of total energy is reproduced; mid/high-k content collapses by the CS bound. Spectrum slope deviation at large k is the irreducible reconstruction floor at K=100.*

For per-band diagnostics over the full time window, see [`docs/assets/exp064/band_energy_rel_error_vs_time.png`](docs/assets/exp064/band_energy_rel_error_vs_time.png).

---

## Filtering vs Smoothing

The reconstruction admits two estimator regimes, switched by `use_bidirectional_cfc` in the config:

| Mode | CfC scan | Decoder readout | Use case |
|---|---|---|---|
| **Filtering** *(default)* | forward only | causal `searchsorted(sensor_time, t_q)` | online / streaming sensors |
| **Smoothing** | forward + independent backward pass | causal lookup over bidirectional memory | offline batch reconstruction |

**Filtering — EXP-064 baseline.** A query at time `t_q` reads only sensor memory built from observations ≤ `t_q`. No future leakage. Engineering-deployable for real-time sensor streams.

**Smoothing — EXP-059, NEGATIVE_RESULT.** Adding a backward CfC pass lets `h(t=0)` see future observations. Mean KE marginally improved (20.6% → 19.1%, −1.5 pp) but `t=0` KE *worsened* (55.5% → 60.4%, +4.9 pp). The diagnosis: `t=0` reconstruction is bottlenecked by **insufficient training signal** (no IC weight), not by causal information asymmetry. Smoothing alone does not break the K=100 information bound.

Bottom line: filtering is the correct default; smoothing is implemented but offers no practical benefit at K=100.

---

## Uncertainty / Information Limit

EXP-064 sits at the **Compressed Sensing (CS) ceiling** for K=100 sensors. The ≈100% mid/high-frequency error is a mathematical consequence, not a training failure. Switching basis (Fourier ↔ wavelet) does not help — sparsity is equivalent across bases.

| Quantity | Value |
|---|---|
| Gini coefficient (u / v, db4 2-D wavelet) | **0.983 / 0.985** |
| Coefficients carrying 99% of energy | **~328** of N = 65 536 (top 0.5%) |
| CS recovery threshold M ≥ O(s · log N) | **~5 000 sensors** (≈50× short of K=100) |
| K=200 partial breakthrough (EXP-066) | band_mid 100% → **32.9%** |

| Band | Energy share | Wavelet DOF needed | Feasible at K=100? | EXP-064 band err (t=5) |
|---|---|---|---|---|
| Low (k ≤ 8) | 94.4% | ~196 | ✓ underdetermined | **3.62%** |
| Mid (k ~ 8..16) | 4.8% | ~588 | ✗ exceeds capacity | ~100% |
| High (k ~ 16..32) | 0.8% | ~1 452 | ✗ far exceeds | ~100% |

**Convergent evidence — three turbulence-aware ablations falsified at K=100:**

| Experiment | Mechanism | KE rel-err | Verdict |
|---|---|---|---|
| EXP-064 | Baseline (sensor physics + GradNorm) | **7.80%** | ★ Active baseline |
| EXP-067 | CfC τ ∈ (e⁻³, e¹) + freq-stratified σ=(1,4,12) | 11.20% | +3.4 pp |
| EXP-068 | PINN causal weighting (eps=1.0, 16 bins) | 9.73% | +1.9 pp / div ↑269% |
| EXP-069 | Three combined | 20.13% | negative interaction |

EXP-064 is the global optimum at the current K=100 + architecture configuration. Further progress requires *changing the problem*, not the optimizer / loss / initialization.

**Paths to break the limit:** K ≥ 5 000 sensors · K = 200+ with extended training (EXP-066) · 4D-Var time-series data assimilation · DNS-POD basis as high-frequency prior *(engineering non-transferable, research-only)*.

→ Full information-theoretic analysis: [Section 09 of the architecture page](https://latteine1217.github.io/pi-lnn/lnn_architecture.html). Detailed records: [`docs/experiment_log.md`](docs/experiment_log.md).

---

## How to Reproduce

```bash
uv sync
git submodule update --init --recursive
# Optional, Apple MPS only — explicit-CPU LinAlg patch (~30% faster than auto-fallback):
./scripts/apply_soap_patches.sh
```

> The SOAP submodule is third-party (`nikhilvyas/SOAP`). MPS-specific QR/eigh
> optimizations live under [`patches/`](patches/) and are applied locally by the
> script above (idempotent; supports `--check` and `--revert`). Re-run after any
> `git submodule update`.

**Train (Re=10000, EXP-064 active baseline):**
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1   # safety net for any MPS-unsupported ops
uv run python src/lnn_kolmogorov.py \
  --config configs/exp_064_re10000_xlarge_sensor_physics.toml \
  --device mps
```

**Evaluate the released checkpoint:**
```bash
uv run python scripts/evaluate_deeponet_cfc.py \
  --config configs/exp_064_re10000_xlarge_sensor_physics.toml \
  --checkpoint artifacts/kolmogorov/deeponet-cfc-re10000-exp064-sensor-physics/checkpoints/lnn_kolmogorov_step_10000.pt \
  --output-dir artifacts/kolmogorov/deeponet-cfc-re10000-exp064-sensor-physics/deeponet-cfc-eval
```

Outputs: `field_comparison_t5.png`, `vorticity_comparison_t5.png`, `energy_spectrum.png`, `kinetic_energy_vs_time.png`, `band_energy_rel_error_vs_time.png`, `divergence_vs_time.png`, `summary.json`, …

### Repository layout

```
src/
  lnn_kolmogorov.py       # entry point: model, training loop, physics residuals
  pi_lnn/                 # encoders, decoder, operator, training utilities
  kolmogorov_dataset.py   # sensor + DNS metadata loader

configs/
  exp_064_re10000_xlarge_sensor_physics.toml   # Re=10000 active baseline (filtering)
  exp_030_re1000_soap_sf_5k.toml               # Re=1000 early validation
  exp_059_re10000_xlarge_bidir_cfc.toml        # smoothing variant (NEGATIVE_RESULT)

scripts/
  evaluate_deeponet_cfc.py
  generate_sensors_qrpivot.py

docs/
  lnn_architecture.html   # interactive architecture + result figures
  experiment_log.md       # full experiment state (decisions, metrics, configs)
  assets/exp064/          # all figures referenced above

artifacts/
  kolmogorov/deeponet-cfc-re10000-exp064-sensor-physics/   # active checkpoint + eval
```
