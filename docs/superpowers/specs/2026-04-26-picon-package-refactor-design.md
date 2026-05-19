# Pi-LNN Package Refactor — Design (A1 / B2 / G1)

**Date:** 2026-04-26
**Scope:** Phase 1 of a longer model-code reorganization effort.
**Status:** Approved for implementation planning.

---

## 1. Problem & Goal

### Current state

`src/lnn_kolmogorov.py` is a 1,913-line monolith that mixes runtime utilities, positional encodings, neural-network blocks, model components, the `LiquidOperator` itself, physics residuals, training helpers, config loading, the 683-line `train_lnn_kolmogorov` function, and the CLI entry point.

**Concrete pains the monolith creates:**
- Long scroll cost on every navigation; structural concepts are not separable in the editor.
- Tests already import individual pieces (`SpatialSetEncoder`, `LearnableFourierEmb`, `make_lnn_model_fn`, `observed_channel_prediction`, `TemporalCfCEncoder`, etc.), proving the boundaries exist conceptually but not physically.
- Two import styles already coexist in the wild: `from lnn_kolmogorov import …` (most callers) and `from src.lnn_kolmogorov import …` (some tests). Any refactor must preserve both.

### Goal of this phase

**Pure structural split.** Carve the monolith into a multi-module package with clear concept-per-file boundaries. **Behavior is unchanged**; this is reorganization, not redesign.

### Explicit non-goals (deferred to a possible Phase 2 — "A2")

- Decomposing the 683-line `train_lnn_kolmogorov` function.
- Touching `kolmogorov_dataset.py`.
- Reorganizing `configs/` (52 TOMLs).
- Adding new tests beyond what's needed to validate equivalence.
- Improving the train loop, optimizer wiring, RAR logic, or any algorithm.

---

## 2. Design Decisions

| ID | Decision | Rationale |
|---|---|---|
| **A1** | Pure structural split — keep `train_lnn_kolmogorov` as one function | Lowest risk; reproducibility is preserved; train-loop decomposition is a separate, larger effort. |
| **B2** | Rename to `src/pi_lnn/`, keep `src/lnn_kolmogorov.py` as a thin compatibility shim | Aligns the package name with the project identity ("Pi-LNN"); the shim keeps all 7 existing callers and the `lnn-kolmogorov-train` console script working unchanged. |
| **G1** | Flat package — 10 files at one level, no subpackages | Matches the actual concept count; subpackages would over-engineer a project with one baseline; one `__init__.py` is the minimum re-export surface to maintain. |

---

## 3. Target Layout

```
src/
  pi_lnn/                     # new package
    __init__.py               # re-exports the full public API (back-compat)
    runtime.py                # ~55 lines — device + grad + json + count_parameters
    config.py                 # ~150 lines — DEFAULT_LNN_ARGS + load_lnn_config
    encodings.py              # ~75 lines — LearnableFourierEmb + 2 helpers
    blocks.py                 # ~85 lines — CfCCell + ResidualMLPBlock + TokenSelfAttentionBlock
    losses.py                 # ~105 lines — GradNormWeights + observed_channel_prediction
    encoders.py               # ~180 lines — SpatialSetEncoder + TemporalCfCEncoder
    decoder.py                # ~155 lines — DeepONetCfCDecoder
    operator.py               # ~180 lines — LiquidOperator + create_lnn_model + make_lnn_model_fn
    physics.py                # ~205 lines — NS/Poisson residuals + RAR + scheduling
    training.py               # ~700 lines — train_lnn_kolmogorov + main (A1: not decomposed)
  lnn_kolmogorov.py           # compat shim: re-exports from pi_lnn
  kolmogorov_dataset.py       # unchanged
```

---

## 4. File-by-File Contents (Source Mapping)

Each row maps a target file to source line ranges in the current `src/lnn_kolmogorov.py`. Symbols listed are exhaustive for that file.

| Target file | Source lines | Symbols |
|---|---|---|
| `runtime.py` | 26–80 | `_resolve_torch_device`, `configure_torch_runtime`, `_grad`, `count_parameters`, `write_json` |
| `encodings.py` | 81–154 | `periodic_fourier_encode`, `LearnableFourierEmb`, `temporal_phase_anchor` |
| `blocks.py` | 155–187, 266–318 | `CfCCell`, `ResidualMLPBlock`, `TokenSelfAttentionBlock` |
| `losses.py` | 188–265, 980–1004 | `GradNormWeights`, `_gradnorm_step`, `observed_channel_prediction` |
| `encoders.py` | 319–496 | `SpatialSetEncoder`, `TemporalCfCEncoder` |
| `decoder.py` | 497–650 | `DeepONetCfCDecoder` |
| `operator.py` | 651–802, 872–899 | `LiquidOperator`, `create_lnn_model`, `make_lnn_model_fn` |
| `physics.py` | 803–871, 900–979, 1005–1058 | `unsteady_ns_residuals`, `pressure_poisson_residual`, `_rar_update_pool`, `physics_points_at_step`, `physics_weight_at_step` |
| `config.py` | 1059–1210 | `DEFAULT_LNN_ARGS`, `_find_project_root`, `_resolve_config_path_value`, `load_lnn_config` |
| `training.py` | 1211–1913 | `train_lnn_kolmogorov`, `main` |

### Rationale for ambiguous placements

- **`make_lnn_model_fn` → `operator.py`**: it is a closure factory over `LiquidOperator`; it belongs with the model abstraction it wraps, not with losses or training.
- **`observed_channel_prediction` → `losses.py`**: it converts model output into a normalized prediction used to compute the data loss; it is loss-side machinery, not a model API.
- **`GradNormWeights` and `_gradnorm_step` → `losses.py`**: they manage relative weighting between data/physics losses; structurally an `nn.Module`, but the concern is loss balancing.
- **`_rar_update_pool` → `physics.py`**: Residual-Adaptive Refinement updates the physics-collocation point pool; it is physics-loss machinery.
- **`physics_points_at_step` and `physics_weight_at_step` → `physics.py`**: scheduling functions that are exclusively physics-loss-related.
- **`DEFAULT_LNN_ARGS` → `config.py`**: it is a defaults table consumed by `load_lnn_config`; it is configuration, not training logic, even though `main` and `train_lnn_kolmogorov` use it.

---

## 5. Cross-Module Dependency Graph

The split is designed so the import graph is a DAG (no cycles).

```
training.py    → operator, physics, losses, config, runtime, kolmogorov_dataset
operator.py    → encoders, decoder
encoders.py    → blocks, encodings
decoder.py     → blocks, encodings
losses.py      → operator   (TYPE-HINT-ONLY; see note below)
physics.py     → runtime    (only _grad)
blocks.py      → (torch only)
encodings.py   → (torch only)
runtime.py     → (stdlib + torch only)
config.py      → (stdlib only)
```

**Cycle-prevention note for `losses.py → operator.py`:**
`observed_channel_prediction` has `net: LiquidOperator` as a type annotation. To avoid a runtime import cycle:
- The current monolith already includes `from __future__ import annotations` at line 8 — this is preserved in every split file.
- Annotations are therefore strings at runtime; `LiquidOperator` does not need to be importable at module-load time.
- For static type-checker support, `losses.py` uses `if TYPE_CHECKING: from .operator import LiquidOperator`.

---

## 6. `__init__.py` Re-Export & Compat Shim

### `src/pi_lnn/__init__.py`

Explicit, exhaustive re-export of every symbol that any caller currently imports from `lnn_kolmogorov`:

```python
"""Pi-LNN: Sparse-sensor physics-constrained operator learning for turbulent flow."""
from __future__ import annotations

from .runtime import configure_torch_runtime, count_parameters, write_json
from .encodings import (
    LearnableFourierEmb,
    periodic_fourier_encode,
    temporal_phase_anchor,
)
from .blocks import CfCCell, ResidualMLPBlock, TokenSelfAttentionBlock
from .losses import GradNormWeights, observed_channel_prediction
from .encoders import SpatialSetEncoder, TemporalCfCEncoder
from .decoder import DeepONetCfCDecoder
from .operator import LiquidOperator, create_lnn_model, make_lnn_model_fn
from .physics import (
    physics_points_at_step,
    physics_weight_at_step,
    pressure_poisson_residual,
    unsteady_ns_residuals,
)
from .config import DEFAULT_LNN_ARGS, load_lnn_config
from .training import main, train_lnn_kolmogorov

__all__ = [
    "CfCCell",
    "DEFAULT_LNN_ARGS",
    "DeepONetCfCDecoder",
    "GradNormWeights",
    "LearnableFourierEmb",
    "LiquidOperator",
    "ResidualMLPBlock",
    "SpatialSetEncoder",
    "TemporalCfCEncoder",
    "TokenSelfAttentionBlock",
    "configure_torch_runtime",
    "count_parameters",
    "create_lnn_model",
    "load_lnn_config",
    "main",
    "make_lnn_model_fn",
    "observed_channel_prediction",
    "periodic_fourier_encode",
    "physics_points_at_step",
    "physics_weight_at_step",
    "pressure_poisson_residual",
    "temporal_phase_anchor",
    "train_lnn_kolmogorov",
    "unsteady_ns_residuals",
    "write_json",
]
```

### `src/lnn_kolmogorov.py` (compat shim, replaces the monolith file)

```python
"""Backward-compatibility shim. Prefer `from pi_lnn import ...` for new code."""
from pi_lnn import (  # noqa: F401  (re-exports for legacy callers)
    CfCCell,
    DEFAULT_LNN_ARGS,
    DeepONetCfCDecoder,
    GradNormWeights,
    LearnableFourierEmb,
    LiquidOperator,
    ResidualMLPBlock,
    SpatialSetEncoder,
    TemporalCfCEncoder,
    TokenSelfAttentionBlock,
    configure_torch_runtime,
    count_parameters,
    create_lnn_model,
    load_lnn_config,
    main,
    make_lnn_model_fn,
    observed_channel_prediction,
    periodic_fourier_encode,
    physics_points_at_step,
    physics_weight_at_step,
    pressure_poisson_residual,
    temporal_phase_anchor,
    train_lnn_kolmogorov,
    unsteady_ns_residuals,
    write_json,
)
```

**No `DeprecationWarning` is added in this phase.** A2 (when/if undertaken) is the natural moment to introduce one, alongside any caller migration.

---

## 7. `pyproject.toml`

**No changes required.**

- `[project.scripts] lnn-kolmogorov-train = "lnn_kolmogorov:main"` continues to work because `src/lnn_kolmogorov.py` still exists (now as a shim) and still exports `main`.
- `[tool.hatch.build.targets.wheel] packages = ["src"]` continues to package the entire `src/` tree, which now includes `pi_lnn/`.

---

## 8. Validation Strategy

The split is correct iff (a) all existing tests pass unchanged, (b) public symbols are object-identical between the two import paths, and (c) a fixed-seed training run produces byte-identical metrics before and after.

### Validation steps (in order)

1. **`uv run pytest tests/ -v`** — all 6 existing test files pass with zero modification.
2. **Dual import smoke:**
   ```bash
   uv run python -c "from lnn_kolmogorov import *; from pi_lnn import *; print('ok')"
   ```
3. **Identity check** (catches the "two copies of the class" bug):
   ```bash
   uv run python -c "
   import lnn_kolmogorov, pi_lnn
   assert lnn_kolmogorov.LiquidOperator is pi_lnn.LiquidOperator
   assert lnn_kolmogorov.train_lnn_kolmogorov is pi_lnn.train_lnn_kolmogorov
   assert lnn_kolmogorov.DEFAULT_LNN_ARGS is pi_lnn.DEFAULT_LNN_ARGS
   print('identity ok')
   "
   ```
4. **Behavioral equivalence — fixed-seed smoke train:**
   - Before refactor: `uv run python src/lnn_kolmogorov.py --config configs/smoke_re1000_uvomega.toml --device cpu` (or mps), capture per-step metrics for the first ~50 steps to a JSON.
   - After refactor: re-run with the same config and seed, capture the same metrics.
   - Assert numerical equality (exact match on metrics for the first 50 steps).
5. **Console script check:** `uv run lnn-kolmogorov-train --help` returns successfully.

### Acceptance criteria

All five steps above pass. If step 4 shows any drift, the refactor is rejected and investigated before merge.

---

## 9. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Hidden module-level state in monolith breaks when split | Low–Med | Step 4 (fixed-seed smoke train) catches any behavioral drift. |
| Missed symbol in `__init__.py` re-export breaks legacy `from lnn_kolmogorov import X` | Low | Explicit listing in shim (no `*`-only); step 1 (full pytest) and step 2 (dual import) catch the obvious cases. |
| Import cycle between `losses.py` and `operator.py` | Low | `from __future__ import annotations` already present at file top; cycle is type-only. |
| Console script `lnn-kolmogorov-train` breaks | Low | Step 5 verifies; the shim re-exports `main`, so the entry point resolves. |
| Path-dependent imports in `training.py` (e.g., `from kolmogorov_dataset import KolmogorovDataset` inside the function) break under new package layout | Med | This relative-to-`src/` import remains valid because all sys.path manipulations still add `src/` to `sys.path` and `kolmogorov_dataset.py` is unchanged at `src/`. Tested by step 1 + step 4. |

### Explicit "won't change" list

- `train_lnn_kolmogorov` function body — A1 boundary.
- `kolmogorov_dataset.py` — out of scope.
- Any `configs/*.toml` file.
- Any `scripts/*.py` or `tests/*.py` file.
- `pyproject.toml`.
- `sys.path.insert(...)` lines in scripts/tests (they continue to point at `src/`).

---

## 10. Rollout

This is a single-PR refactor. The PR contains:
1. New `src/pi_lnn/` package with 10 files.
2. `src/lnn_kolmogorov.py` replaced with the compat shim.
3. Validation evidence pasted in the PR description: pytest output, identity-check output, smoke-train metrics diff (expect: empty diff).

No staged rollout, no feature flag, no migration period — the shim guarantees the change is invisible to all current callers.

---

## 11. After This Phase

Reassess A2 (decomposing `train_lnn_kolmogorov`) once this lands. Signals favoring A2:
- Adding a new optimizer/scheduler is friction.
- Adding a new physics term forces touching the 700-line function.
- New baseline ideas keep diverging by branching the train loop.

If A2 is undertaken later, it builds on the package boundaries established here: optimizer/scheduler builders go in a new file under `pi_lnn/training/`, loss orchestration moves to `losses.py`, and the `train_lnn_kolmogorov` function shrinks to a coordinator.
