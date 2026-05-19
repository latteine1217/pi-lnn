# Pi-LNN Package Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the 1,913-line `src/lnn_kolmogorov.py` monolith into a flat 10-module `src/pi_lnn/` package, replacing the original file with a backward-compat shim. Behavior must be byte-identical to the pre-refactor monolith.

**Architecture:** Incremental extraction (M1). Each task moves one logical group of definitions into a new file under `src/pi_lnn/`, then immediately re-imports those names back into `src/lnn_kolmogorov.py` so all 7 existing callers (scripts/tests) keep working between tasks. The original `lnn_kolmogorov.py` shrinks task by task; the final task replaces what remains with a pure shim.

**Tech Stack:** Python 3.11+, PyTorch 2.6+, uv, pytest, hatchling.

**Spec:** [docs/superpowers/specs/2026-04-26-pi-lnn-package-refactor-design.md](../specs/2026-04-26-pi-lnn-package-refactor-design.md)

**User-policy note (from project CLAUDE.md):** Git commands are not auto-executed. Each task ends with a "Commit" step that lists the exact suggested `git add` + `git commit` commands; the executing agent must surface these to the user rather than running them silently. The user may also defer commits and squash at the end — that is acceptable as long as the working tree is clean before declaring the plan complete.

---

## File Structure (Target — Established by Spec §3)

```
src/
  pi_lnn/
    __init__.py        # public API re-export (Task 12)
    runtime.py         # Task 2  — device/grad/json/count_parameters
    encodings.py       # Task 3  — LearnableFourierEmb + 2 helpers
    blocks.py          # Task 4  — CfCCell + ResidualMLP + TokenSelfAttention
    config.py          # Task 5  — DEFAULT_LNN_ARGS + load_lnn_config
    losses.py          # Task 6  — GradNormWeights + observed_channel_prediction
    encoders.py        # Task 7  — SpatialSetEncoder + TemporalCfCEncoder
    decoder.py         # Task 8  — DeepONetCfCDecoder
    operator.py        # Task 9  — LiquidOperator + create_lnn_model + make_lnn_model_fn
    physics.py         # Task 10 — NS/Poisson + RAR + scheduling
    training.py        # Task 11 — train_lnn_kolmogorov + main (NOT decomposed)
  lnn_kolmogorov.py    # Becomes a compat shim in Task 12
  kolmogorov_dataset.py  # UNCHANGED
```

Order rationale: leaves first (no internal deps), then layers up to `training.py` last. This keeps the monolith importable after every task.

---

## Conventions Used Throughout This Plan

- **"Move" means:** (1) create a new `pi_lnn/X.py` containing the verbatim code chunk, (2) delete the same definitions from `src/lnn_kolmogorov.py`, (3) add `from pi_lnn.X import (...)` near the top of the monolith so the remaining unmoved code keeps resolving those names.
- **Line numbers in this plan refer to the ORIGINAL monolith** (the file as it exists at the start of Task 0). After each task, line numbers in subsequent tasks remain valid against the original file — use the listed symbol names as the source of truth, not absolute positions.
- **`uv run`** is the project's Python invocation — never use bare `python`.
- **No file outside `src/pi_lnn/` and `src/lnn_kolmogorov.py` should be modified by Tasks 0–13** except where explicitly noted (only Task 0 writes a baseline JSON).
- **Pre-existing broken test:** `tests/test_evaluator_dns_idx.py` fails to collect (TDD-RED for an unimplemented `find_dns_time_idx`; out of scope here). Every `pytest` invocation in this plan uses `--ignore=tests/test_evaluator_dns_idx.py`. The expected pass count is **38 tests across 5 test files** (not "6 test files"); this is the pre-refactor baseline state.

---

## Worktree Prerequisites (Already Established by Controller)

The following one-time setup has been performed in this worktree before Task 0 is dispatched. The implementer does NOT need to re-do these — they are listed for the reviewer to verify the environment is sane.

- `data/` is a symlink to the main repo's data directory: `/Users/latteine/Documents/coding/pi-lnn/data` → `<worktree>/data`. (The `data/` path is gitignored, so the symlink does not show up in `git status`.)
- `.python-version` is set to `3.12` (copied from main repo). uv now resolves to Python 3.12.6.
- `uv sync --extra dev` has been run; `pytest`, `torch==2.10.0`, all project deps are installed.
- `uv run pytest tests/ --ignore=tests/test_evaluator_dns_idx.py -v` reports `38 passed`.
- `configs/smoke_re1000_uvomega.toml` had 3 stale fields (`rff_features`, `rff_sigma`, `rff_sigma_bands`) that the current `load_lnn_config` validator rejects as "unsupported". They have been removed (lines 18-20 in the original). This is a 3-line cleanup, NOT part of the refactor itself — the fields were dead and made the config un-runnable. After this cleanup, the config is loadable as-is via `load_lnn_config`.

**Pre-existing layout quirk:** The editable install `_editable_impl_pi_o_net.pth` adds the worktree ROOT (not `src/`) to `sys.path`. Naked `python -c "import lnn_kolmogorov"` therefore fails ModuleNotFoundError, and the `lnn-kolmogorov-train` console script is also broken (`from lnn_kolmogorov import main` cannot resolve). Pytest works because tests do `sys.path.insert(0, "src")`. **This breakage is pre-existing and out of scope for this refactor.** All verification commands in this plan use `PYTHONPATH=src` to compensate, and the console script test in Task 11/13 is replaced with `PYTHONPATH=src uv run python src/lnn_kolmogorov.py --help` which validates the same code path. A proper `pyproject.toml` fix (e.g., switching to a real src-layout) is left to a separate cleanup.

If the implementer hits any "data file not found" or "pytest not installed" error, it indicates the worktree state has been disturbed — surface to controller, do not attempt to recreate the prerequisites.

---

## Task 0: Capture Pre-Refactor Baseline

**Why:** The acceptance criterion (Spec §8 step 4) is "byte-identical metrics on a fixed-seed smoke train". We must have the pre-refactor numbers recorded before any code changes.

**Files:**
- Create: `artifacts/refactor-baseline/pre-refactor-smoke-metrics.json`
- Read-only reference: `configs/smoke_re1000_uvomega.toml`

- [ ] **Step 1: Verify worktree prerequisites are intact**

Run:
```bash
ls -la data/kolmogorov_sensors/re1000/sensors_qrpivot_K100_N128_t0-5.json \
       data/kolmogorov_sensors/re1000/sensors_qrpivot_K100_N128_t0-5_dns_values.npz \
       data/dns/kolmogorov_dns_fp64_etdrk4_Re1000_N128_T5_ds4.npy
uv run python --version
```

Expected: all three files listed (the symlink resolves them); Python version `Python 3.12.6`. If any check fails, **STOP** — the worktree prerequisites have been disturbed.

- [ ] **Step 2: Modify smoke config to make baseline cheap and reproducible**

Edit `configs/smoke_re1000_uvomega.toml`:
- Line 12 (the `dns_paths` value): change `"data/kolmogorov_dns_fp64_etdrk4_Re1000_N128_T5_ds4.npy"` → `"data/dns/kolmogorov_dns_fp64_etdrk4_Re1000_N128_T5_ds4.npy"` (the original config path is wrong; the actual file lives in the `dns/` subfolder).
- Line 45: change `iterations = 150` → `iterations = 50`
- Line 58: change `artifacts_dir = "artifacts/deeponet-cfc-smoke-uvonly-small"` → `artifacts_dir = "artifacts/refactor-baseline/pre-refactor"`

All three edits will be reverted in Step 5.

- [ ] **Step 3: Run baseline smoke train, capturing stdout**

Run (CPU device for byte-determinism — MPS may be non-deterministic across runs):
```bash
mkdir -p artifacts/refactor-baseline
uv run python src/lnn_kolmogorov.py --config configs/smoke_re1000_uvomega.toml --device cpu \
  2>&1 | tee artifacts/refactor-baseline/pre-refactor-smoke.stdout
```

Expected: training completes 50 steps without error. Console output shows per-step metrics; the same content is written to `pre-refactor-smoke.stdout`.

- [ ] **Step 4: Snapshot any structured artifacts emitted by the run**

Run:
```bash
ls -la artifacts/refactor-baseline/pre-refactor/ 2>&1 \
  | tee artifacts/refactor-baseline/pre-refactor-artifacts.ls
if [ -f artifacts/refactor-baseline/pre-refactor/summary.json ]; then
  cp artifacts/refactor-baseline/pre-refactor/summary.json \
     artifacts/refactor-baseline/pre-refactor-summary.json
fi
```

The `pre-refactor-smoke.stdout` from step 3 is the primary baseline. `pre-refactor-summary.json` (if produced) is a secondary cross-check. Both will be diffed in Task 13.

- [ ] **Step 5: Revert smoke config edits**

Run:
```bash
git checkout -- configs/smoke_re1000_uvomega.toml
```

Verify it reverted: `git diff configs/smoke_re1000_uvomega.toml` should be empty.

- [ ] **Step 6: Run baseline pytest**

Run:
```bash
uv run pytest tests/ --ignore=tests/test_evaluator_dns_idx.py -v 2>&1 | tee artifacts/refactor-baseline/pre-refactor-pytest.txt
```

Expected: `38 passed in <N>s` across the 5 test files (`test_cfc_pass_refactor.py`, `test_evaluate_deeponet_cfc.py`, `test_make_lnn_model_fn_cache.py`, `test_observed_channel_prediction_opt.py`, `test_pos_enc_optimization.py`). The pre-existing collection error in `test_evaluator_dns_idx.py` is excluded by `--ignore`. If anything other than 38 passing tests appears, **STOP** — surface to controller.

- [ ] **Step 7: Commit baseline artifacts**

Suggested commit (surface to user, do not auto-run):
```bash
git add artifacts/refactor-baseline/
git commit -m "chore: capture pre-refactor baseline metrics for pi_lnn package split"
```

---

## Task 1: Create Empty `pi_lnn/` Package Skeleton

**Why:** Establish the package directory and an importable (but empty) `__init__.py` so subsequent tasks can extract into a real package.

**Files:**
- Create: `src/pi_lnn/__init__.py`

- [ ] **Step 1: Create the package directory and stub `__init__.py`**

Create `src/pi_lnn/__init__.py` with content:
```python
"""Pi-LNN: Sparse-sensor physics-constrained operator learning for turbulent flow.

This package is currently being populated by an incremental refactor of
src/lnn_kolmogorov.py. Final public API will be re-exported here once the
refactor is complete (see docs/superpowers/specs/2026-04-26-pi-lnn-package-refactor-design.md).
"""
```

- [ ] **Step 2: Verify the package imports**

Run:
```bash
PYTHONPATH=src uv run python -c "import pi_lnn; print(pi_lnn.__doc__.splitlines()[0])"
```

Expected output: `Pi-LNN: Sparse-sensor physics-constrained operator learning for turbulent flow.`

- [ ] **Step 3: Run pytest to confirm no regressions**

Run:
```bash
uv run pytest tests/ --ignore=tests/test_evaluator_dns_idx.py -v
```

Expected: `38 passed` (same baseline as Task 0 step 6).

- [ ] **Step 4: Commit**

Suggested:
```bash
git add src/pi_lnn/__init__.py
git commit -m "refactor(pi_lnn): create empty package skeleton"
```

---

## Task 2: Extract `runtime.py`

**Why:** No internal dependencies — safest first leaf to move.

**Files:**
- Create: `src/pi_lnn/runtime.py`
- Modify: `src/lnn_kolmogorov.py` (delete original `_resolve_torch_device`, `configure_torch_runtime`, `_grad`, `count_parameters`, `write_json` definitions; add re-import line)

**Symbols moved:** `_resolve_torch_device`, `configure_torch_runtime`, `_grad`, `count_parameters`, `write_json`
**Source line range in original monolith:** 26–80

- [ ] **Step 1: Create `src/pi_lnn/runtime.py`**

Content (copied verbatim from monolith lines 1–80, dropping definitions outside this group):
```python
"""Runtime utilities: device resolution, autograd helpers, parameter counting, JSON I/O."""
from __future__ import annotations

import json
from pathlib import Path

import torch


def _resolve_torch_device(device_preference: str) -> torch.device:
    # ... copy verbatim from monolith lines 26-46
    pass


def configure_torch_runtime(device_preference: str) -> torch.device:
    # ... copy verbatim from monolith lines 48-55
    pass


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # ... copy verbatim from monolith lines 57-69
    pass


def count_parameters(model: torch.nn.Module) -> int:
    # ... copy verbatim from monolith lines 71-74
    pass


def write_json(path: Path, data: dict) -> None:
    # ... copy verbatim from monolith lines 76-78
    pass
```

**Note to executor:** The `pass` placeholders above mark where to copy the actual function bodies from the original monolith. Copy line-by-line; do not rewrite.

- [ ] **Step 2: Verify the new module imports cleanly**

Run:
```bash
PYTHONPATH=src uv run python -c "from pi_lnn.runtime import configure_torch_runtime, _grad, count_parameters, write_json, _resolve_torch_device; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Update `src/lnn_kolmogorov.py`**

(a) **Delete** lines 26–80 (the 5 function definitions).

(b) **Add** this import after the existing `import` block (around line 24):
```python
from pi_lnn.runtime import (
    _grad,
    _resolve_torch_device,
    configure_torch_runtime,
    count_parameters,
    write_json,
)
```

- [ ] **Step 4: Run pytest**

Run:
```bash
uv run pytest tests/ --ignore=tests/test_evaluator_dns_idx.py -v
```

Expected: `38 passed` (same baseline as Task 0 step 6).

- [ ] **Step 5: Run dual-import smoke**

Run:
```bash
PYTHONPATH=src uv run python -c "from lnn_kolmogorov import configure_torch_runtime; from pi_lnn.runtime import configure_torch_runtime as r2; assert configure_torch_runtime is r2; print('identity ok')"
```

Expected: `identity ok` (proves the re-import preserved object identity).

- [ ] **Step 6: Commit**

Suggested:
```bash
git add src/pi_lnn/runtime.py src/lnn_kolmogorov.py
git commit -m "refactor(pi_lnn): extract runtime utilities into pi_lnn.runtime"
```

---

## Task 3: Extract `encodings.py`

**Why:** No internal dependencies — second leaf.

**Files:**
- Create: `src/pi_lnn/encodings.py`
- Modify: `src/lnn_kolmogorov.py`

**Symbols moved:** `periodic_fourier_encode`, `LearnableFourierEmb`, `temporal_phase_anchor`
**Source line range in original monolith:** 81–154

- [ ] **Step 1: Create `src/pi_lnn/encodings.py`**

Content:
```python
"""Positional and temporal encodings for sparse-sensor operator learning."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def periodic_fourier_encode(z: torch.Tensor, domain_length: float, n_harmonics: int) -> torch.Tensor:
    # Copy verbatim from monolith lines 81-100
    pass


class LearnableFourierEmb(nn.Module):
    # Copy verbatim from monolith lines 102-130
    pass


def temporal_phase_anchor(t: torch.Tensor, T_total: float, n_harmonics: int = 2) -> torch.Tensor:
    # Copy verbatim from monolith lines 132-153
    pass
```

- [ ] **Step 2: Verify import**

Run:
```bash
PYTHONPATH=src uv run python -c "from pi_lnn.encodings import LearnableFourierEmb, periodic_fourier_encode, temporal_phase_anchor; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Update `src/lnn_kolmogorov.py`**

(a) **Delete** the 3 definitions (originally lines 81–154).

(b) **Add** to the import block:
```python
from pi_lnn.encodings import (
    LearnableFourierEmb,
    periodic_fourier_encode,
    temporal_phase_anchor,
)
```

- [ ] **Step 4: Run pytest**

Run:
```bash
uv run pytest tests/ --ignore=tests/test_evaluator_dns_idx.py -v
```

Expected: `38 passed`. Note: `tests/test_pos_enc_optimization.py` directly imports `LearnableFourierEmb` and `periodic_fourier_encode` from `src.lnn_kolmogorov` — this validates the re-import path works.

- [ ] **Step 5: Commit**

Suggested:
```bash
git add src/pi_lnn/encodings.py src/lnn_kolmogorov.py
git commit -m "refactor(pi_lnn): extract positional encodings into pi_lnn.encodings"
```

---

## Task 4: Extract `blocks.py`

**Why:** No internal dependencies (only torch). Note: source spans two non-contiguous regions in the monolith.

**Files:**
- Create: `src/pi_lnn/blocks.py`
- Modify: `src/lnn_kolmogorov.py`

**Symbols moved:** `CfCCell`, `ResidualMLPBlock`, `TokenSelfAttentionBlock`
**Source line ranges in original monolith:** 155–187 (CfCCell), 266–318 (ResidualMLPBlock + TokenSelfAttentionBlock)

- [ ] **Step 1: Create `src/pi_lnn/blocks.py`**

Content (combine the two source regions in this order):
```python
"""Reusable nn.Module building blocks: CfC cell, residual MLP, token self-attention."""
from __future__ import annotations

import torch
import torch.nn as nn


class CfCCell(nn.Module):
    # Copy verbatim from monolith lines 155-186
    pass


class ResidualMLPBlock(nn.Module):
    # Copy verbatim from monolith lines 266-285
    pass


class TokenSelfAttentionBlock(nn.Module):
    # Copy verbatim from monolith lines 287-318
    pass
```

- [ ] **Step 2: Verify import**

Run:
```bash
PYTHONPATH=src uv run python -c "from pi_lnn.blocks import CfCCell, ResidualMLPBlock, TokenSelfAttentionBlock; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Update `src/lnn_kolmogorov.py`**

(a) **Delete** `CfCCell` (orig lines 155–186).
(b) **Delete** `ResidualMLPBlock` and `TokenSelfAttentionBlock` (orig lines 266–318).
(c) **Add** to the import block:
```python
from pi_lnn.blocks import CfCCell, ResidualMLPBlock, TokenSelfAttentionBlock
```

**Important:** Do NOT delete `GradNormWeights` (orig lines 188–215) or `_gradnorm_step` (orig lines 216–264) — they belong to Task 6.

- [ ] **Step 4: Run pytest**

Run:
```bash
uv run pytest tests/ --ignore=tests/test_evaluator_dns_idx.py -v
```

Expected: `38 passed`. `tests/test_cfc_pass_refactor.py` exercises `TemporalCfCEncoder` which uses `CfCCell` internally — this validates the re-import.

- [ ] **Step 5: Commit**

Suggested:
```bash
git add src/pi_lnn/blocks.py src/lnn_kolmogorov.py
git commit -m "refactor(pi_lnn): extract nn building blocks into pi_lnn.blocks"
```

---

## Task 5: Extract `config.py`

**Why:** No internal dependencies. `DEFAULT_LNN_ARGS` is referenced by `train_lnn_kolmogorov` and `main` but those still resolve via the shim's re-import.

**Files:**
- Create: `src/pi_lnn/config.py`
- Modify: `src/lnn_kolmogorov.py`

**Symbols moved:** `DEFAULT_LNN_ARGS`, `_find_project_root`, `_resolve_config_path_value`, `load_lnn_config`
**Source line range in original monolith:** 1059–1210

- [ ] **Step 1: Create `src/pi_lnn/config.py`**

Content:
```python
"""Default training arguments and TOML config loading."""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


DEFAULT_LNN_ARGS: dict[str, Any] = {
    # Copy verbatim from monolith lines 1059-1150 (the entire dict literal)
}


def _find_project_root(start: Path) -> Path | None:
    # Copy verbatim from monolith lines 1151-1157
    pass


def _resolve_config_path_value(raw_path: str | Path, config_path: Path) -> str:
    # Copy verbatim from monolith lines 1159-1185
    pass


def load_lnn_config(config_path: Path | None) -> dict[str, Any]:
    # Copy verbatim from monolith lines 1187-1210
    pass
```

**Note:** Verify the imports `tomllib`, `Path`, `Any` are sufficient — if `_resolve_config_path_value` uses anything else (e.g., `os`), copy that import too.

- [ ] **Step 2: Verify import**

Run:
```bash
PYTHONPATH=src uv run python -c "from pi_lnn.config import DEFAULT_LNN_ARGS, load_lnn_config; assert isinstance(DEFAULT_LNN_ARGS, dict) and len(DEFAULT_LNN_ARGS) > 10; print('ok', len(DEFAULT_LNN_ARGS), 'keys')"
```

Expected: `ok N keys` where N matches the number of entries in the original `DEFAULT_LNN_ARGS`.

- [ ] **Step 3: Update `src/lnn_kolmogorov.py`**

(a) **Delete** orig lines 1059–1210.
(b) **Add** to the import block:
```python
from pi_lnn.config import (
    DEFAULT_LNN_ARGS,
    _find_project_root,
    _resolve_config_path_value,
    load_lnn_config,
)
```

- [ ] **Step 4: Run pytest**

Run:
```bash
uv run pytest tests/ --ignore=tests/test_evaluator_dns_idx.py -v
```

Expected: `38 passed`.

- [ ] **Step 5: Verify scripts still load configs**

Run:
```bash
PYTHONPATH=src uv run python -c "from lnn_kolmogorov import load_lnn_config; cfg = load_lnn_config(None); print('keys:', sorted(cfg.keys())[:5])"
```

Expected: a list of config keys, no exception.

- [ ] **Step 6: Commit**

Suggested:
```bash
git add src/pi_lnn/config.py src/lnn_kolmogorov.py
git commit -m "refactor(pi_lnn): extract config loader into pi_lnn.config"
```

---

## Task 6: Extract `losses.py`

**Why:** Loss machinery (GradNorm + observed-channel prediction) belongs together. `observed_channel_prediction` has a `LiquidOperator` type annotation — but `from __future__ import annotations` (already in monolith and mandatory in every new file) makes annotations strings at runtime, so no real import cycle.

**Files:**
- Create: `src/pi_lnn/losses.py`
- Modify: `src/lnn_kolmogorov.py`

**Symbols moved:** `GradNormWeights`, `_gradnorm_step`, `observed_channel_prediction`
**Source line ranges in original monolith:** 188–264 (GradNorm + helper), 980–1003 (observed_channel_prediction)

- [ ] **Step 1: Create `src/pi_lnn/losses.py`**

Content:
```python
"""Loss-side machinery: GradNorm dynamic loss weighting + sparse-channel prediction."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from pi_lnn.operator import LiquidOperator  # noqa: F401  (used in annotation only)


class GradNormWeights(nn.Module):
    # Copy verbatim from monolith lines 188-214
    pass


def _gradnorm_step(
    # ... full signature copied from monolith lines 216-264
):
    # Copy verbatim
    pass


def observed_channel_prediction(
    net: "LiquidOperator",
    xy: torch.Tensor,
    t_q: torch.Tensor,
    c_obs: torch.Tensor,
    observed_channel_names: tuple[str, ...],
    observed_channel_mean: torch.Tensor,
    observed_channel_std: torch.Tensor,
    h_states: torch.Tensor,
    s_time: torch.Tensor,
    sensor_pos: torch.Tensor,
) -> torch.Tensor:
    # Copy verbatim from monolith lines 980-1002
    pass
```

- [ ] **Step 2: Verify import**

Run:
```bash
PYTHONPATH=src uv run python -c "from pi_lnn.losses import GradNormWeights, observed_channel_prediction, _gradnorm_step; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Update `src/lnn_kolmogorov.py`**

(a) **Delete** `GradNormWeights` + `_gradnorm_step` (orig lines 188–264).
(b) **Delete** `observed_channel_prediction` (orig lines 980–1003).
(c) **Add** to the import block:
```python
from pi_lnn.losses import GradNormWeights, _gradnorm_step, observed_channel_prediction
```

- [ ] **Step 4: Run pytest**

Run:
```bash
uv run pytest tests/ --ignore=tests/test_evaluator_dns_idx.py -v
```

Expected: `38 passed`. Note: `tests/test_observed_channel_prediction_opt.py` directly imports `observed_channel_prediction` from `src.lnn_kolmogorov` — validates re-import.

- [ ] **Step 5: Commit**

Suggested:
```bash
git add src/pi_lnn/losses.py src/lnn_kolmogorov.py
git commit -m "refactor(pi_lnn): extract loss machinery into pi_lnn.losses"
```

---

## Task 7: Extract `encoders.py`

**Why:** First file with internal `pi_lnn` dependencies (blocks + encodings). Both must already exist (Tasks 3 + 4 done).

**Files:**
- Create: `src/pi_lnn/encoders.py`
- Modify: `src/lnn_kolmogorov.py`

**Symbols moved:** `SpatialSetEncoder`, `TemporalCfCEncoder`
**Source line range in original monolith:** 319–496

- [ ] **Step 1: Create `src/pi_lnn/encoders.py`**

Content:
```python
"""Sensor-side encoders: spatial set encoder + temporal CfC encoder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pi_lnn.blocks import CfCCell, ResidualMLPBlock, TokenSelfAttentionBlock
from pi_lnn.encodings import LearnableFourierEmb


class SpatialSetEncoder(nn.Module):
    # Copy verbatim from monolith lines 319-382
    pass


class TemporalCfCEncoder(nn.Module):
    # Copy verbatim from monolith lines 384-495
    pass
```

**Note:** Verify which symbols `SpatialSetEncoder` and `TemporalCfCEncoder` actually use. If they only need `CfCCell` and `LearnableFourierEmb`, drop the unused imports above. Match the original's import discipline.

- [ ] **Step 2: Verify import**

Run:
```bash
PYTHONPATH=src uv run python -c "from pi_lnn.encoders import SpatialSetEncoder, TemporalCfCEncoder; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Update `src/lnn_kolmogorov.py`**

(a) **Delete** orig lines 319–496.
(b) **Add** to the import block:
```python
from pi_lnn.encoders import SpatialSetEncoder, TemporalCfCEncoder
```

- [ ] **Step 4: Run pytest**

Run:
```bash
uv run pytest tests/ --ignore=tests/test_evaluator_dns_idx.py -v
```

Expected: `38 passed`. `tests/test_pos_enc_optimization.py` imports `SpatialSetEncoder` directly; `tests/test_cfc_pass_refactor.py` imports `TemporalCfCEncoder`.

- [ ] **Step 5: Commit**

Suggested:
```bash
git add src/pi_lnn/encoders.py src/lnn_kolmogorov.py
git commit -m "refactor(pi_lnn): extract encoders into pi_lnn.encoders"
```

---

## Task 8: Extract `decoder.py`

**Why:** Depends on blocks + encodings (both extracted).

**Files:**
- Create: `src/pi_lnn/decoder.py`
- Modify: `src/lnn_kolmogorov.py`

**Symbols moved:** `DeepONetCfCDecoder`
**Source line range in original monolith:** 497–650

- [ ] **Step 1: Create `src/pi_lnn/decoder.py`**

Content:
```python
"""DeepONet-style decoder with CfC trunk and cross-attention branch."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pi_lnn.blocks import CfCCell, ResidualMLPBlock, TokenSelfAttentionBlock
from pi_lnn.encodings import LearnableFourierEmb, temporal_phase_anchor


class DeepONetCfCDecoder(nn.Module):
    # Copy verbatim from monolith lines 497-650
    pass
```

**Note:** Match imports to what the class actually uses. If it doesn't use `temporal_phase_anchor`, drop it.

- [ ] **Step 2: Verify import**

Run:
```bash
PYTHONPATH=src uv run python -c "from pi_lnn.decoder import DeepONetCfCDecoder; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Update `src/lnn_kolmogorov.py`**

(a) **Delete** orig lines 497–650.
(b) **Add** to the import block:
```python
from pi_lnn.decoder import DeepONetCfCDecoder
```

- [ ] **Step 4: Run pytest**

Run:
```bash
uv run pytest tests/ --ignore=tests/test_evaluator_dns_idx.py -v
```

Expected: `38 passed`.

- [ ] **Step 5: Commit**

Suggested:
```bash
git add src/pi_lnn/decoder.py src/lnn_kolmogorov.py
git commit -m "refactor(pi_lnn): extract decoder into pi_lnn.decoder"
```

---

## Task 9: Extract `operator.py`

**Why:** Depends on encoders + decoder (both extracted). This task also moves `make_lnn_model_fn` (a closure factory over `LiquidOperator`).

**Files:**
- Create: `src/pi_lnn/operator.py`
- Modify: `src/lnn_kolmogorov.py`

**Symbols moved:** `LiquidOperator`, `create_lnn_model`, `make_lnn_model_fn`
**Source line ranges in original monolith:** 651–802 (LiquidOperator + create_lnn_model), 872–898 (make_lnn_model_fn)

- [ ] **Step 1: Create `src/pi_lnn/operator.py`**

Content:
```python
"""Pi-LNN main operator: LiquidOperator model class + factory + closure helper."""
from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from pi_lnn.decoder import DeepONetCfCDecoder
from pi_lnn.encoders import SpatialSetEncoder, TemporalCfCEncoder


class LiquidOperator(nn.Module):
    # Copy verbatim from monolith lines 651-767
    pass


def create_lnn_model(cfg: dict[str, Any]) -> LiquidOperator:
    # Copy verbatim from monolith lines 769-801
    pass


def make_lnn_model_fn(
    net: LiquidOperator,
    sensor_vals: torch.Tensor,
    sensor_pos: torch.Tensor,
    re_norm: float,
    sensor_time: torch.Tensor,
    device: torch.device,
    h_states: torch.Tensor | None = None,
    s_time: torch.Tensor | None = None,
) -> Callable:
    # Copy verbatim from monolith lines 872-897
    pass
```

- [ ] **Step 2: Verify import**

Run:
```bash
PYTHONPATH=src uv run python -c "from pi_lnn.operator import LiquidOperator, create_lnn_model, make_lnn_model_fn; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Update `src/lnn_kolmogorov.py`**

(a) **Delete** `LiquidOperator` + `create_lnn_model` (orig lines 651–802).
(b) **Delete** `make_lnn_model_fn` (orig lines 872–898).
(c) **Add** to the import block:
```python
from pi_lnn.operator import LiquidOperator, create_lnn_model, make_lnn_model_fn
```

- [ ] **Step 4: Run pytest**

Run:
```bash
uv run pytest tests/ --ignore=tests/test_evaluator_dns_idx.py -v
```

Expected: `38 passed`. `tests/test_make_lnn_model_fn_cache.py` and `tests/test_pos_enc_optimization.py` import `LiquidOperator` and `make_lnn_model_fn` directly.

- [ ] **Step 5: Commit**

Suggested:
```bash
git add src/pi_lnn/operator.py src/lnn_kolmogorov.py
git commit -m "refactor(pi_lnn): extract LiquidOperator and factories into pi_lnn.operator"
```

---

## Task 10: Extract `physics.py`

**Why:** Depends only on `runtime._grad`. Spans three regions in the monolith.

**Files:**
- Create: `src/pi_lnn/physics.py`
- Modify: `src/lnn_kolmogorov.py`

**Symbols moved:** `unsteady_ns_residuals`, `pressure_poisson_residual`, `_rar_update_pool`, `physics_points_at_step`, `physics_weight_at_step`
**Source line ranges in original monolith:** 803–871 (NS + Poisson), 900–979 (RAR), 1005–1058 (scheduling)

- [ ] **Step 1: Create `src/pi_lnn/physics.py`**

Content:
```python
"""Physics losses: NS/Poisson residuals + Residual-Adaptive Refinement + scheduling."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from pi_lnn.runtime import _grad


def unsteady_ns_residuals(
    # ... full signature copied from monolith lines 803-836
):
    # Copy verbatim from monolith lines 803-836
    pass


def pressure_poisson_residual(
    # ... full signature copied from monolith lines 838-870
):
    # Copy verbatim from monolith lines 838-870
    pass


def _rar_update_pool(
    # ... full signature copied from monolith lines 900-978
):
    # Copy verbatim from monolith lines 900-978
    pass


def physics_points_at_step(
    step: int,
    start: int,
    end: int,
    ramp_steps: int,
    warmup_steps: int = 0,
) -> int:
    # Copy verbatim from monolith lines 1005-1032
    pass


def physics_weight_at_step(
    # ... full signature copied from monolith lines 1034-1057
):
    # Copy verbatim from monolith lines 1034-1057
    pass
```

- [ ] **Step 2: Verify import**

Run:
```bash
PYTHONPATH=src uv run python -c "from pi_lnn.physics import unsteady_ns_residuals, pressure_poisson_residual, physics_points_at_step, physics_weight_at_step, _rar_update_pool; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Update `src/lnn_kolmogorov.py`**

(a) **Delete** NS/Poisson (orig lines 803–871).
(b) **Delete** RAR (orig lines 900–979).
(c) **Delete** scheduling (orig lines 1005–1058).
(d) **Add** to the import block:
```python
from pi_lnn.physics import (
    _rar_update_pool,
    physics_points_at_step,
    physics_weight_at_step,
    pressure_poisson_residual,
    unsteady_ns_residuals,
)
```

- [ ] **Step 4: Run pytest**

Run:
```bash
uv run pytest tests/ --ignore=tests/test_evaluator_dns_idx.py -v
```

Expected: `38 passed`.

- [ ] **Step 5: Commit**

Suggested:
```bash
git add src/pi_lnn/physics.py src/lnn_kolmogorov.py
git commit -m "refactor(pi_lnn): extract physics residuals and RAR into pi_lnn.physics"
```

---

## Task 11: Extract `training.py`

**Why:** Last big chunk. Depends on everything previously extracted. After this task, the monolith is essentially empty.

**Files:**
- Create: `src/pi_lnn/training.py`
- Modify: `src/lnn_kolmogorov.py`

**Symbols moved:** `train_lnn_kolmogorov`, `main`
**Source line range in original monolith:** 1211–1913

- [ ] **Step 1: Create `src/pi_lnn/training.py`**

Content:
```python
"""Pi-LNN training loop and CLI entry point.

A1 boundary: train_lnn_kolmogorov is moved verbatim. Decomposing it is the
deferred A2 phase — see docs/superpowers/specs/2026-04-26-pi-lnn-package-refactor-design.md §11.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pi_lnn.config import DEFAULT_LNN_ARGS, load_lnn_config
from pi_lnn.losses import GradNormWeights, _gradnorm_step, observed_channel_prediction
from pi_lnn.operator import LiquidOperator, create_lnn_model, make_lnn_model_fn
from pi_lnn.physics import (
    _rar_update_pool,
    physics_points_at_step,
    physics_weight_at_step,
    pressure_poisson_residual,
    unsteady_ns_residuals,
)
from pi_lnn.runtime import configure_torch_runtime, count_parameters, write_json


def train_lnn_kolmogorov(
    args: dict[str, Any],
    log_fn: Callable[[int, dict[str, float]], None] | None = None,
) -> None:
    # Copy verbatim from monolith lines 1211-1893.
    # CRITICAL: preserve the inline `from kolmogorov_dataset import KolmogorovDataset`
    # near the top of this function — it relies on src/ being on sys.path,
    # which all callers already arrange.
    pass


def main() -> None:
    # Copy verbatim from monolith lines 1894-1909
    pass


if __name__ == "__main__":
    main()
```

**Notes for executor:**
- The 700-line `train_lnn_kolmogorov` body is copied verbatim. Do not attempt to refactor it — that is the deferred A2 phase.
- Strip imports from the function-level import block above that turn out to be unused after the copy (e.g., if `os` or `math` is not actually referenced by `train_lnn_kolmogorov` or `main`). Keep imports honest.
- The `from kolmogorov_dataset import KolmogorovDataset` line inside `train_lnn_kolmogorov` is intentional — leave it where it is. `kolmogorov_dataset.py` lives at `src/kolmogorov_dataset.py` and is reached via the same `sys.path` insertion that scripts and tests already use.

- [ ] **Step 2: Verify import**

Run:
```bash
PYTHONPATH=src uv run python -c "from pi_lnn.training import train_lnn_kolmogorov, main; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Update `src/lnn_kolmogorov.py`**

(a) **Delete** `train_lnn_kolmogorov` + `main` + the `if __name__ == "__main__"` guard (orig lines 1211–1913).
(b) **Add** to the import block:
```python
from pi_lnn.training import main, train_lnn_kolmogorov
```
(c) **Add** at the very bottom of `src/lnn_kolmogorov.py`:
```python
if __name__ == "__main__":
    main()
```

After this task, `src/lnn_kolmogorov.py` should consist of: the original module docstring, original top-level imports, a long block of `from pi_lnn.X import (...)` re-imports, and the `if __name__` guard. No definitions.

- [ ] **Step 4: Run pytest**

Run:
```bash
uv run pytest tests/ --ignore=tests/test_evaluator_dns_idx.py -v
```

Expected: `38 passed`.

- [ ] **Step 5: Verify console script and CLI smoke**

Run:
```bash
PYTHONPATH=src uv run python src/lnn_kolmogorov.py --help
```

Expected: argparse usage text printed; no traceback.

- [ ] **Step 6: Commit**

Suggested:
```bash
git add src/pi_lnn/training.py src/lnn_kolmogorov.py
git commit -m "refactor(pi_lnn): extract training loop into pi_lnn.training"
```

---

## Task 12: Replace Monolith With Compat Shim + Finalize `__init__.py`

**Why:** Now that all definitions are in `pi_lnn/`, replace the verbose re-import file with a clean compat shim, and populate `pi_lnn/__init__.py` with the public API surface.

**Files:**
- Modify: `src/pi_lnn/__init__.py` (replace stub with full re-export)
- Modify: `src/lnn_kolmogorov.py` (replace verbose re-imports with shim form)

- [ ] **Step 1: Populate `src/pi_lnn/__init__.py`**

Replace the entire file contents with:
```python
"""Pi-LNN: Sparse-sensor physics-constrained operator learning for turbulent flow."""
from __future__ import annotations

from pi_lnn.blocks import CfCCell, ResidualMLPBlock, TokenSelfAttentionBlock
from pi_lnn.config import DEFAULT_LNN_ARGS, load_lnn_config
from pi_lnn.decoder import DeepONetCfCDecoder
from pi_lnn.encoders import SpatialSetEncoder, TemporalCfCEncoder
from pi_lnn.encodings import (
    LearnableFourierEmb,
    periodic_fourier_encode,
    temporal_phase_anchor,
)
from pi_lnn.losses import GradNormWeights, observed_channel_prediction
from pi_lnn.operator import LiquidOperator, create_lnn_model, make_lnn_model_fn
from pi_lnn.physics import (
    physics_points_at_step,
    physics_weight_at_step,
    pressure_poisson_residual,
    unsteady_ns_residuals,
)
from pi_lnn.runtime import configure_torch_runtime, count_parameters, write_json
from pi_lnn.training import main, train_lnn_kolmogorov

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

- [ ] **Step 2: Replace `src/lnn_kolmogorov.py` with compat shim**

Overwrite the entire file contents with:
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


if __name__ == "__main__":
    main()
```

**Note:** The shim re-exports include private symbols only when callers depend on them. Audit confirmed no caller imports any underscore-prefixed name from `lnn_kolmogorov`, so `_grad`, `_gradnorm_step`, `_rar_update_pool`, `_resolve_torch_device`, `_find_project_root`, `_resolve_config_path_value` are intentionally excluded from the shim's public surface.

- [ ] **Step 3: Verify shim imports cleanly**

Run:
```bash
PYTHONPATH=src uv run python -c "from lnn_kolmogorov import LiquidOperator, train_lnn_kolmogorov, DEFAULT_LNN_ARGS; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Run pytest**

Run:
```bash
uv run pytest tests/ --ignore=tests/test_evaluator_dns_idx.py -v
```

Expected: `38 passed`.

- [ ] **Step 5: Commit**

Suggested:
```bash
git add src/pi_lnn/__init__.py src/lnn_kolmogorov.py
git commit -m "refactor(pi_lnn): finalize package __init__ and reduce lnn_kolmogorov to compat shim"
```

---

## Task 13: Full Validation Suite

**Why:** Spec §8 acceptance criteria. Every check below must pass before declaring the refactor complete.

**Files:** No code changes.

- [ ] **Step 1: Full pytest run**

Run:
```bash
uv run pytest tests/ --ignore=tests/test_evaluator_dns_idx.py -v 2>&1 | tee artifacts/refactor-baseline/post-refactor-pytest.txt
```

Expected: `38 passed` (same baseline as Task 0 step 6). Diff against baseline:
```bash
diff artifacts/refactor-baseline/pre-refactor-pytest.txt artifacts/refactor-baseline/post-refactor-pytest.txt
```
Expected diff: only timing lines may differ; pass/fail counts must be identical.

- [ ] **Step 2: Dual import smoke**

Run:
```bash
PYTHONPATH=src uv run python -c "from lnn_kolmogorov import *; from pi_lnn import *; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Identity check (no duplicate class objects)**

Run:
```bash
PYTHONPATH=src uv run python -c "
import lnn_kolmogorov, pi_lnn
assert lnn_kolmogorov.LiquidOperator is pi_lnn.LiquidOperator, 'LiquidOperator identity drift'
assert lnn_kolmogorov.train_lnn_kolmogorov is pi_lnn.train_lnn_kolmogorov, 'train fn identity drift'
assert lnn_kolmogorov.DEFAULT_LNN_ARGS is pi_lnn.DEFAULT_LNN_ARGS, 'DEFAULT_LNN_ARGS identity drift'
assert lnn_kolmogorov.create_lnn_model is pi_lnn.create_lnn_model, 'create_lnn_model identity drift'
assert lnn_kolmogorov.observed_channel_prediction is pi_lnn.observed_channel_prediction, 'observed_channel_prediction identity drift'
print('identity ok across 5 representative symbols')
"
```

Expected: `identity ok across 5 representative symbols`

- [ ] **Step 4: Behavioral equivalence — fixed-seed smoke train**

(a) Edit `configs/smoke_re1000_uvomega.toml` (same three edits as Task 0 step 2, with the artifacts dir pointing to post-refactor this time):
- Line 12: change `dns_paths` value from `"data/kolmogorov_dns_fp64_etdrk4_Re1000_N128_T5_ds4.npy"` → `"data/dns/kolmogorov_dns_fp64_etdrk4_Re1000_N128_T5_ds4.npy"`
- Line 45: `iterations = 150` → `iterations = 50`
- Line 58: `artifacts_dir = "artifacts/deeponet-cfc-smoke-uvonly-small"` → `artifacts_dir = "artifacts/refactor-baseline/post-refactor"`

(b) Run, capturing stdout the same way as the baseline:
```bash
uv run python src/lnn_kolmogorov.py --config configs/smoke_re1000_uvomega.toml --device cpu \
  2>&1 | tee artifacts/refactor-baseline/post-refactor-smoke.stdout
```

(c) Snapshot any structured artifacts:
```bash
if [ -f artifacts/refactor-baseline/post-refactor/summary.json ]; then
  cp artifacts/refactor-baseline/post-refactor/summary.json \
     artifacts/refactor-baseline/post-refactor-summary.json
fi
```

(d) Diff stdout (primary check) against pre-refactor baseline:
```bash
diff artifacts/refactor-baseline/pre-refactor-smoke.stdout \
     artifacts/refactor-baseline/post-refactor-smoke.stdout
```

Expected: empty diff except for non-deterministic noise (timestamps, wall-clock, file paths that include the differing `artifacts_dir`). All numeric metrics (loss values, gradient norms, weight values) MUST match exactly. If any metric value differs, **STOP** — refactor introduced behavioral drift. Bisect by reverting tasks one at a time to find which extraction broke equivalence.

(e) If `summary.json` was produced in both runs, also diff that:
```bash
diff artifacts/refactor-baseline/pre-refactor-summary.json \
     artifacts/refactor-baseline/post-refactor-summary.json
```
Expected: empty diff (or only differences in `artifacts_dir` path string).

(f) Revert smoke config:
```bash
git checkout -- configs/smoke_re1000_uvomega.toml
```

- [ ] **Step 5: Console script check**

Run:
```bash
PYTHONPATH=src uv run python src/lnn_kolmogorov.py --help
```

Expected: argparse usage text. No traceback.

- [ ] **Step 6: Verify the monolith is actually short now**

Run:
```bash
wc -l src/lnn_kolmogorov.py
```

Expected: under 50 lines (the shim plus its imports). If >100 lines, something was missed.

- [ ] **Step 7: Verify all 7 callers still work without modification**

Run:
```bash
git diff --stat origin/main -- scripts/ tests/
```

Expected: zero changes to `scripts/` and `tests/`. If any caller was modified during the refactor, **STOP** — investigate which extraction made an external change necessary (this would violate A1's scope).

- [ ] **Step 8: Final acceptance commit**

Suggested:
```bash
git add artifacts/refactor-baseline/
git commit -m "chore: validate pi_lnn refactor — pytest, identity, smoke metrics all match baseline"
```

---

## Done Criteria

The refactor is complete when:
- [x] All Task 13 sub-steps pass with no diffs.
- [x] `src/lnn_kolmogorov.py` is a < 50-line shim.
- [x] `src/pi_lnn/` contains 11 files (`__init__.py` + 10 modules).
- [x] No file under `scripts/` or `tests/` was modified.
- [x] No `configs/*.toml` was modified (the smoke-config edits in Tasks 0 and 13 were reverted).
- [x] `pyproject.toml` is unchanged.
- [x] Pre/post smoke metrics are byte-identical.

If any of the above is not true, the refactor must be either fixed or reverted — not declared done.

---

## Out of Scope (Explicit Reminder)

- The 700-line `train_lnn_kolmogorov` function is **moved verbatim**, not decomposed.
- `kolmogorov_dataset.py` is **untouched**.
- `configs/`, `scripts/`, `tests/` are **untouched** (modulo the temporary smoke-config edits which are reverted).
- No new features, no new tests beyond the validation script behavior, no documentation updates.

Phase 2 (A2 — train-loop decomposition) is a separate plan to be authored only if/when this phase is accepted and lands.
