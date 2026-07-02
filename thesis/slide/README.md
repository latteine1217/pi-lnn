# Pi-LNN Lab Talk — Slidev Deck

Audience: lab members & advisor (CFD specialists, AI non-specialists).
Language: English. Charts: chart.js with real EXP-064 / AL-series data
extracted from `artifacts/.../summary.json` (no synthetic numbers).

## Run locally

```bash
cd thesis/slide
npm install            # one-off
npm run dev            # opens at http://localhost:3030
```

## Export

```bash
npm run build          # static SPA in dist/
npm run export-pdf     # pi-lnn-talk.pdf  (requires Playwright Chromium; slidev installs on first run)
```

## Layout

- `slides.md` — main deck (28 slides, 8 sections).
- `components/` — UI primitives (auto-loaded by Slidev). Copied verbatim
  from `thesis/components/` plus one new `ChartCanvas.vue` wrapping chart.js.
- `data/` — JSON snapshots of real evaluation summaries (EXP-064 timeseries,
  9-point AL Pareto, ablation tables, per-band errors). Imported directly
  by per-slide `<script setup>` blocks.
- `public/images/` — flow-field PNGs (vorticity, velocity field, sensor
  layout, architecture diagram). Anything that is *not* a numerical chart.
- `setup/main.ts` — global chart.js registration + design-system defaults.
- `style.css` — slide-wide background and font fallbacks.

## Updating numbers

If you re-run an experiment and want fresh charts:

```bash
# example: re-extract EXP-064 timeseries from a new eval
python3 -c "
import json
src='artifacts/eval-rerun-NEW/exp064-main/summary.json'
dst='thesis/slide/data/exp064_timeseries.json'
with open(src) as f: d=json.load(f)
ts={'t':[s['time'] for s in d['steps']],
    'split':[s['split'] for s in d['steps']],
    'ke_rel_err':[s['ke_rel_err'] for s in d['steps']],
    'div_l2':[s['div_l2'] for s in d['steps']],
    'div_ref_l2':[s['div_ref_l2'] for s in d['steps']],
    'ens_rel_err':[s['ens_rel_err'] for s in d['steps']],
    'u_rel_l2':[s['u_rel_l2'] for s in d['steps']],
    'v_rel_l2':[s['v_rel_l2'] for s in d['steps']],
    'omega_rel_l2':[s['omega_rel_l2'] for s in d['steps']],
    'band_low':[s['band_rel_err_low'] for s in d['steps']],
    'band_mid':[s['band_rel_err_mid'] for s in d['steps']],
    'band_high':[s['band_rel_err_high'] for s in d['steps']],
    'kf_amp_ref':[s['kf_amp_ref'] for s in d['steps']],
    'kf_amp_pred':[s['kf_amp_pred'] for s in d['steps']]}
json.dump(ts, open(dst,'w'))
"
```
