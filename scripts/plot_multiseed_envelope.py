#!/usr/bin/env python3
"""scripts/plot_multiseed_envelope.py — Multi-seed mean ± std envelope plotter (C).

What
====
Aggregates per-seed time-series (KE, divergence ratio, u/v rel-L2, forcing-mode
amp/phase) from the EXP-245 multi-seed group (n=5: seeds 42, 1, 2, 3, 4) and
produces envelope figures (mean line + ±1 σ shaded band) that overwrite the
single-seed PNGs currently used in the thesis.

Why
===
Step (A) only annotated captions as "single-seed visualisation". Step (B)
refreshed the single-seed PNGs to the EXP-245_a (seed=42, 20k iter) checkpoint.
Step (C) here removes the single-seed restriction by reporting the multi-seed
distribution directly on the time-series plots, matching the main-table
"5.71 ± 0.11 %" headline at every time step.

Prerequisite — small evaluator patch
====================================
``scripts/evaluate_deeponet_cfc.py`` currently keeps trajectories in an
in-memory ``series_map`` dict (line ~692) but does not write them to disk.
Add an ``--export_arrays`` flag that dumps the trajectories before
``plot_metric_series`` is called, e.g. just before the first
``fig.savefig(...)`` block in ``main()``:

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

Then re-run the evaluator with ``--export_arrays`` for each of the five seeds,
writing to ``artifacts/eval_245_seed{42,1,2,3,4}/``.

Usage
=====
    uv run python scripts/plot_multiseed_envelope.py \\
        --seed_dirs artifacts/eval_245_seed42 \\
                    artifacts/eval_245_seed1  \\
                    artifacts/eval_245_seed2  \\
                    artifacts/eval_245_seed3  \\
                    artifacts/eval_245_seed4  \\
        --output_dir thesis/figures/results

After completion, recompile ``thesis/main.tex`` (2-pass).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pi_con.plot_style import apply_journal_rcparams, DNS, PICON  # noqa: E402

# --- Plot specifications -----------------------------------------------------
# Each entry maps a thesis-figure filename to the series.npz key(s) and labels.
# `dns_key` (optional) overlays the DNS reference line.

PLOT_SPECS: dict[str, dict] = {
    "kinetic_energy_vs_time.png": {
        "keys": ["KE"],
        "dns_key": "KE_dns",
        "labels": ["PI-CON"],
        "ylabel": r"Kinetic energy $\mathrm{KE}(t)$ (m$^2$/s$^2$)",
        "title": "Kinetic energy trajectory",
    },
    "divergence_vs_time.png": {
        "keys": ["div_ratio"],
        "dns_key": None,
        "labels": ["PI-CON"],
        "ylabel": r"div ratio $\Vert\nabla\cdot\mathbf{u}\Vert_2/\Vert\nabla\mathbf{u}\Vert_F^{\rm DNS}$",
        "title": "Divergence ratio (DNS floor 1.04% marked)",
        "hline": 0.0104,  # DNS finite-difference floor
        "hline_label": "DNS finite-difference floor",
    },
    # Standalone twin of panel 3 of the stacked main_trajectories figure, for the
    # defense slides (which place one diagnostic per half-slide and cannot use the
    # 6.6x9.6in stack). RMSE, not relative L2, for the reason given in
    # plot_combined's docstring: the DNS v magnitude falls 37% over t=3->5, so a
    # relative v error turns upward there while the absolute error stays flat.
    # Colours match the stacked figure (u magenta, v green).
    "uv_rmse_vs_time.png": {
        "keys": ["u_rmse", "v_rmse"],
        "dns_key": None,
        # Title and ylabel already say RMSE; the legend only has to name the component.
        "labels": [r"$u$", r"$v$"],
        "colors": ["#CC79A7", "#009E73"],
        "ylabel": r"RMSE (m/s)",
        "title": "Velocity-component RMSE",
        "legend_below": True,
        "legend_ncol": 3,
    },
    "kf_mode_amplitude_vs_time.png": {
        "keys": ["kf_amp"],
        "dns_key": "kf_amp_dns",
        "labels": ["PI-CON"],
        "ylabel": r"$|\hat{u}_{k_f}|$ (m/s)",
        "title": "Forcing-mode amplitude",
        "legend_below": True,
    },
    "kf_mode_phase_vs_time.png": {
        "keys": ["kf_phase"],
        "dns_key": "kf_phase_dns",
        "labels": ["PI-CON"],
        "ylabel": r"$\arg(\hat{u}_{k_f})$ (rad)",
        "title": "Forcing-mode phase",
        "legend_below": True,
    },
}


def load_seed_series(seed_dir: Path) -> dict[str, np.ndarray]:
    """Load series.npz from one seed's evaluator output_dir."""
    npz_path = seed_dir / "series.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} missing — run evaluator with --export_arrays (see docstring)."
        )
    return dict(np.load(npz_path))


def plot_one(
    time: np.ndarray,
    seed_stack: dict[str, np.ndarray],
    dns_curve: np.ndarray | None,
    spec: dict,
    output_path: Path,
) -> None:
    """Draw individual seed lines + mean + ±1σ envelope per series in `spec['keys']`.

    Visual conventions (per CLAUDE.md 圖表規則):
      - Distinct linestyles for distinct curves (DNS solid, mean dashed, seeds dotted)
        so lines remain distinguishable in greyscale / colour-blind viewing.
      - Wong-palette colours (blue / orange / green / red-pink), avoiding red-green.
    """
    apply_journal_rcparams()
    # Wong colour palette — colour-blind safe (blue, orange, vermillion, green).
    palette = [PICON, "#009E73", "#CC79A7"]  # PI-CON blue, green, magenta
    dns_color = DNS  # DNS reference: black

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    if dns_curve is not None:
        # DNS reference: solid, thick.
        ax.plot(time, dns_curve, color=dns_color, linewidth=1.8, linestyle="-",
                label="DNS reference", zorder=5)

    for i, (key, label) in enumerate(zip(spec["keys"], spec["labels"])):
        if key not in seed_stack:
            print(f"  ! missing key {key} in series.npz, skipping", file=sys.stderr)
            continue
        stack = seed_stack[key]  # shape (n_seeds, n_time)
        n_seeds = stack.shape[0]
        mean = stack.mean(axis=0)
        std = stack.std(axis=0, ddof=1) if n_seeds > 1 else np.zeros_like(mean)
        # `colors` lets a spec pin its curves to the same hues the stacked
        # main_trajectories figure uses, so the same quantity is not drawn in
        # two different colours across the thesis and the slides.
        color = spec.get("colors", palette)[i % len(spec.get("colors", palette))]

        # Layer 1: individual seed trajectories (dotted, semi-transparent).
        for s_idx in range(n_seeds):
            ax.plot(time, stack[s_idx], color=color, linewidth=0.7, alpha=0.45,
                    linestyle=":", zorder=2,
                    label=(f"{label} individual seeds (n={n_seeds}, dotted)"
                           if s_idx == 0 else None))
        # Layer 2: ±1σ shading (continuous backdrop).
        ax.fill_between(time, mean - std, mean + std, color=color, alpha=0.20,
                        linewidth=0, label=rf"{label} $\pm 1\sigma$", zorder=3)
        # Layer 3: mean line (dashed, thick) — different linestyle from DNS.
        ax.plot(time, mean, color=color, linewidth=1.8, linestyle="--",
                label=f"{label} mean (dashed)", zorder=4)

    if "hline" in spec:
        ax.axhline(spec["hline"], color="k", linestyle="-.", linewidth=1.0,
                   label=spec.get("hline_label", None), zorder=1)

    ax.set_xlabel(r"time $t$ (s)")
    ax.set_ylabel(spec["ylabel"])
    ax.set_title(spec["title"])
    if spec.get("legend_below"):
        # Legend below the axes so it never overlaps the curves.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=spec.get("legend_ncol", 2), frameon=False, fontsize=8)
    else:
        ax.legend(loc="best", fontsize=7.5, framealpha=0.85,
                  ncol=1 if len(spec["keys"]) == 1 else 2)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if spec.get("legend_below"):
        # Vector copy for the thesis (crisp text at any scale).
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_combined(
    time: np.ndarray,
    seed_stack: dict[str, np.ndarray],
    dns_curves: dict[str, np.ndarray | None],
    output_dir: Path,
) -> None:
    """Single stacked figure (KE / divergence / velocity RMSE / DNS magnitude).

    Why this layout
    ===============
    The previous thesis figure placed three panels in one LaTeX row at
    ~0.32 linewidth each; scaled down, the tick/axis fonts became unreadable, and
    each panel carried its own in-plot legend that overlapped the data. Here the
    diagnostics share one figure as a vertical stack: every panel is
    full text-width (fonts stay near 1:1 when embedded at \\linewidth), and a
    single curated legend sits *below* all panels so it never covers a curve.

    Why RMSE rather than relative L2 for the velocity panel
    ======================================================
    The earlier velocity panel plotted u/v relative L2 error, whose per-component
    denominator is the DNS component magnitude. That magnitude is not stationary:
    over t = 3 -> 5 the DNS redistributes energy between the components — KE_u
    +46.5%, KE_v -60.3%, while total KE moves only +3.3% — so ‖v‖_rms falls 37%.
    The v relative error therefore turns upward late in the window even though the
    v absolute error stays flat (RMSE 0.033 -> 0.031 m/s): an artefact of the
    shrinking denominator, readable as reconstruction decay.
    (The driver of that redistribution is not established here. The forcing
    f = (A sin(k_f y), 0) acts on u alone, but the forcing-mode amplitude
    correlates only weakly with KE_v over the window, r = -0.26.)
    Plotting RMSE (absolute, m/s) removes the artefact; the companion panel carries
    the DNS component magnitudes so the relative-error figures reported in the
    tables remain recoverable from the figure.
    """
    # slot names kept; values set to the semantic palette:
    # "orange" slot = PI-CON blue, "blue" slot = DNS black (see pi_con.plot_style)
    orange, blue, green, magenta = PICON, DNS, "#009E73", "#CC79A7"
    seed_kw = dict(linewidth=0.6, alpha=0.40, linestyle=":")
    band_kw = dict(alpha=0.20, linewidth=0)

    apply_journal_rcparams()
    with plt.rc_context({}):
        fig, (ax_ke, ax_div, ax_uv, ax_mag) = plt.subplots(
            4, 1, figsize=(6.6, 9.6), sharex=True)

        def draw(ax, key, color, scale=1.0):
            stack = seed_stack[key] * scale
            mean = stack.mean(axis=0)
            std = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(mean)
            for s in range(stack.shape[0]):
                ax.plot(time, stack[s], color=color, zorder=2, **seed_kw)
            ax.fill_between(time, mean - std, mean + std, color=color, zorder=3, **band_kw)
            ax.plot(time, mean, color=color, linewidth=1.7, linestyle="--", zorder=4)

        # Panel 1 — kinetic energy (physical units).
        if dns_curves.get("KE_dns") is not None:
            ax_ke.plot(time, dns_curves["KE_dns"], color=blue, linewidth=1.7,
                       linestyle="-", zorder=5)
        draw(ax_ke, "KE", orange)
        ax_ke.set_ylabel(r"KE$(t)$ (m$^2$/s$^2$)")
        ax_ke.set_title("Kinetic energy")

        # Panel 2 — divergence ratio in %, with DNS finite-difference floor.
        draw(ax_div, "div_ratio", orange, scale=100.0)
        ax_div.axhline(1.04, color="k", linestyle="-.", linewidth=1.0, zorder=1)
        ax_div.set_ylabel(r"div ratio $\|\nabla\!\cdot\mathbf{u}\|_2/\|\nabla\mathbf{u}\|_F$ (%)")
        ax_div.set_title("Divergence ratio")

        # Panel 3 — velocity-component RMSE (absolute, denominator-free).
        draw(ax_uv, "u_rmse", magenta)
        draw(ax_uv, "v_rmse", green)
        ax_uv.set_ylabel(r"RMSE (m/s)")
        ax_uv.set_title("Velocity-component RMSE")

        # Panel 4 — DNS component magnitudes, i.e. the denominators the
        # tabulated relative L2 errors are normalised by.
        if dns_curves.get("u_ref_rms") is not None:
            ax_mag.plot(time, dns_curves["u_ref_rms"], color=magenta,
                        linewidth=1.7, linestyle="-", zorder=5)
        if dns_curves.get("v_ref_rms") is not None:
            ax_mag.plot(time, dns_curves["v_ref_rms"], color=green,
                        linewidth=1.7, linestyle="-", zorder=5)
        ax_mag.set_ylabel(r"$\Vert\cdot\Vert_{\rm rms}$ (m/s)")
        ax_mag.set_title("DNS velocity-component magnitude")
        ax_mag.set_xlabel(r"time $t$ (s)")

        for ax in (ax_ke, ax_div, ax_uv, ax_mag):
            ax.grid(True, alpha=0.3, linewidth=0.4)

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [
            Line2D([0], [0], color=blue, lw=1.7, ls="-", label="DNS reference"),
            Line2D([0], [0], color=orange, lw=1.7, ls="--", label="PI-CON mean (KE, divergence)"),
            Line2D([0], [0], color=magenta, lw=1.7, ls="--", label=r"$u$ RMSE mean"),
            Line2D([0], [0], color=green, lw=1.7, ls="--", label=r"$v$ RMSE mean"),
            Line2D([0], [0], color=magenta, lw=1.7, ls="-", label=r"$\Vert u_{\rm DNS}\Vert_{\rm rms}$"),
            Line2D([0], [0], color=green, lw=1.7, ls="-", label=r"$\Vert v_{\rm DNS}\Vert_{\rm rms}$"),
            Line2D([0], [0], color="k", lw=1.0, ls="-.", label="DNS finite-difference floor"),
            Line2D([0], [0], color="0.4", lw=0.8, ls=":", label="individual seeds"),
            Patch(facecolor="0.4", alpha=0.25, label=r"$\pm 1\sigma$ band"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                   fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        for ext in ("pdf", "png"):
            fig.savefig(output_dir / f"main_trajectories.{ext}",
                        dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed PI-CON evaluator series into envelope plots."
    )
    parser.add_argument(
        "--seed_dirs",
        nargs="+",
        type=Path,
        required=True,
        help="One evaluator output_dir per seed (each containing series.npz).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Destination for envelope PNGs (typically thesis/figures/results).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all seeds
    print(f"Loading {len(args.seed_dirs)} seeds ...")
    series_per_seed = [load_seed_series(d) for d in args.seed_dirs]
    time_vals = series_per_seed[0]["time"]
    for i, s in enumerate(series_per_seed[1:], start=2):
        if not np.array_equal(s["time"], time_vals):
            raise ValueError(f"seed_dirs[{i}] time array mismatches seed_dirs[0]")

    # Build seed_stack: dict[series_key, ndarray(n_seeds, n_time)]
    all_keys = set().union(*(s.keys() for s in series_per_seed)) - {"time"}
    seed_stack = {
        k: np.stack([s[k] for s in series_per_seed if k in s])
        for k in all_keys
    }

    # Plot each figure spec
    for fname, spec in PLOT_SPECS.items():
        dns_curve = None
        if spec.get("dns_key"):
            dk = spec["dns_key"]
            # DNS reference is the same across seeds; take seed 0
            if dk in series_per_seed[0]:
                dns_curve = series_per_seed[0][dk]
            else:
                print(f"  ! DNS key {dk} not in series.npz", file=sys.stderr)
        try:
            plot_one(time_vals, seed_stack, dns_curve, spec, args.output_dir / fname)
            print(f"  ✓ wrote {args.output_dir / fname}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {fname}: {exc}", file=sys.stderr)

    # Combined stacked figure used by the thesis (single shared legend below).
    dns_curves = {
        k: (series_per_seed[0][k] if k in series_per_seed[0] else None)
        for k in ("KE_dns",)
    }
    # DNS component magnitudes ‖·‖_rms, recovered exactly from the exported
    # arrays: rmse / rel_L2 = (‖d‖₂/√M) / (‖d‖₂/‖ref‖₂) = ‖ref‖₂/√M. The DNS
    # reference is seed-independent, so seed 0 suffices.
    s0 = series_per_seed[0]
    for comp in ("u", "v"):
        rmse_k, rel_k = f"{comp}_rmse", f"{comp}_rel_L2"
        if rmse_k in s0 and rel_k in s0:
            dns_curves[f"{comp}_ref_rms"] = s0[rmse_k] / np.maximum(s0[rel_k], 1.0e-12)
        else:
            dns_curves[f"{comp}_ref_rms"] = None
            print(f"  ! cannot recover ‖{comp}_DNS‖: {rmse_k}/{rel_k} missing",
                  file=sys.stderr)
    try:
        plot_combined(time_vals, seed_stack, dns_curves, args.output_dir)
        print(f"  ✓ wrote {args.output_dir / 'main_trajectories.{pdf,png}'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ main_trajectories: {exc}", file=sys.stderr)

    print(f"\n✅ Multi-seed envelope figures saved to {args.output_dir}")
    print("   Now recompile thesis/main.tex (2-pass) to embed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
