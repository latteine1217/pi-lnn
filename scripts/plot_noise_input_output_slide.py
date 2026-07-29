"""Create the slide figure linking a full-field noise illustration to reconstruction.

What
====
The left block shows clean DNS u/v fields and the same fields after an
equivalent 10% per-channel Gaussian perturbation. This makes the noise level
visible over the whole domain. The right block compares DNS, clean-input
reconstruction, and 10%-noise reconstruction using shared field and error
colour scales.

Why
===
EXP-290 adds noise to sparse sensor time series before normalization; it does
not provide a noisy full field to the model. The left block is therefore
explicitly labelled as an equivalent full-field visualization, not model input.

Usage
=====
    uv run python scripts/plot_noise_input_output_slide.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pi_con.plot_style import apply_journal_rcparams  # noqa: E402

SENSOR_NPZ = ROOT / (
    "data/kolmogorov_sensors/re10000/"
    "sensors_qrpivot_K100_N256_t0-5_si100_les_n256_T50standalone_dns_values.npz"
)
DNS_NPY = ROOT / "data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy"
CLEAN_FIELDS = ROOT / "artifacts/eval_245_seed42_fields/fields.npz"
NOISY_FIELDS = ROOT / "artifacts/eval_noise_fields/noise10/fields.npz"
OUTPUT = ROOT / "thesis/slide/public/images/noise_input_output_comparison.png"

NOISE_LEVEL = 0.10
RNG_SEED = 42


def _field_axis(ax: plt.Axes) -> None:
    """Use a compact square domain without decorative ticks."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#9CA3AF")


def main() -> None:
    for path in (SENSOR_NPZ, DNS_NPY, CLEAN_FIELDS, NOISY_FIELDS):
        if not path.exists():
            raise SystemExit(f"[abort] missing {path}")

    sensor = np.load(SENSOR_NPZ)
    raw = np.stack([sensor["u"], sensor["v"]], axis=-1).astype(np.float64)
    channel_mean = raw.mean(axis=(0, 1))
    channel_std = raw.std(axis=(0, 1))

    dns = np.load(DNS_NPY, allow_pickle=True).item()
    clean_velocity = np.stack([dns["u"][-1], dns["v"][-1]], axis=-1).astype(np.float64)
    # Equivalent full-field realization of the EXP-290 per-channel noise model.
    # This array is for visualization only; training perturbs K=100 sensor values.
    rng = np.random.default_rng(RNG_SEED)
    perturbation = rng.normal(
        loc=0.0,
        scale=NOISE_LEVEL * channel_std[None, None, :],
        size=clean_velocity.shape,
    )
    noisy_velocity = clean_velocity + perturbation
    clean_eval = np.load(CLEAN_FIELDS)
    noisy_eval = np.load(NOISY_FIELDS)
    omega_ref = clean_eval["omega_ref"][-1].astype(np.float64)
    omega_clean = clean_eval["omega_pred"][-1].astype(np.float64)
    omega_noisy = noisy_eval["omega_pred"][-1].astype(np.float64)
    if not np.allclose(noisy_eval["omega_ref"][-1], omega_ref, rtol=0, atol=1e-6):
        raise SystemExit("[abort] noisy and clean evaluations use different DNS references")

    apply_journal_rcparams()
    fig = plt.figure(figsize=(13.8, 4.35), facecolor="white")
    outer = GridSpec(1, 2, figure=fig, width_ratios=(1.05, 1.55), wspace=0.14,
                     left=0.025, right=0.965, top=0.87, bottom=0.10)
    input_grid = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0], wspace=0.08, hspace=0.18)
    output_grid = GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[1], wspace=0.07, hspace=0.16)

    fig.text(0.20, 0.95, "NOISE VISUALIZATION — full DNS field", ha="center", va="center",
             fontsize=12, fontweight="bold", color="#7F1084")
    fig.text(0.69, 0.95, "OUTPUT — reconstructed vorticity", ha="center", va="center",
             fontsize=12, fontweight="bold", color="#7F1084")

    normalized_fields = [
        (dns["u"][-1] - channel_mean[0]) / channel_std[0],
        (dns["v"][-1] - channel_mean[1]) / channel_std[1],
    ]
    noisy_fields = [
        (noisy_velocity[..., 0] - channel_mean[0]) / channel_std[0],
        (noisy_velocity[..., 1] - channel_mean[1]) / channel_std[1],
    ]
    titles = (("clean DNS u", "DNS u + 10% noise"), ("clean DNS v", "DNS v + 10% noise"))
    velocity_map = None
    for row in range(2):
        for col, field in enumerate((normalized_fields[row], noisy_fields[row])):
            ax = fig.add_subplot(input_grid[row, col])
            # Transpose because stored DNS convention is [x,y], while imshow is [row(y),col(x)].
            velocity_map = ax.imshow(field.T, origin="lower", extent=(0, 1, 0, 1),
                                     cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")
            ax.set_title(titles[row][col], fontsize=9.5, pad=3)
            _field_axis(ax)
    cbar_input = fig.colorbar(velocity_map, ax=[fig.axes[i] for i in range(4)], fraction=0.024, pad=0.018)
    cbar_input.set_label(r"velocity / $\sigma_{\rm channel}$", fontsize=8.5)

    omega_limit = float(np.percentile(np.abs(omega_ref), 99))
    errors = [omega_clean - omega_ref, omega_noisy - omega_ref]
    error_limit = float(np.percentile(np.abs(np.stack(errors)), 99))
    im_field = None
    for col, (title, field) in enumerate((
        ("DNS reference", omega_ref),
        ("clean input", omega_clean),
        ("10% noisy input", omega_noisy),
    )):
        ax = fig.add_subplot(output_grid[0, col])
        im_field = ax.imshow(field.T, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r",
                             vmin=-omega_limit, vmax=omega_limit, interpolation="nearest")
        ax.set_title(title, fontsize=9.5, pad=3)
        _field_axis(ax)

    label_ax = fig.add_subplot(output_grid[1, 0])
    label_ax.axis("off")
    label_ax.text(0.5, 0.53, "error\nvs DNS", ha="center", va="center",
                  fontsize=10, fontweight="bold", color="#6B7280")

    im_error = None
    for col, error in enumerate(errors, start=1):
        ax = fig.add_subplot(output_grid[1, col])
        im_error = ax.imshow(error.T, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r",
                             vmin=-error_limit, vmax=error_limit, interpolation="nearest")
        _field_axis(ax)

    cbar_field = fig.colorbar(im_field, ax=[fig.axes[i] for i in (5, 6, 7)], fraction=0.022, pad=0.015)
    cbar_field.set_label(r"$\omega$ (1/s)", fontsize=8.5)
    cbar_error = fig.colorbar(im_error, ax=[fig.axes[i] for i in (9, 10)], fraction=0.022, pad=0.015)
    cbar_error.set_label(r"$\omega_{\rm pred}-\omega_{\rm DNS}$ (1/s)", fontsize=8.5)

    fig.text(0.20, 0.025, "Equivalent full-field visualization only; the model still receives K = 100 noisy sensors.",
             ha="center", va="center", fontsize=8.5, color="#6B7280")
    fig.text(0.69, 0.025, "Shared colour scales within each output row; t = 5, seed 42.",
             ha="center", va="center", fontsize=8.5, color="#6B7280")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, facecolor="white")
    plt.close(fig)

    rms = np.sqrt(np.mean(perturbation**2, axis=(0, 1))) / channel_std
    print(f"[saved] {OUTPUT.relative_to(ROOT)}")
    print(f"[check] realized RMS noise / channel std: u={rms[0]:.4f}, v={rms[1]:.4f}")


if __name__ == "__main__":
    main()
