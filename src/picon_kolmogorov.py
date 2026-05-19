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
    make_lnn_model_fn_uvp,
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
