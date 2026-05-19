"""Backward-compatibility shim. Prefer `from pi_con import ...` for new code."""
from pi_con import (  # noqa: F401  (re-exports for legacy callers)
    CfCCell,
    DEFAULT_PICON_ARGS,
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
    create_picon_model,
    load_picon_config,
    main,
    make_picon_model_fn,
    make_picon_model_fn_uvp,
    observed_channel_prediction,
    periodic_fourier_encode,
    physics_points_at_step,
    physics_weight_at_step,
    pressure_poisson_residual,
    temporal_phase_anchor,
    train_picon_kolmogorov,
    unsteady_ns_residuals,
    write_json,
)


if __name__ == "__main__":
    main()
