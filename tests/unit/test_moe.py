import jax
import jax.numpy as jnp
import pytest

from lalamo.arrays import FullPrecisionArray
from lalamo.common import ParameterPath
from lalamo.model_import.loaders.huggingface import load_moe
from lalamo.module import RandomInitializer
from lalamo.modules import (
    GELU,
    DenseMLPConfig,
    ForwardPassMode,
    LinearConfig,
    MixtureOfExpertsConfig,
    SoftmaxRouting,
)
from tests.common import assert_close


@pytest.mark.fast
def test_moe_prefill_vs_decode_match() -> None:
    # Model hyperparameters (kept small for speed)
    model_dim = 64
    hidden_dim = 512
    num_routed_experts = 16
    num_active_routed_experts = 2

    # Build Mixture-of-Experts config with full-precision linear layers
    router_config = LinearConfig(precision=jnp.float32)
    expert_linear_config = LinearConfig(precision=jnp.float32)
    expert_config = DenseMLPConfig(
        linear_config=expert_linear_config,
        activation=GELU(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )

    moe_config = MixtureOfExpertsConfig(
        num_routed_experts=num_routed_experts,
        num_active_routed_experts=num_active_routed_experts,
        routing_function=SoftmaxRouting(),
        router_config=router_config,
        router_has_biases=False,
        num_shared_experts=0,
        expert_config=expert_config,
        expert_hidden_dim=hidden_dim,
    )

    # Initialize MoE
    key = jax.random.key(0)
    moe = moe_config.init(
        RandomInitializer(precision=jnp.bfloat16, key=key), model_dim=model_dim, hidden_dim=hidden_dim
    )

    # Random inputs
    batch = 2
    suffix_tokens = 16
    inputs_key = jax.random.key(1)
    inputs = jax.random.normal(inputs_key, (batch, suffix_tokens, model_dim), dtype=jnp.float32)

    # Compare PREFILL vs DECODE
    out_prefill = moe(inputs, forward_pass_mode=ForwardPassMode.MULTI_TOKEN, key=None)
    out_decode = moe(inputs, forward_pass_mode=ForwardPassMode.SINGLE_TOKEN, key=None)

    assert_close(result=out_decode, reference=out_prefill)


@pytest.mark.fast
def test_load_moe_gpt_oss_fused_weights_wraps_linear_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dim = 4
    hidden_dim = 3
    moe_config = MixtureOfExpertsConfig(
        num_routed_experts=2,
        num_active_routed_experts=1,
        routing_function=SoftmaxRouting(),
        router_config=LinearConfig(),
        router_has_biases=False,
        num_shared_experts=0,
        expert_config=DenseMLPConfig(
            linear_config=LinearConfig(),
            activation=GELU(),
            has_up_biases=True,
            has_down_biases=True,
            gate_clipping=None,
            up_clipping=None,
        ),
        expert_hidden_dim=hidden_dim,
    )
    module = moe_config.init(
        RandomInitializer(precision=jnp.float32, key=jax.random.key(0)),
        model_dim=model_dim,
        hidden_dim=hidden_dim,
    )

    gate_up_fused = jnp.arange(2 * 6 * 2 * 2, dtype=jnp.float32).reshape(2, 6, 2, 2)
    down_fused = jnp.arange(2 * 4 * 1 * 3, dtype=jnp.float32).reshape(2, 4, 1, 3)

    def fake_decode_mxfp4(
        blocks: jax.Array,
        scales: jax.Array,  # noqa: ARG001
        *,
        dtype: jnp.dtype,  # noqa: ARG001
        flatten: bool = False,  # noqa: ARG001
    ) -> jax.Array:
        if blocks.shape == (2, 6, 2, 2):
            return gate_up_fused
        if blocks.shape == (2, 4, 1, 3):
            return down_fused
        raise AssertionError(f"Unexpected blocks shape: {blocks.shape}")

    monkeypatch.setattr("lalamo.model_import.loaders.huggingface.decode_mxfp4", fake_decode_mxfp4)

    loaded = load_moe(
        module,
        {
            ParameterPath("moe.router.weight"): jnp.arange(8, dtype=jnp.float32).reshape(2, 4),
            ParameterPath("moe.experts.gate_up_proj_blocks"): jnp.zeros((2, 6, 2, 2), dtype=jnp.uint8),
            ParameterPath("moe.experts.gate_up_proj_scales"): jnp.zeros((2, 6, 2), dtype=jnp.uint8),
            ParameterPath("moe.experts.gate_up_proj_bias"): jnp.arange(6, dtype=jnp.float32),
            ParameterPath("moe.experts.down_proj_blocks"): jnp.zeros((2, 4, 1, 3), dtype=jnp.uint8),
            ParameterPath("moe.experts.down_proj_scales"): jnp.zeros((2, 4, 1), dtype=jnp.uint8),
            ParameterPath("moe.experts.down_proj_bias"): jnp.arange(4, dtype=jnp.float32),
        },
        ParameterPath("moe"),
    )

    assert isinstance(loaded.experts.up_projection.weights, FullPrecisionArray)
    assert isinstance(loaded.experts.down_projection.weights, FullPrecisionArray)
    assert loaded.experts.up_projection.weights.materialize().shape == (2, hidden_dim * 2, model_dim)
    assert loaded.experts.down_projection.weights.materialize().shape == (2, model_dim, hidden_dim)
