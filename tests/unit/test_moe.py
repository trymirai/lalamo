import jax
import jax.numpy as jnp

from lalamo.modules import (
    GELU,
    DenseMLPConfig,
    ForwardPassMode,
    FullPrecisionLinearConfig,
    MixtureOfExpertsConfig,
    SoftmaxRouting,
)
from tests.common import assert_close


def test_moe_prefill_vs_decode_match() -> None:
    # Model hyperparameters (kept small for speed)
    model_dim = 64
    hidden_dim = 512
    num_routed_experts = 16
    num_active_routed_experts = 2

    # Build Mixture-of-Experts config with full-precision linear layers
    router_config = FullPrecisionLinearConfig(precision=jnp.float32)
    expert_linear_config = FullPrecisionLinearConfig(precision=jnp.float32)
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
    key = jax.random.PRNGKey(0)
    moe = moe_config.random_init(model_dim=model_dim, hidden_dim=hidden_dim, key=key)

    # Random inputs
    batch = 2
    suffix_tokens = 16
    inputs_key = jax.random.PRNGKey(1)
    inputs = jax.random.normal(inputs_key, (batch, suffix_tokens, model_dim), dtype=jnp.float32)

    # Compare PREFILL vs DECODE
    out_prefill = moe(inputs, forward_pass_mode=ForwardPassMode.MULTI_TOKEN)
    out_decode = moe(inputs, forward_pass_mode=ForwardPassMode.SINGLE_TOKEN)

    assert_close(result=out_decode, reference=out_prefill)
