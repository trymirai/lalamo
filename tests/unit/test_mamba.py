import jax
import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis import strategies as st

from lalamo.modules import (
    FullPrecisionLinearConfig,
    Identity,
    Mamba2,
    Mamba2Config,
    SeparableCausalConvConfig,
)
from tests.common import assert_close


def make_mamba() -> Mamba2:
    precision = jnp.float32
    config = Mamba2Config(
        in_projection_config=FullPrecisionLinearConfig(precision=precision),
        out_projection_config=FullPrecisionLinearConfig(precision=precision),
        conv_config=SeparableCausalConvConfig(precision=precision, has_biases=True),
        activation=Identity(),
        kernel_size=3,
        num_heads=2,
        num_groups=1,
        head_dim=2,
        state_dim=2,
        expansion_factor=2,
        has_in_biases=True,
        has_out_biases=True,
    )
    model_dim = 4
    return config.random_init(model_dim=model_dim, key=jax.random.PRNGKey(0))


@settings(max_examples=10, deadline=None)
@given(
    seqlen=st.integers(min_value=1, max_value=32),
    padding=st.integers(min_value=0, max_value=32),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_mamba_state_respects_length_without_padding(
    seqlen: int,
    padding: int,
    seed: int,
) -> None:
    mamba = make_mamba()
    model_dim = mamba.model_dim
    full_seqlen = seqlen + padding

    full_input = jax.random.normal(
        jax.random.PRNGKey(seed),
        (full_seqlen, model_dim),
        dtype=jnp.float32,
    )

    padded_result = mamba(
        full_input,
        positional_embeddings=None,
        state=None,
        return_updated_state=True,
        length_without_padding=seqlen,
    )

    reference_result = mamba(
        full_input[:seqlen],
        positional_embeddings=None,
        state=None,
        return_updated_state=True,
        length_without_padding=None,
    )

    assert padded_result.state is not None
    assert reference_result.state is not None

    assert_close(result=padded_result.outputs[:seqlen], reference=reference_result.outputs)
    assert_close(result=padded_result.state.ssm_state, reference=reference_result.state.ssm_state)
    assert_close(result=padded_result.state.conv_state, reference=reference_result.state.conv_state)
