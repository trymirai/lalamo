import jax
import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis import strategies as st

from lalamo.modules import (
    FullPrecisionLinearConfig,
    SeparableCausalConvConfig,
    ShortConv,
    ShortConvConfig,
)
from tests.common import assert_close


def make_short_conv() -> ShortConv:
    precision = jnp.float32
    config = ShortConvConfig(
        in_projection_config=FullPrecisionLinearConfig(precision=precision),
        conv_config=SeparableCausalConvConfig(precision=precision, has_biases=True),
        out_projection_config=FullPrecisionLinearConfig(precision=precision),
        kernel_size=3,
    )
    model_dim = 4
    return config.random_init(model_dim=model_dim, key=jax.random.PRNGKey(0))


@settings(max_examples=10, deadline=None)
@given(
    seqlen=st.integers(min_value=1, max_value=32),
    padding=st.integers(min_value=0, max_value=32),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_short_conv_state_respects_length_without_padding(
    seqlen: int,
    padding: int,
    seed: int,
) -> None:
    short_conv = make_short_conv()
    model_dim = short_conv.model_dim
    full_seqlen = seqlen + padding

    full_input = jax.random.normal(
        jax.random.PRNGKey(seed),
        (full_seqlen, model_dim),
        dtype=jnp.float32,
    )

    padded_result = short_conv(
        full_input,
        positional_embeddings=None,
        state=None,
        return_updated_state=True,
        length_without_padding=seqlen,
    )

    reference_result = short_conv(
        full_input[:seqlen],
        positional_embeddings=None,
        state=None,
        return_updated_state=True,
        length_without_padding=None,
    )

    assert padded_result.state is not None
    assert reference_result.state is not None

    assert_close(result=padded_result.outputs[:seqlen], reference=reference_result.outputs)
    assert_close(result=padded_result.state.conv_state, reference=reference_result.state.conv_state)
