import jax
import jax.numpy as jnp
import pytest
from einops import einsum as einops_einsum
from hypothesis import given, settings
from hypothesis import strategies as st

from lalamo.modules import (
    FullPrecisionLinearConfig,
    Identity,
    Mamba2,
    Mamba2Config,
    SeparableCausalConvConfig,
)
from lalamo.modules.token_mixers.mamba import exp_segsum, fused_ssd_intra_chunk
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


@pytest.mark.parametrize("seq_len", [4, 64, 256])
def test_exp_segsum_numerical_properties(seq_len: int) -> None:
    """Test that exp_segsum handles edge cases correctly."""
    key = jax.random.PRNGKey(123)
    x = jax.random.uniform(key, (2, 3, seq_len), dtype=jnp.float32, minval=-5.0, maxval=-0.1)

    result = exp_segsum(x)

    assert jnp.all(result >= 0), "exp_segsum should produce non-negative values"

    mask = jnp.triu(jnp.ones((seq_len, seq_len), dtype=bool), k=1)
    upper_triangle = result[..., mask]
    assert jnp.allclose(upper_triangle, 0.0), "Upper triangle should be exactly 0"

    diagonal = jnp.diagonal(result, axis1=-2, axis2=-1)
    assert jnp.allclose(diagonal, 1.0), "Diagonal should be 1"

    for i in range(1, min(seq_len, 5)):
        current_diag = jnp.diagonal(result, offset=-i, axis1=-2, axis2=-1)
        prev_diag = jnp.diagonal(result, offset=-(i - 1), axis1=-2, axis2=-1)
        min_len = min(current_diag.shape[-1], prev_diag.shape[-1])
        assert jnp.all(
            current_diag[..., :min_len] <= prev_diag[..., :min_len] + 1e-5,
        ), f"Decay property violated at diagonal {i}"


def _ssd_intra_chunk_reference(a_cumsum: jnp.ndarray, cb: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Reference implementation using exp_segsum + einsum."""
    a = jnp.diff(a_cumsum, axis=-1, prepend=jnp.zeros_like(a_cumsum[..., :1]))
    decay = exp_segsum(a)
    return einops_einsum(cb, decay, x, "c l s g, g r c l s, c s g r p -> c l g r p")


@pytest.mark.parametrize("chunk_size", [16, 64])
@pytest.mark.parametrize("num_chunks", [1, 2])
def test_fused_ssd_intra_chunk_matches_reference(chunk_size: int, num_chunks: int) -> None:
    """Test that fused_ssd_intra_chunk matches the reference implementation."""
    groups = 4
    heads_per_group = 2
    state_dim = 16
    head_dim = 32

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)

    a = -jnp.abs(jax.random.normal(keys[0], (groups, heads_per_group, num_chunks, chunk_size), dtype=jnp.float32))
    a_cumsum = jnp.cumsum(a, axis=-1)

    c = jax.random.normal(keys[1], (num_chunks, chunk_size, groups, state_dim), dtype=jnp.float32)
    b = jax.random.normal(keys[2], (num_chunks, chunk_size, groups, state_dim), dtype=jnp.float32)
    cb = einops_einsum(c, b, "c l g n, c s g n -> c l s g")

    x = jax.random.normal(keys[3], (num_chunks, chunk_size, groups, heads_per_group, head_dim), dtype=jnp.float32)

    reference_result = _ssd_intra_chunk_reference(a_cumsum, cb, x)
    fused_result = fused_ssd_intra_chunk(a_cumsum, cb, x)

    assert_close(
        result=fused_result,
        reference=reference_result,
        operation_name="fused_ssd_intra_chunk vs reference",
    )


@pytest.mark.parametrize("chunk_size", [16, 32, 64])
def test_fused_ssd_preserves_causality(chunk_size: int) -> None:
    """Test that the fused kernel maintains causal masking (no future leakage)."""
    groups = 2
    heads_per_group = 2
    state_dim = 8
    head_dim = 16
    num_chunks = 1

    key = jax.random.PRNGKey(999)
    keys = jax.random.split(key, 4)

    a = -jnp.abs(jax.random.normal(keys[0], (groups, heads_per_group, num_chunks, chunk_size), dtype=jnp.float32))
    a_cumsum = jnp.cumsum(a, axis=-1)

    c = jax.random.normal(keys[1], (num_chunks, chunk_size, groups, state_dim), dtype=jnp.float32)
    b = jax.random.normal(keys[2], (num_chunks, chunk_size, groups, state_dim), dtype=jnp.float32)
    cb = einops_einsum(c, b, "c l g n, c s g n -> c l s g")

    x_base = jnp.arange(1, chunk_size + 1, dtype=jnp.float32)
    x = jnp.broadcast_to(
        x_base[None, :, None, None, None],
        (num_chunks, chunk_size, groups, heads_per_group, head_dim),
    )

    result = fused_ssd_intra_chunk(a_cumsum, cb, x)

    x_modified = x.at[0, chunk_size // 2:, :, :, :].set(1000.0)
    result_modified = fused_ssd_intra_chunk(a_cumsum, cb, x_modified)

    assert_close(
        result=result_modified[0, :chunk_size // 2, :, :, :],
        reference=result[0, :chunk_size // 2, :, :, :],
        operation_name="causality check - first half unchanged",
    )
