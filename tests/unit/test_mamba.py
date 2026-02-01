import jax
import jax.numpy as jnp
import pytest
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


@pytest.mark.parametrize("chunk_size", [4, 16, 256])
@pytest.mark.parametrize("seq_len", [1, 16, 33, 257])
def test_chunked_scan_matches_sequential_scan(chunk_size: int, seq_len: int) -> None:
    """Test that _chunked_scan produces the same outputs as _scan."""
    mamba = make_mamba()
    num_heads = mamba.num_heads
    num_groups = mamba.num_groups
    head_dim = mamba.head_dim
    state_dim = mamba.state_dim

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)

    hidden_states = jax.random.normal(keys[0], (seq_len, num_heads, head_dim), dtype=jnp.float32)
    input_projection = jax.random.normal(keys[1], (seq_len, num_groups, state_dim), dtype=jnp.float32)
    output_projection = jax.random.normal(keys[2], (seq_len, num_groups, state_dim), dtype=jnp.float32)
    time_delta_log = jax.random.normal(keys[3], (seq_len, num_heads), dtype=jnp.float32)
    initial_state = jax.random.normal(keys[4], (num_heads, head_dim, state_dim), dtype=jnp.float32) * 0.1

    num_steps = seq_len

    scan_outputs, scan_final_state = mamba._scan(
        hidden_states,
        input_projection,
        output_projection,
        time_delta_log,
        initial_state,
        num_steps,
    )

    dt = jax.nn.softplus(time_delta_log)
    chunked_outputs, chunked_final_state = mamba._chunked_scan(
        hidden_states,
        input_projection,
        output_projection,
        dt,
        initial_state,
        chunk_size,
        num_steps,
    )

    assert_close(
        result=chunked_outputs,
        reference=scan_outputs,
        operation_name="chunked_scan outputs vs sequential scan",
    )

    assert_close(
        result=chunked_final_state,
        reference=scan_final_state,
        operation_name="chunked_scan final_state vs sequential scan",
    )


@pytest.mark.parametrize("chunk_size", [8, 64])
@pytest.mark.parametrize("seq_len", [16, 65])
@pytest.mark.parametrize("num_steps_offset", [0, -3, -10])
def test_chunked_scan_respects_num_steps(
    chunk_size: int, seq_len: int, num_steps_offset: int
) -> None:
    """Test that _chunked_scan correctly handles num_steps < seq_len (padding case)."""
    mamba = make_mamba()

    num_heads = mamba.num_heads
    num_groups = mamba.num_groups
    head_dim = mamba.head_dim
    state_dim = mamba.state_dim

    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, 5)

    hidden_states = jax.random.normal(keys[0], (seq_len, num_heads, head_dim), dtype=jnp.float32)
    input_projection = jax.random.normal(keys[1], (seq_len, num_groups, state_dim), dtype=jnp.float32)
    output_projection = jax.random.normal(keys[2], (seq_len, num_groups, state_dim), dtype=jnp.float32)
    time_delta_log = jax.random.normal(keys[3], (seq_len, num_heads), dtype=jnp.float32)
    initial_state = jax.random.normal(keys[4], (num_heads, head_dim, state_dim), dtype=jnp.float32) * 0.1

    num_steps = max(1, seq_len + num_steps_offset)

    scan_outputs, scan_final_state = mamba._scan(
        hidden_states,
        input_projection,
        output_projection,
        time_delta_log,
        initial_state,
        num_steps,
    )

    dt = jax.nn.softplus(time_delta_log)
    chunked_outputs, chunked_final_state = mamba._chunked_scan(
        hidden_states,
        input_projection,
        output_projection,
        dt,
        initial_state,
        chunk_size,
        num_steps,
    )

    assert_close(
        result=chunked_outputs[:num_steps],
        reference=scan_outputs[:num_steps],
        operation_name=f"chunked_scan outputs (num_steps={num_steps})",
    )

    assert_close(
        result=chunked_final_state,
        reference=scan_final_state,
        operation_name=f"chunked_scan final_state (num_steps={num_steps})",
    )


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_decode_step_matches_scan(seed: int) -> None:
    """Test that optimized _decode_step produces same results as scan-based path."""
    mamba = make_mamba()
    model_dim = mamba.model_dim

    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 2)

    initial_state = mamba.init_static_state(capacity=1)
    initial_state = initial_state.__class__(
        conv_state=jax.random.normal(keys[0], initial_state.conv_state.shape, dtype=jnp.float32),
        ssm_state=jax.random.normal(keys[1], initial_state.ssm_state.shape, dtype=jnp.float32) * 0.1,
    )

    single_token = jax.random.normal(jax.random.PRNGKey(seed + 1000), (1, model_dim), dtype=jnp.float32)

    decode_result = mamba(
        single_token,
        positional_embeddings=None,
        state=initial_state,
        return_updated_state=True,
    )

    from jax import vmap
    from einops import rearrange

    conv_inputs, gate_values, time_delta_log = vmap(mamba.in_projection)(single_token)
    conv_output, updated_conv_state = mamba.conv(
        conv_inputs,
        None,
        initial_state.conv_state,
        return_updated_state=True,
    )
    conv_activated = mamba.config.activation(conv_output)

    x_channels, input_proj_channels, output_proj_channels = jnp.split(
        conv_activated,
        [mamba.inner_dim, mamba.inner_dim + mamba.num_groups * mamba.state_dim],
        axis=-1,
    )

    hidden_states = rearrange(
        x_channels,
        "suffix_tokens (heads head_channels) -> suffix_tokens heads head_channels",
        heads=mamba.num_heads,
    )
    input_projection = rearrange(
        input_proj_channels,
        "suffix_tokens (groups state_channels) -> suffix_tokens groups state_channels",
        groups=mamba.num_groups,
    )
    output_projection = rearrange(
        output_proj_channels,
        "suffix_tokens (groups state_channels) -> suffix_tokens groups state_channels",
        groups=mamba.num_groups,
    )
    time_delta_log = rearrange(
        time_delta_log,
        "suffix_tokens heads -> suffix_tokens heads",
        heads=mamba.num_heads,
    )

    ssm_outputs, scan_final_ssm_state = mamba._scan(
        hidden_states,
        input_projection,
        output_projection,
        time_delta_log,
        initial_state.ssm_state,
        1,
    )

    skip_contribution = mamba.skip_connection_weight[None, :, None] * hidden_states
    ssm_outputs = ssm_outputs + skip_contribution
    ssm_outputs = rearrange(
        ssm_outputs,
        "suffix_tokens heads head_channels -> suffix_tokens (heads head_channels)",
    )
    gated_outputs = ssm_outputs * jax.nn.silu(gate_values + mamba.gate_bias)
    (scan_outputs,) = vmap(mamba.out_projection)(gated_outputs)

    assert_close(
        result=decode_result.outputs,
        reference=scan_outputs,
        operation_name="decode_step outputs vs scan",
    )

    assert decode_result.state is not None
    assert_close(
        result=decode_result.state.ssm_state,
        reference=scan_final_ssm_state,
        operation_name="decode_step ssm_state vs scan",
    )

    assert updated_conv_state is not None
    assert_close(
        result=decode_result.state.conv_state,
        reference=updated_conv_state,
        operation_name="decode_step conv_state vs scan",
    )


@pytest.mark.parametrize("T", [4, 64, 256])
def test_exp_segsum_numerical_properties(T: int) -> None:
    """Test that exp_segsum handles edge cases correctly."""
    key = jax.random.PRNGKey(123)
    x = jax.random.uniform(key, (2, 3, T), dtype=jnp.float32, minval=-5.0, maxval=-0.1)

    result = exp_segsum(x)

    assert jnp.all(result >= 0), "exp_segsum should produce non-negative values"

    mask = jnp.triu(jnp.ones((T, T), dtype=bool), k=1)
    upper_triangle = result[..., mask]
    assert jnp.allclose(upper_triangle, 0.0), "Upper triangle should be exactly 0"

    diagonal = jnp.diagonal(result, axis1=-2, axis2=-1)
    assert jnp.allclose(diagonal, 1.0), "Diagonal should be 1"

    for i in range(1, min(T, 5)):
        current_diag = jnp.diagonal(result, offset=-i, axis1=-2, axis2=-1)
        prev_diag = jnp.diagonal(result, offset=-(i - 1), axis1=-2, axis2=-1)
        min_len = min(current_diag.shape[-1], prev_diag.shape[-1])
        assert jnp.all(
            current_diag[..., :min_len] <= prev_diag[..., :min_len] + 1e-5
        ), f"Decay property violated at diagonal {i}"


def _ssd_intra_chunk_reference(A_cumsum, CB, X):
    """Reference implementation using exp_segsum + einsum."""
    A = jnp.diff(A_cumsum, axis=-1, prepend=jnp.zeros_like(A_cumsum[..., :1]))
    L = exp_segsum(A)
    from einops import einsum as einops_einsum
    return einops_einsum(CB, L, X, "c l s g, g r c l s, c s g r p -> c l g r p")


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

    A = -jnp.abs(jax.random.normal(keys[0], (groups, heads_per_group, num_chunks, chunk_size), dtype=jnp.float32))
    A_cumsum = jnp.cumsum(A, axis=-1)

    C = jax.random.normal(keys[1], (num_chunks, chunk_size, groups, state_dim), dtype=jnp.float32)
    B = jax.random.normal(keys[2], (num_chunks, chunk_size, groups, state_dim), dtype=jnp.float32)

    from einops import einsum as einops_einsum
    CB = einops_einsum(C, B, "c l g n, c s g n -> c l s g")

    X = jax.random.normal(keys[3], (num_chunks, chunk_size, groups, heads_per_group, head_dim), dtype=jnp.float32)

    reference_result = _ssd_intra_chunk_reference(A_cumsum, CB, X)
    fused_result = fused_ssd_intra_chunk(A_cumsum, CB, X)

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

    A = -jnp.abs(jax.random.normal(keys[0], (groups, heads_per_group, num_chunks, chunk_size), dtype=jnp.float32))
    A_cumsum = jnp.cumsum(A, axis=-1)

    C = jax.random.normal(keys[1], (num_chunks, chunk_size, groups, state_dim), dtype=jnp.float32)
    B = jax.random.normal(keys[2], (num_chunks, chunk_size, groups, state_dim), dtype=jnp.float32)

    from einops import einsum as einops_einsum
    CB = einops_einsum(C, B, "c l g n, c s g n -> c l s g")

    X_base = jnp.arange(1, chunk_size + 1, dtype=jnp.float32)
    X = jnp.broadcast_to(
        X_base[None, :, None, None, None],
        (num_chunks, chunk_size, groups, heads_per_group, head_dim)
    )

    result = fused_ssd_intra_chunk(A_cumsum, CB, X)

    X_modified = X.at[0, chunk_size // 2:, :, :, :].set(1000.0)
    result_modified = fused_ssd_intra_chunk(A_cumsum, CB, X_modified)

    assert_close(
        result=result_modified[0, :chunk_size // 2, :, :, :],
        reference=result[0, :chunk_size // 2, :, :, :],
        operation_name="causality check - first half unchanged",
    )
