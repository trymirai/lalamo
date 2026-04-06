from typing import Any

import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lalamo.common import ParameterPath
from lalamo.model_import.loaders.huggingface import load_delta_net_attention
from lalamo.modules import (
    DeltaNetAttentionConfig,
    FullPrecisionLinearConfig,
    NormalizationConfig,
    SeparableCausalConvConfig,
    UpcastMode,
)
from lalamo.modules.token_mixers.delta_net_attention import DeltaNetAttention
from lalamo.modules.torch_interop import torch_to_jax
from tests.common import assert_close


def _make_hf_delta_net() -> tuple[Any, Any]:
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet

    config = Qwen3NextConfig(
        hidden_size=64,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=128,
        num_hidden_layers=1,
        vocab_size=128,
        max_position_embeddings=64,
        use_cache=False,
    )
    return Qwen3NextGatedDeltaNet(config, layer_idx=0), config


def _make_lalamo_delta_net(hf_config: Any):
    precision = jnp.float32
    norm_config = NormalizationConfig(
        scale_precision=precision,
        accumulation_precision=precision,
        epsilon=hf_config.rms_norm_eps,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
    )
    linear_config = FullPrecisionLinearConfig(precision=precision)
    config = DeltaNetAttentionConfig(
        in_proj_config=linear_config,
        conv_config=SeparableCausalConvConfig(precision=precision, has_biases=False),
        out_proj_config=linear_config,
        norm_config=norm_config,
        num_heads=hf_config.linear_num_value_heads,
        num_groups=hf_config.linear_num_key_heads,
        head_dim=hf_config.linear_key_head_dim,
        value_head_dim=hf_config.linear_value_head_dim,
        kernel_size=hf_config.linear_conv_kernel_dim,
    )
    return config.random_init(model_dim=hf_config.hidden_size, key=jax.random.PRNGKey(0))


def test_delta_net_attention_matches_hf() -> None:
    import torch

    torch.manual_seed(0)
    hf_module, hf_config = _make_hf_delta_net()
    hf_module = hf_module.eval()
    lalamo_module = _make_lalamo_delta_net(hf_config)

    weights = {
        ParameterPath("in_proj_qkvz") / "weight": torch_to_jax(hf_module.in_proj_qkvz.weight),
        ParameterPath("in_proj_ba") / "weight": torch_to_jax(hf_module.in_proj_ba.weight),
        ParameterPath("conv1d") / "weight": torch_to_jax(hf_module.conv1d.weight),
        ParameterPath("out_proj") / "weight": torch_to_jax(hf_module.out_proj.weight),
        ParameterPath("norm") / "weight": torch_to_jax(hf_module.norm.weight),
        ParameterPath("dt_bias"): torch_to_jax(hf_module.dt_bias),
        ParameterPath("A_log"): torch_to_jax(hf_module.A_log),
    }

    lalamo_module = load_delta_net_attention(
        lalamo_module,
        weights,
        ParameterPath(""),
        permute_conv=False,
    )

    inputs = torch.randn(1, 7, hf_config.hidden_size, dtype=torch.float32)
    with torch.no_grad():
        hf_out = hf_module(inputs, attention_mask=None)

    lalamo_out = lalamo_module(
        torch_to_jax(inputs[0]),
        positional_embeddings=None,
        state=None,
        return_updated_state=False,
        length_without_padding=None,
    ).outputs

    assert_close(
        result=lalamo_out,
        reference=torch_to_jax(hf_out[0]),
        atol=1e-3,
        fraction_of_allowed_violations=0.02,
    )


def _make_delta_net(
    chunk_size: int = 4,
    num_heads: int = 2,
    num_groups: int = 2,
) -> DeltaNetAttention:
    precision = jnp.float32
    norm_config = NormalizationConfig(
        scale_precision=precision,
        accumulation_precision=precision,
        epsilon=1e-6,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
    )
    config = DeltaNetAttentionConfig(
        in_proj_config=FullPrecisionLinearConfig(precision=precision),
        conv_config=SeparableCausalConvConfig(precision=precision, has_biases=False),
        out_proj_config=FullPrecisionLinearConfig(precision=precision),
        norm_config=norm_config,
        num_heads=num_heads,
        num_groups=num_groups,
        head_dim=8,
        value_head_dim=8,
        kernel_size=3,
        chunk_size=chunk_size,
    )
    return config.random_init(model_dim=64, key=jax.random.PRNGKey(42))


@pytest.mark.parametrize(
    ("seq_len", "num_steps", "chunk_size", "use_nonzero_state"),
    [
        # No padding (num_steps == seq_len), zero initial state
        (1, 1, 4, False),
        (3, 3, 4, False),
        (4, 4, 4, False),
        (7, 7, 4, False),
        (8, 8, 4, False),
        (16, 16, 4, False),
        # Padding (num_steps < seq_len), zero initial state
        (8, 3, 4, False),
        (8, 4, 4, False),
        (8, 7, 4, False),
        (16, 1, 4, False),
        (16, 9, 4, False),
        (8, 0, 4, False),
        # Non-zero initial state (tests Phase 2 propagation + Phase 3 correction)
        (1, 1, 4, True),
        (7, 7, 4, True),
        (8, 8, 4, True),
        (16, 16, 4, True),
        (8, 3, 4, True),
        (16, 9, 4, True),
        # Degenerate chunk_size=1
        (4, 4, 1, False),
        (4, 4, 1, True),
        # chunk_size > seq_len
        (3, 3, 16, False),
        (3, 3, 16, True),
    ],
)
def test_chunked_scan_matches_sequential(
    seq_len: int, num_steps: int, chunk_size: int, use_nonzero_state: bool,
) -> None:
    module = _make_delta_net(chunk_size=chunk_size)
    k1, k2, k3, k4, k5, k6 = jax.random.split(jax.random.PRNGKey(0), 6)

    queries = jax.random.normal(k1, (seq_len, module.num_heads, module.head_dim))
    keys = jax.random.normal(k2, (seq_len, module.num_heads, module.head_dim))
    values = jax.random.normal(k3, (seq_len, module.num_heads, module.value_head_dim))
    decay_factor = -jnp.abs(jax.random.normal(k4, (seq_len, module.num_heads)))
    beta = jax.nn.sigmoid(jax.random.normal(k5, (seq_len, module.num_heads)))

    state_shape = (module.num_heads, module.value_head_dim, module.head_dim)
    initial_state = (
        jax.random.normal(k6, state_shape) * 0.1
        if use_nonzero_state
        else jnp.zeros(state_shape, dtype=jnp.float32)
    )

    out_seq, state_seq = module._recurrent_scan(
        queries, keys, values, decay_factor, beta, initial_state, num_steps,
    )
    out_chunk, state_chunk = module._chunked_scan(
        queries, keys, values, decay_factor, beta, initial_state, num_steps,
    )

    valid = max(num_steps, 0)
    if valid > 0:
        assert_close(result=out_chunk[:valid], reference=out_seq[:valid], atol=1e-5)
    assert_close(result=state_chunk, reference=state_seq, atol=1e-5)


@settings(max_examples=10, deadline=None)
@given(
    seqlen=st.integers(min_value=1, max_value=32),
    padding=st.integers(min_value=0, max_value=32),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_delta_net_state_respects_length_without_padding(
    seqlen: int,
    padding: int,
    seed: int,
) -> None:
    module = _make_delta_net(chunk_size=4)
    full_input = jax.random.normal(
        jax.random.PRNGKey(seed),
        (seqlen + padding, module.model_dim),
        dtype=jnp.float32,
    )

    padded_result = module(
        full_input,
        positional_embeddings=None,
        state=None,
        return_updated_state=True,
        length_without_padding=seqlen,
    )
    reference_result = module(
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


@pytest.mark.parametrize("seq_len", [8, 16])
def test_delta_net_chunked_preserves_causality(seq_len: int) -> None:
    module = _make_delta_net(chunk_size=4)
    inputs = jax.random.normal(jax.random.PRNGKey(123), (seq_len, module.model_dim))

    result_original = module(inputs, positional_embeddings=None)

    midpoint = seq_len // 2
    modified_inputs = inputs.at[midpoint:].set(
        jax.random.normal(jax.random.PRNGKey(999), (seq_len - midpoint, module.model_dim)),
    )
    result_modified = module(modified_inputs, positional_embeddings=None)

    assert_close(
        result=result_modified.outputs[:midpoint],
        reference=result_original.outputs[:midpoint],
    )


def test_delta_net_chunked_with_gqa() -> None:
    module = _make_delta_net(chunk_size=4, num_heads=4, num_groups=2)
    inputs = jax.random.normal(jax.random.PRNGKey(7), (12, module.model_dim))

    result = module(inputs, positional_embeddings=None, return_updated_state=True)
    assert result.outputs.shape == (12, module.model_dim)
    assert result.state is not None

    # Verify chunked matches sequential with GQA by running the scan-level comparison
    # after __call__ projections (the repeat happens before _scan/_chunked_scan)
    module_seq = _make_delta_net(chunk_size=4, num_heads=4, num_groups=2)
    k1, k2, k3, k4, k5, k6 = jax.random.split(jax.random.PRNGKey(3), 6)
    seq_len = 9

    queries = jax.random.normal(k1, (seq_len, 4, 8))
    keys = jax.random.normal(k2, (seq_len, 4, 8))
    values = jax.random.normal(k3, (seq_len, 4, 8))
    decay_factor = -jnp.abs(jax.random.normal(k4, (seq_len, 4)))
    beta = jax.nn.sigmoid(jax.random.normal(k5, (seq_len, 4)))
    initial_state = jax.random.normal(k6, (4, 8, 8)) * 0.1

    out_seq, state_seq = module_seq._recurrent_scan(
        queries, keys, values, decay_factor, beta, initial_state, seq_len,
    )
    out_chunk, state_chunk = module_seq._chunked_scan(
        queries, keys, values, decay_factor, beta, initial_state, seq_len,
    )

    assert_close(result=out_chunk, reference=out_seq, atol=1e-5)
    assert_close(result=state_chunk, reference=state_seq, atol=1e-5)


def test_delta_net_multi_step_generation() -> None:
    module = _make_delta_net(chunk_size=4)
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(42), 3)
    chunk1 = jax.random.normal(k1, (6, module.model_dim))
    chunk2 = jax.random.normal(k2, (5, module.model_dim))
    chunk3 = jax.random.normal(k3, (7, module.model_dim))

    # Multi-step: process chunks sequentially, passing state forward
    result1 = module(chunk1, positional_embeddings=None, return_updated_state=True)
    assert result1.state is not None
    result2 = module(chunk2, positional_embeddings=None, state=result1.state, return_updated_state=True)
    assert result2.state is not None
    result3 = module(chunk3, positional_embeddings=None, state=result2.state, return_updated_state=True)

    # Single-step: process all at once
    full_input = jnp.concatenate([chunk1, chunk2, chunk3], axis=0)
    result_full = module(full_input, positional_embeddings=None, return_updated_state=True)

    # Outputs must match
    multi_outputs = jnp.concatenate([result1.outputs, result2.outputs, result3.outputs], axis=0)
    assert_close(result=multi_outputs, reference=result_full.outputs, atol=1e-4)

    # Final state must match
    assert result3.state is not None
    assert result_full.state is not None
    assert_close(result=result3.state.ssm_state, reference=result_full.state.ssm_state, atol=1e-4)


@pytest.mark.parametrize(
    ("chunk_size_a", "chunk_size_b"),
    [(1, 4), (2, 8), (3, 7), (4, 16)],
)
def test_delta_net_output_invariant_to_chunk_size(chunk_size_a: int, chunk_size_b: int) -> None:
    module_a = _make_delta_net(chunk_size=chunk_size_a)
    module_b = _make_delta_net(chunk_size=chunk_size_b)
    inputs = jax.random.normal(jax.random.PRNGKey(0), (12, module_a.model_dim))

    result_a = module_a(inputs, positional_embeddings=None, return_updated_state=True)
    result_b = module_b(inputs, positional_embeddings=None, return_updated_state=True)

    assert_close(result=result_a.outputs, reference=result_b.outputs, atol=1e-5)
    assert result_a.state is not None
    assert result_b.state is not None
    assert_close(result=result_a.state.ssm_state, reference=result_b.state.ssm_state, atol=1e-5)


@pytest.mark.parametrize("seq_len", [128, 512])
def test_chunked_scan_matches_sequential_large(seq_len: int) -> None:
    module = _make_delta_net(chunk_size=64)
    k1, k2, k3, k4, k5, k6 = jax.random.split(jax.random.PRNGKey(0), 6)

    queries = jax.random.normal(k1, (seq_len, module.num_heads, module.head_dim))
    keys = jax.random.normal(k2, (seq_len, module.num_heads, module.head_dim))
    values = jax.random.normal(k3, (seq_len, module.num_heads, module.value_head_dim))
    decay_factor = -jnp.abs(jax.random.normal(k4, (seq_len, module.num_heads)))
    beta = jax.nn.sigmoid(jax.random.normal(k5, (seq_len, module.num_heads)))
    initial_state = jax.random.normal(k6, (module.num_heads, module.value_head_dim, module.head_dim)) * 0.1

    out_seq, state_seq = module._recurrent_scan(
        queries, keys, values, decay_factor, beta, initial_state, seq_len,
    )
    out_chunk, state_chunk = module._chunked_scan(
        queries, keys, values, decay_factor, beta, initial_state, seq_len,
    )

    assert_close(result=out_chunk, reference=out_seq, atol=2e-3, fraction_of_allowed_violations=0.01)
    assert_close(result=state_chunk, reference=state_seq, atol=2e-3, fraction_of_allowed_violations=0.01)


@pytest.mark.parametrize(
    ("seq_len", "chunk_size", "min_chunk_len"),
    [
        # seq_len < min_chunk_len: __call__ routes entirely to _scan
        (3, 8, 4, ),
        (1, 4, 2),
        # Tail falls back to _scan: remainder=2 < min_chunk_len=3
        (10, 8, 3),
        # Tail falls back to _scan: remainder=1 < min_chunk_len=4
        (9, 8, 4),
        # No tail: remainder=0 (exact multiple)
        (16, 8, 4),
        # No tail: remainder=4 >= min_chunk_len=4 (padded normally)
        (12, 8, 4),
        # Large min_chunk_len forces most to _scan tail
        (5, 4, 3),
    ],
)
def test_min_chunk_len_matches_sequential(seq_len: int, chunk_size: int, min_chunk_len: int) -> None:
    from dataclasses import replace as dc_replace

    module = _make_delta_net(chunk_size=chunk_size)
    module_with_min = DeltaNetAttention(
        dc_replace(module.config, min_chunk_len=min_chunk_len),
        in_proj=module.in_proj,
        conv=module.conv,
        out_proj=module.out_proj,
        norm=module.norm,
        dt_bias=module.dt_bias,
        a_log=module.a_log,
        num_heads=module.num_heads,
        num_groups=module.num_groups,
        head_dim=module.head_dim,
        value_head_dim=module.value_head_dim,
        kernel_size=module.kernel_size,
    )

    inputs = jax.random.normal(jax.random.PRNGKey(0), (seq_len, module.model_dim))

    result = module_with_min(inputs, positional_embeddings=None, return_updated_state=True)
    reference = module(inputs, positional_embeddings=None, return_updated_state=True)

    assert_close(result=result.outputs, reference=reference.outputs, atol=1e-5)
    assert result.state is not None
    assert reference.state is not None
    assert_close(result=result.state.ssm_state, reference=reference.state.ssm_state, atol=1e-5)
