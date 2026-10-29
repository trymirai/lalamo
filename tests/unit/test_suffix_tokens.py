import jax
import jax.numpy as jnp
import pytest
from jax import Array
from jaxtyping import Int

from lalamo.initializer import RandomInitializer
from lalamo.modules import (
    Decoder,
    DecoderResult,
    EmbeddingForwardPassConfig,
    Keychain,
    ModelForwardPassConfig,
)
from lalamo.modules.flow import Context, EmitConfig, FlowForwardPassConfig, SequenceConfig, residual, run
from lalamo.utils.sharding import ShardingConfig
from tests.common import assert_close
from tests.helpers import (
    build_tiny_attention_decoder,
    make_test_sharding_config,
    tiny_attention_config,
    tiny_mlp_config,
    tiny_norm_config,
)

FORWARD_PASS_CONFIG = ModelForwardPassConfig(
    embedding_forward_pass_config=EmbeddingForwardPassConfig(activation_dtype=jnp.float32),
)


@pytest.fixture
def decoder(request: pytest.FixtureRequest) -> Decoder:
    kv_source_layer_indices: tuple[int | None, ...] = getattr(request, "param", (None,))
    return build_tiny_attention_decoder(kv_source_layer_indices)


def run_decoder(
    decoder: Decoder,
    token_ids: Int[Array, "batch tokens"],
    lengths_without_padding: Int[Array, " batch"],
    return_suffix_tokens: int | None = None,
) -> DecoderResult:
    batch_size, sequence_length = token_ids.shape
    token_positions = jnp.broadcast_to(
        jnp.arange(sequence_length, dtype=jnp.int32),
        (batch_size, sequence_length),
    )
    state = decoder.init_static_state(batch_size, 16, jnp.float32)
    return decoder(
        token_ids,
        token_positions,
        state=state,
        return_updated_state=True,
        lengths_without_padding=lengths_without_padding,
        forward_pass_config=FORWARD_PASS_CONFIG,
        return_suffix_tokens=return_suffix_tokens,
        keychain=Keychain.init(1, sharding_config=make_test_sharding_config()),
    )


@pytest.mark.parametrize("num_suffix_tokens", [1, 3])
@pytest.mark.parametrize(
    "decoder",
    [(), (None, None, None), (None, None, 0, 1)],
    indirect=True,
    ids=["no_temporal_steps", "plain", "trailing_shared_kv"],
)
def test_suffix_logits_match_full_pass(decoder: Decoder, num_suffix_tokens: int) -> None:
    batch_size, sequence_length = 3, 10
    lengths_without_padding = jnp.array([10, 7, 2], dtype=jnp.int32)
    token_ids = jax.random.randint(
        jax.random.key(0),
        (batch_size, sequence_length),
        0,
        decoder.vocab_size,
        dtype=jnp.int32,
    )

    full_result = run_decoder(decoder, token_ids, lengths_without_padding)
    suffix_result = run_decoder(decoder, token_ids, lengths_without_padding, return_suffix_tokens=num_suffix_tokens)

    assert suffix_result.logits.shape == (batch_size, num_suffix_tokens, decoder.vocab_size)
    for row, row_length in enumerate(lengths_without_padding.tolist()):
        num_valid_tokens = min(num_suffix_tokens, row_length)
        assert_close(
            result=suffix_result.logits[row, num_suffix_tokens - num_valid_tokens :],
            reference=full_result.logits[row, row_length - num_valid_tokens : row_length],
            operation_name=f"suffix logits, row {row}",
        )

    assert full_result.updated_state is not None
    assert suffix_result.updated_state is not None
    jax.tree.map(
        lambda suffix_leaf, full_leaf: assert_close(
            result=suffix_leaf,
            reference=full_leaf,
            operation_name="updated state",
        ),
        suffix_result.updated_state,
        full_result.updated_state,
    )


@pytest.mark.parametrize(
    "decoder",
    [(None, None, None), (None, None, 0, 1)],
    indirect=True,
    ids=["plain", "trailing_shared_kv"],
)
def test_suffix_logits_across_chunk_boundary(decoder: Decoder) -> None:
    batch_size, sequence_length, chunk_size = 3, 10, 5
    lengths_without_padding = jnp.array([10, 7, 2], dtype=jnp.int32)
    token_ids = jax.random.randint(
        jax.random.key(0),
        (batch_size, sequence_length),
        0,
        decoder.vocab_size,
        dtype=jnp.int32,
    )

    full_result = run_decoder(decoder, token_ids, lengths_without_padding)

    state = decoder.init_static_state(batch_size, 16, jnp.float32)
    chunk_logits = []
    for chunk_start in range(0, sequence_length, chunk_size):
        chunk_token_ids = token_ids[:, chunk_start : chunk_start + chunk_size]
        chunk_positions = jnp.broadcast_to(
            jnp.arange(chunk_start, chunk_start + chunk_size, dtype=jnp.int32),
            (batch_size, chunk_size),
        )
        sequence_ends = jnp.clip(lengths_without_padding - chunk_start, 0, chunk_size)
        chunk_result = decoder(
            chunk_token_ids,
            chunk_positions,
            state=state,
            return_updated_state=True,
            lengths_without_padding=sequence_ends,
            forward_pass_config=FORWARD_PASS_CONFIG,
            return_suffix_tokens=1,
            keychain=Keychain.init(1, sharding_config=make_test_sharding_config()),
        )
        assert chunk_result.updated_state is not None
        state = chunk_result.updated_state
        chunk_logits.append(chunk_result.logits[:, 0])

    for row, row_length in enumerate(lengths_without_padding.tolist()):
        last_token_chunk = (row_length - 1) // chunk_size
        assert_close(
            result=chunk_logits[last_token_chunk][row],
            reference=full_result.logits[row, row_length - 1],
            operation_name=f"chunked suffix logits, row {row}",
        )


@pytest.mark.parametrize("has_attention", [False, True])
def test_taps_observe_suffix_outputs(*, has_attention: bool) -> None:
    hidden_states = EmitConfig("hidden_states")
    sharding_config = ShardingConfig.replicated(jax.devices("cpu")[:8])
    program = SequenceConfig(
        (
            *((residual(tiny_norm_config(), tiny_attention_config()),) if has_attention else ()),
            residual(tiny_norm_config(), tiny_mlp_config()),
            hidden_states,
            tiny_norm_config(),
        ),
    ).init(RandomInitializer(jnp.float32, sharding_config, key=jax.random.key(4)), model_dim=8)
    batch_size, sequence_length = 2, 6
    inner_features = jax.random.normal(jax.random.key(2), (batch_size, sequence_length, 8), dtype=jnp.float32)
    token_positions = jnp.broadcast_to(jnp.arange(sequence_length, dtype=jnp.int32), (batch_size, sequence_length))
    context = Context(None, token_positions, None, None, FlowForwardPassConfig(), sharding_config)
    keychain = Keychain.init(3, sharding_config=sharding_config)

    full_result = run(program, inner_features, context, keychain=keychain, taps=(hidden_states,), trace=True)
    suffix_result = run(program, inner_features, context, keychain=keychain, taps=(hidden_states,), suffix=1)

    assert full_result.trace is not None
    *_, pre_norm_step, _ = full_result.trace
    (full_hidden_states,) = full_result.taps
    (suffix_hidden_states,) = suffix_result.taps
    assert_close(result=full_hidden_states, reference=pre_norm_step.outputs, operation_name="tap outputs")
    assert_close(
        result=suffix_hidden_states,
        reference=full_hidden_states[:, -1:],
        operation_name="pre-norm suffix outputs",
    )
    assert_close(
        result=suffix_result.outputs,
        reference=full_result.outputs[:, -1:],
        operation_name="post-norm suffix outputs",
    )


@pytest.mark.parametrize("with_prefix", [False, True])
def test_shared_kv_suffix_matches_full_with_dynamic_padding(*, with_prefix: bool) -> None:
    decoder = build_tiny_attention_decoder((None, None, 0, 1))
    sharding = decoder.sharding_config
    keychain = Keychain.init(1, sharding_config=sharding)
    tokens = jax.random.randint(jax.random.key(0), (2, 10), 0, decoder.vocab_size)
    positions = jnp.broadcast_to(jnp.arange(10, dtype=jnp.int32), tokens.shape)
    state = None
    if with_prefix:
        prefix = decoder(
            tokens[:, :4],
            positions[:, :4],
            return_updated_state=True,
            lengths_without_padding=jnp.array([4, 3]),
            forward_pass_config=FORWARD_PASS_CONFIG,
            keychain=keychain,
        )
        state = prefix.updated_state
        assert state is not None
    tokens, positions = tokens[:, 4:], positions[:, 4:]
    lengths = jnp.array([6, 3])
    full = decoder(
        tokens,
        positions,
        state,
        lengths_without_padding=lengths,
        forward_pass_config=FORWARD_PASS_CONFIG,
        keychain=keychain,
    )
    suffix = decoder(
        tokens,
        positions,
        state,
        lengths_without_padding=lengths,
        return_suffix_tokens=2,
        forward_pass_config=FORWARD_PASS_CONFIG,
        keychain=keychain,
    )
    for row, length in enumerate(lengths.tolist()):
        assert_close(
            result=suffix.logits[row],
            reference=full.logits[row, length - 2 : length],
            operation_name="shared-KV dynamic suffix",
        )
