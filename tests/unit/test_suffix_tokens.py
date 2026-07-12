import jax
import jax.numpy as jnp
import pytest
from frozendict import frozendict
from jax import Array
from jaxtyping import Int

from lalamo.modules import (
    Decoder,
    DecoderForwardPassConfig,
    DecoderResult,
    EmbeddingForwardPassConfig,
    Keychain,
)
from tests.common import assert_close
from tests.helpers import build_tiny_attention_decoder, make_test_sharding_config

FORWARD_PASS_CONFIG = DecoderForwardPassConfig(
    embedding_forward_pass_config=EmbeddingForwardPassConfig(activation_dtype=jnp.float32),
)


@pytest.fixture
def decoder(request: pytest.FixtureRequest) -> Decoder:
    num_layers, kv_reuse_map = getattr(request, "param", (1, frozendict()))
    return build_tiny_attention_decoder(num_layers, kv_reuse_map)


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
    [(3, frozendict()), (4, frozendict({2: 0, 3: 1}))],
    indirect=True,
    ids=["plain", "trailing_borrowed_kv"],
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
    [(3, frozendict()), (4, frozendict({2: 0, 3: 1}))],
    indirect=True,
    ids=["plain", "trailing_borrowed_kv"],
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


def test_pre_norm_outputs_populated_only_in_suffix_mode(decoder: Decoder) -> None:
    transformer = decoder.transformer
    batch_size, sequence_length = 2, 6
    inner_features = jax.random.normal(
        jax.random.key(2),
        (batch_size, sequence_length, transformer.config.model_dim),
        dtype=jnp.float32,
    )
    token_positions = jnp.broadcast_to(
        jnp.arange(sequence_length, dtype=jnp.int32),
        (batch_size, sequence_length),
    )

    full_result = transformer(
        inner_features,
        token_positions,
        state=None,
        return_updated_state=False,
        return_layer_results=True,
        return_positional_embeddings=False,
        lengths_without_padding=None,
        keychain=Keychain.init(3, sharding_config=make_test_sharding_config()),
    )
    suffix_result = transformer(
        inner_features,
        token_positions,
        state=None,
        return_updated_state=False,
        return_layer_results=False,
        return_positional_embeddings=False,
        lengths_without_padding=None,
        return_suffix_tokens=1,
        keychain=Keychain.init(3, sharding_config=make_test_sharding_config()),
    )

    assert full_result.pre_norm_outputs is None
    assert suffix_result.pre_norm_outputs is not None
    assert full_result.layer_results is not None
    assert_close(
        result=suffix_result.pre_norm_outputs,
        reference=full_result.layer_results[-1].outputs[:, -1:],
        operation_name="pre-norm suffix outputs",
    )
    assert_close(
        result=suffix_result.outputs,
        reference=full_result.outputs[:, -1:],
        operation_name="post-norm suffix outputs",
    )
