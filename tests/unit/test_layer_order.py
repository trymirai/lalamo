from itertools import accumulate, pairwise

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float, Int

from lalamo.initializer import RandomInitializer
from lalamo.modules import Decoder, DecoderForwardPassConfig, EmbeddingForwardPassConfig, Keychain
from lalamo.utils.sharding import ShardingConfig
from tests.helpers import build_tiny_attention_decoder_config

CHUNK_LENGTHS = (5, 1, 4)


def init_decoder(layer_count: int, layer_order: tuple[int, ...] | None) -> Decoder:
    return build_tiny_attention_decoder_config((None,) * layer_count, layer_order=layer_order).init(
        RandomInitializer(
            default_dtype=jnp.float32,
            sharding_config=ShardingConfig.replicated(jax.devices("cpu")[:8]),
            key=jax.random.key(4),
        ),
    )


def looped_and_unrolled_decoders() -> tuple[Decoder, Decoder]:
    looped = init_decoder(2, (0, 0, 1))
    first_layer, second_layer = looped.transformer.layers
    unrolled = eqx.tree_at(
        lambda decoder: (decoder.embedding, decoder.transformer.layers, decoder.transformer.output_norm),
        init_decoder(3, None),
        (looped.embedding, (first_layer, first_layer, second_layer), looped.transformer.output_norm),
    )
    return looped, unrolled


def chunked_logits(decoder: Decoder, token_ids: Int[Array, "batch tokens"]) -> Float[Array, "batch tokens vocabulary"]:
    batch_size, num_tokens = token_ids.shape
    token_positions = jnp.broadcast_to(jnp.arange(num_tokens, dtype=jnp.int32), (batch_size, num_tokens))
    state = decoder.init_static_state(batch_size, num_tokens, jnp.float32)
    chunk_logits = []
    for start, end in pairwise(accumulate(CHUNK_LENGTHS, initial=0)):
        result = decoder(
            token_ids[:, start:end],
            token_positions[:, start:end],
            state=state,
            return_updated_state=True,
            forward_pass_config=DecoderForwardPassConfig(
                embedding_forward_pass_config=EmbeddingForwardPassConfig(activation_dtype=jnp.float32),
            ),
            keychain=Keychain.init(1, sharding_config=decoder.sharding_config),
        )
        state = result.updated_state
        chunk_logits.append(result.logits)
    return jnp.concatenate(chunk_logits, axis=1)


def test_repeated_layers_match_unrolled_copies() -> None:
    looped, unrolled = looped_and_unrolled_decoders()
    token_ids = jax.random.randint(jax.random.key(0), (2, sum(CHUNK_LENGTHS)), 0, looped.vocab_size, dtype=jnp.int32)

    np.testing.assert_array_equal(chunked_logits(looped, token_ids), chunked_logits(unrolled, token_ids))


@pytest.mark.parametrize(
    ("kv_source_layer_indices", "layer_order", "message"),
    [
        ((None, None), (0, 0), "every one of the 2 layers"),
        ((None, None), (0, 2, 1), "every one of the 2 layers"),
        ((None, 0), (0, 1, 1), "KV-sharing"),
    ],
)
def test_invalid_layer_order_is_rejected(
    kv_source_layer_indices: tuple[int | None, ...],
    layer_order: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_tiny_attention_decoder_config(kv_source_layer_indices, layer_order=layer_order)
