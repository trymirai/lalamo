import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from einops import rearrange
from jaxtyping import Array, Float

from lalamo.initializer import RandomInitializer
from lalamo.modules.normalization import Normalization, NormalizationConfig, UpcastMode
from lalamo.modules.rope import PositionalEmbeddings, UnscaledRoPEConfig
from lalamo.modules.token_mixers.indexer_cache import (
    DynamicIndexerCacheLayer,
    IndexerCacheLayer,
    StaticIndexerCacheLayer,
)
from lalamo.modules.utils import call_vmapped
from tests.common import assert_close
from tests.helpers import make_test_sharding_config

COMPRESSION_RATIO = 4
HEAD_DIM = 16
ROPE_DIM = 8
NUM_TOKENS = 13
POSITION_OFFSET = 7


def make_key_norm() -> Normalization:
    config = NormalizationConfig(
        epsilon=1e-6,
        scale_offset=1.0,
        upcast_mode=UpcastMode.FULL_LAYER,
        subtract_mean=False,
    )
    key_norm = config.init(
        RandomInitializer(
            default_dtype=jnp.float32,
            sharding_config=make_test_sharding_config(),
            key=jax.random.key(0),
        ),
        input_dim=HEAD_DIM,
    )
    return eqx.tree_at(
        lambda norm: norm.scales,
        key_norm,
        jax.random.normal(jax.random.key(1), (HEAD_DIM,), dtype=jnp.float32) * 0.1,
    )


def make_inputs() -> tuple[Float[Array, "tokens head_channels"], PositionalEmbeddings]:
    keys = jax.random.normal(jax.random.key(2), (NUM_TOKENS, HEAD_DIM), dtype=jnp.float32)
    rope_config = UnscaledRoPEConfig(base=10000.0, max_sequence_length=64, head_dim=ROPE_DIM)
    embeddings = rope_config.compute_positional_embeddings(jnp.arange(NUM_TOKENS) + POSITION_OFFSET)
    return keys, embeddings


def reference_compressed_keys(
    keys: Float[Array, "tokens head_channels"],
    embeddings: PositionalEmbeddings,
    key_norm: Normalization,
) -> Float[Array, "blocks head_channels"]:
    num_blocks = NUM_TOKENS // COMPRESSION_RATIO
    grouped_keys = rearrange(
        keys[: num_blocks * COMPRESSION_RATIO],
        "(blocks ratio) head_channels -> blocks ratio head_channels",
        ratio=COMPRESSION_RATIO,
    )
    block_start_embeddings = PositionalEmbeddings(
        cosines=embeddings.cosines[::COMPRESSION_RATIO][:num_blocks],
        sines=embeddings.sines[::COMPRESSION_RATIO][:num_blocks],
    )
    return block_start_embeddings.apply(call_vmapped(key_norm, jnp.mean(grouped_keys, axis=1)))


@eqx.filter_jit
def fill(
    cache: IndexerCacheLayer,
    keys: Float[Array, "tokens head_channels"],
    embeddings: PositionalEmbeddings,
    key_norm: Normalization,
    chunk_lengths: tuple[int, ...],
    num_padding_tokens: int,
) -> IndexerCacheLayer:
    garbage_key = jax.random.key(3)
    start = 0
    for length in chunk_lengths:
        garbage_key, keys_key, rotations_key = jax.random.split(garbage_key, 3)
        padded_keys = jnp.concatenate(
            [keys[start : start + length], jax.random.normal(keys_key, (num_padding_tokens, HEAD_DIM))],
        )
        garbage_rotations = jax.random.normal(rotations_key, (num_padding_tokens, ROPE_DIM))
        padded_embeddings = PositionalEmbeddings(
            cosines=jnp.concatenate([embeddings.cosines[start : start + length], garbage_rotations]),
            sines=jnp.concatenate([embeddings.sines[start : start + length], garbage_rotations]),
        )
        cache = cache.extend(padded_keys, padded_embeddings, key_norm, added_length=jnp.array(length))
        start += length
    return cache


@pytest.mark.parametrize(
    "chunk_lengths",
    [
        pytest.param((13,), id="single-chunk"),
        pytest.param((1,) * 13, id="token-by-token"),
        pytest.param((5, 1, 3, 4), id="straddling-blocks"),
        pytest.param((4, 4, 5), id="block-aligned-chunks"),
        pytest.param((2, 11), id="short-then-long"),
    ],
)
@pytest.mark.parametrize("num_padding_tokens", [0, 3])
@pytest.mark.parametrize("cache_kind", ["dynamic", "static"])
@pytest.mark.usefixtures("fake_mesh")
def test_incremental_extend_matches_reference(
    chunk_lengths: tuple[int, ...],
    num_padding_tokens: int,
    cache_kind: str,
) -> None:
    key_norm = make_key_norm()
    keys, embeddings = make_inputs()
    if cache_kind == "dynamic":
        empty_cache = DynamicIndexerCacheLayer.empty(COMPRESSION_RATIO, HEAD_DIM, jnp.float32)
    else:
        empty_cache = StaticIndexerCacheLayer.init(
            capacity=16,
            compression_ratio=COMPRESSION_RATIO,
            head_dim=HEAD_DIM,
            dtype=jnp.float32,
            sharding_config=make_test_sharding_config(),
        )

    cache = fill(empty_cache, keys, embeddings, key_norm, chunk_lengths, num_padding_tokens)

    num_blocks = NUM_TOKENS // COMPRESSION_RATIO
    tail_start = NUM_TOKENS - (COMPRESSION_RATIO - 1)
    assert int(cache.num_tokens) == NUM_TOKENS
    assert_close(
        result=cache.compressed_keys[:num_blocks],
        reference=reference_compressed_keys(keys, embeddings, key_norm),
    )
    assert_close(result=cache.tail_keys, reference=keys[tail_start:])
    assert_close(result=cache.tail_cosines[:, :ROPE_DIM], reference=embeddings.cosines[tail_start:])
    assert_close(result=cache.tail_sines[:, :ROPE_DIM], reference=embeddings.sines[tail_start:])
