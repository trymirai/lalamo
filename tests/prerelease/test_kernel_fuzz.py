import warnings
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import Array, Bool, Float

from lalamo.kernels.attention import pallas_decode_attention, stable_reduction_attention, xla_attention
from lalamo.kernels.deltanet import deltanet_recurrent_scan, xla_recurrent_scan
from lalamo.utils.sharding import ShardingConfig
from tests.common import assert_close, gpu_only

pytestmark = [gpu_only, pytest.mark.slow]


@dataclass(frozen=True)
class DecodeAttentionShape:
    batch_size: int
    key_value_heads: int
    query_heads_per_key_value_head: int
    head_dim: int
    capacity: int
    must_use_pallas: bool


@dataclass(frozen=True)
class DeltaNetShape:
    batch_size: int
    num_tokens: int
    num_heads: int
    key_head_dim: int
    value_head_dim: int
    must_use_pallas: bool


_PALLAS_DECODE_SHAPES = st.sampled_from(
    (
        DecodeAttentionShape(32, 8, 5, 128, 96, must_use_pallas=True),
        DecodeAttentionShape(32, 4, 4, 128, 129, must_use_pallas=True),
        DecodeAttentionShape(16, 4, 16, 128, 385, must_use_pallas=True),
        DecodeAttentionShape(8, 4, 4, 256, 897, must_use_pallas=True),
        DecodeAttentionShape(32, 8, 2, 256, 127, must_use_pallas=True),
        DecodeAttentionShape(64, 1, 16, 512, 385, must_use_pallas=True),
        DecodeAttentionShape(16, 3, 14, 128, 385, must_use_pallas=True),
        DecodeAttentionShape(16, 3, 14, 96, 385, must_use_pallas=False),
        DecodeAttentionShape(8, 5, 7, 368, 257, must_use_pallas=False),
        DecodeAttentionShape(16, 2, 17, 128, 385, must_use_pallas=False),
    ),
)

_DELTANET_SHAPES = st.sampled_from(
    (
        DeltaNetShape(1, 1, 1, 128, 64, must_use_pallas=True),
        DeltaNetShape(8, 1, 2, 128, 128, must_use_pallas=True),
        DeltaNetShape(14, 1, 3, 128, 192, must_use_pallas=True),
        DeltaNetShape(4, 1, 14, 128, 64, must_use_pallas=True),
        DeltaNetShape(4, 1, 2, 96, 64, must_use_pallas=False),
        DeltaNetShape(2, 1, 1, 368, 64, must_use_pallas=False),
        DeltaNetShape(3, 1, 2, 128, 368, must_use_pallas=False),
        DeltaNetShape(4, 3, 2, 128, 64, must_use_pallas=False),
    ),
)


_batched_pallas_decode_attention = jax.jit(jax.vmap(pallas_decode_attention, in_axes=(0, 0, 0, None, 0, None, None)))
_batched_xla_attention = jax.jit(jax.vmap(xla_attention, in_axes=(0, 0, 0, None, 0, None, None)))


@partial(jax.jit, static_argnames=("tile_size",))
def _batched_stable_reduction_attention(
    queries: Float[Array, "batch 1 query_heads head_dim"],
    keys: Float[Array, "batch capacity key_value_heads head_dim"],
    values: Float[Array, "batch capacity key_value_heads head_dim"],
    masks: Bool[Array, "batch 1 capacity"],
    scale: Float[Array, ""],
    tile_size: int,
) -> Float[Array, "batch 1 query_heads head_dim"]:
    return jax.vmap(
        lambda query, key, value, mask: stable_reduction_attention(
            query,
            key,
            value,
            bias=None,
            mask=mask,
            scale=scale,
            logit_soft_cap=None,
            tile_size=tile_size,
            accumulation_dtype=jnp.float32,
        )
    )(queries, keys, values, masks)


_batched_pallas_deltanet = jax.jit(jax.vmap(deltanet_recurrent_scan))
_batched_xla_deltanet = jax.jit(jax.vmap(xla_recurrent_scan))


@settings(max_examples=1_000, deadline=None)
@given(
    shape=_PALLAS_DECODE_SHAPES,
    seed=st.integers(min_value=0, max_value=2**31 - 5),
    scale=st.one_of(
        st.none(),
        st.floats(
            min_value=0.01,
            max_value=2.0,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
            width=32,
        ),
    ),
    window_divisor=st.one_of(st.none(), st.integers(min_value=1, max_value=16)),
    hole_start_fraction=st.integers(min_value=0, max_value=15),
    hole_size=st.integers(min_value=0, max_value=16),
    tile_size=st.sampled_from((32, 96, 127, 128, 256)),
)
def test_attention_implementations_match_xla(
    shape: DecodeAttentionShape,
    seed: int,
    scale: float | None,
    window_divisor: int | None,
    hole_start_fraction: int,
    hole_size: int,
    tile_size: int,
) -> None:
    sharding_config = ShardingConfig.replicated()
    num_query_heads = shape.key_value_heads * shape.query_heads_per_key_value_head
    queries = jax.device_put(
        (
            jax.random.normal(
                jax.random.key(seed),
                (shape.batch_size, 1, num_query_heads, shape.head_dim),
                dtype=jnp.float32,
            )
            * 0.2
        ).astype(jnp.bfloat16),
        sharding_config.make_sharding((None, None, None, None)),
    )
    keys = jax.device_put(
        (
            jax.random.normal(
                jax.random.key(seed + 1),
                (shape.batch_size, shape.capacity, shape.key_value_heads, shape.head_dim),
                dtype=jnp.float32,
            )
            * 0.1
        ).astype(jnp.bfloat16),
        sharding_config.make_sharding((None, None, None, None)),
    )
    values = jax.device_put(
        (
            jax.random.normal(
                jax.random.key(seed + 2),
                (shape.batch_size, shape.capacity, shape.key_value_heads, shape.head_dim),
                dtype=jnp.float32,
            )
            * 0.1
        ).astype(jnp.bfloat16),
        sharding_config.make_sharding((None, None, None, None)),
    )
    lengths = jax.random.randint(
        jax.random.key(seed + 3),
        (shape.batch_size,),
        minval=1,
        maxval=shape.capacity + 1,
    )
    if window_divisor is None:
        starts = jnp.zeros_like(lengths)
    else:
        window_size = max(shape.capacity // window_divisor, 1)
        starts = jnp.maximum(lengths - window_size, 0)
    token_indices = jnp.arange(shape.capacity)
    masks = (token_indices >= starts[:, None]) & (token_indices < lengths[:, None])
    hole_start = hole_start_fraction * shape.capacity // 16
    hole_end = min(hole_start + hole_size, shape.capacity)
    masks = masks.at[:, hole_start:hole_end].set(False)
    masks = masks.at[jnp.arange(shape.batch_size), lengths - 1].set(True)
    masks = jax.device_put(
        masks[:, None],
        sharding_config.make_sharding((None, None, None)),
    )
    attention_scale = jnp.asarray(shape.head_dim**-0.5 if scale is None else scale, dtype=jnp.float32)

    with warnings.catch_warnings():
        if shape.must_use_pallas:
            warnings.filterwarnings(
                "error",
                message=r"Pallas decode attention .*falling back to XLA attention\.",
                category=RuntimeWarning,
            )
        result = _batched_pallas_decode_attention(
            queries,
            keys,
            values,
            None,
            masks,
            attention_scale,
            None,
        )
    stable_reduction = _batched_stable_reduction_attention(
        queries,
        keys,
        values,
        masks,
        attention_scale,
        tile_size,
    )
    reference = _batched_xla_attention(
        queries,
        keys,
        values,
        None,
        masks,
        attention_scale,
        None,
    )

    assert_close(result=result, reference=reference, atol=2e-2, rtol=3e-2)
    assert_close(result=stable_reduction, reference=reference, atol=2e-2, rtol=3e-2)


@settings(max_examples=1_000, deadline=None)
@given(
    shape=_DELTANET_SHAPES,
    seed=st.integers(min_value=0, max_value=2**31 - 7),
)
def test_deltanet_recurrence_matches_xla(shape: DeltaNetShape, seed: int) -> None:
    sharding_config = ShardingConfig.replicated()
    queries = jax.device_put(
        jax.random.normal(
            jax.random.key(seed),
            (shape.batch_size, shape.num_tokens, shape.num_heads, shape.key_head_dim),
            dtype=jnp.float32,
        )
        * 0.05,
        sharding_config.make_sharding((None, None, None, None)),
    )
    keys = jax.device_put(
        jax.random.normal(
            jax.random.key(seed + 1),
            (shape.batch_size, shape.num_tokens, shape.num_heads, shape.key_head_dim),
            dtype=jnp.float32,
        )
        * 0.05,
        sharding_config.make_sharding((None, None, None, None)),
    )
    values = jax.device_put(
        jax.random.normal(
            jax.random.key(seed + 2),
            (shape.batch_size, shape.num_tokens, shape.num_heads, shape.value_head_dim),
            dtype=jnp.float32,
        )
        * 0.1,
        sharding_config.make_sharding((None, None, None, None)),
    )
    decay_factor = jax.device_put(
        -jax.nn.softplus(
            jax.random.normal(
                jax.random.key(seed + 3),
                (shape.batch_size, shape.num_tokens, shape.num_heads),
                dtype=jnp.float32,
            )
            * 0.5
        ),
        sharding_config.make_sharding((None, None, None)),
    )
    beta = jax.device_put(
        jax.nn.sigmoid(
            jax.random.normal(
                jax.random.key(seed + 4),
                (shape.batch_size, shape.num_tokens, shape.num_heads),
                dtype=jnp.float32,
            )
            * 0.5
        ),
        sharding_config.make_sharding((None, None, None)),
    )
    initial_state = jax.device_put(
        jax.random.normal(
            jax.random.key(seed + 5),
            (shape.batch_size, shape.num_heads, shape.value_head_dim, shape.key_head_dim),
            dtype=jnp.float32,
        )
        * 0.02,
        sharding_config.make_sharding((None, None, None, None)),
    )
    num_steps = jax.device_put(
        jax.random.randint(
            jax.random.key(seed + 6),
            (shape.batch_size,),
            minval=0,
            maxval=shape.num_tokens + 1,
            dtype=jnp.int32,
        ),
        sharding_config.make_sharding((None,)),
    )

    with warnings.catch_warnings():
        if shape.must_use_pallas:
            warnings.filterwarnings(
                "error",
                message=r"Pallas DeltaNet recurrence .*falling back to XLA recurrence\.",
                category=RuntimeWarning,
            )
        outputs, final_state = _batched_pallas_deltanet(
            queries,
            keys,
            values,
            decay_factor,
            beta,
            initial_state,
            num_steps,
        )
    reference_outputs, reference_state = _batched_xla_deltanet(
        queries,
        keys,
        values,
        decay_factor,
        beta,
        initial_state,
        num_steps,
    )

    assert_close(result=outputs, reference=reference_outputs, atol=5e-2, rtol=1e-1)
    assert_close(result=final_state, reference=reference_state, atol=1e-3, rtol=3e-2)
