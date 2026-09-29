import warnings
from dataclasses import dataclass
from enum import StrEnum
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
    query_tokens: int
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


class AttentionMaskPattern(StrEnum):
    EMPTY = "empty"
    FULL = "full"
    SINGLETON = "singleton"
    PREFIX = "prefix"
    WINDOW = "window"
    HOLES = "holes"
    ALTERNATING = "alternating"


class BetaPattern(StrEnum):
    ZERO = "zero"
    ONE = "one"
    NEAR_ZERO = "near_zero"
    NEAR_ONE = "near_one"
    RANDOM = "random"


class NumStepsPattern(StrEnum):
    SHARED = "shared"
    BATCHED = "batched"


_PALLAS_DECODE_SHAPES = st.sampled_from(
    (
        DecodeAttentionShape(18, 1, 8, 1, 128, 1, must_use_pallas=True),
        DecodeAttentionShape(18, 1, 8, 14, 256, 127, must_use_pallas=True),
        DecodeAttentionShape(18, 1, 8, 16, 512, 128, must_use_pallas=True),
        DecodeAttentionShape(9, 1, 8, 1, 128, 129, must_use_pallas=True),
        DecodeAttentionShape(9, 1, 8, 14, 256, 255, must_use_pallas=True),
        DecodeAttentionShape(9, 1, 8, 16, 512, 256, must_use_pallas=True),
        DecodeAttentionShape(9, 1, 8, 7, 128, 257, must_use_pallas=True),
        DecodeAttentionShape(6, 1, 8, 1, 128, 385, must_use_pallas=True),
        DecodeAttentionShape(6, 1, 8, 14, 256, 511, must_use_pallas=True),
        DecodeAttentionShape(6, 1, 8, 16, 512, 512, must_use_pallas=True),
        DecodeAttentionShape(4, 1, 8, 1, 128, 897, must_use_pallas=True),
        DecodeAttentionShape(4, 1, 8, 14, 256, 1024, must_use_pallas=True),
        DecodeAttentionShape(16, 1, 3, 14, 128, 385, must_use_pallas=True),
        DecodeAttentionShape(16, 1, 3, 14, 96, 385, must_use_pallas=False),
        DecodeAttentionShape(8, 1, 5, 7, 368, 257, must_use_pallas=False),
        DecodeAttentionShape(16, 1, 2, 17, 128, 385, must_use_pallas=False),
        DecodeAttentionShape(8, 2, 4, 14, 128, 257, must_use_pallas=False),
        DecodeAttentionShape(4, 4, 7, 1, 256, 129, must_use_pallas=False),
        DecodeAttentionShape(1, 1, 1, 14, 128, 1024, must_use_pallas=False),
        DecodeAttentionShape(8, 1, 4, 16, 512, 128, must_use_pallas=False),
    ),
)

_DELTANET_SHAPES = st.sampled_from(
    (
        DeltaNetShape(1, 1, 1, 128, 64, must_use_pallas=True),
        DeltaNetShape(2, 1, 7, 128, 128, must_use_pallas=True),
        DeltaNetShape(4, 1, 14, 128, 192, must_use_pallas=True),
        DeltaNetShape(8, 1, 1, 128, 256, must_use_pallas=True),
        DeltaNetShape(14, 1, 7, 128, 320, must_use_pallas=True),
        DeltaNetShape(2, 1, 14, 128, 512, must_use_pallas=True),
        DeltaNetShape(2, 1, 1, 14, 64, must_use_pallas=False),
        DeltaNetShape(2, 1, 7, 63, 64, must_use_pallas=False),
        DeltaNetShape(2, 1, 14, 65, 64, must_use_pallas=False),
        DeltaNetShape(2, 1, 1, 96, 64, must_use_pallas=False),
        DeltaNetShape(2, 1, 7, 127, 64, must_use_pallas=False),
        DeltaNetShape(2, 1, 14, 129, 64, must_use_pallas=False),
        DeltaNetShape(2, 1, 1, 368, 64, must_use_pallas=False),
        DeltaNetShape(2, 1, 7, 128, 14, must_use_pallas=False),
        DeltaNetShape(2, 1, 14, 128, 63, must_use_pallas=False),
        DeltaNetShape(2, 1, 1, 128, 65, must_use_pallas=False),
        DeltaNetShape(2, 1, 7, 128, 96, must_use_pallas=False),
        DeltaNetShape(2, 1, 14, 128, 127, must_use_pallas=False),
        DeltaNetShape(2, 1, 1, 128, 129, must_use_pallas=False),
        DeltaNetShape(2, 1, 7, 128, 368, must_use_pallas=False),
        DeltaNetShape(2, 2, 1, 128, 64, must_use_pallas=False),
        DeltaNetShape(2, 3, 7, 128, 64, must_use_pallas=False),
        DeltaNetShape(2, 14, 14, 128, 64, must_use_pallas=False),
    ),
)


_batched_pallas_decode_attention = jax.jit(jax.vmap(pallas_decode_attention, in_axes=(0, 0, 0, None, 0, None, None)))
_batched_xla_attention = jax.jit(jax.vmap(xla_attention, in_axes=(0, 0, 0, None, 0, None, None)))


@partial(jax.jit, static_argnames=("tile_size",))
def _batched_stable_reduction_attention(
    queries: Float[Array, "batch query_tokens query_heads head_dim"],
    keys: Float[Array, "batch capacity key_value_heads head_dim"],
    values: Float[Array, "batch capacity key_value_heads head_dim"],
    bias: Float[Array, "query_heads query_tokens capacity"] | None,
    masks: Bool[Array, "batch query_tokens capacity"],
    scale: float | Float[Array, ""] | None,
    logit_soft_cap: float | Float[Array, ""] | None,
    tile_size: int,
) -> Float[Array, "batch query_tokens query_heads head_dim"]:
    return jax.vmap(
        lambda query, key, value, mask: stable_reduction_attention(
            query,
            key,
            value,
            bias=bias,
            mask=mask,
            scale=scale,
            logit_soft_cap=logit_soft_cap,
            tile_size=tile_size,
            accumulation_dtype=jnp.float32,
        )
    )(queries, keys, values, masks)


_batched_steps_pallas_deltanet = jax.jit(jax.vmap(deltanet_recurrent_scan))
_batched_steps_xla_deltanet = jax.jit(jax.vmap(xla_recurrent_scan))
_shared_steps_pallas_deltanet = jax.jit(jax.vmap(deltanet_recurrent_scan, in_axes=(0, 0, 0, 0, 0, 0, None)))
_shared_steps_xla_deltanet = jax.jit(jax.vmap(xla_recurrent_scan, in_axes=(0, 0, 0, 0, 0, 0, None)))


@settings(max_examples=2_000, deadline=None)
@given(
    shape=_PALLAS_DECODE_SHAPES,
    seed=st.integers(min_value=0, max_value=2**31 - 5),
    mask_pattern=st.sampled_from(tuple(AttentionMaskPattern)),
    scale=st.one_of(
        st.none(),
        st.sampled_from((-2.0, -1.0, -0.015625, 0.0, 0.015625, 1.0, 2.0)),
        st.floats(
            min_value=-2.0,
            max_value=2.0,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
            width=32,
        ),
    ),
    bias_magnitude=st.one_of(st.none(), st.sampled_from((0.0, 0.015625, 0.25, 1.0, 20.0))),
    logit_soft_cap=st.one_of(
        st.none(),
        st.sampled_from((0.015625, 0.25, 1.0, 20.0)),
        st.floats(
            min_value=0.015625,
            max_value=20.0,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
            width=32,
        ),
    ),
    query_key_magnitude=st.sampled_from((0.0, 0.015625, 0.1, 0.25, 1.0)),
    value_magnitude=st.sampled_from((0.0, 0.015625, 0.1, 0.25, 1.0, 4.0)),
    tile_size=st.sampled_from((1, 14, 31, 32, 127, 128, 129, 368, 1024)),
)
def test_attention_implementations_match_xla(
    shape: DecodeAttentionShape,
    seed: int,
    mask_pattern: AttentionMaskPattern,
    scale: float | None,
    bias_magnitude: float | None,
    logit_soft_cap: float | None,
    query_key_magnitude: float,
    value_magnitude: float,
    tile_size: int,
) -> None:
    sharding_config = ShardingConfig.replicated()
    num_query_heads = shape.key_value_heads * shape.query_heads_per_key_value_head
    queries = jax.device_put(
        (
            jax.random.normal(
                jax.random.key(seed),
                (shape.batch_size, shape.query_tokens, num_query_heads, shape.head_dim),
                dtype=jnp.float32,
            )
            * query_key_magnitude
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
            * query_key_magnitude
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
            * value_magnitude
        ).astype(jnp.bfloat16),
        sharding_config.make_sharding((None, None, None, None)),
    )
    token_indices = jnp.arange(shape.capacity)
    mask_shape = (shape.batch_size, shape.query_tokens, shape.capacity)
    if mask_pattern is AttentionMaskPattern.EMPTY:
        masks = jnp.zeros(mask_shape, dtype=jnp.bool_)
    elif mask_pattern is AttentionMaskPattern.FULL:
        masks = jnp.ones(mask_shape, dtype=jnp.bool_)
    elif mask_pattern is AttentionMaskPattern.SINGLETON:
        selected_tokens = jax.random.randint(
            jax.random.key(seed + 3),
            (shape.batch_size, shape.query_tokens, 1),
            minval=0,
            maxval=shape.capacity,
        )
        masks = token_indices == selected_tokens
    elif mask_pattern is AttentionMaskPattern.PREFIX:
        lengths = jax.random.randint(
            jax.random.key(seed + 3),
            (shape.batch_size, shape.query_tokens, 1),
            minval=1,
            maxval=shape.capacity + 1,
        )
        masks = token_indices < lengths
    elif mask_pattern is AttentionMaskPattern.WINDOW:
        ends = jax.random.randint(
            jax.random.key(seed + 3),
            (shape.batch_size, shape.query_tokens, 1),
            minval=1,
            maxval=shape.capacity + 1,
        )
        widths = jax.random.randint(
            jax.random.key(seed + 4),
            (shape.batch_size, shape.query_tokens, 1),
            minval=1,
            maxval=shape.capacity + 1,
        )
        starts = jnp.maximum(ends - widths, 0)
        masks = (token_indices >= starts) & (token_indices < ends)
    elif mask_pattern is AttentionMaskPattern.HOLES:
        masks = jnp.broadcast_to((token_indices + seed) % 14 != 0, mask_shape)
    else:
        batch_indices = jnp.arange(shape.batch_size)[:, None, None]
        query_indices = jnp.arange(shape.query_tokens)[None, :, None]
        masks = (token_indices + batch_indices + query_indices + seed) % 2 == 0
    if mask_pattern is not AttentionMaskPattern.EMPTY:
        guaranteed_tokens = (
            jnp.arange(shape.batch_size)[:, None] + jnp.arange(shape.query_tokens) + seed
        ) % shape.capacity
        masks = masks.at[
            jnp.arange(shape.batch_size)[:, None],
            jnp.arange(shape.query_tokens)[None, :],
            guaranteed_tokens,
        ].set(True)
    masks = jax.device_put(
        masks,
        sharding_config.make_sharding((None, None, None)),
    )
    if bias_magnitude is None:
        bias = None
    else:
        bias = (
            jax.random.normal(
                jax.random.key(seed + 5),
                (num_query_heads, shape.query_tokens, shape.capacity),
                dtype=jnp.float32,
            )
            * bias_magnitude
        )
    if scale is None:
        attention_scale = None
    else:
        attention_scale = jnp.asarray(scale, dtype=jnp.float32)

    with warnings.catch_warnings():
        if shape.must_use_pallas and bias is None and logit_soft_cap is None:
            warnings.filterwarnings(
                "error",
                message=r"Pallas decode attention .*falling back to XLA attention\.",
                category=RuntimeWarning,
            )
        result = _batched_pallas_decode_attention(
            queries,
            keys,
            values,
            bias,
            masks,
            attention_scale,
            logit_soft_cap,
        )
    stable_reduction = _batched_stable_reduction_attention(
        queries,
        keys,
        values,
        bias,
        masks,
        attention_scale,
        logit_soft_cap,
        tile_size,
    )
    reference = _batched_xla_attention(
        queries,
        keys,
        values,
        bias,
        masks,
        attention_scale,
        logit_soft_cap,
    )

    assert_close(result=result, reference=reference, atol=2e-2, rtol=3e-2, operation_name="Pallas attention")
    assert_close(
        result=stable_reduction,
        reference=reference,
        atol=2e-2,
        rtol=3e-2,
        operation_name="stable-reduction attention",
    )


@settings(max_examples=2_000, deadline=None)
@given(
    shape=_DELTANET_SHAPES,
    seed=st.integers(min_value=0, max_value=2**31 - 7),
    input_magnitude=st.sampled_from((0.0, 0.015625, 0.25, 1.0, 4.0)),
    state_magnitude=st.sampled_from((0.0, 0.015625, 0.25, 1.0, 4.0)),
    decay_magnitude=st.sampled_from((0.0, 0.015625, 1.0, 20.0, 100.0)),
    beta_pattern=st.sampled_from(tuple(BetaPattern)),
    num_steps_pattern=st.sampled_from(tuple(NumStepsPattern)),
)
def test_deltanet_recurrence_matches_xla(
    shape: DeltaNetShape,
    seed: int,
    input_magnitude: float,
    state_magnitude: float,
    decay_magnitude: float,
    beta_pattern: BetaPattern,
    num_steps_pattern: NumStepsPattern,
) -> None:
    sharding_config = ShardingConfig.replicated()
    queries = jax.device_put(
        jax.random.normal(
            jax.random.key(seed),
            (shape.batch_size, shape.num_tokens, shape.num_heads, shape.key_head_dim),
            dtype=jnp.float32,
        )
        * input_magnitude,
        sharding_config.make_sharding((None, None, None, None)),
    )
    keys = jax.device_put(
        jax.random.normal(
            jax.random.key(seed + 1),
            (shape.batch_size, shape.num_tokens, shape.num_heads, shape.key_head_dim),
            dtype=jnp.float32,
        )
        * input_magnitude,
        sharding_config.make_sharding((None, None, None, None)),
    )
    values = jax.device_put(
        jax.random.normal(
            jax.random.key(seed + 2),
            (shape.batch_size, shape.num_tokens, shape.num_heads, shape.value_head_dim),
            dtype=jnp.float32,
        )
        * input_magnitude,
        sharding_config.make_sharding((None, None, None, None)),
    )
    decay_factor = jax.device_put(
        -(
            0.5
            + jnp.abs(
                jax.random.normal(
                    jax.random.key(seed + 3),
                    (shape.batch_size, shape.num_tokens, shape.num_heads),
                    dtype=jnp.float32,
                )
            )
        )
        * decay_magnitude,
        sharding_config.make_sharding((None, None, None)),
    )
    factor_shape = (shape.batch_size, shape.num_tokens, shape.num_heads)
    if beta_pattern is BetaPattern.ZERO:
        beta_values = jnp.zeros(factor_shape, dtype=jnp.float32)
    elif beta_pattern is BetaPattern.ONE:
        beta_values = jnp.ones(factor_shape, dtype=jnp.float32)
    elif beta_pattern is BetaPattern.NEAR_ZERO:
        beta_values = jnp.full(factor_shape, 1e-6, dtype=jnp.float32)
    elif beta_pattern is BetaPattern.NEAR_ONE:
        beta_values = jnp.full(factor_shape, 1.0 - 1e-6, dtype=jnp.float32)
    else:
        beta_values = jax.nn.sigmoid(
            jax.random.normal(
                jax.random.key(seed + 4),
                factor_shape,
                dtype=jnp.float32,
            )
        )
    beta = jax.device_put(
        beta_values,
        sharding_config.make_sharding((None, None, None)),
    )
    initial_state = jax.device_put(
        jax.random.normal(
            jax.random.key(seed + 5),
            (shape.batch_size, shape.num_heads, shape.value_head_dim, shape.key_head_dim),
            dtype=jnp.float32,
        )
        * state_magnitude,
        sharding_config.make_sharding((None, None, None, None)),
    )
    if num_steps_pattern is NumStepsPattern.BATCHED:
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
        pallas_update = _batched_steps_pallas_deltanet
        reference_update = _batched_steps_xla_deltanet
    else:
        num_steps = jax.device_put(
            jnp.asarray(seed % (shape.num_tokens + 1), dtype=jnp.int32),
            sharding_config.make_sharding(()),
        )
        pallas_update = _shared_steps_pallas_deltanet
        reference_update = _shared_steps_xla_deltanet

    with warnings.catch_warnings():
        if shape.must_use_pallas:
            warnings.filterwarnings(
                "error",
                message=r"Pallas DeltaNet recurrence .*falling back to XLA recurrence\.",
                category=RuntimeWarning,
            )
        outputs, final_state = pallas_update(
            queries,
            keys,
            values,
            decay_factor,
            beta,
            initial_state,
            num_steps,
        )
    reference_outputs, reference_state = reference_update(
        queries,
        keys,
        values,
        decay_factor,
        beta,
        initial_state,
        num_steps,
    )

    assert_close(
        result=outputs,
        reference=reference_outputs,
        atol=5e-2,
        rtol=1e-1,
        operation_name="DeltaNet outputs",
    )
    assert_close(
        result=final_state,
        reference=reference_state,
        atol=1e-3,
        rtol=3e-2,
        operation_name="DeltaNet final state",
    )
