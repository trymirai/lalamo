from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jax.lax import DotAlgorithmPreset

from lalamo.kernels.ragged_dot import ragged_dot
from lalamo.utils.sharding import ShardingConfig
from tests.common import assert_close

pytestmark = pytest.mark.fast


@dataclass(frozen=True)
class RaggedDotShape:
    group_sizes: tuple[int, ...]
    input_dim: int
    output_dim: int
    tail_size: int = 0


_SHAPES = st.sampled_from(
    (
        RaggedDotShape((1,), 1, 1),
        RaggedDotShape((5,), 4, 3),
        RaggedDotShape((0, 5), 3, 4),
        RaggedDotShape((2, 0, 1, 2), 4, 7),
        RaggedDotShape((0, 1, 0, 3, 2, 0, 1, 1), 7, 4),
        RaggedDotShape((0, 0), 2, 3, tail_size=2),
        RaggedDotShape((1, 0, 2), 5, 1, tail_size=3),
        RaggedDotShape((0, 3, 0, 0, 1), 8, 6, tail_size=1),
        RaggedDotShape((1,) * 8, 3, 5),
        RaggedDotShape((4, 0, 0, 0, 0, 0, 0, 4), 6, 8),
    ),
)


@settings(max_examples=100, deadline=None)
@given(
    shape=_SHAPES,
    seed=st.integers(min_value=0, max_value=2**31 - 3),
    dtype=st.sampled_from((jnp.float16, jnp.bfloat16, jnp.float32)),
)
def test_ragged_dot_matches_grouped_matmul(shape: RaggedDotShape, seed: int, dtype: jnp.dtype) -> None:
    sharding_config = ShardingConfig.replicated()
    num_grouped_tokens = sum(shape.group_sizes)
    num_tokens = num_grouped_tokens + shape.tail_size
    vectors = jax.device_put(
        jax.random.normal(jax.random.key(seed), (num_tokens, shape.input_dim), dtype=jnp.float32).astype(dtype),
        sharding_config.make_sharding((None, None)),
    )
    expert_weights = jax.device_put(
        jax.random.normal(
            jax.random.key(seed + 1),
            (len(shape.group_sizes), shape.input_dim, shape.output_dim),
            dtype=jnp.float32,
        ).astype(dtype),
        sharding_config.make_sharding((None, None, None)),
    )
    group_sizes = jax.device_put(
        jnp.array(shape.group_sizes, dtype=jnp.int32),
        sharding_config.make_sharding((None,)),
    )

    result = ragged_dot(
        vectors,
        expert_weights,
        group_sizes,
        precision=DotAlgorithmPreset.DEFAULT,
    )
    group_ends = jnp.cumsum(group_sizes)
    group_starts = jnp.concatenate((jnp.zeros((1,), dtype=jnp.int32), group_ends[:-1]))
    reference = jnp.concatenate(
        tuple(
            vectors[start:end] @ weights
            for start, end, weights in zip(group_starts, group_ends, expert_weights, strict=True)
        ),
    )

    grouped_result = jax.device_get(result)[:num_grouped_tokens]
    if num_grouped_tokens > 0:
        assert_close(result=grouped_result, reference=reference)
    else:
        assert grouped_result.shape == reference.shape
    assert result.sharding == vectors.sharding


def test_ragged_dot_gradients_match_grouped_matmul() -> None:
    sharding_config = ShardingConfig.replicated()
    vectors = jax.device_put(
        jnp.arange(5 * 4, dtype=jnp.float32).reshape(5, 4) / 10,
        sharding_config.make_sharding((None, None)),
    )
    expert_weights = jax.device_put(
        jnp.arange(3 * 4 * 3, dtype=jnp.float32).reshape(3, 4, 3) / 20,
        sharding_config.make_sharding((None, None, None)),
    )
    group_sizes = jax.device_put(
        jnp.array([2, 0, 3], dtype=jnp.int32),
        sharding_config.make_sharding((None,)),
    )

    def ragged_loss(current_vectors: jax.Array, current_weights: jax.Array) -> jax.Array:
        outputs = ragged_dot(
            current_vectors,
            current_weights,
            group_sizes,
            precision=DotAlgorithmPreset.DEFAULT,
        )
        return jnp.square(outputs).sum()

    def reference_loss(current_vectors: jax.Array, current_weights: jax.Array) -> jax.Array:
        expert_indices = jnp.array([0, 0, 2, 2, 2], dtype=jnp.int32)
        outputs = jnp.einsum("tio,ti->to", current_weights[expert_indices], current_vectors)
        return jnp.square(outputs).sum()

    vector_grad, weight_grad = jax.grad(ragged_loss, argnums=(0, 1))(vectors, expert_weights)
    vector_reference, weight_reference = jax.grad(reference_loss, argnums=(0, 1))(vectors, expert_weights)

    assert_close(result=vector_grad, reference=vector_reference)
    assert_close(result=weight_grad, reference=weight_reference)
