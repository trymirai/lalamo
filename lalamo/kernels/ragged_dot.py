from functools import partial

import jax
from jax import lax
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array, Float, Int

from lalamo.utils.sharding import sharding_of


@partial(jax.custom_jvp, nondiff_argnums=(3,))
def ragged_dot(
    vectors: Float[Array, "tokens input_channels"],
    expert_weights: Float[Array, "experts input_channels output_channels"],
    group_sizes: Int[Array, " experts"],
    precision: DotAlgorithmPreset,
) -> Float[Array, "tokens output_channels"]:
    return lax.ragged_dot(
        vectors,
        expert_weights,
        group_sizes,
        precision=precision,
        out_sharding=sharding_of(vectors),
    )


@ragged_dot.defjvp
def _ragged_dot_jvp(
    precision: DotAlgorithmPreset,
    primals: tuple[Array, Array, Array],
    tangents: tuple[Array, Array, object],
) -> tuple[Array, Array]:
    # JAX's built-in JVP drops explicit out_sharding, so keep both tangent terms on this wrapper.
    vectors, expert_weights, group_sizes = primals
    vector_tangent, weight_tangent, _ = tangents
    result = ragged_dot(vectors, expert_weights, group_sizes, precision)
    tangent = ragged_dot(vector_tangent, expert_weights, group_sizes, precision)
    tangent += ragged_dot(vectors, weight_tangent, group_sizes, precision)
    return result, tangent
