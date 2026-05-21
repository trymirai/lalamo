from collections.abc import Callable, Sequence
from functools import cache
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from lalamo.utils.sharding import sharding_of

__all__ = [
    "hadamard_transform",
]


def hadamard_transform(
    inputs: Float[Array, "... channels"],
    block_size: Literal[32, 64, 128],
) -> Float[Array, "... channels"]:
    sharding = sharding_of(inputs)
    *_, channel_axis = sharding.spec
    if channel_axis is not None:
        raise ValueError("Hadamard transform inputs must not be sharded along the channels axis.")
    return _make_hadamard_transform_for_block_size(block_size)(inputs)


def _jax_hadamard_transform(
    inputs: Float[Array, "... channels"],
    block_size: Literal[32, 64, 128],
) -> Float[Array, "... channels"]:
    *original_leading_dims, input_dim = inputs.shape
    result = jnp.reshape(inputs, (*original_leading_dims, input_dim // block_size, block_size))

    for stage in range(block_size.bit_length() - 1):
        butterfly_size = 2 ** (stage + 1)
        half_butterfly_size = 2**stage
        *block_leading_dims, _ = result.shape
        grouped = jnp.reshape(
            result,
            (*block_leading_dims, block_size // butterfly_size, butterfly_size),
        )
        left = grouped[..., :half_butterfly_size]
        right = grouped[..., half_butterfly_size:]
        result = jnp.reshape(
            jnp.concatenate((left + right, left - right), axis=-1),
            (*block_leading_dims, block_size),
        )

    normalization = jnp.sqrt(jnp.asarray(block_size, dtype=inputs.dtype))
    return jnp.reshape(result / normalization, (*original_leading_dims, input_dim))


@cache
def _make_hadamard_transform_for_block_size(
    block_size: Literal[32, 64, 128],
) -> Callable[[Float[Array, "... channels"]], Float[Array, "... channels"]]:
    if block_size not in (32, 64, 128):
        raise ValueError(f"Block size {block_size} must be one of 32, 64, or 128")

    @jax.custom_vjp
    @jax.custom_batching.custom_vmap
    def transform(inputs: Float[Array, "... channels"]) -> Float[Array, "... channels"]:
        *_, input_dim = inputs.shape
        if input_dim % block_size != 0:
            raise ValueError(
                f"Input dimension {input_dim} must be a multiple of block size {block_size}",
            )

        if jax.default_backend() == "gpu":
            from lalamo.compressed.utils.hadamard_kernels import cute_hadamard_transform  # noqa: PLC0415

            return cute_hadamard_transform(inputs, block_size)
        return _jax_hadamard_transform(inputs, block_size)

    @transform.def_vmap
    def transform_vmap(
        axis_size: int,
        in_batched: Sequence[bool],
        inputs: Float[Array, "... channels"],
    ) -> tuple[Float[Array, "... channels"], bool]:
        del axis_size
        (inputs_batched,) = in_batched
        return transform(inputs), inputs_batched

    def transform_fwd(
        inputs: Float[Array, "... channels"],
    ) -> tuple[Float[Array, "... channels"], None]:
        return transform(inputs), None

    def transform_bwd(
        _: None,
        cotangent: Float[Array, "... channels"],
    ) -> tuple[Float[Array, "... channels"]]:
        return (transform(cotangent),)

    transform.defvjp(transform_fwd, transform_bwd)
    return transform
