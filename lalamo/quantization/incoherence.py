from functools import cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
from einops import rearrange
from jaxtyping import Array, Float, Int, PRNGKeyArray


class RHTFactors(NamedTuple):
    input_factors: Int[Array, " in_channels"]
    output_factors: Int[Array, " out_channels"]


@cache
def get_hadamard_matrix(block_size: int) -> Int[np.ndarray, " block_size block_size"]:
    return scipy.linalg.hadamard(block_size)


def hadamard_transform(
    inputs: Float[Array, " channels"],
    block_size: int,
    inverse: bool = False,
) -> Float[Array, " channels"]:
    if block_size.bit_count() != 1:
        raise ValueError(f"Block size {block_size} must be a power of 2")
    (input_dim,) = inputs.shape
    if input_dim % block_size != 0:
        raise ValueError(f"Input dimension {input_dim} must be a multiple of block size {block_size}")
    transform_matrix = jnp.array(get_hadamard_matrix(block_size), dtype=inputs.dtype) / jnp.sqrt(block_size)
    if inverse:
        transform_matrix = transform_matrix.T
    blocks = rearrange(inputs, "(blocks block_channels) -> blocks block_channels", block_channels=block_size)
    result = jax.vmap(lambda block: transform_matrix @ block)(blocks)
    return rearrange(result, "blocks block_channels -> (blocks block_channels)")


def random_incoherence_keys(input_dim: int, output_dim: int, key: PRNGKeyArray) -> RHTFactors:
    input_key, output_key = jax.random.split(key)
    input_factors = jnp.where(jax.random.bernoulli(input_key, shape=(input_dim,)), 1, -1)
    output_factors = jnp.where(jax.random.bernoulli(output_key, shape=(output_dim,)), 1, -1)
    assert input_factors.dtype == jnp.int32
    return RHTFactors(input_factors, output_factors)
