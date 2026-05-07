from collections.abc import Callable, Sequence
from importlib import import_module

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from lalamo.compressed.cute_w4a16_contract import use_packed_w4a16_route


def _dense_matmul(vectors: Array, dense_weights: Array) -> Array:
    return jnp.einsum("bc,rc->br", vectors, dense_weights.astype(vectors.dtype))


@jax.custom_batching.custom_vmap
def mlx_w4a16_inference_dot(
    vector: Float[Array, " channels"],
    packed_weights: Int[Array, "rows packed_cols"],
    scales: Float[Array, "rows groups"],
    biases: Float[Array, "rows groups"],
    dense_weights: Float[Array, "rows channels"],
) -> Float[Array, " rows"]:
    return mlx_w4a16_inference_matmul(vector[None, :], packed_weights, scales, biases, dense_weights)[0]


@jax.custom_batching.custom_vmap
def awq_w4a16_inference_dot(
    vector: Float[Array, " channels"],
    packed_weights: Int[Array, "rows packed_cols"],
    scales: Float[Array, "rows groups"],
    packed_zero_points: Int[Array, "rows packed_groups"],
    dense_weights: Float[Array, "rows channels"],
) -> Float[Array, " rows"]:
    return awq_w4a16_inference_matmul(
        vector[None, :],
        packed_weights,
        scales,
        packed_zero_points,
        dense_weights,
    )[0]


def mlx_w4a16_inference_matmul(
    vectors: Float[Array, "batch channels"],
    packed_weights: Int[Array, "rows packed_cols"],
    scales: Float[Array, "rows groups"],
    biases: Float[Array, "rows groups"],
    dense_weights: Float[Array, "rows channels"],
) -> Float[Array, "batch rows"]:
    if use_packed_w4a16_route(vectors.shape[0], packed_weights.shape[0]):
        mlx_w4a16_matmul = import_module("lalamo.compressed.cute_w4a16_dot").mlx_w4a16_matmul
        return mlx_w4a16_matmul(vectors, packed_weights, scales, biases)
    return _dense_matmul(vectors, dense_weights)


def awq_w4a16_inference_matmul(
    vectors: Float[Array, "batch channels"],
    packed_weights: Int[Array, "rows packed_cols"],
    scales: Float[Array, "rows groups"],
    packed_zero_points: Int[Array, "rows packed_groups"],
    dense_weights: Float[Array, "rows channels"],
) -> Float[Array, "batch rows"]:
    if use_packed_w4a16_route(vectors.shape[0], packed_weights.shape[0]):
        awq_w4a16_matmul = import_module("lalamo.compressed.cute_w4a16_dot").awq_w4a16_matmul
        return awq_w4a16_matmul(vectors, packed_weights, scales, packed_zero_points)
    return _dense_matmul(vectors, dense_weights)


@mlx_w4a16_inference_dot.def_vmap
def _mlx_vmap(axis_size: int, in_batched: Sequence[bool], *args: Array) -> tuple[Array, bool]:
    return _vmap_dot(axis_size, in_batched, mlx_w4a16_inference_dot, mlx_w4a16_inference_matmul, *args)


@awq_w4a16_inference_dot.def_vmap
def _awq_vmap(axis_size: int, in_batched: Sequence[bool], *args: Array) -> tuple[Array, bool]:
    return _vmap_dot(axis_size, in_batched, awq_w4a16_inference_dot, awq_w4a16_inference_matmul, *args)


def _vmap_dot(
    axis_size: int,
    in_batched: Sequence[bool],
    dot_fn: Callable[[Array, Array, Array, Array, Array], Array],
    matmul_fn: Callable[[Array, Array, Array, Array, Array], Array],
    vector: Array,
    packed_weights: Array,
    scales: Array,
    affine: Array,
    dense_weights: Array,
) -> tuple[Array, bool]:
    batched = tuple(in_batched)
    assert any(batched)
    if batched == (True, False, False, False, False):
        assert vector.shape[0] == axis_size
        return matmul_fn(vector, packed_weights, scales, affine, dense_weights), True

    def take(array: Array, is_batched: bool, index: Array) -> Array:
        if is_batched:
            return jax.lax.dynamic_index_in_dim(array, index, keepdims=False)
        return array

    def body(index: Array) -> Array:
        return dot_fn(
            take(vector, batched[0], index),
            take(packed_weights, batched[1], index),
            take(scales, batched[2], index),
            take(affine, batched[3], index),
            take(dense_weights, batched[4], index),
        )

    return jax.lax.map(body, jnp.arange(axis_size)), True
