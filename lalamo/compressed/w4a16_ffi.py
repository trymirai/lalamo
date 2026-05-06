from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset

if TYPE_CHECKING:
    from jaxtyping import Array, Float, Int

__all__ = [
    "awq_w4a16_dot",
    "mlx_w4a16_dot",
    "register_w4a16_ffi_library",
]

_MLX_TARGETS = {
    jnp.dtype(jnp.float16): "lalamo_w4a16_mlx_f16",
    jnp.dtype(jnp.bfloat16): "lalamo_w4a16_mlx_bf16",
}
_AWQ_TARGETS = {
    jnp.dtype(jnp.float16): "lalamo_w4a16_awq_f16",
    jnp.dtype(jnp.bfloat16): "lalamo_w4a16_awq_bf16",
}

_loaded_libraries: list[ctypes.CDLL] = []


def register_w4a16_ffi_library(path: Path | str) -> None:
    library = ctypes.cdll.LoadLibrary(str(path))
    for target in (*_MLX_TARGETS.values(), *_AWQ_TARGETS.values()):
        jax.ffi.register_ffi_target(
            target,
            jax.ffi.pycapsule(getattr(library, target)),
            platform="CUDA",
        )
    _loaded_libraries.append(library)


def mlx_w4a16_dot(
    vector: Float[Array, " ... channels"],
    packed_weights: Int[Array, " rows packed_channels"],
    scales: Float[Array, " rows groups"],
    biases: Float[Array, " rows groups"],
    *,
    precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT,
) -> Float[Array, " ... rows"]:
    _check_common_inputs(vector, packed_weights, scales)
    _check_precision(precision)
    if biases.shape != scales.shape:
        raise ValueError(f"biases shape {biases.shape} must match scales shape {scales.shape}")
    return _call(_MLX_TARGETS[jnp.dtype(vector.dtype)], vector, packed_weights, scales, biases)


def awq_w4a16_dot(
    vector: Float[Array, " ... channels"],
    packed_weights: Int[Array, " rows packed_channels"],
    scales: Float[Array, " rows groups"],
    packed_zero_points: Int[Array, " rows packed_groups"],
    *,
    precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT,
) -> Float[Array, " ... rows"]:
    _check_common_inputs(vector, packed_weights, scales)
    _check_precision(precision)
    if packed_zero_points.shape != (scales.shape[0], scales.shape[1] // 2):
        raise ValueError(
            f"packed_zero_points shape {packed_zero_points.shape} must be "
            f"{(scales.shape[0], scales.shape[1] // 2)}"
        )
    return _call(_AWQ_TARGETS[jnp.dtype(vector.dtype)], vector, packed_weights, scales, packed_zero_points)


def _check_common_inputs(
    vector: Array,
    packed_weights: Array,
    scales: Array,
) -> None:
    if vector.ndim not in {1, 2}:
        raise ValueError(f"vector must have rank 1 or 2, got {vector.ndim}")
    if packed_weights.ndim != 2:
        raise ValueError(f"packed_weights must have rank 2, got {packed_weights.ndim}")
    if scales.ndim != 2:
        raise ValueError(f"scales must have rank 2, got {scales.ndim}")
    if packed_weights.dtype != jnp.uint8:
        raise TypeError(f"packed_weights must be uint8, got {packed_weights.dtype}")
    vector_dtype = jnp.dtype(vector.dtype)
    if vector_dtype not in _MLX_TARGETS:
        raise TypeError(f"vector must be float16 or bfloat16, got {vector.dtype}")
    if jnp.dtype(scales.dtype) != vector_dtype:
        raise TypeError(f"scales dtype {scales.dtype} must match vector dtype {vector.dtype}")
    rows, packed_channels = packed_weights.shape
    if scales.shape[0] != rows:
        raise ValueError(f"scales rows {scales.shape[0]} must match packed weight rows {rows}")
    if vector.shape[-1] != packed_channels * 2:
        raise ValueError(f"vector channels {vector.shape[-1]} must match packed weight channels {packed_channels * 2}")
    if scales.shape[1] * 32 != vector.shape[-1]:
        raise ValueError(f"scales groups {scales.shape[1]} must cover {vector.shape[-1]} channels with group size 32")


def _check_precision(precision: DotAlgorithmPreset) -> None:
    if precision != DotAlgorithmPreset.DEFAULT:
        raise ValueError(f"w4a16 FFI kernels only support {DotAlgorithmPreset.DEFAULT}, got {precision}")


def _call(target: str, vector: Array, packed_weights: Array, scales: Array, affine: Array) -> Array:
    result_shape = (*vector.shape[:-1], packed_weights.shape[0])
    call = jax.ffi.ffi_call(
        target,
        jax.ShapeDtypeStruct(result_shape, vector.dtype),
        vmap_method="sequential",
    )
    return call(vector, packed_weights, scales, affine)
