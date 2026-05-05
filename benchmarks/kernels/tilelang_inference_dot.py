# ruff: noqa: INP001, PLC0415
"""Benchmark the optional TileLang inference dot path end to end."""

import argparse
import time
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr

from lalamo.compressed.awq import (
    AWQMatrixForInference,
    AWQSpec,
    _awq_dot_output_input_4bit,
    _awq_unpack_master_weights,
)
from lalamo.compressed.mlx import (
    MLXMatrixForInference,
    MLXSpec,
    _mlx_dot_output_input_4bit,
    _mlx_unpack_master_weights,
)
from lalamo.weight_matrix import CompressionImplementation


def _time(fn: Callable[[], jax.Array], iters: int) -> float:
    fn().block_until_ready()
    start = time.perf_counter()
    for _ in range(iters):
        fn().block_until_ready()
    return (time.perf_counter() - start) * 1e6 / iters


def _time_torch(fn: Callable[[], object], iters: int) -> float:
    import torch

    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dim", type=int, default=6144)
    parser.add_argument("--input-dim", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    args = parser.parse_args()

    dtype = jnp.float16 if args.dtype == "fp16" else jnp.bfloat16
    weight_key, x_key = jr.split(jr.key(0))
    weights = jr.normal(weight_key, (args.output_dim, args.input_dim), dtype=dtype)
    vector = jr.normal(x_key, (args.input_dim,), dtype=dtype)
    from torch.utils import dlpack as torch_dlpack

    mlx = cast(
        "MLXMatrixForInference",
        MLXSpec(bits=4, group_size=args.group_size).compress(
            weights,
            implementation=CompressionImplementation.INFERENCE,
        ),
    )
    awq = cast(
        "AWQMatrixForInference",
        AWQSpec(bits=4, group_size=args.group_size).compress(
            weights,
            implementation=CompressionImplementation.INFERENCE,
        ),
    )

    mlx_jax = jax.jit(
        lambda packed, scales, biases, x: _mlx_dot_output_input_4bit(packed, scales, biases, x, args.group_size)
    )
    awq_jax = jax.jit(
        lambda packed, scales, zeros, x: _awq_dot_output_input_4bit(packed, scales, zeros, x, args.group_size)
    )
    mlx_materialized = jax.jit(
        lambda packed, scales, biases, x: _mlx_unpack_master_weights(packed, scales, biases, args.group_size, 4) @ x
    )
    awq_materialized = jax.jit(
        lambda packed, scales, zeros, x: _awq_unpack_master_weights(packed, scales, zeros, args.group_size, 4) @ x
    )

    mlx_reference = mlx_jax(mlx.packed_weights, mlx.scales.astype(dtype), mlx.biases.astype(dtype), vector)
    awq_reference = awq_jax(awq.packed_weights, awq.scales.astype(dtype), awq.packed_zero_points, vector)
    mlx_tilelang_dot = mlx.to_tilelang_dot(dtype)
    awq_tilelang_dot = awq.to_tilelang_dot(dtype)
    vector_torch = torch_dlpack.from_dlpack(vector)
    mlx_tilelang = jnp.from_dlpack(mlx_tilelang_dot(vector_torch).detach())
    awq_tilelang = jnp.from_dlpack(awq_tilelang_dot(vector_torch).detach())
    mlx_diff = jnp.max(jnp.abs(mlx_reference.astype(jnp.float32) - mlx_tilelang.astype(jnp.float32)))
    awq_diff = jnp.max(jnp.abs(awq_reference.astype(jnp.float32) - awq_tilelang.astype(jnp.float32)))

    mlx_materialized_us = _time(
        lambda: mlx_materialized(mlx.packed_weights, mlx.scales.astype(dtype), mlx.biases.astype(dtype), vector),
        args.iters,
    )
    mlx_jax_us = _time(
        lambda: mlx_jax(mlx.packed_weights, mlx.scales.astype(dtype), mlx.biases.astype(dtype), vector),
        args.iters,
    )
    mlx_tilelang_torch_us = _time_torch(lambda: mlx_tilelang_dot(vector_torch), args.iters)
    mlx_tilelang_roundtrip_us = _time(lambda: mlx_tilelang_dot.jax(vector), args.iters)
    awq_materialized_us = _time(
        lambda: awq_materialized(awq.packed_weights, awq.scales.astype(dtype), awq.packed_zero_points, vector),
        args.iters,
    )
    awq_jax_us = _time(
        lambda: awq_jax(awq.packed_weights, awq.scales.astype(dtype), awq.packed_zero_points, vector),
        args.iters,
    )
    awq_tilelang_torch_us = _time_torch(lambda: awq_tilelang_dot(vector_torch), args.iters)
    awq_tilelang_roundtrip_us = _time(lambda: awq_tilelang_dot.jax(vector), args.iters)

    print(f"shape={args.output_dim}x{args.input_dim} group={args.group_size} dtype={args.dtype}")
    print(
        f"MLX diff={float(mlx_diff):.6f} "
        f"materialized={mlx_materialized_us:.2f}us "
        f"jax_grouped={mlx_jax_us:.2f}us "
        f"tilelang_torch={mlx_tilelang_torch_us:.2f}us "
        f"tilelang_roundtrip={mlx_tilelang_roundtrip_us:.2f}us"
    )
    print(
        f"AWQ diff={float(awq_diff):.6f} "
        f"materialized={awq_materialized_us:.2f}us "
        f"jax_grouped={awq_jax_us:.2f}us "
        f"tilelang_torch={awq_tilelang_torch_us:.2f}us "
        f"tilelang_roundtrip={awq_tilelang_roundtrip_us:.2f}us"
    )


if __name__ == "__main__":
    main()
