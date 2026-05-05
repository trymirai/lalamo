# ruff: noqa: INP001
"""Benchmark grouped 4-bit training dot against materialized quantize+matmul."""

import argparse
import time
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax

from lalamo.compressed.awq import (
    _awq_deterministic_quantized_dot_output_input,
    _awq_quantize,
    _awq_quantized_dot_output_input,
)
from lalamo.compressed.mlx import (
    _mlx_deterministic_quantized_dot_output_input,
    _mlx_quantize,
    _mlx_quantized_dot_output_input,
)
from lalamo.compressed.rounding import deterministic_round_to_unsigned_grid


def _time_forward(
    fn: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array],
    weights: jax.Array,
    scales: jax.Array,
    affine: jax.Array,
    xs: jax.Array,
) -> float:
    def repeat(weights: jax.Array, scales: jax.Array, affine: jax.Array, xs: jax.Array) -> jax.Array:
        def body(i: jax.Array, total: jax.Array) -> jax.Array:
            result = fn(weights, scales, affine, xs[i])
            return total + jnp.sum(result.astype(jnp.float32))

        return lax.fori_loop(0, xs.shape[0], body, jnp.float32(0.0))

    compiled = jax.jit(repeat)
    compiled(weights, scales, affine, xs).block_until_ready()
    start = time.perf_counter()
    compiled(weights, scales, affine, xs).block_until_ready()
    return (time.perf_counter() - start) * 1e6 / xs.shape[0]


def _time_backward(
    fn: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array],
    weights: jax.Array,
    scales: jax.Array,
    affine: jax.Array,
    xs: jax.Array,
    cotangents: jax.Array,
) -> float:
    def loss(
        weights: jax.Array,
        scales: jax.Array,
        affine: jax.Array,
        x: jax.Array,
        cotangent: jax.Array,
    ) -> jax.Array:
        result = fn(weights, scales, affine, x)
        return jnp.sum(result.astype(jnp.float32) * cotangent)

    grad_fn = jax.grad(loss, argnums=(0, 1, 2, 3))

    def repeat(
        weights: jax.Array,
        scales: jax.Array,
        affine: jax.Array,
        xs: jax.Array,
        cotangents: jax.Array,
    ) -> jax.Array:
        def body(i: jax.Array, total: jax.Array) -> jax.Array:
            grads = grad_fn(weights, scales, affine, xs[i], cotangents[i])
            grad_sum = sum(jnp.sum(grad.astype(jnp.float32)) for grad in grads)
            return total + grad_sum

        return lax.fori_loop(0, xs.shape[0], body, jnp.float32(0.0))

    compiled = jax.jit(repeat)
    compiled(weights, scales, affine, xs, cotangents).block_until_ready()
    start = time.perf_counter()
    compiled(weights, scales, affine, xs, cotangents).block_until_ready()
    return (time.perf_counter() - start) * 1e6 / xs.shape[0]


def _grad_error(
    fast: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array],
    semantic_reference: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array],
    weights: jax.Array,
    scales: jax.Array,
    affine: jax.Array,
    x: jax.Array,
    cotangent: jax.Array,
) -> tuple[float, float]:
    def loss(
        fn: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array],
        weights: jax.Array,
        scales: jax.Array,
        affine: jax.Array,
        x: jax.Array,
    ) -> jax.Array:
        result = fn(weights, scales, affine, x)
        return jnp.sum(result.astype(jnp.float32) * cotangent)

    fast_grads = jax.grad(partial(loss, fast), argnums=(0, 1, 2, 3))(weights, scales, affine, x)
    reference_grads = jax.grad(partial(loss, semantic_reference), argnums=(0, 1, 2, 3))(weights, scales, affine, x)
    abs_diffs = [
        jnp.abs(fast_grad.astype(jnp.float32) - reference_grad.astype(jnp.float32))
        for fast_grad, reference_grad in zip(fast_grads, reference_grads, strict=True)
    ]
    references = [reference_grad.astype(jnp.float32) for reference_grad in reference_grads]
    max_abs = jnp.max(jnp.stack([jnp.max(diff) for diff in abs_diffs]))
    rms_diff = sum(jnp.sum(jnp.square(diff)) for diff in abs_diffs)
    rms_reference = sum(jnp.sum(jnp.square(reference)) for reference in references)
    rel_rms = jnp.sqrt(rms_diff / jnp.maximum(rms_reference, 1e-20))
    return float(max_abs), float(rel_rms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dim", type=int, default=6144)
    parser.add_argument("--input-dim", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    assert args.input_dim % args.group_size == 0

    dtype = jnp.bfloat16
    key = jr.key(0)
    weight_key, scale_key, affine_key, x_key, cotangent_key = jr.split(key, 5)
    weights = jr.normal(weight_key, (args.output_dim, args.input_dim), dtype=dtype) * jnp.asarray(0.02, dtype=dtype)
    scales = jr.uniform(
        scale_key,
        (args.output_dim, args.input_dim // args.group_size),
        minval=0.001,
        maxval=0.02,
        dtype=dtype,
    )
    affine = jr.normal(affine_key, scales.shape, dtype=dtype) * jnp.asarray(0.02, dtype=dtype)
    xs = jr.normal(x_key, (args.iters, args.input_dim), dtype=dtype)
    cotangents = jr.normal(cotangent_key, (args.iters, args.output_dim), dtype=jnp.float32)
    round_fn = partial(deterministic_round_to_unsigned_grid, bits=4)

    cases = {
        "MLX": (
            lambda w, s, b, x: _mlx_deterministic_quantized_dot_output_input(w, s, b, x, args.group_size, 4),
            lambda w, s, b, x: _mlx_quantized_dot_output_input(w, s, b, x, args.group_size, round_fn),
            lambda w, s, b, x: _mlx_quantize(w, s, b, group_size=args.group_size, round_fn=round_fn) @ x,
            lambda w, s, b, x: jnp.sum(
                _mlx_quantize(w, s, b, group_size=args.group_size, round_fn=round_fn).astype(jnp.float32)
                * x.astype(jnp.float32)[None, :],
                axis=-1,
            ).astype(x.dtype),
        ),
        "AWQ": (
            lambda w, s, z, x: _awq_deterministic_quantized_dot_output_input(w, s, z, x, args.group_size, 4),
            lambda w, s, z, x: _awq_quantized_dot_output_input(w, s, z, x, args.group_size, round_fn),
            lambda w, s, z, x: _awq_quantize(w, s, z, group_size=args.group_size, round_fn=round_fn) @ x,
            lambda w, s, z, x: jnp.sum(
                _awq_quantize(w, s, z, group_size=args.group_size, round_fn=round_fn).astype(jnp.float32)
                * x.astype(jnp.float32)[None, :],
                axis=-1,
            ).astype(x.dtype),
        ),
    }

    print(f"shape={args.output_dim}x{args.input_dim} group={args.group_size} dtype=bf16")
    for name, (fast, grouped_reference, materialized, semantic_reference) in cases.items():
        x = xs[0]
        cotangent = cotangents[0]
        fast_output = jax.jit(fast)(weights, scales, affine, x).block_until_ready()
        reference_output = jax.jit(semantic_reference)(weights, scales, affine, x).block_until_ready()
        forward_diff = jnp.max(jnp.abs(fast_output.astype(jnp.float32) - reference_output.astype(jnp.float32)))
        grad_max_abs, grad_rel_rms = _grad_error(fast, grouped_reference, weights, scales, affine, x, cotangent)
        grouped_max_abs, grouped_rel_rms = _grad_error(
            grouped_reference,
            semantic_reference,
            weights,
            scales,
            affine,
            x,
            cotangent,
        )
        fast_forward = _time_forward(fast, weights, scales, affine, xs)
        reference_forward = _time_forward(materialized, weights, scales, affine, xs)
        fast_backward = _time_backward(fast, weights, scales, affine, xs, cotangents)
        reference_backward = _time_backward(materialized, weights, scales, affine, xs, cotangents)
        print(
            f"{name}: forward_diff={float(forward_diff):.6f} "
            f"custom_grad_max_abs={grad_max_abs:.6f} custom_grad_rel_rms={grad_rel_rms:.6f} "
            f"grouped_vs_materialized_grad_max_abs={grouped_max_abs:.6f} "
            f"grouped_vs_materialized_grad_rel_rms={grouped_rel_rms:.6f} "
            f"forward={reference_forward:.2f}us->{fast_forward:.2f}us "
            f"backward={reference_backward:.2f}us->{fast_backward:.2f}us"
        )


if __name__ == "__main__":
    main()
