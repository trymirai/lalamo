import argparse
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp

from lalamo.compressed.awq import AWQMatrixForInference, AWQSpec, _awq_unpack_master_weights
from lalamo.compressed.cute_w4a16_dot import _call_awq, _call_mlx, awq_w4a16_dot, mlx_w4a16_dot
from lalamo.compressed.mlx import MLXMatrixForInference, MLXSpec, _mlx_unpack_master_weights
from lalamo.weight_matrix import CompressionImplementation

type Benchmark = tuple[Callable[..., jax.Array], tuple[jax.Array, ...]]


def _time_us(fn: Callable[..., jax.Array], args: tuple[jax.Array, ...], *, warmups: int, iters: int) -> float:
    fn = jax.jit(fn)
    for _ in range(warmups):
        fn(*args).block_until_ready()

    start = time.perf_counter()
    for _ in range(iters):
        fn(*args).block_until_ready()
    return (time.perf_counter() - start) * 1_000_000 / iters


def _mlx_packed(
    vectors: jax.Array,
    packed_weights: jax.Array,
    scales: jax.Array,
    biases: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    return jax.vmap(lambda vector: mlx_w4a16_dot(vector, packed_weights, scales, biases, weights))(vectors)


def _mlx_materialize(
    vectors: jax.Array,
    packed_weights: jax.Array,
    scales: jax.Array,
    biases: jax.Array,
) -> jax.Array:
    weights = _mlx_unpack_master_weights(packed_weights, scales, biases, 32, 4).astype(vectors.dtype)
    return jnp.einsum("bc,rc->br", vectors, weights)


def _awq_packed(
    vectors: jax.Array,
    packed_weights: jax.Array,
    scales: jax.Array,
    packed_zero_points: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    return jax.vmap(lambda vector: awq_w4a16_dot(vector, packed_weights, scales, packed_zero_points, weights))(vectors)


def _awq_materialize(
    vectors: jax.Array,
    packed_weights: jax.Array,
    scales: jax.Array,
    packed_zero_points: jax.Array,
) -> jax.Array:
    weights = _awq_unpack_master_weights(packed_weights, scales, packed_zero_points, 32, 4).astype(vectors.dtype)
    return jnp.einsum("bc,rc->br", vectors, weights)


def _dense(vectors: jax.Array, weights: jax.Array) -> jax.Array:
    return jnp.einsum("bc,rc->br", vectors, weights)


def _mlx_paths(matrix: MLXMatrixForInference, vectors: jax.Array) -> dict[str, Benchmark]:
    packed_weights = matrix.packed_weights
    scales = matrix.scales.astype(vectors.dtype)
    biases = matrix.biases.astype(vectors.dtype)
    dense_weights = matrix.weights.astype(vectors.dtype)

    return {
        "direct_cute": (_call_mlx, (vectors, packed_weights, scales, biases)),
        "packed_cute": (_mlx_packed, (vectors, packed_weights, scales, biases, dense_weights)),
        "cached_dense": (_dense, (vectors, dense_weights)),
        "materialize_dense": (_mlx_materialize, (vectors, packed_weights, scales, biases)),
    }


def _awq_paths(matrix: AWQMatrixForInference, vectors: jax.Array) -> dict[str, Benchmark]:
    packed_weights = matrix.packed_weights
    scales = matrix.scales.astype(vectors.dtype)
    packed_zero_points = matrix.packed_zero_points
    dense_weights = matrix.weights.astype(vectors.dtype)

    return {
        "direct_cute": (_call_awq, (vectors, packed_weights, scales, packed_zero_points)),
        "packed_cute": (_awq_packed, (vectors, packed_weights, scales, packed_zero_points, dense_weights)),
        "cached_dense": (_dense, (vectors, dense_weights)),
        "materialize_dense": (_awq_materialize, (vectors, packed_weights, scales, packed_zero_points)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=6144)
    parser.add_argument("--cols", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    args = parser.parse_args()

    assert args.group_size == 32
    assert args.cols % args.group_size == 0
    dtype = jnp.float16 if args.dtype == "float16" else jnp.bfloat16

    key = jax.random.key(0)
    weight_key, vector_key = jax.random.split(key)
    weights = jax.random.normal(weight_key, (args.rows, args.cols), dtype=jnp.float32)
    max_batch = max(args.batches)
    vectors = jax.random.normal(vector_key, (max_batch, args.cols), dtype=dtype)

    mlx_matrix = MLXSpec(bits=4, group_size=args.group_size).compress(
        weights,
        implementation=CompressionImplementation.INFERENCE,
    )
    awq_matrix = AWQSpec(bits=4, group_size=args.group_size).compress(
        weights,
        implementation=CompressionImplementation.INFERENCE,
    )
    assert isinstance(mlx_matrix, MLXMatrixForInference)
    assert isinstance(awq_matrix, AWQMatrixForInference)

    print(f"backend={jax.default_backend()} shape={args.rows}x{args.cols} group={args.group_size} dtype={args.dtype}")
    print("format,batch,direct_cute_us,packed_cute_us,cached_dense_us,materialize_dense_us,best_us,best_path")
    for batch in args.batches:
        batch_vectors = vectors[:batch]
        for name, paths in (
            ("mlx", _mlx_paths(mlx_matrix, batch_vectors)),
            ("awq", _awq_paths(awq_matrix, batch_vectors)),
        ):
            times = {
                path_name: _time_us(path, path_args, warmups=args.warmups, iters=args.iters)
                for path_name, (path, path_args) in paths.items()
            }
            best_path = min(times, key=times.__getitem__)
            print(
                f"{name},{batch},{times['direct_cute']:.2f},{times['packed_cute']:.2f},{times['cached_dense']:.2f},"
                f"{times['materialize_dense']:.2f},{times[best_path]:.2f},{best_path}",
                flush=True,
            )


if __name__ == "__main__":
    main()
