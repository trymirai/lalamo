from collections.abc import Callable, Sequence
from functools import partial

import cuda.bindings.driver as cuda
import cutlass
import jax
import jax.numpy as jnp
from cutlass import cute
from cutlass import jax as cjax
from jaxtyping import Array, Float, Int

from .cute_w4a16_contract import (
    AWQ_ROWS_PER_CTA,
    GROUP_SIZE,
    MICRO_PACKED,
    MICRO_VALUES,
    MLX_ROWS_PER_CTA,
    REDUCE_THREADS,
)

DENSE_PREFILL_BATCH = 2


@cute.kernel
def _mlx_kernel(
    x_tiles: cute.Tensor,
    w_tiles: cute.Tensor,
    scales: cute.Tensor,
    biases: cute.Tensor,
    out: cute.Tensor,
    chunks_per_thread: cutlass.Constexpr,
) -> None:
    tx, ty, _ = cute.arch.thread_idx()
    batch, cta_row, _ = cute.arch.block_idx()
    row = cta_row * MLX_ROWS_PER_CTA + ty

    x_frag = cute.make_rmem_tensor(cute.size(x_tiles, mode=[0]), x_tiles.element_type)
    w_frag = cute.make_rmem_tensor(cute.size(w_tiles, mode=[0]), w_tiles.element_type)

    acc = cutlass.Float32(0.0)
    for block in cutlass.range_constexpr(chunks_per_thread):
        chunk = block * REDUCE_THREADS + tx
        cute.autovec_copy(x_tiles[(None, (batch, chunk))], x_frag)
        cute.autovec_copy(w_tiles[(None, (row, chunk))], w_frag)

        int_dot = cutlass.Float32(0.0)
        x_sum = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(MICRO_PACKED):
            byte = w_frag[j]
            lo = byte & cutlass.Uint8(15)
            hi = (byte >> cutlass.Uint8(4)) & cutlass.Uint8(15)
            x0 = cutlass.Float32(x_frag[j * 2])
            x1 = cutlass.Float32(x_frag[j * 2 + 1])
            int_dot += cutlass.Float32(lo) * x0 + cutlass.Float32(hi) * x1
            x_sum += x0 + x1

        group = chunk * MICRO_VALUES // GROUP_SIZE
        acc += int_dot * cutlass.Float32(scales[row, group]) + x_sum * cutlass.Float32(biases[row, group])

    lane_id = tx & 31
    acc += cute.arch.shuffle_sync_down(acc, offset=16)
    acc += cute.arch.shuffle_sync_down(acc, offset=8)
    acc += cute.arch.shuffle_sync_down(acc, offset=4)
    acc += cute.arch.shuffle_sync_down(acc, offset=2)
    acc += cute.arch.shuffle_sync_down(acc, offset=1)

    if lane_id == 0:
        if out.element_type == cutlass.BFloat16:
            out[batch, row] = cutlass.BFloat16(acc)
        else:
            out[batch, row] = cutlass.Float16(acc)


@cute.kernel
def _awq_kernel(
    x_tiles: cute.Tensor,
    w_tiles: cute.Tensor,
    scales: cute.Tensor,
    zero_tiles: cute.Tensor,
    out: cute.Tensor,
    chunks_per_thread: cutlass.Constexpr,
) -> None:
    tx, ty, _ = cute.arch.thread_idx()
    batch, cta_row, _ = cute.arch.block_idx()
    row = cta_row * AWQ_ROWS_PER_CTA + ty

    x_frag = cute.make_rmem_tensor(cute.size(x_tiles, mode=[0]), x_tiles.element_type)
    w_frag = cute.make_rmem_tensor(cute.size(w_tiles, mode=[0]), w_tiles.element_type)

    acc = cutlass.Float32(0.0)
    for block in cutlass.range_constexpr(chunks_per_thread):
        chunk = block * REDUCE_THREADS + tx
        cute.autovec_copy(x_tiles[(None, (batch, chunk))], x_frag)
        cute.autovec_copy(w_tiles[(None, (row, chunk))], w_frag)

        int_dot = cutlass.Float32(0.0)
        x_sum = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(MICRO_PACKED):
            byte = w_frag[j]
            lo = byte & cutlass.Uint8(15)
            hi = (byte >> cutlass.Uint8(4)) & cutlass.Uint8(15)
            x0 = cutlass.Float32(x_frag[j * 2])
            x1 = cutlass.Float32(x_frag[j * 2 + 1])
            int_dot += cutlass.Float32(lo) * x0 + cutlass.Float32(hi) * x1
            x_sum += x0 + x1

        group = chunk * MICRO_VALUES // GROUP_SIZE
        zero_byte = zero_tiles[row, group // 2]
        zero_point = zero_byte & cutlass.Uint8(15)
        if group % 2 == 1:
            zero_point = (zero_byte >> cutlass.Uint8(4)) & cutlass.Uint8(15)
        acc += (int_dot - x_sum * cutlass.Float32(zero_point)) * cutlass.Float32(scales[row, group])

    lane_id = tx & 31
    acc += cute.arch.shuffle_sync_down(acc, offset=16)
    acc += cute.arch.shuffle_sync_down(acc, offset=8)
    acc += cute.arch.shuffle_sync_down(acc, offset=4)
    acc += cute.arch.shuffle_sync_down(acc, offset=2)
    acc += cute.arch.shuffle_sync_down(acc, offset=1)
    if lane_id == 0:
        if out.element_type == cutlass.BFloat16:
            out[batch, row] = cutlass.BFloat16(acc)
        else:
            out[batch, row] = cutlass.Float16(acc)


@cute.jit
def _launch_mlx(
    stream: cuda.CUstream,
    x: cute.Tensor,
    packed_weights: cute.Tensor,
    scales: cute.Tensor,
    biases: cute.Tensor,
    out: cute.Tensor,
    *,
    chunks_per_thread: int,
) -> None:
    x_tiles = cute.zipped_divide(x, (1, MICRO_VALUES))
    w_tiles = cute.zipped_divide(packed_weights, (1, MICRO_PACKED))
    _mlx_kernel(x_tiles, w_tiles, scales, biases, out, chunks_per_thread).launch(
        grid=[x.shape[0], cute.ceil_div(packed_weights.shape[0], MLX_ROWS_PER_CTA), 1],
        block=[REDUCE_THREADS, MLX_ROWS_PER_CTA, 1],
        stream=stream,
    )


@cute.jit
def _launch_awq(
    stream: cuda.CUstream,
    x: cute.Tensor,
    packed_weights: cute.Tensor,
    scales: cute.Tensor,
    packed_zero_points: cute.Tensor,
    out: cute.Tensor,
    *,
    chunks_per_thread: int,
) -> None:
    x_tiles = cute.zipped_divide(x, (1, MICRO_VALUES))
    w_tiles = cute.zipped_divide(packed_weights, (1, MICRO_PACKED))
    _awq_kernel(x_tiles, w_tiles, scales, packed_zero_points, out, chunks_per_thread).launch(
        grid=[x.shape[0], cute.ceil_div(packed_weights.shape[0], AWQ_ROWS_PER_CTA), 1],
        block=[REDUCE_THREADS, AWQ_ROWS_PER_CTA, 1],
        stream=stream,
    )


@jax.custom_batching.custom_vmap
def mlx_w4a16_dot(
    vector: Float[Array, " channels"],
    packed_weights: Int[Array, "rows packed_cols"],
    scales: Float[Array, "rows groups"],
    biases: Float[Array, "rows groups"],
    weights: Float[Array, "rows channels"],
) -> Float[Array, " rows"]:
    return mlx_w4a16_matmul(vector[None, :], packed_weights, scales, biases, weights)[0]


@jax.custom_batching.custom_vmap
def awq_w4a16_dot(
    vector: Float[Array, " channels"],
    packed_weights: Int[Array, "rows packed_cols"],
    scales: Float[Array, "rows groups"],
    packed_zero_points: Int[Array, "rows packed_groups"],
    weights: Float[Array, "rows channels"],
) -> Float[Array, " rows"]:
    return awq_w4a16_matmul(vector[None, :], packed_weights, scales, packed_zero_points, weights)[0]


def mlx_w4a16_matmul(
    vectors: Float[Array, "batch channels"],
    packed_weights: Int[Array, "rows packed_cols"],
    scales: Float[Array, "rows groups"],
    biases: Float[Array, "rows groups"],
    weights: Float[Array, "rows channels"],
) -> Float[Array, "batch rows"]:
    if vectors.shape[0] >= DENSE_PREFILL_BATCH:
        return jnp.einsum("bc,rc->br", vectors, weights.astype(vectors.dtype))
    return _call_mlx(vectors, packed_weights, scales, biases)


def awq_w4a16_matmul(
    vectors: Float[Array, "batch channels"],
    packed_weights: Int[Array, "rows packed_cols"],
    scales: Float[Array, "rows groups"],
    packed_zero_points: Int[Array, "rows packed_groups"],
    weights: Float[Array, "rows channels"],
) -> Float[Array, "batch rows"]:
    if vectors.shape[0] >= DENSE_PREFILL_BATCH:
        return jnp.einsum("bc,rc->br", vectors, weights.astype(vectors.dtype))
    return _call_awq(vectors, packed_weights, scales, packed_zero_points)


@mlx_w4a16_dot.def_vmap
def _mlx_vmap(axis_size: int, in_batched: Sequence[bool], *args: Array) -> tuple[Array, bool]:
    return _vmap_dot(axis_size, in_batched, mlx_w4a16_dot, mlx_w4a16_matmul, *args)


@awq_w4a16_dot.def_vmap
def _awq_vmap(axis_size: int, in_batched: Sequence[bool], *args: Array) -> tuple[Array, bool]:
    return _vmap_dot(axis_size, in_batched, awq_w4a16_dot, awq_w4a16_matmul, *args)


def _vmap_dot(
    axis_size: int,
    in_batched: Sequence[bool],
    dot_fn: Callable[[Array, Array, Array, Array, Array], Array],
    matmul_fn: Callable[[Array, Array, Array, Array, Array], Array],
    vector: Array,
    packed_weights: Array,
    scales: Array,
    affine: Array,
    weights: Array,
) -> tuple[Array, bool]:
    batched = tuple(in_batched)
    assert any(batched)
    if batched == (True, False, False, False, False):
        assert vector.shape[0] == axis_size
        return matmul_fn(vector, packed_weights, scales, affine, weights), True

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
            take(weights, batched[4], index),
        )

    return jax.lax.map(body, jnp.arange(axis_size)), True


def _call_mlx(vectors: Array, packed_weights: Array, scales: Array, biases: Array) -> Array:
    return _call(_launch_mlx, MLX_ROWS_PER_CTA, vectors, packed_weights, scales, biases)


def _call_awq(vectors: Array, packed_weights: Array, scales: Array, packed_zero_points: Array) -> Array:
    return _call(_launch_awq, AWQ_ROWS_PER_CTA, vectors, packed_weights, scales, packed_zero_points)


@partial(jax.jit, static_argnums=(0, 1))
def _call(
    launcher: Callable[..., None],
    rows_per_cta: int,
    vectors: Array,
    packed_weights: Array,
    scales: Array,
    affine: Array,
) -> Array:
    assert vectors.ndim == 2
    assert vectors.shape[1] == packed_weights.shape[1] * 2
    assert vectors.shape[1] == scales.shape[1] * GROUP_SIZE
    assert vectors.shape[1] % (REDUCE_THREADS * MICRO_VALUES) == 0
    assert vectors.dtype in (jnp.float16, jnp.bfloat16)
    assert packed_weights.shape[0] % rows_per_cta == 0

    call = cjax.cutlass_call(
        launcher,
        output_shape_dtype=jax.ShapeDtypeStruct((vectors.shape[0], packed_weights.shape[0]), vectors.dtype),
        use_static_tensors=True,
        chunks_per_thread=vectors.shape[1] // (REDUCE_THREADS * MICRO_VALUES),
    )
    return call(vectors, packed_weights, scales, affine)
