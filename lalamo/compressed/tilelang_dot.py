# pyrefly: ignore-errors
# ruff: noqa: ANN001, ANN202, ANN204, ERA001, PLC0415, N806, N812
from collections.abc import Callable
from functools import lru_cache
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int

_TileLangScheme = Literal["mlx", "awq"]

__all__ = ["TileLangDot"]


class TileLangDot:
    @classmethod
    def from_mlx_packed(
        cls,
        packed_weights: Int[Array, "rows packed_cols"],
        scales: Float[Array, "rows groups"],
        biases: Float[Array, "rows groups"],
        group_size: int,
        dtype: DTypeLike,
    ) -> "TileLangDot":
        return cls("mlx", packed_weights, scales, biases, group_size, dtype)

    @classmethod
    def from_awq_packed(
        cls,
        packed_weights: Int[Array, "rows packed_cols"],
        scales: Float[Array, "rows groups"],
        packed_zero_points: Int[Array, "rows packed_groups"],
        group_size: int,
        dtype: DTypeLike,
    ) -> "TileLangDot":
        torch = _require_torch()
        zero_points = _jax_to_torch(packed_zero_points).view(torch.uint8)
        lo = (zero_points & 0x0F).to(_torch_dtype(dtype))
        hi = ((zero_points >> 4) & 0x0F).to(_torch_dtype(dtype))
        unpacked_zero_points = torch.stack((lo, hi), dim=-1).reshape(scales.shape)
        return cls("awq", packed_weights, scales, unpacked_zero_points, group_size, dtype)

    def __init__(
        self,
        scheme: _TileLangScheme,
        packed_weights: Int[Array, "rows packed_cols"],
        scales: Float[Array, "rows groups"],
        affine: Array,
        group_size: int,
        dtype: DTypeLike,
    ) -> None:
        torch = _require_torch()
        rows, packed_cols = packed_weights.shape
        channels = packed_cols * 2
        self.rows = rows
        self.channels = channels
        self.group_size = group_size
        self.scheme = scheme
        self.tilelang_dtype = _tilelang_dtype(dtype)
        self.output_dtype = _torch_dtype(dtype)
        self.kernel = _compile_tilelang_gemv(scheme, rows, channels, group_size, self.tilelang_dtype, 1)
        self.weights = _jax_to_torch(packed_weights).view(torch.int8).contiguous()
        self.scales = _jax_to_torch(scales).to(self.output_dtype).contiguous()
        if affine.__class__.__module__.startswith("torch"):
            self.affine = affine.to(self.output_dtype).contiguous()
        else:
            self.affine = _jax_to_torch(affine).to(self.output_dtype).contiguous()
        self.outputs = {1: torch.empty((1, rows), dtype=self.output_dtype, device=self.weights.device)}

    def __call__(self, inputs):
        if inputs.__class__.__module__.startswith("torch"):
            x = inputs.reshape(-1, self.channels)
        else:
            x = _jax_to_torch(inputs).reshape(-1, self.channels)
        if x.dtype != self.output_dtype:
            x = x.to(self.output_dtype)
        if not x.is_contiguous():
            x = x.contiguous()
        batch = x.shape[0]
        output = self.outputs.get(batch)
        if output is None:
            output = _require_torch().empty(
                (batch, self.rows),
                dtype=self.output_dtype,
                device=self.weights.device,
            )
            self.outputs[batch] = output
        kernel = (
            self.kernel
            if batch == 1
            else _compile_tilelang_gemv(
                self.scheme,
                self.rows,
                self.channels,
                self.group_size,
                self.tilelang_dtype,
                batch,
            )
        )
        kernel(x, self.weights, self.scales, self.affine, output)
        return output[0] if inputs.ndim == 1 else output

    def jax(self, vector: Array) -> Array:
        """Correctness-only bridge; the fast path keeps tensors in torch."""
        return jnp.from_dlpack(self(vector).detach())


def _jax_to_torch(value: Array):
    torch = _require_torch()
    from torch.utils import dlpack as torch_dlpack

    if value.dtype == jnp.bfloat16:
        return torch_dlpack.from_dlpack(value.view(jnp.uint16)).view(torch.bfloat16)
    return torch_dlpack.from_dlpack(value)


def _require_torch():
    try:
        import torch
    except ImportError as error:
        raise ImportError("TileLang compressed dot requires torch.") from error
    return torch


def _require_tilelang():
    try:
        import tilelang
        from tilelang import language as T
        from tilelang.quantize import _tir_packed_to_unsigned_convert
    except ImportError as error:
        raise ImportError("TileLang compressed dot requires tilelang.") from error
    return tilelang, T, _tir_packed_to_unsigned_convert


def _torch_dtype(dtype: DTypeLike):
    torch = _require_torch()
    if dtype == jnp.float16:
        return torch.float16
    if dtype == jnp.bfloat16:
        return torch.bfloat16
    raise TypeError(f"Unsupported TileLang dot dtype: {dtype}")


def _tilelang_dtype(dtype: DTypeLike) -> str:
    if dtype == jnp.float16:
        return "float16"
    if dtype == jnp.bfloat16:
        return "bfloat16"
    raise TypeError(f"Unsupported TileLang dot dtype: {dtype}")


@lru_cache(maxsize=32)
def _compile_tilelang_gemv(
    scheme: _TileLangScheme,
    rows: int,
    channels: int,
    group_size: int,
    dtype: str,
    batch: int,
) -> Callable[..., None]:
    tilelang, T, unpack_uint4 = _require_tilelang()
    reduce_thread = 64
    rows_per_block = 2
    micro_k = 16
    packed_micro_k = micro_k // 2
    block_k = reduce_thread * micro_k
    groups = channels // group_size
    assert rows % rows_per_block == 0
    assert channels % block_k == 0

    @tilelang.jit
    def kernel_factory(
        rows: int,
        channels: int,
        group_size: int,
        groups: int,
        dtype: str,
        rows_per_block: int,
        batch: int,
    ):
        @T.prim_func
        def gemv(
            vectors: T.Tensor((batch, channels), dtype),
            weights: T.Tensor((rows, channels // 2), "int8"),
            scales: T.Tensor((rows, groups), dtype),
            affine: T.Tensor((rows, groups), dtype),
            output: T.Tensor((batch, rows), dtype),
        ):
            with T.Kernel(rows // rows_per_block, batch, threads=(reduce_thread, rows_per_block)) as (
                row_block,
                batch_index,
            ):
                vector_values = T.alloc_local((micro_k,), dtype)
                packed_values = T.alloc_local((packed_micro_k,), "int8")
                unpacked_values = T.alloc_local((micro_k,), dtype)
                acc = T.alloc_local((1,), "float32")
                int_dot = T.alloc_local((1,), "float32")
                vector_sum = T.alloc_local((1,), "float32")
                partials = T.alloc_shared((rows_per_block, reduce_thread), "float32")
                reduced = T.alloc_shared((rows_per_block,), "float32")
                thread = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
                row_offset = T.thread_binding(0, rows_per_block, thread="threadIdx.y")
                row = row_block * rows_per_block + row_offset
                T.clear(acc)

                for block in T.serial(channels // block_k):
                    for offset in T.vectorized(micro_k):
                        channel = block * block_k + thread * micro_k + offset
                        vector_values[offset] = vectors[batch_index, channel]

                    for offset in T.vectorized(packed_micro_k):
                        packed_channel = block * (reduce_thread * packed_micro_k) + thread * packed_micro_k + offset
                        packed_values[offset] = weights[row, packed_channel]

                    group = (block * block_k + thread * micro_k) // group_size
                    scale = scales[row, group]
                    affine_value = affine[row, group]
                    unpack = unpack_uint4("int", 8)
                    for offset in T.serial(micro_k):
                        unpacked_values[offset] = unpack(4, packed_values[offset // 2], offset % 2, dtype)

                    T.clear(int_dot)
                    T.clear(vector_sum)
                    for offset in T.serial(micro_k):
                        int_value = T.Cast("float32", unpacked_values[offset])
                        vector_value = T.Cast("float32", vector_values[offset])
                        int_dot[0] += int_value * vector_value
                        vector_sum[0] += vector_value

                    scale_value = T.Cast("float32", scale)
                    affine_float = T.Cast("float32", affine_value)
                    if scheme == "mlx":
                        acc[0] += int_dot[0] * scale_value + vector_sum[0] * affine_float
                    else:
                        acc[0] += (int_dot[0] - affine_float * vector_sum[0]) * scale_value

                partials[row_offset, thread] = acc[0]
                T.reduce_sum(partials, reduced, dim=1, clear=True)
                if thread == 0:
                    output[batch_index, row] = reduced[row_offset]

        return gemv

    return kernel_factory(rows, channels, group_size, groups, dtype, rows_per_block, batch)
