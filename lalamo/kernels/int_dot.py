from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.lax import DotAlgorithmPreset
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith, llvm
from jaxtyping import Array, Float, UInt8

from lalamo.utils.packing import unpack_uint8_to_uint
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.sharding import sharding_of, supports_mosaic_gpu
from lalamo.weight_matrix import Layout

if TYPE_CHECKING:
    from jax.sharding import Mesh

type _Ref = Any

__all__ = [
    "dequantize_int_weights",
    "int_dot",
    "supports_batched_int_dot",
]


def dequantize_int_weights(
    packed_weights: UInt8[Array, "... packed_cols"],
    scales: Float[Array, "... groups"],
    packed_zero_points: UInt8[Array, "... packed_groups"] | None,
    group_size: int,
    bits: int,
) -> Float[Array, "... cols"]:
    int_weights = unpack_uint8_to_uint(
        packed_weights,
        bits=bits,
        dtype=scales.dtype,
        unpacked_last_axis_dim=scales.shape[-1] * group_size,
    )
    if packed_zero_points is None:
        int_zero_points: Array | int = 2 ** (bits - 1)
    else:
        int_zero_points = unpack_uint8_to_uint(
            packed_zero_points,
            bits=bits,
            dtype=scales.dtype,
            unpacked_last_axis_dim=scales.shape[-1],
        )[..., None]

    *leading_dims, num_columns = int_weights.shape
    grouped_weights = int_weights.reshape(*leading_dims, num_columns // group_size, group_size)
    return ((grouped_weights - int_zero_points) * scales[..., None]).reshape(int_weights.shape)


def supports_batched_int_dot(
    batch_size: int,
    rows: int,
    columns: int,
    group_size: int,
    bits: int,
    is_symmetric: bool,
) -> bool:
    return (
        batch_size >= 8
        and rows >= 128
        and rows % 128 == 0
        and columns >= 512
        and columns % max(256, 8 * group_size) == 0
        and group_size >= 8
        and 128 % group_size == 0
        and (bits, is_symmetric) in ((4, False), (8, True))
    )


def int_dot(
    vector: Float[Array, " source_channels"],
    packed_weights: UInt8[Array, "rows packed_cols"],
    scales: Float[Array, "rows groups"],
    packed_zero_points: UInt8[Array, "rows packed_groups"] | None,
    *,
    group_size: int,
    bits: int,
    is_symmetric: bool,
    layout: Layout,
    precision: DotAlgorithmPreset,
    transposed: bool = False,
) -> Float[Array, " target_channels"]:
    rows = packed_weights.shape[0]
    columns = scales.shape[-1] * group_size
    complete_rows = rows - rows % 64
    mesh = cast("Mesh", sharding_of(packed_weights).mesh)
    if (
        supports_mosaic_gpu(mesh, 9)
        and layout == Layout.OUTPUT_INPUT
        and vector.dtype == jnp.bfloat16
        and precision == DotAlgorithmPreset.DEFAULT
        and not transposed
        and complete_rows > 0
        and group_size in (16, 32, 64, 128)
    ):
        if packed_zero_points is None:
            main_zero_points = packed_weights[:complete_rows, :0]
        else:
            main_zero_points = packed_zero_points[:complete_rows]
        output = _make_int_dot(
            complete_rows,
            columns,
            group_size,
            bits,
            is_symmetric,
            use_sm100_batched=supports_mosaic_gpu(mesh, 10),
        )(
            vector,
            packed_weights[:complete_rows],
            scales[:complete_rows].astype(vector.dtype),
            main_zero_points,
        )
        if complete_rows == rows:
            return output

        tail_zero_points = None
        if packed_zero_points is not None:
            tail_zero_points = packed_zero_points[complete_rows:]
        tail_weights = dequantize_int_weights(
            packed_weights[complete_rows:],
            scales[complete_rows:].astype(vector.dtype),
            tail_zero_points,
            group_size,
            bits,
        )
        return jnp.concatenate((output, tail_weights @ vector))

    weights = dequantize_int_weights(
        packed_weights,
        scales.astype(vector.dtype),
        packed_zero_points,
        group_size,
        bits,
    )
    if transposed:
        layout = layout.transpose()
    with use_dot_algorithm_preset(precision):
        return layout.matmul(weights, vector)


@cache
def _make_int_dot(
    rows: int,
    columns: int,
    group_size: int,
    bits: int,
    is_symmetric: bool,
    *,
    use_sm100_batched: bool = False,
) -> Callable[[Array, Array, Array, Array], Array]:
    rows_per_program = 64
    values_per_byte = 8 // bits
    packed_group_size = group_size // values_per_byte
    group_count = columns // group_size
    row_programs = rows // rows_per_program
    splits = min(
        (
            split
            for split in range(1, group_count + 1)
            if group_count % split == 0 and rows_per_program * columns // values_per_byte // split <= 196_608
        ),
        key=lambda split: abs(row_programs * split - 1_440),
    )
    groups_per_split = group_count // splits
    packed_split_size = groups_per_split * packed_group_size
    staged_packed_split_size = packed_split_size
    if packed_group_size < 32:
        staged_packed_split_size = (packed_split_size + 15) // 16 * 16
        if staged_packed_split_size % 32 == 0:
            staged_packed_split_size += 16

    weight_layout = plgpu.Layout.TILED(
        plgpu.Tiling(((64, 8), (16, 8), (8, 8), (4,))),
        warp_dims=(-7,),
        lane_dims=(-5, -3, -2),
        vector_dim=-1,
    )
    row_layout = weight_layout.reduce(1)
    activation_layout = plgpu.Layout.TILED(
        plgpu.Tiling(
            (
                (64, 8 * values_per_byte),
                (16, 8 * values_per_byte),
                (8, 8 * values_per_byte),
                (4 * values_per_byte,),
            )
        ),
        warp_dims=(-7,),
        lane_dims=(-5, -3, -2),
        vector_dim=-1,
    )
    activation_column_layout = activation_layout.reduce(0)
    compiler_params = plgpu.CompilerParams(
        lowering_semantics=plgpu.LoweringSemantics.Lane,
        reduction_scratch_bytes=6_144,
    )

    @plgpu.inline_mgpu(
        arg_types=(weight_layout, activation_layout),
        return_type=plgpu.ShapeDtypeStruct(
            (rows_per_program,),
            jnp.int32,
            layout=row_layout,
        ),
    )
    def packed_dot(
        _ctx: object,
        weights: mgpu.FragmentedArray,
        activations: mgpu.FragmentedArray,
    ) -> mgpu.FragmentedArray:
        i32 = ir.IntegerType.get_signless(32)
        accumulator = cast("ir.Value", mgpu.c(0, i32))
        for register_index, weight_register in enumerate(weights.registers.flat):
            packed_weight = mgpu.utils.bitcast(weight_register, i32)
            if bits == 4:
                i64 = ir.IntegerType.get_signless(64)
                packed_activations = mgpu.utils.bitcast(
                    activations.registers.flat[register_index],
                    i64,
                )
                low_activations = arith.trunci(i32, packed_activations)
                high_activations = arith.trunci(
                    i32,
                    arith.shrui(packed_activations, mgpu.c(32, i64)),
                )
                activation_words = (
                    llvm.inline_asm(
                        i32,
                        [low_activations, high_activations],
                        "prmt.b32 $0, $1, $2, 0x6420;",
                        "=r,r,r",
                        has_side_effects=False,
                    ),
                    llvm.inline_asm(
                        i32,
                        [low_activations, high_activations],
                        "prmt.b32 $0, $1, $2, 0x7531;",
                        "=r,r,r",
                        has_side_effects=False,
                    ),
                )
            else:
                activation_words = (
                    mgpu.utils.bitcast(
                        activations.registers.flat[register_index],
                        i32,
                    ),
                )
            for value_index, packed_vector in enumerate(activation_words):
                if bits == 4:
                    packed_weight_values = arith.andi(
                        arith.shrui(
                            packed_weight,
                            mgpu.c(value_index * bits, i32),
                        ),
                        mgpu.c(0x0F0F0F0F, i32),
                    )
                else:
                    packed_weight_values = packed_weight
                accumulator = cast(
                    "ir.Value",
                    llvm.inline_asm(
                        i32,
                        [
                            cast("ir.Value", packed_weight_values),
                            cast("ir.Value", packed_vector),
                            accumulator,
                        ],
                        "dp4a.u32.s32 $0, $1, $2, $3;",
                        "=r,r,r,r",
                        has_side_effects=False,
                    ),
                )
        accumulator = mgpu.utils.warp_tree_reduce(
            accumulator,
            arith.addi,
            group_size=2,
        )
        return mgpu.FragmentedArray.splat(
            accumulator,
            (rows_per_program,),
            layout=row_layout.to_mgpu(),
            is_signed=True,
        )

    def kernel(
        quantized_vector_ref: jax.Ref,
        activation_scales_ref: jax.Ref,
        activation_sums_ref: jax.Ref,
        packed_weights_ref: jax.Ref,
        scales_ref: jax.Ref,
        packed_zero_points_ref: jax.Ref,
        partial_output_ref: jax.Ref,
        packed_weights_smem_ref: _Ref,
        copy_barrier_ref: jax.Ref,
    ) -> None:
        split_index = pl.program_id(0)
        row_slice = pl.ds(
            pl.program_id(1) * rows_per_program,
            rows_per_program,
        )
        plgpu.copy_gmem_to_smem(
            packed_weights_ref.at[
                row_slice,
                pl.ds(split_index * packed_split_size, staged_packed_split_size),
            ],
            packed_weights_smem_ref,  # pyrefly: ignore[bad-argument-type]
            copy_barrier_ref,  # pyrefly: ignore[bad-argument-type]
            oob_mode=plgpu.OOBFillMode.ZEROS,
        )
        plgpu.barrier_wait(copy_barrier_ref)  # pyrefly: ignore[bad-argument-type]

        @pl.loop(  # pyrefly: ignore[bad-argument-type]
            0,
            groups_per_split,
            init_carry=plgpu.layout_cast(
                jnp.zeros((rows_per_program,), jnp.float32),
                row_layout,
            ),
            unroll=True,
        )
        def accumulate(
            local_group_index: jax.Array,
            accumulator: jax.Array,
        ) -> jax.Array:
            group_index = split_index * groups_per_split + local_group_index
            packed_weights = plgpu.load(
                packed_weights_smem_ref.at[
                    :,
                    pl.ds(
                        local_group_index * packed_group_size,
                        packed_group_size,
                    ),
                ],
                layout=weight_layout,
                optimized=packed_group_size != 8,
            )
            activation_tile = plgpu.layout_cast(
                jax.lax.broadcast_in_dim(
                    plgpu.load(
                        quantized_vector_ref.at[pl.ds(group_index * group_size, group_size)],
                        layout=activation_column_layout,
                        optimized=False,
                    ),
                    (rows_per_program, group_size),
                    (1,),
                ),
                activation_layout,
            )
            integer_dot = packed_dot(packed_weights, activation_tile)
            scales = plgpu.load(
                scales_ref.at[row_slice, group_index],
                layout=row_layout,
                optimized=False,
            ).astype(jnp.float32)
            if is_symmetric:
                zero_points = 2 ** (bits - 1)
            else:
                packed_zero_points = plgpu.load(
                    packed_zero_points_ref.at[
                        row_slice,
                        group_index // values_per_byte,
                    ],
                    layout=row_layout,
                    optimized=False,
                ).astype(jnp.int32)
                zero_points = (packed_zero_points >> ((group_index % values_per_byte) * bits)) & ((1 << bits) - 1)
            corrected_dot = integer_dot - zero_points * activation_sums_ref[group_index]
            return accumulator + corrected_dot.astype(jnp.float32) * activation_scales_ref[group_index] * scales

        partial_output_ref.at[split_index, row_slice][...] = accumulate

    packed_weights_smem = plgpu.SMEM(
        (rows_per_program, staged_packed_split_size),
        jnp.uint8,
    )
    if packed_group_size > 16:
        packed_weights_smem = plgpu.SMEM(
            (rows_per_program, staged_packed_split_size),
            jnp.uint8,
            transforms=(
                plgpu.TilingTransform((8, packed_group_size)),
                plgpu.SwizzleTransform(packed_group_size),
            ),
        )

    matmul = plgpu.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct((splits, rows), jnp.float32),
        grid=(splits, row_programs),
        grid_names=("split", "row"),
        scratch_types=[
            packed_weights_smem,
            plgpu.Barrier(),
        ],
        compiler_params=compiler_params,
    )

    def quantize_kernel(
        vector_ref: jax.Ref,
        quantized_vector_ref: jax.Ref,
        activation_scales_ref: jax.Ref,
        activation_sums_ref: jax.Ref,
    ) -> None:
        group_index = pl.program_id(0)
        group_slice = pl.ds(
            group_index * group_size,
            group_size,
        )
        vector = plgpu.load(
            vector_ref.at[group_slice],
            layout=activation_column_layout,
            optimized=False,
        ).astype(jnp.float32)
        activation_scale = jnp.maximum(
            jnp.max(jnp.abs(vector)) / 127,
            jnp.finfo(jnp.float32).tiny,
        )
        quantized_vector = jnp.clip(
            jnp.rint(vector / activation_scale),
            -127,
            127,
        ).astype(jnp.int8)
        quantized_vector_ref.at[group_slice][...] = quantized_vector
        activation_scales_ref[group_index] = activation_scale
        activation_sums_ref[group_index] = jnp.sum(
            quantized_vector,
            dtype=jnp.int32,
        )

    quantize = plgpu.kernel(
        quantize_kernel,
        out_type=(
            jax.ShapeDtypeStruct(
                (columns,),
                jnp.int8,
            ),
            jax.ShapeDtypeStruct((group_count,), jnp.float32),
            jax.ShapeDtypeStruct((group_count,), jnp.int32),
        ),
        grid=(group_count,),
        grid_names=("group",),
        compiler_params=compiler_params,
    )

    @jax.custom_batching.custom_vmap
    def dot(
        vector: Array,
        packed_weights: Array,
        scales: Array,
        packed_zero_points: Array,
    ) -> Array:
        quantized_vector, activation_scales, activation_sums = quantize(vector)
        partials = matmul(
            quantized_vector,
            activation_scales,
            activation_sums,
            packed_weights,
            scales,
            packed_zero_points,
        )
        return jnp.sum(partials, axis=0).astype(vector.dtype)

    @dot.def_vmap
    def dot_vmap(
        axis_size: int,
        in_batched: list[bool],
        vectors: Array,
        packed_weights: Array,
        scales: Array,
        packed_zero_points: Array,
    ) -> tuple[Array, bool]:
        if in_batched != [True, False, False, False]:
            raise ValueError("Only the input vectors may be batched.")
        if axis_size == 1:
            return (
                dot(
                    vectors[0],
                    packed_weights,
                    scales,
                    packed_zero_points,
                )[None],
                True,
            )

        use_batched_matmul = use_sm100_batched and supports_batched_int_dot(
            axis_size,
            rows,
            columns,
            group_size,
            bits,
            is_symmetric,
        )
        if use_batched_matmul:
            from lalamo.kernels.int_matmul import (  # noqa: PLC0415
                _make_batched_int_matmul,
            )

            return (
                _make_batched_int_matmul(
                    axis_size,
                    rows,
                    columns,
                    group_size,
                    bits,
                    is_symmetric,
                )(
                    vectors,
                    packed_weights,
                    scales.astype(vectors.dtype),
                    packed_zero_points,
                ),
                True,
            )

        zero_points = packed_zero_points
        if is_symmetric:
            zero_points = None
        weights = dequantize_int_weights(
            packed_weights,
            scales.astype(vectors.dtype),
            zero_points,
            group_size,
            bits,
        )
        return (
            vectors @ weights.swapaxes(-1, -2),
            True,
        )

    return dot
