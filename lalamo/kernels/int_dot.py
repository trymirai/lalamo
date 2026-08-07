from collections.abc import Callable
from functools import cache
from math import sqrt
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.pallas import mosaic_gpu as plgpu
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith, llvm
from jaxtyping import Array

from lalamo.kernels.hadamard import fragmented_hadamard

type _Ref = Any


def supports_batched_int_dot(
    batch_size: int,
    rows: int,
    columns: int,
    group_size: int,
    bits: int,
    is_symmetric: bool,
    input_block_size: Literal[32, 64, 128] | None,
) -> bool:
    return (
        batch_size >= 8
        and rows >= 128
        and rows % 128 == 0
        and columns >= 512
        and columns % 256 == 0
        and group_size == 64
        and input_block_size == 32
        and (bits, is_symmetric) in ((4, False), (8, True))
    )


@cache
def make_int_dot(
    rows: int,
    columns: int,
    group_size: int,
    bits: int,
    is_symmetric: bool,
    input_block_size: Literal[32, 64, 128] | None = None,
    *,
    use_sm100_batched: bool = False,
) -> Callable[[Array, Array, Array, Array, Array], Array]:
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

    prepare_rht = None
    if input_block_size is not None and input_block_size <= group_size:
        blocks_per_group = group_size // input_block_size
        groups_per_program = next(candidate for candidate in (4, 2, 1) if group_count % candidate == 0)
        if groups_per_program == 1:
            warp_dims = (plgpu.Replicated(4),)
        elif groups_per_program == 4:
            warp_dims = (-4,)
        else:
            warp_dims = (-4, plgpu.Replicated(4 // groups_per_program))
        input_layout = plgpu.Layout.TILED(
            plgpu.Tiling(((groups_per_program, 1, 32), (1,))),
            warp_dims=warp_dims,
            lane_dims=(-2,),
            vector_dim=-1,
        )
        activation_parameter_layout = input_layout.reduce((1, 2))

        @plgpu.inline_mgpu(
            arg_types=(input_layout,),
            return_type=(
                plgpu.ShapeDtypeStruct(
                    (groups_per_program, blocks_per_group, input_block_size),
                    jnp.int8,
                    layout=input_layout,
                ),
                plgpu.ShapeDtypeStruct(
                    (groups_per_program,),
                    jnp.float32,
                    layout=activation_parameter_layout,
                ),
                plgpu.ShapeDtypeStruct(
                    (groups_per_program,),
                    jnp.int32,
                    layout=activation_parameter_layout,
                ),
            ),
        )
        def prepare_activations(
            _ctx: object,
            values: mgpu.FragmentedArray,
        ) -> tuple[
            mgpu.FragmentedArray,
            mgpu.FragmentedArray,
            mgpu.FragmentedArray,
        ]:
            transformed = fragmented_hadamard(
                values,
                input_block_size,
                blocks_per_warp=blocks_per_group,
            ) / sqrt(input_block_size)
            transformed = transformed.astype(ir.BF16Type.get()).astype(ir.F32Type.get())
            activation_scale = (transformed.abs().reduce("max", axis=(1, 2)) / 127).max(
                float(jnp.finfo(jnp.float32).tiny)
            )
            broadcast_scale = activation_scale.broadcast_in_dim(
                transformed.shape,
                (0,),
                transformed.layout,
            )
            quantized = (
                (transformed / broadcast_scale)
                .round_even()
                .max(-127)
                .min(127)
                .astype(
                    ir.IntegerType.get_signless(8),
                    is_signed=True,
                )
            )
            activation_sums = quantized.astype(
                ir.IntegerType.get_signless(32),
                is_signed=True,
            ).reduce("add", axis=(1, 2))
            return (
                quantized,
                activation_scale,
                activation_sums,
            )

        def prepare_rht_kernel(
            vector_ref: jax.Ref,
            signs_ref: jax.Ref,
            quantized_vector_ref: jax.Ref,
            activation_scales_ref: jax.Ref,
            activation_sums_ref: jax.Ref,
        ) -> None:
            group_slice = pl.ds(
                pl.program_id(0) * groups_per_program,
                groups_per_program,
            )
            values = plgpu.load(
                vector_ref.at[group_slice, :, :],
                layout=input_layout,
                optimized=False,
            ).astype(jnp.float32)
            signs = plgpu.load(
                signs_ref.at[group_slice, :, :],
                layout=input_layout,
                optimized=False,
            ).astype(jnp.float32)
            quantized, scales, activation_sums = prepare_activations(values * signs)
            quantized_vector_ref.at[group_slice, :, :][...] = quantized
            activation_scales_ref.at[group_slice][...] = scales
            activation_sums_ref.at[group_slice][...] = activation_sums

        blocked_shape = (group_count, blocks_per_group, input_block_size)
        prepare_rht = plgpu.kernel(
            prepare_rht_kernel,
            out_type=(
                jax.ShapeDtypeStruct(blocked_shape, jnp.int8),
                jax.ShapeDtypeStruct((group_count,), jnp.float32),
                jax.ShapeDtypeStruct((group_count,), jnp.int32),
            ),
            grid=(group_count // groups_per_program,),
            grid_names=("program",),
            compiler_params=plgpu.CompilerParams(
                lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
            ),
        )

    @jax.custom_batching.custom_vmap
    def dot(
        vector: Array,
        input_signs: Array,
        packed_weights: Array,
        scales: Array,
        packed_zero_points: Array,
    ) -> Array:
        if prepare_rht is None:
            if input_block_size is not None:
                from lalamo.compressed.utils.hadamard import hadamard_transform  # noqa: PLC0415

                vector = hadamard_transform(
                    vector * input_signs.astype(vector.dtype),
                    input_block_size,
                )
            quantized_vector, activation_scales, activation_sums = quantize(vector)
        else:
            assert input_block_size is not None
            blocked_shape = (
                group_count,
                group_size // input_block_size,
                input_block_size,
            )
            quantized_vector, activation_scales, activation_sums = prepare_rht(
                vector.reshape(blocked_shape),
                input_signs.reshape(blocked_shape),
            )
            quantized_vector = quantized_vector.reshape(columns)
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
        input_signs: Array,
        packed_weights: Array,
        scales: Array,
        packed_zero_points: Array,
    ) -> tuple[Array, bool]:
        if in_batched != [True, False, False, False, False]:
            raise ValueError("Only the input vectors may be batched.")
        if axis_size == 1:
            return (
                dot(
                    vectors[0],
                    input_signs,
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
            input_block_size,
        )
        if use_batched_matmul:
            assert input_block_size is not None
            from lalamo.kernels.hadamard import (  # noqa: PLC0415
                gpu_signed_hadamard_transform,
            )
            from lalamo.kernels.int_matmul import (  # noqa: PLC0415
                make_batched_int_matmul,
            )

            transformed_vectors = gpu_signed_hadamard_transform(
                vectors,
                input_signs,
                input_block_size,
                permuted_output=bits == 4,
            )
            return (
                make_batched_int_matmul(
                    axis_size,
                    rows,
                    columns,
                    bits,
                    is_symmetric,
                )(
                    transformed_vectors,
                    packed_weights,
                    scales.astype(vectors.dtype),
                    packed_zero_points,
                ),
                True,
            )

        from lalamo.compressed.int import _packed_weights_to_master_weights  # noqa: PLC0415

        zero_points = packed_zero_points
        if is_symmetric:
            zero_points = None
        weights = _packed_weights_to_master_weights(
            packed_weights,
            scales.astype(vectors.dtype),
            zero_points,
            group_size,
            bits,
        )
        if input_block_size is not None:
            from lalamo.compressed.utils.hadamard import hadamard_transform  # noqa: PLC0415

            vectors = hadamard_transform(
                vectors * input_signs.astype(vectors.dtype),
                input_block_size,
            )
        return (
            vectors @ weights.swapaxes(-1, -2),
            True,
        )

    return dot
