# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
from functools import cache
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.pallas import mosaic_gpu as plgpu
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith, llvm, memref, vector
from jaxtyping import Array

type _Ref = Any
type BatchedIntMatmul = Callable[[Array, Array, Array, Array], Array]

_MAIN_WARPGROUP = 0
_DEQUANT_LOW_WARPGROUP = 1
_DEQUANT_HIGH_WARPGROUP = 2
_STORE_WARPGROUP = _DEQUANT_HIGH_WARPGROUP
_MMA_WARP = 0
_WEIGHT_TMA_WARP = 1
_ACTIVATION_TMA_WARP = 2
_METADATA_TMA_WARP = 3
_TMEM = plgpu.Layout.TCGEN05_TMEM_NATIVE


def _int4_to_bf16(
    weights: Array,
    half: int,
) -> Array:
    @plgpu.inline_mgpu(
        arg_types=(_TMEM(4),),
        return_type=plgpu.ShapeDtypeStruct(
            weights.shape,
            jnp.bfloat16,
            _TMEM(8),
        ),
    )
    def convert(
        _ctx: object,
        values: mgpu.FragmentedArray,
    ) -> mgpu.FragmentedArray:
        i32 = ir.IntegerType.get_signless(32)
        bf16_pair = ir.VectorType.get((2,), ir.BF16Type.get())
        registers_per_half = values.registers.shape[1] // 2
        half_registers = values.registers[
            :,
            half * registers_per_half : (half + 1) * registers_per_half,
        ]
        registers = np.empty_like(half_registers)
        for index, register in np.ndenumerate(half_registers):
            packed_values = mgpu.utils.bitcast(register, i32)
            converted = llvm.inline_asm(
                llvm.StructType.get_literal([i32] * 4),
                [packed_values],
                """
                {
                .reg .b32 evens, odds;
                .reg .b16 part<4>;
                .reg .b16 scale;
                lop3.b32 evens, $4, 0x0F0F0F0F, 0x08080808, (0xf0 & 0xcc) ^ 0xaa;
                lop3.b32 odds, $4, 0xF0F0F0F0, 0x80808080, (0xf0 & 0xcc) ^ 0xaa;
                shl.b32 evens, evens, 4;
                mov.b32 {part0, part1}, evens;
                mov.b32 {part2, part3}, odds;
                mov.b16 scale, 0x8181;
                cvt.rn.scaled::n2::ue8m0.bf16x2.s2f6x2 $0, part0, scale;
                cvt.rn.scaled::n2::ue8m0.bf16x2.s2f6x2 $1, part1, scale;
                cvt.rn.scaled::n2::ue8m0.bf16x2.s2f6x2 $2, part2, scale;
                cvt.rn.scaled::n2::ue8m0.bf16x2.s2f6x2 $3, part3, scale;
                }
                """,
                "=r,=r,=r,=r,r",
            )
            if not isinstance(converted, ir.Value):
                raise TypeError(f"Expected packed converter result, got {converted}.")
            converted_values = mgpu.utils.vector_concat(
                [
                    mgpu.utils.bitcast(
                        llvm.extractvalue(i32, converted, (part,)),
                        bf16_pair,
                    )
                    for part in range(4)
                ]
            )
            registers[index] = vector.shuffle(
                converted_values,
                converted_values,
                [0, 4, 1, 5, 2, 6, 3, 7],
            )
        return mgpu.FragmentedArray(
            _registers=registers,
            _layout=mgpu.tmem_native_layout(8),
            _is_signed=None,
        )

    return convert(weights)


def _dequantize_int4(
    scales_ref: _Ref,
    zero_points_ref: _Ref,
    weights: Array,
    group_offset: jax.Array,
    group_size: int,
) -> Array:
    @plgpu.inline_mgpu(
        arg_types=(
            plgpu.RefType(),
            plgpu.RefType(),
            _TMEM(8),
            plgpu.Layout.WG_SPLAT,
        ),
        return_type=plgpu.ShapeDtypeStruct(
            weights.shape,
            scales_ref.dtype,
            _TMEM(8),
        ),
    )
    def apply_affine(
        _ctx: object,
        scales_smem: _Ref,
        zero_points_smem: _Ref,
        values: mgpu.FragmentedArray,
        first_group: mgpu.FragmentedArray,
    ) -> mgpu.FragmentedArray:
        zero_point_row_stride = zero_points_smem.type.shape[0] * zero_points_smem.type.shape[1] // values.shape[0]
        first_group_value = arith.index_castui(
            ir.IndexType.get(),
            cast("ir.Value", first_group.registers[()]),
        )
        logical_row = next(iter(values.layout.thread_idxs(values.shape)))[0]
        scales = []
        biases = []
        for scale_index in range(scales_smem.type.shape[1]):
            group_index = arith.addi(
                first_group_value,
                arith.constant(ir.IndexType.get(), scale_index),
            )
            zero_point_index = arith.addi(
                arith.muli(
                    logical_row,
                    arith.constant(ir.IndexType.get(), zero_point_row_stride),
                ),
                arith.divui(
                    group_index,
                    arith.constant(ir.IndexType.get(), 2),
                ),
            )
            packed_zero_points = memref.load(
                zero_points_smem,
                (
                    arith.divui(
                        zero_point_index,
                        arith.constant(
                            ir.IndexType.get(),
                            zero_points_smem.type.shape[1],
                        ),
                    ),
                    arith.remui(
                        zero_point_index,
                        arith.constant(
                            ir.IndexType.get(),
                            zero_points_smem.type.shape[1],
                        ),
                    ),
                ),
            )
            zero_point_shift = arith.index_castui(
                ir.IntegerType.get_signless(8),
                arith.muli(
                    arith.remui(
                        group_index,
                        arith.constant(ir.IndexType.get(), 2),
                    ),
                    arith.constant(ir.IndexType.get(), 4),
                ),
            )
            zero_point_integer = arith.andi(
                arith.shrui(
                    packed_zero_points,
                    zero_point_shift,
                ),
                mgpu.c(0xF, ir.IntegerType.get_signless(8)),
            )
            i32 = ir.IntegerType.get_signless(32)
            centered_zero_point = arith.subi(
                arith.extui(i32, zero_point_integer),
                mgpu.c(8, i32),
            )
            zero_point = arith.sitofp(ir.BF16Type.get(), centered_zero_point)
            scale = memref.load(
                scales_smem,
                (
                    logical_row,
                    arith.constant(ir.IndexType.get(), scale_index),
                ),
            )
            scales.append(scale)
            biases.append(
                arith.negf(
                    arith.mulf(
                        zero_point,
                        scale,
                        fastmath=arith.FastMathFlags.fast,
                    ),
                    fastmath=arith.FastMathFlags.fast,
                )
            )

        registers = np.empty_like(values.registers)
        registers_per_group = group_size // 8
        for register_index, register in np.ndenumerate(values.registers):
            scale_index = register_index[1] // registers_per_group
            scale = vector.broadcast(register.type, scales[scale_index])
            bias = vector.broadcast(register.type, biases[scale_index])
            registers[register_index] = arith.addf(
                arith.mulf(
                    register,
                    scale,
                    fastmath=arith.FastMathFlags.fast,
                ),
                bias,
                fastmath=arith.FastMathFlags.fast,
            )
        return mgpu.FragmentedArray(
            _registers=registers,
            _layout=values.layout,
            _is_signed=None,
        )

    return apply_affine(
        scales_ref,
        zero_points_ref,
        weights,
        group_offset,
    )


def _dequantize_int8(
    scales_ref: _Ref,
    weights: Array,
    group_size: int,
) -> Array:
    @plgpu.inline_mgpu(
        arg_types=(plgpu.RefType(), _TMEM(8)),
        return_type=plgpu.ShapeDtypeStruct(
            weights.shape,
            scales_ref.dtype,
            _TMEM(8),
        ),
    )
    def convert_and_scale(
        _ctx: object,
        scales_smem: _Ref,
        values: mgpu.FragmentedArray,
    ) -> mgpu.FragmentedArray:
        i32 = ir.IntegerType.get_signless(32)
        i32_pair = ir.VectorType.get((2,), i32)
        bf16_pair = ir.VectorType.get((2,), ir.BF16Type.get())
        logical_row = next(iter(values.layout.thread_idxs(values.shape)))[0]
        scales = [
            memref.load(
                scales_smem,
                (
                    logical_row,
                    arith.constant(ir.IndexType.get(), scale_index),
                ),
            )
            for scale_index in range(scales_smem.type.shape[1])
        ]
        registers = np.empty_like(values.registers)
        registers_per_group = group_size // 8
        for register_index, register in np.ndenumerate(values.registers):
            packed_pair = mgpu.utils.bitcast(register, i32_pair)
            packed_values = [vector.extract(packed_pair, [], [part]) for part in range(2)]
            converted = llvm.inline_asm(
                llvm.StructType.get_literal([i32] * 4),
                packed_values,
                """
                {
                .reg .b32 signed0, signed1;
                .reg .b16 part<4>;
                .reg .b16 scale;
                xor.b32 signed0, $4, 0x80808080;
                xor.b32 signed1, $5, 0x80808080;
                mov.b32 {part0, part1}, signed0;
                mov.b32 {part2, part3}, signed1;
                mov.b16 scale, 0x8585;
                cvt.rn.scaled::n2::ue8m0.bf16x2.s2f6x2 $0, part0, scale;
                cvt.rn.scaled::n2::ue8m0.bf16x2.s2f6x2 $1, part1, scale;
                cvt.rn.scaled::n2::ue8m0.bf16x2.s2f6x2 $2, part2, scale;
                cvt.rn.scaled::n2::ue8m0.bf16x2.s2f6x2 $3, part3, scale;
                }
                """,
                "=r,=r,=r,=r,r,r",
                has_side_effects=False,
            )
            if not isinstance(converted, ir.Value):
                raise TypeError(f"Expected packed converter result, got {converted}.")
            converted_values = mgpu.utils.vector_concat(
                [
                    mgpu.utils.bitcast(
                        llvm.extractvalue(i32, converted, (part,)),
                        bf16_pair,
                    )
                    for part in range(4)
                ]
            )
            scale = vector.broadcast(
                converted_values.type,
                scales[register_index[1] // registers_per_group],
            )
            registers[register_index] = arith.mulf(
                converted_values,
                scale,
                fastmath=arith.FastMathFlags.fast,
            )
        return mgpu.FragmentedArray(
            _registers=registers,
            _layout=values.layout,
            _is_signed=None,
        )

    return convert_and_scale(scales_ref, weights)


@cache
def _make_batched_int_matmul(
    batch_size: int,
    rows: int,
    columns: int,
    group_size: int,
    bits: int,
    is_symmetric: bool,
) -> BatchedIntMatmul:
    if (bits, is_symmetric) not in ((4, False), (8, True)):
        raise ValueError("The SM100 batched path supports asymmetric W4 and symmetric W8.")
    if batch_size < 8:
        raise ValueError(f"The SM100 batched path requires at least 8 rows, got {batch_size}.")
    if group_size < 8 or 128 % group_size:
        raise ValueError(
            f"The SM100 batched path requires group size to divide 128 and be at least 8, got {group_size}."
        )
    column_multiple = max(256, 8 * group_size)
    if columns % column_multiple:
        raise ValueError(
            f"The contraction dimension must be divisible by {column_multiple} for group size {group_size}, "
            f"got {columns}."
        )
    if rows % 128:
        raise ValueError(f"The output dimension must be divisible by 128, got {rows}.")

    block_m = min(64, 1 << (batch_size - 1).bit_length())
    block_n = 128
    block_k = 256
    num_stages = 2
    group_count = columns // group_size
    packed_zero_point_count = group_count // 2
    m_iters = pl.cdiv(batch_size, block_m)
    n_iters = pl.cdiv(rows, block_n)
    weight_values_per_byte = 8 // bits
    packed_block_k = block_k // weight_values_per_byte
    scales_per_block = block_k // group_size
    scales_per_half = scales_per_block // 2
    if bits == 4:
        main_registers = 152
        producer_registers = 176
        metadata_arrivals = 2
    else:
        main_registers = 88
        producer_registers = 208
        metadata_arrivals = 1

    activation_swizzle = plgpu.find_swizzle(block_k * jnp.dtype(jnp.bfloat16).itemsize * 8)
    activation_transforms = (
        plgpu.TilingTransform((8, activation_swizzle // jnp.dtype(jnp.bfloat16).itemsize)),
        plgpu.SwizzleTransform(activation_swizzle),
    )
    weight_swizzle = plgpu.find_swizzle(block_k * bits)
    weight_transforms = (
        plgpu.TilingTransform((8, weight_swizzle)),
        plgpu.SwizzleTransform(weight_swizzle),
    )

    def kernel(*refs: _Ref, scoped: _Ref) -> None:
        if bits == 4:
            (
                activations_gmem,
                weights_gmem,
                scales_gmem,
                zero_points_gmem,
                output_gmem,
            ) = refs
            (
                (
                    activations_smem,
                    weights_smem,
                    scales_smem,
                    zero_points_smem,
                    weights_bf16_tmem,
                    output_smem,
                    accumulator_tmem,
                ),
                barriers,
            ) = scoped
        else:
            (
                activations_gmem,
                weights_gmem,
                scales_gmem,
                output_gmem,
            ) = refs
            (
                (
                    activations_smem,
                    weights_smem,
                    scales_smem,
                    weights_bf16_tmem,
                    output_smem,
                    accumulator_tmem,
                ),
                barriers,
            ) = scoped
            zero_points_gmem = None
            zero_points_smem = None

        (
            activation_tma_barrier,
            weight_tma_barrier,
            weight_ready_barrier,
            mma_complete_barrier,
            mma_done_barrier,
            metadata_ready_barrier,
        ) = barriers
        warpgroup_index = lax.axis_index("wg")

        batch_tile_index, output_tile_index = plgpu.planar_snake(
            lax.axis_index("sm"),
            (m_iters, n_iters),
            0,
            1,
        )
        batch_start = pl.multiple_of(batch_tile_index * block_m, 8)
        actual_batch_size = lax.min(block_m, batch_size - batch_start)
        batch_slice = pl.ds(batch_start, block_m)
        output_slice = pl.ds(output_tile_index * block_n, block_n)

        @pl.when(actual_batch_size > 0)
        def run_tile() -> None:
            @pl.when(warpgroup_index == _MAIN_WARPGROUP)
            def main_warpgroup() -> None:
                plgpu.set_max_registers(main_registers, action="decrease")

                @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
                def per_warp() -> None:
                    warp_index = lax.axis_index("warp")

                    @pl.when(warp_index == _WEIGHT_TMA_WARP)
                    def load_weights() -> None:
                        def load_weight_tile(
                            k_index: jax.Array,
                            _: None,
                        ) -> None:
                            slot = lax.rem(k_index, num_stages)

                            @pl.when(k_index >= num_stages)
                            def wait_for_consumer() -> None:
                                plgpu.barrier_wait(weight_ready_barrier.at[slot])

                            plgpu.copy_gmem_to_smem(
                                weights_gmem.at[
                                    0,
                                    output_slice,
                                    pl.ds(k_index * packed_block_k, packed_block_k),
                                ],
                                weights_smem.at[slot],
                                weight_tma_barrier.at[slot],
                            )

                        lax.fori_loop(
                            0,
                            columns // block_k,
                            load_weight_tile,
                            None,
                        )

                    @pl.when(warp_index == _ACTIVATION_TMA_WARP)
                    def load_activations() -> None:
                        def load_activation_tile(
                            k_index: jax.Array,
                            _: None,
                        ) -> None:
                            slot = lax.rem(k_index, num_stages)

                            @pl.when(k_index >= num_stages)
                            def wait_for_mma() -> None:
                                plgpu.barrier_wait(mma_complete_barrier.at[slot])

                            plgpu.copy_gmem_to_smem(
                                activations_gmem.at[
                                    batch_slice,
                                    pl.ds(k_index * block_k, block_k),
                                ],
                                activations_smem.at[slot],
                                activation_tma_barrier.at[slot],
                            )

                        lax.fori_loop(
                            0,
                            columns // block_k,
                            load_activation_tile,
                            None,
                        )

                    @pl.when(warp_index == _METADATA_TMA_WARP)
                    def load_metadata() -> None:
                        plgpu.copy_gmem_to_smem(
                            scales_gmem.at[0, output_slice, :],
                            scales_smem,
                            metadata_ready_barrier,
                        )
                        if bits == 4:
                            assert zero_points_gmem is not None
                            assert zero_points_smem is not None
                            plgpu.copy_gmem_to_smem(
                                zero_points_gmem.at[
                                    pl.ds(
                                        output_tile_index * packed_zero_point_count,
                                        packed_zero_point_count,
                                    ),
                                    :,
                                ],
                                zero_points_smem,
                                metadata_ready_barrier,
                            )

                    @pl.when(warp_index == _MMA_WARP)
                    def issue_mma() -> None:
                        def mma(
                            k_index: jax.Array,
                            _: None,
                        ) -> None:
                            slot = lax.rem(k_index, num_stages)
                            plgpu.barrier_wait(weight_ready_barrier.at[slot])
                            plgpu.barrier_wait(activation_tma_barrier.at[slot])
                            plgpu.tcgen05_mma(
                                accumulator_tmem,
                                weights_bf16_tmem.at[
                                    :,
                                    pl.ds(slot * block_k, block_k),
                                ],
                                activations_smem.at[slot].T,
                                mma_complete_barrier.at[slot],
                                accumulate=k_index > 0,
                            )

                            @pl.when(k_index == columns // block_k - 1)
                            def signal_done() -> None:
                                plgpu.tcgen05_commit_arrive(mma_done_barrier)

                        lax.fori_loop(0, columns // block_k, mma, None)

            def dequantize_half(half: int) -> None:
                plgpu.set_max_registers(producer_registers, action="increase")
                plgpu.barrier_wait(metadata_ready_barrier)
                half_block_k = block_k // 2

                def dequantize_tile(
                    k_index: jax.Array,
                    _: None,
                ) -> None:
                    slot = lax.rem(k_index, num_stages)
                    plgpu.barrier_wait(weight_tma_barrier.at[slot])
                    group_offset = k_index * scales_per_block + half * scales_per_half
                    if bits == 4:
                        packed_weights = plgpu.load(
                            weights_smem.at[slot],
                            layout=_TMEM(4),
                            optimized=False,
                        )
                        converted_weights = _int4_to_bf16(
                            packed_weights,
                            half,
                        )
                        assert zero_points_smem is not None
                        dequantized_weights = _dequantize_int4(
                            scales_smem.at[
                                :,
                                pl.ds(
                                    pl.multiple_of(
                                        group_offset,
                                        scales_per_half,
                                    ),
                                    scales_per_half,
                                ),
                            ],
                            zero_points_smem,
                            converted_weights,
                            group_offset,
                            group_size,
                        )
                    else:
                        packed_weights = plgpu.load(
                            weights_smem.at[
                                slot,
                                :,
                                pl.ds(
                                    half * half_block_k,
                                    half_block_k,
                                ),
                            ],
                            layout=_TMEM(8),
                            optimized=False,
                        )
                        dequantized_weights = _dequantize_int8(
                            scales_smem.at[
                                :,
                                pl.ds(
                                    pl.multiple_of(
                                        group_offset,
                                        scales_per_half,
                                    ),
                                    scales_per_half,
                                ),
                            ],
                            packed_weights,
                            group_size,
                        )
                    dequantized_weights = plgpu.layout_cast(
                        dequantized_weights,
                        _TMEM,
                    )

                    @pl.when(k_index >= num_stages)
                    def wait_for_previous_mma() -> None:
                        plgpu.barrier_wait(mma_complete_barrier.at[slot])

                    plgpu.async_store_tmem(
                        weights_bf16_tmem.at[
                            :,
                            pl.ds(
                                slot * block_k + half * half_block_k,
                                half_block_k,
                            ),
                        ],
                        dequantized_weights,
                    )
                    plgpu.commit_tmem()
                    plgpu.barrier_arrive(weight_ready_barrier.at[slot])

                lax.fori_loop(
                    0,
                    columns // block_k,
                    dequantize_tile,
                    None,
                )

            @pl.when(warpgroup_index == _DEQUANT_LOW_WARPGROUP)
            def dequantize_low() -> None:
                dequantize_half(0)

            @pl.when(warpgroup_index == _DEQUANT_HIGH_WARPGROUP)
            def dequantize_high() -> None:
                dequantize_half(1)

            @pl.when(warpgroup_index == _STORE_WARPGROUP)
            def store_output() -> None:
                plgpu.barrier_wait(mma_done_barrier)
                accumulator = plgpu.async_load_tmem(accumulator_tmem)
                plgpu.wait_load_tmem()
                output_smem.at[0].T[...] = plgpu.layout_cast(
                    accumulator.astype(jnp.bfloat16),
                    plgpu.Layout.TCGEN05_TRANSPOSED,
                )
                plgpu.commit_smem()

                length = actual_batch_size

                def store_chunk(
                    offset: int | jax.Array,
                    size: int,
                ) -> None:
                    @pl.when(length & size != 0)
                    def store_rows() -> None:
                        plgpu.copy_smem_to_gmem(
                            output_smem.at[0, pl.ds(offset, size)],
                            output_gmem.at[
                                pl.ds(
                                    batch_start + offset,
                                    size,
                                ),
                                output_slice,
                            ],
                            commit_group=False,
                        )

                offset = 0
                size = 1 << (min(block_m, batch_size).bit_length() - 1)
                while size > 0:
                    store_chunk(offset, size)
                    offset += length & size
                    size //= 2
                plgpu.commit_smem_to_gmem_group()
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    def kernel_entry(*refs: _Ref) -> None:
        activation_smem = plgpu.SMEM(
            (num_stages, block_m, block_k),
            dtype=jnp.bfloat16,
            transforms=activation_transforms,
        )
        weight_smem = plgpu.SMEM(
            (num_stages, block_n, packed_block_k),
            dtype=jnp.uint8,
            transforms=weight_transforms,
        )
        scale_smem = plgpu.SMEM(
            (block_n, group_count),
            dtype=jnp.bfloat16,
        )
        output_smem = plgpu.SMEM(
            (1, block_m, block_n),
            dtype=jnp.bfloat16,
            transforms=(
                plgpu.TilingTransform((1, 64)),
                plgpu.SwizzleTransform(128),
            ),
        )
        weights_bf16_tmem = plgpu.TMEM(
            (block_n, num_stages * block_k),
            dtype=jnp.bfloat16,
            packed=True,
        )
        accumulator_tmem = plgpu.TMEM(
            (block_n, block_m),
            dtype=jnp.float32,
        )
        scratch_buffers = (
            activation_smem,
            weight_smem,
            scale_smem,
        )
        if bits == 4:
            scratch_buffers += (
                plgpu.SMEM(
                    (packed_zero_point_count, block_n),
                    dtype=jnp.uint8,
                ),
            )
        scratch_buffers += (
            weights_bf16_tmem,
            output_smem,
            accumulator_tmem,
        )
        barriers = (
            plgpu.Barrier(num_barriers=num_stages),
            plgpu.Barrier(num_barriers=num_stages),
            plgpu.Barrier(
                num_arrivals=2,
                num_barriers=num_stages,
                orders_tensor_core=True,
            ),
            plgpu.Barrier(
                num_barriers=num_stages,
                orders_tensor_core=True,
            ),
            plgpu.Barrier(orders_tensor_core=True),
            plgpu.Barrier(num_arrivals=metadata_arrivals),
        )
        pl.run_scoped(
            lambda *args: kernel(*refs, scoped=args),
            scratch_buffers,
            barriers,
            collective_axes="wg",
        )

    pallas_matmul = plgpu.kernel(
        kernel_entry,
        out_type=jax.ShapeDtypeStruct(
            (batch_size, rows),
            jnp.bfloat16,
        ),
        num_threads=3,
        thread_name="wg",
        grid=(m_iters * n_iters,),
        grid_names=("sm",),
        kernel_name=f"w{bits}a16_g{group_size}_sm100",
        compiler_params=plgpu.CompilerParams(
            approx_math=True,
            unsafe_no_auto_barriers=True,
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
    )

    def matmul(
        activations: Array,
        packed_weights: Array,
        scales: Array,
        packed_zero_points: Array,
    ) -> Array:
        expected_weight_shape = (
            rows,
            columns // weight_values_per_byte,
        )
        if activations.shape != (batch_size, columns):
            raise ValueError(f"Expected activations with shape {(batch_size, columns)}, got {activations.shape}.")
        if packed_weights.shape != expected_weight_shape:
            raise ValueError(
                f"Expected packed weights with shape {expected_weight_shape}, got {packed_weights.shape}."
            )
        if scales.shape != (rows, group_count):
            raise ValueError(f"Expected scales with shape {(rows, group_count)}, got {scales.shape}.")
        if activations.dtype != jnp.bfloat16 or packed_weights.dtype != jnp.uint8:
            raise TypeError("The SM100 batched path requires BF16 activations and uint8 weight storage.")
        if scales.dtype != jnp.bfloat16:
            raise TypeError(f"The SM100 batched path requires BF16 scales, got {scales.dtype}.")

        kernel_arguments = (
            activations,
            packed_weights[None],
            scales[None],
        )
        if bits == 4:
            expected_zero_point_shape = (rows, packed_zero_point_count)
            if packed_zero_points.shape != expected_zero_point_shape:
                raise ValueError(
                    "Expected packed zero points with shape "
                    f"{expected_zero_point_shape}, got {packed_zero_points.shape}."
                )
            kernel_arguments += (packed_zero_points.reshape(-1, block_n),)
        return pallas_matmul(*kernel_arguments)

    return matmul
