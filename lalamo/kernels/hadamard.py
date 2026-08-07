from collections.abc import Callable
from functools import cache, partial
from math import sqrt
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.pallas.mosaic_gpu.core import ParameterizedLayout
from jax.experimental import pallas as pl
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.pallas import mosaic_gpu as plgpu
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxtyping import Array, DTypeLike, Float

from lalamo.utils.sharding import is_sharded, sharding_of


def fragmented_hadamard(
    values: mgpu.FragmentedArray,
    block_size: Literal[32, 64, 128],
    blocks_per_warp: int,
) -> mgpu.FragmentedArray:
    registers = np.array(values.registers, copy=True)
    i32 = ir.IntegerType.get_signless(32)
    lane_index = arith.remui(mgpu.utils.thread_idx(), mgpu.c(32, i32))
    for stage in range(5):
        mask = 1 << stage
        lower_lane = arith.cmpi(
            arith.CmpIPredicate.eq,
            arith.andi(lane_index, mgpu.c(mask, i32)),
            mgpu.c(0, i32),
        )
        for register_index, value in enumerate(registers.flat):
            partner = mgpu.utils.shfl_bfly(value, mask)
            registers.flat[register_index] = arith.select(
                lower_lane,
                arith.addf(value, partner),
                arith.subf(partner, value),
            )

    registers_per_block = block_size // 32
    for block_index in range(blocks_per_warp):
        block_offset = block_index * registers_per_block
        for stage in range(5, block_size.bit_length() - 1):
            register_stride = 1 << (stage - 5)
            for pair_index in range(registers_per_block // 2):
                group_index = pair_index >> (stage - 5)
                pair_in_group = pair_index - (group_index << (stage - 5))
                left_index = block_offset + (group_index << (stage - 4)) + pair_in_group
                right_index = left_index + register_stride
                left = registers.flat[left_index]
                right = registers.flat[right_index]
                registers.flat[left_index] = arith.addf(left, right)
                registers.flat[right_index] = arith.subf(left, right)

    return mgpu.FragmentedArray(
        _registers=registers,
        _layout=values.layout,
        _is_signed=None,
    )


def _hadamard_layouts(
    blocks_per_program: int,
    *,
    permuted_output: bool,
) -> tuple[ParameterizedLayout, ParameterizedLayout]:
    if blocks_per_program == 1:
        warp_dims = (plgpu.Replicated(4),)
    elif blocks_per_program == 4:
        warp_dims = (-3,)
    else:
        warp_dims = (-3, plgpu.Replicated(4 // blocks_per_program))
    input_layout = plgpu.Layout.TILED(
        plgpu.Tiling(((blocks_per_program, 32), (1,))),
        warp_dims=warp_dims,
        lane_dims=(-2,),
        vector_dim=-1,
    )
    if not permuted_output:
        return input_layout, input_layout

    if blocks_per_program == 1:
        output_warp_dims = (plgpu.Replicated(4),)
    elif blocks_per_program == 4:
        output_warp_dims = (-5,)
    else:
        output_warp_dims = (-5, plgpu.Replicated(2))
    output_layout = plgpu.Layout.TILED(
        plgpu.Tiling(((blocks_per_program, 32), (8,), (4,), (1,))),
        warp_dims=output_warp_dims,
        lane_dims=(-4, -2, -3),
        vector_dim=-1,
    )
    return input_layout, output_layout


@cache
def _make_hadamard_transform(
    element_count: int,
    block_size: Literal[32, 64, 128],
    dtype: DTypeLike,
) -> Callable[[Array], Array]:
    block_count = element_count // block_size
    blocks_per_program = next(candidate for candidate in (4, 2, 1) if block_count % candidate == 0)
    layout, _ = _hadamard_layouts(
        blocks_per_program,
        permuted_output=False,
    )

    @plgpu.inline_mgpu(
        arg_types=(layout,),
        return_type=plgpu.ShapeDtypeStruct(
            (blocks_per_program, block_size),
            jnp.float32,
            layout=layout,
        ),
    )
    def transform(
        _ctx: object,
        values: mgpu.FragmentedArray,
    ) -> mgpu.FragmentedArray:
        return fragmented_hadamard(values, block_size, blocks_per_warp=1)

    def kernel(
        inputs_ref: jax.Ref,
        outputs_ref: jax.Ref,
    ) -> None:
        block_slice = pl.ds(
            pl.program_id(0) * blocks_per_program,
            blocks_per_program,
        )
        values = plgpu.load(
            inputs_ref.at[block_slice, :],
            layout=layout,
            optimized=False,
        ).astype(jnp.float32)
        outputs_ref.at[block_slice, :][...] = (transform(values) / sqrt(block_size)).astype(dtype)

    blocked_shape = (block_count, block_size)
    pallas_transform = plgpu.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct(blocked_shape, dtype),
        grid=(block_count // blocks_per_program,),
        grid_names=("program",),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
    )

    def apply(inputs: Array) -> Array:
        return pallas_transform(inputs.reshape(blocked_shape)).reshape(inputs.shape)

    return apply


@cache
def _make_signed_hadamard_transform(
    element_count: int,
    row_size: int,
    block_size: Literal[32, 64, 128],
    dtype: DTypeLike,
    *,
    permuted_output: bool,
) -> Callable[[Array, Array], Array]:
    block_count = element_count // block_size
    blocks_per_row = row_size // block_size
    blocks_per_program = next(candidate for candidate in (4, 2, 1) if block_count % candidate == 0)
    input_layout, output_layout = _hadamard_layouts(
        blocks_per_program,
        permuted_output=permuted_output,
    )

    @plgpu.inline_mgpu(
        arg_types=(input_layout,),
        return_type=plgpu.ShapeDtypeStruct(
            (blocks_per_program, block_size),
            jnp.float32,
            layout=output_layout,
        ),
    )
    def transform(
        _ctx: object,
        values: mgpu.FragmentedArray,
    ) -> mgpu.FragmentedArray:
        transformed = fragmented_hadamard(values, block_size, blocks_per_warp=1)
        if not permuted_output:
            return transformed
        return mgpu.FragmentedArray(
            _registers=transformed.registers.reshape(
                output_layout.to_mgpu().registers_shape((blocks_per_program, block_size))
            ),
            _layout=output_layout.to_mgpu(),
            _is_signed=None,
        )

    def kernel(
        inputs_ref: jax.Ref,
        signs_ref: jax.Ref,
        outputs_ref: jax.Ref,
    ) -> None:
        block_start = pl.program_id(0) * blocks_per_program
        block_slice = pl.ds(block_start, blocks_per_program)
        values = plgpu.load(
            inputs_ref.at[block_slice, :],
            layout=input_layout,
            optimized=False,
        ).astype(jnp.float32)
        signs = plgpu.load(
            signs_ref.at[
                pl.ds(
                    jax.lax.rem(block_start, blocks_per_row),
                    blocks_per_program,
                ),
                :,
            ],
            layout=input_layout,
            optimized=False,
        ).astype(jnp.float32)
        outputs_ref.at[block_slice, :][...] = (transform(values * signs) / sqrt(block_size)).astype(dtype)

    blocked_shape = (block_count, block_size)
    pallas_transform = plgpu.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct(blocked_shape, dtype),
        grid=(block_count // blocks_per_program,),
        grid_names=("program",),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
    )

    def apply(inputs: Array, signs: Array) -> Array:
        return pallas_transform(
            inputs.reshape(blocked_shape),
            signs.reshape(blocks_per_row, block_size),
        ).reshape(inputs.shape)

    return apply


def _local_hadamard_transform(
    inputs: Float[Array, "... channels"],
    block_size: Literal[32, 64, 128],
) -> Float[Array, "... channels"]:
    return _make_hadamard_transform(
        inputs.size,
        block_size,
        inputs.dtype,
    )(inputs)


def gpu_hadamard_transform(
    inputs: Float[Array, "... channels"],
    block_size: Literal[32, 64, 128],
) -> Float[Array, "... channels"]:
    sharding = sharding_of(inputs)
    if is_sharded(sharding):
        return jax.shard_map(
            partial(
                _local_hadamard_transform,
                block_size=block_size,
            ),
            mesh=sharding.mesh,
            in_specs=sharding.spec,
            out_specs=sharding.spec,
        )(inputs)
    return _local_hadamard_transform(inputs, block_size)


def _local_signed_hadamard_transform(
    inputs: Float[Array, "... channels"],
    signs: Array,
    block_size: Literal[32, 64, 128],
    *,
    permuted_output: bool,
) -> Float[Array, "... channels"]:
    return _make_signed_hadamard_transform(
        inputs.size,
        inputs.shape[-1],
        block_size,
        inputs.dtype,
        permuted_output=permuted_output,
    )(inputs, signs)


def gpu_signed_hadamard_transform(
    inputs: Float[Array, "... channels"],
    signs: Array,
    block_size: Literal[32, 64, 128],
    *,
    permuted_output: bool,
) -> Float[Array, "... channels"]:
    sharding = sharding_of(inputs)
    if is_sharded(sharding):
        signs_sharding = sharding_of(signs)
        return jax.shard_map(
            partial(
                _local_signed_hadamard_transform,
                block_size=block_size,
                permuted_output=permuted_output,
            ),
            mesh=sharding.mesh,
            in_specs=(sharding.spec, signs_sharding.spec),
            out_specs=sharding.spec,
        )(inputs, signs)
    return _local_signed_hadamard_transform(
        inputs,
        signs,
        block_size,
        permuted_output=permuted_output,
    )
