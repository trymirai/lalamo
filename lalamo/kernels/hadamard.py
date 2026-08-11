from collections.abc import Callable, Sequence
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

__all__ = [
    "hadamard_transform",
]


def hadamard_transform(
    inputs: Float[Array, "... channels"],
    block_size: Literal[32, 64, 128],
) -> Float[Array, "... channels"]:
    sharding = sharding_of(inputs)
    *_, channel_axis = sharding.spec
    if channel_axis is not None:
        raise ValueError("Hadamard transform inputs must not be sharded along the channels axis.")
    return _make_hadamard_dispatch(block_size)(inputs)


def _xla_hadamard_transform(
    inputs: Float[Array, "... channels"],
    block_size: Literal[32, 64, 128],
) -> Float[Array, "... channels"]:
    *original_leading_dims, input_dim = inputs.shape
    result = jnp.reshape(inputs, (*original_leading_dims, input_dim // block_size, block_size))

    for stage in range(block_size.bit_length() - 1):
        butterfly_size = 2 ** (stage + 1)
        half_butterfly_size = 2**stage
        *block_leading_dims, _ = result.shape
        grouped = jnp.reshape(
            result,
            (*block_leading_dims, block_size // butterfly_size, butterfly_size),
        )
        left = grouped[..., :half_butterfly_size]
        right = grouped[..., half_butterfly_size:]
        result = jnp.reshape(
            jnp.concatenate((left + right, left - right), axis=-1),
            (*block_leading_dims, block_size),
        )

    normalization = jnp.sqrt(jnp.asarray(block_size, dtype=inputs.dtype))
    return jnp.reshape(result / normalization, (*original_leading_dims, input_dim))


@cache
def _make_hadamard_dispatch(
    block_size: Literal[32, 64, 128],
) -> Callable[[Float[Array, "... channels"]], Float[Array, "... channels"]]:
    if block_size not in (32, 64, 128):
        raise ValueError(f"Block size {block_size} must be one of 32, 64, or 128")

    @jax.custom_vjp
    @jax.custom_batching.custom_vmap
    def transform(inputs: Float[Array, "... channels"]) -> Float[Array, "... channels"]:
        *_, input_dim = inputs.shape
        if input_dim % block_size != 0:
            raise ValueError(
                f"Input dimension {input_dim} must be a multiple of block size {block_size}",
            )

        abstract_device = sharding_of(inputs).mesh.abstract_mesh.abstract_device
        if (
            abstract_device is not None
            and abstract_device.platform == "gpu"
            and abstract_device.device_kind.startswith("NVIDIA")
        ):
            return _pallas_hadamard_transform(inputs, block_size)
        return _xla_hadamard_transform(inputs, block_size)

    @transform.def_vmap
    def transform_vmap(
        axis_size: int,
        in_batched: Sequence[bool],
        inputs: Float[Array, "... channels"],
    ) -> tuple[Float[Array, "... channels"], bool]:
        del axis_size
        (inputs_batched,) = in_batched
        return transform(inputs), inputs_batched

    def transform_fwd(
        inputs: Float[Array, "... channels"],
    ) -> tuple[Float[Array, "... channels"], None]:
        return transform(inputs), None

    def transform_bwd(
        _: None,
        cotangent: Float[Array, "... channels"],
    ) -> tuple[Float[Array, "... channels"]]:
        return (transform(cotangent),)

    transform.defvjp(transform_fwd, transform_bwd)
    return transform


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


def _hadamard_layout(blocks_per_program: int) -> ParameterizedLayout:
    if blocks_per_program == 1:
        warp_dims = (plgpu.Replicated(4),)
    elif blocks_per_program == 4:
        warp_dims = (-3,)
    else:
        warp_dims = (-3, plgpu.Replicated(4 // blocks_per_program))
    return plgpu.Layout.TILED(
        plgpu.Tiling(((blocks_per_program, 32), (1,))),
        warp_dims=warp_dims,
        lane_dims=(-2,),
        vector_dim=-1,
    )


@cache
def _make_pallas_hadamard_transform(
    element_count: int,
    block_size: Literal[32, 64, 128],
    dtype: DTypeLike,
) -> Callable[[Array], Array]:
    block_count = element_count // block_size
    blocks_per_program = next(candidate for candidate in (4, 2, 1) if block_count % candidate == 0)
    layout = _hadamard_layout(blocks_per_program)

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


def _local_pallas_hadamard_transform(
    inputs: Float[Array, "... channels"],
    block_size: Literal[32, 64, 128],
) -> Float[Array, "... channels"]:
    return _make_pallas_hadamard_transform(
        inputs.size,
        block_size,
        inputs.dtype,
    )(inputs)


def _pallas_hadamard_transform(
    inputs: Float[Array, "... channels"],
    block_size: Literal[32, 64, 128],
) -> Float[Array, "... channels"]:
    sharding = sharding_of(inputs)
    if is_sharded(sharding):
        return jax.shard_map(
            partial(
                _local_pallas_hadamard_transform,
                block_size=block_size,
            ),
            mesh=sharding.mesh,
            in_specs=sharding.spec,
            out_specs=sharding.spec,
        )(inputs)
    return _local_pallas_hadamard_transform(inputs, block_size)
