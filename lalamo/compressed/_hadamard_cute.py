from math import sqrt
from typing import Literal

import cutlass
import cutlass.jax as cutlass_jax
import jax
from cuda.bindings import driver as cuda
from cutlass import cute
from cutlass.cute.arch import WARP_SIZE
from jaxtyping import Array, Float


@cute.kernel
def _hadamard_transform_kernel(
    inputs: cute.Tensor,
    outputs: cute.Tensor,
    block_size: cutlass.Constexpr,
) -> None:
    lane_index, block_index, _ = cute.arch.thread_idx()
    _, num_blocks_per_cta, _ = cute.arch.block_dim()
    cta_index, _, _ = cute.arch.block_idx()
    num_subblocks_per_block = block_size // WARP_SIZE
    block_offset = (cta_index * num_blocks_per_cta + block_index) * block_size + lane_index
    subblock_accum = [
        cutlass.Float32(inputs[block_offset + WARP_SIZE * subblock_index])
        for subblock_index in range(num_subblocks_per_block)
    ]

    for stage in cutlass.range_constexpr(WARP_SIZE.bit_length() - 1):
        mask = 1 << stage
        lane_sign = cutlass.Float32(1 - 2 * ((lane_index & mask) >> stage))
        for subblock_index in cutlass.range_constexpr(num_subblocks_per_block):
            partner = cute.arch.shuffle_sync_bfly(subblock_accum[subblock_index], mask)
            subblock_accum[subblock_index] = lane_sign * subblock_accum[subblock_index] + partner

    for stage in cutlass.range_constexpr(num_subblocks_per_block.bit_length() - 1):
        subblock_stride = 1 << stage
        for pair_index in cutlass.range_constexpr(num_subblocks_per_block // 2):
            group_index = pair_index >> stage
            pair_in_group = pair_index - (group_index << stage)
            left_subblock = (group_index << (stage + 1)) + pair_in_group
            right_subblock = left_subblock + subblock_stride
            left = subblock_accum[left_subblock]
            right = subblock_accum[right_subblock]
            subblock_accum[left_subblock] = left + right
            subblock_accum[right_subblock] = left - right

    normalization = cutlass.Float32(1 / sqrt(WARP_SIZE * num_subblocks_per_block))
    for subblock_index in cutlass.range_constexpr(num_subblocks_per_block):
        result = subblock_accum[subblock_index] * normalization
        outputs[block_offset + WARP_SIZE * subblock_index] = result.to(outputs.element_type)


@cute.jit
def _launch_hadamard_transform(
    stream: cuda.CUstream,
    inputs: cute.Tensor,
    outputs: cute.Tensor,
    block_size: cutlass.Constexpr,
    num_blocks_per_cta: cutlass.Constexpr,
) -> None:
    num_ctas = cute.size(inputs) // (num_blocks_per_cta * block_size)
    input_values = cute.make_tensor(inputs.iterator, (cute.size(inputs),))
    output_values = cute.make_tensor(outputs.iterator, (cute.size(outputs),))

    kernel = _hadamard_transform_kernel(input_values, output_values, block_size)
    kernel.launch(
        grid=[num_ctas, 1, 1],
        block=[WARP_SIZE, num_blocks_per_cta, 1],
        stream=stream,
    )


def cute_hadamard_transform(
    inputs: Float[Array, "... channels"],
    block_size: Literal[32, 64, 128],
) -> Float[Array, "... channels"]:
    num_blocks = inputs.size // block_size
    num_blocks_per_cta = 512 // block_size
    while num_blocks % num_blocks_per_cta != 0:
        num_blocks_per_cta //= 2
    return cutlass_jax.cutlass_call(
        _launch_hadamard_transform,
        output_shape_dtype=jax.ShapeDtypeStruct(inputs.shape, inputs.dtype),
        use_static_tensors=True,
        block_size=block_size,
        num_blocks_per_cta=num_blocks_per_cta,
    )(inputs)
