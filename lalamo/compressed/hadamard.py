from functools import partial
from math import prod, sqrt
from typing import Literal

import equinox as eqx
import jax
import jax.experimental.pallas.triton as pltriton
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jaxtyping import Array, Float, Int, Key

__all__ = [
    "RHTFactors",
    "hadamard_transform",
    "random_incoherence_factors",
]


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def hadamard_transform(
    inputs: Float[Array, " channels"],
    block_size: Literal[32, 64, 128],
) -> Float[Array, " channels"]:
    if block_size not in (32, 64, 128):
        raise ValueError(f"Block size {block_size} must be one of 32, 64, or 128")
    return _hadamard_transform_impl(inputs, block_size)


def _hadamard_transform_impl(
    inputs: Float[Array, "... channels"],
    block_size: Literal[32, 64, 128],
) -> Float[Array, "... channels"]:
    *_, input_dim = inputs.shape
    if input_dim % block_size != 0:
        raise ValueError(
            f"Input dimension {input_dim} must be a multiple of block size {block_size}",
        )

    if jax.default_backend() == "gpu":
        return _pallas_hadamard_transform(inputs, block_size)
    return _jax_hadamard_transform(inputs, block_size)


def _jax_hadamard_transform(
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


def _warp_shuffle_xor(
    values: Float[Array, " elements"],
    mask: int,
) -> Float[Array, " elements"]:
    [result] = pltriton.elementwise_inline_asm(
        f"shfl.sync.bfly.b32 $0, $1, {mask}, 31, 0xffffffff;",
        args=[values],
        constraints="=f,f",
        pack=1,
        result_shape_dtypes=[
            jax.ShapeDtypeStruct(values.shape, values.dtype),
        ],
    )
    return result


def _pallas_hadamard_transform_kernel(
    inputs_ref: pl.MemoryRef,
    outputs_ref: pl.MemoryRef,
    *,
    block_size: Literal[32, 64, 128],
    blocks_per_program: Literal[1, 4, 8],
    scale: float,
) -> None:
    block_offsets = jax.lax.broadcasted_iota(jnp.int32, (blocks_per_program, 32), 0) * block_size
    element_offsets = jax.lax.broadcasted_iota(jnp.int32, (blocks_per_program, 32), 1)
    first_quarter_offsets = block_offsets + element_offsets
    first_quarter = inputs_ref[first_quarter_offsets].astype(jnp.float32)
    if block_size >= 64:
        second_quarter_offsets = first_quarter_offsets + 32
        second_quarter = inputs_ref[second_quarter_offsets].astype(jnp.float32)
    if block_size >= 128:
        third_quarter_offsets = first_quarter_offsets + 64
        fourth_quarter_offsets = first_quarter_offsets + 96
        third_quarter = inputs_ref[third_quarter_offsets].astype(jnp.float32)
        fourth_quarter = inputs_ref[fourth_quarter_offsets].astype(jnp.float32)

    for mask in (1, 2, 4, 8, 16):
        first_partners = _warp_shuffle_xor(first_quarter, mask)
        first_quarter = jnp.where(
            (element_offsets & mask) == 0,
            first_quarter + first_partners,
            first_partners - first_quarter,
        )
        if block_size >= 64:
            second_partners = _warp_shuffle_xor(second_quarter, mask)
            second_quarter = jnp.where(
                (element_offsets & mask) == 0,
                second_quarter + second_partners,
                second_partners - second_quarter,
            )
        if block_size >= 128:
            third_partners = _warp_shuffle_xor(third_quarter, mask)
            fourth_partners = _warp_shuffle_xor(fourth_quarter, mask)
            third_quarter = jnp.where(
                (element_offsets & mask) == 0,
                third_quarter + third_partners,
                third_partners - third_quarter,
            )
            fourth_quarter = jnp.where(
                (element_offsets & mask) == 0,
                fourth_quarter + fourth_partners,
                fourth_partners - fourth_quarter,
            )

    if block_size >= 64:
        first_half = first_quarter + second_quarter
        second_half = first_quarter - second_quarter
    if block_size >= 128:
        third_half = third_quarter + fourth_quarter
        fourth_half = third_quarter - fourth_quarter
        first_quarter = first_half + third_half
        second_quarter = second_half + fourth_half
        third_quarter = first_half - third_half
        fourth_quarter = second_half - fourth_half
    elif block_size >= 64:
        first_quarter = first_half
        second_quarter = second_half

    normalization = jnp.asarray(scale, dtype=outputs_ref.dtype).astype(jnp.float32)
    outputs_ref[first_quarter_offsets] = (first_quarter * normalization).astype(outputs_ref.dtype)
    if block_size >= 64:
        outputs_ref[second_quarter_offsets] = (second_quarter * normalization).astype(outputs_ref.dtype)
    if block_size >= 128:
        outputs_ref[third_quarter_offsets] = (third_quarter * normalization).astype(outputs_ref.dtype)
        outputs_ref[fourth_quarter_offsets] = (fourth_quarter * normalization).astype(outputs_ref.dtype)


def _pallas_hadamard_transform(
    inputs: Float[Array, "... channels"],
    block_size: Literal[32, 64, 128],
) -> Float[Array, "... channels"]:
    flat_size = prod(inputs.shape)
    if flat_size % (8 * block_size) == 0:
        blocks_per_program = 8
    elif flat_size % (4 * block_size) == 0:
        blocks_per_program = 4
    else:
        blocks_per_program = 1
    program_size = blocks_per_program * block_size
    flat_inputs = jnp.reshape(inputs, (flat_size,))
    block_spec = pl.BlockSpec(
        block_shape=(program_size,),
        index_map=lambda program_id: (program_id,),
    )
    result = pl.pallas_call(
        partial(
            _pallas_hadamard_transform_kernel,
            block_size=block_size,
            blocks_per_program=blocks_per_program,
            scale=1 / sqrt(block_size),
        ),
        out_shape=jax.ShapeDtypeStruct(flat_inputs.shape, inputs.dtype),
        grid=(flat_size // program_size,),
        in_specs=[block_spec],
        out_specs=block_spec,
        compiler_params=pltriton.CompilerParams(num_warps=blocks_per_program),
        name="hadamard_transform",
    )(flat_inputs)
    return jnp.reshape(result, inputs.shape)


def _hadamard_transform_fwd(
    inputs: Float[Array, " channels"],
    block_size: Literal[32, 64, 128],
) -> tuple[Float[Array, " channels"], None]:
    return hadamard_transform(inputs, block_size), None


def _hadamard_transform_bwd(
    block_size: Literal[32, 64, 128],
    _: None,
    cotangent: Float[Array, " channels"],
) -> tuple[Float[Array, " channels"]]:
    return (hadamard_transform(cotangent, block_size),)


hadamard_transform.defvjp(_hadamard_transform_fwd, _hadamard_transform_bwd)


def random_incoherence_factors(
    channels: int,
    key: Key[Array, ""],
) -> Int[Array, " channels"]:
    factors = jnp.where(jax.random.bernoulli(key, shape=(channels,)), 1, -1)
    return factors.astype(jnp.int32)


class RHTFactors(eqx.Module):
    input_factors: Int[Array, " in_channels"]
    output_factors: Int[Array, " out_channels"]

    @classmethod
    def random_init(
        cls,
        input_dim: int,
        output_dim: int,
        key: Key[Array, ""],
    ) -> "RHTFactors":
        input_key, output_key = jax.random.split(key, 2)
        return cls(
            input_factors=random_incoherence_factors(channels=input_dim, key=input_key),
            output_factors=random_incoherence_factors(
                channels=output_dim,
                key=output_key,
            ),
        )
