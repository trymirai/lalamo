import math
from collections.abc import Callable
from functools import cache
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jaxtyping import Array

__all__ = [
    "COMPILER_PARAMS",
    "KEY_VALUE_HEAD_AXIS",
    "QUERY_HEAD_AXIS",
    "make_split_reduction",
    "tiled_smem",
]

type _Ref = Any

KEY_VALUE_HEAD_AXIS = "pallas_decode_key_value_head"
QUERY_HEAD_AXIS = "pallas_decode_query_head"
COMPILER_PARAMS = plgpu.CompilerParams(
    approx_math=True,
    lowering_semantics=plgpu.LoweringSemantics.Lane,
    reduction_scratch_bytes=8_192,
)


def tiled_smem(
    shape: tuple[int, int],
    dtype: jax.typing.DTypeLike,
    *,
    swizzle: int | None = None,
) -> pl.MemoryRef:
    element_bits = jnp.dtype(dtype).itemsize * 8
    if swizzle is None:
        swizzle = plgpu.find_swizzle(shape[-1] * element_bits)
    return plgpu.SMEM(
        shape,
        dtype,
        transforms=(
            plgpu.TilingTransform((8, 8 * swizzle // element_bits)),
            plgpu.SwizzleTransform(swizzle),
        ),
    )


@cache
def make_split_reduction(
    num_key_value_heads: int,
    padded_query_heads: Literal[8, 16],
    head_dim: Literal[128, 256, 512],
    num_splits: Literal[2, 4, 8],
) -> Callable[[Array, Array, Array], Array]:
    output_layout = plgpu.Layout.WG_STRIDED(
        (head_dim,),
        vec_size=head_dim // 128,
    )

    def kernel(
        main_maxima_ref: _Ref,
        main_normalizers_ref: _Ref,
        main_outputs_ref: _Ref,
        outputs_ref: _Ref,
    ) -> None:
        key_value_head_index = jax.lax.axis_index(KEY_VALUE_HEAD_AXIS)
        query_head_index = jax.lax.axis_index(QUERY_HEAD_AXIS)
        initial_index = (key_value_head_index, 0, query_head_index)
        maximum = plgpu.load(
            main_maxima_ref.at[initial_index],
            layout=plgpu.Layout.WG_SPLAT,
            optimized=False,
        )
        normalizer = plgpu.load(
            main_normalizers_ref.at[initial_index],
            layout=plgpu.Layout.WG_SPLAT,
            optimized=False,
        )
        output = plgpu.load(
            main_outputs_ref.at[*initial_index, :],
            layout=output_layout,
            optimized=False,
        ).astype(jnp.float32)

        for split_index in range(1, num_splits):
            main_index = (key_value_head_index, split_index, query_head_index)
            block_maximum = plgpu.load(
                main_maxima_ref.at[main_index],
                layout=plgpu.Layout.WG_SPLAT,
                optimized=False,
            )
            block_normalizer = plgpu.load(
                main_normalizers_ref.at[main_index],
                layout=plgpu.Layout.WG_SPLAT,
                optimized=False,
            )
            block_output = plgpu.load(
                main_outputs_ref.at[*main_index, :],
                layout=output_layout,
                optimized=False,
            ).astype(jnp.float32)
            has_values = block_normalizer > 0
            updated_maximum = jnp.where(
                has_values,
                jnp.maximum(maximum, block_maximum),
                maximum,
            )
            carry_scale = jnp.where(
                has_values,
                jnp.exp2((maximum - updated_maximum) * math.log2(math.e)),
                1,
            )
            block_scale = jnp.where(
                has_values,
                jnp.exp2((block_maximum - updated_maximum) * math.log2(math.e)),
                0,
            )
            maximum = updated_maximum
            normalizer = normalizer * carry_scale + block_normalizer * block_scale
            output = output * carry_scale + block_output * block_scale

        outputs_ref.at[key_value_head_index, query_head_index, :][...] = jnp.where(
            normalizer > 0,
            output / normalizer,
            0,
        ).astype(jnp.bfloat16)

    return plgpu.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct(
            (num_key_value_heads, padded_query_heads, head_dim),
            jnp.bfloat16,
        ),
        grid=(num_key_value_heads, padded_query_heads),
        grid_names=(KEY_VALUE_HEAD_AXIS, QUERY_HEAD_AXIS),
        compiler_params=COMPILER_PARAMS,
        kernel_name=f"tcgen_decode_fused_reduce_d{head_dim}_q{padded_query_heads}_s{num_splits}",
    )
