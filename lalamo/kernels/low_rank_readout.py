from collections.abc import Callable
from functools import cache
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jaxtyping import Array

type _Ref = Any


@cache
def make_block_topk(
    block_count: int,
    block_size: int,
    count: int,
) -> Callable[[Array], Array]:
    layout = plgpu.Layout.WG_STRIDED(
        (block_size,),
        vec_size=min(8, block_size // 128),
    )
    position_bits = block_size.bit_length() - 1
    position_mask = block_size - 1

    def kernel(scores_ref: _Ref, output_ref: _Ref) -> None:
        block_index = jax.lax.axis_index("block")
        scores = plgpu.load(
            scores_ref.at[block_index, :],
            layout=layout,
        )
        positions = plgpu.layout_cast(jnp.arange(block_size, dtype=jnp.int32), layout)
        bits = jax.lax.bitcast_convert_type(scores, jnp.uint16).astype(jnp.uint32)
        sign = bits & 0x8000 != 0
        score_keys = bits ^ jnp.where(sign, 0xFFFF, 0x8000)
        remaining_keys = ((score_keys + 1) << position_bits) | (position_mask - positions).astype(jnp.uint32)

        @pl.loop(0, count, init_carry=remaining_keys)  # pyrefly: ignore[bad-argument-type]
        def select(rank: jax.Array, remaining: jax.Array) -> jax.Array:
            maximum = jnp.max(remaining)
            winner = (position_mask - (maximum & position_mask)).astype(jnp.int32)
            output_ref.at[block_index, rank][...] = winner
            return jnp.where(positions == winner, jnp.asarray(0, jnp.uint32), remaining)

    return plgpu.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct((block_count, count), jnp.int32),
        grid=(block_count,),
        grid_names=("block",),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Lane,
            reduction_scratch_bytes=8_192,
        ),
        kernel_name="block_topk",
    )


@cache
def make_gathered_dot(
    candidate_count: int,
    hidden_size: int,
) -> Callable[[Array, Array, Array], Array]:
    outputs_per_program = 32
    values_per_thread = hidden_size // 128
    layout = plgpu.Layout.WG_STRIDED(
        (hidden_size,),
        vec_size=min(8, values_per_thread & -values_per_thread),
    )
    padded_candidate_count = (candidate_count + outputs_per_program - 1) // outputs_per_program * outputs_per_program

    def kernel(weights_ref: _Ref, token_ids_ref: _Ref, vectors_ref: _Ref, output_ref: _Ref) -> None:
        candidate_block = jax.lax.axis_index("candidate")
        vector = plgpu.load(vectors_ref.at[:], layout=layout).astype(jnp.float32)

        @pl.loop(0, outputs_per_program)
        def compute(offset: jax.Array) -> None:
            candidate_index = candidate_block * outputs_per_program + offset
            token_id = plgpu.load(
                token_ids_ref.at[candidate_index],
                layout=plgpu.Layout.WG_SPLAT,
            )
            weights = plgpu.load(weights_ref.at[token_id, :], layout=layout).astype(jnp.float32)
            output_ref.at[candidate_index][...] = jnp.sum(weights * vector).astype(jnp.bfloat16)

    gathered_dot = plgpu.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct((padded_candidate_count,), jnp.bfloat16),
        grid=(padded_candidate_count // outputs_per_program,),
        grid_names=("candidate",),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Lane,
            reduction_scratch_bytes=8_192,
        ),
        kernel_name="gathered_dot",
    )

    def call(weights: Array, token_ids: Array, vectors: Array) -> Array:
        padding = padded_candidate_count - candidate_count
        if padding:
            token_ids = jnp.pad(token_ids, (0, padding))
        return gathered_dot(weights, token_ids, vectors)[:candidate_count]

    return call
