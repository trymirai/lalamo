import math
import warnings
from collections.abc import Callable
from functools import cache
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jaxtyping import Array, Float, Int, UInt32

from lalamo.kernels.mosaic import supports_mosaic_gpu
from lalamo.utils.sharding import sharding_of

from .mosaic_common import COMPILER_PARAMS, KEY_VALUE_HEAD_AXIS, make_split_reduction, tiled_smem
from .xla import xla_sparse_attention

__all__ = ["pallas_sparse_attention"]

type _Ref = Any
type _NumSplits = Literal[1, 2, 4, 8]

_TILE_SIZE = 128
_NUM_SPLITS: tuple[_NumSplits, ...] = (1, 2, 4, 8)
_SPLIT_AXIS = "pallas_sparse_split"


@cache
def make_sparse_main(
    num_programs: int,
    num_groups: int,
    query_heads_per_group: int,
    head_dim: int,
    num_slots: int,
    tiles_per_split: int,
    num_splits: _NumSplits,
) -> Callable[..., Any]:
    padded_query_heads = 8 if query_heads_per_group <= 8 else 16
    padded_output_heads = 16
    feature_blocks = head_dim // _TILE_SIZE
    accumulator_layout = plgpu.Layout.TCGEN05
    query_head_layout = accumulator_layout.reduce(0)
    token_layout = accumulator_layout.reduce(1)

    def kernel(*refs: _Ref) -> None:
        if num_splits == 1:
            (
                grouped_queries_ref,
                keys_ref,
                values_ref,
                rows_ref,
                scale_ref,
                final_outputs_ref,
                *scratch,
            ) = refs
            partial_maxima_ref = partial_normalizers_ref = partial_outputs_ref = cast("_Ref", None)
        else:
            (
                grouped_queries_ref,
                keys_ref,
                values_ref,
                rows_ref,
                scale_ref,
                partial_maxima_ref,
                partial_normalizers_ref,
                partial_outputs_ref,
                *scratch,
            ) = refs
            final_outputs_ref = cast("_Ref", None)
        (
            query_smem_ref,
            key_value_smem_union,
            values_low_smem_ref,
            probability_output_smem_union,
            query_barrier_ref,
            key_value_barrier_ref,
            tensor_core_barrier_ref,
            logits_tmem_ref,
            output_tmem_ref,
            *optional_scratch,
        ) = scratch
        if feature_blocks > 1:
            (values_high_barrier_ref,) = optional_scratch
        else:
            values_high_barrier_ref = cast("_Ref", None)
        keys_smem_ref, values_high_smem_ref = key_value_smem_union
        probabilities_smem_ref, output_smem_ref = probability_output_smem_union
        program_index = jax.lax.axis_index(KEY_VALUE_HEAD_AXIS)
        query_index = program_index // num_groups
        column_start = (program_index % num_groups) * head_dim
        if num_splits == 1:
            split_index = 0
        else:
            split_index = jax.lax.axis_index(_SPLIT_AXIS)
        plgpu.copy_gmem_to_smem(
            grouped_queries_ref.at[program_index, :, :],
            query_smem_ref,
            query_barrier_ref,
            oob_mode=plgpu.OOBFillMode.ZEROS,
        )
        initial_maximum = plgpu.layout_cast(
            jnp.full((padded_query_heads,), -jnp.inf, jnp.float32),
            query_head_layout,
        )
        initial_normalizer = plgpu.layout_cast(
            jnp.zeros((padded_query_heads,), jnp.float32),
            query_head_layout,
        )
        plgpu.barrier_wait(query_barrier_ref)

        @pl.loop(
            0,
            tiles_per_split,
            init_carry=(initial_maximum, initial_normalizer),
        )
        def consume_tile(
            local_tile_index: jax.Array,
            carry: tuple[Array, Array],
        ) -> tuple[Array, Array]:
            maximum, normalizer = carry
            tile_index = split_index * tiles_per_split + local_tile_index
            row_slice = pl.ds(tile_index * _TILE_SIZE, _TILE_SIZE)
            gather_rows = plgpu.load(
                rows_ref.at[query_index, row_slice],
                layout=plgpu.Layout.TMA_INDICES,
                optimized=False,
            )
            plgpu.copy_gmem_to_smem(
                keys_ref.at[gather_rows, pl.ds(column_start, head_dim)],
                keys_smem_ref,
                key_value_barrier_ref,
                oob_mode=plgpu.OOBFillMode.ZEROS,
            )
            plgpu.copy_gmem_to_smem(
                values_ref.at[gather_rows, pl.ds(column_start, _TILE_SIZE)],
                values_low_smem_ref,
                key_value_barrier_ref,
                oob_mode=plgpu.OOBFillMode.ZEROS,
            )
            plgpu.barrier_wait(key_value_barrier_ref)
            plgpu.tcgen05_mma(
                logits_tmem_ref,
                keys_smem_ref,
                plgpu.transpose_ref(query_smem_ref, (1, 0)),
                tensor_core_barrier_ref,
                accumulate=False,
            )
            plgpu.barrier_wait(tensor_core_barrier_ref)
            if feature_blocks > 1:
                plgpu.copy_gmem_to_smem(
                    values_ref.at[gather_rows, pl.ds(column_start + _TILE_SIZE, _TILE_SIZE)],
                    values_high_smem_ref,
                    values_high_barrier_ref,
                    oob_mode=plgpu.OOBFillMode.ZEROS,
                )
            logits = (
                plgpu.async_load_tmem(
                    logits_tmem_ref,
                    layout=accumulator_layout,
                )
                * scale_ref[...]
            )
            valid_rows = (
                plgpu.load(
                    rows_ref.at[query_index, row_slice],
                    layout=token_layout,
                    optimized=False,
                )
                < num_slots
            )
            valid = plgpu.layout_cast(
                jax.lax.broadcast_in_dim(
                    valid_rows,
                    (_TILE_SIZE, padded_query_heads),
                    (0,),
                ),
                accumulator_layout,
            )
            logits = jnp.where(valid, logits, -jnp.inf)
            tile_maximum = jnp.max(logits, axis=0)
            tile_has_values = jnp.sum(valid_rows.astype(jnp.int32)) > 0
            updated_maximum = jnp.where(
                tile_has_values,
                jnp.maximum(maximum, tile_maximum),
                maximum,
            )
            previous_scale = jnp.where(
                tile_has_values,
                jnp.exp2(
                    (maximum - updated_maximum) * math.log2(math.e),
                ),
                1,
            )
            updated_maximum_matrix = plgpu.layout_cast(
                jax.lax.broadcast_in_dim(
                    updated_maximum,
                    logits.shape,
                    (1,),
                ),
                accumulator_layout,
            )
            probabilities = jnp.where(
                valid,
                jnp.exp2(
                    (logits - updated_maximum_matrix) * math.log2(math.e),
                ),
                0,
            )
            tile_normalizer = jnp.sum(probabilities, axis=0)

            @pl.when(local_tile_index > 0)
            def rescale_output() -> None:
                output = plgpu.async_load_tmem(
                    output_tmem_ref,
                    layout=accumulator_layout,
                )
                repeated_scale = jnp.concatenate(
                    (previous_scale,) * (padded_output_heads // padded_query_heads * feature_blocks)
                )
                output_scale = plgpu.layout_cast(
                    jax.lax.broadcast_in_dim(
                        repeated_scale,
                        output.shape,
                        (1,),
                    ),
                    accumulator_layout,
                )
                scaled_output = output * output_scale
                plgpu.wait_load_tmem()
                plgpu.async_store_tmem(output_tmem_ref, scaled_output)
                plgpu.commit_tmem()

            if padded_query_heads == padded_output_heads:
                output_probabilities = probabilities
            else:
                output_probabilities = jnp.concatenate((probabilities, probabilities), axis=1)
            probabilities_smem_ref[...] = output_probabilities.astype(jnp.bfloat16)
            plgpu.commit_smem()
            accumulate = local_tile_index > 0
            plgpu.tcgen05_mma(
                output_tmem_ref.at[:, pl.ds(0, padded_output_heads)],
                plgpu.transpose_ref(values_low_smem_ref, (1, 0)),
                probabilities_smem_ref,
                accumulate=accumulate,
            )
            if feature_blocks > 1:
                plgpu.barrier_wait(values_high_barrier_ref)
                plgpu.tcgen05_mma(
                    output_tmem_ref.at[:, pl.ds(padded_output_heads, padded_output_heads)],
                    plgpu.transpose_ref(values_high_smem_ref, (1, 0)),
                    probabilities_smem_ref,
                    accumulate=accumulate,
                )
            plgpu.tcgen05_commit_arrive(tensor_core_barrier_ref)
            plgpu.barrier_wait(tensor_core_barrier_ref)
            plgpu.wait_load_tmem()
            return (
                updated_maximum,
                normalizer * previous_scale + tile_normalizer,
            )

        maximum, normalizer = consume_tile
        if padded_query_heads == padded_output_heads:
            padded_maximum = maximum
            padded_normalizer = normalizer
        else:
            padded_maximum = jnp.concatenate((maximum, maximum))
            padded_normalizer = jnp.concatenate((normalizer, normalizer))
        padded_head_slice = pl.ds(0, padded_output_heads)
        if num_splits > 1:
            partial_maxima_ref.at[program_index, split_index, padded_head_slice][...] = padded_maximum
            partial_normalizers_ref.at[program_index, split_index, padded_head_slice][...] = padded_normalizer

        for feature_block in range(feature_blocks):
            output_block = plgpu.async_load_tmem(
                output_tmem_ref.at[
                    :,
                    pl.ds(feature_block * padded_output_heads, padded_output_heads),
                ],
                layout=accumulator_layout,
            )
            if num_splits == 1:
                normalizer_matrix = plgpu.layout_cast(
                    jax.lax.broadcast_in_dim(
                        padded_normalizer,
                        output_block.shape,
                        (1,),
                    ),
                    accumulator_layout,
                )
                output_block = jnp.where(
                    normalizer_matrix > 0,
                    output_block / normalizer_matrix,
                    0,
                )
            output_smem_ref.T[...] = plgpu.layout_cast(
                output_block.astype(jnp.bfloat16),
                plgpu.Layout.TCGEN05_TRANSPOSED,
            )
            plgpu.commit_smem()
            if num_splits == 1:
                destination_ref = final_outputs_ref.at[
                    program_index,
                    padded_head_slice,
                    pl.ds(feature_block * _TILE_SIZE, _TILE_SIZE),
                ]
            else:
                destination_ref = partial_outputs_ref.at[
                    program_index,
                    split_index,
                    padded_head_slice,
                    pl.ds(feature_block * _TILE_SIZE, _TILE_SIZE),
                ]
            plgpu.copy_smem_to_gmem(output_smem_ref, destination_ref)
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    scratch_types = (
        tiled_smem((padded_query_heads, head_dim), jnp.bfloat16),
        plgpu.RefUnion(
            tiled_smem((_TILE_SIZE, head_dim), jnp.bfloat16),
            tiled_smem((_TILE_SIZE, _TILE_SIZE), jnp.bfloat16, swizzle=32),
        ),
        tiled_smem((_TILE_SIZE, _TILE_SIZE), jnp.bfloat16, swizzle=32),
        plgpu.RefUnion(
            tiled_smem((_TILE_SIZE, padded_output_heads), jnp.bfloat16),
            tiled_smem((padded_output_heads, _TILE_SIZE), jnp.bfloat16),
        ),
        plgpu.Barrier(),
        plgpu.Barrier(num_arrivals=2),
        plgpu.Barrier(orders_tensor_core=True),
        plgpu.TMEM((_TILE_SIZE, padded_query_heads), jnp.float32),
        plgpu.TMEM((_TILE_SIZE, padded_output_heads * feature_blocks), jnp.float32),
    )
    if feature_blocks > 1:
        scratch_types = (*scratch_types, plgpu.Barrier())
    kernel_suffix = f"d{head_dim}_q{query_heads_per_group}"
    if num_splits == 1:
        return plgpu.kernel(
            kernel,
            out_type=jax.ShapeDtypeStruct((num_programs, padded_output_heads, head_dim), jnp.bfloat16),
            grid=(num_programs,),
            grid_names=(KEY_VALUE_HEAD_AXIS,),
            scratch_types=scratch_types,
            compiler_params=COMPILER_PARAMS,
            kernel_name=f"tcgen_sparse_{kernel_suffix}",
        )

    partial_prefix = (num_programs, num_splits, padded_output_heads)
    return plgpu.kernel(
        kernel,
        out_type=(
            jax.ShapeDtypeStruct(partial_prefix, jnp.float32),
            jax.ShapeDtypeStruct(partial_prefix, jnp.float32),
            jax.ShapeDtypeStruct((*partial_prefix, head_dim), jnp.bfloat16),
        ),
        grid=(num_programs, num_splits),
        grid_names=(KEY_VALUE_HEAD_AXIS, _SPLIT_AXIS),
        scratch_types=scratch_types,
        compiler_params=COMPILER_PARAMS,
        kernel_name=f"tcgen_sparse_main_{kernel_suffix}_s{num_splits}",
    )


def folded_pallas_sparse_attention(
    queries: Float[Array, "dst_tokens heads head_dim"],
    keys: Float[Array, "src_tokens key_value_channels"],
    values: Float[Array, "src_tokens key_value_channels"],
    rows: UInt32[Array, "dst_tokens rows"],
    scale: Float[Array, ""],
    num_groups: int,
    num_splits: _NumSplits,
) -> Float[Array, "dst_tokens heads head_dim"]:
    num_queries, num_heads, head_dim = queries.shape
    num_slots, _ = keys.shape
    _, num_rows = rows.shape
    query_heads_per_group = num_heads // num_groups
    padded_query_heads = 8 if query_heads_per_group <= 8 else 16
    num_programs = num_queries * num_groups
    tiles_per_split = math.ceil(num_rows / (_TILE_SIZE * num_splits))
    rows = jnp.pad(
        rows,
        ((0, 0), (0, tiles_per_split * num_splits * _TILE_SIZE - num_rows)),
        constant_values=num_slots,
    )
    grouped_queries = queries.reshape(num_programs, query_heads_per_group, head_dim)
    grouped_queries = jnp.pad(
        grouped_queries,
        ((0, 0), (0, padded_query_heads - query_heads_per_group), (0, 0)),
    )
    main = make_sparse_main(
        num_programs,
        num_groups,
        query_heads_per_group,
        head_dim,
        num_slots,
        tiles_per_split,
        num_splits,
    )
    if num_splits == 1:
        output = main(grouped_queries, keys, values, rows, scale)
    else:
        partials = main(grouped_queries, keys, values, rows, scale)
        output = make_split_reduction(num_programs, padded_query_heads, head_dim, num_splits)(*partials)
    return output[:, :query_heads_per_group].reshape(queries.shape)


pallas_sparse_attention = jax.custom_batching.custom_vmap(xla_sparse_attention)


@pallas_sparse_attention.def_vmap
def pallas_sparse_attention_vmap(
    axis_size: int,
    in_batched: list[bool | None],
    queries: Float[Array, "batch dst_tokens heads head_dim"],
    keys: Float[Array, "batch src_tokens key_value_heads head_dim"],
    values: Float[Array, "batch src_tokens key_value_heads head_dim"],
    rows: Int[Array, "batch dst_tokens rows"],
    scale: float | Float[Array, ""] | None,
    logit_soft_cap: float | Float[Array, ""] | None,
) -> tuple[Float[Array, "batch dst_tokens heads head_dim"], bool]:
    (
        queries_batched,
        keys_batched,
        values_batched,
        rows_batched,
        scale_batched,
        _logit_soft_cap_batched,
    ) = in_batched
    fallback = jax.vmap(
        xla_sparse_attention,
        in_axes=tuple(0 if batched else None for batched in in_batched),
    )
    supports_pallas = (
        all((queries_batched, keys_batched, values_batched, rows_batched))
        and not scale_batched
        and logit_soft_cap is None
    )
    if supports_pallas:
        _, _, num_heads, head_dim = queries.shape
        _, _, num_groups, _ = keys.shape
        supports_pallas = (
            supports_mosaic_gpu(sharding_of(keys).mesh, 10)
            and num_heads % num_groups == 0
            and num_heads // num_groups <= 16
            and head_dim in (128, 256)
            and queries.dtype == keys.dtype == values.dtype == jnp.bfloat16
        )
    if not supports_pallas:
        warnings.warn(
            "Pallas sparse attention does not support this attention configuration; falling back to XLA attention.",
            RuntimeWarning,
            stacklevel=2,
        )
        return fallback(queries, keys, values, rows, scale, logit_soft_cap), True

    batch_size, num_queries, num_heads, head_dim = queries.shape
    _, num_slots, num_groups, _ = keys.shape
    _, _, num_rows = rows.shape
    num_programs = axis_size * num_queries * num_groups
    required_splits = math.ceil(144 / num_programs)
    num_tiles = math.ceil(num_rows / _TILE_SIZE)
    available_splits = [candidate for candidate in _NUM_SPLITS if candidate <= num_tiles]
    num_splits = next(
        (candidate for candidate in available_splits if candidate >= required_splits),
        available_splits[-1],
    )
    batch_offsets = (jnp.arange(batch_size, dtype=jnp.int32) * num_slots)[:, None, None]
    folded_rows = jnp.where(rows < num_slots, rows + batch_offsets, batch_size * num_slots)
    attention_scale = jnp.asarray(head_dim**-0.5 if scale is None else scale, dtype=jnp.float32)
    output = folded_pallas_sparse_attention(
        queries.reshape(batch_size * num_queries, num_heads, head_dim),
        keys.reshape(batch_size * num_slots, num_groups * head_dim),
        values.reshape(batch_size * num_slots, num_groups * head_dim),
        folded_rows.reshape(batch_size * num_queries, num_rows).astype(jnp.uint32),
        attention_scale,
        num_groups,
        num_splits,
    )
    return output.reshape(queries.shape), True
