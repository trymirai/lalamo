import math
import warnings
from collections.abc import Callable
from functools import cache
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jaxtyping import Array, Bool, Float, Int

from lalamo.kernels.mosaic import supports_mosaic_gpu
from lalamo.utils.sharding import sharding_of

from .xla import xla_attention

type _Ref = Any
type _NumSplits = Literal[1, 2, 4, 8]

_BLOCK_SIZE = 128
_NUM_SPLITS: tuple[_NumSplits, ...] = (1, 2, 4, 8)
_KEY_VALUE_HEAD_AXIS = "pallas_decode_key_value_head"
_QUERY_HEAD_AXIS = "pallas_decode_query_head"
_SPLIT_AXIS = "pallas_decode_split"
_COMPILER_PARAMS = plgpu.CompilerParams(
    approx_math=True,
    lowering_semantics=plgpu.LoweringSemantics.Lane,
    reduction_scratch_bytes=8_192,
)


def _tiled_smem(
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
def _make_main(
    sequence_length: int,
    num_key_value_heads: int,
    query_heads_per_key_value_head: int,
    head_dim: int,
    num_splits: _NumSplits,
) -> Callable[..., Any]:
    padded_query_heads = 8 if query_heads_per_key_value_head <= 8 else 16
    padded_output_heads = 16
    feature_blocks = head_dim // _BLOCK_SIZE
    num_blocks = math.ceil(sequence_length / _BLOCK_SIZE)
    blocks_per_split = math.ceil(num_blocks / num_splits)
    accumulator_layout = plgpu.Layout.TCGEN05
    query_head_layout = accumulator_layout.reduce(0)
    token_layout = accumulator_layout.reduce(1)

    def kernel(*refs: _Ref) -> None:
        if num_splits == 1:
            (
                grouped_queries_ref,
                keys_ref,
                values_ref,
                mask_ref,
                start_ref,
                end_ref,
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
                mask_ref,
                start_ref,
                end_ref,
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
        key_value_head_index = jax.lax.axis_index(_KEY_VALUE_HEAD_AXIS)
        # A dynamic slice lets TMA zero-fill the padded query heads.
        dynamic_zero = key_value_head_index - key_value_head_index
        if num_splits == 1:
            split_index = 0
        else:
            split_index = jax.lax.axis_index(_SPLIT_AXIS)
        plgpu.copy_gmem_to_smem(
            grouped_queries_ref.at[
                key_value_head_index,
                pl.ds(dynamic_zero, padded_query_heads),
                :,
            ],
            query_smem_ref,
            query_barrier_ref,
            oob_mode=plgpu.OOBFillMode.ZEROS,
        )
        start = start_ref[...]
        end = end_ref[...]
        first_block = start // _BLOCK_SIZE
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
            blocks_per_split,
            init_carry=(initial_maximum, initial_normalizer),
        )
        def consume_block(
            local_block_index: jax.Array,
            carry: tuple[Array, Array],
        ) -> tuple[Array, Array]:
            maximum, normalizer = carry
            block_index = first_block + split_index * blocks_per_split + local_block_index
            block_start = block_index * _BLOCK_SIZE
            token_slice = pl.ds(block_start, _BLOCK_SIZE)
            plgpu.copy_gmem_to_smem(
                keys_ref.at[token_slice, key_value_head_index, :],
                keys_smem_ref,
                key_value_barrier_ref,
                oob_mode=plgpu.OOBFillMode.ZEROS,
            )
            plgpu.copy_gmem_to_smem(
                values_ref.at[
                    token_slice,
                    key_value_head_index,
                    pl.ds(0, _BLOCK_SIZE),
                ],
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
                    values_ref.at[
                        token_slice,
                        key_value_head_index,
                        pl.ds(_BLOCK_SIZE, _BLOCK_SIZE),
                    ],
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
            token_offsets = plgpu.layout_cast(
                jnp.arange(_BLOCK_SIZE, dtype=jnp.int32),
                token_layout,
            )
            absolute_token_indices = block_start + token_offsets
            attention_mask = (
                plgpu.load(
                    mask_ref.at[token_slice],
                    layout=token_layout,
                    optimized=False,
                )
                != 0
            )
            valid_rows = (
                attention_mask
                & (absolute_token_indices >= start)
                & (absolute_token_indices < end)
                & (absolute_token_indices < sequence_length)
            )
            valid = plgpu.layout_cast(
                jax.lax.broadcast_in_dim(
                    valid_rows,
                    (_BLOCK_SIZE, padded_query_heads),
                    (0,),
                ),
                accumulator_layout,
            )
            logits = jnp.where(valid, logits, -jnp.inf)
            block_maximum = jnp.max(logits, axis=0)
            block_has_values = jnp.sum(valid_rows.astype(jnp.int32)) > 0
            updated_maximum = jnp.where(
                block_has_values,
                jnp.maximum(maximum, block_maximum),
                maximum,
            )
            previous_scale = jnp.where(
                block_has_values,
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
            block_normalizer = jnp.sum(probabilities, axis=0)

            @pl.when(local_block_index > 0)
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
            accumulate = local_block_index > 0
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

            for feature_block in range(2, feature_blocks, 2):
                plgpu.copy_gmem_to_smem(
                    values_ref.at[
                        token_slice,
                        key_value_head_index,
                        pl.ds(feature_block * _BLOCK_SIZE, _BLOCK_SIZE),
                    ],
                    values_low_smem_ref,
                    key_value_barrier_ref,
                    oob_mode=plgpu.OOBFillMode.ZEROS,
                )
                plgpu.copy_gmem_to_smem(
                    values_ref.at[
                        token_slice,
                        key_value_head_index,
                        pl.ds((feature_block + 1) * _BLOCK_SIZE, _BLOCK_SIZE),
                    ],
                    values_high_smem_ref,
                    key_value_barrier_ref,
                    oob_mode=plgpu.OOBFillMode.ZEROS,
                )
                plgpu.barrier_wait(key_value_barrier_ref)
                plgpu.tcgen05_mma(
                    output_tmem_ref.at[
                        :,
                        pl.ds(feature_block * padded_output_heads, padded_output_heads),
                    ],
                    plgpu.transpose_ref(values_low_smem_ref, (1, 0)),
                    probabilities_smem_ref,
                    accumulate=accumulate,
                )
                plgpu.tcgen05_mma(
                    output_tmem_ref.at[
                        :,
                        pl.ds((feature_block + 1) * padded_output_heads, padded_output_heads),
                    ],
                    plgpu.transpose_ref(values_high_smem_ref, (1, 0)),
                    probabilities_smem_ref,
                    accumulate=accumulate,
                )
                plgpu.tcgen05_commit_arrive(tensor_core_barrier_ref)
                plgpu.barrier_wait(tensor_core_barrier_ref)

            plgpu.wait_load_tmem()
            return (
                updated_maximum,
                normalizer * previous_scale + block_normalizer,
            )

        maximum, normalizer = consume_block
        if padded_query_heads == padded_output_heads:
            padded_maximum = maximum
            padded_normalizer = normalizer
        else:
            padded_maximum = jnp.concatenate((maximum, maximum))
            padded_normalizer = jnp.concatenate((normalizer, normalizer))
        if num_splits > 1:
            padded_head_slice = pl.ds(0, padded_output_heads)
            partial_maxima_ref.at[key_value_head_index, split_index, padded_head_slice][...] = padded_maximum
            partial_normalizers_ref.at[key_value_head_index, split_index, padded_head_slice][...] = padded_normalizer
        else:
            padded_head_slice = pl.ds(dynamic_zero, padded_output_heads)

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
                    key_value_head_index,
                    padded_head_slice,
                    pl.ds(feature_block * _BLOCK_SIZE, _BLOCK_SIZE),
                ]
            else:
                destination_ref = partial_outputs_ref.at[
                    key_value_head_index,
                    split_index,
                    padded_head_slice,
                    pl.ds(feature_block * _BLOCK_SIZE, _BLOCK_SIZE),
                ]
            plgpu.copy_smem_to_gmem(output_smem_ref, destination_ref)
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    scratch_types = (
        _tiled_smem((padded_query_heads, head_dim), jnp.bfloat16),
        plgpu.RefUnion(
            _tiled_smem((_BLOCK_SIZE, head_dim), jnp.bfloat16),
            _tiled_smem((_BLOCK_SIZE, _BLOCK_SIZE), jnp.bfloat16, swizzle=32),
        ),
        _tiled_smem((_BLOCK_SIZE, _BLOCK_SIZE), jnp.bfloat16, swizzle=32),
        plgpu.RefUnion(
            _tiled_smem((_BLOCK_SIZE, padded_output_heads), jnp.bfloat16),
            _tiled_smem((padded_output_heads, _BLOCK_SIZE), jnp.bfloat16),
        ),
        plgpu.Barrier(),
        plgpu.Barrier(num_arrivals=2),
        plgpu.Barrier(orders_tensor_core=True),
        plgpu.TMEM((_BLOCK_SIZE, padded_query_heads), jnp.float32),
        plgpu.TMEM((_BLOCK_SIZE, padded_output_heads * feature_blocks), jnp.float32),
    )
    if feature_blocks > 1:
        scratch_types = (*scratch_types, plgpu.Barrier())
    kernel_suffix = f"d{head_dim}_q{query_heads_per_key_value_head}"
    if num_splits == 1:
        return plgpu.kernel(
            kernel,
            out_type=jax.ShapeDtypeStruct(
                (
                    num_key_value_heads,
                    query_heads_per_key_value_head,
                    head_dim,
                ),
                jnp.bfloat16,
            ),
            grid=(num_key_value_heads,),
            grid_names=(_KEY_VALUE_HEAD_AXIS,),
            scratch_types=scratch_types,
            compiler_params=_COMPILER_PARAMS,
            kernel_name=f"tcgen_decode_fused_{kernel_suffix}",
        )

    partial_prefix = (
        num_key_value_heads,
        num_splits,
        padded_output_heads,
    )
    return plgpu.kernel(
        kernel,
        out_type=(
            jax.ShapeDtypeStruct(partial_prefix, jnp.float32),
            jax.ShapeDtypeStruct(partial_prefix, jnp.float32),
            jax.ShapeDtypeStruct((*partial_prefix, head_dim), jnp.bfloat16),
        ),
        grid=(num_key_value_heads, num_splits),
        grid_names=(_KEY_VALUE_HEAD_AXIS, _SPLIT_AXIS),
        scratch_types=scratch_types,
        compiler_params=_COMPILER_PARAMS,
        kernel_name=f"tcgen_decode_fused_main_{kernel_suffix}_s{num_splits}",
    )


@cache
def _make_reduction(
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
        key_value_head_index = jax.lax.axis_index(_KEY_VALUE_HEAD_AXIS)
        query_head_index = jax.lax.axis_index(_QUERY_HEAD_AXIS)
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
        grid_names=(_KEY_VALUE_HEAD_AXIS, _QUERY_HEAD_AXIS),
        compiler_params=_COMPILER_PARAMS,
        kernel_name=f"tcgen_decode_fused_reduce_d{head_dim}_q{padded_query_heads}_s{num_splits}",
    )


def _pallas_decode_attention(
    queries: Float[Array, "1 query_heads head_dim"],
    keys: Float[Array, "capacity key_value_heads head_dim"],
    values: Float[Array, "capacity key_value_heads head_dim"],
    mask: Int[Array, " padded_capacity"],
    start: Int[Array, ""],
    end: Int[Array, ""],
    scale: Float[Array, ""],
    num_splits: _NumSplits,
) -> Float[Array, "1 query_heads head_dim"]:
    num_key_value_heads = keys.shape[1]
    head_dim = keys.shape[2]
    query_heads_per_key_value_head = queries.shape[1] // num_key_value_heads
    padded_query_heads = 8 if query_heads_per_key_value_head <= 8 else 16
    grouped_queries = queries[0].reshape(
        num_key_value_heads,
        query_heads_per_key_value_head,
        head_dim,
    )
    main = _make_main(
        keys.shape[0],
        num_key_value_heads,
        query_heads_per_key_value_head,
        head_dim,
        num_splits,
    )
    if num_splits == 1:
        return main(grouped_queries, keys, values, mask, start, end, scale).reshape(queries.shape)

    partials = main(grouped_queries, keys, values, mask, start, end, scale)
    reduced = _make_reduction(
        num_key_value_heads,
        padded_query_heads,
        head_dim,
        num_splits,
    )(*partials)
    return reduced[:, :query_heads_per_key_value_head].reshape(queries.shape)


pallas_decode_attention = jax.custom_batching.custom_vmap(xla_attention)


@pallas_decode_attention.def_vmap
def _pallas_decode_attention_vmap(
    axis_size: int,
    in_batched: list[bool | None],
    queries: Float[Array, "batch dst_tokens heads head_dim"],
    keys: Float[Array, "batch src_tokens key_value_heads head_dim"],
    values: Float[Array, "batch src_tokens key_value_heads head_dim"],
    bias: Float[Array, "batch heads dst_tokens src_tokens"] | Float[Array, "heads dst_tokens src_tokens"] | None,
    masks: Bool[Array, "batch dst_tokens src_tokens"] | Bool[Array, "dst_tokens src_tokens"],
    scale: float | Float[Array, ""] | None,
    logit_soft_cap: float | None,
) -> tuple[Float[Array, "batch dst_tokens heads head_dim"], bool]:
    (
        queries_batched,
        keys_batched,
        values_batched,
        _bias_batched,
        masks_batched,
        scale_batched,
        _logit_soft_cap_batched,
    ) = in_batched
    fallback = jax.vmap(
        xla_attention,
        in_axes=tuple(0 if batched else None for batched in in_batched),
    )
    arrays_are_batched = all((queries_batched, keys_batched, values_batched, masks_batched))
    supports_pallas = arrays_are_batched and bias is None and logit_soft_cap is None
    if supports_pallas:
        query_heads = queries.shape[2]
        key_value_heads = keys.shape[2]
        head_dim = queries.shape[3]
        supports_pallas = (
            not scale_batched
            and supports_mosaic_gpu(sharding_of(keys).mesh, 10)
            and queries.shape[1] == 1
            and query_heads % key_value_heads == 0
            and query_heads // key_value_heads <= 16
            and head_dim in (128, 256, 512)
            and queries.dtype == keys.dtype == values.dtype == jnp.bfloat16
        )
    if not supports_pallas:
        warnings.warn(
            "Pallas decode attention does not support this attention configuration; falling back to XLA attention.",
            RuntimeWarning,
            stacklevel=2,
        )
        return fallback(queries, keys, values, bias, masks, scale, logit_soft_cap), True

    programs = axis_size * keys.shape[2]
    required_splits = (144 + programs - 1) // programs
    num_blocks = math.ceil(keys.shape[1] / _BLOCK_SIZE)
    num_splits = next(
        (candidate for candidate in _NUM_SPLITS if required_splits <= candidate <= num_blocks),
        None,
    )
    if num_splits is None:
        warnings.warn(
            "Pallas decode attention cannot provide enough parallel work for this batch and cache size; "
            "falling back to XLA attention.",
            RuntimeWarning,
            stacklevel=2,
        )
        return fallback(queries, keys, values, bias, masks, scale, logit_soft_cap), True

    attention_scale = queries.shape[-1] ** -0.5 if scale is None else scale
    attention_scale = jnp.asarray(attention_scale, dtype=jnp.float32)

    unbatched_masks = masks[:, 0]
    has_values = jnp.any(unbatched_masks, axis=-1)
    starts = jnp.where(
        has_values,
        jnp.argmax(unbatched_masks, axis=-1),
        0,
    ).astype(jnp.int32)
    ends = jnp.where(
        has_values,
        masks.shape[-1] - jnp.argmax(unbatched_masks[:, ::-1], axis=-1),
        0,
    ).astype(jnp.int32)
    padded_masks = jnp.pad(
        unbatched_masks,
        ((0, 0), (0, (-masks.shape[-1]) % _BLOCK_SIZE)),
        constant_values=False,
    ).astype(jnp.int32)
    return (
        jax.vmap(
            lambda query, key, value, mask, start, end: _pallas_decode_attention(
                query,
                key,
                value,
                mask,
                start,
                end,
                attention_scale,
                num_splits,
            )
        )(queries, keys, values, padded_masks, starts, ends),
        True,
    )
