# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jaxtyping import Array, Bool, Float


def _flash_attention_kernel(
    queries: jax.Ref,
    keys: jax.Ref,
    values: jax.Ref,
    bias: jax.Ref | None,
    mask: jax.Ref,
    starts: jax.Ref,
    ends: jax.Ref,
    scale: jax.Ref,
    soft_cap: jax.Ref | None,
    outputs: jax.Ref,
    normalizers: jax.Ref,
    maxima: jax.Ref,
    *,
    block_keys: int,
    splits: int,
    heads_per_group: int,
    query_count: int,
) -> None:
    query = queries[:, :]
    first_block = starts[0] // block_keys
    last_block = pl.cdiv(ends[0], block_keys)
    blocks_per_split = pl.cdiv(last_block - first_block, splits)
    first_block += pl.program_id(2) * blocks_per_split
    last_block = jnp.minimum(first_block + blocks_per_split, last_block)
    query_rows, head_dim = query.shape
    rows = pl.program_id(0) * query_rows + jnp.arange(query_rows)
    query_positions = rows // heads_per_group
    query_heads = rows % heads_per_group

    def attend_block(block: Array, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        output, normalizer, maximum = carry
        key_slice = pl.ds(block * block_keys, block_keys)
        key = keys[key_slice, :]
        value = values[key_slice, :]
        scores = plgpu.dot(query, key.T, precision=jax.lax.Precision.HIGHEST) * scale[()]
        if soft_cap is not None:
            scores = jnp.tanh(scores / soft_cap[()]) * soft_cap[()]
        key_positions = block * block_keys + jnp.arange(block_keys)
        if bias is not None:
            scores += plgpu.load(
                bias.at[query_heads[:, None], query_positions[:, None], key_positions[None, :]],
                mask=query_positions[:, None] < query_count,
                other=0.0,
            ).astype(jnp.float32)
        valid = plgpu.load(
            mask.at[query_positions[:, None], key_positions[None, :]],
            mask=query_positions[:, None] < query_count,
            other=False,
        )
        scores = jnp.where(valid, scores, jnp.finfo(jnp.float32).min)
        next_maximum = jnp.maximum(maximum, jnp.max(scores, axis=-1))
        correction = jnp.exp(maximum - next_maximum)
        weights = jnp.where(valid, jnp.exp(scores - next_maximum[:, None]), 0.0)
        return (
            output * correction[:, None]
            + plgpu.dot(weights.astype(value.dtype), value, precision=jax.lax.Precision.HIGHEST),
            normalizer * correction + jnp.sum(weights, axis=-1),
            next_maximum,
        )

    output, normalizer, maximum = jax.lax.fori_loop(
        first_block,
        last_block,
        attend_block,
        (
            jnp.zeros((query_rows, head_dim), dtype=jnp.float32),
            jnp.zeros(query_rows, dtype=jnp.float32),
            jnp.full(query_rows, jnp.finfo(jnp.float32).min, dtype=jnp.float32),
        ),
    )
    outputs[:, :] = output
    normalizers[:] = normalizer
    maxima[:] = maximum


def triton_attention(
    queries: Float[Array, "dst_tokens heads head_dim"],
    keys: Float[Array, "src_tokens groups head_dim"],
    values: Float[Array, "src_tokens groups head_dim"],
    bias: Float[Array, "heads dst_tokens src_tokens"] | None,
    mask: Bool[Array, "dst_tokens src_tokens"],
    scale: float | Float[Array, ""] | None,
    logit_soft_cap: float | Float[Array, ""] | None,
    *,
    batch_size: int,
) -> Float[Array, "dst_tokens heads head_dim"]:
    query_count, heads, head_dim = queries.shape
    key_count, groups, _ = keys.shape
    heads_per_group = heads // groups
    rows_per_group = query_count * heads_per_group
    padded_dim = max(16, pl.next_power_of_2(head_dim))
    block_queries = min(128, max(16, pl.next_power_of_2(rows_per_group)))
    block_keys, num_stages = 128, 2
    if queries.dtype == jnp.float32 or padded_dim * queries.dtype.itemsize >= 512:
        block_queries = min(block_queries, 64)
        num_stages = 1
    if padded_dim * queries.dtype.itemsize > 512:
        block_queries = min(block_queries, 32)
        block_keys = 64
    padded_rows = pl.cdiv(rows_per_group, block_queries) * block_queries
    padded_keys = pl.cdiv(key_count, block_keys) * block_keys
    query_blocks = padded_rows // block_queries
    programs = batch_size * groups * query_blocks
    splits = min(pl.cdiv(144, programs), padded_keys // block_keys, 16)

    # Pallas blocks carry no explicit mesh metadata; normalize it once at this boundary.
    @jax.sharding.auto_axes(
        axes=jax.typeof(queries).sharding.mesh.explicit_axes, out_sharding=jax.typeof(queries).sharding
    )
    def attend(
        queries: Array,
        keys: Array,
        values: Array,
        bias: Array | None,
        mask: Array,
        scale: Array,
        soft_cap: Array | None,
    ) -> Array:
        queries = queries.reshape(query_count, groups, heads_per_group, head_dim)
        queries = queries.transpose(1, 0, 2, 3).reshape(groups, rows_per_group, head_dim)
        queries = jnp.pad(queries, ((0, 0), (0, padded_rows - rows_per_group), (0, padded_dim - head_dim)))
        keys = jnp.pad(keys, ((0, padded_keys - key_count), (0, 0), (0, padded_dim - head_dim)))
        values = jnp.pad(values, ((0, padded_keys - key_count), (0, 0), (0, padded_dim - head_dim)))
        mask = jnp.pad(mask, ((0, 0), (0, padded_keys - key_count)))
        if bias is not None:
            bias = bias.reshape(groups, heads_per_group, query_count, key_count)
            bias = jnp.pad(bias, ((0, 0), (0, 0), (0, 0), (0, padded_keys - key_count)))
        has_values = jnp.any(mask, axis=-1)
        row_starts = jnp.where(has_values, jnp.argmax(mask, axis=-1), padded_keys)
        row_ends = jnp.where(has_values, padded_keys - jnp.argmax(mask[:, ::-1], axis=-1), 0)
        query_positions = jnp.arange(padded_rows).reshape(query_blocks, block_queries) // heads_per_group
        visible_rows = jnp.minimum(query_positions, query_count - 1)
        starts = jnp.min(jnp.where(query_positions < query_count, row_starts[visible_rows], padded_keys), axis=-1)
        ends = jnp.max(jnp.where(query_positions < query_count, row_ends[visible_rows], 0), axis=-1)
        starts = jnp.where(ends > 0, starts, 0).astype(jnp.int32)
        ends = ends.astype(jnp.int32)
        output_shape = (splits, groups, padded_rows, padded_dim)
        residual_shape = output_shape[:-1]
        outputs, normalizers, maxima = pl.pallas_call(
            partial(
                _flash_attention_kernel,
                block_keys=block_keys,
                splits=splits,
                heads_per_group=heads_per_group,
                query_count=query_count,
            ),
            grid=(query_blocks, groups, splits),
            in_specs=(
                pl.BlockSpec((None, block_queries, padded_dim), lambda tile, group, _part: (group, tile, 0)),
                pl.BlockSpec((padded_keys, None, padded_dim), lambda _tile, group, _part: (0, group, 0)),
                pl.BlockSpec((padded_keys, None, padded_dim), lambda _tile, group, _part: (0, group, 0)),
                None
                if bias is None
                else pl.BlockSpec(
                    (None, heads_per_group, query_count, padded_keys), lambda _tile, group, _part: (group, 0, 0, 0)
                ),
                pl.BlockSpec((query_count, padded_keys), lambda _tile, _group, _part: (0, 0)),
                pl.BlockSpec((1,), lambda tile, _head, _part: (tile,)),
                pl.BlockSpec((1,), lambda tile, _head, _part: (tile,)),
                pl.BlockSpec((), lambda _tile, _head, _part: ()),
                None if soft_cap is None else pl.BlockSpec((), lambda _tile, _head, _part: ()),
            ),
            out_specs=(
                pl.BlockSpec(
                    (None, None, block_queries, padded_dim), lambda tile, group, part: (part, group, tile, 0)
                ),
                pl.BlockSpec((None, None, block_queries), lambda tile, group, part: (part, group, tile)),
                pl.BlockSpec((None, None, block_queries), lambda tile, group, part: (part, group, tile)),
            ),
            out_shape=(
                jax.ShapeDtypeStruct(output_shape, jnp.float32),
                jax.ShapeDtypeStruct(residual_shape, jnp.float32),
                jax.ShapeDtypeStruct(residual_shape, jnp.float32),
            ),
            compiler_params=plgpu.CompilerParams(num_warps=8, num_stages=num_stages),
            name="flash_attention",
        )(queries, keys, values, bias, mask, starts, ends, scale, soft_cap)
        maximum = jnp.max(maxima, axis=0)
        corrections = jnp.exp(maxima - maximum[None])
        denominator = jnp.sum(normalizers * corrections, axis=0)
        numerator = jnp.sum(outputs * corrections[..., None], axis=0)
        output = numerator / jnp.maximum(denominator[..., None], jnp.finfo(jnp.float32).tiny)
        output = output[:, :rows_per_group, :head_dim].reshape(groups, query_count, heads_per_group, head_dim)
        return output.transpose(1, 0, 2, 3).reshape(query_count, heads, head_dim).astype(queries.dtype)

    attention_scale = head_dim**-0.5 if scale is None else scale
    soft_cap = None
    if logit_soft_cap is not None:
        soft_cap = jnp.asarray(logit_soft_cap, dtype=jnp.float32)
    return attend(queries, keys, values, bias, mask, jnp.asarray(attention_scale, dtype=jnp.float32), soft_cap)
