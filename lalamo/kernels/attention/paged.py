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
from jaxtyping import Array, Float, Int

__all__ = ["paged_decode_attention"]


# JAX's paged attention wrapper exposes neither window starts nor softmax residuals.
# Keep the paged FlashAttention reduction here so learned sinks enter its denominator exactly once.
def _paged_attention_kernel(
    queries: jax.Ref,
    key_pages: jax.Ref,
    value_pages: jax.Ref,
    block_table: jax.Ref,
    length: jax.Ref,
    outputs: jax.Ref,
    normalizers: jax.Ref,
    maxima: jax.Ref,
    *,
    scale: float,
    logit_soft_cap: float | None,
    sliding_window_size: int | None,
    pages_per_partition: int,
) -> None:
    block_heads, head_dim = queries.shape
    padded_dim = max(16, pl.next_power_of_2(head_dim))
    features = jnp.arange(padded_dim)
    heads = jnp.arange(block_heads)
    _, page_size, _ = key_pages.shape
    end_token = length[0]
    start_token = jnp.zeros((), dtype=jnp.int32)
    if sliding_window_size is not None:
        start_token = jnp.maximum(0, end_token - sliding_window_size)
    first_page = start_token // page_size + pl.program_id(2) * pages_per_partition
    last_page = jnp.minimum(first_page + pages_per_partition, pl.cdiv(end_token, page_size))
    query = plgpu.load(
        queries.at[heads[:, None], features[None, :]],
        mask=features[None, :] < head_dim,
        other=0.0,
    )

    def attend_page(page: Array, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        output, normalizer, maximum = carry
        physical_page = block_table[page]
        indices = (physical_page, jnp.arange(page_size)[:, None], features[None, :])
        keys = plgpu.load(key_pages.at[indices], mask=features[None, :] < head_dim, other=0.0)
        values = plgpu.load(value_pages.at[indices], mask=features[None, :] < head_dim, other=0.0)
        scores = plgpu.dot(query, keys.T, precision=jax.lax.Precision.HIGHEST) * scale
        if logit_soft_cap is not None:
            scores = jnp.tanh(scores / logit_soft_cap) * logit_soft_cap
        positions = page * page_size + jnp.arange(page_size)
        valid = (positions >= start_token) & (positions < end_token)
        scores = jnp.where(valid[None, :], scores, jnp.finfo(jnp.float32).min)
        next_maximum = jnp.maximum(maximum, jnp.max(scores, axis=-1))
        correction = jnp.exp(maximum - next_maximum)
        weights = jnp.where(valid[None, :], jnp.exp(scores - next_maximum[:, None]), 0.0)
        return (
            output * correction[:, None]
            + plgpu.dot(weights.astype(values.dtype), values, precision=jax.lax.Precision.HIGHEST),
            normalizer * correction + jnp.sum(weights, axis=-1),
            next_maximum,
        )

    output, normalizer, maximum = jax.lax.fori_loop(
        first_page,
        jnp.maximum(first_page, last_page),
        attend_page,
        (
            jnp.zeros((block_heads, padded_dim), dtype=jnp.float32),
            jnp.zeros(block_heads, dtype=jnp.float32),
            jnp.full(block_heads, jnp.finfo(jnp.float32).min, dtype=jnp.float32),
        ),
    )
    plgpu.store(outputs.at[heads[:, None], features[None, :]], output, mask=features[None, :] < head_dim)
    normalizers[:] = normalizer
    maxima[:] = maximum


def paged_decode_attention(
    queries: Float[Array, "batch heads head_channels"],
    key_pages: Float[Array, "groups total_pages page_size head_channels"],
    value_pages: Float[Array, "groups total_pages page_size head_channels"],
    block_tables: Int[Array, "batch pages_per_sequence"],
    lengths: Int[Array, " batch"],
    *,
    scale: float,
    logit_soft_cap: float | None,
    sinks: Float[Array, " heads"] | None = None,
    sliding_window_size: int | None = None,
) -> Float[Array, "batch heads head_channels"]:
    # Pallas blocks use local refs; preserve the caller's partition without collecting other shards.
    def attend_batch(
        queries: Array,
        key_pages: Array,
        value_pages: Array,
        block_tables: Array,
        lengths: Array,
        sinks: Array | None,
    ) -> Array:
        _, num_heads, head_dim = queries.shape
        num_groups, total_pages, page_size, _ = key_pages.shape
        heads_per_group = num_heads // num_groups
        block_heads = 16
        padded_heads = pl.cdiv(heads_per_group, block_heads) * block_heads
        pages_per_sequence = block_tables.shape[1]
        active_pages = pages_per_sequence
        if sliding_window_size is not None:
            active_pages = min(active_pages, (sliding_window_size + 2 * page_size - 2) // page_size)
        partitions = min(active_pages, 16)
        pages_per_partition = pl.cdiv(active_pages, partitions)

        def attend(query: Array, block_table: Array, length: Array) -> Array:
            grouped_queries = jnp.pad(
                query.reshape(num_groups, heads_per_group, head_dim),
                ((0, 0), (0, padded_heads - heads_per_group), (0, 0)),
            )
            output_shape = (partitions, num_groups, padded_heads, head_dim)
            residual_shape = output_shape[:-1]
            outputs, normalizers, maxima = pl.pallas_call(
                partial(
                    _paged_attention_kernel,
                    scale=scale,
                    logit_soft_cap=logit_soft_cap,
                    sliding_window_size=sliding_window_size,
                    pages_per_partition=pages_per_partition,
                ),
                grid=(num_groups, padded_heads // block_heads, partitions),
                in_specs=(
                    pl.BlockSpec((None, block_heads, head_dim), lambda group, head, _part: (group, head, 0)),
                    pl.BlockSpec(
                        (None, total_pages, page_size, head_dim), lambda group, _head, _part: (group, 0, 0, 0)
                    ),
                    pl.BlockSpec(
                        (None, total_pages, page_size, head_dim), lambda group, _head, _part: (group, 0, 0, 0)
                    ),
                    pl.BlockSpec((pages_per_sequence,), lambda _group, _head, _part: (0,)),
                    pl.BlockSpec((1,), lambda _group, _head, _part: (0,)),
                ),
                out_specs=(
                    pl.BlockSpec(
                        (None, None, block_heads, head_dim), lambda group, head, part: (part, group, head, 0)
                    ),
                    pl.BlockSpec((None, None, block_heads), lambda group, head, part: (part, group, head)),
                    pl.BlockSpec((None, None, block_heads), lambda group, head, part: (part, group, head)),
                ),
                out_shape=(
                    jax.ShapeDtypeStruct(output_shape, jnp.float32),
                    jax.ShapeDtypeStruct(residual_shape, jnp.float32),
                    jax.ShapeDtypeStruct(residual_shape, jnp.float32),
                ),
                compiler_params=plgpu.CompilerParams(num_warps=8, num_stages=1),
                name="paged_decode_attention",
            )(
                grouped_queries,
                key_pages,
                value_pages,
                block_table,
                length[None],
            )
            outputs = outputs[:, :, :heads_per_group].reshape(partitions, num_heads, head_dim)
            normalizers = normalizers[:, :, :heads_per_group].reshape(partitions, num_heads)
            maxima = maxima[:, :, :heads_per_group].reshape(partitions, num_heads)
            maximum = jnp.max(maxima, axis=0)
            if sinks is not None:
                maximum = jnp.maximum(maximum, sinks.astype(jnp.float32))
            corrections = jnp.exp(maxima - maximum[None, :])
            denominator = jnp.sum(normalizers * corrections, axis=0)
            if sinks is not None:
                denominator += jnp.exp(sinks.astype(jnp.float32) - maximum)
            numerator = jnp.sum(outputs * corrections[:, :, None], axis=0)
            return (numerator / jnp.maximum(denominator[:, None], jnp.finfo(jnp.float32).tiny)).astype(query.dtype)

        return jax.vmap(attend)(queries, block_tables, lengths)

    sharding = jax.typeof(queries).sharding
    if sharding.mesh.explicit_axes:
        attend_batch = jax.shard_map(attend_batch, mesh=sharding.mesh, out_specs=sharding.spec, check_vma=False)
    return attend_batch(queries, key_pages, value_pages, block_tables, lengths, sinks)
