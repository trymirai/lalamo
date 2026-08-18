import jax
import jax.numpy as jnp
from einops import rearrange
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import Array, DTypeLike, Int

from lalamo.modules import Decoder
from lalamo.modules.token_mixer import State, StateLayerBase
from lalamo.modules.token_mixers.attention import Attention
from lalamo.modules.token_mixers.kv_cache import (
    KVCacheLayer,
    PagedKVCacheLayer,
    StaticKVCacheLayer,
)


def init_paged_state(
    decoder: Decoder,
    *,
    slot_count: int,
    total_pages: int,
    page_size: int,
    dtype: DTypeLike,
) -> State:
    page_sharding = decoder.sharding_config.make_sharding((None, None, None, None))
    table_sharding = decoder.sharding_config.make_sharding((None, None))
    length_sharding = decoder.sharding_config.make_sharding((None,))
    state_layers: list[StateLayerBase] = []
    for layer_index in decoder.transformer.kv_source_layer_indices:
        layer = decoder.transformer.layers[layer_index]
        if isinstance(layer.mixer, Attention):
            page_shape = (
                layer.mixer.config.num_groups,
                total_pages,
                page_size,
                layer.mixer.config.head_dim,
            )
            state_layers.append(
                PagedKVCacheLayer(
                    keys=jax.device_put(jnp.zeros(page_shape, dtype=dtype), page_sharding),
                    values=jax.device_put(jnp.zeros(page_shape, dtype=dtype), page_sharding),
                    block_tables=jax.device_put(jnp.empty((0, 0), dtype=jnp.int32), table_sharding),
                    lengths=jax.device_put(jnp.empty((0,), dtype=jnp.int32), length_sharding),
                )
            )
        else:
            state_layers.append(layer.init_static_state(slot_count, 1, dtype))
    return State(state_layers)


def insert_prefill(
    paged_state: State,
    prefill_state: State,
    slot_ids: Int[Array, " batch"],
    block_tables: Int[Array, "batch pages_per_sequence"],
) -> State:
    batch_size, pages_per_sequence = block_tables.shape
    if slot_ids.shape != (batch_size,):
        raise ValueError(f"slot_ids must have shape {(batch_size,)}, got {slot_ids.shape}.")

    def insert_layer(pool: StateLayerBase, prefill: StateLayerBase) -> StateLayerBase:
        if isinstance(pool, PagedKVCacheLayer):
            assert isinstance(prefill, StaticKVCacheLayer)
            page_capacity = pages_per_sequence * pool.page_size
            if prefill.keys.shape[1] < page_capacity:
                raise ValueError(
                    f"Prefill capacity {prefill.keys.shape[1]} is smaller than page capacity {page_capacity}."
                )
            keys = rearrange(
                prefill.keys[:, :page_capacity],
                "batch (pages page_size) groups channels -> groups batch pages page_size channels",
                pages=pages_per_sequence,
                page_size=pool.page_size,
            )
            values = rearrange(
                prefill.values[:, :page_capacity],
                "batch (pages page_size) groups channels -> groups batch pages page_size channels",
                pages=pages_per_sequence,
                page_size=pool.page_size,
            )
            return PagedKVCacheLayer(
                keys=pool.keys.at[:, block_tables, :, :].set(keys, out_sharding=_named_sharding(pool.keys)),
                values=pool.values.at[:, block_tables, :, :].set(values, out_sharding=_named_sharding(pool.values)),
                block_tables=pool.block_tables,
                lengths=pool.lengths,
            )
        assert not isinstance(prefill, KVCacheLayer)
        return jax.tree.map(
            lambda pool_array, prefill_array: pool_array.at[slot_ids].set(
                prefill_array,
                out_sharding=_named_sharding(pool_array),
            ),
            pool,
            prefill,
        )

    return State(insert_layer(pool, prefill) for pool, prefill in zip(paged_state, prefill_state, strict=True))


def select_batch(
    paged_state: State,
    slot_ids: Int[Array, " batch"],
    block_tables: Int[Array, "batch pages_per_sequence"],
    lengths: Int[Array, " batch"],
) -> State:
    batch_size, _ = block_tables.shape
    if slot_ids.shape != (batch_size,):
        raise ValueError(f"slot_ids must have shape {(batch_size,)}, got {slot_ids.shape}.")

    def select_layer(layer: StateLayerBase) -> StateLayerBase:
        if isinstance(layer, PagedKVCacheLayer):
            return PagedKVCacheLayer(
                keys=layer.keys,
                values=layer.values,
                block_tables=block_tables,
                lengths=lengths,
            )
        return jax.tree.map(
            lambda array: array.at[slot_ids].get(out_sharding=_batch_sharding(slot_ids, array.ndim)),
            layer,
        )

    return State(select_layer(layer) for layer in paged_state)


def update_slots(paged_state: State, decoded_state: State, slot_ids: Int[Array, " batch"]) -> State:
    def update_layer(pool: StateLayerBase, decoded: StateLayerBase) -> StateLayerBase:
        if isinstance(pool, PagedKVCacheLayer):
            assert isinstance(decoded, PagedKVCacheLayer)
            return PagedKVCacheLayer(
                keys=decoded.keys,
                values=decoded.values,
                block_tables=pool.block_tables,
                lengths=pool.lengths,
            )
        assert not isinstance(decoded, PagedKVCacheLayer)
        return jax.tree.map(
            lambda pool_array, decoded_array: pool_array.at[slot_ids].set(
                decoded_array,
                out_sharding=_named_sharding(pool_array),
            ),
            pool,
            decoded,
        )

    return State(update_layer(pool, decoded) for pool, decoded in zip(paged_state, decoded_state, strict=True))


def _named_sharding(array: Array) -> NamedSharding | None:
    sharding = jax.typeof(array).sharding
    return sharding if isinstance(sharding, NamedSharding) else None


def _batch_sharding(indices: Int[Array, " batch"], ndim: int) -> NamedSharding | None:
    sharding = jax.typeof(indices).sharding
    if not isinstance(sharding, NamedSharding):
        return None
    return NamedSharding(sharding.mesh, PartitionSpec(*sharding.spec, *((None,) * (ndim - 1))))
