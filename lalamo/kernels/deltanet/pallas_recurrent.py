import warnings
from collections.abc import Callable
from functools import cache

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.sharding import NamedSharding
from jaxtyping import Array

from lalamo.kernels.mosaic import supports_mosaic_gpu

from .xla import xla_recurrent_scan

__all__ = [
    "deltanet_recurrent_scan",
]

type DeltaUpdate = Callable[
    [Array, Array, Array, Array, Array, Array, Array],
    tuple[Array, Array],
]


deltanet_recurrent_scan = jax.custom_batching.custom_vmap(xla_recurrent_scan)


@cache
def _make_batched_update(
    batch_size: int,
    num_heads: int,
    value_head_dim: int,
) -> DeltaUpdate:
    values_per_program = 64
    head_dim = 128
    state_layout = plgpu.Layout.TILED(
        plgpu.Tiling(((64, 8), (16, 8), (8, 8), (2,), (1,))),
        warp_dims=(-8,),
        lane_dims=(-4, -3),
        vector_dim=-1,
    )
    value_layout = state_layout.reduce(1)
    key_layout = state_layout.reduce(0)

    mesh = plgpu.Mesh(
        grid=(batch_size, num_heads, value_head_dim // values_per_program),
        grid_names=("batch", "head", "value"),
        kernel_name=f"deltanet_b{batch_size}",
    )

    def update(
        queries: Array,
        keys: Array,
        values: Array,
        decay: Array,
        beta: Array,
        state: Array,
        active: Array,
    ) -> tuple[Array, Array]:
        queries_ref = jax.new_ref(queries, memory_space=plgpu.GMEM)
        keys_ref = jax.new_ref(keys, memory_space=plgpu.GMEM)
        values_ref = jax.new_ref(values, memory_space=plgpu.GMEM)
        decay_ref = jax.new_ref(decay, memory_space=plgpu.GMEM)
        beta_ref = jax.new_ref(beta, memory_space=plgpu.GMEM)
        state_ref = jax.new_ref(state, memory_space=plgpu.GMEM)
        active_ref = jax.new_ref(active, memory_space=plgpu.GMEM)
        outputs_ref = jax.new_ref(
            jnp.empty((batch_size, num_heads, value_head_dim), dtype=jnp.float32),
            memory_space=plgpu.GMEM,
        )

        @pl.core_map(
            mesh,
            compiler_params=plgpu.CompilerParams(
                lowering_semantics=plgpu.LoweringSemantics.Lane,
                reduction_scratch_bytes=8_192,
            ),
            name=f"deltanet_b{batch_size}",
        )
        def kernel() -> None:
            batch_index = jax.lax.axis_index("batch")
            head_index = jax.lax.axis_index("head")
            value_start = jax.lax.axis_index("value") * values_per_program
            value_slice = pl.ds(value_start, values_per_program)
            query = plgpu.load(
                queries_ref.at[batch_index, head_index, :],
                layout=key_layout,
                optimized=False,
            ).astype(jnp.float32)
            key = plgpu.load(
                keys_ref.at[batch_index, head_index, :],
                layout=key_layout,
                optimized=False,
            ).astype(jnp.float32)
            values = plgpu.load(
                values_ref.at[batch_index, head_index, value_slice],
                layout=value_layout,
                optimized=False,
            ).astype(jnp.float32)
            state = plgpu.load(
                state_ref.at[batch_index, head_index, value_slice, :],
                layout=state_layout,
                optimized=False,
            ).astype(jnp.float32)
            key_matrix = plgpu.layout_cast(
                jax.lax.broadcast_in_dim(
                    key,
                    (values_per_program, head_dim),
                    (1,),
                ),
                state_layout,
            )
            query_matrix = plgpu.layout_cast(
                jax.lax.broadcast_in_dim(
                    query,
                    (values_per_program, head_dim),
                    (1,),
                ),
                state_layout,
            )
            state_times_key = jnp.sum(state * key_matrix, axis=-1)
            state_times_query = jnp.sum(state * query_matrix, axis=-1)
            key_times_query = jnp.sum(key * query)
            decay_value = decay_ref[batch_index, head_index]
            value_delta = beta_ref[batch_index, head_index] * (values - decay_value * state_times_key)
            value_delta_matrix = plgpu.layout_cast(
                jax.lax.broadcast_in_dim(
                    value_delta,
                    (values_per_program, head_dim),
                    (0,),
                ),
                state_layout,
            )
            updated_state = decay_value * state + value_delta_matrix * key_matrix
            state_ref[batch_index, head_index, value_slice, :] = jnp.where(
                active_ref[batch_index],
                updated_state,
                state,
            )
            outputs_ref[batch_index, head_index, value_slice] = (
                decay_value * state_times_query + value_delta * key_times_query
            )

        return jax.freeze(state_ref), jax.freeze(outputs_ref)

    return update


@deltanet_recurrent_scan.def_vmap
def _deltanet_recurrent_scan_vmap(
    axis_size: int,
    in_batched: list[bool],
    queries: Array,
    keys: Array,
    values: Array,
    decay_factor: Array,
    beta: Array,
    initial_state: Array,
    num_steps: Array | int,
) -> tuple[tuple[Array, Array], tuple[bool, bool]]:
    (
        queries_batched,
        keys_batched,
        values_batched,
        decay_factor_batched,
        beta_batched,
        initial_state_batched,
        num_steps_batched,
    ) = in_batched
    if not all(
        (
            queries_batched,
            keys_batched,
            values_batched,
            decay_factor_batched,
            beta_batched,
            initial_state_batched,
        )
    ):
        raise ValueError("Expected queries, keys, values, decay_factor, beta, and initial_state to be batched.")
    if not num_steps_batched:
        num_steps = jnp.broadcast_to(num_steps, (axis_size,))
    _, num_tokens, num_heads, head_dim = queries.shape
    value_head_dim = values.shape[-1]
    state_sharding = jax.typeof(initial_state).sharding
    if (
        num_tokens != 1
        or head_dim != 128
        or value_head_dim % 64 != 0
        or initial_state.dtype != jnp.float32
        or not isinstance(state_sharding, NamedSharding)
        or any(axis is not None for axis in state_sharding.spec)
        or not supports_mosaic_gpu(state_sharding.mesh, 9)
    ):
        warnings.warn(
            "Pallas DeltaNet recurrence does not support this recurrent configuration; "
            "falling back to XLA recurrence.",
            RuntimeWarning,
            stacklevel=2,
        )
        return (
            jax.vmap(xla_recurrent_scan)(
                queries,
                keys,
                values,
                decay_factor,
                beta,
                initial_state,
                num_steps,
            ),
            (True, True),
        )

    final_state, outputs = _make_batched_update(axis_size, num_heads, value_head_dim)(
        queries[:, 0],
        keys[:, 0],
        values[:, 0],
        jnp.exp(decay_factor[:, 0]),
        beta[:, 0],
        initial_state,
        jnp.asarray(num_steps) > 0,
    )
    return (
        (outputs[:, None], final_state),
        (True, True),
    )
