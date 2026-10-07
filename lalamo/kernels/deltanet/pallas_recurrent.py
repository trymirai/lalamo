import warnings
from collections.abc import Callable
from functools import cache
from math import prod

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import Array

from lalamo.kernels.mosaic import supports_mosaic_gpu

from .xla import xla_recurrent_scan

__all__ = [
    "deltanet_recurrent_scan",
]

type DeltaScan = Callable[
    [Array, Array, Array, Array, Array, Array, Array],
    tuple[Array, Array],
]


@jax.custom_batching.custom_vmap
def deltanet_recurrent_scan(
    queries: Array,
    keys: Array,
    values: Array,
    decay_factor: Array,
    beta: Array,
    initial_state: Array,
    num_steps: Array | int,
) -> tuple[Array, Array]:
    (outputs, final_state), _ = _deltanet_recurrent_scan_vmap(
        1,
        [True, True, True, True, True, True, False],
        queries[None],
        keys[None],
        values[None],
        decay_factor[None],
        beta[None],
        initial_state[None],
        num_steps,
    )
    return outputs[0], final_state[0]


@cache
def _make_batched_scan(
    batch_size: int,
    num_tokens: int,
    num_heads: int,
    value_head_dim: int,
) -> DeltaScan:
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

    def kernel(
        queries_ref: jax.Ref,
        keys_ref: jax.Ref,
        values_ref: jax.Ref,
        decay_ref: jax.Ref,
        beta_ref: jax.Ref,
        initial_state_ref: jax.Ref,
        num_steps_ref: jax.Ref,
        final_state_ref: jax.Ref,
        outputs_ref: jax.Ref,
    ) -> None:
        batch_index = jax.lax.axis_index("batch")
        head_index = jax.lax.axis_index("head")
        value_start = jax.lax.axis_index("value") * values_per_program
        value_slice = pl.ds(value_start, values_per_program)
        state = plgpu.load(
            initial_state_ref.at[batch_index, head_index, value_slice, :],
            layout=state_layout,
            optimized=False,
        ).astype(jnp.float32)

        def step(token_index: Array, state: Array) -> Array:
            query = plgpu.load(
                queries_ref.at[batch_index, token_index, head_index, :],
                layout=key_layout,
                optimized=False,
            ).astype(jnp.float32)
            key = plgpu.load(
                keys_ref.at[batch_index, token_index, head_index, :],
                layout=key_layout,
                optimized=False,
            ).astype(jnp.float32)
            values = plgpu.load(
                values_ref.at[batch_index, token_index, head_index, value_slice],
                layout=value_layout,
                optimized=False,
            ).astype(jnp.float32)
            key_matrix = plgpu.layout_cast(
                jax.lax.broadcast_in_dim(key, (values_per_program, head_dim), (1,)),
                state_layout,
            )
            query_matrix = plgpu.layout_cast(
                jax.lax.broadcast_in_dim(query, (values_per_program, head_dim), (1,)),
                state_layout,
            )
            state_times_key = jnp.sum(state * key_matrix, axis=-1)
            state_times_query = jnp.sum(state * query_matrix, axis=-1)
            key_times_query = jnp.sum(key * query)
            decay_value = decay_ref[batch_index, token_index, head_index]
            value_delta = beta_ref[batch_index, token_index, head_index] * (values - decay_value * state_times_key)
            value_delta_matrix = plgpu.layout_cast(
                jax.lax.broadcast_in_dim(value_delta, (values_per_program, head_dim), (0,)),
                state_layout,
            )
            updated_state = decay_value * state + value_delta_matrix * key_matrix
            outputs_ref[batch_index, token_index, head_index, value_slice] = (
                decay_value * state_times_query + value_delta * key_times_query
            )
            return jnp.where(token_index < num_steps_ref[batch_index], updated_state, state)

        final_state_ref[batch_index, head_index, value_slice, :] = jax.lax.fori_loop(0, num_tokens, step, state)

    return plgpu.kernel(
        kernel,
        out_type=(
            jax.ShapeDtypeStruct((batch_size, num_heads, value_head_dim, head_dim), jnp.float32),
            jax.ShapeDtypeStruct((batch_size, num_tokens, num_heads, value_head_dim), jnp.float32),
        ),
        grid=(batch_size, num_heads, value_head_dim // values_per_program),
        grid_names=("batch", "head", "value"),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Lane,
            reduction_scratch_bytes=8_192,
        ),
        kernel_name=f"deltanet_b{batch_size}_t{num_tokens}",
    )


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
    arguments = (queries, keys, values, decay_factor, beta, initial_state, num_steps)
    first_batched = next(argument for argument, batched in zip(arguments, in_batched, strict=True) if batched)
    batch_sharding = jax.typeof(jnp.asarray(first_batched)).sharding
    assert isinstance(batch_sharding, NamedSharding)

    def broadcast(argument: Array | int, batched: bool) -> Array:
        argument = jnp.asarray(argument)
        if batched:
            return argument
        out_sharding = None
        if not batch_sharding.mesh.empty:
            argument_sharding = jax.typeof(argument).sharding
            out_sharding = NamedSharding(
                batch_sharding.mesh,
                PartitionSpec(batch_sharding.spec[0], *argument_sharding.spec),
            )
        return jnp.broadcast_to(argument, (axis_size, *argument.shape), out_sharding=out_sharding)

    queries, keys, values, decay_factor, beta, initial_state, num_steps = (
        broadcast(argument, batched) for argument, batched in zip(arguments, in_batched, strict=True)
    )
    _, num_tokens, num_heads, head_dim = queries.shape
    value_head_dim = values.shape[-1]
    sharding = jax.typeof(queries).sharding
    assert isinstance(sharding, NamedSharding)
    mesh = sharding.mesh
    # Implicit single-device arrays have an empty abstract mesh in JAX's types.
    if mesh.empty:
        mesh = jax.make_mesh((1,), ("replica",), devices=jax.devices()[:1])

    def partition_count(axis: str | tuple[str, ...] | None) -> int:
        if axis is None:
            return 1
        if isinstance(axis, str):
            return mesh.shape[axis]
        return prod(mesh.shape[name] for name in axis)

    batch_axis, token_axis, head_axis, key_axis = sharding.spec
    if (
        head_dim != 128
        or value_head_dim % 64 != 0
        or initial_state.dtype != jnp.float32
        or partition_count(token_axis) > 1
        or partition_count(key_axis) > 1
        or not supports_mosaic_gpu(mesh, 9)
    ):
        if mesh.abstract_mesh.abstract_device.platform != "cpu":
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

    scan = _make_batched_scan(
        axis_size // partition_count(batch_axis),
        num_tokens,
        num_heads // partition_count(head_axis),
        value_head_dim,
    )
    if not sharding.mesh.empty:
        scan = jax.shard_map(
            scan,
            mesh=mesh,
            in_specs=(
                PartitionSpec(batch_axis, None, head_axis),
                PartitionSpec(batch_axis, None, head_axis),
                PartitionSpec(batch_axis, None, head_axis),
                PartitionSpec(batch_axis, None, head_axis),
                PartitionSpec(batch_axis, None, head_axis),
                PartitionSpec(batch_axis, head_axis),
                PartitionSpec(batch_axis),
            ),
            out_specs=(PartitionSpec(batch_axis, head_axis), PartitionSpec(batch_axis, None, head_axis)),
            check_vma=False,
        )
    final_state, outputs = scan(queries, keys, values, jnp.exp(decay_factor), beta, initial_state, num_steps)
    return (
        (outputs, final_state),
        (True, True),
    )
