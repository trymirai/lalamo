from collections.abc import Callable
from functools import partial
from typing import Final, Protocol, overload

import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float, PyTree, Shaped

from lalamo.module import Keychain, KeychainBroadcastMode, ShardingAxis

__all__ = [
    "apply_soft_capping",
    "call_vmapped",
    "call_vmapped_twice",
]


class _Unspecified:
    pass


_UNSPECIFIED: Final = _Unspecified()


class VmappableWithKeychain[*ArgT, ResultT](Protocol):
    def __call__(self, *args: *ArgT, keychain: Keychain) -> ResultT: ...


@overload
def call_vmapped[ResultT](
    fn: Callable[..., ResultT],
    /,
    *args: PyTree[Shaped[Array, "..."]],
    in_axes: PyTree[int | None] = 0,
    out_axes: PyTree[int | None] = 0,
    added_sharding_axis: ShardingAxis | None = None,
) -> ResultT: ...


@overload
def call_vmapped[ResultT](
    fn: Callable[..., ResultT],
    /,
    *args: PyTree[Shaped[Array, "..."]],
    forward_pass_config: object,
    in_axes: PyTree[int | None] = 0,
    out_axes: PyTree[int | None] = 0,
    added_sharding_axis: ShardingAxis | None = None,
) -> ResultT: ...


@overload
def call_vmapped[ResultT](
    fn: Callable[..., ResultT],
    /,
    *args: PyTree[Shaped[Array, "..."]],
    keychain: Keychain,
    in_axes: PyTree[int | None] = 0,
    out_axes: PyTree[int | None] = 0,
    added_sharding_axis: ShardingAxis | None = None,
) -> ResultT: ...


@overload
def call_vmapped[ResultT](
    fn: Callable[..., ResultT],
    /,
    *args: PyTree[Shaped[Array, "..."]],
    forward_pass_config: object,
    keychain: Keychain,
    in_axes: PyTree[int | None] = 0,
    out_axes: PyTree[int | None] = 0,
    added_sharding_axis: ShardingAxis | None = None,
) -> ResultT: ...


@overload
def call_vmapped[*ArgT, ResultT](
    fn: VmappableWithKeychain[*ArgT, ResultT],
    /,
    *args: *ArgT,
    keychain: Keychain,
    in_axes: PyTree[int | None] = 0,
    out_axes: PyTree[int | None] = 0,
    added_sharding_axis: ShardingAxis | None = None,
) -> ResultT: ...


def _bind_shared_kwargs[ResultT](
    fn: Callable[..., ResultT],
    *,
    forward_pass_config: object | _Unspecified = _UNSPECIFIED,
) -> Callable[..., ResultT]:
    mapped_fn: Callable[..., ResultT] = fn
    if not isinstance(forward_pass_config, _Unspecified):
        mapped_fn = partial(mapped_fn, forward_pass_config=forward_pass_config)
    return mapped_fn


def _normalize_in_axes(
    args: tuple[PyTree[Shaped[Array, "..."]], ...],
    in_axes: PyTree[int | None],
) -> tuple[PyTree[int | None], ...]:
    if isinstance(in_axes, tuple) and len(in_axes) == len(args):
        return in_axes
    return (in_axes,) * len(args)


def _key_shape(args: tuple[PyTree[Shaped[Array, "..."]], ...], in_axes: PyTree[int | None]) -> tuple[int, ...]:
    def probe(*_args: PyTree[Shaped[Array, "..."]]) -> Shaped[Array, ""]:
        return jnp.empty((), dtype=jnp.int8)

    return jax.eval_shape(vmap(probe, in_axes=in_axes), *args).shape


def _nested_key_shape(
    args: tuple[PyTree[Shaped[Array, "..."]], ...],
    outer_in_axes: PyTree[int | None],
    inner_in_axes: PyTree[int | None],
) -> tuple[int, ...]:
    def probe(*_args: PyTree[Shaped[Array, "..."]]) -> Shaped[Array, ""]:
        return jnp.empty((), dtype=jnp.int8)

    return jax.eval_shape(vmap(vmap(probe, in_axes=inner_in_axes), in_axes=outer_in_axes), *args).shape


def _call_with_keychain[ResultT](fn: Callable[..., ResultT], *, keychain: Keychain) -> Callable[..., ResultT]:
    def wrapped(*args: PyTree[Shaped[Array, "..."]]) -> ResultT:
        *mapped_args, vmapped_keys = args
        if not isinstance(vmapped_keys, jax.Array):
            raise TypeError(f"Expected JAX array key, got {type(vmapped_keys).__name__}")
        return fn(*mapped_args, keychain=Keychain(vmapped_keys=vmapped_keys, batch_key=keychain.batch_key))

    return wrapped


def call_vmapped[ResultT](
    fn: Callable[..., ResultT],
    /,
    *args: PyTree[Shaped[Array, "..."]],
    forward_pass_config: object | _Unspecified = _UNSPECIFIED,
    keychain: Keychain | _Unspecified = _UNSPECIFIED,
    in_axes: PyTree[int | None] = 0,
    out_axes: PyTree[int | None] = 0,
    added_sharding_axis: ShardingAxis | None = None,
) -> ResultT:
    mapped_fn = _bind_shared_kwargs(
        fn,
        forward_pass_config=forward_pass_config,
    )
    if not isinstance(keychain, _Unspecified):
        mapped_fn = _call_with_keychain(mapped_fn, keychain=keychain)
        keychain = keychain.broadcast(_key_shape(args, in_axes), sharding_axes=(added_sharding_axis,))
        vmapped_fn = vmap(
            mapped_fn,
            in_axes=(*_normalize_in_axes(args, in_axes), 0),
            out_axes=out_axes,
            spmd_axis_name=added_sharding_axis,
        )
        return vmapped_fn(*args, keychain.vmapped_keys)

    vmapped_fn = vmap(mapped_fn, in_axes=in_axes, out_axes=out_axes, spmd_axis_name=added_sharding_axis)
    return vmapped_fn(*args)


@overload
def call_vmapped_twice[ResultT](
    fn: Callable[..., ResultT],
    /,
    *args: PyTree[Shaped[Array, "..."]],
    in_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    out_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    added_sharding_axes: tuple[ShardingAxis | None, ShardingAxis | None] = (None, None),
) -> ResultT: ...


@overload
def call_vmapped_twice[ResultT](
    fn: Callable[..., ResultT],
    /,
    *args: PyTree[Shaped[Array, "..."]],
    forward_pass_config: object,
    in_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    out_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    added_sharding_axes: tuple[ShardingAxis | None, ShardingAxis | None] = (None, None),
) -> ResultT: ...


@overload
def call_vmapped_twice[ResultT](
    fn: Callable[..., ResultT],
    /,
    *args: PyTree[Shaped[Array, "..."]],
    keychain: Keychain,
    in_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    out_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    added_sharding_axes: tuple[ShardingAxis | None, ShardingAxis | None] = (None, None),
) -> ResultT: ...


@overload
def call_vmapped_twice[ResultT](
    fn: Callable[..., ResultT],
    /,
    *args: PyTree[Shaped[Array, "..."]],
    forward_pass_config: object,
    keychain: Keychain,
    in_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    out_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    added_sharding_axes: tuple[ShardingAxis | None, ShardingAxis | None] = (None, None),
) -> ResultT: ...


@overload
def call_vmapped_twice[*ArgT, ResultT](
    fn: VmappableWithKeychain[*ArgT, ResultT],
    /,
    *args: *ArgT,
    keychain: Keychain,
    in_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    out_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    added_sharding_axes: tuple[ShardingAxis | None, ShardingAxis | None] = (None, None),
) -> ResultT: ...


def call_vmapped_twice[ResultT](
    fn: Callable[..., ResultT],
    /,
    *args: PyTree[Shaped[Array, "..."]],
    forward_pass_config: object | _Unspecified = _UNSPECIFIED,
    keychain: Keychain | _Unspecified = _UNSPECIFIED,
    in_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    out_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    added_sharding_axes: tuple[ShardingAxis | None, ShardingAxis | None] = (None, None),
) -> ResultT:
    outer_in_axes, inner_in_axes = in_axes
    outer_out_axes, inner_out_axes = out_axes
    outer_sharding_axis, inner_sharding_axis = added_sharding_axes
    mapped_fn = _bind_shared_kwargs(
        fn,
        forward_pass_config=forward_pass_config,
    )
    if not isinstance(keychain, _Unspecified):
        mapped_fn = _call_with_keychain(mapped_fn, keychain=keychain)
        keychain = keychain.broadcast(
            _nested_key_shape(args, outer_in_axes, inner_in_axes),
            mode=KeychainBroadcastMode.SUFFIX,
            sharding_axes=added_sharding_axes,
        )
        vmapped_once = vmap(
            mapped_fn,
            in_axes=(*_normalize_in_axes(args, inner_in_axes), 0),
            out_axes=inner_out_axes,
            spmd_axis_name=inner_sharding_axis,
        )
        vmapped_twice = vmap(
            vmapped_once,
            in_axes=(*_normalize_in_axes(args, outer_in_axes), 0),
            out_axes=outer_out_axes,
            spmd_axis_name=outer_sharding_axis,
        )
        return vmapped_twice(*args, keychain.vmapped_keys)

    vmapped_once = vmap(
        mapped_fn,
        in_axes=inner_in_axes,
        out_axes=inner_out_axes,
        spmd_axis_name=inner_sharding_axis,
    )
    vmapped_twice = vmap(
        vmapped_once,
        in_axes=outer_in_axes,
        out_axes=outer_out_axes,
        spmd_axis_name=outer_sharding_axis,
    )
    return vmapped_twice(*args)


def apply_soft_capping(
    values: Float[Array, "*"],
    soft_cap: float,
) -> Float[Array, "dst_tokens src_tokens"]:
    return jax.nn.tanh(values / soft_cap) * soft_cap
