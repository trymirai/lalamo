from collections.abc import Callable
from functools import partial
from typing import Final, Protocol, overload, runtime_checkable

import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float, Int, PyTree, Shaped

from lalamo.module import Keychain, KeychainBroadcastMode, LogicalAxis, ShardingConfig

__all__ = [
    "apply_soft_capping",
    "call_vmapped",
    "call_vmapped_twice",
    "gather_suffix_tokens",
]


class _Unspecified:
    pass


_UNSPECIFIED: Final = _Unspecified()


@runtime_checkable
class VmappableWithKeychain[*ArgT, ResultT](Protocol):
    def __call__(self, *args: *ArgT, keychain: Keychain) -> ResultT: ...


@overload
def call_vmapped[ResultT](
    fn: Callable[..., ResultT],
    /,
    *args: PyTree[Shaped[Array, "..."]],
    in_axes: PyTree[int | None] = 0,
    out_axes: PyTree[int | None] = 0,
    added_sharding_axis: str | None = None,
) -> ResultT: ...


@overload
def call_vmapped[ResultT](
    fn: Callable[..., ResultT],
    /,
    *args: PyTree[Shaped[Array, "..."]],
    forward_pass_config: object,
    in_axes: PyTree[int | None] = 0,
    out_axes: PyTree[int | None] = 0,
    added_sharding_axis: str | None = None,
) -> ResultT: ...


@overload
def call_vmapped[ResultT](
    fn: Callable[..., ResultT],
    /,
    *args: PyTree[Shaped[Array, "..."]],
    keychain: Keychain,
    in_axes: PyTree[int | None] = 0,
    out_axes: PyTree[int | None] = 0,
    added_sharding_axis: str | None = None,
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
    added_sharding_axis: str | None = None,
) -> ResultT: ...


@overload
def call_vmapped[*ArgT, ResultT](
    fn: VmappableWithKeychain[*ArgT, ResultT],
    /,
    *args: *ArgT,
    keychain: Keychain,
    in_axes: PyTree[int | None] = 0,
    out_axes: PyTree[int | None] = 0,
    added_sharding_axis: str | None = None,
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
        return fn(
            *mapped_args,
            keychain=Keychain(
                vmapped_keys=vmapped_keys,
                batch_key=keychain.batch_key,
                sharding_config=keychain.sharding_config,
            ),
        )

    return wrapped


def call_vmapped[ResultT](
    fn: Callable[..., ResultT],
    /,
    *args: PyTree[Shaped[Array, "..."]],
    forward_pass_config: object | _Unspecified = _UNSPECIFIED,
    keychain: Keychain | _Unspecified = _UNSPECIFIED,
    in_axes: PyTree[int | None] = 0,
    out_axes: PyTree[int | None] = 0,
    added_sharding_axis: str | None = None,
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
    added_sharding_axes: tuple[str | None, str | None] = (None, None),
) -> ResultT: ...


@overload
def call_vmapped_twice[ResultT](
    fn: Callable[..., ResultT],
    /,
    *args: PyTree[Shaped[Array, "..."]],
    forward_pass_config: object,
    in_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    out_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    added_sharding_axes: tuple[str | None, str | None] = (None, None),
) -> ResultT: ...


@overload
def call_vmapped_twice[ResultT](
    fn: Callable[..., ResultT],
    /,
    *args: PyTree[Shaped[Array, "..."]],
    keychain: Keychain,
    in_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    out_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    added_sharding_axes: tuple[str | None, str | None] = (None, None),
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
    added_sharding_axes: tuple[str | None, str | None] = (None, None),
) -> ResultT: ...


@overload
def call_vmapped_twice[*ArgT, ResultT](
    fn: VmappableWithKeychain[*ArgT, ResultT],
    /,
    *args: *ArgT,
    keychain: Keychain,
    in_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    out_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    added_sharding_axes: tuple[str | None, str | None] = (None, None),
) -> ResultT: ...


def call_vmapped_twice[ResultT](
    fn: Callable[..., ResultT],
    /,
    *args: PyTree[Shaped[Array, "..."]],
    forward_pass_config: object | _Unspecified = _UNSPECIFIED,
    keychain: Keychain | _Unspecified = _UNSPECIFIED,
    in_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    out_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    added_sharding_axes: tuple[str | None, str | None] = (None, None),
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
    values: Float[Array, "..."],
    soft_cap: float,
) -> Float[Array, "..."]:
    return jax.nn.tanh(values / soft_cap) * soft_cap


def gather_suffix_tokens(
    features: Shaped[Array, "batch tokens *channels"],
    lengths_without_padding: Int[Array, " batch"] | None,
    num_suffix_tokens: int,
    sharding_config: ShardingConfig,
) -> Shaped[Array, "batch suffix_tokens *channels"]:
    """Gather the last `num_suffix_tokens` positions of every sequence, counting back from the end
    of its non-padded prefix rather than from the physical end of the token axis.

    Sequences shorter than the window get position-0 duplicates at the window start; callers must
    not consume outputs at those slots.
    """
    batch_size, sequence_length, *_ = features.shape
    if lengths_without_padding is None:
        return features[:, sequence_length - num_suffix_tokens :]

    batch_axis = sharding_config.resolve_axis(LogicalAxis.BATCH)
    suffix_offsets = jnp.arange(-num_suffix_tokens, 0, dtype=jnp.int32)
    suffix_positions = jnp.clip(
        lengths_without_padding[:, None] + suffix_offsets[None, :],
        0,
        sequence_length - 1,
    )
    batch_indices = jax.device_put(
        jnp.arange(batch_size, dtype=jnp.int32)[:, None],
        sharding_config.make_sharding((batch_axis, None)),
    )
    out_sharding = sharding_config.make_sharding((batch_axis, *(None,) * (features.ndim - 1)))
    return features.at[batch_indices, suffix_positions].get(out_sharding=out_sharding)
