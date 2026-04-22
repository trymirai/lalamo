from collections.abc import Callable
from functools import partial
from typing import Final, Protocol, cast, overload

import jax
from jax import vmap
from jaxtyping import Array, Float, Key, PyTree

__all__ = [
    "apply_soft_capping",
    "call_vmapped",
    "call_vmapped_twice",
]


_UNSPECIFIED: Final = object()


class VmappableWithKey[*ArgT, ResultT](Protocol):
    def __call__(self, *args: *ArgT, key: Key[Array, ""]) -> ResultT: ...


@overload
def call_vmapped[ResultT](
    fn: object,
    /,
    *args: object,
    in_axes: PyTree[int | None] = 0,
    out_axes: PyTree[int | None] = 0,
) -> ResultT: ...


@overload
def call_vmapped[ResultT](
    fn: object,
    /,
    *args: object,
    dequant_key: Key[Array, ""],
    in_axes: PyTree[int | None] = 0,
    out_axes: PyTree[int | None] = 0,
) -> ResultT: ...


@overload
def call_vmapped[ResultT](
    fn: object,
    /,
    *args: object,
    forward_pass_config: object,
    dequant_key: Key[Array, ""],
    in_axes: PyTree[int | None] = 0,
    out_axes: PyTree[int | None] = 0,
) -> ResultT: ...


@overload
def call_vmapped[ResultT](
    fn: object,
    /,
    *args: object,
    key: Key[Array, "..."],
    in_axes: PyTree[int | None] = 0,
    out_axes: PyTree[int | None] = 0,
) -> ResultT: ...


@overload
def call_vmapped[*ArgT, ResultT](
    fn: VmappableWithKey[*ArgT, ResultT],
    /,
    *args: *ArgT,
    key: Key[Array, "..."],
    in_axes: PyTree[int | None] = 0,
    out_axes: PyTree[int | None] = 0,
) -> ResultT: ...


def _bind_shared_kwargs[ResultT](
    fn: Callable[..., ResultT],
    *,
    forward_pass_config: object = _UNSPECIFIED,
    dequant_key: Key[Array, ""] | object = _UNSPECIFIED,
) -> Callable[..., ResultT]:
    mapped_fn: Callable[..., ResultT] = fn
    if forward_pass_config is not _UNSPECIFIED:
        mapped_fn = partial(mapped_fn, forward_pass_config=forward_pass_config)
    if dequant_key is not _UNSPECIFIED:
        mapped_fn = partial(mapped_fn, dequant_key=dequant_key)
    return mapped_fn


def _call_with_mapped_key[ResultT](
    fn: Callable[..., ResultT],
    /,
    *args: object,
    key: Key[Array, "..."] | object = _UNSPECIFIED,
) -> ResultT:
    if key is _UNSPECIFIED:
        return fn(*args)
    return fn(*args, key=key)


def call_vmapped[ResultT](
    fn: object,
    /,
    *args: object,
    forward_pass_config: object = _UNSPECIFIED,
    key: Key[Array, "..."] | object = _UNSPECIFIED,
    dequant_key: Key[Array, ""] | object = _UNSPECIFIED,
    in_axes: PyTree[int | None] = 0,
    out_axes: PyTree[int | None] = 0,
) -> ResultT:
    mapped_fn = _bind_shared_kwargs(
        cast("Callable[..., ResultT]", fn),
        forward_pass_config=forward_pass_config,
        dequant_key=dequant_key,
    )
    vmapped_fn = vmap(mapped_fn, in_axes=in_axes, out_axes=out_axes)
    return _call_with_mapped_key(vmapped_fn, *args, key=key)


@overload
def call_vmapped_twice[ResultT](
    fn: object,
    /,
    *args: object,
    in_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    out_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
) -> ResultT: ...


@overload
def call_vmapped_twice[ResultT](
    fn: object,
    /,
    *args: object,
    dequant_key: Key[Array, ""],
    in_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    out_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
) -> ResultT: ...


@overload
def call_vmapped_twice[ResultT](
    fn: object,
    /,
    *args: object,
    forward_pass_config: object,
    dequant_key: Key[Array, ""],
    in_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    out_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
) -> ResultT: ...


@overload
def call_vmapped_twice[ResultT](
    fn: object,
    /,
    *args: object,
    key: Key[Array, "..."],
    in_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    out_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
) -> ResultT: ...


@overload
def call_vmapped_twice[*ArgT, ResultT](
    fn: VmappableWithKey[*ArgT, ResultT],
    /,
    *args: *ArgT,
    key: Key[Array, "..."],
    in_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    out_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
) -> ResultT: ...


def _split_nested_axes(
    axes: tuple[PyTree[int | None], PyTree[int | None]],
    *,
    argument_name: str,
) -> tuple[PyTree[int | None], PyTree[int | None]]:
    try:
        outer_axes, inner_axes = axes
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{argument_name} for call_vmapped_twice must contain exactly two axis specs: "
            "the outer axes and the inner axes.",
        ) from error
    return outer_axes, inner_axes


def call_vmapped_twice[ResultT](
    fn: object,
    /,
    *args: object,
    forward_pass_config: object = _UNSPECIFIED,
    key: Key[Array, "..."] | object = _UNSPECIFIED,
    dequant_key: Key[Array, ""] | object = _UNSPECIFIED,
    in_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
    out_axes: tuple[PyTree[int | None], PyTree[int | None]] = (0, 0),
) -> ResultT:
    outer_in_axes, inner_in_axes = _split_nested_axes(in_axes, argument_name="in_axes")
    outer_out_axes, inner_out_axes = _split_nested_axes(out_axes, argument_name="out_axes")
    mapped_fn = _bind_shared_kwargs(
        cast("Callable[..., ResultT]", fn),
        forward_pass_config=forward_pass_config,
        dequant_key=dequant_key,
    )
    vmapped_once = vmap(mapped_fn, in_axes=inner_in_axes, out_axes=inner_out_axes)
    vmapped_twice = vmap(vmapped_once, in_axes=outer_in_axes, out_axes=outer_out_axes)
    return _call_with_mapped_key(vmapped_twice, *args, key=key)


def apply_soft_capping(
    values: Float[Array, "*"],
    soft_cap: float,
) -> Float[Array, "dst_tokens src_tokens"]:
    return jax.nn.tanh(values / soft_cap) * soft_cap
