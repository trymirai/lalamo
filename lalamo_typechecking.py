import typing
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin

from beartype import BeartypeConf, FrozenDict, beartype
from jax import Array, ShapeDtypeStruct
from jaxtyping import PyTree

__all__ = ["typechecker"]

_ARRAY_OR_DUMMY = Array | ShapeDtypeStruct
_BEARTYPE = beartype(conf=BeartypeConf(hint_overrides=FrozenDict({Array: _ARRAY_OR_DUMMY})))
_SELF_HINT: Any = vars(typing)["Self"]


def _self_name(qualname: str) -> str | None:
    name, separator, _ = qualname.partition(".")
    if not separator or name == "<locals>":
        return None
    return name


def _widen(hint: Any, self_name: str | None) -> Any:
    if hint is _SELF_HINT and self_name is not None:
        if self_name == "ShapeDtypeMatrix":
            return "WeightMatrix"
        return self_name
    if hint is Array:
        return _ARRAY_OR_DUMMY

    dtype = getattr(hint, "dtype", None)
    dim_str = getattr(hint, "dim_str", None)
    if getattr(hint, "array_type", None) is Array and dtype is not None and isinstance(dim_str, str):
        return dtype[_ARRAY_OR_DUMMY, dim_str]

    leaf_type = getattr(hint, "leaftype", None)
    if leaf_type is not None:
        widened_leaf_type = _widen(leaf_type, self_name)
        if widened_leaf_type is leaf_type:
            return hint
        structure = getattr(hint, "structure", None)
        if structure is None:
            return PyTree[widened_leaf_type]
        return PyTree[widened_leaf_type, structure]

    origin = get_origin(hint)
    if origin is None:
        return hint

    if isinstance(origin, type) and issubclass(origin, tuple) and hasattr(origin, "_fields"):
        return tuple

    args = get_args(hint)
    if not args:
        return hint

    widened_args = []
    for arg in args:
        if isinstance(arg, list):
            widened_args.append([_widen(child, self_name) for child in arg])
        else:
            widened_args.append(_widen(arg, self_name))
    if all(widened_arg is arg for widened_arg, arg in zip(widened_args, args, strict=True)):
        return hint

    if origin in (UnionType, Union):
        union_subscript: Any = Union
        return union_subscript[tuple(widened_args)]
    if origin is Annotated:
        annotated_subscript: Any = Annotated
        return annotated_subscript[tuple(widened_args)]
    origin_subscript: Any = origin
    if len(widened_args) == 1:
        return origin_subscript[widened_args[0]]
    return origin_subscript[tuple(widened_args)]


def typechecker(fn: Any) -> Any:
    self_name = _self_name(fn.__qualname__)
    original_annotations = fn.__annotations__
    fn.__annotations__ = {name: _widen(annotation, self_name) for name, annotation in original_annotations.items()}
    try:
        return _BEARTYPE(fn)
    finally:
        fn.__annotations__ = original_annotations
