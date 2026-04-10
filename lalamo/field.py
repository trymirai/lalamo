import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, overload

import equinox as eqx
import jax.tree_util as jtu

if TYPE_CHECKING:
    from lalamo.sharding import ShardingOrder, TensorSharding


class ParameterPath(str):
    """
    ParameterPath is a universal representation, allowing to index into either the pytree, or flattened
    representations of modules. The invariant is as follows:

    For any PyTree and any ParameterPath:
    ```py
    get_by_path(pytree, path) == get_by_path(pytree.to_uzu(), path)

    jax.tree.map_with_path(lambda v, path: v == get_by_path(pytree, ParameterPath(path)), pytree)
    ```

    Basically ParameterPath is a universal str representation for any Path within any PyTree.

    The only constraint is that you can't use dictionary keys with dots inside of them, if you try - we will throw up.
    """

    __slots__ = ()

    @overload
    def __truediv__(self, other: str) -> "ParameterPath": ...
    @overload
    def __truediv__(self, other: int) -> "ParameterPath": ...
    @overload
    def __truediv__(self, other: jtu.GetAttrKey) -> "ParameterPath": ...
    @overload
    def __truediv__(self, other: jtu.SequenceKey) -> "ParameterPath": ...
    @overload
    def __truediv__(self, other: jtu.DictKey) -> "ParameterPath": ...
    @overload
    def __truediv__(self, other: jtu.FlattenedIndexKey) -> "ParameterPath": ...
    @overload
    def __truediv__(self, other: tuple[Any, ...]) -> "ParameterPath": ...

    def __truediv__(  # noqa: PLR0911
        self,
        other: str | int | jtu.GetAttrKey | jtu.SequenceKey | jtu.DictKey | jtu.FlattenedIndexKey | tuple[Any, ...],
    ) -> "ParameterPath":
        match other:
            case str():
                if not self:
                    return ParameterPath(other)
                return ParameterPath(f"{self}.{other}")
            case int():
                return self / str(other)
            case jtu.GetAttrKey(name):
                return self / name
            case jtu.SequenceKey(idx):
                return self / str(idx)
            case jtu.DictKey(key=key):
                key_str = str(key)
                if "." in key_str:
                    raise ValueError(
                        f"DictKey {key_str!r} contains dots, which would be ambiguous in a dot-separated path"
                    )
                return self / key_str
            case jtu.FlattenedIndexKey(key):
                return self / key
            case tuple():
                result = self
                for element in other:
                    result = result / element
                return result
            case _:
                raise TypeError(f"Unsupported type for ParameterPath /: {type(other).__name__}")


class ParameterNorm(Enum):
    SPECTRAL = "spectral"
    L_INF = "l_inf"
    L_2 = "l_2"


@dataclass(frozen=True)
class FieldMetadata:
    tensor_sharding: "TensorSharding | None" = None
    sharding_order: "ShardingOrder | None" = None
    min_size_to_shard: int = 0
    to_uzu: Callable[[Any, Any], Any] | None = None
    from_uzu: Callable[[Any, Any], Any] | None = None
    trainable: bool = True
    norm: ParameterNorm = ParameterNorm.SPECTRAL

    @property
    def has_uzu_converters(self) -> bool:
        return self.to_uzu is not None or self.from_uzu is not None


_EMPTY = FieldMetadata()


def field_metadata_from_path(
    module: Any,  # noqa: ANN401
    path: ParameterPath,
) -> FieldMetadata:
    if not path:
        return _EMPTY

    cur: Any = module
    owner: eqx.Module = module
    owner_field: dataclasses.Field[Any] | None = None

    for component in path.split("."):
        if isinstance(cur, eqx.Module):
            owner = cur
            owner_field = next((f for f in dataclasses.fields(cur) if f.name == component), None)
            cur = getattr(cur, component)
        elif isinstance(cur, (list, tuple)):
            cur = cur[int(component)]
        elif isinstance(cur, dict):
            cur = cur[component]
        else:
            raise TypeError(f"Cannot traverse {type(cur).__name__} with component {component!r} in path {path!r}")

    if owner_field is None:
        return _EMPTY

    meta = owner_field.metadata
    return FieldMetadata(
        tensor_sharding=meta.get("tensor_sharding"),
        sharding_order=getattr(owner, "sharding_order", None),
        min_size_to_shard=meta.get("min_size_to_shard", 0),
        to_uzu=meta.get("to_uzu"),
        from_uzu=meta.get("from_uzu"),
        trainable=meta.get("trainable", True),
        norm=meta.get("norm", ParameterNorm.SPECTRAL),
    )


def find_field_metadata_by_value(module: eqx.Module, target: object) -> FieldMetadata | None:
    flat_with_path, _ = jtu.tree_flatten_with_path(module, is_leaf=lambda x: x is target)
    for path, leaf in flat_with_path:
        if leaf is target:
            return field_metadata_from_path(module, ParameterPath("") / path)
    return None


def field(
    tensor_sharding: Any = None,  # noqa: ANN401
    min_size_to_shard: int = 2**18,
    *,
    to_uzu: Callable[[Any, Any], Any] | None = None,
    from_uzu: Callable[[Any, Any], Any] | None = None,
    trainable: bool = True,
    norm: ParameterNorm = ParameterNorm.SPECTRAL,
    converter: Callable[[Any], Any] | None = None,
    static: bool = False,
    metadata: Mapping[str, Any] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    merged_metadata = dict(metadata or {})
    merged_metadata["tensor_sharding"] = tensor_sharding
    merged_metadata["min_size_to_shard"] = min_size_to_shard
    merged_metadata["to_uzu"] = to_uzu
    merged_metadata["from_uzu"] = from_uzu
    merged_metadata["trainable"] = trainable
    merged_metadata["norm"] = norm
    return eqx.field(
        converter=converter,
        static=static,
        metadata=merged_metadata,
        **kwargs,
    )
