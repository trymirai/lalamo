import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import equinox as eqx
import jax

from lalamo.common import ParameterPath

if TYPE_CHECKING:
    from lalamo.modules.common import ShardingConfig


@dataclass(frozen=True)
class FieldMetadata:
    tensor_sharding: Any = None  # TensorSharding | None, kept as Any to avoid circular import
    sharding_order: Any = None  # ShardingOrder | None
    min_size_to_shard: int = 0
    to_uzu: Callable[[Any, Any], Any] | None = None
    from_uzu: Callable[[Any, Any], Any] | None = None

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
    )


def field(
    tensor_sharding: Any = None,  # noqa: ANN401
    min_size_to_shard: int = 2**18,
    *,
    to_uzu: Callable[[Any, Any], Any] | None = None,
    from_uzu: Callable[[Any, Any], Any] | None = None,
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
    return eqx.field(
        converter=converter,
        static=static,
        metadata=merged_metadata,
        **kwargs,
    )


class UzuSerializable:
    def to_uzu(self) -> dict[str, Any]:
        flat_with_path, _ = jax.tree_util.tree_flatten_with_path(
            self, is_leaf=lambda x: isinstance(x, UzuSerializable) and x is not self
        )
        result: dict[str, Any] = {}
        for path, leaf in flat_with_path:
            key = ParameterPath("") / path
            metadata = field_metadata_from_path(self, key)

            if isinstance(leaf, UzuSerializable) and metadata.has_uzu_converters:
                raise ValueError(
                    f"Both to_uzu inline annotation and UzuSerializable is specified for leaf at {key}, "
                    "which is an undefined behaviour in our framework. You should probably remove the inline "
                    "annotation and make sure to include the desired behaviour in to_uzu method"
                )

            if isinstance(leaf, UzuSerializable):
                for sub_key, value in leaf.to_uzu().items():
                    result[key / sub_key] = value
            elif metadata.to_uzu is not None:
                result[key] = metadata.to_uzu(self, leaf)
            else:
                result[key] = leaf
        return result

    def from_uzu(
        self,
        data: Mapping[str, Any],
        prefix: ParameterPath = ParameterPath(),  # noqa: B008
        sharding_config: "ShardingConfig | None" = None,
    ) -> Self:
        def restore(jax_path: tuple[object, ...], subtree: object) -> object:
            path = prefix / jax_path
            metadata = field_metadata_from_path(self, ParameterPath("") / jax_path)

            if isinstance(subtree, UzuSerializable) and metadata.has_uzu_converters:
                raise ValueError(
                    f"Both to_uzu inline annotation and functional to_uzu converter is specified for leaf at {path}, "
                    "which is an undefined behaviour in our framework. You should probably remove the inline "
                    "annotation and make sure to include the desired behaviour in from_uzu method"
                )

            if isinstance(subtree, UzuSerializable):
                return subtree.from_uzu(data, prefix=path, sharding_config=sharding_config)

            if metadata.from_uzu is not None:
                return metadata.from_uzu(self, data[path])

            if path in data:
                return data[path]

            return subtree

        return jax.tree_util.tree_map_with_path(
            restore, self, is_leaf=lambda x: isinstance(x, UzuSerializable) and x is not self
        )


def strip_uzu_prefix(data: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    prefix_with_separator = f"{prefix}."
    return {
        key.removeprefix(prefix_with_separator): value
        for key, value in data.items()
        if key.startswith(prefix_with_separator)
    }
