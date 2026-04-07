from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Self

import jax

from lalamo.common import stringify_path

if TYPE_CHECKING:
    from lalamo.modules.common import ShardingConfig


class UzuSerializable:
    def to_uzu(self) -> dict[str, Any]:
        flat_with_path, _ = jax.tree_util.tree_flatten_with_path(
            self, is_leaf=lambda x: isinstance(x, UzuSerializable) and x is not self
        )
        result: dict[str, Any] = {}
        for path, leaf in flat_with_path:
            key = stringify_path(path)
            if isinstance(leaf, UzuSerializable):
                for sub_key, value in leaf.to_uzu().items():
                    result[f"{key}.{sub_key}"] = value
            else:
                result[key] = leaf
        return result

    def from_uzu(
        self,
        data: Mapping[str, Any],
        prefix: str = "",
        sharding_config: "ShardingConfig | None" = None,
    ) -> Self:
        def _make_key(path: tuple[object, ...]) -> str:
            key = stringify_path(path)
            return f"{prefix}.{key}" if prefix else key

        def restore(path: tuple[object, ...], leaf: object) -> object:
            key = _make_key(path)
            if isinstance(leaf, UzuSerializable):
                return leaf.from_uzu(data, prefix=key, sharding_config=sharding_config)
            if key in data:
                return data[key]
            return leaf

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
