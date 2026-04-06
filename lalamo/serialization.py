from collections.abc import Mapping
from typing import Any

import jax

from lalamo.common import stringify_path


class Serializable:
    def to_uzu(self) -> dict[str, Any]:
        flat_with_path, _ = jax.tree_util.tree_flatten_with_path(
            self, is_leaf=lambda x: isinstance(x, Serializable) and x is not self
        )
        result: dict[str, Any] = {}
        for path, leaf in flat_with_path:
            key = stringify_path(path)
            if isinstance(leaf, Serializable):
                for sub_key, v in leaf.to_uzu().items():
                    result[f"{key}.{sub_key}"] = v
            else:
                result[key] = leaf
        return result


def strip_uzu_prefix(data: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    prefix_with_separator = f"{prefix}."
    return {
        key.removeprefix(prefix_with_separator): value
        for key, value in data.items()
        if key.startswith(prefix_with_separator)
    }
