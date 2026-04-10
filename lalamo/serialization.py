from collections.abc import Mapping
from typing import Any, Self

import jax

from lalamo.field import ParameterPath, field_metadata_from_path


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
                return subtree.from_uzu(data, prefix=path)

            if metadata.from_uzu is not None:
                return metadata.from_uzu(self, data[path])

            if path in data:
                return data[path]

            return subtree

        return jax.tree_util.tree_map_with_path(
            restore, self, is_leaf=lambda x: isinstance(x, UzuSerializable) and x is not self
        )
