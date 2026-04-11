from collections.abc import Mapping
from typing import Self

import jax.tree_util as jtu
from jaxtyping import Array

from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.surgery import load_as

__all__ = ["Exportable"]


class Exportable:
    def export(self) -> dict[str, Array]:
        flat_with_path, _ = jtu.tree_flatten_with_path(
            self,
            is_leaf=lambda x: isinstance(x, Exportable) and (x is not self),
        )
        result: dict[str, Array] = {}
        for path, leaf in flat_with_path:
            key = ParameterPath("") / path
            if isinstance(leaf, Exportable):
                for sub_key, value in leaf.export().items():
                    result[key / sub_key] = value
            else:
                result[key] = leaf
        return result

    def load_exported(
        self,
        data: Mapping[str, Array],
        allow_dtype_cast: bool = False,
        *,
        prefix: ParameterPath | None = None,
    ) -> Self:
        if prefix is None:
            prefix = ParameterPath()

        def restore(jax_path: tuple[object, ...], subtree: Exportable | Array) -> Exportable | Array:
            path = prefix / jax_path

            if isinstance(subtree, Exportable):
                return subtree.load_exported(data, allow_dtype_cast=allow_dtype_cast, prefix=path)

            return load_as(subtree, data[path], allow_dtype_cast=allow_dtype_cast)

        return jtu.tree_map_with_path(
            restore,
            self,
            is_leaf=lambda x: isinstance(x, Exportable) and x is not self,
        )
