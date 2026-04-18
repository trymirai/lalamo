from typing import NamedTuple, Self

import jax.tree_util as jtu
from jaxtyping import Array

from lalamo.utils.JSON import JSON
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.surgery import load_as

__all__ = [
    "ExportResults",
    "Exportable",
]


class ExportResults(NamedTuple):
    arrays: dict[str, Array]
    metadata: dict[str, JSON]


class Exportable:
    def export(self) -> ExportResults:
        flat_with_path, _ = jtu.tree_flatten_with_path(
            self,
            is_leaf=lambda x: isinstance(x, Exportable) and (x is not self),
        )
        result_arrays: dict[str, Array] = {}
        result_metadata: dict[str, JSON] = {}

        for path, leaf in flat_with_path:
            key = ParameterPath("") / path
            if isinstance(leaf, Exportable):
                leaf_arrays, leaf_metadata = leaf.export()
                for sub_key in leaf_arrays.keys() | leaf_metadata.keys():
                    if sub_key in leaf_arrays:
                        result_arrays[str(key / sub_key)] = leaf_arrays[sub_key]
                    if sub_key in leaf_metadata:
                        result_metadata[str(key / sub_key)] = leaf_metadata[sub_key]
            else:
                result_arrays[key] = leaf
        return ExportResults(result_arrays, result_metadata)

    def load_exported(
        self,
        expored_data: ExportResults,
        allow_dtype_cast: bool = False,
        *,
        prefix: ParameterPath | None = None,
    ) -> Self:
        if prefix is None:
            prefix = ParameterPath()

        def restore(jax_path: tuple[object, ...], subtree: Exportable | Array) -> Exportable | Array:
            path = prefix / jax_path

            if isinstance(subtree, Exportable):
                return subtree.load_exported(expored_data, allow_dtype_cast=allow_dtype_cast, prefix=path)

            return load_as(subtree, expored_data.arrays[path], allow_dtype_cast=allow_dtype_cast)

        return jtu.tree_map_with_path(
            restore,
            self,
            is_leaf=lambda x: isinstance(x, Exportable) and x is not self,
        )
