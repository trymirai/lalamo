from collections.abc import Mapping
from typing import NamedTuple, Self

import jax
import jax.tree_util as jtu
from jax import ShapeDtypeStruct
from jaxtyping import Array

from lalamo.safetensors import SafeTensorArrays
from lalamo.utils.json import JSON
from lalamo.utils.parameter_path import ParameterPath

__all__ = [
    "ExportResults",
    "Exportable",
]


class ExportResults(NamedTuple):
    arrays: Mapping[str, Array]
    metadata: Mapping[str, JSON]


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
            elif isinstance(leaf, jax.Array):
                result_arrays[str(key)] = leaf
        return ExportResults(result_arrays, result_metadata)

    def load_exported(
        self,
        exported_data: ExportResults,
        *,
        prefix: ParameterPath | None = None,
    ) -> Self:
        # Local import avoids Exportable -> surgery -> WeightMatrix -> Exportable at module import time.
        from lalamo.utils.surgery import load_as  # noqa: PLC0415

        if prefix is None:
            prefix = ParameterPath()

        def restore(jax_path: tuple[object, ...], subtree: object) -> object:
            path = prefix / jax_path

            if isinstance(subtree, Exportable):
                return subtree.load_exported(exported_data, prefix=path)
            if not isinstance(subtree, (jax.Array, ShapeDtypeStruct)):
                return subtree

            try:
                if isinstance(exported_data.arrays, SafeTensorArrays):
                    array = exported_data.arrays.read_tensor(path, subtree.sharding)
                else:
                    array = exported_data.arrays[path]
                return load_as(subtree, array)
            except Exception as e:
                raise ValueError(f"Failed to load {path}: {e}") from e

        return jtu.tree_map_with_path(
            restore,
            self,
            is_leaf=lambda x: isinstance(x, Exportable) and x is not self,
        )
