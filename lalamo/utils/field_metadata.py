import equinox as eqx
import jax.tree_util as jtu
from jax import Array

from lalamo.module import FieldMetadata, ParameterNorm

__all__ = [
    "field_metadata_for_leaf",
    "partition_trainable",
]


_DEFAULT_METADATA = FieldMetadata()


def field_metadata_for_leaf(module: eqx.Module, path: tuple[object, ...]) -> FieldMetadata:
    """Resolve the FieldMetadata declared on the dataclass field that owns the leaf at *path*."""
    current: object = module
    for key in path:
        if hasattr(key, "idx"):
            assert isinstance(current, (list, tuple))
            assert isinstance(key.idx, int)
            current = current[key.idx]
            continue
        name = key.name if hasattr(key, "name") else str(key)
        fields = getattr(current, "__dataclass_fields__", None)
        if fields is None or name not in fields:
            return _DEFAULT_METADATA
        meta = fields[name].metadata
        child = getattr(current, name)
        if isinstance(child, (eqx.Module, list, tuple)):
            current = child
            continue
        return FieldMetadata(
            trainable=meta.get("trainable", True),
            norm=meta.get("norm", ParameterNorm.L_INF),
        )
    return _DEFAULT_METADATA


def partition_trainable[M: eqx.Module](module: M) -> tuple[M, M]:
    """Split *module* into trainable and frozen parts using field metadata."""
    leaves_with_paths, treedef = jtu.tree_flatten_with_path(module)
    filter_spec = treedef.unflatten(
        field_metadata_for_leaf(module, path).trainable if isinstance(leaf, Array) else False
        for path, leaf in leaves_with_paths
    )
    return eqx.partition(module, filter_spec)
