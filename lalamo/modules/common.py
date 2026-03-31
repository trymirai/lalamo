import contextlib
import contextvars
import dataclasses
import math
from abc import abstractmethod
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass
from enum import Enum
from types import UnionType
from typing import Any, Self

import equinox as eqx
import jax
import jax.sharding as shd
import jax.tree_util as jtu
from cattrs import Converter
from jax import numpy as jnp
from jax.tree_util import keystr
from jaxtyping import Array, DTypeLike

from lalamo.common import ParameterTree, require_array, require_tree
from lalamo.quantization import QuantizationMode

__all__ = [
    "DummyUnionMember",
    "FieldMetadataInfo",
    "ForwardPassMode",
    "LalamoModule",
    "ParameterLeafInfo",
    "ParameterTree",
    "PositionalEmbeddingSelector",
    "ShardingConfig",
    "ShardingOrder",
    "TensorSharding",
    "combine_parameter_leaves",
    "config_converter",
    "field",
    "field_metadata_by_leaf_id",
    "get_current_sharding_config",
    "iter_parameter_leaves",
    "pad_and_apply_data_sharding",
    "partition_parameter_leaves",
    "register_config_union",
    "require_array",
    "require_tree",
]


class PositionalEmbeddingSelector(Enum):
    GLOBAL = "global"
    LOCAL = "sliding_window"
    NONE = "none"


class ForwardPassMode(Enum):
    MULTI_TOKEN = "multi_token"
    SINGLE_TOKEN = "single_token"


class LalamoModule[ConfigT](eqx.Module):
    config: ConfigT = eqx.field(static=True)

    @property
    @abstractmethod
    def activation_precision(self) -> DTypeLike: ...

    @abstractmethod
    def export_weights(self) -> ParameterTree[Array]: ...

    @abstractmethod
    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self: ...


def _dtype_to_str(dtype: DTypeLike) -> str:
    if dtype == jnp.bfloat16:
        return "bfloat16"
    try:
        return str(dtype.dtype)  # type: ignore
    except AttributeError:
        return str(dtype)


def _str_to_dtype(dtype_str: str) -> jnp.dtype:
    return {
        "int4": jnp.int4,
        "int8": jnp.int8,
        "int16": jnp.int16,
        "int32": jnp.int32,
        "int64": jnp.int64,
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
        "float32": jnp.float32,
        "float64": jnp.float64,
    }[dtype_str]


config_converter = Converter()


config_converter.register_unstructure_hook_func(
    lambda t: t in [jnp.dtype, DTypeLike],
    _dtype_to_str,
)

config_converter.register_structure_hook_func(
    lambda t: t in [jnp.dtype, DTypeLike],
    lambda s, _: _str_to_dtype(s),
)


def register_config_union(union_type: UnionType) -> None:
    union_members = union_type.__args__
    name_to_type = {m.__name__: m for m in union_members}

    def unstructure(obj: Any) -> dict | None:  # noqa: ANN401
        if obj is None:
            return None
        return {
            "type": obj.__class__.__name__,
            **config_converter.unstructure(obj),
        }

    config_converter.register_unstructure_hook(
        union_type,
        unstructure,
    )

    config_converter.register_unstructure_hook(
        union_type | None,
        unstructure,
    )

    def structure[T](config: dict | None, _: type[T]) -> T | None:
        if config is None:
            return None
        new_config = dict(config)
        type_name = new_config.pop("type")
        target_type = name_to_type[type_name]
        return config_converter.structure(new_config, target_type)

    config_converter.register_structure_hook(
        union_type,
        structure,
    )

    config_converter.register_structure_hook(
        union_type | None,
        structure,
    )


@dataclass(frozen=True)
class ShardingConfig:
    mesh: shd.Mesh
    data_axis_name: str = "data"
    tensor_axis_name: str = "tensor"
    fsdp: bool = False

    @classmethod
    def build(
        cls,
        *,
        tensor_parallelism: int | None = None,
        data_parallelism: int | None = None,
        fsdp: bool = False,
        data_axis_name: str = "data",
        tensor_axis_name: str = "tensor",
    ) -> "ShardingConfig":
        device_count = jax.device_count()
        tp = tensor_parallelism
        dp = data_parallelism

        if (tp is not None and tp <= 0) or (dp is not None and dp <= 0):
            raise ValueError("tensor_parallelism and data_parallelism must be positive integers.")

        match (tp, dp):
            case (None, None):
                dp = device_count
                tp = 1
            case (int(), None):
                dp = device_count // tp  # type: ignore
            case (None, int()):
                tp = device_count // dp  # type: ignore

        if tp * dp != device_count:  # type: ignore
            raise ValueError(
                f"data_parallelism * tensor_parallelism must equal the number of devices ({device_count}).",
            )

        mesh = jax.make_mesh(
            (dp, tp),  # type: ignore
            (data_axis_name, tensor_axis_name),
            axis_types=(jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
        )
        return cls(
            mesh=mesh,
            data_axis_name=data_axis_name,
            tensor_axis_name=tensor_axis_name,
            fsdp=fsdp,
        )

    @property
    def data_axis_size(self) -> int:
        return self.mesh.shape[self.data_axis_name]

    @property
    def tensor_axis_size(self) -> int:
        return self.mesh.shape[self.tensor_axis_name]

    def make_sharding(self, pspec: shd.PartitionSpec) -> shd.NamedSharding:
        return shd.NamedSharding(self.mesh, pspec)


_CURRENT_SHARDING: contextvars.ContextVar[ShardingConfig | None] = contextvars.ContextVar(
    "lalamo_current_sharding",
    default=None,
)


@contextlib.contextmanager
def use_sharding(sharding_config: ShardingConfig | None) -> Generator[None, None, None]:
    token = _CURRENT_SHARDING.set(sharding_config)
    try:
        yield
    finally:
        _CURRENT_SHARDING.reset(token)


def get_current_sharding_config() -> ShardingConfig | None:
    return _CURRENT_SHARDING.get()


class ShardingOrder(Enum):
    INPUT = "input"
    OUTPUT = "output"


@dataclass(frozen=True)
class TensorSharding:
    axes: tuple[int, ...]
    axes_names: tuple[ShardingOrder, ...] | None = None

    @classmethod
    def build(
        cls,
        axes: int | tuple[int, ...],
        axes_names: tuple[ShardingOrder, ...] | None = None,
    ) -> "TensorSharding":
        normalized_axes = (axes,) if isinstance(axes, int) else axes
        return TensorSharding(normalized_axes, axes_names)

    def __post_init__(self) -> None:
        if not self.axes:
            raise ValueError("TensorSharding.axes must contain at least one axis.")

        if self.axes_names is not None:
            if len(self.axes_names) != len(self.axes):
                raise ValueError("TensorSharding.axes_names length must match axes length.")
            if len(set(self.axes_names)) != len(self.axes_names):
                raise ValueError("TensorSharding.axes_names must not contain duplicates.")

    def dims_to_try(self, sharding_order: ShardingOrder | None = None) -> tuple[int, ...]:
        if (
            sharding_order is None
            or self.axes_names is None
            or sharding_order not in self.axes_names
            or len(self.axes) == 1
        ):
            return self.axes
        pivot = self.axes_names.index(sharding_order)
        return (self.axes[pivot], *self.axes[:pivot], *self.axes[pivot + 1 :])


@dataclass(frozen=True)
class FieldMetadataInfo:
    owner: eqx.Module
    field: dataclasses.Field


class ParameterNorm(Enum):
    SPECTRAL = "spectral"
    L_INF = "l_inf"


@dataclass(frozen=True)
class ParameterLeafInfo:
    path: str
    owner_type: type[eqx.Module]
    field_name: str
    shape: tuple[int, ...]
    dtype: jnp.dtype
    trainable: bool
    norm: ParameterNorm
    quantized: bool
    quantization_mode: QuantizationMode | None
    tensor_sharding: TensorSharding | None
    min_size_to_shard: int
    alias_of: str | None


@dataclass(frozen=True)
class _ParameterLeafEntry:
    leaf: Array | jax.ShapeDtypeStruct
    canonical_index: int
    info: ParameterLeafInfo


@dataclass(frozen=True)
class _FieldValueEntry:
    path: tuple[Any, ...]
    value: Any
    owner: eqx.Module
    field: dataclasses.Field


def _field_value_entries(module: eqx.Module) -> list[_FieldValueEntry]:
    entries: list[_FieldValueEntry] = []

    def visit(value: Any, *, owner: eqx.Module, field: dataclasses.Field, path: tuple[Any, ...]) -> None:  # noqa: ANN401
        if field.metadata.get("static", False):
            return
        if isinstance(value, eqx.Module):
            for child_field in dataclasses.fields(value):
                visit(
                    getattr(value, child_field.name),
                    owner=value,
                    field=child_field,
                    path=(*path, jtu.GetAttrKey(child_field.name)),
                )
            return
        if isinstance(value, tuple | list):
            for index, item in enumerate(value):
                visit(item, owner=owner, field=field, path=(*path, jtu.SequenceKey(index)))
            return
        if isinstance(value, dict):
            for key in sorted(value):
                item = value[key]
                visit(item, owner=owner, field=field, path=(*path, jtu.DictKey(key)))
            return
        entries.append(_FieldValueEntry(path=path, value=value, owner=owner, field=field))

    for field in dataclasses.fields(module):
        visit(
            getattr(module, field.name),
            owner=module,
            field=field,
            path=(jtu.GetAttrKey(field.name),),
        )
    return entries


def field_metadata_by_leaf_id(module: eqx.Module) -> dict[int, FieldMetadataInfo]:
    field_metadata: dict[int, FieldMetadataInfo] = {}
    for entry in _field_value_entries(module):
        if _is_leaf_array(entry.value):
            field_metadata.setdefault(id(entry.value), FieldMetadataInfo(owner=entry.owner, field=entry.field))
    return field_metadata


def _is_leaf_array(leaf: Any) -> bool:  # noqa: ANN401
    return eqx.is_array(leaf) or isinstance(leaf, jax.ShapeDtypeStruct)


def _infer_quantization_mode(owner: eqx.Module) -> QuantizationMode | None:
    config = getattr(owner, "config", None)
    if config is None:
        return None

    weight_mode = getattr(config, "weight_quantization_mode", None)
    if isinstance(weight_mode, QuantizationMode):
        return weight_mode

    embedding_mode = getattr(config, "embedding_quantization_mode", None)
    if isinstance(embedding_mode, QuantizationMode):
        return embedding_mode

    return None


def _parameter_leaf_entries(module: eqx.Module) -> list[_ParameterLeafEntry]:
    canonical_by_leaf_id: dict[int, tuple[int, str]] = {}
    results: list[_ParameterLeafEntry] = []

    for entry in _field_value_entries(module):
        if not _is_leaf_array(entry.value):
            continue

        leaf = entry.value
        metadata = entry.field.metadata
        path = keystr(entry.path).lstrip(".")
        quantized = metadata.get("quantized", False)

        leaf_key = id(leaf)
        canonical_entry = canonical_by_leaf_id.get(leaf_key)
        if canonical_entry is None:
            canonical_index = len(canonical_by_leaf_id)
            canonical_by_leaf_id[leaf_key] = (canonical_index, path)
            alias_of = None
        else:
            canonical_index, alias_of = canonical_entry

        results.append(
            _ParameterLeafEntry(
                leaf=leaf,
                canonical_index=canonical_index,
                info=ParameterLeafInfo(
                    path=path,
                    owner_type=type(entry.owner),
                    field_name=entry.field.name,
                    shape=tuple(leaf.shape),
                    dtype=jnp.dtype(leaf.dtype),
                    trainable=metadata.get("trainable", True),
                    norm=metadata.get("norm", ParameterNorm.SPECTRAL),
                    quantized=quantized,
                    quantization_mode=_infer_quantization_mode(entry.owner) if quantized else None,
                    tensor_sharding=metadata.get("tensor_sharding"),
                    min_size_to_shard=metadata.get("min_size_to_shard", 0),
                    alias_of=alias_of,
                ),
            ),
        )

    return results


def iter_parameter_leaves(module: eqx.Module) -> list[ParameterLeafInfo]:
    return [entry.info for entry in _parameter_leaf_entries(module)]


def partition_parameter_leaves[M: eqx.Module, LeafT](
    module: M,
    predicate: Callable[[ParameterLeafInfo], bool],
    mapper: Callable[[Array | jax.ShapeDtypeStruct, ParameterLeafInfo], LeafT],
) -> M:
    entries = _parameter_leaf_entries(module)
    flat_leaves, treedef = jtu.tree_flatten(module, is_leaf=lambda leaf: leaf is None)
    if not entries:
        return jtu.tree_unflatten(treedef, [None] * len(flat_leaves))

    selected_by_canonical_index: list[LeafT | None] = [None] * (max(entry.canonical_index for entry in entries) + 1)
    for entry in entries:
        if entry.info.alias_of is not None or not predicate(entry.info):
            continue
        selected_by_canonical_index[entry.canonical_index] = mapper(entry.leaf, entry.info)

    entry_iterator = iter(entries)
    partitioned_leaves: list[object] = []
    for leaf in flat_leaves:
        if not _is_leaf_array(leaf):
            partitioned_leaves.append(None)
            continue
        entry = next(entry_iterator)
        partitioned_leaves.append(
            None if entry.info.alias_of is not None else selected_by_canonical_index[entry.canonical_index],
        )

    assert next(entry_iterator, None) is None
    return jtu.tree_unflatten(treedef, partitioned_leaves)


def combine_parameter_leaves[M: eqx.Module](
    module: M,
    replacements: M,
) -> M:
    entries = _parameter_leaf_entries(module)
    if not entries:
        return module

    flat_leaves, treedef = jtu.tree_flatten(module, is_leaf=lambda leaf: leaf is None)
    replacement_leaves, replacement_treedef = jtu.tree_flatten(replacements, is_leaf=lambda leaf: leaf is None)
    assert replacement_treedef == treedef

    replacement_by_canonical_index: list[object | None] = [None] * (
        max(entry.canonical_index for entry in entries) + 1
    )
    entry_iterator = iter(entries)
    for leaf, replacement in zip(flat_leaves, replacement_leaves, strict=True):
        if not _is_leaf_array(leaf):
            continue
        entry = next(entry_iterator)
        if replacement is None or entry.info.alias_of is not None:
            continue
        replacement_by_canonical_index[entry.canonical_index] = replacement

    assert next(entry_iterator, None) is None
    if all(replacement is None for replacement in replacement_by_canonical_index):
        return module

    entry_iterator = iter(entries)
    combined_leaves: list[object] = []
    for leaf in flat_leaves:
        if not _is_leaf_array(leaf):
            combined_leaves.append(leaf)
            continue
        entry = next(entry_iterator)
        replacement = replacement_by_canonical_index[entry.canonical_index]
        combined_leaves.append(leaf if replacement is None else replacement)

    assert next(entry_iterator, None) is None
    return jtu.tree_unflatten(treedef, combined_leaves)


def field(
    tensor_sharding: TensorSharding | None = None,
    min_size_to_shard: int = 2**18,
    trainable: bool = True,
    norm: ParameterNorm = ParameterNorm.SPECTRAL,
    quantized: bool = False,
    *,
    converter: Callable[[Any], Any] | None = None,
    static: bool = False,
    metadata: Mapping[str, Any] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    merged_metadata = dict(metadata or {})
    merged_metadata["trainable"] = trainable
    merged_metadata["norm"] = norm
    merged_metadata["quantized"] = quantized
    merged_metadata["tensor_sharding"] = tensor_sharding
    merged_metadata["min_size_to_shard"] = min_size_to_shard
    return eqx.field(
        converter=converter,
        static=static,
        metadata=merged_metadata,
        **kwargs,
    )


def shard_batch_axis(array: Array, sharding_config: ShardingConfig, *, batch_axis: int) -> Array:
    if array.ndim == 0:
        return array

    dp = sharding_config.data_axis_size
    batch_size = array.shape[batch_axis]
    padded_size = ((batch_size + dp - 1) // dp) * dp
    if padded_size != batch_size:
        pad_shape = list(array.shape)
        pad_shape[batch_axis] = padded_size - batch_size
        if jnp.issubdtype(array.dtype, jax.dtypes.prng_key):
            padding = jax.random.split(jax.random.key(0), math.prod(pad_shape)).reshape(pad_shape)
        else:
            padding = jnp.zeros(pad_shape, dtype=array.dtype)
        array = jnp.concat([array, padding], axis=batch_axis)

    parts: list[str | None] = [None] * array.ndim
    parts[batch_axis] = sharding_config.data_axis_name
    pspec = shd.PartitionSpec(*parts)
    return jax.lax.with_sharding_constraint(array, sharding_config.make_sharding(pspec))


def pad_and_apply_data_sharding[T](value: T, *, sharding_config: ShardingConfig | None, batch_axis: int) -> T:
    if sharding_config is None:
        return value
    return jax.tree.map(
        lambda leaf: shard_batch_axis(leaf, sharding_config, batch_axis=batch_axis) if eqx.is_array(leaf) else leaf,
        value,
    )


@dataclass
class DummyUnionMember:
    def __getattribute__(self, name: str, /) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        raise NotImplementedError
