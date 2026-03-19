import contextlib
import contextvars
import dataclasses
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
    "apply_data_sharding",
    "combine_parameter_leaves",
    "config_converter",
    "field",
    "find_field_metadata",
    "get_current_sharding_config",
    "iter_parameter_leaves",
    "pad_and_apply_data_sharding",
    "partition_parameter_leaves",
    "register_config_union",
    "require_array",
    "require_tree",
    "sharded_field",
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

    def unstructure(obj: object) -> dict | None:
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

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.field.metadata


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
    tensor_sharding: TensorSharding | None
    min_size_to_shard: int
    alias_of: str | None


@dataclass(frozen=True)
class _ParameterLeafEntry:
    path: tuple[Any, ...]
    canonical_path: tuple[Any, ...]
    leaf: Array | jax.ShapeDtypeStruct
    info: ParameterLeafInfo


def _field_metadata_from_path(module: eqx.Module, path: tuple[Any, ...]) -> FieldMetadataInfo | None:
    cur: object = module
    owner = module
    owner_field: dataclasses.Field | None = None

    for key in path:
        if isinstance(key, jtu.GetAttrKey):
            assert isinstance(cur, eqx.Module)
            owner = cur
            owner_field = next((field for field in dataclasses.fields(cur) if field.name == key.name), None)
            cur = getattr(cur, key.name)
        elif isinstance(key, jtu.SequenceKey):
            cur = cur[key.idx]  # type: ignore[index]
        elif isinstance(key, jtu.DictKey):
            cur = cur[key.key]  # type: ignore[index]
        else:
            raise TypeError(f"Unexpected key type: at {path}, key {key}")

    if owner_field is None:
        return None

    return FieldMetadataInfo(owner=owner, field=owner_field)


def find_field_metadata(module: eqx.Module, target: object) -> FieldMetadataInfo | None:
    flat_with_path, _ = jtu.tree_flatten_with_path(module, is_leaf=lambda value: value is target)

    for path, leaf in flat_with_path:
        if leaf is not target:
            continue
        field_info = _field_metadata_from_path(module, path)
        if field_info is None:
            raise ValueError(f"Field lookup failed for module {module} at {path}")
        return field_info

    return None


def _is_array_like(leaf: object) -> bool:
    return eqx.is_array(leaf) or isinstance(leaf, jax.ShapeDtypeStruct)


def _parameter_leaf_entries(module: eqx.Module) -> list[_ParameterLeafEntry]:
    flat_with_path, _ = jtu.tree_flatten_with_path(module)
    first_paths: dict[int, tuple[Any, ...]] = {}
    results: list[_ParameterLeafEntry] = []

    for path, leaf in flat_with_path:
        if not _is_array_like(leaf):
            continue

        field_info = _field_metadata_from_path(module, path)
        if field_info is None:
            raise ValueError(f"Field lookup failed for module {module} at {path}")

        leaf_key = id(leaf)
        canonical_path = first_paths.get(leaf_key)
        if canonical_path is None:
            canonical_path = path
            first_paths[leaf_key] = path
            alias_of = None
        else:
            alias_of = keystr(canonical_path).lstrip(".")

        results.append(
            _ParameterLeafEntry(
                path=path,
                canonical_path=canonical_path,
                leaf=leaf,
                info=ParameterLeafInfo(
                    path=keystr(path).lstrip("."),
                    owner_type=type(field_info.owner),
                    field_name=field_info.field.name,
                    shape=tuple(leaf.shape),
                    dtype=jnp.dtype(leaf.dtype),
                    trainable=field_info.metadata.get("trainable", True),
                    norm=field_info.metadata.get("norm", ParameterNorm.SPECTRAL),
                    quantized=field_info.metadata.get("quantized", False),
                    tensor_sharding=field_info.metadata.get("tensor_sharding"),
                    min_size_to_shard=field_info.metadata.get("min_size_to_shard", 0),
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
    if not entries:
        return jax.tree.map(lambda _leaf: None, module)

    entry_by_path = {entry.path: entry for entry in entries}
    selected_by_canonical_path = {
        entry.canonical_path: mapper(entry.leaf, entry.info)
        for entry in entries
        if entry.info.alias_of is None and predicate(entry.info)
    }

    def maybe_select(path: tuple[Any, ...], _leaf: object) -> object:
        entry = entry_by_path.get(path)
        if entry is None or entry.info.alias_of is not None:
            return None
        return selected_by_canonical_path.get(entry.canonical_path)

    return jax.tree.map_with_path(maybe_select, module)


def combine_parameter_leaves[M: eqx.Module](
    module: M,
    replacements: M,
) -> M:
    entries = _parameter_leaf_entries(module)
    if not entries:
        return module

    entry_by_path = {entry.path: entry for entry in entries}
    replacement_by_path = dict(
        jtu.tree_flatten_with_path(
            replacements,
            is_leaf=lambda leaf: leaf is None,
        )[0]
    )
    replacement_by_canonical_path: dict[tuple[Any, ...], object] = {}
    for entry in entries:
        replacement = replacement_by_path[entry.path]
        if replacement is None or entry.info.alias_of is not None:
            continue
        replacement_by_canonical_path[entry.canonical_path] = replacement
    if not replacement_by_canonical_path:
        return module

    def maybe_replace(path: tuple[Any, ...], leaf: object) -> object:
        entry = entry_by_path.get(path)
        if entry is None:
            return leaf
        replacement = replacement_by_canonical_path.get(entry.canonical_path)
        return leaf if replacement is None else replacement

    return jax.tree.map_with_path(maybe_replace, module)


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


def sharded_field(
    tensor_sharding: TensorSharding | None = None,
    min_size_to_shard: int = 2**18,
    *,
    converter: Callable[[Any], Any] | None = None,
    static: bool = False,
    metadata: Mapping[str, Any] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    return field(
        tensor_sharding=tensor_sharding,
        min_size_to_shard=min_size_to_shard,
        converter=converter,
        static=static,
        metadata=metadata,
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
            padding = jax.random.split(jax.random.key(0), pad_shape[batch_axis])
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


def apply_data_sharding[T](value: T, *, sharding_config: ShardingConfig | None, batch_axis: int) -> T:
    return pad_and_apply_data_sharding(value, sharding_config=sharding_config, batch_axis=batch_axis)


@dataclass
class DummyUnionMember:
    def __getattribute__(self, name: str, /) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        raise NotImplementedError
