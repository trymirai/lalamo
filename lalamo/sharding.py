import contextlib
import contextvars
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.sharding as shd
from jaxtyping import Array

from lalamo.field import FieldMetadata


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


def apply_parameter_sharding(
    array: Array,
    info: FieldMetadata,
    sharding_config: ShardingConfig | None,
) -> Array:
    if sharding_config is None or info.tensor_sharding is None or array.size < info.min_size_to_shard:
        return array

    parts: list[str | None] = [None] * array.ndim

    tp_dim: int | None = None
    tp_axis_size = sharding_config.tensor_axis_size
    for dim in info.tensor_sharding.dims_to_try(info.sharding_order):
        if dim < array.ndim and array.shape[dim] % tp_axis_size == 0:
            parts[dim] = sharding_config.tensor_axis_name
            tp_dim = dim
            break

    if sharding_config.fsdp:
        dp_axis_size = sharding_config.data_axis_size
        for dim in info.tensor_sharding.axes:
            if dim != tp_dim and dim < array.ndim and array.shape[dim] % dp_axis_size == 0:
                parts[dim] = sharding_config.data_axis_name
                break

    pspec = shd.PartitionSpec(*parts)
    return jax.device_put(array, sharding_config.make_sharding(pspec))


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
