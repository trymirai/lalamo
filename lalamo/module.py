from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from math import prod
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Self, TypeVar, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from cattrs import GenConverter
from jax.sharding import reshard
from jaxtyping import Array, DTypeLike, Key

from lalamo.utils.dummy_array import supports_dummy_arrays
from lalamo.utils.json import JSON
from lalamo.utils.registry_abc import make_registry_abc_converter

from .exportable import Exportable

if TYPE_CHECKING:
    # Import only for static analysis to avoid a runtime cycle: weight_matrix imports this module.
    from lalamo.weight_matrix import CompressionImplementation, WeightMatrix

__all__ = [
    "ForwardPassMode",
    "Keychain",
    "LalamoConfig",
    "LalamoModule",
    "ShardingAxis",
    "field",
]


class ForwardPassMode(StrEnum):
    MULTI_TOKEN = "multi_token"
    SINGLE_TOKEN = "single_token"


class ShardingAxis(StrEnum):
    DATA = "data"
    TENSOR = "tensor"
    EXPERT = "expert"


@supports_dummy_arrays()
def _cast_dtype(
    value: "Array | WeightMatrix | None",
    dtype: DTypeLike,
) -> "Array | WeightMatrix | None":
    if value is None:
        return None
    return value.astype(dtype)


class Keychain(eqx.Module):
    vmapped_keys: Key[Array, "..."]
    batch_key: Key[Array, ""]

    @classmethod
    def init(cls, seed: int, shape: tuple[int, ...] = ()) -> Self:
        vmapped_keys, batch_key = jax.random.split(jax.random.key(seed))
        if shape:
            vmapped_keys = jnp.reshape(jax.random.split(vmapped_keys, prod(shape)), shape)
        return cls(vmapped_keys=vmapped_keys, batch_key=batch_key)

    def broadcast(
        self,
        shape: tuple[int, ...],
        *,
        sharding_axes: tuple[ShardingAxis | None, ...] | None = None,
    ) -> Self:
        leading_shape = _leading_broadcast_shape(self.vmapped_keys.shape, shape)
        if leading_shape:
            flat_keys = jnp.reshape(self.vmapped_keys, (-1,))
            split_keys = jax.vmap(partial(jax.random.split, num=prod(leading_shape)))(flat_keys)
            vmapped_keys = jnp.reshape(jnp.swapaxes(split_keys, 0, 1), shape)
        else:
            vmapped_keys = self.vmapped_keys

        if sharding_axes is not None and not all(axis is None for axis in sharding_axes):
            # Local import avoids a module <-> sharding cycle: sharding helpers are parameterized by ShardingAxis.
            from lalamo.utils.sharding import make_sharding  # noqa: PLC0415

            sharding = make_sharding(sharding_axes)
            if sharding is not None:
                vmapped_keys = reshard(vmapped_keys, sharding)

        return type(self)(vmapped_keys=vmapped_keys, batch_key=self.batch_key)

    def split(self, num: int = 2) -> tuple[Self, ...]:
        vmapped_keys = self.broadcast((num, *self.vmapped_keys.shape)).vmapped_keys
        batch_keys = jax.random.split(self.batch_key, num)
        return tuple(
            type(self)(vmapped_keys=split_vmapped_keys, batch_key=batch_key)
            for split_vmapped_keys, batch_key in zip(vmapped_keys, batch_keys, strict=True)
        )

    def rolling_broadcast(self, shape: tuple[int, ...]) -> Self:
        leading_shape = _leading_broadcast_shape(self.vmapped_keys.shape, shape)
        if not leading_shape:
            return self

        def split_once(carry: Key[Array, ""], _: None) -> tuple[Key[Array, ""], Key[Array, ""]]:
            next_carry, sample_key = jax.random.split(carry)
            return next_carry, sample_key

        def rolling_broadcast_scalar_key(vmapped_keys: Key[Array, ""]) -> Key[Array, " num"]:
            _, keys = jax.lax.scan(split_once, vmapped_keys, xs=None, length=prod(leading_shape))
            return keys

        flat_keys = jnp.reshape(self.vmapped_keys, (-1,))
        split_keys = jax.vmap(rolling_broadcast_scalar_key)(flat_keys)
        vmapped_keys = jnp.reshape(jnp.swapaxes(split_keys, 0, 1), shape)

        return type(self)(
            vmapped_keys=vmapped_keys,
            batch_key=self.batch_key,
        )


def _leading_broadcast_shape(current_shape: tuple[int, ...], target_shape: tuple[int, ...]) -> tuple[int, ...]:
    broadcast_shape = jnp.broadcast_shapes(current_shape, target_shape)
    if broadcast_shape != target_shape:
        raise ValueError(f"Cannot broadcast vmapped_keys from shape {current_shape} to shape {target_shape}.")
    if current_shape and target_shape[-len(current_shape) :] != current_shape:
        raise ValueError(
            f"Expected target shape {target_shape} to end with existing vmapped_keys shape {current_shape}.",
        )
    return target_shape[: len(target_shape) - len(current_shape)]


@dataclass(frozen=True)
class LalamoConfig:
    _converter: ClassVar[GenConverter] = make_registry_abc_converter()

    @classmethod
    def from_json(cls, json_object: JSON) -> Self:
        return cls._converter.structure(json_object, cls)

    def to_json(self) -> JSON:
        return self._converter.unstructure(self)


ConfigT_co = TypeVar("ConfigT_co", bound=LalamoConfig, covariant=True)

type PytreeKey = (
    str | int | jtu.GetAttrKey | jtu.SequenceKey | jtu.DictKey | jtu.FlattenedIndexKey | tuple[PytreeKey, ...]
)


class ParameterNorm(StrEnum):
    L_INF = "l_inf"
    L_2 = "l_2"
    SPECTRAL = "spectral"


@dataclass(frozen=True)
class FieldMetadata:
    trainable: bool = True
    norm: ParameterNorm = ParameterNorm.L_INF


def _field_metadata(
    *,
    trainable: bool | None,
    static: bool,
    norm: ParameterNorm,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if trainable is None:
        trainable = not static
    if static and trainable:
        raise ValueError("A field cannot be both static and trainable.")
    return {"trainable": trainable, "norm": norm, **(metadata or {})}


if TYPE_CHECKING:
    # Pyrefly currently mis-handles wrappers around `eqx.field(...)` when they are
    # inherited through an `eqx.Module` hierarchy. In practice that means a field
    # like `config: ConfigT = field(static=True)` is treated as if it were an
    # ordinary default-valued class attribute rather than a dataclass field, and
    # subclasses then fail with false positives such as:
    #
    #   "Dataclass field x without a default may not follow dataclass field with a default"
    #
    # The failure does not come from our runtime behavior. `eqx.Module` already
    # provides the dataclass transform via its metaclass, and the exact same class
    # hierarchy type-checks correctly when the field helper is written as
    # `eqx.field(...)` directly. Adding another `@dataclass_transform(...)` on
    # `LalamoModule`, or even putting one on a custom metaclass derived from the
    # Equinox metaclass, does not help: pyrefly still fails to recognize the
    # wrapper once the class goes through the `eqx.Module` path. The bug is
    # specifically in the checker's handling of a wrapper around `eqx.field`, even
    # if that wrapper is otherwise a faithful field specifier.
    #
    # So for static analysis we expose `lalamo.field` as a plain alias to
    # `eqx.field`, which lets pyrefly recognize inherited Lalamo fields correctly.
    # At runtime we still use the Lalamo wrapper below, because we need to
    # translate the Lalamo-only `trainable` and `norm` arguments into Equinox
    # metadata before calling `eqx.field(...)`.
    field = eqx.field
else:

    def field(
        *,
        trainable: bool | None = None,
        static: bool = False,
        norm: ParameterNorm = ParameterNorm.L_INF,
        metadata: Mapping[str, Any] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        return eqx.field(
            metadata=_field_metadata(trainable=trainable, static=static, norm=norm, metadata=metadata),
            static=static,
            **kwargs,
        )


class LalamoModule(Exportable, eqx.Module, Generic[ConfigT_co]):  # noqa: UP046
    config: ConfigT_co = field(static=True)

    def astype(self, dtype: DTypeLike) -> Self:
        from lalamo.weight_matrix import WeightMatrix  # noqa: PLC0415

        result = jtu.tree_map(
            lambda value: _cast_dtype(value, dtype),
            self,
            is_leaf=lambda x: isinstance(x, WeightMatrix),
        )
        return cast("Self", result)

    def to_full_precision(self) -> Self:
        from lalamo.utils.surgery import map_nodes_of_type  # noqa: PLC0415
        from lalamo.weight_matrix import WeightMatrix  # noqa: PLC0415

        result = map_nodes_of_type(WeightMatrix, lambda value: value.to_full_precision(), self)
        return cast("Self", result)

    def switch_implementation(self, implementation: "CompressionImplementation") -> Self:
        from lalamo.utils.surgery import map_nodes_of_type  # noqa: PLC0415
        from lalamo.weight_matrix import WeightMatrix  # noqa: PLC0415

        result = map_nodes_of_type(WeightMatrix, lambda value: value.switch_implementation(implementation), self)
        return cast("Self", result)
