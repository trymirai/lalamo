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
    "KeychainBroadcastMode",
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


class KeychainBroadcastMode(StrEnum):
    AUTO = "auto"
    PREFIX = "prefix"
    SUFFIX = "suffix"


@supports_dummy_arrays()
def _cast_dtype(
    value: "Array | WeightMatrix | None",
    dtype: DTypeLike,
) -> "Array | WeightMatrix | None":
    if value is None:
        return None
    return value.astype(dtype)


def _broadcast_prefix_and_suffix(
    current_shape: tuple[int, ...], target_shape: tuple[int, ...], mode: KeychainBroadcastMode
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    broadcast_shape = target_shape
    if not current_shape and mode == KeychainBroadcastMode.AUTO:
        return target_shape, ()

    prefix, suffix = (), ()
    if mode == KeychainBroadcastMode.AUTO:
        matches: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for window_start in range(len(broadcast_shape) - len(current_shape) + 1):
            window_end = window_start + len(current_shape)
            window = broadcast_shape[window_start:window_end]
            if window != current_shape:
                continue

            matches.append((broadcast_shape[:window_start], broadcast_shape[window_end:]))

        if not matches:
            raise ValueError(f"Cannot broadcast vmapped_keys from shape {current_shape} to shape {target_shape}.")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous broadcast from vmapped_keys shape {current_shape} to shape {target_shape}.")
        prefix, suffix = matches[0]

    elif mode == KeychainBroadcastMode.PREFIX:
        window_size = len(current_shape)
        prefix = broadcast_shape[: len(broadcast_shape) - window_size]

        if broadcast_shape[len(broadcast_shape) - window_size :] != current_shape:
            raise ValueError(
                f"Expected target shape {target_shape} to end with existing vmapped_keys shape {current_shape}.",
            )

    elif mode == KeychainBroadcastMode.SUFFIX:
        window_size = len(current_shape)
        suffix = broadcast_shape[window_size:]

        if broadcast_shape[:window_size] != current_shape:
            raise ValueError(
                f"Expected target shape {target_shape} to end with existing vmapped_keys shape {current_shape}.",
            )

    return prefix, suffix


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
        mode: KeychainBroadcastMode = KeychainBroadcastMode.AUTO,
        sharding_axes: tuple[ShardingAxis | None, ...] | None = None,
    ) -> Self:
        prefix_shape, suffix_shape = _broadcast_prefix_and_suffix(self.vmapped_keys.shape, shape, mode)
        combined_shape = prefix_shape + suffix_shape

        if combined_shape:
            flat_keys = jnp.reshape(self.vmapped_keys, (-1,))
            split_keys = jax.vmap(partial(jax.random.split, num=combined_shape))(flat_keys)
            unflattened_keys = jnp.reshape(split_keys, self.vmapped_keys.shape + prefix_shape + suffix_shape)

            original_indices = tuple(range(self.vmapped_keys.ndim))
            prefix_indices = tuple(range(self.vmapped_keys.ndim, self.vmapped_keys.ndim + len(prefix_shape)))
            suffix_indices = tuple(range(self.vmapped_keys.ndim + len(prefix_shape), unflattened_keys.ndim))
            target_indices = tuple(range(len(unflattened_keys.shape)))

            vmapped_keys = jnp.moveaxis(
                unflattened_keys,
                prefix_indices + original_indices + suffix_indices,
                target_indices,
            )
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
        vmapped_keys = self.broadcast(
            (num, *self.vmapped_keys.shape),
            mode=KeychainBroadcastMode.PREFIX,
        ).vmapped_keys
        batch_keys = jax.random.split(self.batch_key, num)
        return tuple(
            type(self)(vmapped_keys=split_vmapped_keys, batch_key=batch_key)
            for split_vmapped_keys, batch_key in zip(vmapped_keys, batch_keys, strict=True)
        )

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> Self:
        return type(self)(vmapped_keys=jnp.squeeze(self.vmapped_keys, axis=axis), batch_key=self.batch_key)

    def rolling_broadcast(
        self,
        shape: tuple[int, ...],
        *,
        mode: KeychainBroadcastMode = KeychainBroadcastMode.AUTO,
    ) -> Self:
        def split_once(carry: Key[Array, ""], _: None) -> tuple[Key[Array, ""], Key[Array, ""]]:
            next_carry, sample_key = jax.random.split(carry)
            return next_carry, sample_key

        def rolling_broadcast_axis(
            vmapped_keys: Key[Array, "..."],
            axis_size: int,
            axis_index: int,
        ) -> Key[Array, "..."]:
            def rolling_broadcast_scalar_key(vmapped_key: Key[Array, ""]) -> Key[Array, " axis"]:
                _, keys = jax.lax.scan(split_once, vmapped_key, xs=None, length=axis_size)
                return keys

            flat_keys = jnp.reshape(vmapped_keys, (-1,))
            split_keys = jax.vmap(rolling_broadcast_scalar_key)(flat_keys)
            unflattened_keys = jnp.reshape(split_keys, (*vmapped_keys.shape, axis_size))

            appended_axis_index = vmapped_keys.ndim
            return jnp.moveaxis(unflattened_keys, appended_axis_index, axis_index)

        prefix_shape, suffix_shape = _broadcast_prefix_and_suffix(self.vmapped_keys.shape, shape, mode)

        vmapped_keys = self.vmapped_keys
        for suffix_axis_size in reversed(suffix_shape):
            vmapped_keys = rolling_broadcast_axis(vmapped_keys, suffix_axis_size, self.vmapped_keys.ndim)
        for prefix_axis_size in reversed(prefix_shape):
            vmapped_keys = rolling_broadcast_axis(vmapped_keys, prefix_axis_size, 0)

        return type(self)(
            vmapped_keys=vmapped_keys,
            batch_key=self.batch_key,
        )


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
    # Pyrefly currently mishandles wrappers around `eqx.field(...)` when they are
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

        return map_nodes_of_type(WeightMatrix, lambda value: value.to_full_precision(), self)

    def switch_implementation(self, implementation: "CompressionImplementation") -> Self:
        from lalamo.utils.surgery import map_nodes_of_type  # noqa: PLC0415
        from lalamo.weight_matrix import WeightMatrix  # noqa: PLC0415

        return map_nodes_of_type(WeightMatrix, lambda value: value.switch_implementation(implementation), self)
