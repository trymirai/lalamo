from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float, PRNGKeyArray

from lalamo.serialization import Serializable

from lalamo.common import ParameterTree


class GradientEstimator(Enum):
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"


@dataclass(frozen=True)
class ArrayForwardPassConfig:
    gradient_estimator: GradientEstimator = GradientEstimator.DETERMINISTIC


class CompressedArray(Serializable, eqx.Module):
    _registry: ClassVar[dict[str, type["CompressedArray"]]] = {}
    kind: ClassVar[str]

    def __init_subclass__(cls, kind: str, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init_subclass__(**kwargs)
        CompressedArray._registry[kind] = cls
        cls.kind = kind

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]: ...

    @property
    @abstractmethod
    def dtype(self) -> DTypeLike: ...

    @abstractmethod
    def materialize(self) -> Float[Array, "... out_channels in_channels"]: ...

    @abstractmethod
    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        key: PRNGKeyArray | None,
        forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig(),  # noqa: B008
    ) -> Float[Array, "... out_channels"]: ...

    def to_uzu(self) -> dict[str, Any]:
        return {"__kind__": self.kind, **super().to_uzu()}

    @classmethod
    def from_uzu(cls, data: Mapping[str, Any]) -> "CompressedArray":
        kind = data["__kind__"]
        if not isinstance(kind, str):
            raise TypeError(f"Expected string kind, got {type(kind)}")
        return cls._registry[kind].from_uzu(data)

    @abc.abstractmethod
    def export_weights(self) -> ParameterTree: ...


def pack_uint_to_uint8(unpacked: Array, bits: int) -> Array:
    if bits == 8:
        return unpacked.astype(jnp.uint8)

    if bits not in {1, 4}:
        raise ValueError(f"Unsupported uint packing width: {bits}")

    if unpacked.ndim == 0:
        raise ValueError("Input array cannot be scalar")

    values_per_byte = 8 // bits
    *_, last_dim = unpacked.shape
    if last_dim % values_per_byte != 0:
        raise ValueError(
            f"Last dimension {last_dim} must be divisible by {values_per_byte} for {bits}-bit packing",
        )

    grouped = rearrange(
        unpacked.astype(jnp.uint8),
        "... (groups packed_values) -> ... groups packed_values",
        packed_values=values_per_byte,
    )
    packed = jnp.zeros(grouped.shape[:-1], dtype=jnp.uint8)
    for shift in range(values_per_byte):
        packed = packed | (grouped[..., shift] << jnp.uint8(shift * bits))
    return packed


def unpack_uint8_to_uint(packed: Array, bits: int) -> Array:
    if bits == 8:
        return packed.astype(jnp.uint8)

    if bits not in {1, 4}:
        raise ValueError(f"Unsupported uint unpacking width: {bits}")

    values_per_byte = 8 // bits
    shifts = jnp.arange(values_per_byte, dtype=jnp.uint8) * jnp.uint8(bits)
    mask = jnp.uint8((1 << bits) - 1)
    unpacked = jnp.bitwise_and(jnp.right_shift(packed[..., None], shifts), mask)
    return rearrange(unpacked, "... groups packed_values -> ... (groups packed_values)")
