import dataclasses
import json
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar, cast

if TYPE_CHECKING:
    from lalamo.modules.common import Initializer, ShardingConfig

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Key

from lalamo.common import ParameterPath
from lalamo.registry_abc import RegistryABC
from lalamo.serialization import UzuSerializable

from lalamo.common import ParameterTree


class GradientEstimator(Enum):
    NONE = "none"
    DETERMINISTIC = "deterministic"
    STOCHASTIC_DROPOUT = "stochastic_dropout"


@dataclass(frozen=True)
class ArrayForwardPassConfig:
    gradient_estimator: GradientEstimator = GradientEstimator.NONE


@dataclass(frozen=True)
class CompressedArraySpec:
    @abstractmethod
    def compress(self, weights: Float[Array, "... out_channels in_channels"]) -> "CompressedArray": ...

    @abstractmethod
    def init(
        self,
        initializer: "Initializer",
        leading_dims: tuple[int, ...],
        out_channels: int,
        in_channels: int,
    ) -> "CompressedArray": ...

    @abstractmethod
    def from_uzu(self, data: Mapping[str, Any], prefix: ParameterPath) -> "CompressedArray": ...


def serialize_spec(spec: CompressedArraySpec) -> str:
    spec_dict: dict[str, Any] = {"__class__": type(spec).__name__}
    for f in dataclasses.fields(spec):
        value = getattr(spec, f.name)
        spec_dict[f.name] = str(jnp.dtype(value)) if f.name == "dtype" else value
    return json.dumps(spec_dict)


def deserialize_spec(spec_json: str) -> CompressedArraySpec:
    spec_dict = json.loads(spec_json)
    class_name = spec_dict.pop("__class__")
    spec_types = {t.__name__: t for t in CompressedArraySpec.__subclasses__()}
    if "dtype" in spec_dict:
        spec_dict["dtype"] = jnp.dtype(spec_dict["dtype"])
    return spec_types[class_name](**spec_dict)


CompressedArraySpecT_co = TypeVar("CompressedArraySpecT_co", bound=CompressedArraySpec | None, covariant=True)


class CompressedArray(UzuSerializable, RegistryABC, eqx.Module, Generic[CompressedArraySpecT_co]):  # noqa: UP046
    spec: CompressedArraySpecT_co = eqx.field(static=True, default=None, kw_only=True)

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]: ...

    @property
    @abstractmethod
    def dtype(self) -> DTypeLike: ...

    def dequantize(
        self, quantized_weights: Float[Array, "... out_channels in_channels"]
    ) -> Float[Array, "... out_channels in_channels"]:
        raise NotImplementedError(f"{type(self).__name__} does not support dequantize")

    @abstractmethod
    def materialize(self) -> Float[Array, "... out_channels in_channels"]: ...

    @abstractmethod
    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        key: Key[Array, ""] | None,
        forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig(),  # noqa: B008
    ) -> Float[Array, "... out_channels"]: ...

    def to_uzu(self) -> dict[str, Any]:
        result: dict[str, Any] = {"__class__": type(self).__name__}
        if self.spec is not None:
            result["__spec__"] = serialize_spec(self.spec)
        result.update(super().to_uzu())
        return result

    def from_uzu(
        self,
        data: Mapping[str, Any],
        prefix: ParameterPath = ParameterPath(),  # noqa: B008
        sharding_config: "ShardingConfig | None" = None,
    ) -> Self:
        spec_key = prefix / "__spec__"
        if spec_key not in data:
            return super().from_uzu(data, prefix=prefix, sharding_config=sharding_config)
        return cast("Self", deserialize_spec(data[spec_key]).from_uzu(data, prefix))
