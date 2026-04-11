import dataclasses
import json
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Self, TypeVar, cast

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Key

from lalamo.common import ParameterPath
from lalamo.module import Initializer
from lalamo.utils.registry_abc import RegistryABC

from lalamo.common import ParameterTree


class GradientEstimator(Enum):
    NONE = "none"
    DETERMINISTIC = "deterministic"
    STOCHASTIC_DROPOUT = "stochastic_dropout"


@dataclass(frozen=True)
class ArrayForwardPassConfig:
    gradient_estimator: GradientEstimator = GradientEstimator.NONE


@dataclass(frozen=True)
class ArraySpec:
    def to_json(self) -> str:
        spec_dict: dict[str, Any] = {"__class__": type(self).__name__}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            spec_dict[f.name] = str(jnp.dtype(value)) if f.name == "float_dtype" else value
        return json.dumps(spec_dict)

    @classmethod
    def from_json(cls, spec_json: str) -> Self:
        spec_dict = json.loads(spec_json)
        class_name = spec_dict.pop("__class__")
        registry: dict[str, type[ArraySpec]] = {}
        queue = list(cls.__subclasses__())
        while queue:
            sub = queue.pop()
            queue.extend(sub.__subclasses__())
            registry[sub.__name__] = sub
        if "dtype" in spec_dict:
            spec_dict["dtype"] = jnp.dtype(spec_dict["dtype"])
        return cast("Self", registry[class_name](**spec_dict))


@dataclass(frozen=True)
class CompressedArraySpec(ArraySpec):
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
            result["__spec__"] = self.spec.to_json()
        result.update(super().to_uzu())
        return result

    def from_uzu(
        self,
        data: Mapping[str, Any],
        prefix: ParameterPath = ParameterPath(),  # noqa: B008
    ) -> Self:
        spec_key = prefix / "__spec__"
        spec = CompressedArraySpec.from_json(data[spec_key]) if spec_key in data else self.spec
        if spec is not None:
            return cast("Self", spec.from_uzu(data, prefix))
        return super().from_uzu(data, prefix=prefix)
