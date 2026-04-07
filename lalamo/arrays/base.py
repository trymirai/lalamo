from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

import equinox as eqx
from jaxtyping import Array, DTypeLike, Float, PRNGKeyArray

from lalamo.serialization import Serializable

from lalamo.common import ParameterTree


class GradientEstimator(Enum):
    NONE = "none"
    DETERMINISTIC = "deterministic"
    STOCHASTIC_DROPOUT = "stochastic_dropout"


@dataclass(frozen=True)
class ArrayForwardPassConfig:
    gradient_estimator: GradientEstimator = GradientEstimator.NONE


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
    def dequantize(
        self, quantized_weights: Float[Array, "... out_channels in_channels"]
    ) -> Float[Array, "... out_channels in_channels"]: ...

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
