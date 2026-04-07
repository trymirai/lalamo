from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Self

if TYPE_CHECKING:
    from lalamo.modules.common import ShardingConfig

import equinox as eqx
from jaxtyping import Array, DTypeLike, Float, Key

from lalamo.serialization import UzuSerializable

from lalamo.common import ParameterTree


class GradientEstimator(Enum):
    NONE = "none"
    DETERMINISTIC = "deterministic"
    STOCHASTIC_DROPOUT = "stochastic_dropout"


@dataclass(frozen=True)
class ArrayForwardPassConfig:
    gradient_estimator: GradientEstimator = GradientEstimator.NONE


class CompressedArray(UzuSerializable, eqx.Module):
    _registry: ClassVar[dict[str, type["CompressedArray"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init_subclass__(**kwargs)
        CompressedArray._registry[cls.__name__] = cls

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
        return {"__class__": type(self).__name__, **super().to_uzu()}

    def from_uzu(
        self,
        data: Mapping[str, Any],
        prefix: str = "",
        sharding_config: "ShardingConfig | None" = None,
    ) -> Self:
        class_key = f"{prefix}.__class__" if prefix else "__class__"
        stored_class_name = data.get(class_key)
        if isinstance(stored_class_name, str) and stored_class_name != type(self).__name__:
            target_cls = CompressedArray._registry[stored_class_name]
            placeholder = target_cls.__new__(target_cls)
            return placeholder.from_uzu(data, prefix=prefix, sharding_config=sharding_config)  # type: ignore[return-value]
        return super().from_uzu(data, prefix=prefix, sharding_config=sharding_config)  # type: ignore[invalid-return-type]
