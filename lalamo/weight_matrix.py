import math
from abc import abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Generic, Self, TypeVar, dataclass_transform

import equinox as eqx
from cattrs import GenConverter
from jax import default_matmul_precision
from jax.lax import Precision
from jaxtyping import Array, DTypeLike, Float, Int, Key

from lalamo.initializer import Initializer
from lalamo.module import ParameterNorm, ShardingAxis, field
from lalamo.utils.json import JSON
from lalamo.utils.registry_abc import RegistryABC, make_registry_abc_converter

__all__ = [
    "GradientEstimator",
    "MatmulConfig",
    "WeightMatrix",
    "WeightMatrixSpec",
]


class GradientEstimator(StrEnum):
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"


@dataclass(frozen=True)
class MatmulConfig:
    gradient_estimator: GradientEstimator = GradientEstimator.DETERMINISTIC
    precision: Precision = Precision.DEFAULT


@dataclass(frozen=True)
class WeightMatrixSpec(RegistryABC):
    _converter: ClassVar[GenConverter] = make_registry_abc_converter()

    @classmethod
    def from_json(cls, json_object: JSON) -> Self:
        return cls._converter.structure(json_object, cls)

    def to_json(self) -> JSON:
        return self._converter.unstructure(self)

    def compress(self, weights: Float[Array, "*components out_channels in_channels"]) -> "WeightMatrix": ...

    @abstractmethod
    def init(
        self,
        initializer: Initializer,
        leading_dims: tuple[int, ...],
        output_dim: int,
        input_dim: int,
    ) -> "WeightMatrix": ...


WeightMatrixSpecT_co = TypeVar("WeightMatrixSpecT_co", bound=WeightMatrixSpec, covariant=True)


@dataclass_transform(field_specifiers=(eqx.field, field))
class WeightMatrix(RegistryABC, eqx.Module, Generic[WeightMatrixSpecT_co]):  # noqa: UP046
    spec: WeightMatrixSpecT_co = field(static=True)

    def _raise_if_batched(self) -> None:
        if self.ndim != 3:
            raise ValueError(
                "Attempted to call a method directly on a batched version of WeightMatrix. Use vmap instead.",
            )

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    @abstractmethod
    def dtype(self) -> DTypeLike: ...

    @abstractmethod
    def decompress(self) -> Float[Array, "... out_channels in_channels"]: ...

    @abstractmethod
    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        key: Key[Array, ""] | None,
        forward_pass_config: MatmulConfig | None = None,
    ) -> Float[Array, "... out_channels"]: ...


class EmbeddingMatrix(WeightMatrix[WeightMatrixSpecT_co]):
    @abstractmethod
    def lookup_embedding(
        self,
        index: int | Int[Array, ""],
        *,
        key: Key[Array, ""] | None,
        forward_pass_config: MatmulConfig | None = None,
    ) -> Float[Array, "... out_channels"]: ...


class Layout(StrEnum):
    OUTPUT_INPUT = "output_input"
    INPUT_OUTPUT = "input_output"


@dataclass(frozen=True)
class FullPrecisionSpec(WeightMatrixSpec):
    layout: Layout = Layout.OUTPUT_INPUT

    def compress(self, weights: Float[Array, "... out_channels in_channels"]) -> "FullPrecisionMatrix":
        if self.layout == Layout.INPUT_OUTPUT:
            weights = weights.swapaxes(-1, -2)
        return FullPrecisionMatrix(spec=self, weights=weights)

    def init(
        self,
        initializer: "Initializer",
        leading_dims: tuple[int, ...],
        output_dim: int,
        input_dim: int,
    ) -> "FullPrecisionMatrix":
        std = 1 / math.sqrt(input_dim)
        partition: tuple[ShardingAxis | None, ...]
        if self.layout == Layout.INPUT_OUTPUT:
            partition = (None, ShardingAxis.TENSOR)
            shape = (*leading_dims, input_dim, output_dim)
        else:
            partition = (ShardingAxis.TENSOR, None)
            shape = (*leading_dims, output_dim, input_dim)
        if leading_dims:
            partition = (ShardingAxis.EXPERT,) * len(leading_dims) + partition
        return FullPrecisionMatrix(
            spec=self,
            weights=initializer.normal(std, shape=shape, partition=partition),
        )


class FullPrecisionMatrix(EmbeddingMatrix[FullPrecisionSpec]):
    weights: Float[Array, "... out_channels in_channels"] = field(norm=ParameterNorm.SPECTRAL)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.weights.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.weights.dtype

    def decompress(self) -> Float[Array, "... out_channels in_channels"]:
        return self.weights

    def lookup_embedding(self, index: int | Int[Array, ""]) -> Float[Array, "... out_channels"]:
        self._raise_if_batched()
        if self.spec.layout == Layout.INPUT_OUTPUT:
            return self.weights[:, index]
        return self.weights[index, :]

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        key: Key[Array, ""] | None,  # noqa: ARG002
        forward_pass_config: MatmulConfig | None = None,
    ) -> Float[Array, "... out_channels"]:
        self._raise_if_batched()
        forward_pass_config = forward_pass_config or MatmulConfig()
        with default_matmul_precision(forward_pass_config.precision):
            if self.spec.layout == Layout.INPUT_OUTPUT:
                return vector @ self.weights
            return self.weights @ vector
