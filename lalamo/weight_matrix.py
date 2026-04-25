import math
from abc import abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Generic, Self, TypeVar

import equinox as eqx
import jax.numpy as jnp
from cattrs import GenConverter
from jax import default_matmul_precision
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array, DTypeLike, Float, Int, Key

from lalamo.exportable import Exportable, ExportResults
from lalamo.initializer import Initializer
from lalamo.module import ParameterNorm, ShardingAxis, field
from lalamo.utils.json import JSON
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.registry_abc import RegistryABC, make_registry_abc_converter

__all__ = [
    "GradientEstimator",
    "MatmulConfig",
    "Preconditioner",
    "WeightMatrix",
    "WeightMatrixSpec",
]


@eqx.filter_jit
def sym_to_tril(sym_matrix: Float[Array, "n n"]) -> Float[Array, " n*(n+1)//2"]:
    tril_indices = jnp.tril_indices_from(sym_matrix)
    return sym_matrix[tril_indices]


@eqx.filter_jit
def tril_to_sym(tril_vector: Float[Array, " n*(n+1)//2"]) -> Float[Array, "n n"]:
    (numel,) = tril_vector.shape
    num_rows = int(math.sqrt(8 * numel + 1) - 1) // 2
    result = jnp.empty((num_rows, num_rows), dtype=tril_vector.dtype)
    tril_rows, tril_cols = jnp.tril_indices_from(result)
    return result.at[tril_cols, tril_rows].set(tril_vector)


class Preconditioner(eqx.Module):
    input_block_tril: Float[Array, " input_channels*(input_channels+1)//2"] | None
    output_block_tril: Float[Array, " output_channels*(output_channels+1)//2"] | None

    @property
    def input_block(self) -> Float[Array, "input_channels input_channels"] | None:
        if self.input_block_tril is None:
            return None
        return tril_to_sym(self.input_block_tril)

    @property
    def output_block(
        self,
    ) -> Float[Array, "output_channels output_channels"] | None:
        if self.output_block_tril is None:
            return None
        return tril_to_sym(self.output_block_tril)

    @classmethod
    def init(
        cls,
        input_block: Float[Array, "input_channels input_channels"] | None = None,
        output_block: Float[Array, "output_channels output_channels"] | None = None,
    ) -> Self:
        return cls(
            input_block_tril=sym_to_tril(input_block) if input_block is not None else None,
            output_block_tril=sym_to_tril(output_block) if output_block is not None else None,
        )

    @classmethod
    def identity(cls) -> Self:
        return cls(
            input_block_tril=None,
            output_block_tril=None,
        )


class GradientEstimator(StrEnum):
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"


@dataclass(frozen=True)
class MatmulConfig:
    gradient_estimator: GradientEstimator = GradientEstimator.DETERMINISTIC
    precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT


@dataclass(frozen=True)
class WeightMatrixSpec(RegistryABC):
    _converter: ClassVar[GenConverter] = make_registry_abc_converter()

    @classmethod
    def from_json(cls, json_object: JSON) -> Self:
        return cls._converter.structure(json_object, cls)

    def to_json(self) -> JSON:
        return self._converter.unstructure(self)

    def compress(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        preconditioner: Preconditioner | None = None,
    ) -> "WeightMatrix": ...

    @abstractmethod
    def init(
        self,
        initializer: Initializer,
        leading_dims: tuple[int, ...],
        output_dim: int,
        input_dim: int,
    ) -> "WeightMatrix": ...


WeightMatrixSpecT_co = TypeVar("WeightMatrixSpecT_co", bound=WeightMatrixSpec, covariant=True)


class WeightMatrix(RegistryABC, Exportable, eqx.Module, Generic[WeightMatrixSpecT_co]):  # noqa: UP046
    spec: WeightMatrixSpecT_co = field(static=True)

    def _raise_if_batched(self) -> None:
        if self.ndim != 2:
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
        dequant_key: Key[Array, ""],
        forward_pass_config: MatmulConfig | None = None,
    ) -> Float[Array, "... out_channels"]: ...

    def export(self) -> ExportResults:
        arrays, metadata = super().export()
        return ExportResults(
            arrays=arrays,
            metadata={"spec": self.spec.to_json(), **metadata},
        )

    def load_exported(
        self,
        expored_data: ExportResults,
        allow_dtype_cast: bool = False,
        *,
        prefix: ParameterPath | None = None,
    ) -> Self:
        if prefix is None:
            prefix = ParameterPath()
        saved_spec = expored_data.metadata[prefix / "spec"]
        loaded_spec = self.spec.from_json(saved_spec)
        if loaded_spec != self.spec:
            raise ValueError(f"WeightMatrix spec mismatch: expected {self.spec}, got {loaded_spec}")
        return super().load_exported(expored_data, allow_dtype_cast=allow_dtype_cast, prefix=prefix)


class EmbeddingMatrix(WeightMatrix[WeightMatrixSpecT_co]):
    @abstractmethod
    def lookup_embedding(
        self,
        index: int | Int[Array, ""],
        *,
        dequant_key: Key[Array, ""],
        forward_pass_config: MatmulConfig | None = None,
    ) -> Float[Array, "... out_channels"]: ...


class Layout(StrEnum):
    OUTPUT_INPUT = "output_input"
    INPUT_OUTPUT = "input_output"


@dataclass(frozen=True)
class FullPrecisionSpec(WeightMatrixSpec):
    layout: Layout = Layout.OUTPUT_INPUT

    def compress(
        self,
        weights: Float[Array, "... out_channels in_channels"],
        preconditioner: Preconditioner | None = None,  # noqa: ARG002
    ) -> "FullPrecisionMatrix":
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

    def lookup_embedding(
        self,
        index: int | Int[Array, ""],
        *,
        dequant_key: Key[Array, ""],  # noqa: ARG002
        forward_pass_config: MatmulConfig | None = None,  # noqa: ARG002
    ) -> Float[Array, "... out_channels"]:
        self._raise_if_batched()
        if self.spec.layout == Layout.INPUT_OUTPUT:
            return self.weights[:, index]
        raise ValueError(f"Embedding lookup not supported for layout {self.spec.layout}")

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        dequant_key: Key[Array, ""],  # noqa: ARG002
        forward_pass_config: MatmulConfig | None = None,
    ) -> Float[Array, "... out_channels"]:
        self._raise_if_batched()
        forward_pass_config = forward_pass_config or MatmulConfig()
        with default_matmul_precision(forward_pass_config.precision):
            if self.spec.layout == Layout.INPUT_OUTPUT:
                return vector @ self.weights
            return self.weights @ vector
