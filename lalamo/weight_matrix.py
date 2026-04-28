import math
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Generic, Self, TypeVar

import equinox as eqx
import jax.numpy as jnp
from cattrs import GenConverter
from jax import ShapeDtypeStruct, device_put
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.exportable import Exportable, ExportResults
from lalamo.initializer import Initializer
from lalamo.module import Keychain, ParameterNorm, ShardingAxis, field
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.json import JSON
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.registry_abc import RegistryABC, make_registry_abc_converter
from lalamo.utils.sharding import make_sharding, reshard_as

__all__ = [
    "CompressionImplementation",
    "GradientEstimator",
    "Layout",
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
    result = result.at[tril_rows, tril_cols].set(tril_vector)
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
    DETERMINISTIC_ROUNDING = "deterministic_rounding"
    STOCHASTIC_ROUNDING = "stochastic_rounding"
    LOCAL_ADDITIVE_NOISE = "local_additive_noise"


class CompressionImplementation(StrEnum):
    TRAINING = "training"
    INFERENCE = "inference"


@dataclass(frozen=True)
class MatmulConfig:
    gradient_estimator: GradientEstimator = GradientEstimator.DETERMINISTIC_ROUNDING
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
        weights: Float[Array | ShapeDtypeStruct, "*components out_channels in_channels"],
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> "WeightMatrix": ...


WeightMatrixSpecT_co = TypeVar("WeightMatrixSpecT_co", bound=WeightMatrixSpec, covariant=True)


class WeightMatrix(RegistryABC, Exportable, eqx.Module, Generic[WeightMatrixSpecT_co]):  # noqa: UP046
    spec: WeightMatrixSpecT_co = field(static=True)

    def _raise_if_batched(self) -> None:
        if self.ndim != 2:
            raise ValueError(
                "Attempted to call a method directly on a batched version of WeightMatrix. Use vmap instead.",
            )

    def switch_implementation(self, implementation: CompressionImplementation) -> Self:  # noqa: ARG002
        return self

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
    def astype(self, dtype: DTypeLike) -> Self: ...

    @abstractmethod
    def decompress(self) -> Float[Array, "... out_channels in_channels"]: ...

    @abstractmethod
    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
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
        loaded_spec = WeightMatrixSpec.from_json(saved_spec)
        if loaded_spec != self.spec:
            raise ValueError(f"WeightMatrix spec mismatch: expected {self.spec}, got {loaded_spec}")
        return super().load_exported(expored_data, allow_dtype_cast=allow_dtype_cast, prefix=prefix)


class EmbeddingMatrix(WeightMatrix[WeightMatrixSpecT_co]):
    @abstractmethod
    def lookup_embedding(
        self,
        index: int | Int[Array, ""],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
    ) -> Float[Array, "... out_channels"]: ...


class Layout(StrEnum):
    OUTPUT_INPUT = "output_input"
    INPUT_OUTPUT = "input_output"

    def weight_partition(self, num_leading_dims: int) -> tuple[ShardingAxis | None, ...]:
        if self == Layout.INPUT_OUTPUT:
            result = (None, ShardingAxis.TENSOR)
        else:
            result = (ShardingAxis.TENSOR, None)
        return (ShardingAxis.EXPERT,) * num_leading_dims + result

    def weight_shape(self, leading_dims: Sequence[int], output_dim: int, input_dim: int) -> tuple[int, ...]:
        if self == Layout.INPUT_OUTPUT:
            return (*leading_dims, input_dim, output_dim)
        return (*leading_dims, output_dim, input_dim)

    def convert_weights(
        self,
        weights: Float[Array | ShapeDtypeStruct, "... out_channels in_channels"],
    ) -> Float[Array, "... rows cols"]:
        sharding = make_sharding(self.weight_partition(weights.ndim - 2))
        if isinstance(weights, ShapeDtypeStruct):
            *leading_dims, output_dim, input_dim = weights.shape
            return dummy_array(
                shape=self.weight_shape(leading_dims, output_dim, input_dim),
                dtype=weights.dtype,
                sharding=sharding,
            )
        if self == Layout.INPUT_OUTPUT:
            weights = weights.swapaxes(-1, -2)
        return device_put(weights, sharding)

    def matmul(
        self,
        weights: Float[Array, "row cols"],
        vector: Float[Array, " in_channels"],
    ) -> Float[Array, " out_channels"]:
        if self == Layout.INPUT_OUTPUT:
            return vector @ weights
        return weights @ vector


@dataclass(frozen=True)
class FullPrecisionSpec(WeightMatrixSpec):
    layout: Layout = Layout.OUTPUT_INPUT

    def compress(
        self,
        weights: Float[Array | ShapeDtypeStruct, "... out_channels in_channels"],
        preconditioner: Preconditioner | None = None,  # noqa: ARG002
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,  # noqa: ARG002
    ) -> "FullPrecisionMatrix":
        return FullPrecisionMatrix(
            spec=self,
            weights=self.layout.convert_weights(weights),
        )

    def init(
        self,
        initializer: "Initializer",
        leading_dims: tuple[int, ...],
        output_dim: int,
        input_dim: int,
    ) -> "FullPrecisionMatrix":
        std = 1 / math.sqrt(input_dim)
        shape = self.layout.weight_shape(leading_dims, output_dim, input_dim)
        partition = self.layout.weight_partition(len(leading_dims))
        return FullPrecisionMatrix(
            spec=self,
            weights=initializer.normal(std, shape=shape, partition=partition),
        )


class FullPrecisionMatrix(EmbeddingMatrix[FullPrecisionSpec]):
    weights: Float[Array, "... m n"] = field(norm=ParameterNorm.SPECTRAL)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.weights.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.weights.dtype

    def astype(self, dtype: DTypeLike) -> "FullPrecisionMatrix":
        return FullPrecisionMatrix(spec=self.spec, weights=self.weights.astype(dtype))

    def decompress(self) -> Float[Array, "... out_channels in_channels"]:
        return self.weights

    def lookup_embedding(
        self,
        index: int | Int[Array, ""],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),  # noqa: ARG002
    ) -> Float[Array, "... out_channels"]:
        self._raise_if_batched()
        if self.spec.layout == Layout.INPUT_OUTPUT:
            return self.weights[index, :]
        raise ValueError(f"Embedding lookup not supported for layout {self.spec.layout}")

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),
    ) -> Float[Array, "... out_channels"]:
        self._raise_if_batched()
        with use_dot_algorithm_preset(forward_pass_config.precision):
            result = self.spec.layout.matmul(self.weights, vector)

        return reshard_as(result, vector)
