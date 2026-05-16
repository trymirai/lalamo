import math
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Generic, Literal, Self, TypeVar, cast, overload

import equinox as eqx
from cattrs import GenConverter
from jax import ShapeDtypeStruct
from jax.lax import DotAlgorithmPreset, dot_general
from jaxtyping import Array, DTypeLike, Float, Int, Key

from lalamo.exportable import Exportable, ExportResults
from lalamo.module import Keychain, ParameterNorm, ShardingAxis, field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import dummy_array, supports_dummy_arrays
from lalamo.utils.json import JSON
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.registry_abc import RegistryABC, make_registry_abc_converter
from lalamo.utils.sharding import lookup_sharded_indices, make_sharding, sharding_of, with_sharding

__all__ = [
    "CompressionImplementation",
    "FullPrecisionMatrix",
    "FullPrecisionSpec",
    "GradientEstimator",
    "Layout",
    "MatmulConfig",
    "WeightMatrix",
    "WeightMatrixSpec",
]


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

    @classmethod
    def for_tracer_tests(cls) -> Self:
        return cls(
            precision=DotAlgorithmPreset.F32_F32_F32,
        )

    @classmethod
    def for_inference(cls, precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT) -> Self:
        return cls(precision=precision)

    @classmethod
    def for_training(
        cls,
        gradient_estimator: GradientEstimator = GradientEstimator.DETERMINISTIC_ROUNDING,
        precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT,
    ) -> Self:
        return cls(
            gradient_estimator=gradient_estimator,
            precision=precision,
        )


@dataclass(frozen=True)
class WeightMatrixSpec(RegistryABC):
    _converter: ClassVar[GenConverter] = make_registry_abc_converter()

    @classmethod
    def from_json(cls, json_object: JSON) -> Self:
        return cls._converter.structure(json_object, cls)

    def to_json(self) -> JSON:
        return self._converter.unstructure(self)

    @supports_dummy_arrays()
    @abstractmethod
    def compress(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        is_sharded: bool = True,
    ) -> "WeightMatrix": ...


WeightMatrixSpecT_co = TypeVar("WeightMatrixSpecT_co", bound=WeightMatrixSpec, covariant=True)


class WeightMatrix(RegistryABC, Exportable, eqx.Module, Generic[WeightMatrixSpecT_co]):  # noqa: UP046
    spec: WeightMatrixSpecT_co = field(static=True)
    is_sharded: bool = field(static=True)

    def _raise_if_batched(self) -> None:
        if self.ndim != 2:
            raise ValueError(
                "Attempted to call a method directly on a batched version of WeightMatrix. Use vmap instead.",
            )

    def switch_implementation(self, implementation: CompressionImplementation) -> Self:  # noqa: ARG002
        return self

    def switch_sharding(self, is_sharded: bool) -> Self:
        if is_sharded == self.is_sharded:
            return self
        weights = self.decompress()
        result = self.spec.compress(
            weights,
            implementation=CompressionImplementation.INFERENCE,
            is_sharded=is_sharded,
        )
        if isinstance(result, type(self)):
            return result
        result = self.spec.compress(
            weights,
            implementation=CompressionImplementation.TRAINING,
            is_sharded=is_sharded,
        )
        assert isinstance(result, type(self))
        return result

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    @abstractmethod
    def dtype(self) -> DTypeLike: ...

    @abstractmethod
    def astype(self, dtype: DTypeLike) -> Self: ...

    @abstractmethod
    def to_full_precision(self) -> "FullPrecisionMatrix": ...

    @abstractmethod
    def decompress(self) -> Float[Array, "*components out_channels in_channels"]: ...

    @overload
    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: Literal[False] = False,
    ) -> Float[Array, " out_channels"]: ...

    @overload
    def dot(
        self,
        vector: Float[Array, " out_channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: Literal[True],
    ) -> Float[Array, " in_channels"]: ...

    @abstractmethod
    def dot(
        self,
        vector: Float[Array, " source_channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Float[Array, " target_channels"]: ...

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
    ) -> "WeightMatrix":
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
        index: int | Int[Array, "*batch"],
        *,
        keychain: Keychain,
        dtype: DTypeLike | None = None,
        forward_pass_config: MatmulConfig = MatmulConfig(),
    ) -> Float[Array, "*batch out_channels"]: ...


class Layout(StrEnum):
    OUTPUT_INPUT = "output_input"
    INPUT_OUTPUT = "input_output"

    def weight_partition(self, num_leading_dims: int, *, is_sharded: bool = True) -> tuple[ShardingAxis | None, ...]:
        if not is_sharded:
            return (None,) * (num_leading_dims + 2)
        if num_leading_dims > 0:
            return (ShardingAxis.EXPERT,) * num_leading_dims + (None, None)
        if self == Layout.INPUT_OUTPUT:
            result = (None, ShardingAxis.TENSOR)
        else:
            result = (ShardingAxis.TENSOR, None)
        return (ShardingAxis.EXPERT,) * num_leading_dims + result

    def weight_shape(self, leading_dims: Sequence[int], output_dim: int, input_dim: int) -> tuple[int, ...]:
        if self == Layout.INPUT_OUTPUT:
            return (*leading_dims, input_dim, output_dim)
        return (*leading_dims, output_dim, input_dim)

    @supports_dummy_arrays()
    def from_output_input(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        *,
        is_sharded: bool = True,
    ) -> Float[Array, "*components rows cols"]:
        sharding = make_sharding(self.weight_partition(weights.ndim - 2, is_sharded=is_sharded))
        if self == Layout.INPUT_OUTPUT:
            weights = weights.swapaxes(-1, -2)
        return with_sharding(weights, sharding)

    @supports_dummy_arrays()
    def to_output_input(
        self,
        weights: Float[Array, "*components rows cols"],
    ) -> Float[Array, "*components out_channels in_channels"]:
        if self == Layout.INPUT_OUTPUT:
            return weights.swapaxes(-1, -2)
        return weights

    @overload
    def matmul(
        self: "Literal[Layout.OUTPUT_INPUT]",
        weights: Float[Array, "out_channels in_channels"],
        vector: Float[Array, " in_channels"],
    ) -> Float[Array, " out_channels"]: ...

    @overload
    def matmul(
        self: "Literal[Layout.INPUT_OUTPUT]",
        weights: Float[Array, "in_channels out_channels"],
        vector: Float[Array, " in_channels"],
    ) -> Float[Array, " out_channels"]: ...

    @overload
    def matmul(
        self: "Layout",
        weights: Float[Array, "rows cols"],
        vector: Float[Array, " source_channels"],
    ) -> Float[Array, " target_channels"]: ...

    def matmul(
        self,
        weights: Float[Array, "rows cols"],
        vector: Float[Array, " source_channels"],
    ) -> Float[Array, " target_channels"]:
        if self == Layout.INPUT_OUTPUT:
            return dot_general(
                vector,
                weights,
                dimension_numbers=(((0,), (0,)), ((), ())),
                out_sharding=sharding_of(vector),
            )
        return dot_general(
            weights,
            vector,
            dimension_numbers=(((1,), (0,)), ((), ())),
            out_sharding=sharding_of(vector),
        )

    def transpose(self) -> "Layout":
        if self == Layout.INPUT_OUTPUT:
            return Layout.OUTPUT_INPUT
        return Layout.INPUT_OUTPUT


@dataclass(frozen=True)
class FullPrecisionSpec(WeightMatrixSpec):
    layout: Layout = Layout.OUTPUT_INPUT

    def compress(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,  # noqa: ARG002
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,  # noqa: ARG002
        is_sharded: bool = True,
    ) -> "FullPrecisionMatrix":
        return FullPrecisionMatrix(
            spec=self,
            is_sharded=is_sharded,
            weights=self.layout.from_output_input(weights, is_sharded=is_sharded),
        )


class FullPrecisionMatrix(EmbeddingMatrix[FullPrecisionSpec]):
    weights: Float[Array, "*components m n"] = field(norm=ParameterNorm.SPECTRAL)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.weights.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.weights.dtype

    def astype(self, dtype: DTypeLike) -> "FullPrecisionMatrix":
        return FullPrecisionMatrix(spec=self.spec, is_sharded=self.is_sharded, weights=self.weights.astype(dtype))

    def to_full_precision(self) -> "FullPrecisionMatrix":
        return self

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        return self.spec.layout.to_output_input(self.weights)

    def lookup_embedding(
        self,
        index: int | Int[Array, "*batch"],
        *,
        dtype: DTypeLike | None = None,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),  # noqa: ARG002
    ) -> Float[Array, "*batch out_channels"]:
        self._raise_if_batched()
        if self.spec.layout == Layout.INPUT_OUTPUT:
            result = lookup_sharded_indices(self.weights, index)
            if dtype is not None:
                return result.astype(dtype)
            return result
        raise ValueError(f"Embedding lookup not supported for layout {self.spec.layout}")

    @overload
    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: Literal[False] = False,
    ) -> Float[Array, " out_channels"]: ...

    @overload
    def dot(
        self,
        vector: Float[Array, " out_channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: Literal[True],
    ) -> Float[Array, " in_channels"]: ...

    def dot(
        self,
        vector: Float[Array, " source_channels"],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Float[Array, " target_channels"]:
        self._raise_if_batched()
        weights = self.weights.astype(vector.dtype)
        layout = self.spec.layout
        if transposed:
            layout = layout.transpose()
        with use_dot_algorithm_preset(forward_pass_config.precision):
            return layout.matmul(weights, vector)


@dataclass(frozen=True)
class ShapeDtypeSpec(WeightMatrixSpec):
    layout: Layout = Layout.OUTPUT_INPUT

    def compress(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,  # noqa: ARG002
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,  # noqa: ARG002
        is_sharded: bool = True,
    ) -> "ShapeDtypeMatrix":
        if not isinstance(weights, ShapeDtypeStruct):
            raise TypeError("Can only compress ShapeDtypeStructs to ShapeDtypeMatrices")
        *mixture_dims, output_dim, input_dim = weights.shape
        return ShapeDtypeMatrix(
            spec=self,
            is_sharded=is_sharded,
            mixture_dims=tuple(mixture_dims),
            input_dim=input_dim,
            output_dim=output_dim,
            dtype_=weights.dtype,
        )


class ShapeDtypeMatrix(EmbeddingMatrix[ShapeDtypeSpec]):
    mixture_dims: tuple[int, ...]
    input_dim: int
    output_dim: int
    dtype_: DTypeLike

    @property
    def shape(self) -> tuple[int, ...]:
        return self.spec.layout.weight_shape(self.mixture_dims, self.output_dim, self.input_dim)

    @property
    def dtype(self) -> DTypeLike:
        return self.dtype_

    def astype(self, dtype: DTypeLike) -> "ShapeDtypeMatrix":
        return ShapeDtypeMatrix(
            spec=self.spec,
            is_sharded=self.is_sharded,
            mixture_dims=self.mixture_dims,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            dtype_=dtype,
        )

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        return dummy_array(
            shape=(*self.mixture_dims, self.output_dim, self.input_dim),
            dtype=self.dtype_,
        )

    def load_exported(
        self,
        expored_data: ExportResults,
        allow_dtype_cast: bool = False,
        *,
        prefix: ParameterPath | None = None,
    ) -> WeightMatrix:
        if prefix is None:
            prefix = ParameterPath()

        # You call this an ugly hack, I call this an elegant solution to a difficult problem (@norpadon).
        saved_spec = expored_data.metadata[prefix / "spec"]
        loaded_spec = WeightMatrixSpec.from_json(saved_spec)
        dummy_layer = loaded_spec.compress(self.decompress(), is_sharded=self.is_sharded)
        result = dummy_layer.load_exported(expored_data, allow_dtype_cast=allow_dtype_cast, prefix=prefix)
        return cast("Self", result)

    def to_full_precision(self) -> "FullPrecisionMatrix":
        raise TypeError("Can't cast dummy ShapeDtypeMatrix to FullPrecisionMatrix")

    def lookup_embedding(
        self,
        index: int | Int[Array, "*batch"],  # noqa: ARG002
        *,
        dtype: DTypeLike | None = None,  # noqa: ARG002
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),  # noqa: ARG002
    ) -> Float[Array, "*batch out_channels"]:
        raise TypeError("Cannot perform embedding lookup on ShapeDtypeMatrix")

    def dot(
        self,
        vector: Float[Array, " source_channels"],  # noqa: ARG002
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),  # noqa: ARG002
        transposed: bool = False,  # noqa: ARG002
    ) -> Float[Array, " target_channels"]:
        raise TypeError("Cannot perform matmul on ShapeDtypeMatrix")

    def export(self) -> ExportResults:
        raise TypeError("Cannot export ShapeDtypeMatrix")
