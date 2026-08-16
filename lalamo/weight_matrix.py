import math
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Generic, Literal, Self, TypeVar, cast, overload

import equinox as eqx
import jax.tree_util as jtu
from cattrs import GenConverter
from jax import ShapeDtypeStruct
from jax.lax import DotAlgorithmPreset, dot_general
from jax.sharding import NamedSharding
from jaxtyping import Array, DTypeLike, Float, Int, Key

from lalamo.exportable import Exportable, ExportResults
from lalamo.module import Keychain, ParameterNorm, field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import dummy_array, supports_dummy_arrays
from lalamo.utils.json import JSON
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.registry_abc import RegistryABC, make_registry_abc_converter
from lalamo.utils.sharding import (
    LogicalAxis,
    ShardingConfig,
    lookup_sharded_indices,
    sharding_of,
    with_sharding,
)

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
        sharding_config: ShardingConfig,
        is_sharded: bool = True,
    ) -> "WeightMatrix": ...


WeightMatrixSpecT_co = TypeVar("WeightMatrixSpecT_co", bound=WeightMatrixSpec, covariant=True)


class WeightMatrix(RegistryABC, Exportable, eqx.Module, Generic[WeightMatrixSpecT_co]):  # noqa: UP046
    spec: WeightMatrixSpecT_co = field(static=True)
    sharding_config: ShardingConfig = field(static=True)
    is_sharded: bool = field(static=True)

    def _raise_if_batched(self) -> None:
        if self.ndim != 2:
            raise ValueError(
                "Attempted to call a method directly on a batched version of WeightMatrix. Use vmap instead.",
            )

    def _resolve_axis(self, logical_axis: LogicalAxis | None) -> str | None:
        return self.sharding_config.resolve_axis(logical_axis)

    def _resolve_sharding(self, logical_axes: Iterable[LogicalAxis | None]) -> NamedSharding:
        return self.sharding_config.resolve_sharding(logical_axes)

    def switch_implementation(self, implementation: CompressionImplementation) -> Self:  # noqa: ARG002
        return self

    def switch_sharding_config(self, sharding_config: ShardingConfig) -> Self:
        if sharding_config == self.sharding_config:
            return self
        weights = self.decompress()
        result = self.spec.compress(
            weights,
            implementation=CompressionImplementation.INFERENCE,
            sharding_config=sharding_config,
            is_sharded=self.is_sharded,
        )
        if isinstance(result, type(self)):
            return result
        result = self.spec.compress(
            weights,
            implementation=CompressionImplementation.TRAINING,
            sharding_config=sharding_config,
            is_sharded=self.is_sharded,
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
        exported_data: ExportResults,
        *,
        prefix: ParameterPath | None = None,
    ) -> "WeightMatrix":
        if prefix is None:
            prefix = ParameterPath()
        saved_spec = exported_data.metadata[prefix / "spec"]
        loaded_spec = WeightMatrixSpec.from_json(saved_spec)
        if loaded_spec != self.spec:
            raise ValueError(f"WeightMatrix spec mismatch: expected {self.spec}, got {loaded_spec}")
        return super().load_exported(exported_data, prefix=prefix)


class EmbeddingMatrix(WeightMatrix[WeightMatrixSpecT_co]):
    @abstractmethod
    def lookup_embedding(
        self,
        row_index: int | Int[Array, "*batch"],
        *,
        keychain: Keychain,
        dtype: DTypeLike | None = None,
        forward_pass_config: MatmulConfig = MatmulConfig(),
    ) -> Float[Array, "*batch out_channels"]: ...


class Layout(StrEnum):
    OUTPUT_INPUT = "output_input"
    INPUT_OUTPUT = "input_output"

    def weight_partition(
        self,
        num_leading_dims: int,
        *,
        is_sharded: bool = True,
    ) -> tuple[LogicalAxis | None, ...]:
        if not is_sharded:
            return (None,) * (num_leading_dims + 2)
        if num_leading_dims > 0:
            return (LogicalAxis.MIXTURE,) * num_leading_dims + (None, None)
        if self == Layout.INPUT_OUTPUT:
            return (None, LogicalAxis.MATRIX)
        return (LogicalAxis.MATRIX, None)

    def weight_shape(self, leading_dims: Sequence[int], output_dim: int, input_dim: int) -> tuple[int, ...]:
        if self == Layout.INPUT_OUTPUT:
            return (*leading_dims, input_dim, output_dim)
        return (*leading_dims, output_dim, input_dim)

    @supports_dummy_arrays()
    def from_output_input(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        *,
        sharding: NamedSharding,
    ) -> Float[Array, "*components rows cols"]:
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
        out_sharding = sharding_of(vector)
        if self == Layout.INPUT_OUTPUT:
            return dot_general(
                vector,
                weights,
                dimension_numbers=(((0,), (0,)), ((), ())),
                out_sharding=out_sharding,
            )
        return dot_general(
            weights,
            vector,
            dimension_numbers=(((1,), (0,)), ((), ())),
            out_sharding=out_sharding,
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
        sharding_config: ShardingConfig,
        is_sharded: bool = True,
    ) -> "FullPrecisionMatrix":
        logical_axes = self.layout.weight_partition(weights.ndim - 2, is_sharded=is_sharded)
        sharding = sharding_config.resolve_sharding(logical_axes)
        return FullPrecisionMatrix(
            spec=self,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
            weights=self.layout.from_output_input(
                weights,
                sharding=sharding,
            ),
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
        return FullPrecisionMatrix(
            spec=self.spec,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
            weights=self.weights.astype(dtype),
        )

    def to_full_precision(self) -> "FullPrecisionMatrix":
        return self

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        return self.spec.layout.to_output_input(self.weights)

    def lookup_embedding(
        self,
        row_index: int | Int[Array, "*batch"],
        *,
        dtype: DTypeLike | None = None,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),  # noqa: ARG002
    ) -> Float[Array, "*batch out_channels"]:
        self._raise_if_batched()
        if self.spec.layout == Layout.INPUT_OUTPUT:
            result = lookup_sharded_indices(self.weights, row_index)
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
        sharding_config: ShardingConfig,
        is_sharded: bool = True,
    ) -> "ShapeDtypeMatrix":
        if not isinstance(weights, ShapeDtypeStruct):
            raise TypeError("Can only compress ShapeDtypeStructs to ShapeDtypeMatrices")
        *mixture_dims, _output_dim, _input_dim = weights.shape
        logical_axes = Layout.OUTPUT_INPUT.weight_partition(len(mixture_dims), is_sharded=is_sharded)
        sharding = sharding_config.resolve_sharding(logical_axes)
        dummy_weights = dummy_array(
            weights.shape,
            weights.dtype,
            sharding,
            weak_type=weights.weak_type,
        )
        return ShapeDtypeMatrix(
            spec=self,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
            dummy_weights=dummy_weights,
        )


class ShapeDtypeMatrix(EmbeddingMatrix[ShapeDtypeSpec]):
    dummy_weights: Float[Array, "*components out_channels in_channels"]

    @property
    def mixture_dims(self) -> tuple[int, ...]:
        *mixture_dims, _output_dim, _input_dim = self.dummy_weights.shape
        return tuple(mixture_dims)

    @property
    def input_dim(self) -> int:
        *_, input_dim = self.dummy_weights.shape
        return input_dim

    @property
    def output_dim(self) -> int:
        *_, output_dim, _input_dim = self.dummy_weights.shape
        return output_dim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.spec.layout.weight_shape(self.mixture_dims, self.output_dim, self.input_dim)

    @property
    def dtype(self) -> DTypeLike:
        return self.dummy_weights.dtype

    def astype(self, dtype: DTypeLike) -> "ShapeDtypeMatrix":
        return ShapeDtypeMatrix(
            spec=self.spec,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
            dummy_weights=dummy_array(
                shape=self.dummy_weights.shape,
                dtype=dtype,
                sharding=sharding_of(self.dummy_weights),
            ),
        )

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        return dummy_array(
            self.dummy_weights.shape,
            self.dummy_weights.dtype,
            self.sharding_config.resolve_sharding((None,) * len(self.dummy_weights.shape)),
            weak_type=self.dummy_weights.weak_type,
        )

    def load_exported(
        self,
        exported_data: ExportResults,
        *,
        prefix: ParameterPath | None = None,
    ) -> WeightMatrix:
        if prefix is None:
            prefix = ParameterPath()

        saved_spec = exported_data.metadata[prefix / "spec"]
        loaded_spec = WeightMatrixSpec.from_json(saved_spec)
        dummy_layer = loaded_spec.compress(
            self.decompress(),
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
        )
        if self.dummy_weights.weak_type:

            def keep_weak(leaf: ShapeDtypeStruct) -> Array:
                return dummy_array(leaf.shape, leaf.dtype, sharding_of(leaf), weak_type=True)

            dummy_layer = cast("WeightMatrix", jtu.tree_map(keep_weak, dummy_layer))
        result = dummy_layer.load_exported(exported_data, prefix=prefix)
        return cast("Self", result)

    def to_full_precision(self) -> "FullPrecisionMatrix":
        raise TypeError("Can't cast dummy ShapeDtypeMatrix to FullPrecisionMatrix")

    def lookup_embedding(
        self,
        row_index: int | Int[Array, "*batch"],  # noqa: ARG002
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
