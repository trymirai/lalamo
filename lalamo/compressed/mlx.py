from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple, Self

import jax.numpy as jnp
from jax.lax import stop_gradient
from jaxtyping import Array, DTypeLike, Float, Int, Key, UInt8

from lalamo.exportable import ExportResults
from lalamo.module import Keychain, ParameterNorm, field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import preserve_first_input_sharding, supports_dummy_arrays
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.sharding import ShardingConfig, lookup_sharded_indices, with_sharding
from lalamo.utils.surgery import load_as
from lalamo.weight_matrix import (
    CompressionImplementation,
    EmbeddingMatrix,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    MatmulConfig,
    WeightMatrixSpec,
)

from .quantized_spec import QuantizedSpec
from .utils.gaussian_order_statistics import standard_normal_range_squared
from .utils.grouping import (
    expand_last_axis_groups,
    group_by_last_axis,
    min_max_within_groups,
    scale_from_min_max,
)
from .utils.packing import pack_uint_to_uint8, unpack_uint8_to_uint
from .utils.rounding import deterministic_round_to_unsigned_grid, round_to_unsigned_grid
from .utils.yaqa import yaqa_round_fixpoint

__all__ = [
    "MLXMatrix",
    "MLXMatrixForInference",
    "MLXMatrixForTraining",
    "MLXSpec",
]


class MLXAffineParameters(NamedTuple):
    scales: Float[Array, "*components rows groups"]
    biases: Float[Array, "*components rows groups"]

    @classmethod
    @supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
    def from_weights(
        cls,
        weights: Float[Array, "*components rows cols"],
        bits: int,
        group_size: int,
    ) -> Self:
        group_min_max = min_max_within_groups(group_by_last_axis(weights, group_size=group_size))
        scales = scale_from_min_max(group_min_max, bits=bits, dtype=weights.dtype)
        return cls(scales=scales, biases=group_min_max.min)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _master_weights_to_int_scale_weights(
    weights: Float[Array, "... cols"],
    scales: Float[Array, "... groups"],
    biases: Float[Array, "... groups"],
    group_size: int,
) -> Float[Array, "... cols"]:
    expanded_scales = expand_last_axis_groups(scales, group_size=group_size)
    expanded_biases = expand_last_axis_groups(biases, group_size=group_size)
    return (weights - expanded_biases) / expanded_scales


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _quantize(
    weights: Float[Array, "... cols"],
    scales: Float[Array, "... groups"],
    biases: Float[Array, "... groups"],
    group_size: int,
    round_fn: Callable[[Float[Array, "... cols"]], Float[Array, "... cols"]],
) -> Float[Array, "... cols"]:
    expanded_scales = expand_last_axis_groups(scales, group_size=group_size)
    expanded_biases = expand_last_axis_groups(biases, group_size=group_size)
    int_scale_weights = (weights - stop_gradient(expanded_biases)) / stop_gradient(expanded_scales)

    return round_fn(int_scale_weights) * expanded_scales + expanded_biases


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _master_weights_to_packed_weights(
    weights: Float[Array, "*components rows cols"],
    scales: Float[Array, "*components rows groups"],
    biases: Float[Array, "*components rows groups"],
    group_size: int,
    bits: int,
    *,
    sharding_config: ShardingConfig,
) -> UInt8[Array, "*components rows packed_cols"]:
    int_scale_weights = _master_weights_to_int_scale_weights(
        weights,
        scales,
        biases,
        group_size,
    )
    rounded_weights = deterministic_round_to_unsigned_grid(int_scale_weights, bits=bits)
    return pack_uint_to_uint8(rounded_weights.astype(jnp.uint8), bits, sharding_config=sharding_config)


def _packed_weights_to_master_weights(
    packed_weights: UInt8[Array, "... packed_cols"],
    scales: Float[Array, "... groups"],
    biases: Float[Array, "... groups"],
    group_size: int,
    bits: int,
) -> Float[Array, "... cols"]:
    int_weights = unpack_uint8_to_uint(packed_weights, bits=bits)
    expanded_scales = expand_last_axis_groups(scales, group_size=group_size)
    expanded_biases = expand_last_axis_groups(biases, group_size=group_size)
    return int_weights.astype(scales.dtype) * expanded_scales + expanded_biases


@dataclass(frozen=True)
class MLXSpec(QuantizedSpec):
    bits: Literal[4, 8]
    group_size: int
    layout: Layout = Layout.OUTPUT_INPUT

    @property
    def input_block_size(self) -> int:
        if self.layout == Layout.INPUT_OUTPUT:
            return 1
        return self.group_size

    @property
    def output_block_size(self) -> int:
        if self.layout == Layout.INPUT_OUTPUT:
            return self.group_size
        return 1

    @property
    def rate(self) -> float:
        return float(self.bits + 64 / self.group_size)

    @property
    def distortion(self) -> float:
        qmax = (2**self.bits) - 1
        endpoint_correction = (self.group_size - 2) / self.group_size
        return endpoint_correction * standard_normal_range_squared(self.group_size) / (12 * qmax**2)

    def compress(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        sharding_config: ShardingConfig,
        is_sharded: bool = True,
    ) -> "MLXMatrix":
        if preconditioner is not None:
            weights = yaqa_round_fixpoint(
                weights,
                preconditioner,
                self,
                sharding_config=sharding_config,
            )

        weight_axes = self.layout.weight_partition(weights.ndim - 2, is_sharded=is_sharded)
        weight_sharding = sharding_config.resolve_sharding(weight_axes)
        stored_weights = self.layout.from_output_input(weights, sharding=weight_sharding)
        affine_parameters = MLXAffineParameters.from_weights(
            stored_weights,
            bits=self.bits,
            group_size=self.group_size,
        )
        if implementation == CompressionImplementation.TRAINING:
            result = MLXMatrixForTraining(
                spec=self,
                sharding_config=sharding_config,
                is_sharded=is_sharded,
                master_weights=stored_weights,
                scales=affine_parameters.scales,
                biases=affine_parameters.biases,
            )
        else:
            packed_int_weights = _master_weights_to_packed_weights(
                stored_weights,
                affine_parameters.scales,
                affine_parameters.biases,
                self.group_size,
                self.bits,
                sharding_config=sharding_config,
            )
            result = self.from_packed_parameters(
                packed_weights=packed_int_weights,
                scales=affine_parameters.scales,
                biases=affine_parameters.biases,
                implementation=CompressionImplementation.INFERENCE,
                sharding_config=sharding_config,
                is_sharded=is_sharded,
            )

        return result

    def from_packed_parameters(
        self,
        *,
        packed_weights: UInt8[Array, "*components rows packed_cols"],
        scales: Float[Array, "*components rows groups"],
        biases: Float[Array, "*components rows groups"],
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        sharding_config: ShardingConfig,
        is_sharded: bool = True,
    ) -> "MLXMatrix":
        weight_axes = self.layout.weight_partition(scales.ndim - 2, is_sharded=is_sharded)
        weight_sharding = sharding_config.resolve_sharding(weight_axes)
        packed_weights = with_sharding(packed_weights, weight_sharding)
        scales = with_sharding(scales, weight_sharding)
        biases = with_sharding(biases, weight_sharding)

        if implementation == CompressionImplementation.INFERENCE:
            return MLXMatrixForInference(
                spec=self,
                sharding_config=sharding_config,
                is_sharded=is_sharded,
                packed_weights=packed_weights,
                scales=scales,
                biases=biases,
            )

        weights = _packed_weights_to_master_weights(
            packed_weights,
            scales,
            biases,
            self.group_size,
            self.bits,
        )
        return MLXMatrixForTraining(
            spec=self,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
            master_weights=with_sharding(weights, weight_sharding),
            scales=scales,
            biases=biases,
        )


class MLXMatrix(EmbeddingMatrix[MLXSpec]):
    scales: Float[Array, "*components rows groups"] = field(norm=ParameterNorm.L_INF)
    biases: Float[Array, "*components rows groups"] = field(norm=ParameterNorm.L_INF)

    @property
    @abstractmethod
    def _packed_quantized_weights(self) -> UInt8[Array, "*components rows packed_cols"]: ...

    def export(self) -> ExportResults:
        return ExportResults(
            arrays={
                "weights": self._packed_quantized_weights,
                "scales": self.scales,
                "biases": self.biases,
            },
            metadata={"spec": self.spec.to_json()},
        )

    @abstractmethod
    def load_exported(
        self,
        exported_data: ExportResults,
        allow_dtype_cast: bool = False,
        *,
        prefix: ParameterPath | None = None,
    ) -> "MLXMatrix": ...

    @abstractmethod
    def switch_implementation(self, implementation: CompressionImplementation) -> "MLXMatrix": ...

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(layout=self.spec.layout).compress(
            self.decompress(),
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
        )


class MLXMatrixForTraining(MLXMatrix):
    master_weights: Float[Array, "*components rows cols"] = field(norm=ParameterNorm.SPECTRAL)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.master_weights.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.master_weights.dtype

    @property
    def _packed_quantized_weights(self) -> UInt8[Array, "*components rows packed_cols"]:
        return _master_weights_to_packed_weights(
            self.master_weights,
            self.scales,
            self.biases,
            self.spec.group_size,
            self.spec.bits,
            sharding_config=self.sharding_config,
        )

    def astype(self, dtype: DTypeLike) -> "MLXMatrixForTraining":
        return MLXMatrixForTraining(
            spec=self.spec,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
            master_weights=self.master_weights.astype(dtype),
            scales=self.scales.astype(dtype),
            biases=self.biases.astype(dtype),
        )

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        weights = _quantize(
            self.master_weights,
            self.scales,
            self.biases,
            group_size=self.spec.group_size,
            round_fn=partial(deterministic_round_to_unsigned_grid, bits=self.spec.bits),
        )
        return self.spec.layout.to_output_input(weights)

    def lookup_embedding(
        self,
        row_index: int | Int[Array, "*batch"],
        *,
        dtype: DTypeLike | None = None,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
    ) -> Float[Array, "*batch out_channels"]:
        self._raise_if_batched()
        if self.spec.layout != Layout.INPUT_OUTPUT:
            raise ValueError(f"Embedding lookup not supported for layout {self.spec.layout}")
        if dtype is None:
            dtype = self.dtype
        weights = lookup_sharded_indices(self.master_weights, row_index).astype(dtype)
        return _quantize(
            weights,
            lookup_sharded_indices(self.scales, row_index).astype(dtype),
            lookup_sharded_indices(self.biases, row_index).astype(dtype),
            group_size=self.spec.group_size,
            round_fn=partial(
                round_to_unsigned_grid,
                bits=self.spec.bits,
                keychain=keychain,
                gradient_estimator=forward_pass_config.gradient_estimator,
            ),
        )

    def dot(
        self,
        vector: Float[Array, " source_channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Float[Array, " target_channels"]:
        self._raise_if_batched()
        dequantized_weights = _quantize(
            self.master_weights.astype(vector.dtype),
            self.scales.astype(vector.dtype),
            self.biases.astype(vector.dtype),
            group_size=self.spec.group_size,
            round_fn=partial(
                round_to_unsigned_grid,
                bits=self.spec.bits,
                keychain=keychain,
                gradient_estimator=forward_pass_config.gradient_estimator,
            ),
        )
        layout = self.spec.layout
        if transposed:
            layout = layout.transpose()
        with use_dot_algorithm_preset(forward_pass_config.precision):
            return layout.matmul(dequantized_weights, vector)

    def load_exported(
        self,
        exported_data: ExportResults,
        allow_dtype_cast: bool = False,
        *,
        prefix: ParameterPath | None = None,
    ) -> MLXMatrix:
        if prefix is None:
            prefix = ParameterPath()
        saved_spec = exported_data.metadata[prefix / "spec"]
        loaded_spec = WeightMatrixSpec.from_json(saved_spec)
        if loaded_spec != self.spec:
            raise ValueError(f"WeightMatrix spec mismatch: expected {self.spec}, got {loaded_spec}")

        packed_weights = load_as(
            self._packed_quantized_weights,
            exported_data.arrays[prefix / "weights"],
            allow_dtype_cast=False,
        )
        scales = load_as(
            self.scales,
            exported_data.arrays[prefix / "scales"],
            allow_dtype_cast=allow_dtype_cast,
        )
        biases = load_as(
            self.biases,
            exported_data.arrays[prefix / "biases"],
            allow_dtype_cast=allow_dtype_cast,
        )
        return self.spec.from_packed_parameters(
            packed_weights=packed_weights,
            scales=scales,
            biases=biases,
            implementation=CompressionImplementation.TRAINING,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> MLXMatrix:
        if implementation == CompressionImplementation.TRAINING:
            return self
        return self.spec.from_packed_parameters(
            packed_weights=self._packed_quantized_weights,
            scales=self.scales,
            biases=self.biases,
            implementation=CompressionImplementation.INFERENCE,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
        )


class MLXMatrixForInference(MLXMatrix):
    packed_weights: UInt8[Array, "*components rows packed_cols"]
    scales: Float[Array, "*components rows groups"]
    biases: Float[Array, "*components rows groups"]

    @property
    def shape(self) -> tuple[int, ...]:
        *leading_dims, rows, groups = self.scales.shape
        return (*leading_dims, rows, groups * self.spec.group_size)

    @property
    def dtype(self) -> DTypeLike:
        return self.scales.dtype

    @property
    def _packed_quantized_weights(self) -> UInt8[Array, "*components rows packed_cols"]:
        return self.packed_weights

    def astype(self, dtype: DTypeLike) -> "MLXMatrixForInference":
        return MLXMatrixForInference(
            spec=self.spec,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
            packed_weights=self.packed_weights,
            scales=self.scales.astype(dtype),
            biases=self.biases.astype(dtype),
        )

    def load_exported(
        self,
        exported_data: ExportResults,
        allow_dtype_cast: bool = False,
        *,
        prefix: ParameterPath | None = None,
    ) -> MLXMatrix:
        if prefix is None:
            prefix = ParameterPath()
        saved_spec = exported_data.metadata[prefix / "spec"]
        loaded_spec = WeightMatrixSpec.from_json(saved_spec)
        if loaded_spec != self.spec:
            raise ValueError(f"WeightMatrix spec mismatch: expected {self.spec}, got {loaded_spec}")

        packed_weights = load_as(
            self.packed_weights,
            exported_data.arrays[prefix / "weights"],
            allow_dtype_cast=False,
        )
        scales = load_as(
            self.scales,
            exported_data.arrays[prefix / "scales"],
            allow_dtype_cast=allow_dtype_cast,
        )
        biases = load_as(
            self.biases,
            exported_data.arrays[prefix / "biases"],
            allow_dtype_cast=allow_dtype_cast,
        )
        return MLXMatrixForInference(
            spec=self.spec,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
            packed_weights=packed_weights,
            scales=scales,
            biases=biases,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> MLXMatrix:
        if implementation == CompressionImplementation.INFERENCE:
            return self
        return self.spec.from_packed_parameters(
            packed_weights=self.packed_weights,
            scales=self.scales,
            biases=self.biases,
            implementation=CompressionImplementation.TRAINING,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
        )

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        weights = _packed_weights_to_master_weights(
            self.packed_weights,
            self.scales,
            self.biases,
            self.spec.group_size,
            self.spec.bits,
        )
        return self.spec.layout.to_output_input(weights)

    def lookup_embedding(
        self,
        row_index: int | Int[Array, "*batch"],
        *,
        dtype: DTypeLike | None = None,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),  # noqa: ARG002
    ) -> Float[Array, "*batch out_channels"]:
        self._raise_if_batched()
        if self.spec.layout != Layout.INPUT_OUTPUT:
            raise ValueError(f"Embedding lookup not supported for layout {self.spec.layout}")
        if dtype is None:
            dtype = self.dtype
        packed_weights = lookup_sharded_indices(self.packed_weights, row_index)
        return _packed_weights_to_master_weights(
            packed_weights,
            lookup_sharded_indices(self.scales, row_index).astype(dtype),
            lookup_sharded_indices(self.biases, row_index).astype(dtype),
            self.spec.group_size,
            self.spec.bits,
        )

    def dot(
        self,
        vector: Float[Array, " source_channels"],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Float[Array, " target_channels"]:
        self._raise_if_batched()
        weights = _packed_weights_to_master_weights(
            self.packed_weights,
            self.scales.astype(vector.dtype),
            self.biases.astype(vector.dtype),
            self.spec.group_size,
            self.spec.bits,
        )
        layout = self.spec.layout
        if transposed:
            layout = layout.transpose()
        with use_dot_algorithm_preset(forward_pass_config.precision):
            return layout.matmul(weights, vector)
