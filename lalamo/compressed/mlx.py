from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple, Self

import jax.numpy as jnp
from jax import ShapeDtypeStruct
from jax.lax import stop_gradient
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.exportable import ExportResults
from lalamo.module import Keychain, ParameterNorm, field
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.surgery import load_as
from lalamo.weight_matrix import (
    CompressionImplementation,
    EmbeddingMatrix,
    Layout,
    MatmulConfig,
    Preconditioner,
    WeightMatrixSpec,
)

from .packing import pack_uint_to_uint8, unpack_uint8_to_uint
from .rounding import deterministic_round_to_unsigned_grid, round_to_unsigned_grid
from .utils import (
    expand_last_axis_groups,
    group_by_last_axis,
    grouped_last_axis_shape,
    min_max_within_groups,
    scale_from_min_max,
)

__all__ = [
    "MLXMatrix",
    "MLXMatrixForInference",
    "MLXMatrixForTraining",
    "MLXSpec",
]


class MLXAffineParameters(NamedTuple):
    scales: Float[Array, "... groups"]
    biases: Float[Array, "... groups"]

    @classmethod
    def from_weights(
        cls,
        weights: Float[Array | ShapeDtypeStruct, "... rows cols"],
        bits: int,
        group_size: int,
    ) -> Self:
        if isinstance(weights, ShapeDtypeStruct):
            grouped_shape = grouped_last_axis_shape(weights.shape, group_size=group_size)
            return cls(
                scales=dummy_array(grouped_shape, weights.dtype, weights.sharding),
                biases=dummy_array(grouped_shape, weights.dtype, weights.sharding),
            )

        group_min_max = min_max_within_groups(group_by_last_axis(weights, group_size=group_size))
        scales = scale_from_min_max(group_min_max, bits=bits, dtype=weights.dtype)
        return cls(scales=scales, biases=group_min_max.min)


def _mlx_master_weights_to_int_scale[ArrayT: Array | ShapeDtypeStruct](
    weights: Float[ArrayT, "... cols"],
    scales: Float[ArrayT, "... groups"],
    biases: Float[ArrayT, "... groups"],
    group_size: int,
) -> Float[Array, "... rows cols"]:
    if not isinstance(weights, Array):
        return dummy_array(weights.shape, weights.dtype, weights.sharding)
    expanded_scales = expand_last_axis_groups(scales, group_size=group_size)
    expanded_biases = expand_last_axis_groups(biases, group_size=group_size)
    return (weights - expanded_biases) / expanded_scales


def _mlx_quantize[ArrayT: Array | ShapeDtypeStruct](
    weights: Float[ArrayT, "... cols"],
    scales: Float[ArrayT, "... groups"],
    biases: Float[ArrayT, "... groups"],
    group_size: int,
    round_fn: Callable[[Float[Array, "... rows cols"]], Float[Array, "... rows cols"]],
) -> Float[Array, "... rows cols"]:
    if not isinstance(weights, Array):
        return dummy_array(weights.shape, weights.dtype, weights.sharding)

    expanded_scales = expand_last_axis_groups(scales, group_size=group_size)
    expanded_biases = expand_last_axis_groups(biases, group_size=group_size)
    int_scale_weights = (weights - stop_gradient(expanded_biases)) / stop_gradient(expanded_scales)

    return round_fn(int_scale_weights) * expanded_scales + expanded_biases


def _mlx_pack_master_weights(
    weights: Float[Array, "... rows cols"],
    scales: Float[Array, "... rows groups"],
    biases: Float[Array, "... rows groups"],
    group_size: int,
    bits: int,
) -> Int[Array, "... rows packed_cols"]:
    int_scale_weights = _mlx_master_weights_to_int_scale(
        weights,
        scales,
        biases,
        group_size,
    )
    int_weights = deterministic_round_to_unsigned_grid(int_scale_weights, bits=bits).astype(jnp.int8)
    return pack_uint_to_uint8(int_weights, bits)


def _mlx_unpack_master_weights(
    packed_weights: Int[Array, "... rows packed_cols"],
    scales: Float[Array, "... rows groups"],
    biases: Float[Array, "... rows groups"],
    group_size: int,
    bits: int,
) -> Float[Array, "... rows cols"]:
    int_weights = unpack_uint8_to_uint(packed_weights, bits=bits)
    expanded_scales = expand_last_axis_groups(scales, group_size=group_size)
    expanded_biases = expand_last_axis_groups(biases, group_size=group_size)
    return int_weights.astype(scales.dtype) * expanded_scales + expanded_biases


@dataclass(frozen=True)
class MLXSpec(WeightMatrixSpec):
    bits: Literal[4, 8]
    group_size: int
    layout: Layout = Layout.OUTPUT_INPUT

    def compress(
        self,
        weights: Float[Array | ShapeDtypeStruct, "... out_channels in_channels"],
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> "MLXMatrix":
        if preconditioner is not None:
            raise ValueError("Preconditioned rounding is not implemented yet.")

        weights = self.layout.convert_weights(weights)
        affine_parameters = MLXAffineParameters.from_weights(
            weights,
            bits=self.bits,
            group_size=self.group_size,
        )
        if implementation == CompressionImplementation.TRAINING:
            result = MLXMatrixForTraining(
                spec=self,
                weights=weights,
                scales=affine_parameters.scales,
                biases=affine_parameters.biases,
            )
        else:
            packed_int_weights = _mlx_pack_master_weights(
                weights,
                affine_parameters.scales,
                affine_parameters.biases,
                self.group_size,
                self.bits,
            )
            result = MLXMatrixForInference(
                spec=self,
                packed_weights=packed_int_weights,
                scales=affine_parameters.scales,
                biases=affine_parameters.biases,
            )

        return result


class MLXMatrix(EmbeddingMatrix[MLXSpec]):
    scales: Float[Array, "... rows groups"] = field(norm=ParameterNorm.L_INF)
    biases: Float[Array, "... rows groups"] = field(norm=ParameterNorm.L_INF)

    @property
    @abstractmethod
    def _packed_quantized_weights(self) -> Int[Array, "... rows packed_cols"]: ...

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
        expored_data: ExportResults,
        allow_dtype_cast: bool = False,
        *,
        prefix: ParameterPath | None = None,
    ) -> "MLXMatrix": ...

    @abstractmethod
    def switch_implementation(self, implementation: CompressionImplementation) -> "MLXMatrix": ...


class MLXMatrixForTraining(MLXMatrix):
    weights: Float[Array, "... rows cols"] = field(norm=ParameterNorm.SPECTRAL)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.weights.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.weights.dtype

    @property
    def _packed_quantized_weights(self) -> Int[Array, "... rows packed_cols"]:
        return _mlx_pack_master_weights(
            self.weights,
            self.scales,
            self.biases,
            self.spec.group_size,
            self.spec.bits,
        )

    def astype(self, dtype: DTypeLike) -> "MLXMatrixForTraining":
        return MLXMatrixForTraining(
            spec=self.spec,
            weights=self.weights.astype(dtype),
            scales=self.scales.astype(dtype),
            biases=self.biases.astype(dtype),
        )

    def decompress(self) -> Float[Array, "..."]:
        return _mlx_quantize(
            self.weights,
            self.scales,
            self.biases,
            group_size=self.spec.group_size,
            round_fn=partial(deterministic_round_to_unsigned_grid, bits=self.spec.bits),
        )

    def lookup_embedding(
        self,
        index: int | Int[Array, ""],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
    ) -> Float[Array, "... out_channels"]:
        self._raise_if_batched()
        if self.spec.layout != Layout.INPUT_OUTPUT:
            raise ValueError(f"Embedding lookup not supported for layout {self.spec.layout}")
        return _mlx_quantize(
            self.weights[index, :],
            self.scales[index, :],
            self.biases[index, :],
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
        vector: Float[Array, " in_channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
    ) -> Float[Array, "... out_channels"]:
        self._raise_if_batched()
        dequantized_weights = _mlx_quantize(
            self.weights,
            self.scales,
            self.biases,
            group_size=self.spec.group_size,
            round_fn=partial(
                round_to_unsigned_grid,
                bits=self.spec.bits,
                keychain=keychain,
                gradient_estimator=forward_pass_config.gradient_estimator,
            ),
        )
        with use_dot_algorithm_preset(forward_pass_config.precision):
            return self.spec.layout.matmul(dequantized_weights, vector)

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

        packed_weights = expored_data.arrays[prefix / "weights"]
        scales = load_as(self.scales, expored_data.arrays[prefix / "scales"], allow_dtype_cast)
        biases = load_as(self.biases, expored_data.arrays[prefix / "biases"], allow_dtype_cast)
        weights = _mlx_unpack_master_weights(
            packed_weights,
            scales,
            biases,
            self.spec.group_size,
            self.spec.bits,
        )
        weights = load_as(self.weights, weights, allow_dtype_cast=allow_dtype_cast)
        return type(self)(spec=self.spec, weights=weights, scales=scales, biases=biases)

    def switch_implementation(self, implementation: CompressionImplementation) -> MLXMatrix:
        if implementation == CompressionImplementation.TRAINING:
            return self
        return MLXMatrixForInference(
            spec=self.spec,
            packed_weights=self._packed_quantized_weights,
            scales=self.scales,
            biases=self.biases,
        )


class MLXMatrixForInference(MLXMatrix):
    packed_weights: Int[Array, "... rows packed_cols"]
    scales: Float[Array, "... rows groups"]
    biases: Float[Array, "... rows groups"]

    @property
    def shape(self) -> tuple[int, ...]:
        *leading_dims, rows, groups = self.scales.shape
        return (*leading_dims, rows, groups * self.spec.group_size)

    @property
    def dtype(self) -> DTypeLike:
        return self.scales.dtype

    @property
    def _packed_quantized_weights(self) -> Int[Array, "... rows packed_cols"]:
        return self.packed_weights

    def astype(self, dtype: DTypeLike) -> "MLXMatrixForInference":
        return MLXMatrixForInference(
            spec=self.spec,
            packed_weights=self.packed_weights,
            scales=self.scales.astype(dtype),
            biases=self.biases.astype(dtype),
        )

    def load_exported(
        self,
        expored_data: ExportResults,
        allow_dtype_cast: bool = False,
        *,
        prefix: ParameterPath | None = None,
    ) -> MLXMatrix:
        if prefix is None:
            prefix = ParameterPath()
        saved_spec = expored_data.metadata[prefix / "spec"]
        loaded_spec = self.spec.from_json(saved_spec)
        if loaded_spec != self.spec:
            raise ValueError(f"WeightMatrix spec mismatch: expected {self.spec}, got {loaded_spec}")

        packed_weights = load_as(
            self.packed_weights,
            expored_data.arrays[prefix / "weights"],
            allow_dtype_cast=False,
        )
        scales = load_as(self.scales, expored_data.arrays[prefix / "scales"], allow_dtype_cast)
        biases = load_as(self.biases, expored_data.arrays[prefix / "biases"], allow_dtype_cast)
        return MLXMatrixForInference(
            spec=self.spec,
            packed_weights=packed_weights,
            scales=scales,
            biases=biases,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> MLXMatrix:
        if implementation == CompressionImplementation.INFERENCE:
            return self
        weights = _mlx_unpack_master_weights(
            self.packed_weights,
            self.scales,
            self.biases,
            self.spec.group_size,
            self.spec.bits,
        )
        return MLXMatrixForTraining(spec=self.spec, weights=weights, scales=self.scales, biases=self.biases)

    def decompress(self) -> Float[Array, "..."]:
        return _mlx_unpack_master_weights(
            self.packed_weights,
            self.scales,
            self.biases,
            self.spec.group_size,
            self.spec.bits,
        )

    def lookup_embedding(
        self,
        index: int | Int[Array, ""],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),  # noqa: ARG002
    ) -> Float[Array, "... out_channels"]:
        self._raise_if_batched()
        if self.spec.layout != Layout.INPUT_OUTPUT:
            raise ValueError(f"Embedding lookup not supported for layout {self.spec.layout}")
        packed_row = self.packed_weights[index, :]
        weights = unpack_uint8_to_uint(packed_row, self.spec.bits, dtype=self.scales.dtype)
        expanded_scales = expand_last_axis_groups(self.scales[index, :], group_size=self.spec.group_size)
        expanded_biases = expand_last_axis_groups(self.biases[index, :], group_size=self.spec.group_size)
        return weights * expanded_scales + expanded_biases

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),
    ) -> Float[Array, "... out_channels"]:
        self._raise_if_batched()
        weights = _mlx_unpack_master_weights(
            self.packed_weights,
            self.scales,
            self.biases,
            self.spec.group_size,
            self.spec.bits,
        )
        with use_dot_algorithm_preset(forward_pass_config.precision):
            return self.spec.layout.matmul(weights, vector)
