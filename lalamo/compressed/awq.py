from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple, Self

import jax.numpy as jnp
from jax import ShapeDtypeStruct
from jax.lax import stop_gradient
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.exportable import ExportResults
from lalamo.module import Keychain, ParameterNorm, field
from lalamo.utils.dummy_array import dummy_array, preserve_first_input_sharding, supports_dummy_arrays
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.sharding import is_sharded, reshard_as, sharding_of, with_sharding
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
    "AWQMatrix",
    "AWQMatrixForInference",
    "AWQMatrixForTraining",
    "AWQSpec",
]


class AWQAffineParameters(NamedTuple):
    scales: Float[Array, "... groups"]
    zero_points: Float[Array, "... groups"]

    @classmethod
    def from_weights(
        cls,
        weights: Float[Array, "... rows cols"],
        bits: int,
        group_size: int,
    ) -> Self:
        if isinstance(weights, ShapeDtypeStruct):
            grouped_shape = grouped_last_axis_shape(weights.shape, group_size=group_size)
            return cls(
                scales=dummy_array(grouped_shape, weights.dtype, weights.sharding),
                zero_points=dummy_array(grouped_shape, weights.dtype, weights.sharding),
            )

        group_min_max = min_max_within_groups(group_by_last_axis(weights, group_size=group_size))
        scales = scale_from_min_max(group_min_max, bits=bits, dtype=weights.dtype)
        zero_points = jnp.nan_to_num(
            -group_min_max.min,
            nan=0,
            posinf=jnp.finfo(weights.dtype).max,
            neginf=0,
        )
        return cls(scales=scales, zero_points=zero_points)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _awq_master_zero_points_to_int_scale(
    zero_points: Float[Array, "... groups"],
    scales: Float[Array, "... groups"],
) -> Float[Array, "... groups"]:
    return zero_points / scales


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _awq_int_scale_zero_points_to_master(
    int_scale_zero_points: Float[Array, "... groups"],
    scales: Float[Array, "... groups"],
) -> Float[Array, "... groups"]:
    return int_scale_zero_points.astype(scales.dtype) * scales


def _drop_last_axis_sharding(array: Array) -> Array:
    source_sharding = sharding_of(array)
    if not is_sharded(source_sharding):
        return array

    source_partition = tuple(source_sharding.spec)
    output_sharding = NamedSharding(source_sharding.mesh, PartitionSpec(*source_partition[:-1], None))
    return with_sharding(array, output_sharding)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _awq_master_weights_to_int_scale(
    weights: Float[Array, "... cols"],
    scales: Float[Array, "... groups"],
    zero_points: Float[Array, "... groups"],
    group_size: int,
) -> Float[Array, "... rows cols"]:
    expanded_scales = expand_last_axis_groups(scales, group_size=group_size)
    expanded_zero_points = expand_last_axis_groups(zero_points, group_size=group_size)
    return (weights + expanded_zero_points) / expanded_scales


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _awq_quantize(
    weights: Float[Array, "... cols"],
    scales: Float[Array, "... groups"],
    zero_points: Float[Array, "... groups"],
    group_size: int,
    round_fn: Callable[[Float[Array, "..."]], Float[Array, "..."]],
) -> Float[Array, "... rows cols"]:
    expanded_scales = expand_last_axis_groups(scales, group_size=group_size)
    int_scale_zero_points = _awq_master_zero_points_to_int_scale(zero_points, stop_gradient(scales))
    int_zero_points = round_fn(int_scale_zero_points)
    rounded_zero_points = _awq_int_scale_zero_points_to_master(int_zero_points, scales)
    expanded_rounded_zero_points = expand_last_axis_groups(rounded_zero_points, group_size=group_size)
    int_scale_weights = (weights + stop_gradient(expanded_rounded_zero_points)) / stop_gradient(expanded_scales)
    int_weights = round_fn(int_scale_weights)
    expanded_int_zero_points = expand_last_axis_groups(int_zero_points, group_size=group_size)

    return (int_weights - expanded_int_zero_points) * expanded_scales


def _awq_pack_master_zero_points(
    zero_points: Float[Array, "... rows groups"],
    scales: Float[Array, "... rows groups"],
    bits: int,
) -> Int[Array, "... rows packed_groups"]:
    int_scale_zero_points = _awq_master_zero_points_to_int_scale(zero_points, scales)
    int_zero_points = deterministic_round_to_unsigned_grid(int_scale_zero_points, bits=bits)
    int_zero_points = _drop_last_axis_sharding(int_zero_points)
    return pack_uint_to_uint8(int_zero_points, bits)


def _awq_pack_master_weights(
    weights: Float[Array, "... rows cols"],
    scales: Float[Array, "... rows groups"],
    zero_points: Float[Array, "... rows groups"],
    group_size: int,
    bits: int,
) -> Int[Array, "... rows packed_cols"]:
    int_scale_zero_points = _awq_master_zero_points_to_int_scale(zero_points, scales)
    int_zero_points = deterministic_round_to_unsigned_grid(int_scale_zero_points, bits=bits)
    rounded_zero_points = _awq_int_scale_zero_points_to_master(int_zero_points, scales)
    int_scale_weights = _awq_master_weights_to_int_scale(
        weights,
        scales,
        rounded_zero_points,
        group_size,
    )
    int_weights = deterministic_round_to_unsigned_grid(int_scale_weights, bits=bits)
    return pack_uint_to_uint8(int_weights, bits)


def _awq_unpack_master_zero_points(
    packed_zero_points: Int[Array, "... rows packed_groups"],
    scales: Float[Array, "... rows groups"],
    bits: int,
) -> Float[Array, "... rows groups"]:
    int_zero_points = unpack_uint8_to_uint(packed_zero_points, bits=bits, dtype=scales.dtype)
    return _awq_int_scale_zero_points_to_master(int_zero_points, scales)


def _awq_unpack_master_weights(
    packed_weights: Int[Array, "... rows packed_cols"],
    scales: Float[Array, "... rows groups"],
    packed_zero_points: Int[Array, "... rows packed_groups"],
    group_size: int,
    bits: int,
) -> Float[Array, "... rows cols"]:
    int_weights = unpack_uint8_to_uint(packed_weights, bits=bits, dtype=scales.dtype)
    int_zero_points = unpack_uint8_to_uint(packed_zero_points, bits=bits, dtype=scales.dtype)
    expanded_scales = expand_last_axis_groups(scales, group_size=group_size)
    expanded_int_zero_points = expand_last_axis_groups(int_zero_points, group_size=group_size)
    return (int_weights - expanded_int_zero_points) * expanded_scales


@dataclass(frozen=True)
class AWQSpec(WeightMatrixSpec):
    bits: Literal[4, 8]
    group_size: int
    layout: Layout = Layout.OUTPUT_INPUT

    def compress(
        self,
        weights: Float[Array, "... out_channels in_channels"],
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> "AWQMatrix":
        if preconditioner is not None:
            raise ValueError("Preconditioned rounding is not implemented yet.")

        weights = self.layout.convert_weights(weights)
        affine_parameters = AWQAffineParameters.from_weights(
            weights,
            bits=self.bits,
            group_size=self.group_size,
        )
        if implementation == CompressionImplementation.TRAINING:
            result = AWQMatrixForTraining(
                spec=self,
                weights=weights,
                scales=affine_parameters.scales,
                zero_points=affine_parameters.zero_points,
            )
        else:
            packed_weights = _awq_pack_master_weights(
                weights,
                affine_parameters.scales,
                affine_parameters.zero_points,
                self.group_size,
                self.bits,
            )
            packed_zero_points = _awq_pack_master_zero_points(
                affine_parameters.zero_points,
                affine_parameters.scales,
                self.bits,
            )
            result = AWQMatrixForInference(
                spec=self,
                packed_weights=packed_weights,
                scales=affine_parameters.scales,
                packed_zero_points=packed_zero_points,
            )

        return result


class AWQMatrix(EmbeddingMatrix[AWQSpec]):
    scales: Float[Array, "... rows groups"] = field(norm=ParameterNorm.L_INF)

    @property
    @abstractmethod
    def _packed_quantized_weights(self) -> Int[Array, "... rows packed_cols"]: ...

    @property
    @abstractmethod
    def _packed_quantized_zero_points(self) -> Int[Array, "... rows packed_groups"]: ...

    def export(self) -> ExportResults:
        return ExportResults(
            arrays={
                "weights": self._packed_quantized_weights,
                "scales": self.scales,
                "zero_points": self._packed_quantized_zero_points,
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
    ) -> "AWQMatrix": ...

    @abstractmethod
    def switch_implementation(self, implementation: CompressionImplementation) -> "AWQMatrix": ...


class AWQMatrixForTraining(AWQMatrix):
    weights: Float[Array, "... rows cols"] = field(norm=ParameterNorm.SPECTRAL)
    zero_points: Float[Array, "... rows groups"] = field(norm=ParameterNorm.L_INF)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.weights.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.weights.dtype

    @property
    def _packed_quantized_weights(self) -> Int[Array, "... rows packed_cols"]:
        return _awq_pack_master_weights(
            self.weights,
            self.scales,
            self.zero_points,
            self.spec.group_size,
            self.spec.bits,
        )

    @property
    def _packed_quantized_zero_points(self) -> Int[Array, "... rows packed_groups"]:
        return _awq_pack_master_zero_points(
            self.zero_points,
            self.scales,
            self.spec.bits,
        )

    def astype(self, dtype: DTypeLike) -> "AWQMatrixForTraining":
        return AWQMatrixForTraining(
            spec=self.spec,
            weights=self.weights.astype(dtype),
            scales=self.scales.astype(dtype),
            zero_points=self.zero_points.astype(dtype),
        )

    def decompress(self) -> Float[Array, "..."]:
        return _awq_quantize(
            self.weights,
            self.scales,
            self.zero_points,
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
        return _awq_quantize(
            self.weights[index, :],
            self.scales[index, :],
            self.zero_points[index, :],
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
        dequantized_weights = _awq_quantize(
            self.weights,
            self.scales,
            self.zero_points,
            group_size=self.spec.group_size,
            round_fn=partial(
                round_to_unsigned_grid,
                bits=self.spec.bits,
                keychain=keychain,
                gradient_estimator=forward_pass_config.gradient_estimator,
            ),
        )
        with use_dot_algorithm_preset(forward_pass_config.precision):
            result = self.spec.layout.matmul(dequantized_weights, vector)

        return reshard_as(result, vector)

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

        packed_weights = reshard_as(expored_data.arrays[prefix / "weights"], self.weights)
        scales = load_as(self.scales, expored_data.arrays[prefix / "scales"], allow_dtype_cast=allow_dtype_cast)
        packed_zero_points_template = _drop_last_axis_sharding(self.zero_points)
        packed_zero_points = reshard_as(expored_data.arrays[prefix / "zero_points"], packed_zero_points_template)
        weights = _awq_unpack_master_weights(
            packed_weights,
            scales,
            packed_zero_points,
            self.spec.group_size,
            self.spec.bits,
        )
        zero_points = _awq_unpack_master_zero_points(
            packed_zero_points,
            scales,
            self.spec.bits,
        )
        weights = load_as(self.weights, weights, allow_dtype_cast=allow_dtype_cast)
        zero_points = load_as(self.zero_points, zero_points, allow_dtype_cast=allow_dtype_cast)
        return type(self)(spec=self.spec, weights=weights, scales=scales, zero_points=zero_points)

    def switch_implementation(self, implementation: CompressionImplementation) -> AWQMatrix:
        if implementation == CompressionImplementation.TRAINING:
            return self
        return AWQMatrixForInference(
            spec=self.spec,
            packed_weights=self._packed_quantized_weights,
            scales=self.scales,
            packed_zero_points=self._packed_quantized_zero_points,
        )


class AWQMatrixForInference(AWQMatrix):
    packed_weights: Int[Array, "... rows packed_cols"]
    scales: Float[Array, "... rows groups"]
    packed_zero_points: Int[Array, "... rows packed_groups"]

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

    @property
    def _packed_quantized_zero_points(self) -> Int[Array, "... rows packed_groups"]:
        return self.packed_zero_points

    def astype(self, dtype: DTypeLike) -> "AWQMatrixForInference":
        return AWQMatrixForInference(
            spec=self.spec,
            packed_weights=self.packed_weights,
            scales=self.scales.astype(dtype),
            packed_zero_points=self.packed_zero_points,
        )

    def load_exported(
        self,
        expored_data: ExportResults,
        allow_dtype_cast: bool = False,
        *,
        prefix: ParameterPath | None = None,
    ) -> AWQMatrix:
        if prefix is None:
            prefix = ParameterPath()
        saved_spec = expored_data.metadata[prefix / "spec"]
        loaded_spec = WeightMatrixSpec.from_json(saved_spec)
        if loaded_spec != self.spec:
            raise ValueError(f"WeightMatrix spec mismatch: expected {self.spec}, got {loaded_spec}")

        packed_weights = load_as(
            self.packed_weights,
            expored_data.arrays[prefix / "weights"],
            allow_dtype_cast=False,
        )
        scales = load_as(self.scales, expored_data.arrays[prefix / "scales"], allow_dtype_cast=allow_dtype_cast)
        packed_zero_points = load_as(
            self.packed_zero_points,
            expored_data.arrays[prefix / "zero_points"],
            allow_dtype_cast=False,
        )
        return AWQMatrixForInference(
            spec=self.spec,
            packed_weights=packed_weights,
            scales=scales,
            packed_zero_points=packed_zero_points,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> AWQMatrix:
        if implementation == CompressionImplementation.INFERENCE:
            return self
        weights = _awq_unpack_master_weights(
            self.packed_weights,
            self.scales,
            self.packed_zero_points,
            self.spec.group_size,
            self.spec.bits,
        )
        zero_points = _awq_unpack_master_zero_points(
            self.packed_zero_points,
            self.scales,
            self.spec.bits,
        )
        return AWQMatrixForTraining(
            spec=self.spec,
            weights=weights,
            scales=self.scales,
            zero_points=zero_points,
        )

    def decompress(self) -> Float[Array, "..."]:
        return _awq_unpack_master_weights(
            self.packed_weights,
            self.scales,
            self.packed_zero_points,
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
        packed_weights = self.packed_weights[index, :]
        packed_zero_points = self.packed_zero_points[index, :]
        scales = self.scales[index, :]
        int_weights = unpack_uint8_to_uint(packed_weights, self.spec.bits, dtype=scales.dtype)
        int_zero_points = unpack_uint8_to_uint(
            packed_zero_points,
            self.spec.bits,
            dtype=scales.dtype,
        )
        expanded_scales = expand_last_axis_groups(scales, group_size=self.spec.group_size)
        expanded_int_zero_points = expand_last_axis_groups(int_zero_points, group_size=self.spec.group_size)
        return (int_weights - expanded_int_zero_points) * expanded_scales

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),
    ) -> Float[Array, "... out_channels"]:
        self._raise_if_batched()
        weights = _awq_unpack_master_weights(
            self.packed_weights,
            self.scales,
            self.packed_zero_points,
            self.spec.group_size,
            self.spec.bits,
        )
        with use_dot_algorithm_preset(forward_pass_config.precision):
            result = self.spec.layout.matmul(weights, vector)

        return reshard_as(result, vector)
