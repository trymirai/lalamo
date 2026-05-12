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
from lalamo.utils.sharding import (
    make_sharding,
    reshard_as,
    use_out_sharding,
    with_sharding,
)
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
from .utils.gaussian_order_statistics import standard_normal_absmax_squared, standard_normal_range_squared
from .utils.grouping import (
    group_by_last_axis,
    min_max_within_groups,
    scale_from_min_max,
)
from .utils.packing import pack_uint_to_uint8, unpack_uint8_to_uint
from .utils.rounding import deterministic_round_to_unsigned_grid, round_to_unsigned_grid
from .utils.yaqa import yaqa_round_weights

__all__ = [
    "AWQMatrix",
    "AWQMatrixForInference",
    "AWQMatrixForTraining",
    "AWQSpec",
]


class AWQAffineParameters(NamedTuple):
    scales: Float[Array, "*components rows groups"]
    zero_points: Float[Array, "*components rows groups"] | None

    @classmethod
    @supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
    def from_weights(
        cls,
        weights: Float[Array, "*components rows cols"],
        bits: int,
        group_size: int,
        is_symmetric: bool,
    ) -> Self:
        group_min_max = min_max_within_groups(group_by_last_axis(weights, group_size=group_size))
        if is_symmetric:
            finfo = jnp.finfo(weights.dtype)
            max_abs = jnp.maximum(jnp.abs(group_min_max.min), jnp.abs(group_min_max.max))
            scales = jnp.nan_to_num(
                max_abs / ((2 ** (bits - 1)) - 1),
                nan=finfo.eps,
                posinf=finfo.max,
                neginf=finfo.eps,
            )
            zero_points = None
        else:
            scales = scale_from_min_max(group_min_max, bits=bits, dtype=weights.dtype)
            zero_points = jnp.nan_to_num(
                -group_min_max.min,
                nan=0,
                posinf=jnp.finfo(weights.dtype).max,
                neginf=0,
            )
        return cls(scales=scales, zero_points=zero_points)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _awq_master_weights_to_int_scale(
    weights: Float[Array, "... cols"],
    scales: Float[Array, "... groups"],
    int_scale_zero_points: Float[Array, "... groups"] | None,
    group_size: int,
    bits: int,
) -> Float[Array, "... cols"]:
    grouped_weights = weights.reshape(*weights.shape[:-1], weights.shape[-1] // group_size, group_size)
    if int_scale_zero_points is None:
        int_scale_zero_points = 2 ** (bits - 1)
    else:
        int_scale_zero_points = int_scale_zero_points[..., None]

    grouped_int_scale_weights = grouped_weights / scales[..., None] + int_scale_zero_points
    return grouped_int_scale_weights.reshape(weights.shape)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _awq_int_scale_weights_to_master(
    int_scale_weights: Float[Array, "... cols"],
    scales: Float[Array, "... groups"],
    int_scale_zero_points: Float[Array, "... groups"] | None,
    group_size: int,
    bits: int,
) -> Float[Array, "... cols"]:
    grouped_int_scale_weights = int_scale_weights.reshape(
        *int_scale_weights.shape[:-1],
        int_scale_weights.shape[-1] // group_size,
        group_size,
    )
    if int_scale_zero_points is None:
        int_scale_zero_points = 2 ** (bits - 1)
    else:
        int_scale_zero_points = int_scale_zero_points[..., None]

    grouped_weights = (grouped_int_scale_weights - int_scale_zero_points) * scales[..., None]
    return grouped_weights.reshape(int_scale_weights.shape)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _awq_quantize(
    weights: Float[Array, "... cols"],
    scales: Float[Array, "... groups"],
    zero_points: Float[Array, "... groups"] | None,
    group_size: int,
    bits: int,
    round_fn: Callable[[Float[Array, "..."]], Float[Array, "..."]],
) -> Float[Array, "... cols"]:
    if zero_points is None:
        int_scale_zero_points = None
    else:
        int_scale_zero_points = round_fn(zero_points / stop_gradient(scales))

    int_scale_weights = _awq_master_weights_to_int_scale(
        weights,
        stop_gradient(scales),
        stop_gradient(int_scale_zero_points),
        group_size,
        bits,
    )
    int_weights = round_fn(int_scale_weights)
    return _awq_int_scale_weights_to_master(int_weights, scales, int_scale_zero_points, group_size, bits)


@supports_dummy_arrays()
def _awq_pack_master_zero_points(
    zero_points: Float[Array, "*components rows groups"] | None,
    scales: Float[Array, "*components rows groups"],
    bits: int,
) -> UInt8[Array, "*components rows packed_groups"] | None:
    if zero_points is None:
        return None

    int_scale_zero_points = deterministic_round_to_unsigned_grid(zero_points / scales, bits=bits)
    return pack_uint_to_uint8(int_scale_zero_points, bits)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _awq_pack_master_weights(
    weights: Float[Array, "*components rows cols"],
    scales: Float[Array, "*components rows groups"],
    zero_points: Float[Array, "*components rows groups"] | None,
    group_size: int,
    bits: int,
) -> UInt8[Array, "*components rows packed_cols"]:
    if zero_points is None:
        int_scale_zero_points = None
    else:
        int_scale_zero_points = deterministic_round_to_unsigned_grid(zero_points / scales, bits=bits)
    int_scale_weights = _awq_master_weights_to_int_scale(
        weights,
        scales,
        int_scale_zero_points,
        group_size,
        bits,
    )
    int_weights = deterministic_round_to_unsigned_grid(int_scale_weights, bits=bits)
    return pack_uint_to_uint8(int_weights, bits)


def _awq_unpack_master_zero_points(
    packed_zero_points: UInt8[Array, "*components rows packed_groups"] | None,
    scales: Float[Array, "*components rows groups"],
    bits: int,
) -> Float[Array, "*components rows groups"] | None:
    if packed_zero_points is None:
        return None

    int_scale_zero_points = unpack_uint8_to_uint(packed_zero_points, bits=bits, dtype=scales.dtype)
    return int_scale_zero_points.astype(scales.dtype) * scales


def _awq_unpack_master_weights(
    packed_weights: UInt8[Array, "... packed_cols"],
    scales: Float[Array, "... groups"],
    packed_zero_points: UInt8[Array, "... packed_groups"] | None,
    group_size: int,
    bits: int,
) -> Float[Array, "... cols"]:
    int_weights = unpack_uint8_to_uint(packed_weights, bits=bits, dtype=scales.dtype)
    if packed_zero_points is None:
        int_scale_zero_points = None
    else:
        int_scale_zero_points = unpack_uint8_to_uint(packed_zero_points, bits=bits, dtype=scales.dtype)

    return _awq_int_scale_weights_to_master(int_weights, scales, int_scale_zero_points, group_size, bits)


@dataclass(frozen=True)
class AWQSpec(QuantizedSpec):
    bits: Literal[4, 8]
    group_size: int
    is_symmetric: bool = False
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
        if self.is_symmetric:
            zero_point_bits = 0
        else:
            zero_point_bits = self.bits
        return float(self.bits + (16 + zero_point_bits) / self.group_size)

    @property
    def distortion(self) -> float:
        if self.is_symmetric:
            qmax = (2 ** (self.bits - 1)) - 1
            endpoint_correction = (self.group_size - 1) / self.group_size
            return endpoint_correction * standard_normal_absmax_squared(self.group_size) / (12 * qmax**2)

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
        is_sharded: bool = True,
    ) -> "AWQMatrix":
        if preconditioner is not None:
            weights = yaqa_round_weights(weights, preconditioner, self, is_sharded=is_sharded)

        stored_weights = self.layout.from_output_input(weights, is_sharded=is_sharded)
        affine_parameters = AWQAffineParameters.from_weights(
            stored_weights,
            bits=self.bits,
            group_size=self.group_size,
            is_symmetric=self.is_symmetric,
        )
        if implementation == CompressionImplementation.TRAINING:
            result = AWQMatrixForTraining(
                spec=self,
                is_sharded=is_sharded,
                weights=stored_weights,
                scales=affine_parameters.scales,
                zero_points=affine_parameters.zero_points,
            )
        else:
            packed_weights = _awq_pack_master_weights(
                stored_weights,
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
            result = self.from_packed_parameters(
                packed_weights=packed_weights,
                scales=affine_parameters.scales,
                packed_zero_points=packed_zero_points,
                implementation=CompressionImplementation.INFERENCE,
                is_sharded=is_sharded,
            )

        return result

    def from_packed_parameters(
        self,
        *,
        packed_weights: UInt8[Array, "*components rows packed_cols"],
        scales: Float[Array, "*components rows groups"],
        packed_zero_points: UInt8[Array, "*components rows packed_groups"] | None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        is_sharded: bool = True,
    ) -> "AWQMatrix":
        if self.is_symmetric:
            if packed_zero_points is not None:
                raise ValueError("Symmetric AWQ parameters must not include packed zero points.")
        elif packed_zero_points is None:
            raise ValueError("Asymmetric AWQ parameters require packed zero points.")

        weight_partition = self.layout.weight_partition(scales.ndim - 2, is_sharded=is_sharded)
        weight_sharding = make_sharding(weight_partition)

        packed_weights = with_sharding(packed_weights, weight_sharding)
        scales = with_sharding(scales, weight_sharding)
        if packed_zero_points is not None:
            packed_zero_points_sharding = make_sharding((*weight_partition[:-1], None))
            packed_zero_points = with_sharding(packed_zero_points, packed_zero_points_sharding)

        if implementation == CompressionImplementation.INFERENCE:
            return AWQMatrixForInference(
                spec=self,
                is_sharded=is_sharded,
                packed_weights=packed_weights,
                scales=scales,
                packed_zero_points=packed_zero_points,
            )

        weights = _awq_unpack_master_weights(
            packed_weights,
            scales,
            packed_zero_points,
            self.group_size,
            self.bits,
        )
        zero_points = _awq_unpack_master_zero_points(
            packed_zero_points,
            scales,
            self.bits,
        )
        if zero_points is not None:
            zero_points = with_sharding(zero_points, weight_sharding)
        return AWQMatrixForTraining(
            spec=self,
            is_sharded=is_sharded,
            weights=with_sharding(weights, weight_sharding),
            scales=scales,
            zero_points=zero_points,
        )


class AWQMatrix(EmbeddingMatrix[AWQSpec]):
    scales: Float[Array, "*components rows groups"] = field(norm=ParameterNorm.L_INF)

    @property
    @abstractmethod
    def _packed_quantized_weights(self) -> UInt8[Array, "*components rows packed_cols"]: ...

    @property
    @abstractmethod
    def _packed_quantized_zero_points(self) -> UInt8[Array, "*components rows packed_groups"] | None: ...

    def export(self) -> ExportResults:
        arrays = {
            "weights": self._packed_quantized_weights,
            "scales": self.scales,
        }
        packed_zero_points = self._packed_quantized_zero_points
        if packed_zero_points is not None:
            arrays["zero_points"] = packed_zero_points

        return ExportResults(
            arrays=arrays,
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

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(layout=self.spec.layout).compress(self.decompress(), is_sharded=self.is_sharded)


class AWQMatrixForTraining(AWQMatrix):
    weights: Float[Array, "*components rows cols"] = field(norm=ParameterNorm.SPECTRAL)
    zero_points: Float[Array, "*components rows groups"] | None = field(norm=ParameterNorm.L_INF)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.weights.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.weights.dtype

    @property
    def _packed_quantized_weights(self) -> UInt8[Array, "*components rows packed_cols"]:
        return _awq_pack_master_weights(
            self.weights,
            self.scales,
            self.zero_points,
            self.spec.group_size,
            self.spec.bits,
        )

    @property
    def _packed_quantized_zero_points(self) -> UInt8[Array, "*components rows packed_groups"] | None:
        return _awq_pack_master_zero_points(
            self.zero_points,
            self.scales,
            self.spec.bits,
        )

    def astype(self, dtype: DTypeLike) -> "AWQMatrixForTraining":
        zero_points = self.zero_points
        if zero_points is not None:
            zero_points = zero_points.astype(dtype)
        return AWQMatrixForTraining(
            spec=self.spec,
            is_sharded=self.is_sharded,
            weights=self.weights.astype(dtype),
            scales=self.scales.astype(dtype),
            zero_points=zero_points,
        )

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        weights = _awq_quantize(
            self.weights,
            self.scales,
            self.zero_points,
            group_size=self.spec.group_size,
            bits=self.spec.bits,
            round_fn=partial(deterministic_round_to_unsigned_grid, bits=self.spec.bits),
        )
        return self.spec.layout.to_output_input(weights)

    @use_out_sharding((None,))
    def lookup_embedding(
        self,
        index: int | Int[Array, ""],
        *,
        dtype: DTypeLike | None = None,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
    ) -> Float[Array, " out_channels"]:
        self._raise_if_batched()
        if self.spec.layout != Layout.INPUT_OUTPUT:
            raise ValueError(f"Embedding lookup not supported for layout {self.spec.layout}")
        if dtype is None:
            dtype = self.dtype
        zero_points = self.zero_points
        if zero_points is not None:
            zero_points = zero_points[index, :].astype(dtype)
        return _awq_quantize(
            self.weights[index, :].astype(dtype),
            self.scales[index, :].astype(dtype),
            zero_points,
            group_size=self.spec.group_size,
            bits=self.spec.bits,
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
        zero_points = self.zero_points
        if zero_points is not None:
            zero_points = zero_points.astype(vector.dtype)
        dequantized_weights = _awq_quantize(
            self.weights.astype(vector.dtype),
            self.scales.astype(vector.dtype),
            zero_points,
            group_size=self.spec.group_size,
            bits=self.spec.bits,
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
            result = layout.matmul(dequantized_weights, vector)

        return reshard_as(result, vector)

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
            self._packed_quantized_weights,
            expored_data.arrays[prefix / "weights"],
            allow_dtype_cast=False,
        )
        scales = load_as(self.scales, expored_data.arrays[prefix / "scales"], allow_dtype_cast=allow_dtype_cast)
        packed_zero_points = self._packed_quantized_zero_points
        if packed_zero_points is not None:
            packed_zero_points = load_as(
                packed_zero_points,
                expored_data.arrays[prefix / "zero_points"],
                allow_dtype_cast=False,
            )
        return self.spec.from_packed_parameters(
            packed_weights=packed_weights,
            scales=scales,
            packed_zero_points=packed_zero_points,
            implementation=CompressionImplementation.TRAINING,
            is_sharded=self.is_sharded,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> AWQMatrix:
        if implementation == CompressionImplementation.TRAINING:
            return self
        return self.spec.from_packed_parameters(
            packed_weights=self._packed_quantized_weights,
            scales=self.scales,
            packed_zero_points=self._packed_quantized_zero_points,
            implementation=CompressionImplementation.INFERENCE,
            is_sharded=self.is_sharded,
        )


class AWQMatrixForInference(AWQMatrix):
    packed_weights: UInt8[Array, "*components rows packed_cols"]
    scales: Float[Array, "*components rows groups"]
    packed_zero_points: UInt8[Array, "*components rows packed_groups"] | None

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

    @property
    def _packed_quantized_zero_points(self) -> UInt8[Array, "*components rows packed_groups"] | None:
        return self.packed_zero_points

    def astype(self, dtype: DTypeLike) -> "AWQMatrixForInference":
        return AWQMatrixForInference(
            spec=self.spec,
            is_sharded=self.is_sharded,
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
        packed_zero_points = self._packed_quantized_zero_points
        if packed_zero_points is not None:
            packed_zero_points = load_as(
                packed_zero_points,
                expored_data.arrays[prefix / "zero_points"],
                allow_dtype_cast=False,
            )
        return self.spec.from_packed_parameters(
            packed_weights=packed_weights,
            scales=scales,
            packed_zero_points=packed_zero_points,
            implementation=CompressionImplementation.INFERENCE,
            is_sharded=self.is_sharded,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> AWQMatrix:
        if implementation == CompressionImplementation.INFERENCE:
            return self
        return self.spec.from_packed_parameters(
            packed_weights=self.packed_weights,
            scales=self.scales,
            packed_zero_points=self.packed_zero_points,
            implementation=CompressionImplementation.TRAINING,
            is_sharded=self.is_sharded,
        )

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        weights = _awq_unpack_master_weights(
            self.packed_weights,
            self.scales,
            self.packed_zero_points,
            self.spec.group_size,
            self.spec.bits,
        )
        return self.spec.layout.to_output_input(weights)

    @use_out_sharding((None,))
    def lookup_embedding(
        self,
        index: int | Int[Array, ""],
        *,
        dtype: DTypeLike | None = None,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),  # noqa: ARG002
    ) -> Float[Array, " out_channels"]:
        self._raise_if_batched()
        if self.spec.layout != Layout.INPUT_OUTPUT:
            raise ValueError(f"Embedding lookup not supported for layout {self.spec.layout}")
        if dtype is None:
            dtype = self.dtype
        packed_zero_points = self.packed_zero_points
        if packed_zero_points is not None:
            packed_zero_points = packed_zero_points[index, :]
        return _awq_unpack_master_weights(
            self.packed_weights[index, :],
            self.scales[index, :].astype(dtype),
            packed_zero_points,
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
        weights = _awq_unpack_master_weights(
            self.packed_weights,
            self.scales.astype(vector.dtype),
            self.packed_zero_points,
            self.spec.group_size,
            self.spec.bits,
        )
        layout = self.spec.layout
        if transposed:
            layout = layout.transpose()
        with use_dot_algorithm_preset(forward_pass_config.precision):
            result = layout.matmul(weights, vector)

        return reshard_as(result, vector)
