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
    ShardingConfig,
    lookup_sharded_indices,
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
from .utils.yaqa import yaqa_round_blockwise

__all__ = [
    "IntMatrix",
    "IntMatrixForInference",
    "IntMatrixForTraining",
    "IntSpec",
]


class IntAffineParameters(NamedTuple):
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
def _master_weights_to_int_scale_weights(
    weights: Float[Array, "... cols"],
    scales: Float[Array, "... groups"],
    int_scale_zero_points: Float[Array, "... groups"] | None,
    group_size: int,
    bits: int,
) -> Float[Array, "... cols"]:
    *leading_dims, num_cols = weights.shape
    grouped_weights = weights.reshape(*leading_dims, num_cols // group_size, group_size)
    if int_scale_zero_points is None:
        int_scale_zero_points = 2 ** (bits - 1)
    else:
        int_scale_zero_points = int_scale_zero_points[..., None]

    grouped_int_scale_weights = grouped_weights / scales[..., None] + int_scale_zero_points
    return grouped_int_scale_weights.reshape(weights.shape)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _integer_scaled_weights_to_master_weights(
    int_scale_weights: Float[Array, "... cols"],
    scales: Float[Array, "... groups"],
    int_scale_zero_points: Float[Array, "... groups"] | None,
    group_size: int,
    bits: int,
) -> Float[Array, "... cols"]:
    *leading_dims, num_cols = int_scale_weights.shape
    grouped_int_scale_weights = int_scale_weights.reshape(
        *leading_dims,
        num_cols // group_size,
        group_size,
    )
    if int_scale_zero_points is None:
        int_scale_zero_points = 2 ** (bits - 1)
    else:
        int_scale_zero_points = int_scale_zero_points[..., None]

    grouped_weights = (grouped_int_scale_weights - int_scale_zero_points) * scales[..., None]
    return grouped_weights.reshape(int_scale_weights.shape)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _quantize(
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

    int_scale_weights = _master_weights_to_int_scale_weights(
        weights,
        stop_gradient(scales),
        stop_gradient(int_scale_zero_points),
        group_size,
        bits,
    )
    int_weights = round_fn(int_scale_weights)
    return _integer_scaled_weights_to_master_weights(int_weights, scales, int_scale_zero_points, group_size, bits)


@supports_dummy_arrays()
def _master_zero_points_to_packed_zero_points(
    zero_points: Float[Array, "*components rows groups"] | None,
    scales: Float[Array, "*components rows groups"],
    bits: int,
    *,
    sharding_config: ShardingConfig,
) -> UInt8[Array, "*components rows packed_groups"] | None:
    if zero_points is None:
        return None

    int_scale_zero_points = deterministic_round_to_unsigned_grid(zero_points / scales, bits=bits)
    return pack_uint_to_uint8(int_scale_zero_points, bits, sharding_config=sharding_config)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _master_weights_to_packed_weights(
    weights: Float[Array, "*components rows cols"],
    scales: Float[Array, "*components rows groups"],
    zero_points: Float[Array, "*components rows groups"] | None,
    group_size: int,
    bits: int,
    *,
    sharding_config: ShardingConfig,
) -> UInt8[Array, "*components rows packed_cols"]:
    if zero_points is None:
        int_scale_zero_points = None
    else:
        int_scale_zero_points = deterministic_round_to_unsigned_grid(zero_points / scales, bits=bits)
    int_scale_weights = _master_weights_to_int_scale_weights(
        weights,
        scales,
        int_scale_zero_points,
        group_size,
        bits,
    )
    int_weights = deterministic_round_to_unsigned_grid(int_scale_weights, bits=bits)
    return pack_uint_to_uint8(int_weights, bits, sharding_config=sharding_config)


def _packed_zero_points_to_master_zero_points(
    packed_zero_points: UInt8[Array, "*components rows packed_groups"] | None,
    scales: Float[Array, "*components rows groups"],
    bits: int,
) -> Float[Array, "*components rows groups"] | None:
    if packed_zero_points is None:
        return None

    *_, num_groups = scales.shape
    int_scale_zero_points = unpack_uint8_to_uint(
        packed_zero_points,
        bits=bits,
        dtype=scales.dtype,
        unpacked_last_axis_dim=num_groups,
    )
    return int_scale_zero_points.astype(scales.dtype) * scales


def _packed_weights_to_master_weights(
    packed_weights: UInt8[Array, "... packed_cols"],
    scales: Float[Array, "... groups"],
    packed_zero_points: UInt8[Array, "... packed_groups"] | None,
    group_size: int,
    bits: int,
) -> Float[Array, "... cols"]:
    int_weights = unpack_uint8_to_uint(
        packed_weights,
        bits=bits,
        dtype=scales.dtype,
        unpacked_last_axis_dim=scales.shape[-1] * group_size,
    )
    if packed_zero_points is None:
        int_scale_zero_points = None
    else:
        *_, num_groups = scales.shape
        int_scale_zero_points = unpack_uint8_to_uint(
            packed_zero_points,
            bits=bits,
            dtype=scales.dtype,
            unpacked_last_axis_dim=num_groups,
        )

    return _integer_scaled_weights_to_master_weights(int_weights, scales, int_scale_zero_points, group_size, bits)


@dataclass(frozen=True)
class IntSpec(QuantizedSpec):
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
        return float(self.bits + (32 + zero_point_bits) / self.group_size)

    @property
    def distortion(self) -> float:
        if self.is_symmetric:
            qmax = (2 ** (self.bits - 1)) - 1
            endpoint_correction = (self.group_size - 1) / self.group_size
            return endpoint_correction * standard_normal_absmax_squared(self.group_size) / (12 * qmax**2)

        qmax = (2**self.bits) - 1
        return standard_normal_range_squared(self.group_size) / (12 * qmax**2)

    def compress(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        sharding_config: ShardingConfig,
        is_sharded: bool = True,
    ) -> "IntMatrix":
        if preconditioner is not None:
            weights = yaqa_round_blockwise(
                weights,
                preconditioner,
                self,
                sharding_config=sharding_config,
            )

        weight_axes = self.layout.weight_partition(weights.ndim - 2, is_sharded=is_sharded)
        weight_sharding = sharding_config.resolve_sharding(weight_axes)
        stored_weights = self.layout.from_output_input(weights, sharding=weight_sharding)
        affine_parameters = IntAffineParameters.from_weights(
            stored_weights,
            bits=self.bits,
            group_size=self.group_size,
            is_symmetric=self.is_symmetric,
        )
        if implementation == CompressionImplementation.TRAINING:
            result = IntMatrixForTraining(
                spec=self,
                sharding_config=sharding_config,
                is_sharded=is_sharded,
                master_weights=stored_weights,
                scales=affine_parameters.scales,
                master_zero_points=affine_parameters.zero_points,
            )
        else:
            packed_weights = _master_weights_to_packed_weights(
                stored_weights,
                affine_parameters.scales,
                affine_parameters.zero_points,
                self.group_size,
                self.bits,
                sharding_config=sharding_config,
            )
            packed_zero_points = _master_zero_points_to_packed_zero_points(
                affine_parameters.zero_points,
                affine_parameters.scales,
                self.bits,
                sharding_config=sharding_config,
            )
            result = self.from_packed_parameters(
                packed_weights=packed_weights,
                scales=affine_parameters.scales,
                packed_zero_points=packed_zero_points,
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
        packed_zero_points: UInt8[Array, "*components rows packed_groups"] | None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        sharding_config: ShardingConfig,
        is_sharded: bool = True,
    ) -> "IntMatrix":
        if self.is_symmetric:
            if packed_zero_points is not None:
                raise ValueError("Symmetric Int parameters must not include packed zero points.")
        elif packed_zero_points is None:
            raise ValueError("Asymmetric Int parameters require packed zero points.")

        weight_axes = self.layout.weight_partition(scales.ndim - 2, is_sharded=is_sharded)
        *scale_axes, _grouped_axis = weight_axes
        weight_sharding = sharding_config.resolve_sharding(weight_axes)
        packed_zero_points_sharding = sharding_config.resolve_sharding((*scale_axes, None))

        packed_weights = with_sharding(packed_weights, weight_sharding)
        scales = with_sharding(scales, weight_sharding)
        if packed_zero_points is not None:
            packed_zero_points = with_sharding(packed_zero_points, packed_zero_points_sharding)

        if implementation == CompressionImplementation.INFERENCE:
            return IntMatrixForInference(
                spec=self,
                sharding_config=sharding_config,
                is_sharded=is_sharded,
                packed_weights=packed_weights,
                scales=scales,
                packed_zero_points=packed_zero_points,
            )

        weights = _packed_weights_to_master_weights(
            packed_weights,
            scales,
            packed_zero_points,
            self.group_size,
            self.bits,
        )
        zero_points = _packed_zero_points_to_master_zero_points(
            packed_zero_points,
            scales,
            self.bits,
        )
        if zero_points is not None:
            zero_points = with_sharding(zero_points, weight_sharding)
        return IntMatrixForTraining(
            spec=self,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
            master_weights=with_sharding(weights, weight_sharding),
            scales=scales,
            master_zero_points=zero_points,
        )


class IntMatrix(EmbeddingMatrix[IntSpec]):
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
        exported_data: ExportResults,
        *,
        prefix: ParameterPath | None = None,
    ) -> "IntMatrix": ...

    @abstractmethod
    def switch_implementation(self, implementation: CompressionImplementation) -> "IntMatrix": ...

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(layout=self.spec.layout).compress(
            self.decompress(),
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
        )


class IntMatrixForTraining(IntMatrix):
    master_weights: Float[Array, "*components rows cols"] = field(norm=ParameterNorm.SPECTRAL)
    master_zero_points: Float[Array, "*components rows groups"] | None = field(norm=ParameterNorm.L_INF)

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
            self.master_zero_points,
            self.spec.group_size,
            self.spec.bits,
            sharding_config=self.sharding_config,
        )

    @property
    def _packed_quantized_zero_points(self) -> UInt8[Array, "*components rows packed_groups"] | None:
        return _master_zero_points_to_packed_zero_points(
            self.master_zero_points,
            self.scales,
            self.spec.bits,
            sharding_config=self.sharding_config,
        )

    def astype(self, dtype: DTypeLike) -> "IntMatrixForTraining":
        zero_points = self.master_zero_points
        if zero_points is not None:
            zero_points = zero_points.astype(dtype)
        return IntMatrixForTraining(
            spec=self.spec,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
            master_weights=self.master_weights.astype(dtype),
            scales=self.scales.astype(dtype),
            master_zero_points=zero_points,
        )

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        weights = _quantize(
            self.master_weights,
            self.scales,
            self.master_zero_points,
            group_size=self.spec.group_size,
            bits=self.spec.bits,
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
        zero_points = self.master_zero_points
        if zero_points is not None:
            zero_points = lookup_sharded_indices(zero_points, row_index).astype(dtype)
        weights = lookup_sharded_indices(self.master_weights, row_index).astype(dtype)
        return _quantize(
            weights,
            lookup_sharded_indices(self.scales, row_index).astype(dtype),
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
        zero_points = self.master_zero_points
        if zero_points is not None:
            zero_points = zero_points.astype(vector.dtype)
        dequantized_weights = _quantize(
            self.master_weights.astype(vector.dtype),
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
            return layout.matmul(dequantized_weights, vector)

    def load_exported(
        self,
        exported_data: ExportResults,
        *,
        prefix: ParameterPath | None = None,
    ) -> IntMatrix:
        if prefix is None:
            prefix = ParameterPath()
        saved_spec = exported_data.metadata[prefix / "spec"]
        loaded_spec = WeightMatrixSpec.from_json(saved_spec)
        if loaded_spec != self.spec:
            raise ValueError(f"WeightMatrix spec mismatch: expected {self.spec}, got {loaded_spec}")

        packed_weights = load_as(
            self._packed_quantized_weights,
            exported_data.arrays[prefix / "weights"],
        )
        scales = load_as(
            self.scales,
            exported_data.arrays[prefix / "scales"],
        )
        packed_zero_points = self._packed_quantized_zero_points
        if packed_zero_points is not None:
            packed_zero_points = load_as(
                packed_zero_points,
                exported_data.arrays[prefix / "zero_points"],
            )
        return self.spec.from_packed_parameters(
            packed_weights=packed_weights,
            scales=scales,
            packed_zero_points=packed_zero_points,
            implementation=CompressionImplementation.TRAINING,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> IntMatrix:
        if implementation == CompressionImplementation.TRAINING:
            return self
        return self.spec.from_packed_parameters(
            packed_weights=self._packed_quantized_weights,
            scales=self.scales,
            packed_zero_points=self._packed_quantized_zero_points,
            implementation=CompressionImplementation.INFERENCE,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
        )


class IntMatrixForInference(IntMatrix):
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

    def astype(self, dtype: DTypeLike) -> "IntMatrixForInference":
        return IntMatrixForInference(
            spec=self.spec,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
            packed_weights=self.packed_weights,
            scales=self.scales.astype(dtype),
            packed_zero_points=self.packed_zero_points,
        )

    def load_exported(
        self,
        exported_data: ExportResults,
        *,
        prefix: ParameterPath | None = None,
    ) -> IntMatrix:
        if prefix is None:
            prefix = ParameterPath()
        saved_spec = exported_data.metadata[prefix / "spec"]
        loaded_spec = WeightMatrixSpec.from_json(saved_spec)
        if loaded_spec != self.spec:
            raise ValueError(f"WeightMatrix spec mismatch: expected {self.spec}, got {loaded_spec}")

        packed_weights = load_as(
            self.packed_weights,
            exported_data.arrays[prefix / "weights"],
        )
        scales = load_as(
            self.scales,
            exported_data.arrays[prefix / "scales"],
        )
        packed_zero_points = self._packed_quantized_zero_points
        if packed_zero_points is not None:
            packed_zero_points = load_as(
                packed_zero_points,
                exported_data.arrays[prefix / "zero_points"],
            )
        return IntMatrixForInference(
            spec=self.spec,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
            packed_weights=packed_weights,
            scales=scales,
            packed_zero_points=packed_zero_points,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> IntMatrix:
        if implementation == CompressionImplementation.INFERENCE:
            return self
        return self.spec.from_packed_parameters(
            packed_weights=self.packed_weights,
            scales=self.scales,
            packed_zero_points=self.packed_zero_points,
            implementation=CompressionImplementation.TRAINING,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
        )

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        weights = _packed_weights_to_master_weights(
            self.packed_weights,
            self.scales,
            self.packed_zero_points,
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
        packed_zero_points = self.packed_zero_points
        if packed_zero_points is not None:
            packed_zero_points = lookup_sharded_indices(packed_zero_points, row_index)
        packed_weights = lookup_sharded_indices(self.packed_weights, row_index)
        return _packed_weights_to_master_weights(
            packed_weights,
            lookup_sharded_indices(self.scales, row_index).astype(dtype),
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
        weights = _packed_weights_to_master_weights(
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
            return layout.matmul(weights, vector)
