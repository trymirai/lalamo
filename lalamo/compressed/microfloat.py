from abc import abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
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
from lalamo.utils.sharding import lookup_sharded_indices, make_sharding, with_sharding
from lalamo.utils.surgery import load_as
from lalamo.weight_matrix import (
    CompressionImplementation,
    EmbeddingMatrix,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    GradientEstimator,
    Layout,
    MatmulConfig,
    WeightMatrixSpec,
)

from .data.distortion import distortion_estimate
from .quantized_spec import QuantizedSpec
from .utils.grouping import group_by_last_axis
from .utils.packing import pack_uint_to_uint8, unpack_uint8_to_uint
from .utils.rounding import (
    deterministic_round_to_minifloat,
    e8m0_scale_values,
    lut_values_at,
    pack_e4m3_scales,
    pack_e8m0_scales,
    round_to_minifloat,
)
from .utils.yaqa import yaqa_round_weights

__all__ = [
    "MicrofloatMatrix",
    "MicrofloatMatrixForInference",
    "MicrofloatMatrixForTraining",
    "MicrofloatScaleMode",
    "MicrofloatSpec",
]


class MicrofloatScaleMode(StrEnum):
    MXFP4 = "mxfp4"
    NVFP4 = "nvfp4"


def _codebook(dtype: DTypeLike) -> Float[Array, " levels"]:
    positive_values = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    negative_values = tuple(-value for value in positive_values)
    return jnp.array(positive_values + negative_values, dtype=dtype)


class MicrofloatParameters(NamedTuple):
    scales: Float[Array, "*components rows groups"]
    global_scale: Float[Array, "*components"]

    @classmethod
    @supports_dummy_arrays()
    def from_weights(
        cls,
        weights: Float[Array, "*components rows cols"],
        group_size: int,
        scale_mode: MicrofloatScaleMode,
    ) -> Self:
        group_absmax = jnp.max(jnp.abs(group_by_last_axis(weights, group_size=group_size)), axis=-1)
        matrix_absmax = jnp.max(jnp.abs(weights), axis=(-2, -1))
        if scale_mode == MicrofloatScaleMode.MXFP4:
            return cls(
                scales=e8m0_scale_values(pack_e8m0_scales(group_absmax / 4), weights.dtype),
                global_scale=jnp.ones_like(matrix_absmax),
            )

        scale_max = jnp.asarray(jnp.finfo(jnp.float8_e4m3fn).max, dtype=matrix_absmax.dtype) * 6
        global_scale = jnp.where(matrix_absmax == 0, 1, matrix_absmax / scale_max)
        return cls(
            scales=pack_e4m3_scales(group_absmax / (6 * global_scale[..., None, None])).astype(weights.dtype),
            global_scale=global_scale,
        )


def _pack_scales(
    master_scales: Float[Array, "... groups"],
    scale_mode: MicrofloatScaleMode,
) -> UInt8[Array, "... groups"] | Float[Array, "... groups"]:
    if scale_mode == MicrofloatScaleMode.MXFP4:
        return pack_e8m0_scales(master_scales)
    return pack_e4m3_scales(master_scales)


def _unpack_scales(
    packed_scales: UInt8[Array, "... groups"] | Float[Array, "... groups"],
    scale_mode: MicrofloatScaleMode,
    dtype: DTypeLike,
) -> Float[Array, "... groups"]:
    if scale_mode == MicrofloatScaleMode.MXFP4:
        return e8m0_scale_values(packed_scales, dtype)
    return packed_scales.astype(dtype)


def _master_scales_to_scale_values(
    master_scales: Float[Array, "... groups"],
    scale_mode: MicrofloatScaleMode,
    dtype: DTypeLike,
) -> Float[Array, "... groups"]:
    if scale_mode == MicrofloatScaleMode.MXFP4:
        clipped_scales = jnp.maximum(master_scales.astype(jnp.float32), jnp.float32(2.0**-127))
        return deterministic_round_to_minifloat(clipped_scales, dtype=jnp.float8_e8m0fnu).astype(dtype)
    return pack_e4m3_scales(master_scales).astype(dtype)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _master_weights_to_quantized_weights(
    master_weights: Float[Array, "... cols"],
    scale_values: Float[Array, "... groups"],
    *,
    group_size: int,
    keychain: Keychain | None = None,
    gradient_estimator: GradientEstimator = GradientEstimator.DETERMINISTIC_ROUNDING,
) -> Float[Array, "... cols"]:
    safe_scale_values = jnp.where(scale_values == 0, 1, scale_values)
    grouped_weights = group_by_last_axis(master_weights, group_size=group_size)
    normalized_weights = grouped_weights / stop_gradient(safe_scale_values)[..., None]
    grouped_values = round_to_minifloat(
        normalized_weights,
        dtype=jnp.float4_e2m1fn,
        keychain=keychain,
        gradient_estimator=gradient_estimator,
    )
    grouped_weights = grouped_values * scale_values[..., None]
    *leading_dims, groups, group_width = grouped_weights.shape
    return grouped_weights.reshape(*leading_dims, groups * group_width)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _master_weights_to_packed_weights(
    master_weights: Float[Array, "*components rows cols"],
    master_scales: Float[Array, "*components rows groups"],
    *,
    global_scale: Float[Array, "*components"],
    group_size: int,
    scale_mode: MicrofloatScaleMode,
) -> UInt8[Array, "*components rows packed_cols"]:
    packed_scales = _pack_scales(master_scales, scale_mode)
    scale_values = (
        _unpack_scales(packed_scales, scale_mode, master_weights.dtype)
        * global_scale.astype(master_weights.dtype)[..., None, None]
    )
    codebook = _codebook(master_weights.dtype)
    safe_scale_values = jnp.where(scale_values == 0, 1, scale_values)
    normalized_weights = group_by_last_axis(master_weights, group_size=group_size) / safe_scale_values[..., None]
    quantized_values = round_to_minifloat(
        normalized_weights,
        dtype=jnp.float4_e2m1fn,
        keychain=None,
        gradient_estimator=GradientEstimator.DETERMINISTIC_ROUNDING,
    )
    grouped_indices = jnp.argmin(jnp.abs(quantized_values[..., None] - codebook), axis=-1).astype(jnp.uint8)
    *leading_dims, groups, group_width = grouped_indices.shape
    unpacked_indices = grouped_indices.reshape(*leading_dims, groups * group_width)
    return pack_uint_to_uint8(unpacked_indices, 4)


def _packed_weights_to_master_weights(
    packed_weights: UInt8[Array, "... packed_cols"],
    packed_scales: UInt8[Array, "... groups"] | Float[Array, "... groups"],
    *,
    global_scale: Float[Array, "*components"],
    dtype: DTypeLike,
    group_size: int,
    scale_mode: MicrofloatScaleMode,
) -> Float[Array, "... cols"]:
    scale_values = _unpack_scales(packed_scales, scale_mode, dtype) * global_scale.astype(dtype)
    *leading_dims, groups = packed_scales.shape
    indices = unpack_uint8_to_uint(packed_weights, bits=4, unpacked_last_axis_dim=groups * group_size)
    grouped_indices = indices.reshape(*leading_dims, groups, group_size)
    grouped_values = lut_values_at(grouped_indices, _codebook(dtype))
    grouped_weights = grouped_values * scale_values[..., None]
    *leading_dims, groups, group_width = grouped_weights.shape
    return grouped_weights.reshape(*leading_dims, groups * group_width)


@dataclass(frozen=True)
class MicrofloatSpec(QuantizedSpec):
    bits: Literal[4] = 4
    group_size: int = 32
    scale_mode: MicrofloatScaleMode = MicrofloatScaleMode.MXFP4
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
        return self.bits + 8 / self.group_size

    @cached_property
    def distortion(self) -> float:
        return distortion_estimate(
            format_name="microfloat",
            bits=self.bits,
            group_size=self.group_size,
            scale_mode=self.scale_mode.value,
        )

    def compress(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        is_sharded: bool = True,
    ) -> "MicrofloatMatrix":
        if preconditioner is not None:
            weights = yaqa_round_weights(weights, preconditioner, self, is_sharded=is_sharded)

        stored_weights = self.layout.from_output_input(weights, is_sharded=is_sharded)
        parameters = MicrofloatParameters.from_weights(stored_weights, self.group_size, self.scale_mode)
        weight_partition = self.layout.weight_partition(parameters.scales.ndim - 2, is_sharded=is_sharded)
        parameter_sharding = make_sharding((*weight_partition[:-1], None))
        master_scales = with_sharding(parameters.scales, parameter_sharding)
        global_scale = with_sharding(parameters.global_scale, make_sharding(weight_partition[:-2]))
        if implementation == CompressionImplementation.TRAINING:
            return MicrofloatMatrixForTraining(
                spec=self,
                is_sharded=is_sharded,
                master_weights=stored_weights,
                master_scales=master_scales,
                global_scale=global_scale,
            )

        return self.from_packed_parameters(
            packed_weights=_master_weights_to_packed_weights(
                stored_weights,
                master_scales,
                global_scale=global_scale,
                group_size=self.group_size,
                scale_mode=self.scale_mode,
            ),
            packed_scales=_pack_scales(master_scales, self.scale_mode),
            global_scale=global_scale,
            dtype=stored_weights.dtype,
            implementation=CompressionImplementation.INFERENCE,
            is_sharded=is_sharded,
        )

    def from_packed_parameters(
        self,
        *,
        packed_weights: UInt8[Array, "*components rows packed_cols"],
        packed_scales: UInt8[Array, "*components rows groups"] | Float[Array, "*components rows groups"],
        global_scale: Float[Array, "*components"],
        dtype: DTypeLike = jnp.float32,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        is_sharded: bool = True,
    ) -> "MicrofloatMatrix":
        weight_partition = self.layout.weight_partition(packed_scales.ndim - 2, is_sharded=is_sharded)
        parameter_sharding = make_sharding((*weight_partition[:-1], None))
        packed_weights = with_sharding(packed_weights, parameter_sharding)
        packed_scales = with_sharding(packed_scales, parameter_sharding)
        global_scale = with_sharding(global_scale, make_sharding(weight_partition[:-2]))

        if implementation == CompressionImplementation.INFERENCE:
            return MicrofloatMatrixForInference(
                spec=self,
                is_sharded=is_sharded,
                dtype_=jnp.dtype(dtype),
                packed_weights=packed_weights,
                packed_scales=packed_scales,
                global_scale=global_scale,
            )

        weights = _packed_weights_to_master_weights(
            packed_weights,
            packed_scales,
            global_scale=global_scale[..., None, None],
            dtype=dtype,
            group_size=self.group_size,
            scale_mode=self.scale_mode,
        )
        return MicrofloatMatrixForTraining(
            spec=self,
            is_sharded=is_sharded,
            master_weights=with_sharding(weights, make_sharding(weight_partition)),
            master_scales=with_sharding(_unpack_scales(packed_scales, self.scale_mode, dtype), parameter_sharding),
            global_scale=global_scale,
        )


class MicrofloatMatrix(EmbeddingMatrix[MicrofloatSpec]):
    global_scale: Float[Array, "*components"] = field(norm=ParameterNorm.L_INF)

    @property
    @abstractmethod
    def _packed_weights(self) -> UInt8[Array, "*components rows packed_cols"]: ...

    @property
    @abstractmethod
    def _packed_scales(self) -> UInt8[Array, "*components rows groups"] | Float[Array, "*components rows groups"]: ...

    @abstractmethod
    def _weights_for_forward(
        self,
        dtype: DTypeLike,
        keychain: Keychain | None,
        forward_pass_config: MatmulConfig,
        index: int | Int[Array, "*batch"] | None = None,
    ) -> Float[Array, "... cols"]: ...

    def export(self) -> ExportResults:
        arrays = {
            "weights": self._packed_weights,
            "scales": self._packed_scales,
            "global_scale": self.global_scale,
        }
        return ExportResults(arrays=arrays, metadata={"spec": self.spec.to_json()})

    def _load_exported(
        self,
        expored_data: ExportResults,
        allow_dtype_cast: bool = False,
        *,
        implementation: CompressionImplementation,
        prefix: ParameterPath | None = None,
    ) -> "MicrofloatMatrix":
        if prefix is None:
            prefix = ParameterPath()
        loaded_spec = WeightMatrixSpec.from_json(expored_data.metadata[prefix / "spec"])
        if loaded_spec != self.spec:
            raise ValueError(f"WeightMatrix spec mismatch: expected {self.spec}, got {loaded_spec}")
        return self.spec.from_packed_parameters(
            packed_weights=load_as(
                self._packed_weights,
                expored_data.arrays[prefix / "weights"],
                allow_dtype_cast=False,
            ),
            packed_scales=load_as(self._packed_scales, expored_data.arrays[prefix / "scales"], allow_dtype_cast=False),
            global_scale=load_as(
                self.global_scale,
                expored_data.arrays[prefix / "global_scale"],
                allow_dtype_cast=allow_dtype_cast,
            ),
            dtype=self.dtype,
            implementation=implementation,
            is_sharded=self.is_sharded,
        )

    def _from_packed_parameters(self, implementation: CompressionImplementation) -> "MicrofloatMatrix":
        return self.spec.from_packed_parameters(
            packed_weights=self._packed_weights,
            packed_scales=self._packed_scales,
            global_scale=self.global_scale,
            dtype=self.dtype,
            implementation=implementation,
            is_sharded=self.is_sharded,
        )

    @abstractmethod
    def load_exported(
        self,
        expored_data: ExportResults,
        allow_dtype_cast: bool = False,
        *,
        prefix: ParameterPath | None = None,
    ) -> "MicrofloatMatrix": ...

    @abstractmethod
    def switch_implementation(self, implementation: CompressionImplementation) -> "MicrofloatMatrix": ...

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(layout=self.spec.layout).compress(self.decompress(), is_sharded=self.is_sharded)

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        return self.spec.layout.to_output_input(self._weights_for_forward(self.dtype, None, MatmulConfig()))

    def lookup_embedding(
        self,
        index: int | Int[Array, "*batch"],
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
        return self._weights_for_forward(dtype, keychain, forward_pass_config, index)

    def dot(
        self,
        vector: Float[Array, " source_channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Float[Array, " target_channels"]:
        self._raise_if_batched()
        layout = self.spec.layout
        if transposed:
            layout = layout.transpose()
        with use_dot_algorithm_preset(forward_pass_config.precision):
            return layout.matmul(self._weights_for_forward(vector.dtype, keychain, forward_pass_config), vector)


class MicrofloatMatrixForTraining(MicrofloatMatrix):
    master_weights: Float[Array, "*components rows cols"] = field(norm=ParameterNorm.SPECTRAL)
    master_scales: Float[Array, "*components rows groups"] = field(norm=ParameterNorm.L_INF)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.master_weights.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.master_weights.dtype

    @property
    def _packed_weights(self) -> UInt8[Array, "*components rows packed_cols"]:
        return _master_weights_to_packed_weights(
            self.master_weights,
            self.master_scales,
            global_scale=self.global_scale,
            group_size=self.spec.group_size,
            scale_mode=self.spec.scale_mode,
        )

    @property
    def _packed_scales(self) -> UInt8[Array, "*components rows groups"] | Float[Array, "*components rows groups"]:
        return _pack_scales(self.master_scales, self.spec.scale_mode)

    def astype(self, dtype: DTypeLike) -> "MicrofloatMatrixForTraining":
        return MicrofloatMatrixForTraining(
            spec=self.spec,
            is_sharded=self.is_sharded,
            master_weights=self.master_weights.astype(dtype),
            master_scales=self.master_scales.astype(dtype),
            global_scale=self.global_scale.astype(dtype),
        )

    def load_exported(
        self,
        expored_data: ExportResults,
        allow_dtype_cast: bool = False,
        *,
        prefix: ParameterPath | None = None,
    ) -> MicrofloatMatrix:
        return self._load_exported(
            expored_data,
            allow_dtype_cast=allow_dtype_cast,
            implementation=CompressionImplementation.TRAINING,
            prefix=prefix,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> MicrofloatMatrix:
        if implementation == CompressionImplementation.TRAINING:
            return self
        return self._from_packed_parameters(implementation)

    def _weights_for_forward(
        self,
        dtype: DTypeLike,
        keychain: Keychain | None,
        forward_pass_config: MatmulConfig,
        index: int | Int[Array, "*batch"] | None = None,
    ) -> Float[Array, "... cols"]:
        weights = self.master_weights
        scales = self.master_scales
        global_scale = self.global_scale[..., None, None]
        if index is not None:
            weights = lookup_sharded_indices(weights, index)
            scales = lookup_sharded_indices(scales, index)
            global_scale = self.global_scale
        scales = scales.astype(dtype)
        scale_values = _master_scales_to_scale_values(
            scales,
            self.spec.scale_mode,
            dtype,
        )
        scale_values = scale_values * global_scale.astype(dtype)
        return _master_weights_to_quantized_weights(
            weights.astype(dtype),
            scale_values,
            group_size=self.spec.group_size,
            keychain=keychain,
            gradient_estimator=forward_pass_config.gradient_estimator,
        )


class MicrofloatMatrixForInference(MicrofloatMatrix):
    dtype_: DTypeLike = field(static=True)
    packed_weights: UInt8[Array, "*components rows packed_cols"]
    packed_scales: UInt8[Array, "*components rows groups"] | Float[Array, "*components rows groups"]

    @property
    def shape(self) -> tuple[int, ...]:
        *leading_dims, groups = self.packed_scales.shape
        return (*leading_dims, groups * self.spec.group_size)

    @property
    def dtype(self) -> DTypeLike:
        return self.dtype_

    @property
    def _packed_weights(self) -> UInt8[Array, "*components rows packed_cols"]:
        return self.packed_weights

    @property
    def _packed_scales(self) -> UInt8[Array, "*components rows groups"] | Float[Array, "*components rows groups"]:
        return self.packed_scales

    def astype(self, dtype: DTypeLike) -> "MicrofloatMatrixForInference":
        return MicrofloatMatrixForInference(
            spec=self.spec,
            is_sharded=self.is_sharded,
            dtype_=jnp.dtype(dtype),
            packed_weights=self.packed_weights,
            packed_scales=self.packed_scales,
            global_scale=self.global_scale.astype(dtype),
        )

    def load_exported(
        self,
        expored_data: ExportResults,
        allow_dtype_cast: bool = False,
        *,
        prefix: ParameterPath | None = None,
    ) -> MicrofloatMatrix:
        return self._load_exported(
            expored_data,
            allow_dtype_cast=allow_dtype_cast,
            implementation=CompressionImplementation.INFERENCE,
            prefix=prefix,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> MicrofloatMatrix:
        if implementation == CompressionImplementation.INFERENCE:
            return self
        return self._from_packed_parameters(implementation)

    def _weights_for_forward(
        self,
        dtype: DTypeLike,
        keychain: Keychain | None,  # noqa: ARG002
        forward_pass_config: MatmulConfig,  # noqa: ARG002
        index: int | Int[Array, "*batch"] | None = None,
    ) -> Float[Array, "... cols"]:
        packed_weights = self.packed_weights
        packed_scales = self.packed_scales
        global_scale = self.global_scale[..., None, None]
        if index is not None:
            packed_weights = lookup_sharded_indices(packed_weights, index)
            packed_scales = lookup_sharded_indices(packed_scales, index)
            global_scale = self.global_scale
        return _packed_weights_to_master_weights(
            packed_weights,
            packed_scales,
            global_scale=global_scale,
            dtype=dtype,
            group_size=self.spec.group_size,
            scale_mode=self.spec.scale_mode,
        )
