from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal, NamedTuple, Self

import jax.numpy as jnp
from einops import rearrange
from jax.lax import stop_gradient
from jax.scipy.special import ndtri
from jax.sharding import PartitionSpec
from jaxtyping import Array, DTypeLike, Float, Int, Key, UInt8

from lalamo.exportable import ExportResults
from lalamo.module import Keychain, ParameterNorm, field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import preserve_first_input_sharding, supports_dummy_arrays
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.sharding import (
    is_sharded,
    make_sharding,
    reshard_as,
    sharding_of,
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
from .utils.gaussian_order_statistics import standard_normal_absmax_squared
from .utils.grouping import group_by_last_axis
from .utils.packing import pack_uint_to_uint8, unpack_uint8_to_uint
from .utils.yaqa import yaqa_round_weights

__all__ = [
    "NormalFloatMatrix",
    "NormalFloatMatrixForInference",
    "NormalFloatMatrixForTraining",
    "NormalFloatSpec",
]

type NormalFloatBits = Literal[2, 3, 4, 6, 8]

_NORMAL_FLOAT_OFFSET = 0.9677083


class NormalFloatParameters(NamedTuple):
    scales: Float[Array, "*components rows groups"]

    @classmethod
    @supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
    def from_weights(
        cls,
        weights: Float[Array, "*components rows cols"],
        group_size: int,
    ) -> Self:
        grouped_weights = group_by_last_axis(weights, group_size=group_size)
        scales = jnp.max(jnp.abs(grouped_weights), axis=-1)
        return cls(scales=scales)


def _normal_float_codebook(
    bits: int,
    preserve_zero: bool,
    dtype: DTypeLike,
) -> Float[Array, " levels"]:
    num_levels = 2**bits
    if preserve_zero:
        positive_levels = num_levels // 2
        negative_levels = num_levels - positive_levels - 1
        positive_probabilities = jnp.linspace(_NORMAL_FLOAT_OFFSET, 0.5, positive_levels + 1)[:-1]
        negative_probabilities = jnp.linspace(_NORMAL_FLOAT_OFFSET, 0.5, negative_levels + 1)[:-1]
        values = jnp.concatenate(
            [
                -ndtri(negative_probabilities),
                jnp.zeros((1,), dtype=jnp.float32),
                ndtri(positive_probabilities),
            ],
        )
        values = jnp.sort(values)
    else:
        probabilities = (jnp.arange(num_levels, dtype=jnp.float32) + 0.5) / num_levels
        values = ndtri(probabilities)

    values = values / jnp.max(jnp.abs(values))
    return values.astype(dtype)


def _normal_float_grouped_indices(
    weights: Float[Array, "... cols"],
    scales: Float[Array, "... groups"],
    *,
    bits: int,
    group_size: int,
    preserve_zero: bool,
) -> UInt8[Array, "... groups group_size"]:
    codebook = _normal_float_codebook(bits, preserve_zero, weights.dtype)
    grouped_weights = rearrange(weights, "... (groups group_size) -> ... groups group_size", group_size=group_size)
    safe_scales = jnp.where(scales == 0, jnp.ones_like(scales), scales)
    normalized_weights = grouped_weights / safe_scales[..., None]
    thresholds = (codebook[:-1] + codebook[1:]) / 2
    return jnp.sum(normalized_weights[..., None] > thresholds, axis=-1).astype(jnp.uint8)


def _normal_float_indices_to_master_weights(
    indices: UInt8[Array, "... cols"],
    scales: Float[Array, "... groups"],
    *,
    bits: int,
    group_size: int,
    preserve_zero: bool,
) -> Float[Array, "... cols"]:
    codebook = _normal_float_codebook(bits, preserve_zero, scales.dtype)
    grouped_indices = rearrange(
        indices.astype(jnp.int32),
        "... (groups group_size) -> ... groups group_size",
        group_size=group_size,
    )
    if scales.ndim == 1:
        levels = jnp.arange(2**bits, dtype=jnp.int32)
        grouped_values = jnp.sum((grouped_indices[..., None] == levels).astype(scales.dtype) * codebook, axis=-1)
    else:
        scales_sharding = sharding_of(scales)
        values_sharding = None
        if is_sharded(scales_sharding):
            values_sharding = PartitionSpec(*tuple(scales_sharding.spec), None)
        grouped_values = codebook.at[grouped_indices].get(out_sharding=values_sharding)
    grouped_weights = grouped_values * scales[..., None]
    return rearrange(grouped_weights, "... groups group_size -> ... (groups group_size)")


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _normal_float_quantize(
    weights: Float[Array, "... cols"],
    scales: Float[Array, "... groups"],
    *,
    bits: int,
    group_size: int,
    preserve_zero: bool,
) -> Float[Array, "... cols"]:
    grouped_indices = _normal_float_grouped_indices(
        weights,
        stop_gradient(scales),
        bits=bits,
        group_size=group_size,
        preserve_zero=preserve_zero,
    )
    indices = rearrange(grouped_indices, "... groups group_size -> ... (groups group_size)")
    quantized_weights = _normal_float_indices_to_master_weights(
        indices,
        scales,
        bits=bits,
        group_size=group_size,
        preserve_zero=preserve_zero,
    )
    return quantized_weights + weights - stop_gradient(weights)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _normal_float_pack_master_weights(
    weights: Float[Array, "*components rows cols"],
    scales: Float[Array, "*components rows groups"],
    *,
    bits: int,
    group_size: int,
    preserve_zero: bool,
) -> UInt8[Array, "*components rows packed_cols"]:
    grouped_indices = _normal_float_grouped_indices(
        weights,
        scales,
        bits=bits,
        group_size=group_size,
        preserve_zero=preserve_zero,
    )
    indices = rearrange(grouped_indices, "... groups group_size -> ... (groups group_size)")
    return pack_uint_to_uint8(indices, bits)


def _normal_float_unpack_master_weights(
    packed_weights: UInt8[Array, "... packed_cols"],
    scales: Float[Array, "... groups"],
    *,
    bits: int,
    group_size: int,
    preserve_zero: bool,
) -> Float[Array, "... cols"]:
    indices = unpack_uint8_to_uint(
        packed_weights,
        bits=bits,
        unpacked_last_axis_dim=scales.shape[-1] * group_size,
    )
    return _normal_float_indices_to_master_weights(
        indices,
        scales,
        bits=bits,
        group_size=group_size,
        preserve_zero=preserve_zero,
    )


@dataclass(frozen=True)
class NormalFloatSpec(QuantizedSpec):
    bits: NormalFloatBits
    group_size: int
    preserve_zero: bool = True
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
        return float(self.bits + 16 / self.group_size)

    @property
    def distortion(self) -> float:
        return standard_normal_absmax_squared(self.group_size) / (12 * (2**self.bits - 1) ** 2)

    def compress(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        is_sharded: bool = True,
    ) -> "NormalFloatMatrix":
        if preconditioner is not None:
            weights = yaqa_round_weights(weights, preconditioner, self, is_sharded=is_sharded)

        stored_weights = self.layout.from_output_input(weights, is_sharded=is_sharded)
        parameters = NormalFloatParameters.from_weights(stored_weights, group_size=self.group_size)
        weight_partition = self.layout.weight_partition(parameters.scales.ndim - 2, is_sharded=is_sharded)
        parameter_sharding = make_sharding((*weight_partition[:-1], None))
        scales = with_sharding(parameters.scales, parameter_sharding)
        if implementation == CompressionImplementation.TRAINING:
            return NormalFloatMatrixForTraining(
                spec=self,
                is_sharded=is_sharded,
                weights=stored_weights,
                scales=scales,
            )

        packed_weights = _normal_float_pack_master_weights(
            stored_weights,
            scales,
            bits=self.bits,
            group_size=self.group_size,
            preserve_zero=self.preserve_zero,
        )
        return self.from_packed_parameters(
            packed_weights=packed_weights,
            scales=scales,
            implementation=CompressionImplementation.INFERENCE,
            is_sharded=is_sharded,
        )

    def from_packed_parameters(
        self,
        *,
        packed_weights: UInt8[Array, "*components rows packed_cols"],
        scales: Float[Array, "*components rows groups"],
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        is_sharded: bool = True,
    ) -> "NormalFloatMatrix":
        weight_partition = self.layout.weight_partition(scales.ndim - 2, is_sharded=is_sharded)
        weight_sharding = make_sharding(weight_partition)
        parameter_sharding = make_sharding((*weight_partition[:-1], None))
        packed_weights = with_sharding(packed_weights, parameter_sharding)
        scales = with_sharding(scales, parameter_sharding)

        if implementation == CompressionImplementation.INFERENCE:
            return NormalFloatMatrixForInference(
                spec=self,
                is_sharded=is_sharded,
                packed_weights=packed_weights,
                scales=scales,
            )

        weights = _normal_float_unpack_master_weights(
            packed_weights,
            scales,
            bits=self.bits,
            group_size=self.group_size,
            preserve_zero=self.preserve_zero,
        )
        return NormalFloatMatrixForTraining(
            spec=self,
            is_sharded=is_sharded,
            weights=with_sharding(weights, weight_sharding),
            scales=scales,
        )


class NormalFloatMatrix(EmbeddingMatrix[NormalFloatSpec]):
    scales: Float[Array, "*components rows groups"] = field(norm=ParameterNorm.L_INF)

    @property
    @abstractmethod
    def _packed_quantized_weights(self) -> UInt8[Array, "*components rows packed_cols"]: ...

    def export(self) -> ExportResults:
        return ExportResults(
            arrays={
                "weights": self._packed_quantized_weights,
                "scales": self.scales,
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
    ) -> "NormalFloatMatrix": ...

    @abstractmethod
    def switch_implementation(self, implementation: CompressionImplementation) -> "NormalFloatMatrix": ...

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(layout=self.spec.layout).compress(self.decompress(), is_sharded=self.is_sharded)


class NormalFloatMatrixForTraining(NormalFloatMatrix):
    weights: Float[Array, "*components rows cols"] = field(norm=ParameterNorm.SPECTRAL)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.weights.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.weights.dtype

    @property
    def _packed_quantized_weights(self) -> UInt8[Array, "*components rows packed_cols"]:
        return _normal_float_pack_master_weights(
            self.weights,
            self.scales,
            bits=self.spec.bits,
            group_size=self.spec.group_size,
            preserve_zero=self.spec.preserve_zero,
        )

    def astype(self, dtype: DTypeLike) -> "NormalFloatMatrixForTraining":
        return NormalFloatMatrixForTraining(
            spec=self.spec,
            is_sharded=self.is_sharded,
            weights=self.weights.astype(dtype),
            scales=self.scales.astype(dtype),
        )

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        weights = _normal_float_quantize(
            self.weights,
            self.scales,
            bits=self.spec.bits,
            group_size=self.spec.group_size,
            preserve_zero=self.spec.preserve_zero,
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
        return _normal_float_quantize(
            self.weights[index, :].astype(dtype),
            self.scales[index, :].astype(dtype),
            bits=self.spec.bits,
            group_size=self.spec.group_size,
            preserve_zero=self.spec.preserve_zero,
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
        weights = _normal_float_quantize(
            self.weights.astype(vector.dtype),
            self.scales.astype(vector.dtype),
            bits=self.spec.bits,
            group_size=self.spec.group_size,
            preserve_zero=self.spec.preserve_zero,
        )
        layout = self.spec.layout
        if transposed:
            layout = layout.transpose()
        with use_dot_algorithm_preset(forward_pass_config.precision):
            result = layout.matmul(weights, vector)

        return reshard_as(result, vector)

    def load_exported(
        self,
        expored_data: ExportResults,
        allow_dtype_cast: bool = False,
        *,
        prefix: ParameterPath | None = None,
    ) -> NormalFloatMatrix:
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
        return self.spec.from_packed_parameters(
            packed_weights=packed_weights,
            scales=scales,
            implementation=CompressionImplementation.TRAINING,
            is_sharded=self.is_sharded,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> NormalFloatMatrix:
        if implementation == CompressionImplementation.TRAINING:
            return self
        return self.spec.from_packed_parameters(
            packed_weights=self._packed_quantized_weights,
            scales=self.scales,
            implementation=CompressionImplementation.INFERENCE,
            is_sharded=self.is_sharded,
        )


class NormalFloatMatrixForInference(NormalFloatMatrix):
    packed_weights: UInt8[Array, "*components rows packed_cols"]
    scales: Float[Array, "*components rows groups"]

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

    def astype(self, dtype: DTypeLike) -> "NormalFloatMatrixForInference":
        return NormalFloatMatrixForInference(
            spec=self.spec,
            is_sharded=self.is_sharded,
            packed_weights=self.packed_weights,
            scales=self.scales.astype(dtype),
        )

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        weights = _normal_float_unpack_master_weights(
            self.packed_weights,
            self.scales,
            bits=self.spec.bits,
            group_size=self.spec.group_size,
            preserve_zero=self.spec.preserve_zero,
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
        return _normal_float_unpack_master_weights(
            self.packed_weights[index, :],
            self.scales[index, :].astype(dtype),
            bits=self.spec.bits,
            group_size=self.spec.group_size,
            preserve_zero=self.spec.preserve_zero,
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
        weights = _normal_float_unpack_master_weights(
            self.packed_weights,
            self.scales.astype(vector.dtype),
            bits=self.spec.bits,
            group_size=self.spec.group_size,
            preserve_zero=self.spec.preserve_zero,
        )
        layout = self.spec.layout
        if transposed:
            layout = layout.transpose()
        with use_dot_algorithm_preset(forward_pass_config.precision):
            result = layout.matmul(weights, vector)

        return reshard_as(result, vector)

    def load_exported(
        self,
        expored_data: ExportResults,
        allow_dtype_cast: bool = False,
        *,
        prefix: ParameterPath | None = None,
    ) -> NormalFloatMatrix:
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
        return self.spec.from_packed_parameters(
            packed_weights=packed_weights,
            scales=scales,
            implementation=CompressionImplementation.INFERENCE,
            is_sharded=self.is_sharded,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> NormalFloatMatrix:
        if implementation == CompressionImplementation.INFERENCE:
            return self
        return self.spec.from_packed_parameters(
            packed_weights=self.packed_weights,
            scales=self.scales,
            implementation=CompressionImplementation.TRAINING,
            is_sharded=self.is_sharded,
        )
