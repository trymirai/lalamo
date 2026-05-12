from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal, NamedTuple, Self

import jax
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
    GradientEstimator,
    Layout,
    MatmulConfig,
    WeightMatrixSpec,
)

from .quantized_spec import QuantizedSpec
from .utils.gaussian_order_statistics import standard_normal_absmax_squared
from .utils.grouping import group_by_last_axis
from .utils.nf4_tables import (
    NORMAL_FLOAT_CODEBOOK_OFFSET,
    NORMAL_FLOAT_RVQ_BIAS_TABLES,
)
from .utils.packing import pack_uint_to_uint8, unpack_uint8_to_uint
from .utils.rounding import round_up_to_e4m3
from .utils.yaqa import yaqa_round_weights

__all__ = [
    "NormalFloatMatrix",
    "NormalFloatMatrixForInference",
    "NormalFloatMatrixForTraining",
    "NormalFloatSpec",
]


def _normal_float_scale_values(
    scales: Array,
    global_scales: Float[Array, "..."],
    dtype: DTypeLike,
) -> Float[Array, "... groups"]:
    scale_values = scales.astype(dtype)
    global_scale_values = global_scales.astype(dtype)
    while global_scale_values.ndim > scale_values.ndim:
        global_scale_values = jnp.squeeze(global_scale_values, axis=-1)
    return scale_values * global_scale_values


def _normal_float_values_at(
    indices: Array,
    values: Float[Array, " levels"],
    dtype: DTypeLike,
) -> Float[Array, "..."]:
    levels = jnp.arange(values.shape[0], dtype=jnp.int32)
    return jnp.sum((indices[..., None].astype(jnp.int32) == levels).astype(dtype) * values, axis=-1)


def _normal_float_rvq_bias_table(group_size: int, dtype: DTypeLike) -> Float[Array, " bias_levels"]:
    return jnp.array(NORMAL_FLOAT_RVQ_BIAS_TABLES[group_size], dtype=dtype)


class NormalFloatParameters(NamedTuple):
    scales: Array
    global_scales: Float[Array, "*components one_row one_group"]

    @classmethod
    @supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
    def from_weights(
        cls,
        weights: Float[Array, "*components rows cols"],
        group_size: int,
    ) -> Self:
        grouped_weights = group_by_last_axis(weights, group_size=group_size)
        group_absmax = jnp.max(jnp.abs(grouped_weights), axis=-1)
        matrix_absmax = jnp.max(jnp.abs(weights), axis=(-2, -1), keepdims=True)
        e4m3_max = jnp.asarray(jnp.finfo(jnp.float8_e4m3fn).max, dtype=matrix_absmax.dtype)
        global_scales = jnp.where(matrix_absmax == 0, jnp.ones_like(matrix_absmax), matrix_absmax / e4m3_max)
        scale_multipliers = round_up_to_e4m3(group_absmax / global_scales)
        return cls(scales=scale_multipliers.astype(jnp.float8_e4m3fn), global_scales=global_scales)


def _normal_float_codebook(
    bits: int,
    preserve_zero: bool,
    dtype: DTypeLike,
) -> Float[Array, " levels"]:
    num_levels = 2**bits
    if preserve_zero:
        positive_levels = num_levels // 2
        negative_levels = num_levels - positive_levels - 1
        positive_probabilities = jnp.linspace(NORMAL_FLOAT_CODEBOOK_OFFSET, 0.5, positive_levels + 1)[:-1]
        negative_probabilities = jnp.linspace(NORMAL_FLOAT_CODEBOOK_OFFSET, 0.5, negative_levels + 1)[:-1]
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
    scales: Array,
    *,
    global_scales: Float[Array, "..."],
    biases: Float[Array, "... groups"] | None,
    bits: int,
    group_size: int,
    preserve_zero: bool,
    keychain: Keychain | None = None,
    gradient_estimator: GradientEstimator = GradientEstimator.DETERMINISTIC_ROUNDING,
) -> UInt8[Array, "... groups group_size"]:
    codebook = _normal_float_codebook(bits, preserve_zero, weights.dtype)
    grouped_weights = rearrange(weights, "... (groups group_size) -> ... groups group_size", group_size=group_size)
    scales = _normal_float_scale_values(scales, global_scales, weights.dtype)
    normalized_weights = grouped_weights / scales[..., None]
    if biases is not None:
        normalized_weights = normalized_weights + biases[..., None]

    if gradient_estimator == GradientEstimator.STOCHASTIC_ROUNDING:
        if keychain is None:
            raise ValueError("Stochastic NormalFloat rounding requires a keychain.")
        lower_indices = jnp.clip(jnp.sum(normalized_weights[..., None] > codebook, axis=-1) - 1, 0, codebook.size - 1)
        upper_indices = jnp.minimum(lower_indices + 1, codebook.size - 1)
        lower_values = _normal_float_values_at(lower_indices, codebook, weights.dtype)
        upper_values = _normal_float_values_at(upper_indices, codebook, weights.dtype)
        upper_probability = jnp.clip(
            (normalized_weights - lower_values)
            / jnp.maximum(upper_values - lower_values, jnp.finfo(weights.dtype).eps),
            0,
            1,
        )
        samples = jax.random.uniform(keychain.batch_key, normalized_weights.shape, dtype=weights.dtype)
        return jnp.where(samples < upper_probability, upper_indices, lower_indices).astype(jnp.uint8)

    if gradient_estimator == GradientEstimator.LOCAL_ADDITIVE_NOISE:
        raise ValueError("Local additive noise is not implemented.")

    thresholds = (codebook[:-1] + codebook[1:]) / 2
    return jnp.sum(normalized_weights[..., None] > thresholds, axis=-1).astype(jnp.uint8)


def _normal_float_indices_to_master_weights(
    indices: UInt8[Array, "... cols"],
    scales: Array,
    *,
    global_scales: Float[Array, "..."],
    biases: Float[Array, "... groups"] | None,
    bits: int,
    group_size: int,
    preserve_zero: bool,
    dtype: DTypeLike,
) -> Float[Array, "... cols"]:
    codebook = _normal_float_codebook(bits, preserve_zero, dtype)
    grouped_indices = rearrange(
        indices.astype(jnp.int32),
        "... (groups group_size) -> ... groups group_size",
        group_size=group_size,
    )
    if scales.ndim == 1:
        grouped_values = _normal_float_values_at(grouped_indices, codebook, dtype)
    else:
        scales_sharding = sharding_of(scales)
        values_sharding = None
        if is_sharded(scales_sharding):
            values_sharding = PartitionSpec(*tuple(scales_sharding.spec), None)
        grouped_values = codebook.at[grouped_indices].get(out_sharding=values_sharding)
    if biases is not None:
        grouped_values = grouped_values - biases[..., None]
    scales = _normal_float_scale_values(scales, global_scales, dtype)
    grouped_weights = grouped_values * scales[..., None]
    return rearrange(grouped_weights, "... groups group_size -> ... (groups group_size)")


def _normal_float_grouped_bias_indices(
    weights: Float[Array, "... cols"],
    scales: Array,
    *,
    global_scales: Float[Array, "..."],
    bits: int,
    group_size: int,
    preserve_zero: bool,
) -> UInt8[Array, "... groups"]:
    codebook = _normal_float_codebook(bits, preserve_zero, weights.dtype)
    bias_table = _normal_float_rvq_bias_table(group_size, weights.dtype)
    grouped_weights = rearrange(weights, "... (groups group_size) -> ... groups group_size", group_size=group_size)
    scale_values = _normal_float_scale_values(scales, global_scales, weights.dtype)
    normalized_weights = grouped_weights / scale_values[..., None]
    candidate_biases = bias_table.reshape((1,) * (normalized_weights.ndim - 1) + (bias_table.shape[0], 1))
    thresholds = (codebook[:-1] + codebook[1:]) / 2
    shifted_weights = normalized_weights[..., None, :] + candidate_biases
    indices = jnp.sum(shifted_weights[..., None] > thresholds, axis=-1).astype(jnp.int32)
    code_values = _normal_float_values_at(indices, codebook, weights.dtype)
    residuals = code_values - candidate_biases - normalized_weights[..., None, :]
    return jnp.argmin(jnp.mean(residuals * residuals, axis=-1), axis=-1).astype(jnp.uint8)


def _normal_float_bias_values_from_indices(
    bias_indices: UInt8[Array, "... groups"] | None,
    *,
    group_size: int,
    dtype: DTypeLike,
) -> Float[Array, "... groups"] | None:
    if bias_indices is None:
        return None

    bias_table = _normal_float_rvq_bias_table(group_size, dtype)
    return _normal_float_values_at(bias_indices, bias_table, dtype)


def _normal_float_pack_bias_indices(
    bias_indices: UInt8[Array, "*components rows groups"] | None,
    *,
    bias_bits: int,
) -> UInt8[Array, "*components rows packed_groups"] | None:
    if bias_indices is None:
        return None
    return pack_uint_to_uint8(bias_indices, bias_bits)


def _normal_float_unpack_bias_indices(
    packed_bias_indices: UInt8[Array, "... packed_groups"] | None,
    *,
    bias_bits: int,
    groups: int,
) -> UInt8[Array, "... groups"] | None:
    if packed_bias_indices is None:
        return None
    return unpack_uint8_to_uint(packed_bias_indices, bits=bias_bits, unpacked_last_axis_dim=groups)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _normal_float_quantize(
    weights: Float[Array, "... cols"],
    scales: Array,
    *,
    global_scales: Float[Array, "..."],
    bias_indices: UInt8[Array, "... groups"] | None,
    bits: int,
    group_size: int,
    preserve_zero: bool,
    keychain: Keychain | None = None,
    gradient_estimator: GradientEstimator = GradientEstimator.DETERMINISTIC_ROUNDING,
) -> Float[Array, "... cols"]:
    biases = _normal_float_bias_values_from_indices(bias_indices, group_size=group_size, dtype=weights.dtype)
    grouped_indices = _normal_float_grouped_indices(
        weights,
        stop_gradient(scales),
        global_scales=stop_gradient(global_scales),
        biases=stop_gradient(biases) if biases is not None else None,
        bits=bits,
        group_size=group_size,
        preserve_zero=preserve_zero,
        keychain=keychain,
        gradient_estimator=gradient_estimator,
    )
    indices = rearrange(grouped_indices, "... groups group_size -> ... (groups group_size)")
    quantized_weights = _normal_float_indices_to_master_weights(
        indices,
        scales,
        global_scales=global_scales,
        biases=biases,
        bits=bits,
        group_size=group_size,
        preserve_zero=preserve_zero,
        dtype=weights.dtype,
    )
    return quantized_weights + weights - stop_gradient(weights)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _normal_float_pack_master_weights(
    weights: Float[Array, "*components rows cols"],
    scales: Array,
    *,
    global_scales: Float[Array, "*components one_row one_group"],
    bias_indices: UInt8[Array, "*components rows groups"] | None,
    bits: int,
    group_size: int,
    preserve_zero: bool,
) -> UInt8[Array, "*components rows packed_cols"]:
    biases = _normal_float_bias_values_from_indices(bias_indices, group_size=group_size, dtype=weights.dtype)
    grouped_indices = _normal_float_grouped_indices(
        weights,
        scales,
        global_scales=global_scales,
        biases=biases,
        bits=bits,
        group_size=group_size,
        preserve_zero=preserve_zero,
    )
    indices = rearrange(grouped_indices, "... groups group_size -> ... (groups group_size)")
    return pack_uint_to_uint8(indices, bits)


def _normal_float_unpack_master_weights(
    packed_weights: UInt8[Array, "... packed_cols"],
    scales: Array,
    *,
    global_scales: Float[Array, "..."],
    bias_indices: UInt8[Array, "... groups"] | None,
    dtype: DTypeLike,
    bits: int,
    group_size: int,
    preserve_zero: bool,
) -> Float[Array, "... cols"]:
    biases = _normal_float_bias_values_from_indices(bias_indices, group_size=group_size, dtype=dtype)
    indices = unpack_uint8_to_uint(
        packed_weights,
        bits=bits,
        unpacked_last_axis_dim=scales.shape[-1] * group_size,
    )
    return _normal_float_indices_to_master_weights(
        indices,
        scales,
        global_scales=global_scales,
        biases=biases,
        bits=bits,
        group_size=group_size,
        preserve_zero=preserve_zero,
        dtype=dtype,
    )


@dataclass(frozen=True)
class NormalFloatSpec(QuantizedSpec):
    bits: Literal[2, 3, 4, 6, 8]
    group_size: int
    bias_bits: Literal[0, 4] = 0
    preserve_zero: bool = True
    layout: Layout = Layout.OUTPUT_INPUT

    def __post_init__(self) -> None:
        if self.bias_bits == 4 and (self.bits != 4 or not self.preserve_zero):
            raise ValueError("NormalFloat RVQ biases are only available for preserve-zero NF4")
        if self.bias_bits == 4 and self.group_size not in NORMAL_FLOAT_RVQ_BIAS_TABLES:
            raise ValueError(f"NormalFloat RVQ biases are not available for group size {self.group_size}")

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
        return float(self.bits + (8 + self.bias_bits) / self.group_size)

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
        parameters = NormalFloatParameters.from_weights(
            stored_weights,
            group_size=self.group_size,
        )
        weight_partition = self.layout.weight_partition(parameters.scales.ndim - 2, is_sharded=is_sharded)
        parameter_sharding = make_sharding((*weight_partition[:-1], None))
        scales = with_sharding(parameters.scales, parameter_sharding)
        global_scales = with_sharding(parameters.global_scales, make_sharding((*weight_partition[:-2], None, None)))
        bias_indices = None
        if self.bias_bits:
            bias_indices = _normal_float_grouped_bias_indices(
                stored_weights,
                scales,
                global_scales=global_scales,
                bits=self.bits,
                group_size=self.group_size,
                preserve_zero=self.preserve_zero,
            )
        if implementation == CompressionImplementation.TRAINING:
            return NormalFloatMatrixForTraining(
                spec=self,
                is_sharded=is_sharded,
                dtype_=stored_weights.dtype,
                weights=stored_weights,
                scales=scales,
                global_scales=global_scales,
                bias_indices=bias_indices,
            )

        packed_weights = _normal_float_pack_master_weights(
            stored_weights,
            scales,
            global_scales=global_scales,
            bias_indices=bias_indices,
            bits=self.bits,
            group_size=self.group_size,
            preserve_zero=self.preserve_zero,
        )
        return self.from_packed_parameters(
            packed_weights=packed_weights,
            scales=scales,
            global_scales=global_scales,
            packed_bias_indices=_normal_float_pack_bias_indices(bias_indices, bias_bits=self.bias_bits),
            dtype=stored_weights.dtype,
            implementation=CompressionImplementation.INFERENCE,
            is_sharded=is_sharded,
        )

    def from_packed_parameters(
        self,
        *,
        packed_weights: UInt8[Array, "*components rows packed_cols"],
        scales: Array,
        global_scales: Float[Array, "*components one_row one_group"],
        packed_bias_indices: UInt8[Array, "*components rows packed_groups"] | None = None,
        dtype: DTypeLike = jnp.float32,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        is_sharded: bool = True,
    ) -> "NormalFloatMatrix":
        if self.bias_bits and packed_bias_indices is None:
            raise ValueError("NormalFloat RVQ biases require packed bias indices")

        weight_partition = self.layout.weight_partition(scales.ndim - 2, is_sharded=is_sharded)
        weight_sharding = make_sharding(weight_partition)
        parameter_sharding = make_sharding((*weight_partition[:-1], None))
        packed_weights = with_sharding(packed_weights, parameter_sharding)
        scales = with_sharding(scales, parameter_sharding)
        global_scales = with_sharding(global_scales, make_sharding((*weight_partition[:-2], None, None)))
        packed_bias_indices = (
            None if packed_bias_indices is None else with_sharding(packed_bias_indices, parameter_sharding)
        )

        if implementation == CompressionImplementation.INFERENCE:
            return NormalFloatMatrixForInference(
                spec=self,
                is_sharded=is_sharded,
                dtype_=jnp.dtype(dtype),
                packed_weights=packed_weights,
                scales=scales,
                global_scales=global_scales,
                packed_bias_indices=packed_bias_indices,
            )

        bias_indices = _normal_float_unpack_bias_indices(
            packed_bias_indices,
            bias_bits=self.bias_bits,
            groups=scales.shape[-1],
        )
        weights = _normal_float_unpack_master_weights(
            packed_weights,
            scales,
            global_scales=global_scales,
            bias_indices=bias_indices,
            dtype=dtype,
            bits=self.bits,
            group_size=self.group_size,
            preserve_zero=self.preserve_zero,
        )
        return NormalFloatMatrixForTraining(
            spec=self,
            is_sharded=is_sharded,
            dtype_=jnp.dtype(dtype),
            weights=with_sharding(weights, weight_sharding),
            scales=scales,
            global_scales=global_scales,
            bias_indices=bias_indices,
        )


class NormalFloatMatrix(EmbeddingMatrix[NormalFloatSpec]):
    dtype_: DTypeLike = field(static=True)
    scales: Array = field(trainable=False)
    global_scales: Float[Array, "*components one_row one_group"] = field(norm=ParameterNorm.L_INF)

    @property
    @abstractmethod
    def _packed_quantized_weights(self) -> UInt8[Array, "*components rows packed_cols"]: ...

    @property
    @abstractmethod
    def _packed_bias_indices(self) -> UInt8[Array, "*components rows packed_groups"] | None: ...

    def export(self) -> ExportResults:
        packed_bias_indices = self._packed_bias_indices
        arrays = {
            "weights": self._packed_quantized_weights,
            "scales": self.scales,
            "global_scales": self.global_scales,
        }
        if packed_bias_indices is not None:
            arrays["bias_indices"] = packed_bias_indices

        return ExportResults(arrays=arrays, metadata={"spec": self.spec.to_json()})

    def _load_exported(
        self,
        expored_data: ExportResults,
        allow_dtype_cast: bool,
        *,
        prefix: ParameterPath | None,
        implementation: CompressionImplementation,
    ) -> "NormalFloatMatrix":
        if prefix is None:
            prefix = ParameterPath()
        loaded_spec = WeightMatrixSpec.from_json(expored_data.metadata[prefix / "spec"])
        if loaded_spec != self.spec:
            raise ValueError(f"WeightMatrix spec mismatch: expected {self.spec}, got {loaded_spec}")

        packed_bias_indices = None
        if self.spec.bias_bits:
            packed_bias_indices = load_as(
                self._packed_bias_indices,
                expored_data.arrays[prefix / "bias_indices"],
                allow_dtype_cast=False,
            )
        return self.spec.from_packed_parameters(
            packed_weights=load_as(
                self._packed_quantized_weights,
                expored_data.arrays[prefix / "weights"],
                allow_dtype_cast=False,
            ),
            scales=load_as(self.scales, expored_data.arrays[prefix / "scales"], allow_dtype_cast=False),
            global_scales=load_as(
                self.global_scales,
                expored_data.arrays[prefix / "global_scales"],
                allow_dtype_cast=allow_dtype_cast,
            ),
            packed_bias_indices=packed_bias_indices,
            dtype=self.dtype,
            implementation=implementation,
            is_sharded=self.is_sharded,
        )

    def _from_current_packed_parameters(self, implementation: CompressionImplementation) -> "NormalFloatMatrix":
        return self.spec.from_packed_parameters(
            packed_weights=self._packed_quantized_weights,
            scales=self.scales,
            global_scales=self.global_scales,
            packed_bias_indices=self._packed_bias_indices,
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
    ) -> "NormalFloatMatrix": ...

    @abstractmethod
    def switch_implementation(self, implementation: CompressionImplementation) -> "NormalFloatMatrix": ...

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(layout=self.spec.layout).compress(self.decompress(), is_sharded=self.is_sharded)


class NormalFloatMatrixForTraining(NormalFloatMatrix):
    weights: Float[Array, "*components rows cols"] = field(norm=ParameterNorm.SPECTRAL)
    bias_indices: UInt8[Array, "*components rows groups"] | None = field(trainable=False, default=None)

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
            global_scales=self.global_scales,
            bias_indices=self.bias_indices,
            bits=self.spec.bits,
            group_size=self.spec.group_size,
            preserve_zero=self.spec.preserve_zero,
        )

    @property
    def _packed_bias_indices(self) -> UInt8[Array, "*components rows packed_groups"] | None:
        return _normal_float_pack_bias_indices(self.bias_indices, bias_bits=self.spec.bias_bits)

    def astype(self, dtype: DTypeLike) -> "NormalFloatMatrixForTraining":
        return NormalFloatMatrixForTraining(
            spec=self.spec,
            is_sharded=self.is_sharded,
            dtype_=jnp.dtype(dtype),
            weights=self.weights.astype(dtype),
            scales=self.scales,
            global_scales=self.global_scales.astype(dtype),
            bias_indices=self.bias_indices,
        )

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        weights = _normal_float_quantize(
            self.weights,
            self.scales,
            global_scales=self.global_scales,
            bias_indices=self.bias_indices,
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
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
    ) -> Float[Array, " out_channels"]:
        self._raise_if_batched()
        if self.spec.layout != Layout.INPUT_OUTPUT:
            raise ValueError(f"Embedding lookup not supported for layout {self.spec.layout}")
        if dtype is None:
            dtype = self.dtype
        bias_indices = None if self.bias_indices is None else self.bias_indices[index, :]
        return _normal_float_quantize(
            self.weights[index, :].astype(dtype),
            self.scales[index, :],
            global_scales=self.global_scales,
            bias_indices=bias_indices,
            bits=self.spec.bits,
            group_size=self.spec.group_size,
            preserve_zero=self.spec.preserve_zero,
            keychain=keychain,
            gradient_estimator=forward_pass_config.gradient_estimator,
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
        weights = _normal_float_quantize(
            self.weights.astype(vector.dtype),
            self.scales,
            global_scales=self.global_scales,
            bias_indices=self.bias_indices,
            bits=self.spec.bits,
            group_size=self.spec.group_size,
            preserve_zero=self.spec.preserve_zero,
            keychain=keychain,
            gradient_estimator=forward_pass_config.gradient_estimator,
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
        return self._load_exported(
            expored_data,
            allow_dtype_cast,
            prefix=prefix,
            implementation=CompressionImplementation.TRAINING,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> NormalFloatMatrix:
        if implementation == CompressionImplementation.TRAINING:
            return self
        return self._from_current_packed_parameters(CompressionImplementation.INFERENCE)


class NormalFloatMatrixForInference(NormalFloatMatrix):
    packed_weights: UInt8[Array, "*components rows packed_cols"]
    packed_bias_indices: UInt8[Array, "*components rows packed_groups"] | None

    @property
    def shape(self) -> tuple[int, ...]:
        *leading_dims, rows, groups = self.scales.shape
        return (*leading_dims, rows, groups * self.spec.group_size)

    @property
    def dtype(self) -> DTypeLike:
        return self.dtype_

    @property
    def _packed_quantized_weights(self) -> UInt8[Array, "*components rows packed_cols"]:
        return self.packed_weights

    @property
    def _packed_bias_indices(self) -> UInt8[Array, "*components rows packed_groups"] | None:
        return self.packed_bias_indices

    @property
    def _bias_indices(self) -> UInt8[Array, "*components rows groups"] | None:
        return _normal_float_unpack_bias_indices(
            self.packed_bias_indices,
            bias_bits=self.spec.bias_bits,
            groups=self.scales.shape[-1],
        )

    def astype(self, dtype: DTypeLike) -> "NormalFloatMatrixForInference":
        return NormalFloatMatrixForInference(
            spec=self.spec,
            is_sharded=self.is_sharded,
            dtype_=jnp.dtype(dtype),
            packed_weights=self.packed_weights,
            scales=self.scales,
            global_scales=self.global_scales.astype(dtype),
            packed_bias_indices=self.packed_bias_indices,
        )

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        weights = _normal_float_unpack_master_weights(
            self.packed_weights,
            self.scales,
            global_scales=self.global_scales,
            bias_indices=self._bias_indices,
            dtype=self.dtype,
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
        bias_indices = self._bias_indices
        return _normal_float_unpack_master_weights(
            self.packed_weights[index, :],
            self.scales[index, :],
            global_scales=self.global_scales,
            bias_indices=None if bias_indices is None else bias_indices[index, :],
            dtype=dtype,
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
            self.scales,
            global_scales=self.global_scales,
            bias_indices=self._bias_indices,
            dtype=vector.dtype,
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
        return self._load_exported(
            expored_data,
            allow_dtype_cast,
            prefix=prefix,
            implementation=CompressionImplementation.INFERENCE,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> NormalFloatMatrix:
        if implementation == CompressionImplementation.INFERENCE:
            return self
        return self._from_current_packed_parameters(CompressionImplementation.TRAINING)
