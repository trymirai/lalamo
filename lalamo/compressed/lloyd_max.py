from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import jax.numpy as jnp
from jax.lax import stop_gradient
from jax.sharding import NamedSharding
from jaxtyping import Array, DTypeLike, Float, Int, Key, UInt8

from lalamo.exportable import ExportResults
from lalamo.module import Keychain, ParameterNorm, field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import (
    preserve_first_input_sharding,
    supports_dummy_arrays,
)
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
    GradientEstimator,
    Layout,
    MatmulConfig,
    WeightMatrixSpec,
)

from .data.distortion import distortion_estimate
from .data.lloyd_max import bias_lut_values, codebook_values
from .quantized_spec import QuantizedSpec
from .utils.grouping import group_by_last_axis
from .utils.packing import pack_uint_to_uint8, unpack_uint8_to_uint
from .utils.rounding import (
    lut_values_at,
    pack_e4m3_scales,
    round_to_sorted_lut_table,
    round_to_sorted_lut_table_indices,
)
from .utils.yaqa import yaqa_round_blockwise

__all__ = [
    "LloydMaxMatrix",
    "LloydMaxMatrixForInference",
    "LloydMaxMatrixForTraining",
    "LloydMaxSpec",
]

_MAX_BIAS_SEARCH_ELEMENTS_PER_CHUNK = 8_388_608


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _master_weights_to_master_scales(
    master_weights: Float[Array, "... cols"],
    group_size: int,
) -> Float[Array, "... groups"]:
    grouped_weights = group_by_last_axis(master_weights, group_size=group_size)
    return jnp.max(jnp.abs(grouped_weights), axis=-1)


def _codebook(
    bits: int,
    group_size: int,
    dtype: DTypeLike,
    bias_bits: int | None = None,
) -> Float[Array, " levels"]:
    values = codebook_values(bits=bits, group_size=group_size, bias_bits=bias_bits)
    return jnp.array(values, dtype=dtype)


def _bias_lut(
    *,
    group_size: int,
    bias_bits: int,
    bits: int,
    dtype: DTypeLike,
) -> Float[Array, " levels"]:
    values = bias_lut_values(bits=bits, group_size=group_size, bias_bits=bias_bits)
    return jnp.array(values, dtype=dtype)


def _master_weights_to_grouped_weight_indices(
    master_weights: Float[Array, "... cols"],
    scale_values: Float[Array, "... groups"],
    *,
    master_biases: Float[Array, "... groups"] | None,
    bits: int,
    group_size: int,
    bias_bits: int | None = None,
    keychain: Keychain | None = None,
    gradient_estimator: GradientEstimator = GradientEstimator.DETERMINISTIC_ROUNDING,
    out_sharding: NamedSharding | None = None,
) -> UInt8[Array, "... groups group_size"]:
    codebook = _codebook(bits, group_size, master_weights.dtype, bias_bits=bias_bits)
    safe_scale_values = jnp.where(scale_values == 0, 1, scale_values)
    grouped_weights = group_by_last_axis(master_weights, group_size=group_size)
    normalized_weights = grouped_weights / safe_scale_values.astype(master_weights.dtype)[..., None]
    if master_biases is not None:
        normalized_weights = normalized_weights + master_biases[..., None]
    if out_sharding is not None:
        normalized_weights = with_sharding(normalized_weights, out_sharding)

    return round_to_sorted_lut_table_indices(
        normalized_weights,
        codebook,
        keychain=keychain,
        gradient_estimator=gradient_estimator,
    )


def _unpacked_weight_indices_to_master_weights(
    unpacked_weight_indices: UInt8[Array, "... cols"],
    scale_values: Float[Array, "... groups"],
    *,
    master_biases: Float[Array, "... groups"] | None,
    bits: int,
    bias_bits: int | None,
    group_size: int,
    dtype: DTypeLike,
    grouped_values_sharding: NamedSharding | None = None,
    out_sharding: NamedSharding | None = None,
) -> Float[Array, "... cols"]:
    codebook = _codebook(bits, group_size, dtype, bias_bits=bias_bits)
    *leading_dims, cols = unpacked_weight_indices.shape
    groups = cols // group_size
    grouped_weight_indices = unpacked_weight_indices.reshape(*leading_dims, groups, group_size)
    if grouped_values_sharding is not None:
        grouped_weight_indices = with_sharding(grouped_weight_indices, grouped_values_sharding)
    grouped_values = lut_values_at(grouped_weight_indices, codebook)
    if master_biases is not None:
        grouped_values = grouped_values - master_biases[..., None]

    grouped_master_weights = grouped_values * scale_values.astype(dtype)[..., None]
    *leading_dims, groups, group_width = grouped_master_weights.shape
    return jnp.reshape(grouped_master_weights, (*leading_dims, groups * group_width), out_sharding=out_sharding)


def _bias_costs_for_candidate_biases(
    normalized_weights: Float[Array, "... groups group_size"],
    codebook: Float[Array, " levels"],
    candidate_biases: Float[Array, " bias_levels"],
    *,
    code_values_sharding: NamedSharding | None = None,
) -> Float[Array, "... groups bias_levels"]:
    candidate_bias_shape = (1,) * (normalized_weights.ndim - 1) + (candidate_biases.size, 1)
    candidate_biases = candidate_biases.reshape(candidate_bias_shape)
    shifted_weights = normalized_weights[..., None, :] + candidate_biases
    shifted_weight_indices = round_to_sorted_lut_table_indices(
        shifted_weights,
        codebook,
        gradient_estimator=GradientEstimator.DETERMINISTIC_ROUNDING,
    )
    if code_values_sharding is not None:
        shifted_weight_indices = with_sharding(shifted_weight_indices, code_values_sharding)
    code_values = lut_values_at(shifted_weight_indices, codebook)
    residuals = code_values - candidate_biases - normalized_weights[..., None, :]
    return jnp.mean(residuals * residuals, axis=-1)


def _master_weights_to_master_biases(
    master_weights: Float[Array, "... cols"],
    scale_values: Float[Array, "... groups"],
    *,
    bits: int,
    bias_bits: int,
    group_size: int,
    bias_sharding: NamedSharding | None = None,
    bias_cost_sharding: NamedSharding | None = None,
) -> Float[Array, "... groups"]:
    codebook = _codebook(bits, group_size, dtype=master_weights.dtype, bias_bits=bias_bits)
    bias_table = _bias_lut(
        group_size=group_size,
        bias_bits=bias_bits,
        bits=bits,
        dtype=master_weights.dtype,
    )
    safe_scale_values = jnp.where(scale_values == 0, 1, scale_values)
    grouped_weights = group_by_last_axis(master_weights, group_size=group_size)
    normalized_weights = grouped_weights / safe_scale_values.astype(master_weights.dtype)[..., None]
    (bias_levels,) = bias_table.shape

    chunk_size = max(
        1,
        min(
            bias_levels,
            _MAX_BIAS_SEARCH_ELEMENTS_PER_CHUNK // normalized_weights.size,
        ),
    )
    grouped_weight_template = normalized_weights.at[..., 0].get(out_sharding=bias_sharding)
    best_bias_costs = jnp.full_like(grouped_weight_template, jnp.inf)
    bias_indices = jnp.zeros_like(grouped_weight_template, dtype=jnp.int32)
    for candidate_start in range(0, bias_levels, chunk_size):
        bias_costs = _bias_costs_for_candidate_biases(
            normalized_weights,
            codebook,
            bias_table[candidate_start : candidate_start + chunk_size],
            code_values_sharding=bias_cost_sharding,
        )
        chunk_bias_indices = jnp.argmin(bias_costs, axis=-1)
        chunk_bias_costs = jnp.min(bias_costs, axis=-1)
        is_better = chunk_bias_costs < best_bias_costs
        bias_indices = jnp.where(is_better, candidate_start + chunk_bias_indices, bias_indices)
        best_bias_costs = jnp.where(is_better, chunk_bias_costs, best_bias_costs)
    if bias_sharding is not None:
        bias_indices = with_sharding(bias_indices, bias_sharding)
    return lut_values_at(bias_indices, bias_table)


def _bias_indices_to_master_biases(
    bias_indices: UInt8[Array, "... groups"],
    *,
    bits: int,
    bias_bits: int,
    group_size: int,
    dtype: DTypeLike,
    out_sharding: NamedSharding | None = None,
) -> Float[Array, "... groups"]:
    bias_table = _bias_lut(
        group_size=group_size,
        bias_bits=bias_bits,
        bits=bits,
        dtype=dtype,
    )
    if out_sharding is not None:
        bias_indices = with_sharding(bias_indices, out_sharding)
    return lut_values_at(bias_indices, bias_table)


def _master_biases_to_bias_indices(
    master_biases: Float[Array, "... groups"],
    *,
    bits: int,
    bias_bits: int,
    group_size: int,
) -> UInt8[Array, "... groups"]:
    bias_table = _bias_lut(
        group_size=group_size,
        bias_bits=bias_bits,
        bits=bits,
        dtype=master_biases.dtype,
    )
    return round_to_sorted_lut_table_indices(
        master_biases,
        bias_table,
        gradient_estimator=GradientEstimator.DETERMINISTIC_ROUNDING,
    )


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _master_biases_to_packed_bias_indices(
    master_biases: Float[Array, "*components rows groups"] | None,
    *,
    bits: int,
    bias_bits: int | None,
    group_size: int,
    sharding_config: ShardingConfig,
) -> UInt8[Array, "*components rows packed_groups"] | None:
    if master_biases is None:
        return None
    assert bias_bits is not None
    bias_indices = _master_biases_to_bias_indices(
        master_biases,
        bits=bits,
        bias_bits=bias_bits,
        group_size=group_size,
    )
    return pack_uint_to_uint8(bias_indices, bias_bits, sharding_config=sharding_config)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _master_weights_to_quantized_weights(
    master_weights: Float[Array, "... cols"],
    scale_values: Float[Array, "... groups"],
    *,
    master_biases: Float[Array, "... groups"] | None,
    bits: int,
    bias_bits: int | None,
    group_size: int,
    keychain: Keychain | None = None,
    gradient_estimator: GradientEstimator = GradientEstimator.DETERMINISTIC_ROUNDING,
    bias_sharding: NamedSharding | None = None,
    out_sharding: NamedSharding | None = None,
    grouped_values_sharding: NamedSharding | None = None,
) -> Float[Array, "... cols"]:
    bias_keychain = keychain
    weight_keychain = keychain
    if master_biases is not None and keychain is not None:
        bias_keychain, weight_keychain = keychain.split()

    if master_biases is not None:
        assert bias_bits is not None
        bias_table = _bias_lut(
            group_size=group_size,
            bias_bits=bias_bits,
            bits=bits,
            dtype=master_weights.dtype,
        )
        if bias_sharding is not None:
            master_biases = with_sharding(master_biases, bias_sharding)
        master_biases = round_to_sorted_lut_table(
            master_biases,
            bias_table,
            keychain=bias_keychain,
            gradient_estimator=gradient_estimator,
        )

    safe_scale_values = jnp.where(scale_values == 0, 1, scale_values)
    grouped_weights = group_by_last_axis(master_weights, group_size=group_size)
    normalized_weights = grouped_weights / stop_gradient(safe_scale_values)[..., None]
    if master_biases is not None:
        normalized_weights = normalized_weights + stop_gradient(master_biases)[..., None]
    if grouped_values_sharding is not None:
        normalized_weights = with_sharding(normalized_weights, grouped_values_sharding)

    grouped_values = round_to_sorted_lut_table(
        normalized_weights,
        _codebook(bits, group_size, master_weights.dtype, bias_bits=bias_bits),
        keychain=weight_keychain,
        gradient_estimator=gradient_estimator,
    )
    if master_biases is not None:
        grouped_values = grouped_values - master_biases[..., None]

    grouped_quantized_weights = grouped_values * scale_values[..., None]
    *leading_dims, groups, group_width = grouped_quantized_weights.shape
    return jnp.reshape(grouped_quantized_weights, (*leading_dims, groups * group_width), out_sharding=out_sharding)


@supports_dummy_arrays()
def _master_weights_to_packed_weight_indices(
    master_weights: Float[Array, "*components rows cols"],
    master_scales: Float[Array, "*components rows groups"],
    *,
    master_biases: Float[Array, "*components rows groups"] | None,
    bits: int,
    bias_bits: int | None,
    group_size: int,
    weight_sharding: NamedSharding,
    parameter_sharding: NamedSharding,
    grouped_values_sharding: NamedSharding,
    sharding_config: ShardingConfig,
) -> UInt8[Array, "*components rows packed_cols"]:
    packed_scales = pack_e4m3_scales(master_scales, out_sharding=parameter_sharding)
    if master_biases is not None:
        assert bias_bits is not None
        bias_indices = _master_biases_to_bias_indices(
            master_biases,
            bits=bits,
            bias_bits=bias_bits,
            group_size=group_size,
        )
        master_biases = _bias_indices_to_master_biases(
            bias_indices,
            bits=bits,
            bias_bits=bias_bits,
            group_size=group_size,
            dtype=master_weights.dtype,
            out_sharding=parameter_sharding,
        )
    grouped_weight_indices = _master_weights_to_grouped_weight_indices(
        master_weights,
        packed_scales.astype(master_weights.dtype),
        master_biases=master_biases,
        bits=bits,
        bias_bits=bias_bits,
        group_size=group_size,
        out_sharding=grouped_values_sharding,
    )
    *leading_dims, groups, group_width = grouped_weight_indices.shape
    unpacked_weight_indices = jnp.reshape(
        grouped_weight_indices,
        (*leading_dims, groups * group_width),
        out_sharding=weight_sharding,
    )
    packed_indices = pack_uint_to_uint8(unpacked_weight_indices, bits, sharding_config=sharding_config)
    return with_sharding(packed_indices, parameter_sharding)


@dataclass(frozen=True)
class LloydMaxSpec(QuantizedSpec):
    bits: Literal[2, 3, 4, 6, 8]
    group_size: int
    bias_bits: Literal[2, 3, 4, 6, 8] | None = None
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
        bias_bits = 0
        if self.bias_bits is not None:
            bias_bits = self.bias_bits
        return self.bits + (8 + bias_bits) / self.group_size

    @cached_property
    def distortion(self) -> float:
        return distortion_estimate(
            format_name="lloyd_max",
            bits=self.bits,
            bias_bits=self.bias_bits,
            group_size=self.group_size,
        )

    def compress(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        sharding_config: ShardingConfig,
        is_sharded: bool = True,
    ) -> "LloydMaxMatrix":
        if preconditioner is not None:
            weights = yaqa_round_blockwise(
                weights,
                preconditioner,
                self,
                sharding_config=sharding_config,
            )

        weight_axes = self.layout.weight_partition(weights.ndim - 2, is_sharded=is_sharded)
        *parameter_axes, grouped_axis = weight_axes
        weight_sharding = sharding_config.resolve_sharding(weight_axes)
        parameter_sharding = sharding_config.resolve_sharding((*parameter_axes, None))
        grouped_values_sharding = sharding_config.resolve_sharding(
            (*parameter_axes, None, grouped_axis),
        )
        bias_cost_sharding = sharding_config.resolve_sharding(
            (*parameter_axes, None, None, grouped_axis),
        )
        stored_weights = self.layout.from_output_input(weights, sharding=weight_sharding)
        master_scales = _master_weights_to_master_scales(stored_weights, group_size=self.group_size)
        master_scales = with_sharding(master_scales, parameter_sharding)
        packed_scales = pack_e4m3_scales(master_scales, out_sharding=parameter_sharding)
        if self.bias_bits is None:
            master_biases = None
        else:
            scale_values = packed_scales.astype(stored_weights.dtype)
            master_biases = _master_weights_to_master_biases(
                stored_weights,
                scale_values,
                bits=self.bits,
                bias_bits=self.bias_bits,
                group_size=self.group_size,
                bias_sharding=parameter_sharding,
                bias_cost_sharding=bias_cost_sharding,
            )
        if implementation == CompressionImplementation.TRAINING:
            return LloydMaxMatrixForTraining(
                spec=self,
                sharding_config=sharding_config,
                is_sharded=is_sharded,
                master_weights=with_sharding(stored_weights, weight_sharding),
                master_scales=master_scales,
                master_biases=master_biases,
            )

        packed_weight_indices = _master_weights_to_packed_weight_indices(
            stored_weights,
            master_scales,
            master_biases=master_biases,
            bits=self.bits,
            bias_bits=self.bias_bits,
            group_size=self.group_size,
            weight_sharding=weight_sharding,
            parameter_sharding=parameter_sharding,
            grouped_values_sharding=grouped_values_sharding,
            sharding_config=sharding_config,
        )
        packed_bias_indices = _master_biases_to_packed_bias_indices(
            master_biases,
            bits=self.bits,
            bias_bits=self.bias_bits,
            group_size=self.group_size,
            sharding_config=sharding_config,
        )
        return self.from_packed_parameters(
            packed_weight_indices=packed_weight_indices,
            packed_scales=packed_scales,
            packed_bias_indices=packed_bias_indices,
            dtype=stored_weights.dtype,
            implementation=CompressionImplementation.INFERENCE,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
        )

    def from_packed_parameters(
        self,
        *,
        packed_weight_indices: UInt8[Array, "*components rows packed_cols"],
        packed_scales: Float[Array, "*components rows groups"],
        packed_bias_indices: UInt8[Array, "*components rows packed_groups"] | None = None,
        dtype: DTypeLike = jnp.float32,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        sharding_config: ShardingConfig,
        is_sharded: bool = True,
    ) -> "LloydMaxMatrix":
        if self.bias_bits is None:
            if packed_bias_indices is not None:
                raise ValueError("Bias-free LloydMax parameters must not include packed bias indices.")
        elif packed_bias_indices is None:
            raise ValueError("LloydMax biases require packed bias indices.")

        weight_axes = self.layout.weight_partition(packed_scales.ndim - 2, is_sharded=is_sharded)
        *parameter_axes, grouped_axis = weight_axes
        parameter_sharding = sharding_config.resolve_sharding((*parameter_axes, None))
        weight_sharding = sharding_config.resolve_sharding(weight_axes)
        grouped_values_sharding = sharding_config.resolve_sharding(
            (*parameter_axes, None, grouped_axis),
        )
        packed_weight_indices = with_sharding(packed_weight_indices, parameter_sharding)
        packed_scales = with_sharding(packed_scales, parameter_sharding)
        if packed_bias_indices is not None:
            packed_bias_indices = with_sharding(packed_bias_indices, parameter_sharding)

        if implementation == CompressionImplementation.INFERENCE:
            return LloydMaxMatrixForInference(
                spec=self,
                sharding_config=sharding_config,
                is_sharded=is_sharded,
                dtype_=jnp.dtype(dtype),
                packed_weight_indices=packed_weight_indices,
                packed_scales=packed_scales,
                packed_bias_indices=packed_bias_indices,
            )

        *_, groups = packed_scales.shape
        if packed_bias_indices is None:
            master_biases = None
        else:
            assert self.bias_bits is not None
            bias_indices = unpack_uint8_to_uint(
                packed_bias_indices,
                bits=self.bias_bits,
                unpacked_last_axis_dim=groups,
            )
            master_biases = _bias_indices_to_master_biases(
                bias_indices,
                bits=self.bits,
                bias_bits=self.bias_bits,
                group_size=self.group_size,
                dtype=dtype,
                out_sharding=parameter_sharding,
            )
        unpacked_weight_indices = unpack_uint8_to_uint(
            packed_weight_indices,
            bits=self.bits,
            unpacked_last_axis_dim=groups * self.group_size,
        )
        weights = _unpacked_weight_indices_to_master_weights(
            unpacked_weight_indices,
            packed_scales.astype(dtype),
            master_biases=master_biases,
            dtype=dtype,
            bits=self.bits,
            bias_bits=self.bias_bits,
            group_size=self.group_size,
            grouped_values_sharding=grouped_values_sharding,
            out_sharding=weight_sharding,
        )
        return LloydMaxMatrixForTraining(
            spec=self,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
            master_weights=with_sharding(weights, weight_sharding),
            master_scales=with_sharding(packed_scales.astype(dtype), parameter_sharding),
            master_biases=master_biases,
        )


class LloydMaxMatrix(EmbeddingMatrix[LloydMaxSpec]):
    @property
    @abstractmethod
    def _packed_weight_indices(
        self,
    ) -> UInt8[Array, "*components rows packed_cols"]: ...

    @property
    @abstractmethod
    def _packed_scales(self) -> Float[Array, "*components rows groups"]: ...

    @property
    @abstractmethod
    def _packed_bias_indices(
        self,
    ) -> UInt8[Array, "*components rows packed_groups"] | None: ...

    @abstractmethod
    def _weights_for_forward(
        self,
        dtype: DTypeLike,
        keychain: Keychain | None,
        forward_pass_config: MatmulConfig,
        row_index: int | Int[Array, "*batch"] | None = None,
    ) -> Float[Array, "... cols"]: ...

    def export(self) -> ExportResults:
        packed_bias_indices = self._packed_bias_indices
        arrays = {
            "weights": self._packed_weight_indices,
            "scales": self._packed_scales,
        }
        if packed_bias_indices is not None:
            arrays["bias_indices"] = packed_bias_indices

        return ExportResults(arrays=arrays, metadata={"spec": self.spec.to_json()})

    def _load_exported(
        self,
        exported_data: ExportResults,
        *,
        implementation: CompressionImplementation,
        prefix: ParameterPath | None = None,
    ) -> "LloydMaxMatrix":
        if prefix is None:
            prefix = ParameterPath()
        loaded_spec = WeightMatrixSpec.from_json(exported_data.metadata[prefix / "spec"])
        if loaded_spec != self.spec:
            raise ValueError(f"WeightMatrix spec mismatch: expected {self.spec}, got {loaded_spec}")

        if self.spec.bias_bits is None:
            packed_bias_indices = None
        else:
            packed_bias_indices = load_as(
                self._packed_bias_indices,
                exported_data.arrays[prefix / "bias_indices"],
            )
        packed_weight_indices = load_as(
            self._packed_weight_indices,
            exported_data.arrays[prefix / "weights"],
        )
        packed_scales = load_as(
            self._packed_scales,
            exported_data.arrays[prefix / "scales"],
        )
        if implementation == CompressionImplementation.INFERENCE:
            return LloydMaxMatrixForInference(
                spec=self.spec,
                sharding_config=self.sharding_config,
                is_sharded=self.is_sharded,
                dtype_=jnp.dtype(self.dtype),
                packed_weight_indices=packed_weight_indices,
                packed_scales=packed_scales,
                packed_bias_indices=packed_bias_indices,
            )
        return self.spec.from_packed_parameters(
            packed_weight_indices=packed_weight_indices,
            packed_scales=packed_scales,
            packed_bias_indices=packed_bias_indices,
            dtype=self.dtype,
            implementation=implementation,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
        )

    def _from_packed_parameters(self, implementation: CompressionImplementation) -> "LloydMaxMatrix":
        return self.spec.from_packed_parameters(
            packed_weight_indices=self._packed_weight_indices,
            packed_scales=self._packed_scales,
            packed_bias_indices=self._packed_bias_indices,
            dtype=self.dtype,
            implementation=implementation,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
        )

    @abstractmethod
    def load_exported(
        self,
        exported_data: ExportResults,
        *,
        prefix: ParameterPath | None = None,
    ) -> "LloydMaxMatrix": ...

    @abstractmethod
    def switch_implementation(self, implementation: CompressionImplementation) -> "LloydMaxMatrix": ...

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(layout=self.spec.layout).compress(
            self.decompress(),
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
        )

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        weights = self._weights_for_forward(self.dtype, None, MatmulConfig())
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
        return self._weights_for_forward(dtype, keychain, forward_pass_config, row_index)

    def dot(
        self,
        vector: Float[Array, " source_channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Float[Array, " target_channels"]:
        self._raise_if_batched()
        weights = self._weights_for_forward(vector.dtype, keychain, forward_pass_config)
        layout = self.spec.layout
        if transposed:
            layout = layout.transpose()
        with use_dot_algorithm_preset(forward_pass_config.precision):
            return layout.matmul(weights, vector)


class LloydMaxMatrixForTraining(LloydMaxMatrix):
    master_weights: Float[Array, "*components rows cols"] = field(norm=ParameterNorm.SPECTRAL)
    master_scales: Float[Array, "*components rows groups"] = field(norm=ParameterNorm.L_INF)
    master_biases: Float[Array, "*components rows groups"] | None = field(norm=ParameterNorm.L_INF, default=None)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.master_weights.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.master_weights.dtype

    @property
    def _packed_weight_indices(self) -> UInt8[Array, "*components rows packed_cols"]:
        weight_axes = self.spec.layout.weight_partition(self.master_scales.ndim - 2, is_sharded=self.is_sharded)
        *parameter_axes, grouped_axis = weight_axes
        return _master_weights_to_packed_weight_indices(
            self.master_weights,
            self.master_scales,
            master_biases=self.master_biases,
            bits=self.spec.bits,
            bias_bits=self.spec.bias_bits,
            group_size=self.spec.group_size,
            weight_sharding=self._resolve_sharding(weight_axes),
            parameter_sharding=self._resolve_sharding((*parameter_axes, None)),
            grouped_values_sharding=self._resolve_sharding((*parameter_axes, None, grouped_axis)),
            sharding_config=self.sharding_config,
        )

    @property
    def _packed_scales(self) -> Float[Array, "*components rows groups"]:
        weight_axes = self.spec.layout.weight_partition(self.master_scales.ndim - 2, is_sharded=self.is_sharded)
        *parameter_axes, _grouped_axis = weight_axes
        return pack_e4m3_scales(
            self.master_scales,
            out_sharding=self._resolve_sharding((*parameter_axes, None)),
        )

    @property
    def _packed_bias_indices(
        self,
    ) -> UInt8[Array, "*components rows packed_groups"] | None:
        return _master_biases_to_packed_bias_indices(
            self.master_biases,
            bits=self.spec.bits,
            bias_bits=self.spec.bias_bits,
            group_size=self.spec.group_size,
            sharding_config=self.sharding_config,
        )

    def astype(self, dtype: DTypeLike) -> "LloydMaxMatrixForTraining":
        master_biases = self.master_biases
        if master_biases is not None:
            master_biases = master_biases.astype(dtype)
        return LloydMaxMatrixForTraining(
            spec=self.spec,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
            master_weights=self.master_weights.astype(dtype),
            master_scales=self.master_scales.astype(dtype),
            master_biases=master_biases,
        )

    def load_exported(
        self,
        exported_data: ExportResults,
        *,
        prefix: ParameterPath | None = None,
    ) -> LloydMaxMatrix:
        return self._load_exported(
            exported_data,
            implementation=CompressionImplementation.TRAINING,
            prefix=prefix,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> LloydMaxMatrix:
        if implementation == CompressionImplementation.TRAINING:
            return self
        return self._from_packed_parameters(implementation)

    def _weights_for_forward(
        self,
        dtype: DTypeLike,
        keychain: Keychain | None,
        forward_pass_config: MatmulConfig,
        row_index: int | Int[Array, "*batch"] | None = None,
    ) -> Float[Array, "... cols"]:
        weights = self.master_weights
        scales = self.master_scales
        master_biases = self.master_biases
        if row_index is not None:
            weights = lookup_sharded_indices(weights, row_index)
            scales = lookup_sharded_indices(scales, row_index)
            if master_biases is not None:
                master_biases = lookup_sharded_indices(master_biases, row_index)
        scales = scales.astype(dtype)
        if row_index is None:
            weight_axes = self.spec.layout.weight_partition(scales.ndim - 2, is_sharded=self.is_sharded)
            *parameter_axes, grouped_axis = weight_axes
            weight_sharding = self._resolve_sharding(weight_axes)
            scale_sharding = self._resolve_sharding((*parameter_axes, None))
            grouped_values_sharding = self._resolve_sharding((*parameter_axes, None, grouped_axis))
        else:
            weight_sharding = None
            scale_sharding = None
            grouped_values_sharding = None
        scale_values = pack_e4m3_scales(scales, out_sharding=scale_sharding).astype(scales.dtype)
        if master_biases is not None:
            master_biases = master_biases.astype(dtype)
        return _master_weights_to_quantized_weights(
            weights.astype(dtype),
            scale_values,
            master_biases=master_biases,
            bits=self.spec.bits,
            bias_bits=self.spec.bias_bits,
            group_size=self.spec.group_size,
            keychain=keychain,
            gradient_estimator=forward_pass_config.gradient_estimator,
            bias_sharding=scale_sharding,
            out_sharding=weight_sharding,
            grouped_values_sharding=grouped_values_sharding,
        )


class LloydMaxMatrixForInference(LloydMaxMatrix):
    dtype_: DTypeLike = field(static=True)
    packed_weight_indices: UInt8[Array, "*components rows packed_cols"]
    packed_scales: Float[Array, "*components rows groups"]
    packed_bias_indices: UInt8[Array, "*components rows packed_groups"] | None

    @property
    def shape(self) -> tuple[int, ...]:
        *leading_dims, groups = self.packed_scales.shape
        return (*leading_dims, groups * self.spec.group_size)

    @property
    def dtype(self) -> DTypeLike:
        return self.dtype_

    @property
    def _packed_weight_indices(self) -> UInt8[Array, "*components rows packed_cols"]:
        return self.packed_weight_indices

    @property
    def _packed_scales(self) -> Float[Array, "*components rows groups"]:
        return self.packed_scales

    @property
    def _packed_bias_indices(
        self,
    ) -> UInt8[Array, "*components rows packed_groups"] | None:
        return self.packed_bias_indices

    def astype(self, dtype: DTypeLike) -> "LloydMaxMatrixForInference":
        return LloydMaxMatrixForInference(
            spec=self.spec,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
            dtype_=jnp.dtype(dtype),
            packed_weight_indices=self.packed_weight_indices,
            packed_scales=self.packed_scales,
            packed_bias_indices=self.packed_bias_indices,
        )

    def load_exported(
        self,
        exported_data: ExportResults,
        *,
        prefix: ParameterPath | None = None,
    ) -> LloydMaxMatrix:
        return self._load_exported(
            exported_data,
            implementation=CompressionImplementation.INFERENCE,
            prefix=prefix,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> LloydMaxMatrix:
        if implementation == CompressionImplementation.INFERENCE:
            return self
        return self._from_packed_parameters(implementation)

    def _weights_for_forward(
        self,
        dtype: DTypeLike,
        keychain: Keychain | None,  # noqa: ARG002
        forward_pass_config: MatmulConfig,  # noqa: ARG002
        row_index: int | Int[Array, "*batch"] | None = None,
    ) -> Float[Array, "... cols"]:
        packed_weight_indices = self.packed_weight_indices
        packed_scales = self.packed_scales
        packed_bias_indices = self.packed_bias_indices
        if row_index is not None:
            packed_weight_indices = lookup_sharded_indices(packed_weight_indices, row_index)
            packed_scales = lookup_sharded_indices(packed_scales, row_index)
            if packed_bias_indices is not None:
                packed_bias_indices = lookup_sharded_indices(packed_bias_indices, row_index)
        if row_index is None:
            weight_axes = self.spec.layout.weight_partition(packed_scales.ndim - 2, is_sharded=self.is_sharded)
            *parameter_axes, grouped_axis = weight_axes
            weight_sharding = self._resolve_sharding(weight_axes)
            parameter_sharding = self._resolve_sharding((*parameter_axes, None))
            grouped_values_sharding = self._resolve_sharding((*parameter_axes, None, grouped_axis))
        else:
            weight_sharding = None
            parameter_sharding = None
            grouped_values_sharding = None
        *_, groups = packed_scales.shape
        if packed_bias_indices is None:
            master_biases = None
        else:
            assert self.spec.bias_bits is not None
            bias_indices = unpack_uint8_to_uint(
                packed_bias_indices,
                bits=self.spec.bias_bits,
                unpacked_last_axis_dim=groups,
            )
            master_biases = _bias_indices_to_master_biases(
                bias_indices,
                bits=self.spec.bits,
                bias_bits=self.spec.bias_bits,
                group_size=self.spec.group_size,
                dtype=dtype,
                out_sharding=parameter_sharding,
            )
        unpacked_weight_indices = unpack_uint8_to_uint(
            packed_weight_indices,
            bits=self.spec.bits,
            unpacked_last_axis_dim=groups * self.spec.group_size,
        )
        return _unpacked_weight_indices_to_master_weights(
            unpacked_weight_indices,
            packed_scales.astype(dtype),
            master_biases=master_biases,
            dtype=dtype,
            bits=self.spec.bits,
            bias_bits=self.spec.bias_bits,
            group_size=self.spec.group_size,
            grouped_values_sharding=grouped_values_sharding,
            out_sharding=weight_sharding,
        )
