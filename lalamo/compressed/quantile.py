from abc import abstractmethod
from dataclasses import dataclass
from functools import cache, cached_property
from typing import Literal

import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
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
    make_sharding,
    reshard_as,
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
from .utils.grouping import group_by_last_axis
from .utils.packing import pack_uint_to_uint8, unpack_uint8_to_uint
from .utils.rounding import (
    lut_values_at,
    pack_e4m3_scales,
    round_to_sorted_lut_table,
    round_to_sorted_lut_table_indices,
)
from .utils.yaqa import yaqa_round_weights

__all__ = [
    "QuantileMatrix",
    "QuantileMatrixForInference",
    "QuantileMatrixForTraining",
    "QuantileSpec",
]


def _sample_values_to_lloyd_lut_values(
    values: Float[Array, " samples"],
    weights: Float[Array, " samples"],
    *,
    num_levels: int,
    steps: int,
    initial_centers: Float[Array, " levels"] | None = None,
) -> Float[Array, " levels"]:
    if initial_centers is None:
        sorted_indices = jnp.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]
        cumulative_weights = jnp.cumsum(sorted_weights)
        total_weight = cumulative_weights[-1]
        center_weight_targets = (jnp.arange(num_levels) + 0.5) * total_weight / num_levels
        center_indices = jnp.searchsorted(cumulative_weights, center_weight_targets, side="left", method="compare_all")
        center_indices = jnp.minimum(center_indices, values.size - 1)
        centers = sorted_values[center_indices]
    else:
        centers = initial_centers
    for _ in range(steps):
        thresholds = (centers[:-1] + centers[1:]) / 2
        assignments = jnp.searchsorted(thresholds, values, side="left", method="compare_all")
        weight_sums = jnp.bincount(assignments, weights=weights, length=num_levels)
        value_sums = jnp.bincount(assignments, weights=weights * values, length=num_levels)
        safe_weight_sums = jnp.maximum(weight_sums, jnp.finfo(values.dtype).tiny)
        centers = jnp.where(weight_sums > 0, value_sums / safe_weight_sums, centers)
        centers = jnp.sort(centers)
    return centers


def _unbiased_codebook_values(
    bits: int,
    normalized_weights: Float[Array, "groups group_size"],
    scale_values: Float[Array, "groups 1"],
) -> Float[Array, " levels"]:
    num_levels = 2**bits
    flat_normalized_weights = normalized_weights.reshape(-1)
    unweighted_centers = _sample_values_to_lloyd_lut_values(
        flat_normalized_weights,
        jnp.ones_like(flat_normalized_weights),
        num_levels=num_levels,
        steps=32,
    )
    half_levels = num_levels // 2
    negative_centers = unweighted_centers[:half_levels]
    positive_centers = unweighted_centers[half_levels:]
    initial_positive_values = (jnp.flip(jnp.abs(negative_centers)) + positive_centers) / 2
    initial_positive_values = jnp.sort(initial_positive_values)
    weights = jnp.broadcast_to(jnp.square(scale_values), normalized_weights.shape)
    positive_values = _sample_values_to_lloyd_lut_values(
        jnp.abs(flat_normalized_weights),
        weights.reshape(-1),
        num_levels=half_levels,
        steps=32,
        initial_centers=initial_positive_values,
    )
    return jnp.concatenate([-jnp.flip(positive_values), positive_values])


def _weighted_bias_costs(
    normalized_weights: Float[Array, "groups group_size"],
    scale_values: Float[Array, "groups 1"],
    codebook: Float[Array, " levels"],
    bias_candidates: Float[Array, " candidates"],
    chunk_size: int,
) -> Float[Array, " groups candidates"]:
    block_weights = jnp.square(scale_values[:, 0])
    weighted_mse_chunks = []
    for candidate_start in range(0, bias_candidates.size, chunk_size):
        candidate_stop = candidate_start + chunk_size
        candidate_biases = bias_candidates[candidate_start:candidate_stop]
        shifted_weights = normalized_weights[:, None, :] + candidate_biases[None, :, None]
        shifted_weight_indices = round_to_sorted_lut_table_indices(
            shifted_weights,
            codebook,
            gradient_estimator=GradientEstimator.DETERMINISTIC_ROUNDING,
        )
        code_values = lut_values_at(shifted_weight_indices, codebook)
        residuals = code_values - candidate_biases[None, :, None] - normalized_weights[:, None, :]
        block_mse = jnp.mean(residuals * residuals, axis=-1)
        weighted_mse_chunks.append(block_mse * block_weights[:, None])
    return jnp.concatenate(weighted_mse_chunks, axis=-1)


def _update_codebook(
    normalized_weights: Float[Array, "groups group_size"],
    scale_values: Float[Array, "groups 1"],
    biases: Float[Array, " groups"],
    codebook: Float[Array, " levels"],
) -> Float[Array, " levels"]:
    shifted_weights = normalized_weights + biases[:, None]
    weight_indices = round_to_sorted_lut_table_indices(
        shifted_weights,
        codebook,
        gradient_estimator=GradientEstimator.DETERMINISTIC_ROUNDING,
    ).astype(jnp.int32)
    (num_levels,) = codebook.shape
    half_levels = num_levels // 2
    pair_indices = jnp.where(
        weight_indices < half_levels, half_levels - 1 - weight_indices, weight_indices - half_levels
    )
    signs = jnp.where(weight_indices < half_levels, -1, 1).astype(shifted_weights.dtype)
    weights = jnp.broadcast_to(jnp.square(scale_values), normalized_weights.shape)
    pair_weights = jnp.bincount(pair_indices.reshape(-1), weights=weights.reshape(-1), length=half_levels)
    pair_sums = jnp.bincount(
        pair_indices.reshape(-1),
        weights=(weights * signs * shifted_weights).reshape(-1),
        length=half_levels,
    )
    safe_pair_weights = jnp.maximum(pair_weights, jnp.finfo(codebook.dtype).tiny)
    positive_values = pair_sums / safe_pair_weights
    positive_values = jnp.where(pair_weights > 0, positive_values, codebook[half_levels:])
    positive_values = jnp.maximum(positive_values, jnp.finfo(codebook.dtype).tiny)
    positive_values = jnp.sort(positive_values)
    return jnp.concatenate([-jnp.flip(positive_values), positive_values])


def _update_bias_indices(
    weighted_bias_costs: Float[Array, " groups candidates"],
    center_indices: Int[Array, " bias_levels"],
) -> Int[Array, " bias_levels"]:
    (num_bias_levels,) = center_indices.shape
    center_costs = weighted_bias_costs[:, center_indices]
    assignments = jnp.argmin(center_costs, axis=-1)
    assignment_counts = jnp.bincount(assignments, length=num_bias_levels)
    assignment_weights = jax.nn.one_hot(assignments, num_bias_levels, dtype=weighted_bias_costs.dtype)
    candidate_costs = assignment_weights.T @ weighted_bias_costs
    updated_center_indices = jnp.argmin(candidate_costs, axis=-1)
    center_indices = jnp.where(assignment_counts > 0, updated_center_indices, center_indices)
    return jnp.sort(center_indices)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _master_weights_to_master_scales(
    master_weights: Float[Array, "... cols"],
    group_size: int,
) -> Float[Array, "... groups"]:
    grouped_weights = group_by_last_axis(master_weights, group_size=group_size)
    return jnp.max(jnp.abs(grouped_weights), axis=-1)


@cache
def _lut_values(
    bits: int,
    group_size: int,
    bias_bits: int | None,
) -> tuple[tuple[float, ...], tuple[float, ...] | None]:
    sample_groups = 32_768
    key = jax.random.PRNGKey(0)
    samples = jax.random.normal(key, (sample_groups, group_size), dtype=jnp.float32)
    absmax = jnp.max(jnp.abs(samples), axis=-1, keepdims=True)
    scale_values = pack_e4m3_scales(absmax).astype(samples.dtype)
    safe_scale_values = jnp.where(scale_values == 0, 1, scale_values)
    normalized_weights = samples / safe_scale_values
    codebook = _unbiased_codebook_values(bits, normalized_weights, scale_values)
    if bias_bits is None:
        codebook_values = tuple(float(value) for value in jax.device_get(codebook))
        return codebook_values, None

    num_bias_levels = 2**bias_bits
    bias_candidates = jnp.linspace(-1, 1, 512, dtype=jnp.float32)
    bias_candidate_chunk_size = 64
    weighted_bias_costs = _weighted_bias_costs(
        normalized_weights,
        scale_values,
        codebook,
        bias_candidates,
        bias_candidate_chunk_size,
    )
    best_bias_indices = jnp.argmin(weighted_bias_costs, axis=-1)
    best_biases = bias_candidates[best_bias_indices]
    centers = _sample_values_to_lloyd_lut_values(
        best_biases,
        jnp.square(scale_values[:, 0]),
        num_levels=num_bias_levels,
        steps=16,
    )
    candidate_thresholds = (bias_candidates[:-1] + bias_candidates[1:]) / 2
    center_indices = jnp.searchsorted(candidate_thresholds, centers, side="right", method="compare_all")
    center_indices = jnp.minimum(center_indices, bias_candidates.size - 1)

    for _ in range(8):
        bias_table = bias_candidates[center_indices]
        center_costs = weighted_bias_costs[:, center_indices]
        assigned_bias_indices = jnp.argmin(center_costs, axis=-1)
        biases = lut_values_at(assigned_bias_indices.astype(jnp.uint8), bias_table)
        codebook = _update_codebook(normalized_weights, scale_values, biases, codebook)
        weighted_bias_costs = _weighted_bias_costs(
            normalized_weights,
            scale_values,
            codebook,
            bias_candidates,
            bias_candidate_chunk_size,
        )
        center_indices = _update_bias_indices(weighted_bias_costs, center_indices)

    codebook_values = tuple(float(value) for value in jax.device_get(codebook))
    bias_values = tuple(float(value) for value in jax.device_get(bias_candidates[center_indices]))
    return codebook_values, bias_values


def _codebook_values(
    bits: int,
    group_size: int,
    bias_bits: int | None,
) -> tuple[float, ...]:
    codebook_values, _ = _lut_values(bits, group_size, bias_bits)
    return codebook_values


def _codebook(
    bits: int,
    group_size: int,
    dtype: DTypeLike,
    bias_bits: int | None = None,
) -> Float[Array, " levels"]:
    return jnp.array(_codebook_values(bits, group_size, bias_bits), dtype=dtype)


def _bias_lut_values(
    group_size: int,
    bias_bits: int,
    bits: int,
) -> tuple[float, ...]:
    _, bias_values = _lut_values(bits, group_size, bias_bits)
    assert bias_values is not None
    return bias_values


def _bias_lut(
    *,
    group_size: int,
    bias_bits: int,
    bits: int,
    dtype: DTypeLike,
) -> Float[Array, " levels"]:
    return jnp.array(_bias_lut_values(group_size, bias_bits, bits), dtype=dtype)


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
) -> UInt8[Array, "... groups group_size"]:
    codebook = _codebook(bits, group_size, master_weights.dtype, bias_bits=bias_bits)
    safe_scale_values = jnp.where(scale_values == 0, 1, scale_values)
    grouped_weights = group_by_last_axis(master_weights, group_size=group_size)
    normalized_weights = grouped_weights / safe_scale_values.astype(master_weights.dtype)[..., None]
    if master_biases is not None:
        normalized_weights = normalized_weights + master_biases[..., None]

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
) -> Float[Array, "... cols"]:
    codebook = _codebook(bits, group_size, dtype, bias_bits=bias_bits)
    *leading_dims, cols = unpacked_weight_indices.shape
    groups = cols // group_size
    grouped_weight_indices = unpacked_weight_indices.reshape(*leading_dims, groups, group_size)
    grouped_values = lut_values_at(grouped_weight_indices, codebook)
    if master_biases is not None:
        grouped_values = grouped_values - master_biases[..., None]

    grouped_master_weights = grouped_values * scale_values.astype(dtype)[..., None]
    *leading_dims, groups, group_width = grouped_master_weights.shape
    return grouped_master_weights.reshape(*leading_dims, groups * group_width)


def _master_weights_to_master_biases(
    master_weights: Float[Array, "... cols"],
    scale_values: Float[Array, "... groups"],
    *,
    bits: int,
    bias_bits: int,
    group_size: int,
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
    candidate_bias_shape = (1,) * (normalized_weights.ndim - 1) + (bias_levels, 1)
    candidate_biases = bias_table.reshape(candidate_bias_shape)
    shifted_weights = normalized_weights[..., None, :] + candidate_biases
    shifted_weight_indices = round_to_sorted_lut_table_indices(
        shifted_weights,
        codebook,
        gradient_estimator=GradientEstimator.DETERMINISTIC_ROUNDING,
    )
    code_values = lut_values_at(shifted_weight_indices, codebook)
    residuals = code_values - candidate_biases - normalized_weights[..., None, :]
    bias_indices = jnp.argmin(jnp.mean(residuals * residuals, axis=-1), axis=-1)
    return lut_values_at(bias_indices, bias_table)


def _bias_indices_to_master_biases(
    bias_indices: UInt8[Array, "... groups"],
    *,
    bits: int,
    bias_bits: int,
    group_size: int,
    dtype: DTypeLike,
) -> Float[Array, "... groups"]:
    bias_table = _bias_lut(
        group_size=group_size,
        bias_bits=bias_bits,
        bits=bits,
        dtype=dtype,
    )
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
    return pack_uint_to_uint8(bias_indices, bias_bits)


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
    return grouped_quantized_weights.reshape(*leading_dims, groups * group_width)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _master_weights_to_packed_weight_indices(
    master_weights: Float[Array, "*components rows cols"],
    master_scales: Float[Array, "*components rows groups"],
    *,
    master_biases: Float[Array, "*components rows groups"] | None,
    bits: int,
    bias_bits: int | None,
    group_size: int,
) -> UInt8[Array, "*components rows packed_cols"]:
    packed_scales = pack_e4m3_scales(master_scales)
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
        )
    grouped_weight_indices = _master_weights_to_grouped_weight_indices(
        master_weights,
        packed_scales.astype(master_weights.dtype),
        master_biases=master_biases,
        bits=bits,
        bias_bits=bias_bits,
        group_size=group_size,
    )
    *leading_dims, groups, group_width = grouped_weight_indices.shape
    unpacked_weight_indices = grouped_weight_indices.reshape(*leading_dims, groups * group_width)
    return pack_uint_to_uint8(unpacked_weight_indices, bits)


@dataclass(frozen=True)
class QuantileSpec(QuantizedSpec):
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
        sample_groups = 32_768
        key = jax.random.PRNGKey(0)
        weights = jax.random.normal(key, (sample_groups, self.group_size), dtype=jnp.float32)
        master_scales = _master_weights_to_master_scales(weights, group_size=self.group_size)
        scale_values = pack_e4m3_scales(master_scales).astype(weights.dtype)
        master_biases = None
        if self.bias_bits is not None:
            master_biases = _master_weights_to_master_biases(
                weights,
                scale_values,
                bits=self.bits,
                bias_bits=self.bias_bits,
                group_size=self.group_size,
            )

        quantized_weights = _master_weights_to_quantized_weights(
            weights,
            scale_values,
            master_biases=master_biases,
            bits=self.bits,
            bias_bits=self.bias_bits,
            group_size=self.group_size,
        )
        distortion = jnp.mean(jnp.square(weights - quantized_weights))
        return float(jax.device_get(distortion))

    def compress(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        is_sharded: bool = True,
    ) -> "QuantileMatrix":
        if preconditioner is not None:
            weights = yaqa_round_weights(weights, preconditioner, self, is_sharded=is_sharded)

        stored_weights = self.layout.from_output_input(weights, is_sharded=is_sharded)
        master_scales = _master_weights_to_master_scales(stored_weights, group_size=self.group_size)
        weight_partition = self.layout.weight_partition(master_scales.ndim - 2, is_sharded=is_sharded)
        parameter_sharding = make_sharding((*weight_partition[:-1], None))
        master_scales = with_sharding(master_scales, parameter_sharding)
        packed_scales = pack_e4m3_scales(master_scales)
        master_biases = None
        if self.bias_bits is not None:
            scale_values = packed_scales.astype(stored_weights.dtype)
            master_biases = _master_weights_to_master_biases(
                stored_weights,
                scale_values,
                bits=self.bits,
                bias_bits=self.bias_bits,
                group_size=self.group_size,
            )
        if implementation == CompressionImplementation.TRAINING:
            return QuantileMatrixForTraining(
                spec=self,
                is_sharded=is_sharded,
                master_weights=stored_weights,
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
        )
        packed_bias_indices = _master_biases_to_packed_bias_indices(
            master_biases,
            bits=self.bits,
            bias_bits=self.bias_bits,
            group_size=self.group_size,
        )
        return self.from_packed_parameters(
            packed_weight_indices=packed_weight_indices,
            packed_scales=packed_scales,
            packed_bias_indices=packed_bias_indices,
            dtype=stored_weights.dtype,
            implementation=CompressionImplementation.INFERENCE,
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
        is_sharded: bool = True,
    ) -> "QuantileMatrix":
        if self.bias_bits is not None and packed_bias_indices is None:
            raise ValueError("Quantile biases require packed bias indices")

        weight_partition = self.layout.weight_partition(packed_scales.ndim - 2, is_sharded=is_sharded)
        parameter_sharding = make_sharding((*weight_partition[:-1], None))
        packed_weight_indices = with_sharding(packed_weight_indices, parameter_sharding)
        packed_scales = with_sharding(packed_scales, parameter_sharding)
        if packed_bias_indices is not None:
            packed_bias_indices = with_sharding(packed_bias_indices, parameter_sharding)

        if implementation == CompressionImplementation.INFERENCE:
            return QuantileMatrixForInference(
                spec=self,
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
        )
        return QuantileMatrixForTraining(
            spec=self,
            is_sharded=is_sharded,
            master_weights=with_sharding(weights, make_sharding(weight_partition)),
            master_scales=with_sharding(packed_scales.astype(dtype), parameter_sharding),
            master_biases=master_biases,
        )


class QuantileMatrix(EmbeddingMatrix[QuantileSpec]):
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
        index: int | Int[Array, ""] | None = None,
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
        expored_data: ExportResults,
        allow_dtype_cast: bool = False,  # noqa: ARG002
        *,
        implementation: CompressionImplementation,
        prefix: ParameterPath | None = None,
    ) -> "QuantileMatrix":
        if prefix is None:
            prefix = ParameterPath()
        loaded_spec = WeightMatrixSpec.from_json(expored_data.metadata[prefix / "spec"])
        if loaded_spec != self.spec:
            raise ValueError(f"WeightMatrix spec mismatch: expected {self.spec}, got {loaded_spec}")

        packed_bias_indices = None
        if self.spec.bias_bits is not None:
            packed_bias_indices = load_as(
                self._packed_bias_indices,
                expored_data.arrays[prefix / "bias_indices"],
                allow_dtype_cast=False,
            )
        return self.spec.from_packed_parameters(
            packed_weight_indices=load_as(
                self._packed_weight_indices,
                expored_data.arrays[prefix / "weights"],
                allow_dtype_cast=False,
            ),
            packed_scales=load_as(
                self._packed_scales,
                expored_data.arrays[prefix / "scales"],
                allow_dtype_cast=False,
            ),
            packed_bias_indices=packed_bias_indices,
            dtype=self.dtype,
            implementation=implementation,
            is_sharded=self.is_sharded,
        )

    def _from_packed_parameters(self, implementation: CompressionImplementation) -> "QuantileMatrix":
        return self.spec.from_packed_parameters(
            packed_weight_indices=self._packed_weight_indices,
            packed_scales=self._packed_scales,
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
    ) -> "QuantileMatrix": ...

    @abstractmethod
    def switch_implementation(self, implementation: CompressionImplementation) -> "QuantileMatrix": ...

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(layout=self.spec.layout).compress(self.decompress(), is_sharded=self.is_sharded)

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        weights = self._weights_for_forward(self.dtype, None, MatmulConfig())
        return self.spec.layout.to_output_input(weights)

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
        weights = self._weights_for_forward(vector.dtype, keychain, forward_pass_config)
        layout = self.spec.layout
        if transposed:
            layout = layout.transpose()
        with use_dot_algorithm_preset(forward_pass_config.precision):
            return reshard_as(layout.matmul(weights, vector), vector)


class QuantileMatrixForTraining(QuantileMatrix):
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
        return _master_weights_to_packed_weight_indices(
            self.master_weights,
            self.master_scales,
            master_biases=self.master_biases,
            bits=self.spec.bits,
            bias_bits=self.spec.bias_bits,
            group_size=self.spec.group_size,
        )

    @property
    def _packed_scales(self) -> Float[Array, "*components rows groups"]:
        return pack_e4m3_scales(self.master_scales)

    @property
    def _packed_bias_indices(
        self,
    ) -> UInt8[Array, "*components rows packed_groups"] | None:
        return _master_biases_to_packed_bias_indices(
            self.master_biases,
            bits=self.spec.bits,
            bias_bits=self.spec.bias_bits,
            group_size=self.spec.group_size,
        )

    def astype(self, dtype: DTypeLike) -> "QuantileMatrixForTraining":
        master_biases = self.master_biases
        if master_biases is not None:
            master_biases = master_biases.astype(dtype)
        return QuantileMatrixForTraining(
            spec=self.spec,
            is_sharded=self.is_sharded,
            master_weights=self.master_weights.astype(dtype),
            master_scales=self.master_scales.astype(dtype),
            master_biases=master_biases,
        )

    def load_exported(
        self,
        expored_data: ExportResults,
        allow_dtype_cast: bool = False,
        *,
        prefix: ParameterPath | None = None,
    ) -> QuantileMatrix:
        return self._load_exported(
            expored_data,
            allow_dtype_cast=allow_dtype_cast,
            implementation=CompressionImplementation.TRAINING,
            prefix=prefix,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> QuantileMatrix:
        if implementation == CompressionImplementation.TRAINING:
            return self
        return self._from_packed_parameters(implementation)

    def _weights_for_forward(
        self,
        dtype: DTypeLike,
        keychain: Keychain | None,
        forward_pass_config: MatmulConfig,
        index: int | Int[Array, ""] | None = None,
    ) -> Float[Array, "... cols"]:
        weights = self.master_weights
        scales = self.master_scales
        master_biases = self.master_biases
        if index is not None:
            weights = weights[index, :]
            scales = scales[index, :]
            if master_biases is not None:
                master_biases = master_biases[index, :]
        scales = scales.astype(dtype)
        scale_values = pack_e4m3_scales(scales).astype(scales.dtype)
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
        )


class QuantileMatrixForInference(QuantileMatrix):
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

    @property
    def _master_biases(self) -> Float[Array, "*components rows groups"] | None:
        if self.packed_bias_indices is None:
            return None
        assert self.spec.bias_bits is not None
        *_, groups = self.packed_scales.shape
        bias_indices = unpack_uint8_to_uint(
            self.packed_bias_indices,
            bits=self.spec.bias_bits,
            unpacked_last_axis_dim=groups,
        )
        return _bias_indices_to_master_biases(
            bias_indices,
            bits=self.spec.bits,
            bias_bits=self.spec.bias_bits,
            group_size=self.spec.group_size,
            dtype=self.dtype,
        )

    def astype(self, dtype: DTypeLike) -> "QuantileMatrixForInference":
        return QuantileMatrixForInference(
            spec=self.spec,
            is_sharded=self.is_sharded,
            dtype_=jnp.dtype(dtype),
            packed_weight_indices=self.packed_weight_indices,
            packed_scales=self.packed_scales,
            packed_bias_indices=self.packed_bias_indices,
        )

    def load_exported(
        self,
        expored_data: ExportResults,
        allow_dtype_cast: bool = False,
        *,
        prefix: ParameterPath | None = None,
    ) -> QuantileMatrix:
        return self._load_exported(
            expored_data,
            allow_dtype_cast=allow_dtype_cast,
            implementation=CompressionImplementation.INFERENCE,
            prefix=prefix,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> QuantileMatrix:
        if implementation == CompressionImplementation.INFERENCE:
            return self
        return self._from_packed_parameters(implementation)

    def _weights_for_forward(
        self,
        dtype: DTypeLike,
        keychain: Keychain | None,  # noqa: ARG002
        forward_pass_config: MatmulConfig,  # noqa: ARG002
        index: int | Int[Array, ""] | None = None,
    ) -> Float[Array, "... cols"]:
        master_biases = self._master_biases
        packed_weight_indices = self.packed_weight_indices
        packed_scales = self.packed_scales
        if index is not None:
            packed_weight_indices = packed_weight_indices[index, :]
            packed_scales = packed_scales[index, :]
            if master_biases is not None:
                master_biases = master_biases[index, :]
        *_, groups = packed_scales.shape
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
        )
