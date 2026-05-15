from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property, partial
from itertools import product
from typing import Literal, NamedTuple, Self

import jax
import jax.numpy as jnp
from einops import rearrange
from jax.lax import stop_gradient
from jax.sharding import PartitionSpec
from jaxtyping import Array, DTypeLike, Float, Int, Key, UInt8, UInt16

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

from .data.distortion import distortion_estimate
from .quantized_spec import QuantizedSpec
from .utils.yaqa import yaqa_round_weights

__all__ = [
    "E8PMatrix",
    "E8PMatrixForInference",
    "E8PMatrixForTraining",
    "E8PSpec",
]

_E8P_VECTOR_SIZE = 8
_E8P_CODEBOOK_SIZE = 1 << 16
_E8P_DEFAULT_SCALE = {
    2: 1.03,
    3: 0.98,
    4: 1.03,
}
_E8P_DEFAULT_RESIDUAL_SCALE = {
    3: 2.04,
    4: 3.45,
}
_E8P_SHUFFLE_MAP = (0, 4, 1, 5, 2, 6, 3, 7)

_E8P_NORM12_DOUBLED = (
    (3, 1, 1, 1, 3, 3, 3, 3),
    (1, 3, 1, 1, 3, 3, 3, 3),
    (1, 1, 3, 1, 3, 3, 3, 3),
    (1, 1, 1, 3, 3, 3, 3, 3),
    (3, 3, 3, 1, 3, 3, 1, 1),
    (3, 3, 3, 1, 3, 1, 3, 1),
    (3, 3, 3, 1, 1, 3, 3, 1),
    (3, 3, 3, 1, 3, 1, 1, 3),
    (3, 3, 3, 1, 1, 3, 1, 3),
    (3, 3, 3, 1, 1, 1, 3, 3),
    (3, 3, 1, 3, 3, 3, 1, 1),
    (3, 3, 1, 3, 3, 1, 3, 1),
    (3, 3, 1, 3, 1, 3, 3, 1),
    (3, 3, 1, 3, 3, 1, 1, 3),
    (3, 3, 1, 3, 1, 3, 1, 3),
    (3, 3, 1, 3, 1, 1, 3, 3),
    (3, 1, 3, 3, 3, 3, 1, 1),
    (3, 1, 3, 3, 3, 1, 3, 1),
    (3, 1, 3, 3, 1, 3, 3, 1),
    (3, 1, 3, 3, 3, 1, 1, 3),
    (3, 1, 3, 3, 1, 3, 1, 3),
    (1, 3, 3, 3, 1, 1, 3, 3),
    (1, 3, 3, 3, 3, 3, 1, 1),
    (1, 3, 3, 3, 3, 1, 3, 1),
    (1, 3, 3, 3, 1, 3, 3, 1),
    (1, 3, 3, 3, 3, 1, 1, 3),
    (1, 3, 3, 3, 1, 3, 1, 3),
    (1, 1, 3, 3, 1, 3, 3, 3),
    (3, 3, 1, 1, 3, 3, 3, 1),
)


def _abs_grid_values() -> tuple[tuple[float, ...], ...]:
    base_values = (0.5, 1.5, 2.5, 3.5)
    d8_abs = tuple(
        values
        for values in product(base_values, repeat=_E8P_VECTOR_SIZE)
        if sum(value * value for value in values) <= 10
    )
    norm12 = tuple(tuple(value / 2 for value in row) for row in _E8P_NORM12_DOUBLED)
    return (*d8_abs, *norm12)


def _packed_abs_grid_values() -> tuple[int, ...]:
    result = []
    for values in _abs_grid_values():
        doubled_shifted = [int(2 * value + 8) for value in values]
        shuffled = [doubled_shifted[index] for index in (0, 2, 4, 6, 1, 3, 5, 7)]
        if sum(values) % 2 == 1:
            shuffled[7] = 16 - shuffled[7]
        packed = 0
        for index, value in enumerate(shuffled):
            packed |= value << (4 * index)
        result.append(packed)
    return tuple(result)


def _e81b_grid_values() -> tuple[tuple[float, ...], ...]:
    zero = ((0.0,) * _E8P_VECTOR_SIZE,)
    integer_roots = tuple(
        tuple(
            (first_sign if coord == first_axis else second_sign if coord == second_axis else 0.0)
            for coord in range(_E8P_VECTOR_SIZE)
        )
        for first_axis in range(_E8P_VECTOR_SIZE)
        for second_axis in range(first_axis + 1, _E8P_VECTOR_SIZE)
        for first_sign in (-1.0, 1.0)
        for second_sign in (-1.0, 1.0)
    )
    half_roots = tuple(
        tuple(0.5 if (signs >> coord) & 1 else -0.5 for coord in range(_E8P_VECTOR_SIZE))
        for signs in range(1 << _E8P_VECTOR_SIZE)
        if signs.bit_count() % 2 == 0
    )
    norm4_roots = tuple(
        tuple((sign if coord == axis else 0.0) for coord in range(_E8P_VECTOR_SIZE))
        for axis in range(_E8P_VECTOR_SIZE)
        for sign in ((2.0,) if axis == _E8P_VECTOR_SIZE - 1 else (-2.0, 2.0))
    )
    return (*zero, *integer_roots, *half_roots, *norm4_roots)


_E8P_PACKED_ABS_GRID_VALUES = _packed_abs_grid_values()
_E81B_GRID_VALUES = _e81b_grid_values()


class E8PParameters(NamedTuple):
    scale: Float[Array, "*components"]

    @classmethod
    @supports_dummy_arrays()
    def from_weights(
        cls,
        weights: Float[Array, "*components rows cols"],
        *,
        scale_normalization: float,
    ) -> Self:
        rms = jnp.sqrt(jnp.mean(jnp.square(weights.astype(jnp.float32)), axis=(-2, -1)))
        scale = jnp.where(rms > 0, rms / jnp.float32(scale_normalization), jnp.ones_like(rms))
        return cls(scale=scale.astype(weights.dtype))


def _scale_broadcast(
    scale: Array,
    ndim: int,
) -> Array:
    return scale.reshape(*scale.shape, *((1,) * (ndim - scale.ndim)))


def _codebook(dtype: DTypeLike) -> Float[Array, "codes vector"]:
    packed_abs_grid = jnp.array(_E8P_PACKED_ABS_GRID_VALUES, dtype=jnp.uint32)
    code_indices = jnp.arange(_E8P_CODEBOOK_SIZE, dtype=jnp.uint32)
    sign_masks = code_indices & jnp.uint32(0xFF)
    abs_indices = code_indices >> jnp.uint32(8)
    sign_bits = (sign_masks[..., None] >> jnp.arange(_E8P_VECTOR_SIZE, dtype=jnp.uint32)) & jnp.uint32(1)
    parity = jnp.sum(sign_bits, axis=-1).astype(jnp.uint32) & jnp.uint32(1)
    sign_masks = sign_masks ^ parity

    abs_codes = packed_abs_grid[abs_indices]
    shuffled_indices = jnp.array(_E8P_SHUFFLE_MAP, dtype=jnp.uint32)
    abs_values = ((abs_codes[..., None] >> (4 * shuffled_indices)) & jnp.uint32(0xF)).astype(jnp.float32)
    values = (abs_values - 8) * 0.5
    signs = ((sign_masks[..., None] >> shuffled_indices) & jnp.uint32(1)).astype(jnp.bool_)
    values = jnp.where(signs, -values, values)
    values = values + jnp.where(parity[..., None] == 1, -0.25, 0.25)
    return values.astype(dtype)


def _abs_codebook(dtype: DTypeLike) -> Float[Array, "codes vector"]:
    packed_abs_grid = jnp.array(_E8P_PACKED_ABS_GRID_VALUES, dtype=jnp.uint32)
    shuffled_indices = jnp.array(_E8P_SHUFFLE_MAP, dtype=jnp.uint32)
    abs_values = ((packed_abs_grid[..., None] >> (4 * shuffled_indices)) & jnp.uint32(0xF)).astype(jnp.float32)
    return ((abs_values - 8) * 0.5).astype(dtype)


def _e81b_codebook(dtype: DTypeLike) -> Float[Array, "codes vector"]:
    return jnp.array(_E81B_GRID_VALUES, dtype=dtype)


def _round_to_codebook(
    vectors: Float[Array, "... vector"],
) -> tuple[Float[Array, "... vector"], UInt16[Array, "..."]]:
    flat_vectors = vectors.reshape(-1, _E8P_VECTOR_SIZE).astype(jnp.float32)
    abs_codebook = _abs_codebook(jnp.float32)
    abs_codebook_norms = jnp.sum(jnp.square(abs_codebook), axis=-1)
    shuffled_indices = jnp.array(_E8P_SHUFFLE_MAP, dtype=jnp.uint32)

    vector_sum = jnp.sum(flat_vectors, axis=-1)
    best_scores = jnp.full_like(vector_sum, -jnp.inf)
    best_indices = jnp.zeros_like(vector_sum, dtype=jnp.uint16)
    coordinate_indices = jnp.arange(_E8P_VECTOR_SIZE)
    chunk_size = 64

    for shift_index, shift_float in enumerate((0.25, -0.25)):
        shift = jnp.asarray(shift_float, dtype=jnp.float32)
        shift_score = 2 * shift * vector_sum - _E8P_VECTOR_SIZE * shift * shift
        for chunk_start in range(0, len(_E8P_PACKED_ABS_GRID_VALUES), chunk_size):
            chunk_stop = chunk_start + chunk_size
            abs_values = abs_codebook[chunk_start:chunk_stop]
            coefficients = 2 * abs_values[None, :, :] * (flat_vectors[:, None, :] - shift)
            abs_coefficients = jnp.abs(coefficients)
            negative_signs = coefficients < 0
            odd_parity = (jnp.sum(negative_signs, axis=-1) % 2).astype(jnp.bool_)
            flip_indices = jnp.argmin(abs_coefficients, axis=-1)
            flip_mask = coordinate_indices == flip_indices[..., None]
            sign_bits = negative_signs ^ (odd_parity[..., None] & flip_mask)

            sign_scores = jnp.sum(abs_coefficients, axis=-1)
            sign_scores -= jnp.where(odd_parity, 2 * jnp.min(abs_coefficients, axis=-1), 0.0)
            scores = sign_scores - abs_codebook_norms[chunk_start:chunk_stop][None, :] + shift_score[:, None]

            chunk_offsets = jnp.argmax(scores, axis=-1)
            chunk_scores = jnp.take_along_axis(scores, chunk_offsets[:, None], axis=-1)[:, 0]
            chunk_sign_bits = jnp.take_along_axis(sign_bits, chunk_offsets[:, None, None], axis=1)[:, 0, :]
            sign_masks = jnp.sum(chunk_sign_bits.astype(jnp.uint32) << shuffled_indices[None, :], axis=-1)
            if shift_index == 1:
                sign_masks = sign_masks ^ jnp.uint32(1)
            abs_indices = jnp.asarray(chunk_start, dtype=jnp.uint32) + chunk_offsets.astype(jnp.uint32)
            chunk_indices = ((abs_indices << jnp.uint32(8)) | sign_masks).astype(jnp.uint16)

            is_better = chunk_scores > best_scores
            best_scores = jnp.where(is_better, chunk_scores, best_scores)
            best_indices = jnp.where(is_better, chunk_indices, best_indices)

    flat_values = _codebook_values_by_indices(_codebook(vectors.dtype), best_indices)
    return flat_values.reshape(vectors.shape), best_indices.reshape(vectors.shape[:-1])


def _round_to_dense_codebook(
    vectors: Float[Array, "... vector"],
    codebook: Float[Array, "codes vector"],
) -> tuple[Float[Array, "... vector"], Int[Array, "..."]]:
    flat_vectors = vectors.reshape(-1, _E8P_VECTOR_SIZE).astype(jnp.float32)
    codebook = codebook.astype(jnp.float32)
    codebook_norms = jnp.sum(jnp.square(codebook), axis=-1)
    scores = 2 * flat_vectors @ codebook.T - codebook_norms
    flat_indices = jnp.argmax(scores, axis=-1).astype(jnp.int32)
    values_sharding = None
    indices_sharding = sharding_of(flat_indices)
    if flat_indices.ndim > 0 and is_sharded(indices_sharding):
        values_sharding = PartitionSpec(*tuple(indices_sharding.spec), None)
    flat_values = codebook.at[flat_indices].get(out_sharding=values_sharding)
    return flat_values.reshape(vectors.shape).astype(vectors.dtype), flat_indices.reshape(vectors.shape[:-1])


def _codebook_values_by_indices(
    codebook: Float[Array, "codes vector"],
    indices: UInt8[Array, "..."] | UInt16[Array, "..."],
) -> Float[Array, "... vector"]:
    indices = indices.astype(jnp.int32)
    values_sharding = None
    indices_sharding = sharding_of(indices)
    if indices.ndim > 0 and is_sharded(indices_sharding):
        values_sharding = PartitionSpec(*tuple(indices_sharding.spec), None)
    return codebook.at[indices].get(out_sharding=values_sharding)


def _weights_to_grouped_codes(
    weights: Float[Array, "... cols"],
    scale: Float[Array, "*components"],
    *,
    bits: int,
    residual_scale: float,
) -> tuple[Float[Array, "... groups vector"], UInt16[Array, "... groups"], Array | None]:
    if weights.shape[-1] % _E8P_VECTOR_SIZE != 0:
        raise ValueError(f"E8P last axis must be divisible by {_E8P_VECTOR_SIZE}, got {weights.shape[-1]}")
    grouped_weights = rearrange(
        weights / _scale_broadcast(scale, weights.ndim),
        "... (groups vector) -> ... groups vector",
        vector=_E8P_VECTOR_SIZE,
    )
    quantized, codes = _round_to_codebook(grouped_weights)
    if bits == 2:
        return quantized, codes.astype(jnp.uint16), None

    residual = (grouped_weights - quantized) * jnp.asarray(residual_scale, dtype=grouped_weights.dtype)
    if bits == 3:
        residual_quantized, residual_codes = _round_to_dense_codebook(residual, _e81b_codebook(grouped_weights.dtype))
        quantized = quantized + residual_quantized / jnp.asarray(residual_scale, dtype=grouped_weights.dtype)
        return quantized, codes.astype(jnp.uint16), residual_codes.astype(jnp.uint8)

    residual_quantized, residual_codes = _round_to_codebook(residual)
    quantized = quantized + residual_quantized / jnp.asarray(residual_scale, dtype=grouped_weights.dtype)
    return quantized, codes.astype(jnp.uint16), residual_codes.astype(jnp.uint16)


def _codes_to_master_weights(
    codes: UInt16[Array, "... groups"],
    residual_codes: Array | None,
    scale: Float[Array, "*components"],
    *,
    bits: int,
    dtype: DTypeLike,
    residual_scale: float,
) -> Float[Array, "... cols"]:
    e8p_codebook = _codebook(dtype)
    grouped_weights = _codebook_values_by_indices(e8p_codebook, codes)
    if bits == 3:
        if residual_codes is None:
            raise ValueError("3-bit E8P parameters require residual codes.")
        residual_values = _codebook_values_by_indices(_e81b_codebook(dtype), residual_codes)
        grouped_weights = grouped_weights + residual_values / jnp.asarray(residual_scale, dtype=dtype)
    if bits == 4:
        if residual_codes is None:
            raise ValueError("4-bit E8P parameters require residual codes.")
        residual_values = _codebook_values_by_indices(e8p_codebook, residual_codes)
        grouped_weights = grouped_weights + residual_values / jnp.asarray(residual_scale, dtype=dtype)

    scaled_weights = grouped_weights * _scale_broadcast(scale.astype(dtype), grouped_weights.ndim)
    return rearrange(scaled_weights, "... groups vector -> ... (groups vector)")


def _quantized_weights_and_normalized_values(
    weights: Float[Array, "... cols"],
    scale: Float[Array, "*components"],
    *,
    bits: int,
    residual_scale: float,
) -> tuple[Float[Array, "... cols"], Float[Array, "... cols"]]:
    grouped_weights, _codes, _residual_codes = _weights_to_grouped_codes(
        weights,
        stop_gradient(scale),
        bits=bits,
        residual_scale=residual_scale,
    )
    normalized_values = rearrange(grouped_weights, "... groups vector -> ... (groups vector)")
    quantized_weights = normalized_values * _scale_broadcast(scale, weights.ndim)
    return quantized_weights, normalized_values


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _quantize_with_vjp(
    weights: Float[Array, "... cols"],
    scale: Float[Array, "*components"],
    bits: int,
    residual_scale: float,
) -> Float[Array, "... cols"]:
    quantized_weights, _normalized_values = _quantized_weights_and_normalized_values(
        weights,
        scale,
        bits=bits,
        residual_scale=residual_scale,
    )
    return quantized_weights


def _quantize_with_vjp_fwd(
    weights: Float[Array, "... cols"],
    scale: Float[Array, "*components"],
    bits: int,
    residual_scale: float,
) -> tuple[Float[Array, "... cols"], tuple[Float[Array, "... cols"], Float[Array, "*components"]]]:
    quantized_weights, normalized_values = _quantized_weights_and_normalized_values(
        weights,
        scale,
        bits=bits,
        residual_scale=residual_scale,
    )
    return quantized_weights, (normalized_values, scale)


def _quantize_with_vjp_bwd(
    bits: int,  # noqa: ARG001
    residual_scale: float,  # noqa: ARG001
    residuals: tuple[Float[Array, "... cols"], Float[Array, "*components"]],
    gradients: Float[Array, "... cols"],
) -> tuple[Float[Array, "... cols"], Float[Array, "*components"]]:
    normalized_values, scale = residuals
    scale_axes = tuple(range(scale.ndim, gradients.ndim))
    scale_gradients = jnp.sum(gradients * normalized_values, axis=scale_axes)
    return gradients, scale_gradients


_quantize_with_vjp.defvjp(
    _quantize_with_vjp_fwd,
    _quantize_with_vjp_bwd,
)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _quantize(
    weights: Float[Array, "... cols"],
    scale: Float[Array, "*components"],
    *,
    bits: int,
    residual_scale: float,
) -> Float[Array, "... cols"]:
    return _quantize_with_vjp(weights, scale, bits, residual_scale)


@supports_dummy_arrays()
def _master_weights_to_codes(
    weights: Float[Array, "*components rows cols"],
    scale: Float[Array, "*components"],
    *,
    bits: int,
    residual_scale: float,
) -> tuple[UInt16[Array, "*components rows groups"], Array | None]:
    _quantized, codes, residual_codes = _weights_to_grouped_codes(
        weights,
        scale,
        bits=bits,
        residual_scale=residual_scale,
    )
    return codes, residual_codes


def _codes_to_weights(
    codes: UInt16[Array, "... groups"],
    residual_codes: Array | None,
    scale: Float[Array, "*components"],
    *,
    bits: int,
    dtype: DTypeLike,
    residual_scale: float,
) -> Float[Array, "... cols"]:
    return _codes_to_master_weights(
        codes,
        residual_codes,
        scale,
        bits=bits,
        dtype=dtype,
        residual_scale=residual_scale,
    )


@dataclass(frozen=True)
class E8PSpec(QuantizedSpec):
    bits: Literal[2, 3, 4] = 2
    layout: Layout = Layout.OUTPUT_INPUT
    scale_normalization: float | None = None
    residual_scale: float | None = None

    @property
    def input_block_size(self) -> int:
        if self.layout == Layout.INPUT_OUTPUT:
            return 1
        return _E8P_VECTOR_SIZE

    @property
    def output_block_size(self) -> int:
        if self.layout == Layout.INPUT_OUTPUT:
            return _E8P_VECTOR_SIZE
        return 1

    @property
    def rate(self) -> float:
        return float(self.bits)

    @cached_property
    def distortion(self) -> float:
        residual_scale = None
        if self.bits != 2:
            residual_scale = self.residual_scale
        return distortion_estimate(
            format_name="e8p",
            bits=self.bits,
            group_size=_E8P_VECTOR_SIZE,
            scale_normalization=self.scale_normalization,
            residual_scale=residual_scale,
        )

    @property
    def scale_normalization_value(self) -> float:
        if self.scale_normalization is not None:
            return self.scale_normalization
        return _E8P_DEFAULT_SCALE[self.bits]

    @property
    def residual_scale_value(self) -> float:
        if self.bits == 2:
            return 1.0
        if self.residual_scale is not None:
            return self.residual_scale
        return _E8P_DEFAULT_RESIDUAL_SCALE[self.bits]

    def compress(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        is_sharded: bool = True,
    ) -> "E8PMatrix":
        if preconditioner is not None:
            weights = yaqa_round_weights(weights, preconditioner, self, is_sharded=is_sharded)

        stored_weights = self.layout.from_output_input(weights, is_sharded=is_sharded)
        parameters = E8PParameters.from_weights(stored_weights, scale_normalization=self.scale_normalization_value)
        weight_partition = self.layout.weight_partition(stored_weights.ndim - 2, is_sharded=is_sharded)
        scale = with_sharding(parameters.scale, make_sharding(weight_partition[:-2]))
        if implementation == CompressionImplementation.TRAINING:
            return E8PMatrixForTraining(
                spec=self,
                is_sharded=is_sharded,
                master_weights=stored_weights,
                scale=scale,
            )

        codes, residual_codes = _master_weights_to_codes(
            stored_weights,
            scale,
            bits=self.bits,
            residual_scale=self.residual_scale_value,
        )
        return self.from_packed_parameters(
            codes=codes,
            residual_codes=residual_codes,
            scale=scale,
            dtype=stored_weights.dtype,
            implementation=CompressionImplementation.INFERENCE,
            is_sharded=is_sharded,
        )

    def from_packed_parameters(
        self,
        *,
        codes: UInt16[Array, "*components rows groups"],
        residual_codes: Array | None,
        scale: Float[Array, "*components"],
        dtype: DTypeLike = jnp.float32,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        is_sharded: bool = True,
    ) -> "E8PMatrix":
        if self.bits == 2 and residual_codes is not None:
            raise ValueError("2-bit E8P parameters must not include residual codes.")
        if self.bits > 2 and residual_codes is None:
            raise ValueError(f"{self.bits}-bit E8P parameters require residual codes.")

        weight_partition = self.layout.weight_partition(codes.ndim - 2, is_sharded=is_sharded)
        weight_sharding = make_sharding(weight_partition)
        parameter_sharding = make_sharding((*weight_partition[:-1], None))
        codes = with_sharding(codes, parameter_sharding)
        if residual_codes is not None:
            residual_codes = with_sharding(residual_codes, parameter_sharding)
        scale = with_sharding(scale, make_sharding(weight_partition[:-2]))

        if implementation == CompressionImplementation.INFERENCE:
            return E8PMatrixForInference(
                spec=self,
                is_sharded=is_sharded,
                dtype_=jnp.dtype(dtype),
                codes=codes,
                residual_codes=residual_codes,
                scale=scale,
            )

        weights = _codes_to_weights(
            codes,
            residual_codes,
            scale,
            bits=self.bits,
            dtype=dtype,
            residual_scale=self.residual_scale_value,
        )
        return E8PMatrixForTraining(
            spec=self,
            is_sharded=is_sharded,
            master_weights=with_sharding(weights, weight_sharding),
            scale=scale,
        )


class E8PMatrix(EmbeddingMatrix[E8PSpec]):
    scale: Float[Array, "*components"] = field(norm=ParameterNorm.L_INF)

    @property
    @abstractmethod
    def _codes(self) -> UInt16[Array, "*components rows groups"]: ...

    @property
    @abstractmethod
    def _residual_codes(self) -> Array | None: ...

    def export(self) -> ExportResults:
        arrays = {
            "weights": self._codes,
            "scale": self.scale,
        }
        residual_codes = self._residual_codes
        if residual_codes is not None:
            arrays["residual_weights"] = residual_codes
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
    ) -> "E8PMatrix": ...

    @abstractmethod
    def switch_implementation(self, implementation: CompressionImplementation) -> "E8PMatrix": ...

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(layout=self.spec.layout).compress(self.decompress(), is_sharded=self.is_sharded)


class E8PMatrixForTraining(E8PMatrix):
    master_weights: Float[Array, "*components rows cols"] = field(norm=ParameterNorm.SPECTRAL)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.master_weights.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.master_weights.dtype

    @property
    def _codes(self) -> UInt16[Array, "*components rows groups"]:
        codes, _residual_codes = _master_weights_to_codes(
            self.master_weights,
            self.scale,
            bits=self.spec.bits,
            residual_scale=self.spec.residual_scale_value,
        )
        return codes

    @property
    def _residual_codes(self) -> Array | None:
        _codes, residual_codes = _master_weights_to_codes(
            self.master_weights,
            self.scale,
            bits=self.spec.bits,
            residual_scale=self.spec.residual_scale_value,
        )
        return residual_codes

    def astype(self, dtype: DTypeLike) -> "E8PMatrixForTraining":
        return E8PMatrixForTraining(
            spec=self.spec,
            is_sharded=self.is_sharded,
            master_weights=self.master_weights.astype(dtype),
            scale=self.scale.astype(dtype),
        )

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        weights = _quantize(
            self.master_weights,
            self.scale,
            bits=self.spec.bits,
            residual_scale=self.spec.residual_scale_value,
        )
        return self.spec.layout.to_output_input(weights)

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
        return _quantize(
            self.master_weights[index, :].astype(dtype),
            self.scale.astype(dtype),
            bits=self.spec.bits,
            residual_scale=self.spec.residual_scale_value,
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
        weights = _quantize(
            self.master_weights.astype(vector.dtype),
            self.scale.astype(vector.dtype),
            bits=self.spec.bits,
            residual_scale=self.spec.residual_scale_value,
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
    ) -> E8PMatrix:
        if prefix is None:
            prefix = ParameterPath()
        saved_spec = expored_data.metadata[prefix / "spec"]
        loaded_spec = WeightMatrixSpec.from_json(saved_spec)
        if loaded_spec != self.spec:
            raise ValueError(f"WeightMatrix spec mismatch: expected {self.spec}, got {loaded_spec}")

        residual_codes = self._residual_codes
        if residual_codes is not None:
            residual_codes = load_as(
                residual_codes,
                expored_data.arrays[prefix / "residual_weights"],
                allow_dtype_cast=False,
            )
        return self.spec.from_packed_parameters(
            codes=load_as(self._codes, expored_data.arrays[prefix / "weights"], allow_dtype_cast=False),
            residual_codes=residual_codes,
            scale=load_as(
                self.scale,
                expored_data.arrays[prefix / "scale"],
                allow_dtype_cast=allow_dtype_cast,
            ),
            dtype=self.dtype,
            implementation=CompressionImplementation.TRAINING,
            is_sharded=self.is_sharded,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> E8PMatrix:
        if implementation == CompressionImplementation.TRAINING:
            return self
        return self.spec.from_packed_parameters(
            codes=self._codes,
            residual_codes=self._residual_codes,
            scale=self.scale,
            dtype=self.dtype,
            implementation=CompressionImplementation.INFERENCE,
            is_sharded=self.is_sharded,
        )


class E8PMatrixForInference(E8PMatrix):
    dtype_: DTypeLike = field(static=True)
    codes: UInt16[Array, "*components rows groups"]
    residual_codes: Array | None
    scale: Float[Array, "*components"]

    @property
    def shape(self) -> tuple[int, ...]:
        *leading_dims, rows, groups = self.codes.shape
        return (*leading_dims, rows, groups * _E8P_VECTOR_SIZE)

    @property
    def dtype(self) -> DTypeLike:
        return self.dtype_

    @property
    def _codes(self) -> UInt16[Array, "*components rows groups"]:
        return self.codes

    @property
    def _residual_codes(self) -> Array | None:
        return self.residual_codes

    def astype(self, dtype: DTypeLike) -> "E8PMatrixForInference":
        return E8PMatrixForInference(
            spec=self.spec,
            is_sharded=self.is_sharded,
            dtype_=jnp.dtype(dtype),
            codes=self.codes,
            residual_codes=self.residual_codes,
            scale=self.scale.astype(dtype),
        )

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        weights = _codes_to_weights(
            self.codes,
            self.residual_codes,
            self.scale,
            bits=self.spec.bits,
            dtype=self.dtype,
            residual_scale=self.spec.residual_scale_value,
        )
        return self.spec.layout.to_output_input(weights)

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
        residual_codes = self.residual_codes
        if residual_codes is not None:
            residual_codes = residual_codes[index, :]
        return _codes_to_weights(
            self.codes[index, :],
            residual_codes,
            self.scale.astype(dtype),
            bits=self.spec.bits,
            dtype=dtype,
            residual_scale=self.spec.residual_scale_value,
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
        weights = _codes_to_weights(
            self.codes,
            self.residual_codes,
            self.scale.astype(vector.dtype),
            bits=self.spec.bits,
            dtype=vector.dtype,
            residual_scale=self.spec.residual_scale_value,
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
    ) -> E8PMatrix:
        if prefix is None:
            prefix = ParameterPath()
        saved_spec = expored_data.metadata[prefix / "spec"]
        loaded_spec = WeightMatrixSpec.from_json(saved_spec)
        if loaded_spec != self.spec:
            raise ValueError(f"WeightMatrix spec mismatch: expected {self.spec}, got {loaded_spec}")

        residual_codes = self.residual_codes
        if residual_codes is not None:
            residual_codes = load_as(
                residual_codes,
                expored_data.arrays[prefix / "residual_weights"],
                allow_dtype_cast=False,
            )
        return self.spec.from_packed_parameters(
            codes=load_as(self.codes, expored_data.arrays[prefix / "weights"], allow_dtype_cast=False),
            residual_codes=residual_codes,
            scale=load_as(
                self.scale,
                expored_data.arrays[prefix / "scale"],
                allow_dtype_cast=allow_dtype_cast,
            ),
            dtype=self.dtype,
            implementation=CompressionImplementation.INFERENCE,
            is_sharded=self.is_sharded,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> E8PMatrix:
        if implementation == CompressionImplementation.INFERENCE:
            return self
        return self.spec.from_packed_parameters(
            codes=self.codes,
            residual_codes=self.residual_codes,
            scale=self.scale,
            dtype=self.dtype,
            implementation=CompressionImplementation.TRAINING,
            is_sharded=self.is_sharded,
        )
