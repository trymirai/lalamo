from abc import abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Self

import jax.numpy as jnp
from einops import rearrange
from jax.lax import stop_gradient
from jax.sharding import PartitionSpec
from jaxtyping import Array, DTypeLike, Float, Int, Key, UInt8

from lalamo.exportable import ExportResults
from lalamo.module import Keychain, ParameterNorm, field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import preserve_first_input_sharding, supports_dummy_arrays
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.sharding import is_sharded, make_sharding, reshard_as, sharding_of, use_out_sharding, with_sharding
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
    "MXFP4Matrix",
    "MXFP4MatrixForInference",
    "MXFP4MatrixForTraining",
    "MXFP4Spec",
]

_MXFP4_BITS = 4
_MXFP4_GROUP_SIZE = 32
_MXFP4_E8M0_BIAS = 127

_MXFP4_CODEBOOK_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


class MXFP4Parameters(NamedTuple):
    scale_exponents: UInt8[Array, "*components rows groups"]

    @classmethod
    @supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
    def from_weights(
        cls,
        weights: Float[Array, "*components rows cols"],
        group_size: int,
    ) -> Self:
        grouped_weights = group_by_last_axis(weights, group_size=group_size)
        max_abs = jnp.max(jnp.abs(grouped_weights), axis=-1).astype(jnp.float32)
        safe_max_abs = jnp.where(max_abs > 0, max_abs, jnp.float32(2.0**-127))
        scale_exponents = jnp.floor(jnp.log2(safe_max_abs)).astype(jnp.int32) - 2
        scale_exponents = jnp.clip(scale_exponents, -_MXFP4_E8M0_BIAS, _MXFP4_E8M0_BIAS)
        return cls(scale_exponents=(scale_exponents + _MXFP4_E8M0_BIAS).astype(jnp.uint8))


def _mxfp4_codebook(dtype: DTypeLike) -> Float[Array, " levels"]:
    return jnp.array(_MXFP4_CODEBOOK_VALUES, dtype=dtype)


def _mxfp4_scale_values(
    scale_exponents: UInt8[Array, "... groups"],
    dtype: DTypeLike,
) -> Float[Array, "... groups"]:
    exponents = scale_exponents.astype(jnp.int32) - _MXFP4_E8M0_BIAS
    return jnp.ldexp(jnp.ones_like(scale_exponents, dtype=dtype), exponents)


def _mxfp4_grouped_indices(
    weights: Float[Array, "... cols"],
    scale_exponents: UInt8[Array, "... groups"],
    *,
    group_size: int,
) -> UInt8[Array, "... groups group_size"]:
    codebook = _mxfp4_codebook(weights.dtype)
    grouped_weights = rearrange(weights, "... (groups group_size) -> ... groups group_size", group_size=group_size)
    scales = _mxfp4_scale_values(scale_exponents, weights.dtype)
    normalized_weights = grouped_weights / scales[..., None]
    quantized_values = normalized_weights.astype(jnp.float4_e2m1fn).astype(weights.dtype)
    distances = jnp.abs(quantized_values[..., None] - codebook)
    return jnp.argmin(distances, axis=-1).astype(jnp.uint8)


def _mxfp4_indices_to_master_weights(
    indices: UInt8[Array, "... cols"],
    scale_exponents: UInt8[Array, "... groups"],
    *,
    dtype: DTypeLike,
    group_size: int,
) -> Float[Array, "... cols"]:
    codebook = _mxfp4_codebook(dtype)
    grouped_indices = rearrange(
        indices.astype(jnp.int32),
        "... (groups group_size) -> ... groups group_size",
        group_size=group_size,
    )
    scales = _mxfp4_scale_values(scale_exponents, dtype)
    if scale_exponents.ndim == 1:
        levels = jnp.arange(len(_MXFP4_CODEBOOK_VALUES), dtype=jnp.int32)
        grouped_values = jnp.sum((grouped_indices[..., None] == levels).astype(scales.dtype) * codebook, axis=-1)
    else:
        scale_sharding = sharding_of(scale_exponents)
        values_sharding = None
        if is_sharded(scale_sharding):
            values_sharding = PartitionSpec(*tuple(scale_sharding.spec), None)
        grouped_values = codebook.at[grouped_indices].get(out_sharding=values_sharding)
    grouped_weights = grouped_values * scales[..., None]
    return rearrange(grouped_weights, "... groups group_size -> ... (groups group_size)")


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _mxfp4_quantize(
    weights: Float[Array, "... cols"],
    scale_exponents: UInt8[Array, "... groups"],
    *,
    group_size: int,
) -> Float[Array, "... cols"]:
    grouped_indices = _mxfp4_grouped_indices(
        weights,
        stop_gradient(scale_exponents),
        group_size=group_size,
    )
    indices = rearrange(grouped_indices, "... groups group_size -> ... (groups group_size)")
    quantized_weights = _mxfp4_indices_to_master_weights(
        indices,
        scale_exponents,
        dtype=weights.dtype,
        group_size=group_size,
    )
    return quantized_weights + weights - stop_gradient(weights)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _mxfp4_pack_master_weights(
    weights: Float[Array, "*components rows cols"],
    scale_exponents: UInt8[Array, "*components rows groups"],
    *,
    group_size: int,
) -> UInt8[Array, "*components rows packed_cols"]:
    grouped_indices = _mxfp4_grouped_indices(
        weights,
        scale_exponents,
        group_size=group_size,
    )
    indices = rearrange(grouped_indices, "... groups group_size -> ... (groups group_size)")
    return pack_uint_to_uint8(indices, _MXFP4_BITS)


def _mxfp4_unpack_master_weights(
    packed_weights: UInt8[Array, "... packed_cols"],
    scale_exponents: UInt8[Array, "... groups"],
    *,
    dtype: DTypeLike,
    group_size: int,
) -> Float[Array, "... cols"]:
    indices = unpack_uint8_to_uint(
        packed_weights,
        bits=_MXFP4_BITS,
        unpacked_last_axis_dim=scale_exponents.shape[-1] * group_size,
    )
    return _mxfp4_indices_to_master_weights(
        indices,
        scale_exponents,
        dtype=dtype,
        group_size=group_size,
    )


@dataclass(frozen=True)
class MXFP4Spec(QuantizedSpec):
    group_size: int = _MXFP4_GROUP_SIZE
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
        return float(_MXFP4_BITS + 8 / self.group_size)

    @property
    def distortion(self) -> float:
        return standard_normal_absmax_squared(self.group_size) / (12 * (2**_MXFP4_BITS - 1) ** 2)

    def compress(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        is_sharded: bool = True,
    ) -> "MXFP4Matrix":
        if preconditioner is not None:
            weights = yaqa_round_weights(weights, preconditioner, self, is_sharded=is_sharded)

        stored_weights = self.layout.from_output_input(weights, is_sharded=is_sharded)
        parameters = MXFP4Parameters.from_weights(stored_weights, group_size=self.group_size)
        weight_partition = self.layout.weight_partition(parameters.scale_exponents.ndim - 2, is_sharded=is_sharded)
        parameter_sharding = make_sharding((*weight_partition[:-1], None))
        scale_exponents = with_sharding(parameters.scale_exponents, parameter_sharding)
        if implementation == CompressionImplementation.TRAINING:
            return MXFP4MatrixForTraining(
                spec=self,
                is_sharded=is_sharded,
                dtype_=stored_weights.dtype,
                weights=stored_weights,
                scale_exponents=scale_exponents,
            )

        packed_weights = _mxfp4_pack_master_weights(
            stored_weights,
            scale_exponents,
            group_size=self.group_size,
        )
        return self.from_packed_parameters(
            packed_weights=packed_weights,
            scale_exponents=scale_exponents,
            dtype=stored_weights.dtype,
            implementation=CompressionImplementation.INFERENCE,
            is_sharded=is_sharded,
        )

    def from_packed_parameters(
        self,
        *,
        packed_weights: UInt8[Array, "*components rows packed_cols"],
        scale_exponents: UInt8[Array, "*components rows groups"],
        dtype: DTypeLike = jnp.float32,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        is_sharded: bool = True,
    ) -> "MXFP4Matrix":
        weight_partition = self.layout.weight_partition(scale_exponents.ndim - 2, is_sharded=is_sharded)
        weight_sharding = make_sharding(weight_partition)
        parameter_sharding = make_sharding((*weight_partition[:-1], None))
        packed_weights = with_sharding(packed_weights, parameter_sharding)
        scale_exponents = with_sharding(scale_exponents, parameter_sharding)

        if implementation == CompressionImplementation.INFERENCE:
            return MXFP4MatrixForInference(
                spec=self,
                is_sharded=is_sharded,
                dtype_=jnp.dtype(dtype),
                packed_weights=packed_weights,
                scale_exponents=scale_exponents,
            )

        weights = _mxfp4_unpack_master_weights(
            packed_weights,
            scale_exponents,
            dtype=dtype,
            group_size=self.group_size,
        )
        return MXFP4MatrixForTraining(
            spec=self,
            is_sharded=is_sharded,
            dtype_=jnp.dtype(dtype),
            weights=with_sharding(weights, weight_sharding),
            scale_exponents=scale_exponents,
        )


class MXFP4Matrix(EmbeddingMatrix[MXFP4Spec]):
    dtype_: DTypeLike = field(static=True)
    scale_exponents: UInt8[Array, "*components rows groups"]

    @property
    @abstractmethod
    def _packed_quantized_weights(self) -> UInt8[Array, "*components rows packed_cols"]: ...

    def export(self) -> ExportResults:
        return ExportResults(
            arrays={
                "weights": self._packed_quantized_weights,
                "scale_exponents": self.scale_exponents,
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
    ) -> "MXFP4Matrix": ...

    @abstractmethod
    def switch_implementation(self, implementation: CompressionImplementation) -> "MXFP4Matrix": ...

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(layout=self.spec.layout).compress(self.decompress(), is_sharded=self.is_sharded)


class MXFP4MatrixForTraining(MXFP4Matrix):
    weights: Float[Array, "*components rows cols"] = field(norm=ParameterNorm.SPECTRAL)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.weights.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.weights.dtype

    @property
    def _packed_quantized_weights(self) -> UInt8[Array, "*components rows packed_cols"]:
        return _mxfp4_pack_master_weights(
            self.weights,
            self.scale_exponents,
            group_size=self.spec.group_size,
        )

    def astype(self, dtype: DTypeLike) -> "MXFP4MatrixForTraining":
        return MXFP4MatrixForTraining(
            spec=self.spec,
            is_sharded=self.is_sharded,
            dtype_=jnp.dtype(dtype),
            weights=self.weights.astype(dtype),
            scale_exponents=self.scale_exponents,
        )

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        weights = _mxfp4_quantize(
            self.weights,
            self.scale_exponents,
            group_size=self.spec.group_size,
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
        return _mxfp4_quantize(
            self.weights[index, :].astype(dtype),
            self.scale_exponents[index, :],
            group_size=self.spec.group_size,
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
        weights = _mxfp4_quantize(
            self.weights.astype(vector.dtype),
            self.scale_exponents,
            group_size=self.spec.group_size,
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
        allow_dtype_cast: bool = False,  # noqa: ARG002
        *,
        prefix: ParameterPath | None = None,
    ) -> MXFP4Matrix:
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
        scale_exponents = load_as(
            self.scale_exponents,
            expored_data.arrays[prefix / "scale_exponents"],
            allow_dtype_cast=False,
        )
        return self.spec.from_packed_parameters(
            packed_weights=packed_weights,
            scale_exponents=scale_exponents,
            dtype=self.dtype,
            implementation=CompressionImplementation.TRAINING,
            is_sharded=self.is_sharded,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> MXFP4Matrix:
        if implementation == CompressionImplementation.TRAINING:
            return self
        return self.spec.from_packed_parameters(
            packed_weights=self._packed_quantized_weights,
            scale_exponents=self.scale_exponents,
            dtype=self.dtype,
            implementation=CompressionImplementation.INFERENCE,
            is_sharded=self.is_sharded,
        )


class MXFP4MatrixForInference(MXFP4Matrix):
    packed_weights: UInt8[Array, "*components rows packed_cols"]
    scale_exponents: UInt8[Array, "*components rows groups"]

    @property
    def shape(self) -> tuple[int, ...]:
        *leading_dims, rows, groups = self.scale_exponents.shape
        return (*leading_dims, rows, groups * self.spec.group_size)

    @property
    def dtype(self) -> DTypeLike:
        return self.dtype_

    @property
    def _packed_quantized_weights(self) -> UInt8[Array, "*components rows packed_cols"]:
        return self.packed_weights

    def astype(self, dtype: DTypeLike) -> "MXFP4MatrixForInference":
        return MXFP4MatrixForInference(
            spec=self.spec,
            is_sharded=self.is_sharded,
            dtype_=jnp.dtype(dtype),
            packed_weights=self.packed_weights,
            scale_exponents=self.scale_exponents,
        )

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        weights = _mxfp4_unpack_master_weights(
            self.packed_weights,
            self.scale_exponents,
            dtype=self.dtype,
            group_size=self.spec.group_size,
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
        return _mxfp4_unpack_master_weights(
            self.packed_weights[index, :],
            self.scale_exponents[index, :],
            dtype=dtype,
            group_size=self.spec.group_size,
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
        weights = _mxfp4_unpack_master_weights(
            self.packed_weights,
            self.scale_exponents,
            dtype=vector.dtype,
            group_size=self.spec.group_size,
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
        allow_dtype_cast: bool = False,  # noqa: ARG002
        *,
        prefix: ParameterPath | None = None,
    ) -> MXFP4Matrix:
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
        scale_exponents = load_as(
            self.scale_exponents,
            expored_data.arrays[prefix / "scale_exponents"],
            allow_dtype_cast=False,
        )
        return self.spec.from_packed_parameters(
            packed_weights=packed_weights,
            scale_exponents=scale_exponents,
            dtype=self.dtype,
            implementation=CompressionImplementation.INFERENCE,
            is_sharded=self.is_sharded,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> MXFP4Matrix:
        if implementation == CompressionImplementation.INFERENCE:
            return self
        return self.spec.from_packed_parameters(
            packed_weights=self.packed_weights,
            scale_exponents=self.scale_exponents,
            dtype=self.dtype,
            implementation=CompressionImplementation.TRAINING,
            is_sharded=self.is_sharded,
        )
