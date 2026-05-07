from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple, Self

import jax
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset, stop_gradient
from jaxtyping import Array, DTypeLike, Float, Int, Key

from lalamo.exportable import ExportResults
from lalamo.module import Keychain, ParameterNorm, field
from lalamo.utils.dummy_array import (
    dummy_array,
    is_dummy_array,
    preserve_first_input_sharding,
    supports_dummy_arrays,
)
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.sharding import make_sharding, reshard_as, use_out_sharding, with_sharding
from lalamo.utils.surgery import load_as
from lalamo.weight_matrix import (
    CompressionImplementation,
    EmbeddingMatrix,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    GradientEstimator,
    Layout,
    MatmulConfig,
    Preconditioner,
    WeightMatrixSpec,
)

from .cute_w4a16_contract import CHANNEL_MULTIPLE, GROUP_SIZE, MLX_ROWS_PER_CTA
from .packing import pack_uint_to_uint8, unpack_uint8_to_uint
from .rounding import deterministic_round_to_unsigned_grid, round_to_unsigned_grid
from .utils import (
    expand_last_axis_groups,
    group_by_last_axis,
    min_max_within_groups,
    scale_from_min_max,
)

__all__ = [
    "MLXMatrix",
    "MLXMatrixForInference",
    "MLXMatrixForTraining",
    "MLXSpec",
]


class MLXAffineParameters(NamedTuple):
    scales: Float[Array, "... groups"]
    biases: Float[Array, "... groups"]

    @classmethod
    @supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
    def from_weights(
        cls,
        weights: Float[Array, "... rows cols"],
        bits: int,
        group_size: int,
    ) -> Self:
        group_min_max = min_max_within_groups(group_by_last_axis(weights, group_size=group_size))
        scales = scale_from_min_max(group_min_max, bits=bits, dtype=weights.dtype)
        return cls(scales=scales, biases=group_min_max.min)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _mlx_master_weights_to_int_scale(
    weights: Float[Array, "... cols"],
    scales: Float[Array, "... groups"],
    biases: Float[Array, "... groups"],
    group_size: int,
) -> Float[Array, "... rows cols"]:
    expanded_scales = expand_last_axis_groups(scales, group_size=group_size)
    expanded_biases = expand_last_axis_groups(biases, group_size=group_size)
    return (weights - expanded_biases) / expanded_scales


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _mlx_quantize(
    weights: Float[Array, "... cols"],
    scales: Float[Array, "... groups"],
    biases: Float[Array, "... groups"],
    group_size: int,
    round_fn: Callable[[Float[Array, "... rows cols"]], Float[Array, "... rows cols"]],
) -> Float[Array, "... rows cols"]:
    expanded_scales = expand_last_axis_groups(scales, group_size=group_size)
    expanded_biases = expand_last_axis_groups(biases, group_size=group_size)
    int_scale_weights = (weights - stop_gradient(expanded_biases)) / stop_gradient(expanded_scales)

    return round_fn(int_scale_weights) * expanded_scales + expanded_biases


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _mlx_pack_master_weights(
    weights: Float[Array, "... rows cols"],
    scales: Float[Array, "... rows groups"],
    biases: Float[Array, "... rows groups"],
    group_size: int,
    bits: int,
) -> Int[Array, "... rows packed_cols"]:
    int_scale_weights = _mlx_master_weights_to_int_scale(
        weights,
        scales,
        biases,
        group_size,
    )
    rounded_weights = deterministic_round_to_unsigned_grid(int_scale_weights, bits=bits)
    return pack_uint_to_uint8(rounded_weights.astype(jnp.uint8), bits)


def _mlx_unpack_master_weights(
    packed_weights: Int[Array, "... rows packed_cols"],
    scales: Float[Array, "... rows groups"],
    biases: Float[Array, "... rows groups"],
    group_size: int,
    bits: int,
) -> Float[Array, "... rows cols"]:
    int_weights = unpack_uint8_to_uint(packed_weights, bits=bits)
    expanded_scales = expand_last_axis_groups(scales, group_size=group_size)
    expanded_biases = expand_last_axis_groups(biases, group_size=group_size)
    return int_weights.astype(scales.dtype) * expanded_scales + expanded_biases


def _mlx_grouped_dot_output_input(
    weights: Float[Array, "rows cols"],
    scales: Float[Array, "rows groups"],
    biases: Float[Array, "rows groups"],
    vector: Float[Array, " channels"],
    group_size: int,
    bits: int,
) -> Float[Array, " rows"]:
    assert vector.shape[0] == weights.shape[-1]
    assert weights.shape[-1] % group_size == 0
    grouped_weights = group_by_last_axis(weights, group_size=group_size)
    int_scale_weights = (grouped_weights - stop_gradient(biases[..., None])) / stop_gradient(scales[..., None])
    rounded_weights = deterministic_round_to_unsigned_grid(int_scale_weights, bits=bits)

    vector_groups = vector.reshape(vector.shape[0] // group_size, group_size).astype(jnp.float32)
    int_dot = jnp.sum(rounded_weights.astype(jnp.float32) * vector_groups[None, :, :], axis=-1)
    vector_sums = jnp.sum(vector_groups, axis=-1)
    group_outputs = int_dot * scales.astype(jnp.float32) + vector_sums[None, :] * biases.astype(jnp.float32)
    return jnp.sum(group_outputs, axis=-1).astype(vector.dtype)


def _use_cute_w4a16_dot(
    spec: "MLXSpec",
    packed_weights: Array,
    vector: Array,
    forward_pass_config: MatmulConfig,
    transposed: bool,
) -> bool:
    return (
        jax.default_backend() == "gpu"
        and spec.bits == 4
        and spec.group_size == GROUP_SIZE
        and spec.layout == Layout.OUTPUT_INPUT
        and vector.ndim == 1
        and vector.dtype in (jnp.float16, jnp.bfloat16)
        and vector.shape[0] % CHANNEL_MULTIPLE == 0
        and packed_weights.shape[0] % MLX_ROWS_PER_CTA == 0
        and forward_pass_config.precision == DotAlgorithmPreset.DEFAULT
        and not transposed
    )


@dataclass(frozen=True)
class MLXSpec(WeightMatrixSpec):
    bits: Literal[4, 8]
    group_size: int
    layout: Layout = Layout.OUTPUT_INPUT

    def compress(
        self,
        weights: Float[Array, "... out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> "MLXMatrix":
        if preconditioner is not None:
            raise ValueError("Preconditioned rounding is not implemented yet.")

        weights = self.layout.from_output_input(weights)
        affine_parameters = MLXAffineParameters.from_weights(
            weights,
            bits=self.bits,
            group_size=self.group_size,
        )
        if implementation == CompressionImplementation.TRAINING:
            result = MLXMatrixForTraining(
                spec=self,
                weights=weights,
                scales=affine_parameters.scales,
                biases=affine_parameters.biases,
            )
        else:
            packed_int_weights = _mlx_pack_master_weights(
                weights,
                affine_parameters.scales,
                affine_parameters.biases,
                self.group_size,
                self.bits,
            )
            result = self.from_packed_parameters(
                packed_weights=packed_int_weights,
                scales=affine_parameters.scales,
                biases=affine_parameters.biases,
                implementation=CompressionImplementation.INFERENCE,
            )

        return result

    def from_packed_parameters(
        self,
        *,
        packed_weights: Int[Array, "... rows packed_cols"],
        scales: Float[Array, "... rows groups"],
        biases: Float[Array, "... rows groups"],
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> "MLXMatrix":
        weight_sharding = make_sharding(self.layout.weight_partition(scales.ndim - 2))
        packed_weights = with_sharding(packed_weights, weight_sharding)
        scales = with_sharding(scales, weight_sharding)
        biases = with_sharding(biases, weight_sharding)

        if implementation == CompressionImplementation.INFERENCE:
            if is_dummy_array(packed_weights):
                weights = dummy_array(
                    (*packed_weights.shape[:-1], packed_weights.shape[-1] * (8 // self.bits)),
                    scales.dtype,
                )
            else:
                weights = _mlx_unpack_master_weights(
                    packed_weights,
                    scales,
                    biases,
                    self.group_size,
                    self.bits,
                )
            return MLXMatrixForInference(
                spec=self,
                packed_weights=packed_weights,
                weights=with_sharding(weights, weight_sharding),
                scales=scales,
                biases=biases,
            )

        weights = _mlx_unpack_master_weights(
            packed_weights,
            scales,
            biases,
            self.group_size,
            self.bits,
        )
        return MLXMatrixForTraining(
            spec=self,
            weights=with_sharding(weights, weight_sharding),
            scales=scales,
            biases=biases,
        )


class MLXMatrix(EmbeddingMatrix[MLXSpec]):
    scales: Float[Array, "... rows groups"] = field(norm=ParameterNorm.L_INF)
    biases: Float[Array, "... rows groups"] = field(norm=ParameterNorm.L_INF)

    @property
    @abstractmethod
    def _packed_quantized_weights(self) -> Int[Array, "... rows packed_cols"]: ...

    def export(self) -> ExportResults:
        return ExportResults(
            arrays={
                "weights": self._packed_quantized_weights,
                "scales": self.scales,
                "biases": self.biases,
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
    ) -> "MLXMatrix": ...

    @abstractmethod
    def switch_implementation(self, implementation: CompressionImplementation) -> "MLXMatrix": ...

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(layout=self.spec.layout).compress(self.decompress())


class MLXMatrixForTraining(MLXMatrix):
    weights: Float[Array, "... rows cols"] = field(norm=ParameterNorm.SPECTRAL)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.weights.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.weights.dtype

    @property
    def _packed_quantized_weights(self) -> Int[Array, "... rows packed_cols"]:
        return _mlx_pack_master_weights(
            self.weights,
            self.scales,
            self.biases,
            self.spec.group_size,
            self.spec.bits,
        )

    def astype(self, dtype: DTypeLike) -> "MLXMatrixForTraining":
        return MLXMatrixForTraining(
            spec=self.spec,
            weights=self.weights.astype(dtype),
            scales=self.scales.astype(dtype),
            biases=self.biases.astype(dtype),
        )

    def decompress(self) -> Float[Array, "... out_channels in_channels"]:
        weights = _mlx_quantize(
            self.weights,
            self.scales,
            self.biases,
            group_size=self.spec.group_size,
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
        return _mlx_quantize(
            self.weights[index, :].astype(dtype),
            self.scales[index, :].astype(dtype),
            self.biases[index, :].astype(dtype),
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
        vector: Float[Array, " channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Float[Array, "... channels"]:
        self._raise_if_batched()
        round_fn = partial(
            round_to_unsigned_grid,
            bits=self.spec.bits,
            keychain=keychain,
            gradient_estimator=forward_pass_config.gradient_estimator,
        )
        uses_grouped_dot = (
            self.spec.layout == Layout.OUTPUT_INPUT
            and vector.dtype in (jnp.bfloat16, jnp.float16)
            and forward_pass_config.gradient_estimator == GradientEstimator.DETERMINISTIC_ROUNDING
            and forward_pass_config.precision == DotAlgorithmPreset.DEFAULT
            and not transposed
        )
        if uses_grouped_dot:
            result = _mlx_grouped_dot_output_input(
                self.weights.astype(vector.dtype),
                self.scales.astype(vector.dtype),
                self.biases.astype(vector.dtype),
                vector,
                self.spec.group_size,
                self.spec.bits,
            )
            return reshard_as(result, vector)

        dequantized_weights = _mlx_quantize(
            self.weights.astype(vector.dtype),
            self.scales.astype(vector.dtype),
            self.biases.astype(vector.dtype),
            group_size=self.spec.group_size,
            round_fn=round_fn,
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
    ) -> MLXMatrix:
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
        biases = load_as(self.biases, expored_data.arrays[prefix / "biases"], allow_dtype_cast=allow_dtype_cast)
        return self.spec.from_packed_parameters(
            packed_weights=packed_weights,
            scales=scales,
            biases=biases,
            implementation=CompressionImplementation.TRAINING,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> MLXMatrix:
        if implementation == CompressionImplementation.TRAINING:
            return self
        return self.spec.from_packed_parameters(
            packed_weights=self._packed_quantized_weights,
            scales=self.scales,
            biases=self.biases,
            implementation=CompressionImplementation.INFERENCE,
        )


class MLXMatrixForInference(MLXMatrix):
    packed_weights: Int[Array, "... rows packed_cols"]
    weights: Float[Array, "... rows cols"]
    scales: Float[Array, "... rows groups"]
    biases: Float[Array, "... rows groups"]

    @property
    def shape(self) -> tuple[int, ...]:
        *leading_dims, rows, groups = self.scales.shape
        return (*leading_dims, rows, groups * self.spec.group_size)

    @property
    def dtype(self) -> DTypeLike:
        return self.weights.dtype

    @property
    def _packed_quantized_weights(self) -> Int[Array, "... rows packed_cols"]:
        return self.packed_weights

    def astype(self, dtype: DTypeLike) -> "MLXMatrixForInference":
        return MLXMatrixForInference(
            spec=self.spec,
            packed_weights=self.packed_weights,
            weights=self.weights.astype(dtype),
            scales=self.scales.astype(dtype),
            biases=self.biases.astype(dtype),
        )

    def load_exported(
        self,
        expored_data: ExportResults,
        allow_dtype_cast: bool = False,
        *,
        prefix: ParameterPath | None = None,
    ) -> MLXMatrix:
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
        biases = load_as(self.biases, expored_data.arrays[prefix / "biases"], allow_dtype_cast=allow_dtype_cast)
        return self.spec.from_packed_parameters(
            packed_weights=packed_weights,
            scales=scales,
            biases=biases,
            implementation=CompressionImplementation.INFERENCE,
        )

    def switch_implementation(self, implementation: CompressionImplementation) -> MLXMatrix:
        if implementation == CompressionImplementation.INFERENCE:
            return self
        return self.spec.from_packed_parameters(
            packed_weights=self.packed_weights,
            scales=self.scales,
            biases=self.biases,
            implementation=CompressionImplementation.TRAINING,
        )

    def decompress(self) -> Float[Array, "... out_channels in_channels"]:
        return self.spec.layout.to_output_input(self.weights)

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
        return self.weights[index, :].astype(dtype)

    def dot(
        self,
        vector: Float[Array, " channels"],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Float[Array, "... channels"]:
        self._raise_if_batched()
        if _use_cute_w4a16_dot(self.spec, self.packed_weights, vector, forward_pass_config, transposed):
            from .cute_w4a16_dot import mlx_w4a16_dot

            result = mlx_w4a16_dot(
                vector,
                self.packed_weights,
                self.scales.astype(vector.dtype),
                self.biases.astype(vector.dtype),
            )
            return reshard_as(result, vector)

        weights = self.weights.astype(vector.dtype)
        layout = self.spec.layout
        if transposed:
            layout = layout.transpose()
        with use_dot_algorithm_preset(forward_pass_config.precision):
            result = layout.matmul(weights, vector)

        return reshard_as(result, vector)
