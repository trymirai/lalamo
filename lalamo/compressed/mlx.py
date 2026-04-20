from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int, Key

from lalamo.compressed.utils import into_layout
from lalamo.initializer import Initializer
from lalamo.weight_matrix import (
    EmbeddingMatrix,
    GradientEstimator,
    Layout,
    MatmulConfig,
    Preconditioner,
    WeightMatrixSpec,
)

from .quantization_helpers import quantize_to_grid
from .utils import (
    expand_group_parameter,
    group_by_last_axis,
    group_parameter_partition,
    lookup_group_parameter,
    lookup_output,
    matmul_with_layout,
    weight_matrix_partition,
)

__all__ = [
    "MLXMatrix",
    "MLXSpec",
]


def _expand_lookup_parameter(
    grouped: Float[Array, " groups"],
    *,
    group_size: int,
) -> Float[Array, " in_channels"]:
    return jnp.repeat(grouped, group_size)


@dataclass(frozen=True)
class MLXSpec(WeightMatrixSpec):
    bits: Literal[4, 8]
    group_size: int
    layout: Layout = Layout.OUTPUT_INPUT

    def compress(
        self, weights: Float[Array, "... out_channels in_channels"], preconditioner: Preconditioner | None = None
    ) -> "MLXMatrix":
        stored_weights = into_layout(weights, self.layout)
        grouped = group_by_last_axis(stored_weights, group_size=self.group_size)
        group_mins = jnp.min(grouped, axis=-1)
        group_maxs = jnp.max(grouped, axis=-1)
        quant_levels = (2**self.bits) - 1
        scales = jnp.maximum((group_maxs - group_mins) / quant_levels, jnp.finfo(stored_weights.dtype).eps)
        safe_scales = expand_group_parameter(scales, layout=self.layout, group_size=self.group_size)
        expanded_biases = expand_group_parameter(
            group_mins,
            layout=self.layout,
            group_size=self.group_size,
        )
        quantized_weights = quantize_to_grid((stored_weights - expanded_biases) / safe_scales, self.bits)
        return MLXMatrix(
            spec=self,
            weights=quantized_weights,
            scales=scales.astype(self.float_dtype),
            biases=group_mins.astype(self.float_dtype),
        )

    def init(
        self,
        initializer: Initializer,
        leading_dims: tuple[int, ...],
        output_dim: int,
        input_dim: int,
    ) -> "MLXMatrix":
        if input_dim % self.group_size != 0:
            raise ValueError(f"Input dimension {input_dim} must be divisible by group size {self.group_size}")
        num_groups = input_dim // self.group_size
        if self.layout == Layout.INPUT_OUTPUT:
            weight_shape = (*leading_dims, input_dim, output_dim)
            grouped_shape = (*leading_dims, num_groups, output_dim)
        else:
            weight_shape = (*leading_dims, output_dim, input_dim)
            grouped_shape = (*leading_dims, output_dim, num_groups)
        return MLXMatrix(
            spec=self,
            weights=initializer.zeros(weight_shape, partition=weight_matrix_partition(self.layout, leading_dims)),
            scales=initializer.ones(grouped_shape, partition=group_parameter_partition(self.layout, leading_dims)),
            biases=initializer.zeros(grouped_shape, partition=group_parameter_partition(self.layout, leading_dims)),
        )


class MLXMatrix(EmbeddingMatrix[MLXSpec]):
    weights: Float[Array, "..."]
    scales: Float[Array, "..."]
    biases: Float[Array, "..."]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.weights.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.scales.dtype

    def _dequantize(
        self,
        quantized_weights: Float[Array, "..."],
    ) -> Float[Array, "..."]:
        expanded_scales = expand_group_parameter(
            self.scales,
            layout=self.spec.layout,
            group_size=self.spec.group_size,
        )
        expanded_biases = expand_group_parameter(
            self.biases,
            layout=self.spec.layout,
            group_size=self.spec.group_size,
        )
        return quantized_weights * expanded_scales + expanded_biases

    def _quantized_weights(self) -> Float[Array, "..."]:
        return quantize_to_grid(self.weights, self.spec.bits)

    def _projected_weights(self) -> Float[Array, "..."]:
        quantized_weights = self._quantized_weights()
        return self.weights + jax.lax.stop_gradient(quantized_weights - self.weights)

    def _lookup_dequantized_output(
        self,
        index: int | Int[Array, ""],
        quantized_weights: Float[Array, "..."],
    ) -> Float[Array, " in_channels"]:
        quantized_output = lookup_output(quantized_weights, layout=self.spec.layout, index=index)
        scales = lookup_group_parameter(self.scales, layout=self.spec.layout, index=index)
        biases = lookup_group_parameter(self.biases, layout=self.spec.layout, index=index)
        expanded_scales = _expand_lookup_parameter(scales, group_size=self.spec.group_size)
        expanded_biases = _expand_lookup_parameter(biases, group_size=self.spec.group_size)
        return quantized_output * expanded_scales + expanded_biases

    def _lookup_variance(
        self,
        index: int | Int[Array, ""],
        projected_weights: Float[Array, "..."],
    ) -> Float[Array, " in_channels"]:
        squared_error = lookup_output(
            jax.lax.stop_gradient((self.weights - projected_weights) ** 2),
            layout=self.spec.layout,
            index=index,
        )
        scales = lookup_group_parameter(self.scales, layout=self.spec.layout, index=index)
        expanded_scale_sq = _expand_lookup_parameter(scales**2, group_size=self.spec.group_size)
        return squared_error * expanded_scale_sq

    def decompress(self) -> Float[Array, "..."]:
        return self._dequantize(self._quantized_weights())

    def lookup_embedding(
        self,
        index: int | Int[Array, ""],
        *,
        dequant_key: Key[Array, ""],
        forward_pass_config: MatmulConfig | None = None,
    ) -> Float[Array, "... out_channels"]:
        self._raise_if_batched()
        forward_pass_config = forward_pass_config or MatmulConfig()
        projected_weights = self._projected_weights()
        result = self._lookup_dequantized_output(index, projected_weights)
        if forward_pass_config.gradient_estimator == GradientEstimator.STOCHASTIC:
            variance = self._lookup_variance(index, projected_weights)
            noise = jax.random.normal(dequant_key, result.shape, dtype=result.dtype)
            result = result + jnp.sqrt(variance + 1e-6) * noise
        return result

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        dequant_key: Key[Array, ""],
        forward_pass_config: MatmulConfig | None = None,
    ) -> Float[Array, "... out_channels"]:
        self._raise_if_batched()
        forward_pass_config = forward_pass_config or MatmulConfig()
        projected_weights = self._projected_weights()
        mean = matmul_with_layout(
            self._dequantize(projected_weights),
            vector,
            layout=self.spec.layout,
        )
        if forward_pass_config.gradient_estimator == GradientEstimator.DETERMINISTIC:
            return mean

        squared_error = jax.lax.stop_gradient((self.weights - projected_weights) ** 2)
        expanded_scale_sq = expand_group_parameter(
            self.scales**2,
            layout=self.spec.layout,
            group_size=self.spec.group_size,
        )
        variance = matmul_with_layout(
            squared_error * expanded_scale_sq,
            vector**2,
            layout=self.spec.layout,
        )
        noise = jax.random.normal(dequant_key, mean.shape, dtype=mean.dtype)
        return mean + jnp.sqrt(variance + 1e-6) * noise
