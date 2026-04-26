import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple, Self

from jax import ShapeDtypeStruct, default_matmul_precision, device_put, lax
from jax.sharding import Sharding
from jaxtyping import Array, DTypeLike, Float, Int, Key

from lalamo.initializer import Initializer
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding
from lalamo.weight_matrix import (
    EmbeddingMatrix,
    Layout,
    MatmulConfig,
    Preconditioner,
    WeightMatrixSpec,
    matmul_with_layout,
    partition_and_shape_for_layout,
    to_stored_layout,
)

from .utils import (
    expand_last_axis_groups,
    group_by_last_axis,
    grouped_last_axis_shape,
    min_max_scale,
    min_max_within_groups,
    round_to_unsigned_grid,
    round_to_unsigned_grid_for_config,
)

__all__ = [
    "MLXMatrix",
    "MLXSpec",
]


class MLXAffineParameters(NamedTuple):
    scales: Float[Array, "... rows groups"]
    biases: Float[Array, "... rows groups"]

    @classmethod
    def from_weights(
        cls,
        weights: Float[Array | ShapeDtypeStruct, "... rows cols"],
        *,
        bits: int,
        group_size: int,
        sharding: Sharding | None,
    ) -> Self:
        grouped_shape = grouped_last_axis_shape(weights.shape, group_size=group_size)
        if isinstance(weights, ShapeDtypeStruct):
            return cls(
                scales=dummy_array(grouped_shape, weights.dtype, sharding),
                biases=dummy_array(grouped_shape, weights.dtype, sharding),
            )

        group_min_max = min_max_within_groups(group_by_last_axis(weights, group_size=group_size))
        scales = min_max_scale(group_min_max, bits=bits, dtype=weights.dtype)
        return cls(scales=device_put(scales, sharding), biases=device_put(group_min_max.min, sharding))


def _affine_quantize(
    weights: Float[Array, "... rows cols"],
    scales: Float[Array, "... rows groups"],
    biases: Float[Array, "... rows groups"],
    *,
    group_size: int,
    round_fn: Callable[[Float[Array, "... rows cols"]], Float[Array, "... rows cols"]],
) -> Float[Array, "... rows cols"]:
    expanded_scales = lax.stop_gradient(expand_last_axis_groups(scales, group_size=group_size))
    expanded_biases = lax.stop_gradient(expand_last_axis_groups(biases, group_size=group_size))
    normalized_weights = (weights - expanded_biases) / expanded_scales
    return round_fn(normalized_weights) * expanded_scales + expanded_biases


@dataclass(frozen=True)
class MLXSpec(WeightMatrixSpec):
    bits: Literal[4, 8]
    group_size: int
    layout: Layout = Layout.OUTPUT_INPUT

    def compress(
        self,
        weights: Float[Array | ShapeDtypeStruct, "... out_channels in_channels"],
        preconditioner: Preconditioner | None = None,
    ) -> "MLXMatrix":
        if preconditioner is not None:
            raise ValueError("Preconditioned rounding is not implemented yet.")
        *leading_dims, output_dim, input_dim = weights.shape
        partition = partition_and_shape_for_layout(
            layout=self.layout,
            leading_dims=leading_dims,
            output_dim=output_dim,
            input_dim=input_dim,
        ).partition
        sharding = make_sharding(partition)

        stored_weights = to_stored_layout(weights, layout=self.layout, sharding=sharding)
        affine_parameters = MLXAffineParameters.from_weights(
            stored_weights,
            bits=self.bits,
            group_size=self.group_size,
            sharding=sharding,
        )
        return MLXMatrix(
            spec=self,
            weights=stored_weights,
            scales=affine_parameters.scales,
            biases=affine_parameters.biases,
        )

    def init(
        self,
        initializer: Initializer,
        leading_dims: tuple[int, ...],
        output_dim: int,
        input_dim: int,
    ) -> "MLXMatrix":
        _, weight_shape = partition_and_shape_for_layout(
            layout=Layout.OUTPUT_INPUT,
            leading_dims=leading_dims,
            output_dim=output_dim,
            input_dim=input_dim,
        )
        weights = initializer.normal(1 / math.sqrt(input_dim), shape=weight_shape)
        return self.compress(weights)


class MLXMatrix(EmbeddingMatrix[MLXSpec]):
    weights: Float[Array, "... rows cols"]
    scales: Float[Array, "... rows groups"]
    biases: Float[Array, "... rows groups"]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.weights.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.weights.dtype

    def astype(self, dtype: DTypeLike) -> "MLXMatrix":
        return MLXMatrix(
            spec=self.spec,
            weights=self.weights.astype(dtype),
            scales=self.scales.astype(dtype),
            biases=self.biases.astype(dtype),
        )

    def decompress(self) -> Float[Array, "..."]:
        return _affine_quantize(
            self.weights,
            self.scales,
            self.biases,
            group_size=self.spec.group_size,
            round_fn=partial(round_to_unsigned_grid, bits=self.spec.bits),
        )

    def lookup_embedding(
        self,
        index: int | Int[Array, ""],
        *,
        dequant_key: Key[Array, ""],
        forward_pass_config: MatmulConfig | None = None,
    ) -> Float[Array, "... out_channels"]:
        self._raise_if_batched()
        forward_pass_config = forward_pass_config or MatmulConfig()
        if self.spec.layout != Layout.INPUT_OUTPUT:
            raise ValueError(f"Embedding lookup not supported for layout {self.spec.layout}")
        return _affine_quantize(
            self.weights[index, :],
            self.scales[index, :],
            self.biases[index, :],
            group_size=self.spec.group_size,
            round_fn=partial(
                round_to_unsigned_grid_for_config,
                bits=self.spec.bits,
                dequant_key=dequant_key,
                forward_pass_config=forward_pass_config,
            ),
        )

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        dequant_key: Key[Array, ""],
        forward_pass_config: MatmulConfig | None = None,
    ) -> Float[Array, "... out_channels"]:
        self._raise_if_batched()
        forward_pass_config = forward_pass_config or MatmulConfig()
        dequantized_weights = _affine_quantize(
            self.weights,
            self.scales,
            self.biases,
            group_size=self.spec.group_size,
            round_fn=partial(
                round_to_unsigned_grid_for_config,
                bits=self.spec.bits,
                dequant_key=dequant_key,
                forward_pass_config=forward_pass_config,
            ),
        )
        with default_matmul_precision(forward_pass_config.precision):
            return matmul_with_layout(dequantized_weights, vector, layout=self.spec.layout)
