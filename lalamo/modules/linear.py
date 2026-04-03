from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import accumulate
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, PRNGKeyArray

from lalamo.arrays import ArrayForwardPassConfig, CompressedArray, FullPrecisionArray, quant_array_import_weights
from lalamo.arrays.quant_format import QuantFormat
from lalamo.common import ParameterTree, dummy_array, require_array, require_mapping
from lalamo.quantization import QuantizationMode, dynamically_quantize_activations

from .common import LalamoModule, ShardingOrder, TensorSharding, sharded_field

__all__ = [
    "Linear",
    "LinearConfig",
    "QuantFormat",
]


@dataclass(frozen=True)
class LinearConfig:
    precision: DTypeLike
    quant_format: QuantFormat = QuantFormat.FULL_PRECISION
    group_size: int | None = None
    bits: int | None = None
    activation_quantization_mode: QuantizationMode | None = None

    def __post_init__(self) -> None:
        if self.quant_format == QuantFormat.FULL_PRECISION:
            if self.group_size is not None or self.bits is not None:
                raise ValueError("group_size and bits must be None for FULL_PRECISION format")
        else:
            if self.group_size is None or self.bits is None:
                raise ValueError(f"group_size and bits are required for {self.quant_format.name} format")

    def _quant_kwargs(self) -> dict[str, int]:
        if self.quant_format == QuantFormat.FULL_PRECISION:
            return {}
        return dict(group_size=self.group_size, bits=self.bits)  # type: ignore[arg-type]

    def _build(
        self,
        leading_dims: tuple[int, ...],
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> Linear:
        total_out = sum(output_dims)
        weights = self.quant_format.array_class.empty(
            leading_dims, total_out, input_dim, self.precision, **self._quant_kwargs()
        )
        biases = dummy_array((*leading_dims, total_out), self.precision) if has_biases else None
        return Linear(config=self, output_dims=output_dims, weights=weights, biases=biases)

    def empty(self, input_dim: int, output_dims: tuple[int, ...], has_biases: bool) -> Linear:
        return self._build((), input_dim, output_dims, has_biases)

    def empty_mixture(
        self, mixture_size: int, input_dim: int, output_dims: tuple[int, ...], has_biases: bool
    ) -> Linear:
        return self._build((mixture_size,), input_dim, output_dims, has_biases)

    def random_init(
        self, input_dim: int, output_dims: tuple[int, ...], has_biases: bool, *, key: PRNGKeyArray
    ) -> Linear:
        total_out = sum(output_dims)
        scale = 1 / math.sqrt(input_dim)
        raw = jax.random.uniform(key, (total_out, input_dim), minval=-scale, maxval=scale, dtype=self.precision)
        full = FullPrecisionArray(raw=raw)
        if self.quant_format == QuantFormat.FULL_PRECISION:
            weights: CompressedArray = full
        else:
            exported = full.export_weights()
            weights = quant_array_import_weights(
                exported,
                quant_format=self.quant_format,
                precision=self.precision,
                group_size=self.group_size,
                bits=self.bits,
            )
        biases = jnp.zeros((total_out,), dtype=self.precision) if has_biases else None
        return Linear(config=self, output_dims=output_dims, weights=weights, biases=biases)

    def random_init_mixture(
        self,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        key: PRNGKeyArray,
    ) -> Linear:
        subkeys = jax.random.split(key, mixture_size)
        return eqx.filter_vmap(lambda k: self.random_init(input_dim, output_dims, has_biases, key=k))(subkeys)


class Linear(LalamoModule[LinearConfig]):
    output_dims: tuple[int, ...] = eqx.field(static=True)
    weights: CompressedArray = sharded_field(
        tensor_sharding=TensorSharding(
            axes=(-2, -1),
            axes_names=(ShardingOrder.OUTPUT, ShardingOrder.INPUT),
        ),
    )
    biases: Float[Array, "*batch total_out_channels"] | None
    sharding_order: ShardingOrder | None = eqx.field(static=True, default=None, kw_only=True)

    def __check_init__(self) -> None:
        *_, weight_out, weight_in = self.weights.materialize().shape
        expected_out = sum(self.output_dims)
        if weight_out != expected_out:
            raise ValueError(f"Weight out_channels ({weight_out}) != sum(output_dims) ({expected_out})")
        if self.biases is not None:
            *_, bias_out = self.biases.shape
            if bias_out != expected_out:
                raise ValueError(f"Bias size ({bias_out}) != sum(output_dims) ({expected_out})")

    @property
    def input_dim(self) -> int:
        return self.weights.materialize().shape[-1]

    @property
    def has_biases(self) -> bool:
        return self.biases is not None

    @property
    def mixture_size(self) -> int | None:
        return self.weights.materialize().shape[0] if self.weights.materialize().ndim > 2 else None

    @property
    def num_outputs(self) -> int:
        return len(self.output_dims)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @staticmethod
    def get_split_points(output_dims: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(accumulate(output_dims[:-1]))

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, " in_channels"],
        forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig(),
    ) -> tuple[Float[Array, " out_channels"], ...]:
        if self.mixture_size is not None:
            raise ValueError(
                "Mixtures of linear layers cannot be called directly. Use eqx.filter_vmap or lax.scan instead.",
            )
        if self.config.activation_quantization_mode is not None:
            inputs = dynamically_quantize_activations(inputs, self.config.activation_quantization_mode)
        result = self.weights.dot(inputs, forward_pass_config)
        if self.biases is not None:
            result = result + self.biases
        return tuple(jnp.split(result, self.get_split_points(self.output_dims)))

    def export_weights(self) -> ParameterTree:
        result = dict(self.weights.export_weights())
        if self.biases is not None:
            result["biases"] = self.biases
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights_map = require_mapping(weights)
        new_biases = require_array(weights_map["biases"]) if self.has_biases else None
        base = quant_array_import_weights(
            weights_map,
            quant_format=self.config.quant_format,
            precision=self.config.precision,
            group_size=self.config.group_size,
            bits=self.config.bits,
        )
        return eqx.tree_at(
            lambda m: (m.weights, m.biases),
            self,
            (base, new_biases),
            is_leaf=lambda x: x is None,
        )  # type: ignore[return-value]
