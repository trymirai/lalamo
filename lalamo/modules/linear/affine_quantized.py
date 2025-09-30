import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, dummy_array
from lalamo.quantization import AffineQuantizationMode, affine_quantize_weights, dynamically_quantize_activations

from .full_precision import LinearBase, LinearConfigBase

__all__ = [
    "AffineQuantizedLinear",
    "AffineQuantizedLinearConfig",
]


@dataclass(frozen=True)
class AffineQuantizedLinearConfig(LinearConfigBase):
    group_size: int
    weight_quantization_mode: AffineQuantizationMode
    activation_quantization_mode: AffineQuantizationMode | None
    activation_precision: DTypeLike

    def random_init(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        key: PRNGKeyArray,
    ) -> "AffineQuantizedLinearBase":
        min_val, max_val = self.weight_quantization_mode.range
        weights = jax.random.uniform(
            key,
            (sum(output_dims), input_dim),
            minval=min_val - 1,
            maxval=max_val + 1,
            dtype=self.activation_precision,
        )
        num_groups = input_dim // self.group_size
        scale = 1 / ((max_val - min_val) / 2 * math.sqrt(input_dim))
        scales = scale * jnp.ones((sum(output_dims), num_groups), dtype=self.activation_precision)

        if has_biases:
            biases = jnp.zeros((sum(output_dims),), dtype=self.activation_precision)
        else:
            biases = None

        zero_point = min_val + 2 ** (self.weight_quantization_mode.bits - 1)
        zero_points = zero_point * jnp.ones((sum(output_dims), num_groups), dtype=self.activation_precision)

        return AffineQuantizedLinear(
            config=self,
            output_dims=output_dims,
            weights=weights,
            scales=scales,
            zero_points=zero_points,
            biases=biases,
        )

    def random_init_mixture(
        self,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        key: PRNGKeyArray,
    ) -> "AffineQuantizedLinearBase":
        subkeys = jax.random.split(key, mixture_size)
        return eqx.filter_vmap(lambda key: self.random_init(input_dim, output_dims, has_biases, key=key))(subkeys)

    def _empty_general(
        self,
        leading_dims: tuple[int, ...],
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "AffineQuantizedLinearBase":
        weights = dummy_array(
            (*leading_dims, sum(output_dims), input_dim),
            dtype=self.activation_precision,
        )
        num_groups = input_dim // self.group_size
        scales = dummy_array((*leading_dims, sum(output_dims), num_groups), dtype=self.activation_precision)

        if has_biases:
            biases = dummy_array((*leading_dims, sum(output_dims)), dtype=self.activation_precision)
        else:
            biases = None
        zero_points = dummy_array((*leading_dims, sum(output_dims), num_groups), dtype=self.activation_precision)

        return AffineQuantizedLinear(
            config=self,
            output_dims=output_dims,
            weights=weights,
            scales=scales,
            zero_points=zero_points,
            biases=biases,
        )

    def empty(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "AffineQuantizedLinearBase":
        return self._empty_general((), input_dim, output_dims, has_biases)

    def empty_mixture(
        self,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "AffineQuantizedLinearBase":
        return self._empty_general((mixture_size,), input_dim, output_dims, has_biases)


class AffineQuantizedLinearBase[ConfigT: AffineQuantizedLinearConfig](LinearBase[ConfigT]):
    weights: Float[Array, "*components total_out_channels in_channels"]
    scales: Float[Array, "*components total_out_channels groups"]
    zero_points: Float[Array, "*components total_out_channels groups"]
    biases: Float[Array, "*components total_out_channels"] | None

    @property
    def mixture_size(self) -> int | None:
        match self.weights.shape:
            case [num_components, _, _]:
                return num_components
            case _:
                return None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.activation_precision

    @property
    def input_dim(self) -> int:
        *_, _, input_dim = self.weights.shape
        return input_dim

    @property
    def has_biases(self) -> bool:
        return self.biases is not None

    @property
    def num_groups(self) -> int:
        return self.input_dim // self.config.group_size

    @property
    def int_weights(self) -> Int[Array, "*components in_channels out_channels"]:
        result = affine_quantize_weights(self.weights, self.config.weight_quantization_mode)
        return result.astype(self.config.weight_quantization_mode.dtype)

    @property
    def int_zero_points(self) -> Int[Array, "*components groups out_channels"]:
        result = affine_quantize_weights(self.zero_points, self.config.weight_quantization_mode)
        return result.astype(self.config.weight_quantization_mode.dtype)

    def __post_init__(self) -> None:  # noqa: PLR0912
        if self.weights.dtype != self.config.activation_precision:
            raise ValueError(
                f"Weight dtype ({self.weights.dtype}) is not equal to specified activation precision"
                f" ({self.config.activation_precision}).",
                " Quantized layers require parameter dtypes to be equal to the activation precision.",
            )
        *w_num_components, w_output_dim, _ = self.weights.shape
        if w_output_dim != sum(self.output_dims):
            raise ValueError(
                f"Number of output channels in weights ({w_output_dim}) is not"
                f" equal to sum of output dims ({sum(self.output_dims)}).",
            )

        if self.scales.dtype != self.config.activation_precision:
            raise ValueError(
                f"Scale dtype ({self.scales.dtype}) is not equal to specified activation precision"
                f" ({self.config.activation_precision}).",
                " Quantized layers require parameter dtypes to be equal to the activation precision.",
            )
        *s_num_components, s_output_dim, s_num_groups = self.scales.shape
        if w_output_dim != s_output_dim:
            raise ValueError(
                f"Number of output channels in weights ({w_output_dim}) is not"
                f" equal to number of output channels in scales ({s_output_dim}).",
            )
        if tuple(s_num_components) != tuple(w_num_components):
            raise ValueError(
                f"Number of mixture components in weights ({w_num_components}) is not"
                f" equal to number of mixture components in scales ({s_num_components}).",
            )
        if s_num_groups != self.num_groups:
            raise ValueError(
                f"Number of groups in scales ({s_num_groups}) is incompatible with"
                f" the specified group size ({self.config.group_size}).",
            )

        if self.zero_points.dtype != self.config.activation_precision:
            raise ValueError(
                f"Zero point dtype ({self.zero_points.dtype}) is not equal to specified activation precision"
                f" ({self.config.activation_precision}).",
                " Quantized layers require parameter dtypes to be equal to the activation precision.",
            )
        *zp_num_components, zp_output_dim, zp_num_groups = self.zero_points.shape
        if w_output_dim != zp_output_dim:
            raise ValueError(
                f"Number of output channels in weights ({w_output_dim}) is not"
                f" equal to number of output channels in zero points ({zp_output_dim}).",
            )
        if tuple(zp_num_components) != tuple(w_num_components):
            raise ValueError(
                f"Number of mixture components in weights ({w_num_components}) is not"
                f" equal to number of mixture components in zero points ({zp_num_components}).",
            )
        if self.num_groups != zp_num_groups:
            raise ValueError(
                f"Number of groups in zero points ({zp_num_groups}) is incompatible with"
                f" the specified group size ({self.config.group_size}).",
            )

        if self.biases is not None:
            if self.biases.dtype != self.config.activation_precision:
                raise ValueError(
                    f"Bias dtype ({self.biases.dtype}) is not equal to specified activation precision"
                    f" ({self.config.activation_precision}).",
                    " Quantized layers require parameter dtypes to be equal to the activation precision.",
                )
            *b_num_components, b_output_dim = self.biases.shape
            if w_output_dim != b_output_dim:
                raise ValueError(
                    f"Number of output channels in weights ({w_output_dim}) is not"
                    f" equal to number of output channels in biases ({b_output_dim}).",
                )
            if tuple(b_num_components) != tuple(w_num_components):
                raise ValueError(
                    f"Number of mixture components in weights ({w_num_components}) is not"
                    f" equal to number of mixture components in biases ({b_num_components}).",
                )

    def _prepare_scaled_weights(self) -> Float[Array, "*components in_channels total_out_channels"]:
        quantized_weights = affine_quantize_weights(self.weights, self.config.weight_quantization_mode)
        grouped_weights = rearrange(
            quantized_weights,
            "... total_out_channels (groups group_channels) -> ... total_out_channels groups group_channels",
            groups=self.num_groups,
        )

        zero_points = rearrange(self.zero_points, "... total_out_channels groups -> ... total_out_channels groups 1")
        grouped_weights = grouped_weights - zero_points

        scales = rearrange(self.scales, "... total_out_channels groups -> ... total_out_channels groups 1")
        scaled_grouped_weights = grouped_weights * scales
        result = rearrange(
            scaled_grouped_weights,
            "... total_out_channels groups group_channels -> ... total_out_channels (groups group_channels)",
        )
        return result

    def _apply_weights(self, inputs: Float[Array, " in_channels"]) -> Float[Array, " total_out_channels"]:
        if self.config.activation_quantization_mode is not None:
            inputs = dynamically_quantize_activations(inputs, self.config.activation_quantization_mode)
        return self._prepare_scaled_weights() @ inputs

    @eqx.filter_jit
    def __call__(self, inputs: Float[Array, " in_channels"]) -> tuple[Float[Array, " out_channels"], ...]:
        if self.mixture_size is not None:
            raise ValueError(
                "Mixtures of linear layers cannot be called directly."
                "They are intended to be used with methods eqx.filter_vmap or lax.scan instead.",
            )
        result = self._apply_weights(inputs)
        if self.biases is not None:
            result = result + self.biases
        return tuple(jnp.split(result, self._get_split_points(self.output_dims)))

    def export_weights(self) -> ParameterTree:
        result = dict(
            weights=self.int_weights,
            zero_points=self.int_zero_points,
            scales=self.scales,
        )
        if self.biases is not None:
            result["biases"] = self.biases
        return result

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        assert isinstance(weights, Mapping)
        assert isinstance(weights["weights"], Array)
        assert isinstance(weights["zero_points"], Array)
        return replace(
            self,
            weights=weights["weights"].astype(self.weights.dtype),
            scales=weights["scales"],
            zero_points=weights["zero_points"].astype(self.zero_points.dtype),
            biases=weights["biases"] if self.has_biases else None,
        )


class AffineQuantizedLinear(AffineQuantizedLinearBase[AffineQuantizedLinearConfig]):
    pass
