import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, PRNGKeyArray

from lalamo.common import ParameterTree, dummy_array

from .affine_quantized import AffineQuantizedLinear, AffineQuantizedLinearBase, AffineQuantizedLinearConfig

__all__ = [
    "QLoRALinear",
    "QLoRALinearConfig",
]


@dataclass(frozen=True)
class QLoRALinearConfig(AffineQuantizedLinearConfig):
    lora_rank: int
    lora_scale: float
    activation_precision: DTypeLike

    def random_init(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        key: PRNGKeyArray,
    ) -> "QLoRALinear":
        base_key, derived_key = jax.random.split(key)
        group_quantized_linear = super().random_init(input_dim, output_dims, has_biases, key=base_key)
        assert isinstance(group_quantized_linear, AffineQuantizedLinear)

        down_key, up_key_root = jax.random.split(derived_key)
        hidden_lora_rank = len(output_dims) * self.lora_rank
        max_down_abs_value = 1 / math.sqrt(input_dim)
        lora_down_weights = jax.random.uniform(
            down_key,
            (input_dim, hidden_lora_rank),
            minval=-max_down_abs_value,
            maxval=max_down_abs_value,
            dtype=self.activation_precision,
        )

        up_keys = jax.random.split(up_key_root, len(output_dims))
        max_up_abs_value = 1 / math.sqrt(hidden_lora_rank)
        lora_up_weights = tuple(
            jax.random.uniform(
                up_key,
                (self.lora_rank, output_dim),
                minval=-max_up_abs_value,
                maxval=max_up_abs_value,
                dtype=self.activation_precision,
            )
            for up_key, output_dim in zip(up_keys, output_dims, strict=True)
        )

        return QLoRALinear(
            config=self,
            output_dims=output_dims,
            weights=group_quantized_linear.weights,
            scales=group_quantized_linear.scales,
            biases=group_quantized_linear.biases,
            zero_points=group_quantized_linear.zero_points,
            lora_down_weights=lora_down_weights,
            lora_up_weights=lora_up_weights,
        )

    def random_init_mixture(
        self,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        key: PRNGKeyArray,
    ) -> "QLoRALinear":
        subkeys = jax.random.split(key, mixture_size)
        return eqx.filter_vmap(lambda k: self.random_init(input_dim, output_dims, has_biases, key=k))(subkeys)

    def empty(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "QLoRALinear":
        group_quantized_linear = super().empty(input_dim, output_dims, has_biases)
        assert isinstance(group_quantized_linear, AffineQuantizedLinear)
        hidden_lora_rank = len(output_dims) * self.lora_rank
        lora_down_weights = dummy_array(
            (input_dim, hidden_lora_rank),
            dtype=self.activation_precision,
        )
        lora_up_weights = tuple(
            dummy_array(
                (self.lora_rank, output_dim),
                dtype=self.activation_precision,
            )
            for output_dim in output_dims
        )

        return QLoRALinear(
            config=self,
            output_dims=output_dims,
            weights=group_quantized_linear.weights,
            scales=group_quantized_linear.scales,
            biases=group_quantized_linear.biases,
            zero_points=group_quantized_linear.zero_points,
            lora_down_weights=lora_down_weights,
            lora_up_weights=lora_up_weights,
        )

    def _empty_general(
        self,
        leading_dims: tuple[int, ...],
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "QLoRALinear":
        group_quantized_linear = super().empty(input_dim, output_dims, has_biases)
        assert isinstance(group_quantized_linear, AffineQuantizedLinear)

        hidden_lora_rank = len(output_dims) * self.lora_rank
        lora_down_weights = dummy_array(
            (*leading_dims, input_dim, hidden_lora_rank),
            dtype=self.activation_precision,
        )
        lora_up_weights = tuple(
            dummy_array(
                (*leading_dims, self.lora_rank, output_dim),
                dtype=self.activation_precision,
            )
            for output_dim in output_dims
        )

        return QLoRALinear(
            config=self,
            output_dims=output_dims,
            weights=group_quantized_linear.weights,
            scales=group_quantized_linear.scales,
            biases=group_quantized_linear.biases,
            zero_points=group_quantized_linear.zero_points,
            lora_down_weights=lora_down_weights,
            lora_up_weights=lora_up_weights,
        )

    def empty_mixture(
        self,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "QLoRALinear":
        return self._empty_general((mixture_size,), input_dim, output_dims, has_biases)


class QLoRALinear(AffineQuantizedLinearBase[QLoRALinearConfig]):
    lora_down_weights: Float[Array, "*components in_channels total_lora_channels"]
    lora_up_weights: tuple[Float[Array, "*components lora_channels out_channels"], ...]

    def _split_biases(self) -> tuple[Float[Array, "*components out_channels"] | None, ...]:
        if self.biases is not None:
            return tuple(jnp.split(self.biases, self._get_split_points(self.output_dims)))
        return (None,) * len(self.output_dims)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.lora_down_weights.dtype != self.config.activation_precision:
            raise ValueError(
                f"LORA down weight dtype ({self.lora_down_weights.dtype}) is not equal to the"
                f" specified activation precision ({self.config.activation_precision}).",
                " Quantized layers require parameter dtypes to be equal to the activation precision.",
            )
        *ld_num_components, lora_down_input_dim, lora_down_output_dim = self.lora_down_weights.shape
        if lora_down_output_dim != self.config.lora_rank * self.num_outputs:
            raise ValueError(
                f"Number of output channels in LORA down weights ({lora_down_output_dim}) is not"
                f" equal to lora_rank * num_outputs ({self.config.lora_rank * self.num_outputs}).",
            )
        if lora_down_input_dim != self.input_dim:
            raise ValueError(
                f"Number of input channels in LORA down weights ({lora_down_input_dim}) is not"
                f" equal to input_dim ({self.input_dim}).",
            )
        *w_num_components, _, _ = self.weights.shape
        if tuple(ld_num_components) != tuple(w_num_components):
            raise ValueError(
                f"Number of mixture components in LORA down weights ({ld_num_components}) is not"
                f" equal to number of mixture components in base weights ({w_num_components}).",
            )
        if len(self.lora_up_weights) != self.num_outputs:
            raise ValueError(
                f"Expected {self.num_outputs} LORA up weights, got {len(self.lora_up_weights)}.",
            )
        for lora_up_weight, output_dim in zip(self.lora_up_weights, self.output_dims, strict=True):
            if lora_up_weight.dtype != self.config.activation_precision:
                raise ValueError(
                    f"LORA up weight dtype ({lora_up_weight.dtype}) is not equal to specified activation precision"
                    f" ({self.config.activation_precision}).",
                    " Quantized layers require parameter dtypes to be equal to the activation precision.",
                )
            *lu_num_components, lora_up_input_dim, lora_up_output_dim = lora_up_weight.shape
            if lora_up_output_dim != output_dim:
                raise ValueError(
                    f"Number of output channels in LORA up weights ({lora_up_output_dim}) is not"
                    f" equal to number of output dims ({self.output_dims}).",
                )
            if lora_up_input_dim != self.config.lora_rank:
                raise ValueError(
                    f"Number of input channels in LORA up weights ({lora_up_input_dim}) is not"
                    f" equal to lora_rank ({self.config.lora_rank}).",
                )
            if tuple(lu_num_components) != tuple(w_num_components):
                raise ValueError(
                    f"Number of mixture components in LORA up weights ({lu_num_components}) is not"
                    f" equal to number of mixture components in base weights ({w_num_components}).",
                )

    @eqx.filter_jit
    def __call__(self, inputs: Float[Array, " in_channels"]) -> tuple[Float[Array, " out_channels"], ...]:
        if self.mixture_size is not None:
            raise ValueError(
                "Mixtures of linear layers cannot be called directly."
                "They are intended to be used with methods eqx.filter_vmap or lax.scan instead.",
            )
        joint_q_out = self._apply_weights(inputs)
        q_outs = jnp.split(joint_q_out, self._get_split_points(self.output_dims))

        joint_lora_hidden = inputs @ self.lora_down_weights
        lora_hiddens = jnp.split(joint_lora_hidden, self._get_split_points([self.config.lora_rank] * self.num_outputs))
        lora_outs = [
            lora_hidden @ lora_up_weight
            for lora_up_weight, lora_hidden in zip(self.lora_up_weights, lora_hiddens, strict=True)
        ]

        results = []
        for q_out, lora_out, bias in zip(q_outs, lora_outs, self._split_biases(), strict=True):
            result = q_out + self.config.lora_scale * lora_out
            if bias is not None:
                result = result + bias
            results.append(result)

        return tuple(results)

    def export_weights(self) -> ParameterTree:
        quantized_linear_weights: dict[str, ParameterTree] = super().export_weights()  # type: ignore
        return dict(
            down_weights=self.lora_down_weights,
            up_weights=self.lora_up_weights,
            **quantized_linear_weights,
        )

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        base = super().import_weights(weights)
        assert isinstance(weights, Mapping)
        assert isinstance(weights["up_weights"], Sequence)
        return replace(
            base,
            lora_down_weights=weights["down_weights"],
            lora_up_weights=tuple(up_weights for up_weights in weights["up_weights"]),
        )
