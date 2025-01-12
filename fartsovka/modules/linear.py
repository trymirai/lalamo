import math
from collections.abc import Sequence
from dataclasses import dataclass

import equinox as eqx
import jax
from einops import rearrange
from jax import numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from fartsovka.common import DEFAULT_PRECISION, DType
from fartsovka.quantization import QuantizationMode, dynamically_quantize_activations, quantize_weights

__all__ = [
    "GroupQuantizedLinear",
    "GroupQuantizedLinearFactory",
    "Linear",
    "LinearBase",
    "LinearFactory",
    "LinearFactoryBase",
    "QLoRALinear",
    "QLoRALinearFactory",
]


class LinearBase(eqx.Module):
    input_dim: int = eqx.field(static=True)
    output_dims: tuple[int, ...] = eqx.field(static=True)

    @property
    def num_outputs(self) -> int:
        return len(self.output_dims)

    @classmethod
    def _split_points(cls, output_dims: Sequence[int]) -> tuple[int, ...]:
        result = []
        last_split_point = 0
        for dim in output_dims[:-1]:
            last_split_point += dim
            result.append(last_split_point)
        return tuple(result)

    def __call__(
        self,
        x: Float[Array, " in_channels"],
    ) -> tuple[Float[Array, " out_channels"], ...]:
        raise NotImplementedError


@dataclass
class LinearFactoryBase[LinearType: LinearBase]:
    def __call__(self, input_dim: int, output_dims: tuple[int, ...], *, key: PRNGKeyArray) -> LinearType:
        raise NotImplementedError


class Linear(LinearBase):
    weights: Float[Array, "total_out_channels in_channels"]

    precision: DType = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        precision: DType,
        *,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__(input_dim=input_dim, output_dims=output_dims)

        self.precision = precision
        max_abs_value = 1 / math.sqrt(input_dim)
        self.weights = jax.random.uniform(
            key,
            (sum(output_dims), input_dim),
            minval=-max_abs_value,
            maxval=max_abs_value,
            dtype=self.precision,
        )

    def __call__(self, x: Float[Array, " in_channels"]) -> tuple[Float[Array, " out_channels"], ...]:
        return tuple(jnp.split(self.weights @ x, self._split_points(self.output_dims)))


@dataclass
class LinearFactory(LinearFactoryBase[Linear]):
    precision: DType = DEFAULT_PRECISION

    def __call__(self, input_dim: int, output_dims: tuple[int, ...], *, key: PRNGKeyArray) -> Linear:
        return Linear(input_dim, output_dims, precision=self.precision, key=key)


class GroupQuantizedLinear(LinearBase):
    group_size: int = eqx.field(static=True)
    weight_quantization_mode: QuantizationMode = eqx.field(static=True)
    activation_quantization_mode: QuantizationMode | None = eqx.field(static=True)

    @property
    def int_weights(self) -> Int[Array, "out_channels (groups in_channels)"]:
        result = quantize_weights(self.weights, self.weight_quantization_mode)
        return result.astype(self.weight_quantization_mode.dtype)

    @property
    def num_groups(self) -> int:
        return self.input_dim // self.group_size

    weights: Float[Array, "total_out_channels in_channels"]
    scales: Float[Array, "total_out_channels groups"]

    activation_precision: DType = eqx.field(static=True)

    def __init__(
        self,
        *,
        input_dim: int,
        output_dims: tuple[int, ...],
        group_size: int,
        weight_quantization_mode: QuantizationMode,
        activation_quantization_mode: QuantizationMode | None,
        activation_precision: DType,
        key: PRNGKeyArray,
    ) -> None:
        if input_dim % group_size != 0:
            raise ValueError(f"input_dim {input_dim} must be divisible by group_size {group_size}")
        super().__init__(input_dim=input_dim, output_dims=output_dims)
        self.group_size = group_size
        self.weight_quantization_mode = weight_quantization_mode
        self.activation_quantization_mode = activation_quantization_mode
        self.activation_precision = activation_precision

        min_val, max_val = weight_quantization_mode.range
        self.weights = jax.random.uniform(
            key,
            (sum(output_dims), input_dim),
            minval=min_val,
            maxval=max_val,
            dtype=activation_precision,
        )
        self.scales = jnp.ones((sum(output_dims), self.num_groups), dtype=activation_precision)

    def prepare_weights(self) -> Float[Array, "total_out_channels in_channels"]:
        quantized_weights = quantize_weights(self.weights, self.weight_quantization_mode)
        grouped_weights = rearrange(
            quantized_weights,
            "total_out_channels (groups group_channels) -> total_out_channels groups group_channels",
            groups=self.num_groups,
        )
        scales = rearrange(self.scales, "total_out_channels groups -> total_out_channels groups 1")
        scaled_grouped_weights = grouped_weights * scales
        result = rearrange(
            scaled_grouped_weights,
            "total_out_channels groups group_channels -> total_out_channels (groups group_channels)",
        )
        return result

    def __call__(self, x: Float[Array, " in_channels"]) -> tuple[Float[Array, " out_channels"], ...]:
        if self.activation_quantization_mode is not None:
            x = dynamically_quantize_activations(x, self.activation_quantization_mode)
        weights = self.prepare_weights()
        return tuple(jnp.split(weights @ x, self._split_points(self.output_dims)))


@dataclass
class GroupQuantizedLinearFactory(LinearFactoryBase[GroupQuantizedLinear]):
    group_size: int
    weight_quantization_mode: QuantizationMode
    activation_quantization_mode: QuantizationMode | None = None
    activation_precision: DType = DEFAULT_PRECISION

    def __call__(self, input_dim: int, output_dims: tuple[int, ...], *, key: PRNGKeyArray) -> GroupQuantizedLinear:
        return GroupQuantizedLinear(
            input_dim=input_dim,
            output_dims=output_dims,
            group_size=self.group_size,
            weight_quantization_mode=self.weight_quantization_mode,
            activation_quantization_mode=self.activation_quantization_mode,
            activation_precision=self.activation_precision,
            key=key,
        )


class QLoRALinear(GroupQuantizedLinear):
    lora_rank: int = eqx.field(static=True)
    lora_scale: float = eqx.field(static=True)

    lora_down_weights: Float[Array, "total_lora_channels in_channels"]
    lora_up_weights: tuple[Float[Array, "out_channels lora_channels"], ...]

    def __init__(
        self,
        *,
        input_dim: int,
        output_dims: tuple[int, ...],
        group_size: int,
        weight_quantization_mode: QuantizationMode,
        activation_quantization_mode: QuantizationMode | None,
        lora_rank: int,
        lora_scale: float,
        activation_precision: DType,
        key: PRNGKeyArray,
    ) -> None:
        linear_key, down_key, up_key_root = jax.random.split(key, 3)

        super().__init__(
            input_dim=input_dim,
            output_dims=output_dims,
            group_size=group_size,
            weight_quantization_mode=weight_quantization_mode,
            activation_quantization_mode=activation_quantization_mode,
            activation_precision=activation_precision,
            key=linear_key,
        )

        self.lora_rank = lora_rank
        self.lora_scale = lora_scale

        hidden_lora_rank = len(output_dims) * lora_rank
        max_down_abs_value = 1 / math.sqrt(input_dim)
        self.lora_down_weights = jax.random.uniform(
            down_key,
            (hidden_lora_rank, input_dim),
            minval=-max_down_abs_value,
            maxval=max_down_abs_value,
            dtype=activation_precision,
        )

        up_keys = jax.random.split(up_key_root, len(output_dims))
        max_up_abs_value = 1 / math.sqrt(hidden_lora_rank)
        self.lora_up_weights = tuple(
            jax.random.uniform(
                up_key,
                (output_dim, lora_rank),
                minval=-max_up_abs_value,
                maxval=max_up_abs_value,
                dtype=activation_precision,
            )
            for up_key, output_dim in zip(up_keys, output_dims, strict=True)
        )

    def __call__(self, x: Float[Array, " in_channels"]) -> tuple[Float[Array, " out_channels"], ...]:
        quantized_outputs = super().__call__(x)
        joint_lora_hidden = self.lora_down_weights @ x
        lora_hiddens = jnp.split(joint_lora_hidden, self._split_points([self.lora_rank] * len(self.output_dims)))
        lora_outputs = tuple(
            lora_up_weight @ lora_hidden
            for lora_up_weight, lora_hidden in zip(self.lora_up_weights, lora_hiddens, strict=True)
        )
        return tuple(
            quantized_output + self.lora_scale * lora_output
            for quantized_output, lora_output in zip(quantized_outputs, lora_outputs, strict=True)
        )


@dataclass
class QLoRALinearFactory(LinearFactoryBase[QLoRALinear]):
    group_size: int
    weight_quantization_mode: QuantizationMode
    activation_quantization_mode: QuantizationMode | None
    lora_rank: int
    lora_scale: float = 2.0
    activation_precision: DType = DEFAULT_PRECISION

    def __call__(self, input_dim: int, output_dims: tuple[int, ...], *, key: PRNGKeyArray) -> QLoRALinear:
        return QLoRALinear(
            input_dim=input_dim,
            output_dims=output_dims,
            group_size=self.group_size,
            weight_quantization_mode=self.weight_quantization_mode,
            activation_quantization_mode=self.activation_quantization_mode,
            lora_rank=self.lora_rank,
            lora_scale=self.lora_scale,
            activation_precision=self.activation_precision,
            key=key,
        )
