import math
from dataclasses import dataclass
from dataclasses import field as dataclass_field

import equinox as eqx
import jax
from einops import rearrange
from jax import numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from fartsovka.quantization import QuantizationMode, quantize

from .common import DEFAULT_PRECISION, DType

__all__ = ["LinearBase", "LinearFactoryBase", "Linear", "LinearFactory"]


class LinearBase(eqx.Module):
    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)

    def __call__(self, x: Float[Array, " in_channels"]) -> Float[Array, " out_channels"]:
        raise NotImplementedError


@dataclass
class LinearFactoryBase[LinearType: LinearBase]:
    def __call__(self, input_dim: int, output_dim: int, *, key: PRNGKeyArray) -> LinearType:
        raise NotImplementedError


class Linear(LinearBase):
    weights: Float[Array, "out_channels in_channels"]

    precision: DType = eqx.field(static=True, default=DEFAULT_PRECISION)

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        precision: DType = DEFAULT_PRECISION,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.precision = precision
        max_abs_value = 1 / math.sqrt(input_dim)
        self.weights = jax.random.uniform(
            key,
            (output_dim, input_dim),
            minval=-max_abs_value,
            maxval=max_abs_value,
            dtype=self.precision,
        )

    def __call__(self, x: Float[Array, " in_channels"]) -> Float[Array, " out_channels"]:
        return self.weights @ x


@dataclass
class LinearFactory(LinearFactoryBase[Linear]):
    precision: DType = dataclass_field(default=DEFAULT_PRECISION)

    def __call__(self, input_dim: int, output_dim: int, *, key: PRNGKeyArray) -> Linear:
        return Linear(input_dim, output_dim, precision=self.precision, key=key)


class GroupQuantizedLinear(LinearBase):
    num_groups: int = eqx.field(static=True)
    mode: QuantizationMode = eqx.field(static=True)
    accumulation_precision: DType = eqx.field(static=True)

    @property
    def group_size(self) -> int:
        return self.output_dim // self.num_groups

    weights: Float[Array, "out_channels in_channels"]
    scales: Float[Array, "out_channels groups"]

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        num_groups: int,
        mode: QuantizationMode,
        accumulation_precision: DType,
        key: PRNGKeyArray,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_groups = num_groups
        self.mode = mode
        self.accumulation_precision = accumulation_precision
        min_val, max_val = mode.range
        self.weights = jax.random.uniform(
            key,
            (output_dim, input_dim),
            minval=min_val,
            maxval=max_val,
            dtype=self.accumulation_precision,
        )
        self.scales = jnp.ones((output_dim, num_groups), dtype=self.accumulation_precision)

    def prepare_weights(self) -> Float[Array, "out_channels in_channels"]:
        quantized_weights = quantize(self.weights, self.mode)
        grouped_weights = rearrange(
            quantized_weights,
            "out_channels (groups group_channels) -> out_channels groups group_channels",
            group_size=self.group_size,
        )
        scales = rearrange(self.scales, "out_channels groups -> out_channels groups 1")
        scaled_grouped_weights = grouped_weights * scales
        result = rearrange(
            scaled_grouped_weights,
            "out_channels groups group_channels -> out_channels (groups group_channels)",
        )
        return result

    def __call__(self, x: Float[Array, " in_channels"]) -> Float[Array, " out_channels"]:
        return self.prepare_weights() @ x


@dataclass
class GroupQuantizedLinearFactory(LinearFactoryBase[GroupQuantizedLinear]):
    num_groups: int = dataclass_field(default=4)
    mode: QuantizationMode = dataclass_field(default=QuantizationMode.INT4)
    accumulation_precision: DType = dataclass_field(default=DEFAULT_PRECISION)

    def __call__(self, input_dim: int, output_dim: int, *, key: PRNGKeyArray) -> GroupQuantizedLinear:
        return GroupQuantizedLinear(
            input_dim=input_dim,
            output_dim=output_dim,
            num_groups=self.num_groups,
            mode=self.mode,
            accumulation_precision=self.accumulation_precision,
            key=key,
        )


class QLoRALinear(LinearBase):
    num_groups: int = eqx.field(static=True)
    mode: QuantizationMode = eqx.field(static=True)
    lora_precision: DType = eqx.field(static=True)
    lora_rank: int = eqx.field(static=True)
    lora_scale: float = eqx.field(static=True)

    quantized_linear: GroupQuantizedLinear
    down_weights: Float[Array, "lora_channels in_channels"]
    up_weights: Float[Array, "out_channels lora_channels"]

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        num_groups: int,
        mode: QuantizationMode,
        lora_rank: int,
        lora_scale: float,
        lora_precision: DType,
        key: PRNGKeyArray,
    ) -> None:
        linear_key, up_key, down_key = jax.random.split(key, 3)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_groups = num_groups
        self.mode = mode
        self.lora_rank = lora_rank
        self.lora_scale = lora_scale

        self.quantized_linear = GroupQuantizedLinear(
            input_dim=input_dim,
            output_dim=output_dim,
            num_groups=num_groups,
            mode=mode,
            accumulation_precision=lora_precision,
            key=linear_key,
        )

        max_down_abs_value = 1 / math.sqrt(input_dim)
        self.down_weights = jax.random.uniform(
            down_key,
            (output_dim, input_dim),
            minval=-max_down_abs_value,
            maxval=max_down_abs_value,
            dtype=self.lora_precision,
        )

        max_up_abs_value = 1 / math.sqrt(lora_rank)
        self.up_weights = jax.random.uniform(
            up_key,
            (output_dim, lora_rank),
            minval=-max_up_abs_value,
            maxval=max_up_abs_value,
            dtype=self.lora_precision,
        )

    def __call__(self, x: Float[Array, " in_channels"]) -> Float[Array, " out_channels"]:
        quantized_linear_output = self.quantized_linear(x)
        lora_hidden = self.down_weights @ x
        lora_output = self.up_weights @ lora_hidden
        return quantized_linear_output + self.lora_scale * lora_output


@dataclass
class QLoRALinearFactory(LinearFactoryBase[QLoRALinear]):
    num_groups: int
    mode: QuantizationMode
    lora_precision: DType
    lora_rank: int
    lora_scale: float = 2.0

    def __call__(self, input_dim: int, output_dim: int, *, key: PRNGKeyArray) -> QLoRALinear:
        return QLoRALinear(
            input_dim=input_dim,
            output_dim=output_dim,
            num_groups=self.num_groups,
            mode=self.mode,
            lora_rank=self.lora_rank,
            lora_scale=self.lora_scale,
            lora_precision=self.lora_precision,
            key=key,
        )
