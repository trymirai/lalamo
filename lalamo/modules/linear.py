import math
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.common import RegistryABC
from lalamo.quantization import QuantizationMode, dynamically_quantize_activations, quantize_weights
from lalamo.utils import jax_uint4_to_packed_uint8

from .common import (
    Initializer,
    LalamoConfig,
    LalamoModule,
)

__all__ = [
    "FullPrecisionLinear",
    "FullPrecisionLinearConfig",
    "GroupQuantizedLinear",
    "GroupQuantizedLinearConfig",
    "LinearBase",
    "LinearConfigBase",
    "MLXQuantizedLinear",
    "MLXQuantizedLinearConfig",
]


class LinearBase(LalamoModule):
    output_dims: tuple[int, ...] = eqx.field(static=True)

    @property
    @abstractmethod
    def mixture_size(self) -> int | None: ...

    @property
    @abstractmethod
    def input_dim(self) -> int: ...

    @property
    def num_outputs(self) -> int:
        return len(self.output_dims)

    @property
    @abstractmethod
    def has_biases(self) -> bool: ...

    @abstractmethod
    def __call__(
        self,
        inputs: Float[Array, " in_channels"],
    ) -> tuple[Float[Array, " out_channels"], ...]: ...

    @staticmethod
    def get_split_points(output_dims: Sequence[int]) -> tuple[int, ...]:
        result = []
        last_split_point = 0
        for dim in output_dims[:-1]:
            last_split_point += dim
            result.append(last_split_point)
        return tuple(result)


@dataclass(frozen=True)
class LinearConfigBase(LalamoConfig, RegistryABC):
    @abstractmethod
    def init(
        self,
        initializer: Initializer,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> LinearBase: ...

    @abstractmethod
    def init_mixture(
        self,
        initializer: Initializer,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> LinearBase: ...


@dataclass(frozen=True)
class FullPrecisionLinearConfig(LinearConfigBase):
    precision: DTypeLike

    def _init_general(
        self,
        initializer: Initializer,
        leading_dims: tuple[int, ...],
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "FullPrecisionLinear":
        std = 1 / math.sqrt(input_dim)
        weights = initializer.normal(std, (*leading_dims, sum(output_dims), input_dim), self.precision)
        if has_biases:
            biases = initializer.zeros((*leading_dims, sum(output_dims)), self.precision)
        else:
            biases = None

        return FullPrecisionLinear(
            weights=weights,
            biases=biases,
            output_dims=output_dims,
            activation_precision=self.precision,
        )

    def init(
        self,
        initializer: Initializer,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "FullPrecisionLinear":
        return self._init_general(initializer, (), input_dim, output_dims, has_biases)

    def init_mixture(
        self,
        initializer: Initializer,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "FullPrecisionLinear":
        return self._init_general(initializer, (mixture_size,), input_dim, output_dims, has_biases)


class FullPrecisionLinear(LinearBase):
    weights: Float[Array, "*components total_out_channels in_channels"]
    biases: Float[Array, "*components total_out_channels"] | None

    activation_precision: DTypeLike = eqx.field(static=True)

    @property
    def mixture_size(self) -> int | None:
        match self.weights.shape:
            case [num_components, _, _]:
                return num_components
            case _:
                return None

    @property
    def input_dim(self) -> int:
        *_, _, input_dim = self.weights.shape
        return input_dim

    @property
    def has_biases(self) -> bool:
        return self.biases is not None

    @eqx.filter_jit
    def __call__(self, inputs: Float[Array, " in_channels"]) -> tuple[Float[Array, " out_channels"], ...]:
        if self.mixture_size is not None:
            raise ValueError(
                "Mixtures of linear layers cannot be called directly."
                "They are intended to be used with methods eqx.filter_vmap or lax.scan instead.",
            )
        result = self.weights @ inputs
        if self.biases is not None:
            result = result + self.biases
        return tuple(jnp.split(result, self.get_split_points(self.output_dims)))


@dataclass(frozen=True)
class QuantizedLinearConfigBase(LinearConfigBase):
    group_size: int
    weight_quantization_mode: QuantizationMode
    activation_quantization_mode: QuantizationMode | None
    activation_precision: DTypeLike


class QuantizedLinearBase(LinearBase):
    biases: Float[Array, "*components total_out_channels"] | None

    activation_precision: DTypeLike = eqx.field(static=True)
    activation_quantization_mode: QuantizationMode | None = eqx.field(static=True)

    @abstractmethod
    def _prepare_scaled_weights(self) -> Float[Array, "*components in_channels total_out_channels"]: ...

    def _apply_weights(self, inputs: Float[Array, " in_channels"]) -> Float[Array, " total_out_channels"]:
        if self.activation_quantization_mode is not None:
            inputs = dynamically_quantize_activations(inputs, self.activation_quantization_mode)
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
        return tuple(jnp.split(result, self.get_split_points(self.output_dims)))


@dataclass(frozen=True)
class GroupQuantizedLinearConfig(QuantizedLinearConfigBase):
    def _init_general(
        self,
        initializer: Initializer,
        leading_dims: tuple[int, ...],
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "GroupQuantizedLinear":
        num_groups = input_dim // self.group_size
        weights = initializer.zeros((*leading_dims, sum(output_dims), input_dim), self.activation_precision)
        scales = initializer.ones((*leading_dims, sum(output_dims), num_groups), self.activation_precision)
        zero_points = initializer.zeros((*leading_dims, sum(output_dims), num_groups), self.activation_precision)

        if has_biases:
            biases = initializer.zeros((*leading_dims, sum(output_dims)), self.activation_precision)
        else:
            biases = None

        return GroupQuantizedLinear(
            weights=weights,
            scales=scales,
            zero_points=zero_points,
            biases=biases,
            output_dims=output_dims,
            activation_precision=self.activation_precision,
            activation_quantization_mode=self.activation_quantization_mode,
            group_size=self.group_size,
            weight_quantization_mode=self.weight_quantization_mode,
        )

    def init(
        self,
        initializer: Initializer,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "GroupQuantizedLinear":
        return self._init_general(initializer, (), input_dim, output_dims, has_biases)

    def init_mixture(
        self,
        initializer: Initializer,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "GroupQuantizedLinear":
        return self._init_general(initializer, (mixture_size,), input_dim, output_dims, has_biases)


class GroupQuantizedLinearBase(QuantizedLinearBase):
    weights: Float[Array, "*components total_out_channels in_channels"]
    scales: Float[Array, "*components total_out_channels groups"]
    zero_points: Float[Array, "*components total_out_channels groups"]
    biases: Float[Array, "*components total_out_channels"] | None

    group_size: int = eqx.field(static=True)
    weight_quantization_mode: QuantizationMode = eqx.field(static=True)

    @property
    def mixture_size(self) -> int | None:
        match self.weights.shape:
            case [num_components, _, _]:
                return num_components
            case _:
                return None

    @property
    def input_dim(self) -> int:
        *_, _, input_dim = self.weights.shape
        return input_dim

    @property
    def has_biases(self) -> bool:
        return self.biases is not None

    @property
    def num_groups(self) -> int:
        return self.input_dim // self.group_size

    @property
    def int_weights(self) -> Int[Array, "*components in_channels out_channels"]:
        quantized = quantize_weights(self.weights, self.weight_quantization_mode)
        casted = quantized.astype(self.weight_quantization_mode.dtype)

        if self.weight_quantization_mode == QuantizationMode.UINT4:
            packed = jax_uint4_to_packed_uint8(casted)
        else:
            packed = casted

        return packed

    @property
    def int_zero_points(self) -> Int[Array, "*components groups out_channels"]:
        quantized = quantize_weights(self.zero_points, self.weight_quantization_mode)
        casted = quantized.astype(self.weight_quantization_mode.dtype)

        if self.weight_quantization_mode == QuantizationMode.UINT4:
            packed = jax_uint4_to_packed_uint8(casted)
        else:
            packed = casted

        return packed

    def _prepare_scaled_weights(self) -> Float[Array, "*components in_channels total_out_channels"]:
        quantized_weights = quantize_weights(self.weights, self.weight_quantization_mode)
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


class GroupQuantizedLinear(GroupQuantizedLinearBase):
    pass


@dataclass(frozen=True)
class MLXQuantizedLinearConfig(QuantizedLinearConfigBase):
    def _init_general(
        self,
        initializer: Initializer,
        leading_dims: tuple[int, ...],
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "MLXQuantizedLinear":
        num_groups = input_dim // self.group_size
        weights = initializer.zeros((*leading_dims, sum(output_dims), input_dim), self.activation_precision)
        scales = initializer.ones((*leading_dims, sum(output_dims), num_groups), self.activation_precision)
        deq_biases = initializer.zeros((*leading_dims, sum(output_dims), num_groups), self.activation_precision)

        if has_biases:
            biases = initializer.zeros((*leading_dims, sum(output_dims)), self.activation_precision)
        else:
            biases = None

        return MLXQuantizedLinear(
            weights=weights,
            scales=scales,
            deq_biases=deq_biases,
            biases=biases,
            output_dims=output_dims,
            activation_precision=self.activation_precision,
            activation_quantization_mode=self.activation_quantization_mode,
            group_size=self.group_size,
            weight_quantization_mode=self.weight_quantization_mode,
        )

    def init(
        self,
        initializer: Initializer,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "MLXQuantizedLinear":
        return self._init_general(initializer, (), input_dim, output_dims, has_biases)

    def init_mixture(
        self,
        initializer: Initializer,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "MLXQuantizedLinear":
        return self._init_general(initializer, (mixture_size,), input_dim, output_dims, has_biases)


class MLXQuantizedLinearBase(QuantizedLinearBase):
    weights: Float[Array, "*components total_out_channels in_channels"]
    scales: Float[Array, "*components total_out_channels groups"]
    deq_biases: Float[Array, "*components total_out_channels groups"]
    biases: Float[Array, "*components total_out_channels"] | None

    group_size: int = eqx.field(static=True)
    weight_quantization_mode: QuantizationMode = eqx.field(static=True)

    @property
    def mixture_size(self) -> int | None:
        match self.weights.shape:
            case [num_components, _, _]:
                return num_components
            case _:
                return None

    @property
    def input_dim(self) -> int:
        *_, _, input_dim = self.weights.shape
        return input_dim

    @property
    def has_biases(self) -> bool:
        return self.biases is not None

    @property
    def num_groups(self) -> int:
        return self.input_dim // self.group_size

    @property
    def int_weights(self) -> Int[Array, "*components in_channels out_channels"]:
        quantized = quantize_weights(self.weights, self.weight_quantization_mode)
        casted = quantized.astype(self.weight_quantization_mode.dtype)

        if self.weight_quantization_mode == QuantizationMode.UINT4:
            packed = jax_uint4_to_packed_uint8(casted)
        else:
            packed = casted

        return packed

    def _prepare_scaled_weights(self) -> Float[Array, "*components in_channels total_out_channels"]:
        quantized_weights = quantize_weights(self.weights, self.weight_quantization_mode)
        grouped_weights = rearrange(
            quantized_weights,
            "... total_out_channels (groups group_channels) -> ... total_out_channels groups group_channels",
            groups=self.num_groups,
        )

        scales = rearrange(self.scales, "... total_out_channels groups -> ... total_out_channels groups 1")
        deq_biases = rearrange(self.deq_biases, "... total_out_channels groups -> ... total_out_channels groups 1")

        scaled_grouped_weights = grouped_weights * scales + deq_biases
        result = rearrange(
            scaled_grouped_weights,
            "... total_out_channels groups group_channels -> ... total_out_channels (groups group_channels)",
        )
        return result


class MLXQuantizedLinear(MLXQuantizedLinearBase):
    pass
