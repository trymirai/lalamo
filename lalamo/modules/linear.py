import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Key

from lalamo.arrays import CompressedArray, FullPrecisionArray
from lalamo.arrays.base import ArrayForwardPassConfig
from lalamo.quantization import QuantizationMode, dynamically_quantize_activations

from .common import Initializer, LalamoModule, ShardingOrder, TensorSharding, sharded_field

__all__ = [
    "Linear",
    "LinearConfig",
]


@dataclass(frozen=True)
class LinearConfig:
    activation_quantization_mode: QuantizationMode | None = None

    def init(
        self,
        initializer: Initializer,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "Linear":
        total_out = sum(output_dims)
        scale = 1 / math.sqrt(input_dim)
        biases = initializer.zeros((total_out,), initializer.precision) if has_biases else None
        weights = FullPrecisionArray(initializer.normal(scale, (total_out, input_dim), initializer.precision))
        return Linear(
            config=self,
            weights=weights,
            biases=biases,
            output_dims=output_dims,
        )

    def init_mixture(
        self,
        initializer: Initializer,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "Linear":
        total_out = sum(output_dims)
        scale = 1 / math.sqrt(input_dim)
        weights = FullPrecisionArray(
            initializer.normal(scale, (mixture_size, total_out, input_dim), initializer.precision)
        )
        biases = initializer.zeros((mixture_size, total_out), initializer.precision) if has_biases else None
        return Linear(config=self, output_dims=output_dims, weights=weights, biases=biases)


class Linear(LalamoModule[LinearConfig]):
    output_dims: tuple[int, ...] = eqx.field(static=True)
    # sharding order specifies in which order do we attempt to shard
    sharding_order: ShardingOrder | None = eqx.field(static=True, default=None, kw_only=True)

    @property
    def input_dim(self) -> int:
        *_, _out, in_dim = self.weights.shape
        return in_dim

    @property
    def has_biases(self) -> bool:
        return self.biases is not None

    @property
    def num_outputs(self) -> int:
        return len(self.output_dims)

    @property
    def mixture_size(self) -> int | None:
        match self.weights.shape:
            case [num_components, _, _]:
                return num_components
            case _:
                return None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.weights.dtype

    @staticmethod
    def get_split_points(output_dims: Sequence[int]) -> tuple[int, ...]:
        result = []
        last_split_point = 0
        for dim in output_dims[:-1]:
            last_split_point += dim
            result.append(last_split_point)
        return tuple(result)


@dataclass(frozen=True)
class LinearConfigBase(ABC):
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
    def _init_general(
        self,
        initializer: Initializer,
        leading_dims: tuple[int, ...],
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "FullPrecisionLinear":
        std = 1 / math.sqrt(input_dim)
        weights = initializer.normal(std, (*leading_dims, sum(output_dims), input_dim), initializer.precision)
        biases = initializer.zeros((*leading_dims, sum(output_dims)), initializer.precision) if has_biases else None
        return FullPrecisionLinear(
            config=self,
            weights=weights,
            biases=biases,
            output_dims=output_dims,
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


class FullPrecisionLinear(LinearBase["FullPrecisionLinearConfig"]):
    weights: Float[Array, "*components total_out_channels in_channels"] = sharded_field(
        tensor_sharding=TensorSharding(
            axes=(-2, -1),
            axes_names=(ShardingOrder.OUTPUT, ShardingOrder.INPUT),
        ),
    )
    biases: Float[Array, "*components total_out_channels"] | None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.weights.dtype

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
    def __call__(
        self,
        inputs: Float[Array, " in_channels"],
        *,
        key: Key[Array, ""] | None,
        forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig(),  # noqa: B008
    ) -> tuple[Float[Array, " out_channels"], ...]:
        if self.config.activation_quantization_mode is not None:
            inputs = dynamically_quantize_activations(inputs, self.config.activation_quantization_mode)
        result = self.weights.dot(inputs, key=key, forward_pass_config=forward_pass_config)
        if self.biases is not None:
            result = result + self.biases
        return tuple(jnp.split(result, self.get_split_points(self.output_dims)))


@dataclass(frozen=True)
class QuantizedLinearConfigBase(LinearConfigBase):
    group_size: int
    weight_quantization_mode: QuantizationMode
    activation_quantization_mode: QuantizationMode | None


class QuantizedLinearBase[ConfigT: QuantizedLinearConfigBase](LinearBase[ConfigT]):
    biases: Float[Array, "*components total_out_channels"] | None

    @abstractmethod
    def _prepare_scaled_weights(self) -> Float[Array, "*components in_channels total_out_channels"]: ...

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
        min_val, max_val = self.weight_quantization_mode.range
        num_groups = input_dim // self.group_size
        total_out = sum(output_dims)
        std = (max_val - min_val + 2) / 2
        weights = initializer.normal(std, (*leading_dims, total_out, input_dim), initializer.precision)
        scales = initializer.zeros((*leading_dims, total_out, num_groups), initializer.precision)
        biases = initializer.zeros((*leading_dims, total_out), initializer.precision) if has_biases else None
        zero_points = initializer.zeros((*leading_dims, total_out, num_groups), initializer.precision)

        return GroupQuantizedLinear(
            config=self,
            output_dims=output_dims,
            weights=weights,
            scales=scales,
            zero_points=zero_points,
            biases=biases,
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


class GroupQuantizedLinearBase[ConfigT: GroupQuantizedLinearConfig](QuantizedLinearBase[ConfigT]):
    weights: Float[Array, "*components total_out_channels in_channels"] = sharded_field(
        tensor_sharding=TensorSharding(
            axes=(-2, -1),
            axes_names=(ShardingOrder.OUTPUT, ShardingOrder.INPUT),
        ),
    )
    scales: Float[Array, "*components total_out_channels groups"]
    zero_points: Float[Array, "*components total_out_channels groups"]
    biases: Float[Array, "*components total_out_channels"] | None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.scales.dtype

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
        return self.input_dim // self.config.group_size

    @property
    def int_weights(self) -> Int[Array, "*components in_channels out_channels"]:
        quantized = quantize_weights(self.weights, self.config.weight_quantization_mode)
        casted = quantized.astype(self.config.weight_quantization_mode.dtype)

        if self.config.weight_quantization_mode == QuantizationMode.UINT4:
            packed = jax_uint4_to_packed_uint8(casted)
        else:
            packed = casted

        return packed

    @property
    def int_zero_points(self) -> Int[Array, "*components groups out_channels"]:
        quantized = quantize_weights(self.zero_points, self.config.weight_quantization_mode)
        casted = quantized.astype(self.config.weight_quantization_mode.dtype)

        if self.config.weight_quantization_mode == QuantizationMode.UINT4:
            packed = jax_uint4_to_packed_uint8(casted)
        else:
            packed = casted

        return packed

    def _prepare_scaled_weights(self) -> Float[Array, "*components in_channels total_out_channels"]:
        quantized_weights = quantize_weights(self.weights, self.config.weight_quantization_mode)
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

    def export_weights(self) -> ParameterTree:
        result = dict(
            weights=self.int_weights,
            zero_points=self.int_zero_points,
            scales=self.scales,
        )
        if self.biases is not None:
            result["biases"] = self.biases
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights = require_mapping(weights)
        unpacked_weights = require_array(weights["weights"])
        unpacked_zero_points = require_array(weights["zero_points"])
        if self.config.weight_quantization_mode == QuantizationMode.UINT4:
            unpacked_weights = jax_uint8_to_unpacked_uint4(unpacked_weights)
            unpacked_zero_points = jax_uint8_to_unpacked_uint4(unpacked_zero_points)
        result = replace(
            self,
            weights=unpacked_weights.astype(self.weights.dtype),
            scales=require_array(weights["scales"]),
            zero_points=unpacked_zero_points.astype(self.zero_points.dtype),
            biases=require_array(weights["biases"]) if self.has_biases else None,
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
        min_val, max_val = self.weight_quantization_mode.range
        num_groups = input_dim // self.group_size
        total_out = sum(output_dims)
        std = (max_val - min_val + 2) / 2
        weights = initializer.normal(std, (*leading_dims, total_out, input_dim), initializer.precision)
        scales = initializer.zeros((*leading_dims, total_out, num_groups), initializer.precision)
        biases = initializer.zeros((*leading_dims, total_out), initializer.precision) if has_biases else None
        deq_biases = initializer.zeros((*leading_dims, total_out, num_groups), initializer.precision)

        return MLXQuantizedLinear(
            config=self,
            output_dims=output_dims,
            weights=weights,
            scales=scales,
            deq_biases=deq_biases,
            biases=biases,
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


class MLXQuantizedLinearBase[ConfigT: MLXQuantizedLinearConfig](QuantizedLinearBase[ConfigT]):
    weights: Float[Array, "*components total_out_channels in_channels"] = sharded_field(
        tensor_sharding=TensorSharding(
            axes=(-2, -1),
            axes_names=(ShardingOrder.OUTPUT, ShardingOrder.INPUT),
        ),
    )
    scales: Float[Array, "*components total_out_channels groups"]
    deq_biases: Float[Array, "*components total_out_channels groups"]
    biases: Float[Array, "*components total_out_channels"] | None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.scales.dtype

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
        return self.input_dim // self.config.group_size

    @property
    def int_weights(self) -> Int[Array, "*components in_channels out_channels"]:
        quantized = quantize_weights(self.weights, self.config.weight_quantization_mode)
        casted = quantized.astype(self.config.weight_quantization_mode.dtype)

        if self.config.weight_quantization_mode == QuantizationMode.UINT4:
            packed = jax_uint4_to_packed_uint8(casted)
        else:
            packed = casted

        return packed

    def _prepare_scaled_weights(self) -> Float[Array, "*components in_channels total_out_channels"]:
        quantized_weights = quantize_weights(self.weights, self.config.weight_quantization_mode)
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

    def export_weights(self) -> ParameterTree:
        result = dict(
            weights=self.int_weights,
            scales=self.scales,
            deq_biases=self.deq_biases,
        )
        if self.biases is not None:
            result["biases"] = self.biases
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights = require_mapping(weights)
        unpacked_weights = require_array(weights["weights"])
        if self.config.weight_quantization_mode == QuantizationMode.UINT4:
            unpacked_weights = jax_uint8_to_unpacked_uint4(unpacked_weights)
        result = replace(
            self,
            weights=unpacked_weights.astype(self.weights.dtype),
            scales=require_array(weights["scales"]),
            deq_biases=require_array(weights["deq_biases"]),
            biases=require_array(weights["biases"]) if self.has_biases else None,
        )
        return result


class MLXQuantizedLinear(MLXQuantizedLinearBase):
    pass


@dataclass(frozen=True)
class QLoRALinearConfig(GroupQuantizedLinearConfig):
    lora_rank: int
    lora_scale: float

    def _init_general(  # type: ignore[override]
        self,
        initializer: Initializer,
        leading_dims: tuple[int, ...],
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "QLoRALinear":
        group_quantized_linear = super()._init_general(initializer, leading_dims, input_dim, output_dims, has_biases)

        hidden_lora_rank = len(output_dims) * self.lora_rank
        lora_down_std = 1 / math.sqrt(input_dim)
        lora_down_weights = initializer.normal(
            lora_down_std,
            (*leading_dims, input_dim, hidden_lora_rank),
            self.activation_precision,
        )

        lora_up_std = 1 / math.sqrt(hidden_lora_rank)
        lora_up_weights = tuple(
            initializer.normal(
                lora_up_std,
                (*leading_dims, self.lora_rank, output_dim),
                self.activation_precision,
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

    def init(  # type: ignore[override]
        self,
        initializer: Initializer,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "QLoRALinear":
        return self._init_general(initializer, (), input_dim, output_dims, has_biases)

    def init_mixture(  # type: ignore[override]
        self,
        initializer: Initializer,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "QLoRALinear":
        return self._init_general(initializer, (mixture_size,), input_dim, output_dims, has_biases)


class QLoRALinear(GroupQuantizedLinearBase):
    lora_down_weights: Float[Array, "*components in_channels total_lora_channels"] = sharded_field(
        tensor_sharding=TensorSharding(
            axes=(-2, -1),
            axes_names=(ShardingOrder.INPUT, ShardingOrder.OUTPUT),
        ),
    )
    lora_up_weights: tuple[Float[Array, "*components lora_channels out_channels"], ...]

    def _split_biases(self) -> tuple[Float[Array, "*components out_channels"] | None, ...]:
        if self.biases is not None:
            return tuple(jnp.split(self.biases, self.get_split_points(self.output_dims)))
        return (None,) * len(self.output_dims)

    @eqx.filter_jit
    def __call__(self, inputs: Float[Array, " in_channels"]) -> tuple[Float[Array, " out_channels"], ...]:
        if self.mixture_size is not None:
            raise ValueError(
                "Mixtures of linear layers cannot be called directly."
                "They are intended to be used with methods eqx.filter_vmap or lax.scan instead.",
            )
        joint_q_out = self._apply_weights(inputs)
        q_outs = jnp.split(joint_q_out, self.get_split_points(self.output_dims))

        joint_lora_hidden = inputs @ self.lora_down_weights
        lora_hiddens = jnp.split(joint_lora_hidden, self.get_split_points([self.config.lora_rank] * self.num_outputs))
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


LinearConfig = FullPrecisionLinearConfig | GroupQuantizedLinearConfig | MLXQuantizedLinearConfig | QLoRALinearConfig


register_config_union(LinearConfig)
