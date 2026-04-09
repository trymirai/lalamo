import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Self, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float, Int, Key

from lalamo.common import ParameterTree, dummy_array, require_array, require_mapping, require_tree
from lalamo.quantization import QuantizationMode, dynamically_quantize_activations, quantize_weights
from lalamo.utils import jax_uint4_to_packed_uint8, jax_uint8_to_unpacked_uint4

from .common import (
    LalamoModule,
    ParameterNorm,
    ShardingOrder,
    TensorSharding,
    config_converter,
    field,
    register_config_union,
)

__all__ = [
    "FullPrecisionLinear",
    "FullPrecisionLinearConfig",
    "GroupQuantizedLinear",
    "GroupQuantizedLinearConfig",
    "LinearBase",
    "LinearConfig",
    "QLoRALinear",
    "QLoRALinearConfig",
]


class LinearBase[ConfigT: LinearConfigBase](LalamoModule[ConfigT]):
    output_dims: tuple[int, ...] = eqx.field(static=True)
    # sharding order specifies in which order do we attempt to shard
    sharding_order: ShardingOrder | None = eqx.field(static=True, default=None, kw_only=True)

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

    def __post_init__(self) -> None:
        assert isinstance(self.output_dims, tuple)

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
    def random_init(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        key: Key[Array, ""],
    ) -> LinearBase: ...

    @abstractmethod
    def random_init_mixture(
        self,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        key: Key[Array, ""],
    ) -> LinearBase: ...

    @abstractmethod
    def empty(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> LinearBase: ...

    @abstractmethod
    def empty_mixture(
        self,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> LinearBase: ...


@dataclass(frozen=True)
class FullPrecisionLinearConfig(LinearConfigBase):
    precision: DTypeLike

    def random_init(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        key: Key[Array, ""],
    ) -> "FullPrecisionLinear":
        scale = 1 / math.sqrt(input_dim)
        weights = jax.random.uniform(
            key,
            (sum(output_dims), input_dim),
            minval=-scale,
            maxval=scale,
            dtype=self.precision,
        )
        if has_biases:
            biases = jnp.zeros((sum(output_dims),), dtype=self.precision)
        else:
            biases = None

        return FullPrecisionLinear(
            config=self,
            output_dims=output_dims,
            weights=weights,
            biases=biases,
        )

    def random_init_mixture(
        self,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        key: Key[Array, ""],
    ) -> LinearBase:
        subkeys = jax.random.split(key, mixture_size)
        return eqx.filter_vmap(lambda key: self.random_init(input_dim, output_dims, has_biases, key=key))(subkeys)

    def _empty_general(
        self,
        leading_dims: tuple[int, ...],
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "FullPrecisionLinear":
        weights = dummy_array(
            (*leading_dims, sum(output_dims), input_dim),
            dtype=self.precision,
        )
        if has_biases:
            biases = dummy_array((*leading_dims, sum(output_dims)), dtype=self.precision)
        else:
            biases = None

        return FullPrecisionLinear(
            config=self,
            output_dims=output_dims,
            weights=weights,
            biases=biases,
        )

    def empty(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "FullPrecisionLinear":
        return self._empty_general((), input_dim, output_dims, has_biases)

    def empty_mixture(
        self,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "FullPrecisionLinear":
        return self._empty_general((mixture_size,), input_dim, output_dims, has_biases)


class FullPrecisionLinear(LinearBase[FullPrecisionLinearConfig]):
    weights: Float[Array, "*components total_out_channels in_channels"] = field(
        tensor_sharding=TensorSharding(
            axes=(-2, -1),
            axes_names=(ShardingOrder.OUTPUT, ShardingOrder.INPUT),
        ),
    )
    biases: Float[Array, "*components total_out_channels"] | None = field()

    @property
    def mixture_size(self) -> int | None:
        match self.weights.shape:
            case [num_components, _, _]:
                return num_components
            case _:
                return None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def input_dim(self) -> int:
        *_, _, input_dim = self.weights.shape
        return input_dim

    @property
    def has_biases(self) -> bool:
        return self.biases is not None

    def __post_init__(self) -> None:
        if self.weights.dtype != self.config.precision:
            raise ValueError(
                f"Weight dtype ({self.weights.dtype}) is not equal to specified precision ({self.config.precision}).",
            )
        *w_num_components, w_output_dim, _ = self.weights.shape
        if w_output_dim != sum(self.output_dims):
            raise ValueError(
                f"Number of output channels in weights ({w_output_dim}) is not"
                f" equal to sum of output dims ({sum(self.output_dims)}).",
            )
        if self.biases is None:
            return
        *b_num_components, b_output_dim = self.biases.shape
        if w_output_dim != b_output_dim:
            raise ValueError(
                f"Number of output channels in weights ({w_output_dim}) is not"
                f" equal to number of output channels in biases ({b_output_dim}).",
            )
        if self.biases.dtype != self.config.precision:
            raise ValueError(
                f"Bias dtype ({self.biases.dtype}) is not equal to specified precision ({self.config.precision}).",
            )
        if b_num_components != w_num_components:
            raise ValueError(
                f"Number of mixture components in weights ({w_num_components}) is not"
                f" equal to number of mixture components in biases ({b_num_components}).",
            )

    @eqx.filter_jit
    def __call__(self, inputs: Float[Array, " in_channels"]) -> tuple[Float[Array, " out_channels"], ...]:
        if self.mixture_size is not None:
            raise ValueError(
                "Mixtures of linear layers cannot be called directly."
                "They are intended to be used with methods eqx.filter_vmap or lax.scan instead.",
            )
        result = jnp.dot(self.weights, inputs)
        if self.biases is not None:
            result = result + self.biases
        return tuple(jnp.split(result, self.get_split_points(self.output_dims)))

    def export_weights(self) -> ParameterTree:
        result = dict(weights=self.weights)
        if self.biases is not None:
            result["biases"] = self.biases
        return result

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        params = _linear_params(weights)
        return replace(
            self,
            weights=require_array(params["weights"]),
            biases=require_array(params["biases"]) if self.has_biases else None,
        )


@dataclass(frozen=True)
class QuantizedLinearConfigBase(LinearConfigBase):
    group_size: int
    weight_quantization_mode: QuantizationMode
    activation_quantization_mode: QuantizationMode | None
    activation_precision: DTypeLike

    @property
    def quantization(self) -> QuantizationMode:
        return self.weight_quantization_mode

    def group_count(self, input_dim: int) -> int:
        assert input_dim % self.group_size == 0
        return input_dim // self.group_size


class QuantizedLinearBase[ConfigT: QuantizedLinearConfigBase](LinearBase[ConfigT]):
    biases: Float[Array, "*components total_out_channels"] | None

    @abstractmethod
    def _dequantize_weights(self) -> Float[Array, "*components in_channels total_out_channels"]: ...

    def _apply_weights(self, inputs: Float[Array, " in_channels"]) -> Float[Array, " total_out_channels"]:
        if self.config.activation_quantization_mode is not None:
            inputs = dynamically_quantize_activations(inputs, self.config.activation_quantization_mode)
        return self._dequantize_weights() @ inputs

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
    def random_init(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        key: Key[Array, ""],
    ) -> LinearBase:
        min_val, max_val = self.weight_quantization_mode.range
        weights = jax.random.uniform(
            key,
            (sum(output_dims), input_dim),
            minval=min_val - 1,
            maxval=max_val + 1,
            dtype=self.activation_precision,
        )
        num_groups = self.group_count(input_dim)
        scale = 1 / ((max_val - min_val) / 2 * math.sqrt(input_dim))
        scales = scale * jnp.ones((sum(output_dims), num_groups), dtype=self.activation_precision)

        if has_biases:
            biases = jnp.zeros((sum(output_dims),), dtype=self.activation_precision)
        else:
            biases = None

        zero_point = min_val + 2 ** (self.weight_quantization_mode.bits - 1)
        zero_points = zero_point * jnp.ones((sum(output_dims), num_groups), dtype=self.activation_precision)

        return GroupQuantizedLinear(
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
        key: Key[Array, ""],
    ) -> LinearBase:
        subkeys = jax.random.split(key, mixture_size)
        return eqx.filter_vmap(lambda key: self.random_init(input_dim, output_dims, has_biases, key=key))(subkeys)

    def _empty_general(
        self,
        leading_dims: tuple[int, ...],
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> LinearBase:
        weights = dummy_array(
            (*leading_dims, sum(output_dims), input_dim),
            dtype=self.activation_precision,
        )
        num_groups = self.group_count(input_dim)
        scales = dummy_array((*leading_dims, sum(output_dims), num_groups), dtype=self.activation_precision)

        if has_biases:
            biases = dummy_array((*leading_dims, sum(output_dims)), dtype=self.activation_precision)
        else:
            biases = None
        zero_points = dummy_array((*leading_dims, sum(output_dims), num_groups), dtype=self.activation_precision)

        return GroupQuantizedLinear(
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
    ) -> LinearBase:
        return self._empty_general((), input_dim, output_dims, has_biases)

    def empty_mixture(
        self,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> LinearBase:
        return self._empty_general((mixture_size,), input_dim, output_dims, has_biases)


class GroupQuantizedLinearBase[ConfigT: GroupQuantizedLinearConfig](QuantizedLinearBase[ConfigT]):
    weights: Float[Array, "*components total_out_channels in_channels"] = field(
        quantized=True,
        tensor_sharding=TensorSharding(
            axes=(-2, -1),
            axes_names=(ShardingOrder.OUTPUT, ShardingOrder.INPUT),
        ),
    )
    scales: Float[Array, "*components total_out_channels groups"] = field(norm=ParameterNorm.L_INF)
    zero_points: Float[Array, "*components total_out_channels groups"] = field(norm=ParameterNorm.L_INF)
    biases: Float[Array, "*components total_out_channels"] | None = field()

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

    def _quantized_weights_for_export(self) -> Int[Array, "*components in_channels out_channels"]:
        quantized = quantize_weights(self.weights, self.config.weight_quantization_mode)
        casted = quantized.astype(self.config.weight_quantization_mode.dtype)

        if self.config.weight_quantization_mode == QuantizationMode.UINT4:
            packed = jax_uint4_to_packed_uint8(casted)
        else:
            packed = casted

        return packed

    def _quantized_zero_points_for_export(self) -> Int[Array, "*components groups out_channels"]:
        quantized = quantize_weights(self.zero_points, self.config.weight_quantization_mode)
        casted = quantized.astype(self.config.weight_quantization_mode.dtype)

        if self.config.weight_quantization_mode == QuantizationMode.UINT4:
            packed = jax_uint4_to_packed_uint8(casted)
        else:
            packed = casted

        return packed

    def __post_init__(self) -> None:
        group_count = self.config.group_count(self.input_dim)
        output_dim = sum(self.output_dims)
        component_shape = self.weights.shape[:-2]
        if self.weights.dtype != self.config.activation_precision:
            raise ValueError(
                f"Weight dtype ({self.weights.dtype}) is not equal to specified activation precision"
                f" ({self.config.activation_precision})."
                " Quantized layers require parameter dtypes to be equal to the activation precision.",
            )
        if self.weights.shape[-2] != output_dim:
            raise ValueError(
                f"Number of output channels in weights ({self.weights.shape[-2]}) is not"
                f" equal to sum of output dims ({output_dim}).",
            )

        for name, value in (("scales", self.scales), ("zero points", self.zero_points)):
            if value.dtype != self.config.activation_precision:
                raise ValueError(
                    f"{name.title()} dtype ({value.dtype}) is not equal to specified activation precision"
                    f" ({self.config.activation_precision})."
                    " Quantized layers require parameter dtypes to be equal to the activation precision.",
                )
            if value.shape[:-2] != component_shape:
                raise ValueError(
                    f"Number of mixture components in {name} ({value.shape[:-2]}) is not"
                    f" equal to number of mixture components in weights ({component_shape}).",
                )
            if value.shape[-2] != output_dim:
                raise ValueError(
                    f"Number of output channels in {name} ({value.shape[-2]}) is not"
                    f" equal to number of output channels in weights ({output_dim}).",
                )
            if value.shape[-1] != group_count:
                raise ValueError(
                    f"Number of groups in {name} ({value.shape[-1]}) is incompatible with"
                    f" the specified group size ({self.config.group_size}).",
                )

        if self.biases is not None:
            if self.biases.dtype != self.config.activation_precision:
                raise ValueError(
                    f"Bias dtype ({self.biases.dtype}) is not equal to specified activation precision"
                    f" ({self.config.activation_precision})."
                    " Quantized layers require parameter dtypes to be equal to the activation precision.",
                )
            if self.biases.shape[:-1] != component_shape:
                raise ValueError(
                    f"Number of mixture components in biases ({self.biases.shape[:-1]}) is not"
                    f" equal to number of mixture components in weights ({component_shape}).",
                )
            if self.biases.shape[-1] != output_dim:
                raise ValueError(
                    f"Number of output channels in biases ({self.biases.shape[-1]}) is not"
                    f" equal to number of output channels in weights ({output_dim}).",
                )

    def _dequantize_weights(self) -> Float[Array, "*components in_channels total_out_channels"]:
        group_count = self.config.group_count(self.input_dim)
        quantized_weights = quantize_weights(self.weights, self.config.weight_quantization_mode)
        grouped_weights = rearrange(
            quantized_weights,
            "... total_out_channels (groups group_channels) -> ... total_out_channels groups group_channels",
            groups=group_count,
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
            weights=self._quantized_weights_for_export(),
            zero_points=self._quantized_zero_points_for_export(),
            scales=self.scales,
        )
        if self.biases is not None:
            result["biases"] = self.biases
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        params = _linear_params(weights)
        unpacked_weights = require_array(params["weights"])
        unpacked_zero_points = require_array(params["zero_points"])
        if self.config.weight_quantization_mode == QuantizationMode.UINT4:
            unpacked_weights = jax_uint8_to_unpacked_uint4(unpacked_weights)
            unpacked_zero_points = jax_uint8_to_unpacked_uint4(unpacked_zero_points)
        return replace(
            self,
            weights=unpacked_weights.astype(self.weights.dtype),
            scales=require_array(params["scales"]),
            zero_points=unpacked_zero_points.astype(self.zero_points.dtype),
            biases=require_array(params["biases"]) if self.has_biases else None,
        )


class GroupQuantizedLinear(GroupQuantizedLinearBase[GroupQuantizedLinearConfig]):
    pass


@dataclass(frozen=True)
class MLXQuantizedLinearConfig(QuantizedLinearConfigBase):
    def random_init(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        key: Key[Array, ""],
    ) -> LinearBase:
        min_val, max_val = self.weight_quantization_mode.range
        weights = jax.random.uniform(
            key,
            (sum(output_dims), input_dim),
            minval=min_val - 1,
            maxval=max_val + 1,
            dtype=self.activation_precision,
        )
        num_groups = self.group_count(input_dim)
        scale = 1 / ((max_val - min_val) / 2 * math.sqrt(input_dim))
        scales = scale * jnp.ones((sum(output_dims), num_groups), dtype=self.activation_precision)

        if has_biases:
            biases = jnp.zeros((sum(output_dims),), dtype=self.activation_precision)
        else:
            biases = None

        zero_point = min_val + 2 ** (self.weight_quantization_mode.bits - 1)
        offsets = -zero_point * scales

        return MLXQuantizedLinear(
            config=self,
            output_dims=output_dims,
            weights=weights,
            scales=scales,
            offsets=offsets,
            biases=biases,
        )

    def random_init_mixture(
        self,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        key: Key[Array, ""],
    ) -> LinearBase:
        subkeys = jax.random.split(key, mixture_size)
        return eqx.filter_vmap(lambda key: self.random_init(input_dim, output_dims, has_biases, key=key))(subkeys)

    def _empty_general(
        self,
        leading_dims: tuple[int, ...],
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> LinearBase:
        weights = dummy_array(
            (*leading_dims, sum(output_dims), input_dim),
            dtype=self.activation_precision,
        )
        num_groups = self.group_count(input_dim)
        scales = dummy_array((*leading_dims, sum(output_dims), num_groups), dtype=self.activation_precision)

        if has_biases:
            biases = dummy_array((*leading_dims, sum(output_dims)), dtype=self.activation_precision)
        else:
            biases = None
        offsets = dummy_array((*leading_dims, sum(output_dims), num_groups), dtype=self.activation_precision)

        return MLXQuantizedLinear(
            config=self,
            output_dims=output_dims,
            weights=weights,
            scales=scales,
            offsets=offsets,
            biases=biases,
        )

    def empty(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> LinearBase:
        return self._empty_general((), input_dim, output_dims, has_biases)

    def empty_mixture(
        self,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> LinearBase:
        return self._empty_general((mixture_size,), input_dim, output_dims, has_biases)


class MLXQuantizedLinearBase[ConfigT: MLXQuantizedLinearConfig](QuantizedLinearBase[ConfigT]):
    weights: Float[Array, "*components total_out_channels in_channels"] = field(
        quantized=True,
        tensor_sharding=TensorSharding(
            axes=(-2, -1),
            axes_names=(ShardingOrder.OUTPUT, ShardingOrder.INPUT),
        ),
    )
    scales: Float[Array, "*components total_out_channels groups"] = field(norm=ParameterNorm.L_INF)
    offsets: Float[Array, "*components total_out_channels groups"] = field(norm=ParameterNorm.L_INF)
    biases: Float[Array, "*components total_out_channels"] | None = field()

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

    def _quantized_weights_for_export(self) -> Int[Array, "*components in_channels out_channels"]:
        quantized = quantize_weights(self.weights, self.config.weight_quantization_mode)
        casted = quantized.astype(self.config.weight_quantization_mode.dtype)

        if self.config.weight_quantization_mode == QuantizationMode.UINT4:
            packed = jax_uint4_to_packed_uint8(casted)
        else:
            packed = casted

        return packed

    def __post_init__(self) -> None:
        group_count = self.config.group_count(self.input_dim)
        output_dim = sum(self.output_dims)
        component_shape = self.weights.shape[:-2]
        if self.weights.dtype != self.config.activation_precision:
            raise ValueError(
                f"Weight dtype ({self.weights.dtype}) is not equal to specified activation precision"
                f" ({self.config.activation_precision})."
                " Quantized layers require parameter dtypes to be equal to the activation precision.",
            )
        if self.weights.shape[-2] != output_dim:
            raise ValueError(
                f"Number of output channels in weights ({self.weights.shape[-2]}) is not"
                f" equal to sum of output dims ({output_dim}).",
            )

        for name, value in (("scales", self.scales), ("offsets", self.offsets)):
            if value.dtype != self.config.activation_precision:
                raise ValueError(
                    f"{name.title()} dtype ({value.dtype}) is not equal to specified activation precision"
                    f" ({self.config.activation_precision})."
                    " Quantized layers require parameter dtypes to be equal to the activation precision.",
                )
            if value.shape[:-2] != component_shape:
                raise ValueError(
                    f"Number of mixture components in {name} ({value.shape[:-2]}) is not"
                    f" equal to number of mixture components in weights ({component_shape}).",
                )
            if value.shape[-2] != output_dim:
                raise ValueError(
                    f"Number of output channels in {name} ({value.shape[-2]}) is not"
                    f" equal to number of output channels in weights ({output_dim}).",
                )
            if value.shape[-1] != group_count:
                raise ValueError(
                    f"Number of groups in {name} ({value.shape[-1]}) is incompatible with"
                    f" the specified group size ({self.config.group_size}).",
                )

        if self.biases is not None:
            if self.biases.dtype != self.config.activation_precision:
                raise ValueError(
                    f"Bias dtype ({self.biases.dtype}) is not equal to specified activation precision"
                    f" ({self.config.activation_precision})."
                    " Quantized layers require parameter dtypes to be equal to the activation precision.",
                )
            if self.biases.shape[:-1] != component_shape:
                raise ValueError(
                    f"Number of mixture components in biases ({self.biases.shape[:-1]}) is not"
                    f" equal to number of mixture components in weights ({component_shape}).",
                )
            if self.biases.shape[-1] != output_dim:
                raise ValueError(
                    f"Number of output channels in biases ({self.biases.shape[-1]}) is not"
                    f" equal to number of output channels in weights ({output_dim}).",
                )

    def _dequantize_weights(self) -> Float[Array, "*components in_channels total_out_channels"]:
        group_count = self.config.group_count(self.input_dim)
        quantized_weights = quantize_weights(self.weights, self.config.weight_quantization_mode)
        grouped_weights = rearrange(
            quantized_weights,
            "... total_out_channels (groups group_channels) -> ... total_out_channels groups group_channels",
            groups=group_count,
        )

        scales = rearrange(self.scales, "... total_out_channels groups -> ... total_out_channels groups 1")
        offsets = rearrange(self.offsets, "... total_out_channels groups -> ... total_out_channels groups 1")

        scaled_grouped_weights = grouped_weights * scales + offsets
        result = rearrange(
            scaled_grouped_weights,
            "... total_out_channels groups group_channels -> ... total_out_channels (groups group_channels)",
        )
        return result

    def export_weights(self) -> ParameterTree:
        result = {
            "weights": self._quantized_weights_for_export(),
            "scales": self.scales,
            "deq_biases": self.offsets,
        }
        if self.biases is not None:
            result["biases"] = self.biases
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        params = _linear_params(weights)
        unpacked_weights = require_array(params["weights"])
        if self.config.weight_quantization_mode == QuantizationMode.UINT4:
            unpacked_weights = jax_uint8_to_unpacked_uint4(unpacked_weights)
        return replace(
            self,
            weights=unpacked_weights.astype(self.weights.dtype),
            scales=require_array(params["scales"]),
            offsets=require_array(params["deq_biases"]),
            biases=require_array(params["biases"]) if self.has_biases else None,
        )


class MLXQuantizedLinear(MLXQuantizedLinearBase[MLXQuantizedLinearConfig]):
    pass


@dataclass(frozen=True)
class QLoRALinearConfig(GroupQuantizedLinearConfig):
    lora_rank: int
    lora_scale: float
    activation_precision: DTypeLike

    def random_init(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        key: Key[Array, ""],
    ) -> LinearBase:
        base_key, derived_key = jax.random.split(key)
        group_quantized_linear = super().random_init(input_dim, output_dims, has_biases, key=base_key)
        assert isinstance(group_quantized_linear, GroupQuantizedLinear)

        down_key, up_key = jax.random.split(derived_key)
        combined_lora_rank = self.lora_rank * len(output_dims)
        max_down_abs_value = 1 / math.sqrt(input_dim)
        lora_down_weights = jax.random.uniform(
            down_key,
            (combined_lora_rank, input_dim),
            minval=-max_down_abs_value,
            maxval=max_down_abs_value,
            dtype=self.activation_precision,
        )

        max_up_abs_value = 1 / math.sqrt(self.lora_rank)
        lora_up_weights = jax.random.uniform(
            up_key,
            (sum(output_dims), combined_lora_rank),
            minval=-max_up_abs_value,
            maxval=max_up_abs_value,
            dtype=self.activation_precision,
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
        key: Key[Array, ""],
    ) -> LinearBase:
        subkeys = jax.random.split(key, mixture_size)
        return eqx.filter_vmap(lambda k: self.random_init(input_dim, output_dims, has_biases, key=k))(subkeys)

    def empty(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> LinearBase:
        return self._empty_general((), input_dim, output_dims, has_biases)

    def _empty_general(
        self,
        leading_dims: tuple[int, ...],
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> LinearBase:
        weights = dummy_array(
            (*leading_dims, sum(output_dims), input_dim),
            dtype=self.activation_precision,
        )
        num_groups = self.group_count(input_dim)
        scales = dummy_array(
            (*leading_dims, sum(output_dims), num_groups),
            dtype=self.activation_precision,
        )
        zero_points = dummy_array(
            (*leading_dims, sum(output_dims), num_groups),
            dtype=self.activation_precision,
        )
        if has_biases:
            biases = dummy_array((*leading_dims, sum(output_dims)), dtype=self.activation_precision)
        else:
            biases = None

        combined_lora_rank = self.lora_rank * len(output_dims)
        lora_down_weights = dummy_array(
            (*leading_dims, combined_lora_rank, input_dim),
            dtype=self.activation_precision,
        )
        lora_up_weights = dummy_array(
            (*leading_dims, sum(output_dims), combined_lora_rank),
            dtype=self.activation_precision,
        )

        return QLoRALinear(
            config=self,
            output_dims=output_dims,
            weights=weights,
            scales=scales,
            biases=biases,
            zero_points=zero_points,
            lora_down_weights=lora_down_weights,
            lora_up_weights=lora_up_weights,
        )

    def empty_mixture(
        self,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> LinearBase:
        return self._empty_general((mixture_size,), input_dim, output_dims, has_biases)


class QLoRALinear(GroupQuantizedLinearBase[QLoRALinearConfig]):
    lora_down_weights: Float[Array, "*components lora_channels in_channels"] = field(
        tensor_sharding=TensorSharding(
            axes=(-2, -1),
            axes_names=(ShardingOrder.OUTPUT, ShardingOrder.INPUT),
        ),
    )
    lora_up_weights: Float[Array, "*components total_out_channels lora_channels"] = field(
        tensor_sharding=TensorSharding(
            axes=(-2, -1),
            axes_names=(ShardingOrder.OUTPUT, ShardingOrder.INPUT),
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.lora_down_weights.dtype != self.config.activation_precision:
            raise ValueError(
                f"LORA down weight dtype ({self.lora_down_weights.dtype}) is not equal to the"
                f" specified activation precision ({self.config.activation_precision})."
                " Quantized layers require parameter dtypes to be equal to the activation precision.",
            )
        combined_lora_rank = self.config.lora_rank * len(self.output_dims)
        *ld_num_components, lora_down_output_dim, lora_down_input_dim = self.lora_down_weights.shape
        if lora_down_output_dim != combined_lora_rank:
            raise ValueError(
                f"Number of output channels in LORA down weights ({lora_down_output_dim}) is not"
                f" equal to lora_rank * num_outputs ({combined_lora_rank}).",
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
        if self.lora_up_weights.dtype != self.config.activation_precision:
            raise ValueError(
                f"LORA up weight dtype ({self.lora_up_weights.dtype}) is not equal to specified activation precision"
                f" ({self.config.activation_precision})."
                " Quantized layers require parameter dtypes to be equal to the activation precision.",
            )
        *lu_num_components, lora_up_output_dim, lora_up_input_dim = self.lora_up_weights.shape
        if lora_up_output_dim != sum(self.output_dims):
            raise ValueError(
                f"Number of output channels in LORA up weights ({lora_up_output_dim}) is not"
                f" equal to sum of output dims ({sum(self.output_dims)}).",
            )
        if lora_up_input_dim != combined_lora_rank:
            raise ValueError(
                f"Number of input channels in LORA up weights ({lora_up_input_dim}) is not"
                f" equal to lora_rank * num_outputs ({combined_lora_rank}).",
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
        lora_hidden = jnp.dot(self.lora_down_weights, inputs)
        joint_lora_out = jnp.dot(self.lora_up_weights, lora_hidden)
        joint_out = joint_q_out + self.config.lora_scale * joint_lora_out
        if self.biases is not None:
            joint_out = joint_out + self.biases
        return tuple(jnp.split(joint_out, self.get_split_points(self.output_dims)))

    def export_weights(self) -> ParameterTree:
        return dict(
            down_weights=self.lora_down_weights,
            up_weights=self.lora_up_weights,
            **cast("dict[str, ParameterTree]", super().export_weights()),
        )

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> "QLoRALinear":
        base = super().import_weights(weights)
        params = _linear_params(weights)
        n_outs = len(self.output_dims)
        lora_rank = self.config.lora_rank

        down_weights = jnp.swapaxes(require_array(params["down_weights"]), -1, -2)

        up_blocks = [jnp.swapaxes(require_array(w), -1, -2) for w in require_tree(params["up_weights"])]
        up_weights = jnp.concatenate(
            [
                jnp.concatenate(
                    [
                        jnp.zeros((dim, lora_rank * i), dtype=up_blocks[i].dtype),
                        up_blocks[i],
                        jnp.zeros((dim, lora_rank * (n_outs - i - 1)), dtype=up_blocks[i].dtype),
                    ],
                    axis=-1,
                )
                for i, dim in enumerate(self.output_dims)
            ],
            axis=-2,
        )

        return replace(base, lora_down_weights=down_weights, lora_up_weights=up_weights)


LinearConfig = FullPrecisionLinearConfig | GroupQuantizedLinearConfig | MLXQuantizedLinearConfig | QLoRALinearConfig


register_config_union(LinearConfig)


def _linear_params(weights: ParameterTree[Array]) -> Mapping[str, Any]:
    params = require_mapping(weights)
    if "inner_linear" not in params:
        return params
    return require_mapping(require_tree(params["inner_linear"]))


def _structure_linear_config(
    config: dict | None,
    _: type[LinearConfig | None],
) -> LinearConfig | None:
    if config is None:
        return None
    if config["type"] == "RHTLinearWrapperConfig":
        config = config["inner_config"]
    type_name = config["type"]
    target_type = {
        "FullPrecisionLinearConfig": FullPrecisionLinearConfig,
        "GroupQuantizedLinearConfig": GroupQuantizedLinearConfig,
        "MLXQuantizedLinearConfig": MLXQuantizedLinearConfig,
        "QLoRALinearConfig": QLoRALinearConfig,
    }[type_name]
    return config_converter.structure({k: v for k, v in config.items() if k != "type"}, target_type)


config_converter.register_structure_hook(LinearConfig, _structure_linear_config)
config_converter.register_structure_hook(LinearConfig | None, _structure_linear_config)
