import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Self, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, dummy_array, require_array
from lalamo.quantization import QuantizationMode, dynamically_quantize_activations, quantize_weights
from lalamo.utils import jax_uint4_to_packed_uint8, jax_uint8_to_unpacked_uint4

from .common import (
    LalamoModule,
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
        key: PRNGKeyArray,
    ) -> LinearBase: ...

    @abstractmethod
    def random_init_mixture(
        self,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        key: PRNGKeyArray,
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
        key: PRNGKeyArray,
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
        key: PRNGKeyArray,
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
    weights: Float[Array, "*components total_out_channels in_channels"]
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
        result = self.weights @ inputs
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
        assert isinstance(weights, Mapping)
        return replace(
            self,
            weights=weights["weights"],
            biases=weights["biases"] if self.has_biases else None,
        )


@dataclass(frozen=True)
class QuantizedLinearConfigBase(LinearConfigBase):
    group_size: int
    weight_quantization_mode: QuantizationMode
    activation_quantization_mode: QuantizationMode | None
    activation_precision: DTypeLike


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
    def random_init(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        key: PRNGKeyArray,
    ) -> LinearBase:
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
        key: PRNGKeyArray,
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
        num_groups = input_dim // self.group_size
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

    def __post_init__(self) -> None:
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
        assert isinstance(weights, Mapping)
        unpacked_weights = require_array(weights["weights"])
        unpacked_zero_points = require_array(weights["zero_points"])
        if self.config.weight_quantization_mode == QuantizationMode.UINT4:
            unpacked_weights = jax_uint8_to_unpacked_uint4(unpacked_weights)
            unpacked_zero_points = jax_uint8_to_unpacked_uint4(unpacked_zero_points)
        return replace(
            self,
            weights=unpacked_weights.astype(self.weights.dtype),
            scales=require_array(weights["scales"]),
            zero_points=unpacked_zero_points.astype(self.zero_points.dtype),
            biases=require_array(weights["biases"]) if self.has_biases else None,
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
        key: PRNGKeyArray,
    ) -> LinearBase:
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

        deq_bias = min_val + 2 ** (self.weight_quantization_mode.bits - 1)
        deq_biases = deq_bias * jnp.ones((sum(output_dims), num_groups), dtype=self.activation_precision)

        return MLXQuantizedLinear(
            config=self,
            output_dims=output_dims,
            weights=weights,
            scales=scales,
            deq_biases=deq_biases,
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
        num_groups = input_dim // self.group_size
        scales = dummy_array((*leading_dims, sum(output_dims), num_groups), dtype=self.activation_precision)

        if has_biases:
            biases = dummy_array((*leading_dims, sum(output_dims)), dtype=self.activation_precision)
        else:
            biases = None
        deq_biases = dummy_array((*leading_dims, sum(output_dims), num_groups), dtype=self.activation_precision)

        return MLXQuantizedLinear(
            config=self,
            output_dims=output_dims,
            weights=weights,
            scales=scales,
            deq_biases=deq_biases,
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
    weights: Float[Array, "*components total_out_channels in_channels"]
    scales: Float[Array, "*components total_out_channels groups"]
    deq_biases: Float[Array, "*components total_out_channels groups"]
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
        quantized = quantize_weights(self.weights, self.config.weight_quantization_mode)
        casted = quantized.astype(self.config.weight_quantization_mode.dtype)

        if self.config.weight_quantization_mode == QuantizationMode.UINT4:
            packed = jax_uint4_to_packed_uint8(casted)
        else:
            packed = casted

        return packed

    def __post_init__(self) -> None:
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

        if self.deq_biases.dtype != self.config.activation_precision:
            raise ValueError(
                f"Dequantization bias dtype ({self.deq_biases.dtype}) is not equal to specified activation precision"
                f" ({self.config.activation_precision}).",
                " Quantized layers require parameter dtypes to be equal to the activation precision.",
            )
        *zp_num_components, zp_output_dim, zp_num_groups = self.deq_biases.shape
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
        assert isinstance(weights, Mapping)
        unpacked_weights = require_array(weights["weights"])
        if self.config.weight_quantization_mode == QuantizationMode.UINT4:
            unpacked_weights = jax_uint8_to_unpacked_uint4(unpacked_weights)
        return replace(
            self,
            weights=unpacked_weights.astype(self.weights.dtype),
            scales=require_array(weights["scales"]),
            deq_biases=require_array(weights["deq_biases"]),
            biases=require_array(weights["biases"]) if self.has_biases else None,
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
        key: PRNGKeyArray,
    ) -> LinearBase:
        base_key, derived_key = jax.random.split(key)
        group_quantized_linear = super().random_init(input_dim, output_dims, has_biases, key=base_key)
        assert isinstance(group_quantized_linear, GroupQuantizedLinear)

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
    ) -> LinearBase:
        subkeys = jax.random.split(key, mixture_size)
        return eqx.filter_vmap(lambda k: self.random_init(input_dim, output_dims, has_biases, key=k))(subkeys)

    def empty(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> LinearBase:
        group_quantized_linear = super().empty(input_dim, output_dims, has_biases)
        assert isinstance(group_quantized_linear, GroupQuantizedLinear)
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
    ) -> LinearBase:
        group_quantized_linear = super().empty(input_dim, output_dims, has_biases)
        assert isinstance(group_quantized_linear, GroupQuantizedLinear)

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
    ) -> LinearBase:
        return self._empty_general((mixture_size,), input_dim, output_dims, has_biases)


class QLoRALinear(GroupQuantizedLinearBase[QLoRALinearConfig]):
    lora_down_weights: Float[Array, "*components in_channels total_lora_channels"]
    lora_up_weights: tuple[Float[Array, "*components lora_channels out_channels"], ...]

    def _split_biases(self) -> tuple[Float[Array, "*components out_channels"] | None, ...]:
        if self.biases is not None:
            return tuple(jnp.split(self.biases, self.get_split_points(self.output_dims)))
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
        base = cast("Self", super().import_weights(weights)) # ty bug
        assert isinstance(weights, Mapping)
        assert isinstance(weights["up_weights"], Sequence)
        return replace(
            base,
            lora_down_weights=weights["down_weights"],
            lora_up_weights=tuple(up_weights for up_weights in weights["up_weights"]),
        )


LinearConfig = FullPrecisionLinearConfig | GroupQuantizedLinearConfig | MLXQuantizedLinearConfig | QLoRALinearConfig


register_config_union(LinearConfig)
