import math
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
from einops import rearrange
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterDict
from lalamo.quantization import QuantizationMode, dynamically_quantize_activations, quantize_weights

from .common import LalamoModule, WeightLayout, register_config_union

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

    @classmethod
    def _default_weight_layout(cls) -> WeightLayout:
        return WeightLayout.INPUT_OUTPUT

    @classmethod
    def _into_layout(
        cls,
        weights: Float[Array, "in_channels out_channels"],
        layout: WeightLayout,
    ) -> Float[Array, "in_channels out_channels"] | Float[Array, "out_channels in_channels"]:
        if layout == WeightLayout.AUTO:
            layout = cls._default_weight_layout()
        match layout:
            case WeightLayout.OUTPUT_INPUT:
                return weights
            case WeightLayout.INPUT_OUTPUT:
                return rearrange(
                    weights,
                    "total_out_channels in_channels -> in_channels total_out_channels",
                )
        raise ValueError(f"Unsupported weight layout: {layout}")

    @classmethod
    def _get_split_points(cls, output_dims: Sequence[int]) -> tuple[int, ...]:
        result = []
        last_split_point = 0
        for dim in output_dims[:-1]:
            last_split_point += dim
            result.append(last_split_point)
        return tuple(result)


@dataclass(frozen=True)
class LinearConfigBase:
    @abstractmethod
    def random_init(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        key: PRNGKeyArray,
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
    ) -> LinearBase:
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


class FullPrecisionLinear(LinearBase[FullPrecisionLinearConfig]):
    weights: Float[Array, "total_out_channels in_channels"]
    biases: Float[Array, " total_out_channels"] | None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def input_dim(self) -> int:
        _, input_dim = self.weights.shape
        return input_dim

    @property
    def has_biases(self) -> bool:
        return self.biases is not None

    def __post_init__(self) -> None:
        if self.weights.dtype != self.config.precision:
            raise ValueError(
                f"Weight dtype ({self.weights.dtype}) is not equal to specified precision ({self.config.precision}).",
            )
        w_output_dim, w_input_dim = self.weights.shape
        if w_output_dim != sum(self.output_dims):
            raise ValueError(
                f"Number of output channels in weights ({w_output_dim}) is not"
                f" equal to sum of output dims ({sum(self.output_dims)}).",
            )
        if self.biases is None:
            return
        (b_output_dim,) = self.biases.shape
        if w_output_dim != b_output_dim:
            raise ValueError(
                f"Number of output channels in weights ({w_output_dim}) is not"
                f" equal to number of output channels in biases ({b_output_dim}).",
            )
        if self.biases.dtype != self.config.precision:
            raise ValueError(
                f"Bias dtype ({self.biases.dtype}) is not equal to specified precision ({self.config.precision}).",
            )

    def __call__(self, inputs: Float[Array, " in_channels"]) -> tuple[Float[Array, " out_channels"], ...]:
        result = self.weights @ inputs
        if self.biases is not None:
            result = result + self.biases
        return tuple(jnp.split(result, self._get_split_points(self.output_dims)))

    def export_weights(self, weight_layout: WeightLayout = WeightLayout.AUTO) -> ParameterDict:
        result = ParameterDict(weights=self._into_layout(self.weights, weight_layout))
        if self.biases is not None:
            result["biases"] = self.biases
        return result


@dataclass(frozen=True)
class GroupQuantizedLinearConfig(LinearConfigBase):
    group_size: int
    weight_quantization_mode: QuantizationMode
    activation_quantization_mode: QuantizationMode | None
    activation_precision: DTypeLike

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


class RequantizedWeights(NamedTuple):
    weights: Int[Array, "total_out_channels in_channels"]
    zero_points: Int[Array, "groups in_channels"]
    scales: Float[Array, "groups in_channels"]


class GroupQuantizedLinearBase[ConfigT: GroupQuantizedLinearConfig](LinearBase[ConfigT]):
    weights: Float[Array, "total_out_channels in_channels"]
    scales: Float[Array, "total_out_channels groups"]
    zero_points: Float[Array, "total_out_channels groups"]
    biases: Float[Array, " total_out_channels"] | None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.activation_precision

    @property
    def input_dim(self) -> int:
        _, input_dim = self.weights.shape
        return input_dim

    @property
    def has_biases(self) -> bool:
        return self.biases is not None

    @property
    def num_groups(self) -> int:
        return self.input_dim // self.config.group_size

    @property
    def int_weights(self) -> Int[Array, "out_channels (groups in_channels)"]:
        result = quantize_weights(self.weights, self.config.weight_quantization_mode)
        return result.astype(self.config.weight_quantization_mode.dtype)

    @property
    def int_zero_points(self) -> Int[Array, "out_channels (groups in_channels)"]:
        result = quantize_weights(self.zero_points, self.config.weight_quantization_mode)
        return result.astype(self.config.weight_quantization_mode.dtype)

    def __post_init__(self) -> None:
        if self.weights.dtype != self.config.activation_precision:
            raise ValueError(
                f"Weight dtype ({self.weights.dtype}) is not equal to specified activation precision"
                f" ({self.config.activation_precision}).",
                " Quantized layers require parameter dtypes to be equal to the activation precision.",
            )
        w_output_dim, w_input_dim = self.weights.shape
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
        s_output_dim, s_num_groups = self.scales.shape
        if w_output_dim != s_output_dim:
            raise ValueError(
                f"Number of output channels in weights ({w_output_dim}) is not"
                f" equal to number of output channels in scales ({s_output_dim}).",
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
        (zp_output_dim, zp_num_groups) = self.zero_points.shape
        if w_output_dim != zp_output_dim:
            raise ValueError(
                f"Number of output channels in weights ({w_output_dim}) is not"
                f" equal to number of output channels in zero points ({zp_output_dim}).",
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
            (b_output_dim,) = self.biases.shape
            if w_output_dim != b_output_dim:
                raise ValueError(
                    f"Number of output channels in weights ({w_output_dim}) is not"
                    f" equal to number of output channels in biases ({b_output_dim}).",
                )

    def _prepare_scaled_weights(self) -> Float[Array, "total_out_channels in_channels"]:
        quantized_weights = quantize_weights(self.weights, self.config.weight_quantization_mode)
        grouped_weights = rearrange(
            quantized_weights,
            "total_out_channels (groups group_channels) -> total_out_channels groups group_channels",
            groups=self.num_groups,
        )

        zero_points = rearrange(self.zero_points, "total_out_channels groups -> total_out_channels groups 1")
        grouped_weights = grouped_weights - zero_points

        scales = rearrange(self.scales, "total_out_channels groups -> total_out_channels groups 1")
        scaled_grouped_weights = grouped_weights * scales
        result = rearrange(
            scaled_grouped_weights,
            "total_out_channels groups group_channels -> total_out_channels (groups group_channels)",
        )
        return result

    def _apply_weights(self, inputs: Float[Array, " in_channels"]) -> Float[Array, " total_out_channels"]:
        if self.config.activation_quantization_mode is not None:
            inputs = dynamically_quantize_activations(inputs, self.config.activation_quantization_mode)
        return self._prepare_scaled_weights() @ inputs

    def __call__(self, inputs: Float[Array, " in_channels"]) -> tuple[Float[Array, " out_channels"], ...]:
        result = self._apply_weights(inputs)
        if self.biases is not None:
            result = result + self.biases
        return tuple(jnp.split(result, self._get_split_points(self.output_dims)))

    def requantize_weights(self, weights, zero_points, scales):
        """
        Requantize weights from [20, 6144] grouping to [2560, 48] grouping.

        Args:
            weights: uint4 array of shape [M, N]
            zero_points: uint4 array of shape [M//group_size_0, N//group_size_1]
            scales: float16 array of shape [M//group_size_0, N//group_size_1]

        Returns:
            new_weights: uint4 array of shape [M, N]
            new_zero_points: uint4 array of shape [M, N//128]
            new_scales: float16 array of shape [M, N//128]
        """
        # Get dimensions
        M, N = weights.shape
        old_groups_0, old_groups_1 = zero_points.shape

        # Calculate old group sizes
        old_group_size_0 = M // old_groups_0  # 2560 // 20 = 128
        old_group_size_1 = N // old_groups_1  # 6144 // 6144 = 1

        # New group sizes
        new_group_size_0 = 1  # 2560 // 2560 = 1
        new_group_size_1 = self.config.group_size  # 6144 // 48 = 128

        # Step 1: Dequantize with original parameters
        # Expand zero_points and scales to match weights shape
        zp_expanded = jnp.repeat(jnp.repeat(zero_points, old_group_size_0, axis=0), old_group_size_1, axis=1)
        scales_expanded = jnp.repeat(jnp.repeat(scales, old_group_size_0, axis=0), old_group_size_1, axis=1)

        # Dequantize (convert to float for computation)
        weights_float = weights.astype(jnp.float32)
        zp_float = zp_expanded.astype(jnp.float32)
        dequantized = (weights_float - zp_float) * scales_expanded.astype(jnp.float32)

        # Step 2: Requantize with new group structure [2560, 48]
        # Reshape for new groups
        dequantized_reshaped = dequantized.reshape(
            M // new_group_size_0,
            new_group_size_0,
            N // new_group_size_1,
            new_group_size_1,
        )

        # Compute new scales and zero points per group
        # Move group dimensions to the end for reduction
        dequantized_groups = dequantized_reshaped.transpose(0, 2, 1, 3)  # [2560, 48, 1, 128]

        # Find min and max per group
        group_min = dequantized_groups.min(axis=(2, 3), keepdims=True)
        group_max = dequantized_groups.max(axis=(2, 3), keepdims=True)

        # Compute scales (with small epsilon to avoid division by zero)
        eps = 1e-6
        new_scales = ((group_max - group_min) / 15.0 + eps).astype(scales.dtype)
        new_scales = new_scales.squeeze(axis=(2, 3))  # [2560, 48]

        # Compute zero points (quantize to uint4 range 0-15)
        new_zero_points = jnp.round(-group_min.squeeze(axis=(2, 3)) / new_scales).astype(jnp.uint4)
        new_zero_points = jnp.clip(new_zero_points, 0, 15)

        # Quantize with new parameters
        scales_expanded_new = jnp.repeat(new_scales, new_group_size_1, axis=1).reshape(M, N)
        zp_expanded_new = jnp.repeat(new_zero_points, new_group_size_1, axis=1).reshape(M, N)

        new_weights = jnp.round(
            dequantized / scales_expanded_new.astype(jnp.float32) + zp_expanded_new.astype(jnp.float32),
        )
        new_weights = jnp.clip(new_weights, 0, 15).astype(jnp.uint4)

        return new_weights, new_zero_points, new_scales

    def export_weights(self, weight_layout: WeightLayout = WeightLayout.AUTO) -> ParameterDict:
        exported_weights = self._into_layout(self.int_weights, weight_layout)

        exported_zero_points = self._into_layout(self.int_zero_points, weight_layout)

        exported_scales = self._into_layout(self.scales, weight_layout)

        # CRIMINAL HACK!!!
        exported_weights, exported_zero_points, exported_scales = self.requantize_weights(
            exported_weights,
            exported_zero_points,
            exported_scales,
        )

        result = ParameterDict(
            weights=exported_weights,
            zero_points=exported_zero_points,
            scales=exported_scales,
        )
        if self.biases is not None:
            result["biases"] = self.biases
        return result


class GroupQuantizedLinear(GroupQuantizedLinearBase[GroupQuantizedLinearConfig]):
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
            (hidden_lora_rank, input_dim),
            minval=-max_down_abs_value,
            maxval=max_down_abs_value,
            dtype=self.activation_precision,
        )

        up_keys = jax.random.split(up_key_root, len(output_dims))
        max_up_abs_value = 1 / math.sqrt(hidden_lora_rank)
        lora_up_weights = tuple(
            jax.random.uniform(
                up_key,
                (output_dim, self.lora_rank),
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


class QLoRALinear(GroupQuantizedLinearBase[QLoRALinearConfig]):
    lora_down_weights: Float[Array, "total_lora_channels in_channels"]
    lora_up_weights: tuple[Float[Array, "out_channels lora_channels"], ...]

    def _split_biases(self) -> tuple[Float[Array, " out_channels"] | None, ...]:
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
        lora_down_output_dim, lora_down_input_dim = self.lora_down_weights.shape
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
            lora_up_output_dim, lora_up_input_dim = lora_up_weight.shape
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

    def __call__(self, inputs: Float[Array, " in_channels"]) -> tuple[Float[Array, " out_channels"], ...]:
        joint_q_out = self._apply_weights(inputs)
        q_outs = jnp.split(joint_q_out, self._get_split_points(self.output_dims))

        joint_lora_hidden = self.lora_down_weights @ inputs
        lora_hiddens = jnp.split(joint_lora_hidden, self._get_split_points([self.config.lora_rank] * self.num_outputs))
        lora_outs = [
            lora_up_weight @ lora_hidden
            for lora_up_weight, lora_hidden in zip(self.lora_up_weights, lora_hiddens, strict=True)
        ]

        results = []
        for q_out, lora_out, bias in zip(q_outs, lora_outs, self._split_biases(), strict=True):
            result = q_out + self.config.lora_scale * lora_out
            if bias is not None:
                result = result + bias
            results.append(result)

        return tuple(results)

    def export_weights(self, weight_layout: WeightLayout = WeightLayout.AUTO) -> ParameterDict:
        quantized_linear_weights = super().export_weights()
        exported_lora_down_weights = self._into_layout(self.lora_down_weights, weight_layout)
        exported_lora_up_weights = tuple(
            self._into_layout(lora_up_weight, weight_layout) for lora_up_weight in self.lora_up_weights
        )
        return ParameterDict(
            **quantized_linear_weights,
            down_weights=exported_lora_down_weights,
            up_weights=exported_lora_up_weights,
        )


LinearConfig = FullPrecisionLinearConfig | GroupQuantizedLinearConfig | QLoRALinearConfig


register_config_union(LinearConfig)
