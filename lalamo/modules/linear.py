import math
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import NamedTuple, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, dummy_array
from lalamo.quantization import QuantizationMode, dynamically_quantize_activations, quantize_weights

from .common import (
    LalamoModule,
    WeightLayout,
    from_layout,
    into_layout,
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
    def _get_split_points(output_dims: Sequence[int]) -> tuple[int, ...]:
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

    @abstractmethod
    def empty(
        self,
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

    def empty(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "FullPrecisionLinear":
        weights = dummy_array(
            (sum(output_dims), input_dim),
            dtype=self.precision,
        )
        if has_biases:
            biases = dummy_array((sum(output_dims),), dtype=self.precision)
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
        w_output_dim, _ = self.weights.shape
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

    @eqx.filter_jit
    def __call__(self, inputs: Float[Array, " in_channels"]) -> tuple[Float[Array, " out_channels"], ...]:
        result = self.weights @ inputs
        if self.biases is not None:
            result = result + self.biases
        return tuple(jnp.split(result, self._get_split_points(self.output_dims)))

    def export_weights(self, weight_layout: WeightLayout = WeightLayout.AUTO) -> ParameterTree:
        result = dict(weights=into_layout(self.weights, weight_layout))
        if self.biases is not None:
            result["biases"] = self.biases
        return result

    def import_weights(
        self,
        weights: ParameterTree[Array],
        weight_layout: WeightLayout = WeightLayout.AUTO,
    ) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            weights=from_layout(weights["weights"], weight_layout),
            biases=weights["biases"] if self.has_biases else None,
        )


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

    def empty(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> LinearBase:
        weights = dummy_array(
            (sum(output_dims), input_dim),
            dtype=self.activation_precision,
        )
        num_groups = input_dim // self.group_size
        scales = dummy_array((sum(output_dims), num_groups), dtype=self.activation_precision)

        if has_biases:
            biases = dummy_array((sum(output_dims),), dtype=self.activation_precision)
        else:
            biases = None
        zero_points = dummy_array((sum(output_dims), num_groups), dtype=self.activation_precision)

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
        w_output_dim, _ = self.weights.shape
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

    @eqx.filter_jit
    def __call__(self, inputs: Float[Array, " in_channels"]) -> tuple[Float[Array, " out_channels"], ...]:
        result = self._apply_weights(inputs)
        if self.biases is not None:
            result = result + self.biases
        return tuple(jnp.split(result, self._get_split_points(self.output_dims)))

    def export_weights(self, weight_layout: WeightLayout = WeightLayout.AUTO) -> ParameterTree:
        expected_weight_layout = WeightLayout.OUTPUT_INPUT
        exported_weights = into_layout(self.int_weights, expected_weight_layout)
        exported_zero_points = into_layout(self.int_zero_points, expected_weight_layout)
        exported_scales = into_layout(self.scales, expected_weight_layout)

        result = dict(
            weights=exported_weights,
            zero_points=exported_zero_points,
            scales=exported_scales,
        )
        if self.biases is not None:
            result["biases"] = self.biases
        return result

    def import_weights(
        self,
        weights: ParameterTree[Array],
        weight_layout: WeightLayout = WeightLayout.AUTO,
    ) -> Self:
        assert isinstance(weights, Mapping)
        assert isinstance(weights["weights"], Array)
        return replace(
            self,
            weights=from_layout(weights["weights"].astype(self.weights.dtype), weight_layout),
            scales=from_layout(weights["scales"], weight_layout),
            zero_points=from_layout(weights["zero_points"], weight_layout).astype(self.zero_points.dtype),
            biases=weights["biases"] if self.has_biases else None,
        )


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
            (hidden_lora_rank, input_dim),
            dtype=self.activation_precision,
        )
        lora_up_weights = tuple(
            dummy_array(
                (output_dim, self.lora_rank),
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

    @eqx.filter_jit
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

    def export_weights(self, weight_layout: WeightLayout = WeightLayout.AUTO) -> ParameterTree:
        quantized_linear_weights: dict[str, ParameterTree] = super().export_weights()  # type: ignore
        exported_lora_down_weights = into_layout(self.lora_down_weights, weight_layout)
        exported_lora_up_weights = [
            into_layout(lora_up_weight, weight_layout) for lora_up_weight in self.lora_up_weights
        ]
        return dict(
            down_weights=into_layout(exported_lora_down_weights, weight_layout),
            up_weights=[into_layout(w, weight_layout) for w in exported_lora_up_weights],
            **quantized_linear_weights,
        )

    def import_weights(
        self,
        weights: ParameterTree[Array],
        weight_layout: WeightLayout = WeightLayout.AUTO,
    ) -> Self:
        base = super().import_weights(weights, weight_layout)
        assert isinstance(weights, Mapping)
        assert isinstance(weights["up_weights"], Sequence)
        return replace(
            base,
            lora_down_weights=from_layout(weights["down_weights"], weight_layout),
            lora_up_weights=tuple(from_layout(up_weights, weight_layout) for up_weights in weights["up_weights"]),
        )


LinearConfig = FullPrecisionLinearConfig | GroupQuantizedLinearConfig | QLoRALinearConfig


register_config_union(LinearConfig)
