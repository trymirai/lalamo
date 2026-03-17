import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax, vmap
from jaxtyping import Array, DTypeLike, Float, PRNGKeyArray

from lalamo.common import ParameterTree, dummy_array, require_array, require_mapping, require_tree
from lalamo.modules.activations import Activation
from lalamo.modules.common import LalamoModule, register_config_union
from lalamo.modules.linear import FullPrecisionLinear, FullPrecisionLinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig


class Conv1dPadding(Enum):
    CAUSAL = "causal"
    SYMMETRIC = "symmetric"


def _get_extra_padding_for_conv1d(length: int, kernel_size: int, stride: int, padding_total: int = 0) -> int:
    """Calculate extra padding needed to ensure output length is correct."""
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return int(ideal_length - length)


@dataclass(frozen=True)
class Conv1dConfig:
    precision: DTypeLike
    has_biases: bool
    padding: Conv1dPadding = Conv1dPadding.CAUSAL

    def random_init(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        *,
        key: PRNGKeyArray,
    ) -> "Conv1d":
        effective_kernel_size = (kernel_size - 1) * dilation + 1
        weights = jax.random.normal(
            key,
            shape=(out_channels, in_channels // groups, kernel_size),
            dtype=self.precision,
        )

        if self.has_biases:
            biases = jnp.zeros((out_channels,), dtype=self.precision)
        else:
            biases = None

        return Conv1d(
            config=self,
            weights=weights,
            biases=biases,
            stride=stride,
            dilation=dilation,
            groups=groups,
            effective_kernel_size=effective_kernel_size,
        )

    def empty(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ) -> "Conv1d":
        effective_kernel_size = (kernel_size - 1) * dilation + 1

        weights = dummy_array(
            (out_channels, in_channels // groups, kernel_size),
            dtype=self.precision,
        )

        if self.has_biases:
            biases = dummy_array((out_channels,), dtype=self.precision)
        else:
            biases = None

        return Conv1d(
            config=self,
            weights=weights,
            biases=biases,
            stride=stride,
            dilation=dilation,
            groups=groups,
            effective_kernel_size=effective_kernel_size,
        )


class Conv1d(LalamoModule[Conv1dConfig]):
    weights: Float[Array, "out_channels in_channels_per_group kernel_size"]
    biases: Float[Array, " out_channels"] | None

    stride: int = eqx.field(static=True)
    dilation: int = eqx.field(static=True)
    groups: int = eqx.field(static=True)
    effective_kernel_size: int = eqx.field(static=True)

    @property
    def padding_mode(self) -> Conv1dPadding:
        return self.config.padding

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def out_channels(self) -> int:
        out_channels, _, _ = self.weights.shape
        return out_channels

    @property
    def in_channels(self) -> int:
        _, in_channels_per_group, _ = self.weights.shape
        return in_channels_per_group * self.groups

    @property
    def kernel_size(self) -> int:
        _, _, kernel_size = self.weights.shape
        return kernel_size

    @property
    def causal_padding(self) -> int:
        return self.effective_kernel_size - self.stride

    def _call_causal(
        self,
        x: Float[Array, "batch sequence in_channels"],
    ) -> Float[Array, "batch sequence_out out_channels"]:
        length = x.shape[1]
        pad = self.causal_padding
        extra_padding = _get_extra_padding_for_conv1d(length, self.effective_kernel_size, self.stride, pad)
        x_padded = jnp.pad(x, ((0, 0), (pad, extra_padding), (0, 0)), mode="constant", constant_values=0)
        return lax.conv_general_dilated(
            x_padded,
            self.weights,
            window_strides=(self.stride,),
            padding="VALID",
            rhs_dilation=(self.dilation,),
            feature_group_count=self.groups,
            dimension_numbers=("NHC", "OIH", "NHC"),
        )

    def _call_symmetric(
        self,
        x: Float[Array, "batch sequence in_channels"],
    ) -> Float[Array, "batch sequence out_channels"]:
        assert self.stride == 1, "Symmetric padding requires stride=1"
        pad = (self.kernel_size - 1) // 2 * self.dilation
        x_padded = jnp.pad(x, ((0, 0), (pad, pad), (0, 0)))
        return lax.conv_general_dilated(
            x_padded,
            self.weights,
            window_strides=(1,),
            padding="VALID",
            rhs_dilation=(self.dilation,),
            feature_group_count=self.groups,
            dimension_numbers=("NHC", "OIH", "NHC"),
        )

    def __call__(
        self,
        x: Float[Array, "batch sequence in_channels"],
    ) -> Float[Array, "batch sequence_out out_channels"]:
        match self.padding_mode:
            case Conv1dPadding.CAUSAL:
                output = self._call_causal(x)
            case Conv1dPadding.SYMMETRIC:
                output = self._call_symmetric(x)

        if self.biases is not None:
            output = output + self.biases[None, None, :]

        return output

    def export_weights(self) -> ParameterTree[Array]:
        result: dict[str, Array] = {"weights": self.weights}
        if self.biases is not None:
            result["biases"] = self.biases
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        assert isinstance(weights["weights"], Array)
        if self.biases is not None:
            assert isinstance(weights["biases"], Array)
            biases = weights["biases"]
        else:
            biases = None
        return replace(
            self,
            weights=weights["weights"],
            biases=biases,
        )


@dataclass(frozen=True)
class CausalTransposeConv1dConfig:
    precision: DTypeLike
    has_biases: bool

    def random_init(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        *,
        key: PRNGKeyArray,
    ) -> "CausalTransposeConv1d":
        # Weight shape: (out_channels, in_channels // groups, kernel_size) - JAX OIK format
        in_per_group = in_channels // groups
        weights = jax.random.normal(
            key,
            (out_channels, in_per_group, kernel_size),
            dtype=self.precision,
        )

        if self.has_biases:
            biases = jnp.zeros((out_channels,), dtype=self.precision)
        else:
            biases = None

        return CausalTransposeConv1d(
            config=self,
            weights=weights,
            biases=biases,
            in_channels=in_channels,
            stride=stride,
            groups=groups,
        )

    def empty(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
    ) -> "CausalTransposeConv1d":
        # Weight shape: (out_channels, in_channels // groups, kernel_size) - JAX OIK format
        in_per_group = in_channels // groups
        weights = dummy_array(
            (out_channels, in_per_group, kernel_size),
            dtype=self.precision,
        )

        if self.has_biases:
            biases = dummy_array((out_channels,), dtype=self.precision)
        else:
            biases = None

        return CausalTransposeConv1d(
            config=self,
            weights=weights,
            biases=biases,
            in_channels=in_channels,
            stride=stride,
            groups=groups,
        )


class CausalTransposeConv1d(LalamoModule[CausalTransposeConv1dConfig]):
    """Causal transposed 1D convolution (deconvolution) with groups support.

    Implements causal transposed convolution by removing appropriate padding
    from the output. Supports grouped convolutions for efficient upsampling.

    Weight format: (out_channels, in_channels // groups, kernel_size) - JAX OIK format
    with kernel already flipped for transposed convolution.
    """

    weights: Float[Array, "out_channels in_channels_per_group kernel_size"]
    biases: Float[Array, " out_channels"] | None

    in_channels: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    groups: int = eqx.field(static=True)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def out_channels(self) -> int:
        out_channels, _, _ = self.weights.shape
        return out_channels

    @property
    def kernel_size(self) -> int:
        _, _, kernel_size = self.weights.shape
        return kernel_size

    def __call__(
        self,
        x: Float[Array, "batch sequence in_channels"],
    ) -> Float[Array, "batch sequence_out out_channels"]:
        # Input: (N, S, C) - NSC format
        # Kernel: (O, I/g, K) - JAX OIK format, already transformed and flipped
        # Output: (N, S, C) - NSC format

        # Calculate padding for transposed convolution
        # Output length = (input_length - 1) * stride + kernel_size
        # We want VALID-like behavior then trim
        padding_needed = self.kernel_size - 1
        padding = ((padding_needed, padding_needed),)

        output = lax.conv_general_dilated(
            x,
            self.weights,
            window_strides=(1,),
            padding=padding,
            lhs_dilation=(self.stride,),
            rhs_dilation=(1,),
            dimension_numbers=("NHC", "OIH", "NHC"),
            feature_group_count=self.groups,
        )

        if self.biases is not None:
            output = output + self.biases[None, None, :]

        # Remove padding to maintain causality
        # NanoCodec uses trim_right_ratio=1, meaning all padding trimmed from right
        pad = self.kernel_size - self.stride
        padding_right = math.ceil(pad)
        padding_left = pad - padding_right

        # Unpad: remove padding_left from start and padding_right from end
        if padding_left > 0 or padding_right > 0:
            end = output.shape[1] - padding_right if padding_right > 0 else output.shape[1]
            output = output[:, padding_left:end, :]

        return output

    def export_weights(self) -> ParameterTree[Array]:
        result: dict[str, Array] = {"weights": self.weights}
        if self.biases is not None:
            result["biases"] = self.biases
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        assert isinstance(weights["weights"], Array)
        if self.biases is not None:
            assert isinstance(weights["biases"], Array)
            biases = weights["biases"]
        else:
            biases = None
        return replace(
            self,
            weights=weights["weights"],
            biases=biases,
        )


@dataclass(frozen=True)
class Snake1dConfig:
    precision: DTypeLike

    def empty(self, channels: int) -> "Snake1d":
        alpha = dummy_array((channels,), dtype=self.precision)
        return Snake1d(config=self, alpha=alpha)

    def random_init(self, channels: int) -> "Snake1d":
        alpha = jnp.ones((channels,), dtype=self.precision)
        return Snake1d(config=self, alpha=alpha)


class Snake1d(LalamoModule[Snake1dConfig]):
    """Snake1d activation module.
    Implements the Snake activation function: x + (1/alpha) * sin^2(alpha * x)
    """

    alpha: Float[Array, " channels"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def channels(self) -> int:
        return self.alpha.shape[0]

    def __call__(
        self,
        x: Float[Array, "batch sequence channels"],
    ) -> Float[Array, "batch sequence channels"]:
        # alpha is (channels,), broadcast to (1, 1, channels) for (batch, seq, channels) input
        alpha = self.alpha[None, None, :]
        # Snake activation: x + (1/alpha) * sin^2(alpha * x)
        return x + jnp.reciprocal(alpha + 1e-9) * jnp.square(jnp.sin(alpha * x))

    def export_weights(self) -> ParameterTree[Array]:
        return {"alpha": self.alpha}

    def import_weights(self, weights: ParameterTree[Array]) -> "Snake1d":
        assert isinstance(weights, Mapping)
        assert isinstance(weights["alpha"], Array)
        return replace(self, alpha=weights["alpha"])


@dataclass(frozen=True)
class SnakeBetaConfig:
    precision: DTypeLike
    alpha_init: float = 1.0
    no_div_by_zero: float = 1e-9

    def empty(self, channels: int) -> "SnakeBeta":
        alpha = jnp.full((channels,), self.alpha_init, dtype=self.precision)
        beta = jnp.full((channels,), self.alpha_init, dtype=self.precision)
        return SnakeBeta(config=self, alpha=alpha, beta=beta)

    def random_init(self, channels: int) -> "SnakeBeta":
        return self.empty(channels)


class SnakeBeta(LalamoModule[SnakeBetaConfig]):
    alpha: Float[Array, " channels"]
    beta: Float[Array, " channels"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        x: Float[Array, "batch tokens channels"],
    ) -> Float[Array, "batch tokens channels"]:
        alpha = jnp.exp(self.alpha)[None, None, :]
        beta = jnp.exp(self.beta)[None, None, :]
        return x + (jnp.reciprocal(beta + self.config.no_div_by_zero) * jnp.square(jnp.sin(x * alpha)))

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            alpha=require_array(weights["alpha"]),
            beta=require_array(weights["beta"]),
        )


SnakeActivation = Snake1d | SnakeBeta
SnakeActivationConfig = Snake1dConfig | SnakeBetaConfig
register_config_union(SnakeActivationConfig)


@dataclass(frozen=True)
class ConvNeXtBlockConfig:
    precision: DTypeLike
    activation: Activation
    conv_config: Conv1dConfig
    norm_config: NormalizationConfig
    linear_config: FullPrecisionLinearConfig
    gamma_init: float | None = None

    def empty(
        self,
        dim: int,
        *,
        kernel_size: int = 7,
        dilation: int = 1,
        mlp_ratio: float = 4.0,
    ) -> "ConvNeXtBlock":
        depthwise_conv = self.conv_config.empty(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=1,
            groups=dim,
        )
        norm = self.norm_config.empty(dim)
        hidden_dim = int(mlp_ratio * dim)
        pointwise_conv1 = self.linear_config.empty(dim, (hidden_dim,), has_biases=True)
        pointwise_conv2 = self.linear_config.empty(hidden_dim, (dim,), has_biases=True)
        gamma = jnp.ones((dim,), dtype=self.precision) * self.gamma_init if self.gamma_init is not None else None
        return ConvNeXtBlock(
            config=self,
            depthwise_conv=depthwise_conv,
            norm=norm,
            pointwise_conv1=pointwise_conv1,
            pointwise_conv2=pointwise_conv2,
            gamma=gamma,
        )

    def random_init(
        self,
        dim: int,
        *,
        kernel_size: int = 7,
        dilation: int = 1,
        mlp_ratio: float = 4.0,
        key: PRNGKeyArray,
    ) -> "ConvNeXtBlock":
        key_dw, key_pw1, key_pw2 = jax.random.split(key, 3)
        depthwise_conv = self.conv_config.random_init(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=1,
            groups=dim,
            key=key_dw,
        )
        norm = self.norm_config.init(dim)
        hidden_dim = int(mlp_ratio * dim)
        pointwise_conv1 = self.linear_config.random_init(dim, (hidden_dim,), has_biases=True, key=key_pw1)
        pointwise_conv2 = self.linear_config.random_init(hidden_dim, (dim,), has_biases=True, key=key_pw2)
        gamma = jnp.ones((dim,), dtype=self.precision) * self.gamma_init if self.gamma_init is not None else None
        return ConvNeXtBlock(
            config=self,
            depthwise_conv=depthwise_conv,
            norm=norm,
            pointwise_conv1=pointwise_conv1,
            pointwise_conv2=pointwise_conv2,
            gamma=gamma,
        )


class ConvNeXtBlock(LalamoModule[ConvNeXtBlockConfig]):
    depthwise_conv: Conv1d
    norm: Normalization
    pointwise_conv1: FullPrecisionLinear
    pointwise_conv2: FullPrecisionLinear
    gamma: Float[Array, " channels"] | None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        x: Float[Array, "batch tokens channels"],
    ) -> Float[Array, "batch tokens channels"]:
        residual = x
        x = self.depthwise_conv(x)
        x = vmap(vmap(self.norm))(x)
        (x,) = vmap(vmap(self.pointwise_conv1))(x)
        x = vmap(vmap(self.config.activation))(x)
        (x,) = vmap(vmap(self.pointwise_conv2))(x)
        if self.gamma is not None:
            x = x * self.gamma[None, None, :]
        return residual + x

    def export_weights(self) -> ParameterTree[Array]:
        result: dict[str, Array | ParameterTree[Array]] = {
            "depthwise_conv": self.depthwise_conv.export_weights(),
            "norm": self.norm.export_weights(),
            "pointwise_conv1": self.pointwise_conv1.export_weights(),
            "pointwise_conv2": self.pointwise_conv2.export_weights(),
        }
        if self.gamma is not None:
            result["gamma"] = self.gamma
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        gamma = require_array(weights["gamma"]) if self.gamma is not None else None
        return replace(
            self,
            depthwise_conv=self.depthwise_conv.import_weights(require_tree(weights["depthwise_conv"])),
            norm=self.norm.import_weights(require_tree(weights["norm"])),
            pointwise_conv1=self.pointwise_conv1.import_weights(require_tree(weights["pointwise_conv1"])),
            pointwise_conv2=self.pointwise_conv2.import_weights(require_tree(weights["pointwise_conv2"])),
            gamma=gamma,
        )


@dataclass(frozen=True)
class ResidualUnitSpatialParams:
    dilation: int = 1
    kernel_size: int = 7


@dataclass(frozen=True)
class ResidualUnitConfig:
    precision: DTypeLike
    snake_config: SnakeActivationConfig
    conv_config: Conv1dConfig
    causal: bool = True

    def empty(
        self,
        dim: int,
        spatial_params: ResidualUnitSpatialParams,
    ) -> "ResidualUnit":
        if not self.causal:
            raise NotImplementedError("Non-causal ResidualUnit is not implemented")

        act1 = self.snake_config.empty(dim)
        conv1 = self.conv_config.empty(
            in_channels=dim,
            out_channels=dim,
            kernel_size=spatial_params.kernel_size,
            stride=1,
            dilation=spatial_params.dilation,
            groups=1,
        )
        act2 = self.snake_config.empty(dim)
        conv2 = self.conv_config.empty(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            dilation=1,
            groups=1,
        )

        return ResidualUnit(config=self, act1=act1, conv1=conv1, act2=act2, conv2=conv2)

    def random_init(
        self,
        dim: int,
        spatial_params: ResidualUnitSpatialParams,
        *,
        key: PRNGKeyArray,
    ) -> "ResidualUnit":
        if not self.causal:
            raise NotImplementedError("Non-causal ResidualUnit is not implemented")

        key1, key2 = jax.random.split(key, 2)

        act1 = self.snake_config.random_init(dim)
        conv1 = self.conv_config.random_init(
            in_channels=dim,
            out_channels=dim,
            kernel_size=spatial_params.kernel_size,
            stride=1,
            dilation=spatial_params.dilation,
            groups=1,
            key=key1,
        )
        act2 = self.snake_config.random_init(dim)
        conv2 = self.conv_config.random_init(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            dilation=1,
            groups=1,
            key=key2,
        )

        return ResidualUnit(config=self, act1=act1, conv1=conv1, act2=act2, conv2=conv2)


class ResidualUnit(LalamoModule[ResidualUnitConfig]):
    act1: SnakeActivation
    conv1: Conv1d
    act2: SnakeActivation
    conv2: Conv1d

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        x: Float[Array, "batch sequence channels"],
    ) -> Float[Array, "batch sequence channels"]:
        y = self.act1(x)
        y = self.conv1(y)
        y = self.act2(y)
        y = self.conv2(y)

        pad = x.shape[1] - y.shape[1]
        if pad > 0:
            x = x[:, :-pad, :]

        return x + y

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "act1": self.act1.export_weights(),
            "conv1": self.conv1.export_weights(),
            "act2": self.act2.export_weights(),
            "conv2": self.conv2.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            act1=self.act1.import_weights(require_tree(weights["act1"])),
            conv1=self.conv1.import_weights(require_tree(weights["conv1"])),
            act2=self.act2.import_weights(require_tree(weights["act2"])),
            conv2=self.conv2.import_weights(require_tree(weights["conv2"])),
        )


@dataclass(frozen=True)
class AudioDecoderBlockSpatialParams:
    input_dim: int
    output_dim: int
    stride: int


@dataclass(frozen=True)
class DecoderBlockConfig:
    precision: DTypeLike
    snake_config: SnakeActivationConfig
    trans_conv_config: CausalTransposeConv1dConfig
    res_unit_config: ResidualUnitConfig
    causal: bool = True

    def empty(
        self,
        spatial_params: AudioDecoderBlockSpatialParams,
    ) -> "DecoderBlock":
        input_dim = spatial_params.input_dim
        output_dim = spatial_params.output_dim
        stride = spatial_params.stride

        snake = self.snake_config.empty(input_dim)
        trans_conv = self.trans_conv_config.empty(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=2 * stride,
            stride=stride,
        )
        residual_units = (
            self.res_unit_config.empty(output_dim, ResidualUnitSpatialParams(dilation=1)),
            self.res_unit_config.empty(output_dim, ResidualUnitSpatialParams(dilation=3)),
            self.res_unit_config.empty(output_dim, ResidualUnitSpatialParams(dilation=9)),
        )

        return DecoderBlock(config=self, snake=snake, trans_conv=trans_conv, residual_units=residual_units)

    def random_init(
        self,
        spatial_params: AudioDecoderBlockSpatialParams,
        *,
        key: PRNGKeyArray,
    ) -> "DecoderBlock":
        input_dim = spatial_params.input_dim
        output_dim = spatial_params.output_dim
        stride = spatial_params.stride

        key1, key2, key3, key4 = jax.random.split(key, 4)

        snake = self.snake_config.random_init(input_dim)
        trans_conv = self.trans_conv_config.random_init(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=2 * stride,
            stride=stride,
            key=key1,
        )
        residual_units = (
            self.res_unit_config.random_init(output_dim, ResidualUnitSpatialParams(dilation=1), key=key2),
            self.res_unit_config.random_init(output_dim, ResidualUnitSpatialParams(dilation=3), key=key3),
            self.res_unit_config.random_init(output_dim, ResidualUnitSpatialParams(dilation=9), key=key4),
        )

        return DecoderBlock(config=self, snake=snake, trans_conv=trans_conv, residual_units=residual_units)


class DecoderBlock(LalamoModule[DecoderBlockConfig]):
    snake: SnakeActivation
    trans_conv: CausalTransposeConv1d
    residual_units: tuple[ResidualUnit, ...]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def input_dim(self) -> int:
        return self.trans_conv.in_channels

    @property
    def output_dim(self) -> int:
        return self.trans_conv.out_channels

    def __call__(
        self,
        x: Float[Array, "batch sequence in_channels"],
    ) -> Float[Array, "batch sequence_out out_channels"]:
        x = self.snake(x)
        x = self.trans_conv(x)
        for residual_unit in self.residual_units:
            x = residual_unit(x)
        return x

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "snake": self.snake.export_weights(),
            "trans_conv": self.trans_conv.export_weights(),
            "residual_units": [u.export_weights() for u in self.residual_units],
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        residual_unit_weights = weights["residual_units"]
        assert isinstance(residual_unit_weights, Sequence)
        return replace(
            self,
            snake=self.snake.import_weights(require_tree(weights["snake"])),
            trans_conv=self.trans_conv.import_weights(require_tree(weights["trans_conv"])),
            residual_units=tuple(
                u.import_weights(require_tree(w))
                for u, w in zip(self.residual_units, residual_unit_weights, strict=True)
            ),
        )


@dataclass(frozen=True)
class TransposeConvSpatialParams:
    in_channels: int
    out_channels: int
    upsample_kernel_size: int
    upsample_stride: int


@dataclass(frozen=True)
class ConvNeXtSpatialParams:
    mlp_ratio: float = 4.0
    kernel_size: int = 7
    dilation: int = 1


@dataclass(frozen=True)
class UpsamplingBlockConfig:
    precision: DTypeLike
    trans_conv_config: CausalTransposeConv1dConfig
    convnext_config: ConvNeXtBlockConfig

    def random_init(
        self,
        trans_conv_params: TransposeConvSpatialParams,
        convnext_spatial_params: ConvNeXtSpatialParams,
        *,
        key: PRNGKeyArray,
    ) -> "UpsamplingBlock":
        key1, key2 = jax.random.split(key)

        trans_conv = self.trans_conv_config.random_init(
            in_channels=trans_conv_params.in_channels,
            out_channels=trans_conv_params.out_channels,
            kernel_size=trans_conv_params.upsample_kernel_size,
            stride=trans_conv_params.upsample_stride,
            key=key1,
        )

        convnext = self.convnext_config.random_init(
            dim=trans_conv_params.out_channels,
            kernel_size=convnext_spatial_params.kernel_size,
            dilation=convnext_spatial_params.dilation,
            mlp_ratio=convnext_spatial_params.mlp_ratio,
            key=key2,
        )

        return UpsamplingBlock(config=self, trans_conv=trans_conv, convnext=convnext)

    def empty(
        self,
        trans_conv_params: TransposeConvSpatialParams,
        convnext_spatial_params: ConvNeXtSpatialParams,
    ) -> "UpsamplingBlock":
        trans_conv = self.trans_conv_config.empty(
            in_channels=trans_conv_params.in_channels,
            out_channels=trans_conv_params.out_channels,
            kernel_size=trans_conv_params.upsample_kernel_size,
            stride=trans_conv_params.upsample_stride,
        )

        convnext = self.convnext_config.empty(
            dim=trans_conv_params.out_channels,
            kernel_size=convnext_spatial_params.kernel_size,
            dilation=convnext_spatial_params.dilation,
            mlp_ratio=convnext_spatial_params.mlp_ratio,
        )

        return UpsamplingBlock(config=self, trans_conv=trans_conv, convnext=convnext)


class UpsamplingBlock(LalamoModule[UpsamplingBlockConfig]):
    trans_conv: CausalTransposeConv1d
    convnext: ConvNeXtBlock

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def in_channels(self) -> int:
        return self.trans_conv.in_channels

    @property
    def out_channels(self) -> int:
        return self.trans_conv.out_channels

    def __call__(
        self,
        x: Float[Array, "batch sequence in_channels"],
    ) -> Float[Array, "batch sequence_out out_channels"]:
        x = self.trans_conv(x)
        x = self.convnext(x)
        return x

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "trans_conv": self.trans_conv.export_weights(),
            "convnext": self.convnext.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> "UpsamplingBlock":
        assert isinstance(weights, Mapping)
        return replace(
            self,
            trans_conv=self.trans_conv.import_weights(require_tree(weights["trans_conv"])),
            convnext=self.convnext.import_weights(require_tree(weights["convnext"])),
        )


@dataclass(frozen=True)
class DACDecoderSpatialParams:
    input_channel: int
    channels: int
    rates: tuple[int, ...]
    d_out: int = 1


@dataclass(frozen=True)
class DACDecoderConfig:
    precision: DTypeLike
    conv_config: Conv1dConfig
    snake_config: SnakeActivationConfig
    decoder_block_config: DecoderBlockConfig
    causal: bool = True

    def empty(
        self,
        spatial_params: DACDecoderSpatialParams,
    ) -> "DACDecoder":
        if not self.causal:
            raise NotImplementedError("Non-causal AudioDecoder is not implemented")

        input_channel = spatial_params.input_channel
        channels = spatial_params.channels
        rates = spatial_params.rates
        d_out = spatial_params.d_out

        first_conv = self.conv_config.empty(
            in_channels=input_channel,
            out_channels=channels,
            kernel_size=7,
            stride=1,
            dilation=1,
            groups=1,
        )

        decoder_blocks: list[DecoderBlock] = []
        for i, stride in enumerate(rates):
            block_input_dim = channels // (2**i)
            block_output_dim = channels // (2 ** (i + 1))

            block_spatial = AudioDecoderBlockSpatialParams(
                input_dim=block_input_dim,
                output_dim=block_output_dim,
                stride=stride,
            )
            block = self.decoder_block_config.empty(spatial_params=block_spatial)
            decoder_blocks.append(block)

        final_dim = channels // (2 ** len(rates))

        final_snake = self.snake_config.empty(final_dim)

        final_conv = self.conv_config.empty(
            in_channels=final_dim,
            out_channels=d_out,
            kernel_size=7,
            stride=1,
            dilation=1,
            groups=1,
        )

        return DACDecoder(
            config=self,
            first_conv=first_conv,
            decoder_blocks=tuple(decoder_blocks),
            final_snake=final_snake,
            final_conv=final_conv,
        )

    def random_init(
        self,
        spatial_params: DACDecoderSpatialParams,
        *,
        key: PRNGKeyArray,
    ) -> "DACDecoder":
        if not self.causal:
            raise NotImplementedError("Non-causal AudioDecoder is not implemented")

        input_channel = spatial_params.input_channel
        channels = spatial_params.channels
        rates = spatial_params.rates
        d_out = spatial_params.d_out

        num_keys = 2 + len(rates)
        keys = jax.random.split(key, num_keys)

        first_conv = self.conv_config.random_init(
            in_channels=input_channel,
            out_channels=channels,
            kernel_size=7,
            stride=1,
            dilation=1,
            groups=1,
            key=keys[0],
        )

        decoder_blocks: list[DecoderBlock] = []
        for i, stride in enumerate(rates):
            block_input_dim = channels // (2**i)
            block_output_dim = channels // (2 ** (i + 1))

            block_spatial = AudioDecoderBlockSpatialParams(
                input_dim=block_input_dim,
                output_dim=block_output_dim,
                stride=stride,
            )
            block = self.decoder_block_config.random_init(spatial_params=block_spatial, key=keys[1 + i])
            decoder_blocks.append(block)

        final_dim = channels // (2 ** len(rates))

        final_snake = self.snake_config.random_init(final_dim)

        final_conv = self.conv_config.random_init(
            in_channels=final_dim,
            out_channels=d_out,
            kernel_size=7,
            stride=1,
            dilation=1,
            groups=1,
            key=keys[-1],
        )

        return DACDecoder(
            config=self,
            first_conv=first_conv,
            decoder_blocks=tuple(decoder_blocks),
            final_snake=final_snake,
            final_conv=final_conv,
        )


class DACDecoder(LalamoModule[DACDecoderConfig]):
    first_conv: Conv1d
    decoder_blocks: tuple[DecoderBlock, ...]
    final_snake: SnakeActivation
    final_conv: Conv1d

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def input_channels(self) -> int:
        return self.first_conv.in_channels

    @property
    def output_channels(self) -> int:
        return self.final_conv.out_channels

    @property
    def num_blocks(self) -> int:
        return len(self.decoder_blocks)

    def __call__(
        self,
        x: Float[Array, "batch sequence in_channels"],
    ) -> Float[Array, "batch sequence_out out_channels"]:
        x = self.first_conv(x)

        for block in self.decoder_blocks:
            x = block(x)

        x = self.final_snake(x)
        x = self.final_conv(x)

        return x

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "first_conv": self.first_conv.export_weights(),
            "decoder_blocks": [block.export_weights() for block in self.decoder_blocks],
            "final_snake": self.final_snake.export_weights(),
            "final_conv": self.final_conv.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights = require_mapping(weights)
        block_weights = weights["decoder_blocks"]
        assert isinstance(block_weights, Sequence)

        return replace(
            self,
            first_conv=self.first_conv.import_weights(require_tree(weights["first_conv"])),
            decoder_blocks=tuple(
                block.import_weights(require_tree(w))
                for block, w in zip(self.decoder_blocks, block_weights, strict=True)
            ),
            final_snake=self.final_snake.import_weights(require_tree(weights["final_snake"])),
            final_conv=self.final_conv.import_weights(require_tree(weights["final_conv"])),
        )
