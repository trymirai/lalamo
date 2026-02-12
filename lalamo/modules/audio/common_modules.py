import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, DTypeLike, Float, PRNGKeyArray

from lalamo.common import ParameterTree, dummy_array
from lalamo.modules.common import LalamoModule


def _get_extra_padding_for_conv1d(length: int, kernel_size: int, stride: int, padding_total: int = 0) -> int:
    """Calculate extra padding needed to ensure output length is correct."""
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return int(ideal_length - length)


@dataclass(frozen=True)
class CausalConv1dConfig:
    precision: DTypeLike
    has_biases: bool

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
    ) -> "CausalConv1d":
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

        return CausalConv1d(
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
    ) -> "CausalConv1d":
        effective_kernel_size = (kernel_size - 1) * dilation + 1

        weights = dummy_array(
            (out_channels, in_channels // groups, kernel_size),
            dtype=self.precision,
        )

        if self.has_biases:
            biases = dummy_array((out_channels,), dtype=self.precision)
        else:
            biases = None

        return CausalConv1d(
            config=self,
            weights=weights,
            biases=biases,
            stride=stride,
            dilation=dilation,
            groups=groups,
            effective_kernel_size=effective_kernel_size,
        )


class CausalConv1d(LalamoModule[CausalConv1dConfig]):
    """Causal 1D convolution module.
    Implements causal convolution by left-padding the input with zeros.
    The output at position t only depends on inputs at positions <= t.
    """

    weights: Float[Array, "out_channels in_channels_per_group kernel_size"]
    biases: Float[Array, " out_channels"] | None

    stride: int = eqx.field(static=True)
    dilation: int = eqx.field(static=True)
    groups: int = eqx.field(static=True)
    effective_kernel_size: int = eqx.field(static=True)

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
    def padding(self) -> int:
        return self.effective_kernel_size - self.stride

    def __call__(
        self,
        x: Float[Array, "batch sequence in_channels"],
    ) -> Float[Array, "batch sequence_out out_channels"]:
        length = x.shape[1]  # sequence dimension is axis 1
        pad = self.padding
        extra_padding = _get_extra_padding_for_conv1d(length, self.effective_kernel_size, self.stride, pad)

        # Pad on the left (causal padding) and extra on the right if needed
        # Input is (batch, sequence, channels), pad the sequence dimension (axis 1)
        x_padded = jnp.pad(x, ((0, 0), (pad, extra_padding), (0, 0)), mode="constant", constant_values=0)

        # Input: (N, S, C) - NSC format
        # Kernel: (O, I, K) - OIH format
        # Output: (N, S, C) - NSC format
        output = lax.conv_general_dilated(
            x_padded,
            self.weights,
            window_strides=(self.stride,),
            padding="VALID",
            rhs_dilation=(self.dilation,),
            feature_group_count=self.groups,
            dimension_numbers=("NHC", "OIH", "NHC"),
        )

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
    # TODO(peter.glushkov):  Once FishAudio is merged, add groups support to
    # fishaudio's CausalTransposeConv1d and reuse it here instead of this variant.

    """Configuration for CausalTransposeConv1d with groups support.

    This is a causal transposed 1D convolution (deconvolution) that removes
    padding from the output to maintain causality. Supports grouped convolutions.

    Weight format: (out_channels, in_channels // groups, kernel_size) - JAX OIK format
    with kernel already flipped for transposed convolution.

    When importing from PyTorch, weights must be transformed from PyTorch format
    (in_channels, out_channels // groups, kernel_size) to JAX format using
    `transform_pytorch_transpose_conv_weights()`.

    Args:
        precision: Data type for computations.
        has_biases: Whether to include bias terms.
    """

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

    Input format: (batch, sequence, channels) - NSC format (JAX convention)
    Output format: (batch, sequence_out, channels) - NSC format (JAX convention)

    Weight format: (out_channels, in_channels // groups, kernel_size) - JAX OIK format
    with kernel already flipped for transposed convolution.

    Reference: NVIDIA NeMo CausalConvTranspose1dNorm
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

    Input format: (batch, sequence, channels) - NSC format (JAX convention)
    Output format: (batch, sequence, channels) - NSC format (JAX convention)
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
