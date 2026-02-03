"""Lalamo implementation of NanoCodec modules from NVIDIA NeMo.

This module implements the Finite Scalar Quantization (FSQ) components
and activation functions from NVIDIA's NanoCodec.

Reference: https://github.com/NVIDIA-NeMo/NeMo/blob/v2.3.0/nemo/collections/tts/modules/audio_codec_modules.py
Reference: Mentzer et al., Finite Scalar Quantization: VQ-VAE Made Simple (https://arxiv.org/abs/2309.15505v1)
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, dummy_array, require_array, require_tree
from lalamo.modules.audio.fishaudio.fishaudio_modules import (
    CausalConv1d,
    CausalConv1dConfig,
    Snake1d,
    Snake1dConfig,
)
from lalamo.modules.common import LalamoModule

__all__ = [
    "CausalHiFiGANDecoder",
    "CausalHiFiGANDecoderConfig",
    "CausalTransposeConv1d",
    "CausalTransposeConv1dConfig",
    "FiniteScalarQuantizer",
    "FiniteScalarQuantizerConfig",
    "GroupFiniteScalarQuantizer",
    "GroupFiniteScalarQuantizerConfig",
    "HalfSnake",
    "HalfSnakeConfig",
    "HiFiGANResBlock",
    "HiFiGANResBlockConfig",
    "HiFiGANResLayer",
    "HiFiGANResLayerConfig",
    "ResidualBlock",
    "ResidualBlockConfig",
]


# =============================================================================
# Finite Scalar Quantization (FSQ)
# =============================================================================


@dataclass(frozen=True)
class FiniteScalarQuantizerConfig:
    """Configuration for Finite Scalar Quantizer.

    FSQ quantizes each element of the input vector independently into discrete levels.
    Unlike traditional VQ-VAE which learns a codebook, FSQ uses implicit quantization
    based on the number of levels per dimension.

    Args:
        num_levels: Number of quantization levels for each dimension.
        eps: Small regularization constant for scaling to avoid boundary issues.
        precision: Data type for computations.
    """

    num_levels: tuple[int, ...]
    eps: float = 1e-3
    precision: DTypeLike = jnp.float32

    @property
    def dim(self) -> int:
        """Dimension of the input/output vectors."""
        return len(self.num_levels)

    @property
    def codebook_size(self) -> int:
        """Size of the implicit codebook."""
        result = 1
        for level in self.num_levels:
            result *= level
        return result

    def empty(self) -> "FiniteScalarQuantizer":
        """Create quantizer with buffer arrays.

        Note: FSQ has no learnable weights, only precomputed buffers.
        """
        num_levels = jnp.array(self.num_levels, dtype=jnp.int32)
        # dim_base_index = cumprod([1, levels[0], levels[0]*levels[1], ...])
        # Used to convert per-dimension indices to a single codebook index
        dim_base_index = jnp.cumprod(jnp.concatenate([jnp.array([1], dtype=jnp.int32), num_levels[:-1]]))
        return FiniteScalarQuantizer(
            config=self,
            num_levels_buffer=num_levels,
            dim_base_index=dim_base_index,
        )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002 (unused - FSQ has no learnable weights)
    ) -> "FiniteScalarQuantizer":
        """Create quantizer (FSQ has no learnable weights)."""
        return self.empty()


class FiniteScalarQuantizer(LalamoModule[FiniteScalarQuantizerConfig]):
    """Finite Scalar Quantizer.

    Quantizes each element of the input vector independently into a number of levels.
    Uses tanh compression and rounding with straight-through estimator for gradients.

    Input shape: [batch, dim, seq] (NCT format like PyTorch)
    Output indices shape: [1, batch, seq] (codebook dimension added for RVQ compatibility)

    Reference: Mentzer et al., Finite Scalar Quantization: VQ-VAE Made Simple
    """

    num_levels_buffer: Int[Array, " dim"]
    dim_base_index: Int[Array, " dim"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def dim(self) -> int:
        return self.config.dim

    @property
    def codebook_size(self) -> int:
        return self.config.codebook_size

    def _compress(self, inputs: Float[Array, "batch dim seq"]) -> Float[Array, "batch dim seq"]:
        """Apply tanh compression to map continuous values to quantization range.

        The compression ensures values are bounded within the quantization levels.
        For a dimension with L levels, outputs are mapped to range [-(L-1)/2, (L-1)/2]
        with appropriate handling for even/odd number of levels.
        """
        num_levels = self.num_levels_buffer.astype(self.config.precision)

        # Output scale: (num_levels - 1) / 2
        # Scaled down slightly to avoid rounding issues at boundaries
        output_scale = (num_levels - 1) / 2
        output_scale = output_scale * (1 - self.config.eps)

        # Offset for even number of levels (shifts grid by 0.5)
        output_offset = jnp.where(self.num_levels_buffer % 2 == 0, 0.5, 0.0)

        # Input shift for even levels (pre-tanh adjustment)
        input_shift = jnp.tan(output_offset / output_scale)

        # Apply compression: scale * tanh(input + shift) - offset
        output = (
            output_scale[None, :, None] * jnp.tanh(inputs + input_shift[None, :, None]) - output_offset[None, :, None]
        )
        return output

    def _round_ste(self, inputs: Float[Array, "batch dim seq"]) -> Float[Array, "batch dim seq"]:
        """Round to nearest integer with straight-through estimator.

        The gradient flows through as if no rounding occurred.
        """
        rounded = jnp.round(inputs)
        # Straight-through: gradient of rounded w.r.t inputs = 1
        return inputs + jax.lax.stop_gradient(rounded - inputs)

    def _inputs_to_codes(self, inputs: Float[Array, "batch dim seq"]) -> Float[Array, "batch dim seq"]:
        """Convert continuous inputs to quantized codes normalized to [-1, 1].

        Steps:
        1. Compress inputs using tanh to bound them
        2. Round to nearest integer level
        3. Normalize to [-1, 1] range
        """
        compressed = self._compress(inputs)
        codes = self._round_ste(compressed)
        # Normalize to [-1, 1]
        scale = (self.num_levels_buffer // 2).astype(self.config.precision)
        codes = codes / scale[None, :, None]
        return codes

    def _codes_to_nonnegative(self, codes: Float[Array, "batch dim seq"]) -> Float[Array, "batch dim seq"]:
        """Convert codes centered around zero to nonnegative integer indices."""
        scale = (self.num_levels_buffer // 2).astype(self.config.precision)
        offset = scale
        return scale[None, :, None] * codes + offset[None, :, None]

    def _nonnegative_to_codes(self, codes_nonneg: Float[Array, "dim"]) -> Float[Array, "dim"]:
        """Convert nonnegative indices back to codes centered around zero."""
        scale = (self.num_levels_buffer // 2).astype(self.config.precision)
        offset = scale
        return (codes_nonneg - offset) / scale

    def _codes_to_indices(self, codes: Float[Array, "batch dim seq"]) -> Int[Array, "batch seq"]:
        """Convert per-dimension code vectors to single indices.

        Uses the dim_base_index to compute: sum(code_d * base_d) for d in dimensions
        """
        nonneg = self._codes_to_nonnegative(codes)
        # Sum over dimensions weighted by base index
        indices = jnp.sum(nonneg * self.dim_base_index[None, :, None].astype(self.config.precision), axis=1)
        return indices.astype(jnp.int32)

    def _indices_to_codes(self, index: Int[Array, " 1"]) -> Float[Array, " dim"]:
        """Convert single indices to per-dimension code vectors.
        Reverses the indexing: code_d = (index // base_d) % levels_d
        """

        codes_nonnegative = (index // self.dim_base_index) % self.num_levels_buffer
        return self._nonnegative_to_codes(codes_nonnegative.astype(self.config.precision))

    def encode(
        self,
        inputs: Float[Array, "batch dim seq"],
    ) -> Int[Array, "batch seq"]:
        """Encode continuous inputs to discrete indices.

        Args:
            inputs: Input tensor of shape [batch, dim, seq]

        Returns:
            Indices tensor of shape [1, batch, seq] (1 codebook for RVQ compatibility)
        """
        codes = self._inputs_to_codes(inputs)
        indices = self._codes_to_indices(codes)
        # Add codebook dimension for compatibility with RVQ API
        return indices

    def decode(
        self,
        indices: Int[Array, " seq"],
    ) -> Float[Array, "seq dim"]:
        """Decode discrete indices back to continuous code vectors.

        Args:
            indices: Indices tensor of shape [1, batch, seq] (1 codebook)

        Returns:
            Decoded codes tensor of shape [batch, dim, seq] in range [-1, 1]
        """
        return jax.vmap(self._indices_to_codes)(indices)

    def __call__(
        self,
        inputs: Float[Array, "batch seq"],
    ) -> Float[Array, "batch seq dim"]:
        """
        Forward pass: dequantize batch of input indices vectors to continous representation.
        """
        return jax.vmap(self.decode, in_axes=0)(inputs)

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "num_levels_buffer": self.num_levels_buffer,
            "dim_base_index": self.dim_base_index,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            num_levels_buffer=require_array(weights["num_levels_buffer"]),
            dim_base_index=require_array(weights["dim_base_index"]),
        )


@dataclass(frozen=True)
class GroupFiniteScalarQuantizerConfig:
    """Configuration for Group Finite Scalar Quantizer.

    Splits the input vector into groups and applies FSQ on each group separately.
    This allows for hierarchical quantization where different groups can be
    decoded independently.

    Args:
        num_groups: Number of groups to split the input into.
        quantizer_config: Configuration for the FSQ applied to each group.

    Reference: Yang et al, HiFi-Codec: Group-residual Vector quantization for High Fidelity Audio Codec
    """

    num_groups: int
    quantizer_config: FiniteScalarQuantizerConfig

    @property
    def precision(self) -> DTypeLike:
        return self.quantizer_config.precision

    @property
    def codebook_dim_per_group(self) -> int:
        """Dimension per group (number of levels dimensions)."""
        return self.quantizer_config.dim

    @property
    def codebook_dim(self) -> int:
        """Total input dimension (groups * dim_per_group)."""
        return self.codebook_dim_per_group * self.num_groups

    @property
    def codebook_size_per_group(self) -> int:
        """Codebook size for each group (product of levels)."""
        return self.quantizer_config.codebook_size

    def empty(self) -> "GroupFiniteScalarQuantizer":
        """Create group quantizer with FSQ for each group."""
        quantizers = tuple(self.quantizer_config.empty() for _ in range(self.num_groups))
        return GroupFiniteScalarQuantizer(config=self, quantizers=quantizers)

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002 (unused - FSQ has no learnable weights)
    ) -> "GroupFiniteScalarQuantizer":
        """Create group quantizer (FSQ has no learnable weights)."""
        return self.empty()


class GroupFiniteScalarQuantizer(LalamoModule[GroupFiniteScalarQuantizerConfig]):
    """Group Finite Scalar Quantizer.

    Splits the input vector into groups and applies FSQ on each group separately.
    This enables hierarchical/group-wise quantization where each group produces
    independent indices.

    Input shape: [batch, channels, seq] where channels = num_groups * codebook_dim_per_group
    Output indices shape: [num_groups, batch, seq]

    Reference: Yang et al, HiFi-Codec: Group-residual Vector quantization for High Fidelity Audio Codec
    """

    quantizers: tuple[FiniteScalarQuantizer, ...]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def num_groups(self) -> int:
        return self.config.num_groups

    @property
    def codebook_dim(self) -> int:
        return self.config.codebook_dim

    @property
    def codebook_dim_per_group(self) -> int:
        return self.config.codebook_dim_per_group

    def encode(
        self,
        inputs: Float[Array, "batch channels seq"],
    ) -> Int[Array, "num_groups batch seq"]:
        """Encode inputs to indices for each group.

        Args:
            inputs: Input tensor of shape [batch, channels, seq]

        Returns:
            Indices tensor of shape [num_groups, batch, seq]
        """
        # Split input along channel dimension into groups
        inputs_grouped = jnp.split(inputs, self.num_groups, axis=1)

        indices_list = []
        for in_group, quantizer in zip(inputs_grouped, self.quantizers, strict=True):
            idx = quantizer.encode(in_group)[None, :, :]  # [1, batch, seq]
            indices_list.append(idx)

        # Concatenate along codebook/group dimension
        return jnp.concatenate(indices_list, axis=0)  # [num_groups, batch, seq]

    def decode(
        self,
        indices: Int[Array, "batch seq num_groups"],
    ) -> Float[Array, "batch seq channels"]:
        """Decode batch of indices vectors back to continuous representation."""
        # # Split indices along group dimension
        indices_grouped = jnp.split(indices, self.num_groups, axis=2)

        dequantized_list = []
        for idx_group, quantizer in zip(indices_grouped, self.quantizers, strict=True):
            deq = quantizer(idx_group)
            dequantized_list.append(deq)

        return jnp.concatenate(dequantized_list, axis=2)

    def __call__(
        self,
        inputs: Float[Array, "batch seq num_groups"],
    ) -> Float[Array, "batch seq channels"]:
        return self.decode(inputs)

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "quantizers": [q.export_weights() for q in self.quantizers],
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        quantizer_weights = weights["quantizers"]
        assert isinstance(quantizer_weights, Sequence)

        new_quantizers = tuple(
            q.import_weights(require_tree(w)) for q, w in zip(self.quantizers, quantizer_weights, strict=True)
        )
        return replace(self, quantizers=new_quantizers)


# =============================================================================
# Activation Functions
# =============================================================================


@dataclass(frozen=True)
class HalfSnakeConfig:
    """Configuration for HalfSnake activation.

    HalfSnake applies Snake activation to the first half of channels
    and LeakyReLU to the second half. This is used in NanoCodec's
    CausalHiFiGAN decoder.

    Args:
        snake_config: Configuration for the Snake1d applied to first half.
        leaky_relu_negative_slope: Negative slope for LeakyReLU.
    """

    snake_config: Snake1dConfig
    leaky_relu_negative_slope: float

    @property
    def precision(self) -> DTypeLike:
        return self.snake_config.precision

    def empty(self, channels: int) -> "HalfSnake":
        """Create HalfSnake with placeholder weights.

        Args:
            channels: Total number of input channels. Snake gets floor(channels/2),
                      LeakyReLU gets the rest.
        """
        snake_channels = channels // 2
        snake = self.snake_config.empty(snake_channels)
        return HalfSnake(config=self, snake=snake, total_channels=channels)

    def random_init(
        self,
        channels: int,
        *,
        key: PRNGKeyArray,  # noqa: ARG002 (unused - Snake uses ones initialization)
    ) -> "HalfSnake":
        """Create HalfSnake with initialized weights.

        Args:
            channels: Total number of input channels. Snake gets floor(channels/2),
                      LeakyReLU gets the rest.
            key: PRNG key (unused, Snake uses ones initialization).
        """
        snake_channels = channels // 2
        snake = self.snake_config.random_init(snake_channels)
        return HalfSnake(config=self, snake=snake, total_channels=channels)


class HalfSnake(LalamoModule[HalfSnakeConfig]):
    """HalfSnake activation function.

    Applies Snake activation to the first half of channels and
    LeakyReLU to the second half.

    Input format: (batch, sequence, channels) - NSC format (JAX convention)
    Output format: (batch, sequence, channels) - NSC format (JAX convention)

    Reference: NVIDIA NeMo audio_codec_modules.py
    """

    snake: Snake1d
    total_channels: int

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def channels(self) -> int:
        return self.total_channels

    @property
    def snake_channels(self) -> int:
        return self.total_channels // 2

    def __call__(
        self,
        x: Float[Array, "batch sequence channels"],
    ) -> Float[Array, "batch sequence channels"]:
        # Split along channel dimension (axis=2 for NSC format)
        snake_input = x[:, :, : self.snake_channels]
        lrelu_input = x[:, :, self.snake_channels :]

        # Apply Snake to first half
        snake_out = self.snake(snake_input)

        # Apply LeakyReLU to second half
        negative_slope = self.config.leaky_relu_negative_slope
        lrelu_out = jnp.where(lrelu_input >= 0, lrelu_input, negative_slope * lrelu_input)

        # Concatenate along channel dimension
        return jnp.concatenate([snake_out, lrelu_out], axis=2)

    def export_weights(self) -> ParameterTree[Array]:
        return {"snake": self.snake.export_weights()}

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        snake_weights = require_tree(weights["snake"])
        new_snake = self.snake.import_weights(snake_weights)
        return replace(self, snake=new_snake)


# =============================================================================
# Convolutions
# =============================================================================
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


# =============================================================================
# Residual Blocks
# =============================================================================


@dataclass(frozen=True)
class ResidualBlockConfig:
    """Configuration for ResidualBlock.

    A residual block applies: activation → conv → activation → conv + residual.
    Used in HiFi-GAN decoder architecture.

    Args:
        activation_config: Config for HalfSnake activations (used for both activations).
        conv_config: Config for CausalConv1d layers (used for both convolutions).
    """

    activation_config: HalfSnakeConfig
    conv_config: CausalConv1dConfig

    @property
    def precision(self) -> DTypeLike:
        return self.conv_config.precision

    def empty(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
    ) -> "ResidualBlock":
        """Create ResidualBlock with placeholder weights.

        Args:
            channels: Number of input/output channels (also used as intermediate channels).
            kernel_size: Kernel size for both convolutions.
            dilation: Dilation for the first convolution (second conv uses dilation=1).
        """
        input_activation = self.activation_config.empty(channels)
        skip_activation = self.activation_config.empty(channels)

        input_conv = self.conv_config.empty(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        skip_conv = self.conv_config.empty(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=1,
        )

        return ResidualBlock(
            config=self,
            input_activation=input_activation,
            skip_activation=skip_activation,
            input_conv=input_conv,
            skip_conv=skip_conv,
        )

    def random_init(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        *,
        key: PRNGKeyArray,
    ) -> "ResidualBlock":
        """Create ResidualBlock with randomly initialized weights."""
        input_act_key, skip_act_key, input_conv_key, skip_conv_key = jax.random.split(key, 4)

        input_activation = self.activation_config.random_init(channels, key=input_act_key)
        skip_activation = self.activation_config.random_init(channels, key=skip_act_key)

        input_conv = self.conv_config.random_init(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            key=input_conv_key,
        )
        skip_conv = self.conv_config.random_init(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=1,
            key=skip_conv_key,
        )

        return ResidualBlock(
            config=self,
            input_activation=input_activation,
            skip_activation=skip_activation,
            input_conv=input_conv,
            skip_conv=skip_conv,
        )


class ResidualBlock(LalamoModule[ResidualBlockConfig]):
    """Residual block for HiFi-GAN decoder.

    Applies: activation → conv → activation → conv + residual

    Input format: (batch, sequence, channels) - NSC format (JAX convention)
    Output format: (batch, sequence, channels) - NSC format (JAX convention)

    Reference: NVIDIA NeMo ResidualBlock
    """

    input_activation: HalfSnake
    skip_activation: HalfSnake
    input_conv: CausalConv1d
    skip_conv: CausalConv1d

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def channels(self) -> int:
        return self.input_conv.in_channels

    def __call__(
        self,
        x: Float[Array, "batch sequence channels"],
    ) -> Float[Array, "batch sequence channels"]:
        # activation → conv → activation → conv
        conv_input = self.input_activation(x)
        skip_input = self.input_conv(conv_input)
        skip_input = self.skip_activation(skip_input)
        res = self.skip_conv(skip_input)

        # Residual connection
        return x + res

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "input_activation": self.input_activation.export_weights(),
            "skip_activation": self.skip_activation.export_weights(),
            "input_conv": self.input_conv.export_weights(),
            "skip_conv": self.skip_conv.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)

        input_act_weights = weights["input_activation"]
        skip_act_weights = weights["skip_activation"]
        input_conv_weights = weights["input_conv"]
        skip_conv_weights = weights["skip_conv"]

        assert isinstance(input_act_weights, Mapping)
        assert isinstance(skip_act_weights, Mapping)
        assert isinstance(input_conv_weights, Mapping)
        assert isinstance(skip_conv_weights, Mapping)

        return replace(
            self,
            input_activation=self.input_activation.import_weights(input_act_weights),
            skip_activation=self.skip_activation.import_weights(skip_act_weights),
            input_conv=self.input_conv.import_weights(input_conv_weights),
            skip_conv=self.skip_conv.import_weights(skip_conv_weights),
        )


@dataclass(frozen=True)
class HiFiGANResBlockConfig:
    """Configuration for HiFiGANResBlock.

    Wraps multiple ResidualBlocks with different dilations, applied sequentially.
    Used in HiFi-GAN decoder architecture.

    Args:
        residual_block_config: Config for ResidualBlock layers (shared by all blocks).
    """

    residual_block_config: ResidualBlockConfig

    @property
    def precision(self) -> DTypeLike:
        return self.residual_block_config.precision

    def empty(
        self,
        channels: int,
        kernel_size: int,
        dilations: tuple[int, ...],
    ) -> "HiFiGANResBlock":
        """Create HiFiGANResBlock with placeholder weights.

        Args:
            channels: Number of input/output channels.
            kernel_size: Kernel size for convolutions.
            dilations: Tuple of dilation values, one ResidualBlock per dilation.
        """
        res_blocks = tuple(
            self.residual_block_config.empty(
                channels=channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            for dilation in dilations
        )
        return HiFiGANResBlock(config=self, res_blocks=res_blocks)

    def random_init(
        self,
        channels: int,
        kernel_size: int,
        dilations: tuple[int, ...],
        *,
        key: PRNGKeyArray,
    ) -> "HiFiGANResBlock":
        """Create HiFiGANResBlock with randomly initialized weights."""
        keys = jax.random.split(key, len(dilations))
        res_blocks = tuple(
            self.residual_block_config.random_init(
                channels=channels,
                kernel_size=kernel_size,
                dilation=dilation,
                key=block_key,
            )
            for dilation, block_key in zip(dilations, keys, strict=True)
        )
        return HiFiGANResBlock(config=self, res_blocks=res_blocks)


class HiFiGANResBlock(LalamoModule[HiFiGANResBlockConfig]):
    """HiFiGAN residual block wrapper.

    Creates multiple ResidualBlocks with different dilations and applies them sequentially.

    Input format: (batch, sequence, channels) - NSC format (JAX convention)
    Output format: (batch, sequence, channels) - NSC format (JAX convention)

    Reference: NVIDIA NeMo HiFiGANResBlock
    """

    res_blocks: tuple[ResidualBlock, ...]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def channels(self) -> int:
        return self.res_blocks[0].channels

    def __call__(
        self,
        x: Float[Array, "batch sequence channels"],
    ) -> Float[Array, "batch sequence channels"]:
        out = x
        for res_block in self.res_blocks:
            out = res_block(out)
        return out

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "res_blocks": [block.export_weights() for block in self.res_blocks],
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        block_weights = weights["res_blocks"]
        assert isinstance(block_weights, Sequence)

        new_blocks = tuple(
            block.import_weights(require_tree(w)) for block, w in zip(self.res_blocks, block_weights, strict=True)
        )
        return replace(self, res_blocks=new_blocks)


@dataclass(frozen=True)
class HiFiGANResLayerConfig:
    """Configuration for HiFiGANResLayer.

    Creates multiple HiFiGANResBlocks with different kernel sizes.
    The outputs of all blocks are averaged.

    Args:
        hifigan_res_block_config: Config for HiFiGANResBlock layers (shared by all blocks).
    """

    hifigan_res_block_config: HiFiGANResBlockConfig

    @property
    def precision(self) -> DTypeLike:
        return self.hifigan_res_block_config.precision

    def empty(
        self,
        channels: int,
        kernel_sizes: tuple[int, ...],
        dilations: tuple[int, ...],
    ) -> "HiFiGANResLayer":
        """Create HiFiGANResLayer with placeholder weights.

        Args:
            channels: Number of input/output channels.
            kernel_sizes: Tuple of kernel sizes, one HiFiGANResBlock per kernel size.
            dilations: Tuple of dilations shared by all HiFiGANResBlocks.
        """
        res_blocks = tuple(
            self.hifigan_res_block_config.empty(
                channels=channels,
                kernel_size=kernel_size,
                dilations=dilations,
            )
            for kernel_size in kernel_sizes
        )
        return HiFiGANResLayer(config=self, res_blocks=res_blocks)

    def random_init(
        self,
        channels: int,
        kernel_sizes: tuple[int, ...],
        dilations: tuple[int, ...],
        *,
        key: PRNGKeyArray,
    ) -> "HiFiGANResLayer":
        """Create HiFiGANResLayer with randomly initialized weights."""
        keys = jax.random.split(key, len(kernel_sizes))
        res_blocks = tuple(
            self.hifigan_res_block_config.random_init(
                channels=channels,
                kernel_size=kernel_size,
                dilations=dilations,
                key=block_key,
            )
            for kernel_size, block_key in zip(kernel_sizes, keys, strict=True)
        )
        return HiFiGANResLayer(config=self, res_blocks=res_blocks)


class HiFiGANResLayer(LalamoModule[HiFiGANResLayerConfig]):
    """HiFiGAN residual layer.

    Creates multiple HiFiGANResBlocks with different kernel sizes.
    Each block processes the same input, and their outputs are averaged.

    Input format: (batch, sequence, channels) - NSC format (JAX convention)
    Output format: (batch, sequence, channels) - NSC format (JAX convention)

    Reference: NVIDIA NeMo HiFiGANResLayer
    """

    res_blocks: tuple[HiFiGANResBlock, ...]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def channels(self) -> int:
        return self.res_blocks[0].channels

    def __call__(
        self,
        x: Float[Array, "batch sequence channels"],
    ) -> Float[Array, "batch sequence channels"]:
        residuals = jnp.stack([res_block(x) for res_block in self.res_blocks], axis=0)
        return jnp.mean(residuals, axis=0)

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "res_blocks": [block.export_weights() for block in self.res_blocks],
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        block_weights = weights["res_blocks"]
        assert isinstance(block_weights, Sequence)

        new_blocks = tuple(
            block.import_weights(require_tree(w)) for block, w in zip(self.res_blocks, block_weights, strict=True)
        )
        return replace(self, res_blocks=new_blocks)


@dataclass(frozen=True)
class CausalHiFiGANDecoderConfig:
    """Configuration for CausalHiFiGANDecoder.

    HiFi-GAN decoder with causal convolutions for audio generation.

    Args:
        activation_config: Config for HalfSnake activations (shared by all).
        pre_conv_config: Config for input convolution.
        transpose_conv_config: Config for upsampling transposed convolutions.
        res_layer_config: Config for HiFiGANResLayer blocks.
        post_conv_config: Config for output convolution.
    """

    activation_config: HalfSnakeConfig
    pre_conv_config: CausalConv1dConfig
    transpose_conv_config: CausalTransposeConv1dConfig
    res_layer_config: HiFiGANResLayerConfig
    post_conv_config: CausalConv1dConfig

    @property
    def precision(self) -> DTypeLike:
        return self.pre_conv_config.precision

    def empty(
        self,
        input_dim: int,
        base_channels: int,
        up_sample_rates: tuple[int, ...],
        in_kernel_size: int,
        out_kernel_size: int,
        resblock_kernel_sizes: tuple[int, ...],
        resblock_dilations: tuple[int, ...],
    ) -> "CausalHiFiGANDecoder":
        """Create CausalHiFiGANDecoder with placeholder weights.

        Args:
            input_dim: Input dimension (from quantizer).
            base_channels: Initial number of channels after pre_conv.
            up_sample_rates: Upsample rate for each stage. Channels halve at each stage.
            in_kernel_size: Kernel size for pre_conv.
            out_kernel_size: Kernel size for post_conv.
            resblock_kernel_sizes: Kernel sizes for HiFiGANResLayer.
            resblock_dilations: Dilations for HiFiGANResLayer.
        """
        # Pre-conv: input_dim -> base_channels
        pre_conv = self.pre_conv_config.empty(
            in_channels=input_dim,
            out_channels=base_channels,
            kernel_size=in_kernel_size,
        )

        # Build stages: each stage has activation, upsample conv, res_layer
        activations = []
        upsample_convs = []
        res_layers = []

        in_channels = base_channels
        for up_sample_rate in up_sample_rates:
            out_channels = in_channels // 2
            kernel_size = 2 * up_sample_rate

            # Activation before upsample
            activation = self.activation_config.empty(in_channels)
            activations.append(activation)

            # Upsample conv (groups=out_channels for depthwise-like operation)
            upsample_conv = self.transpose_conv_config.empty(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=up_sample_rate,
                groups=out_channels,
            )
            upsample_convs.append(upsample_conv)

            # Res layer (at reduced channel count)
            res_layer = self.res_layer_config.empty(
                channels=out_channels,
                kernel_sizes=resblock_kernel_sizes,
                dilations=resblock_dilations,
            )
            res_layers.append(res_layer)

            in_channels = out_channels

        # Post activation and conv
        post_activation = self.activation_config.empty(in_channels)
        post_conv = self.post_conv_config.empty(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=out_kernel_size,
        )

        return CausalHiFiGANDecoder(
            config=self,
            pre_conv=pre_conv,
            activations=tuple(activations),
            upsample_convs=tuple(upsample_convs),
            res_layers=tuple(res_layers),
            post_activation=post_activation,
            post_conv=post_conv,
            up_sample_rates=up_sample_rates,
        )

    def random_init(
        self,
        input_dim: int,
        base_channels: int,
        up_sample_rates: tuple[int, ...],
        in_kernel_size: int,
        out_kernel_size: int,
        resblock_kernel_sizes: tuple[int, ...],
        resblock_dilations: tuple[int, ...],
        *,
        key: PRNGKeyArray,
    ) -> "CausalHiFiGANDecoder":
        """Create CausalHiFiGANDecoder with randomly initialized weights."""
        # Split keys for all components
        num_stages = len(up_sample_rates)
        # pre_conv, post_activation, post_conv + 3 per stage (activation, upsample, res_layer)
        num_keys = 3 + 3 * num_stages
        keys = jax.random.split(key, num_keys)
        key_idx = 0

        pre_conv = self.pre_conv_config.random_init(
            in_channels=input_dim,
            out_channels=base_channels,
            kernel_size=in_kernel_size,
            key=keys[key_idx],
        )
        key_idx += 1

        activations = []
        upsample_convs = []
        res_layers = []

        in_channels = base_channels
        for up_sample_rate in up_sample_rates:
            out_channels = in_channels // 2
            kernel_size = 2 * up_sample_rate

            activation = self.activation_config.random_init(in_channels, key=keys[key_idx])
            key_idx += 1
            activations.append(activation)

            upsample_conv = self.transpose_conv_config.random_init(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=up_sample_rate,
                groups=out_channels,
                key=keys[key_idx],
            )
            key_idx += 1
            upsample_convs.append(upsample_conv)

            res_layer = self.res_layer_config.random_init(
                channels=out_channels,
                kernel_sizes=resblock_kernel_sizes,
                dilations=resblock_dilations,
                key=keys[key_idx],
            )
            key_idx += 1
            res_layers.append(res_layer)

            in_channels = out_channels

        post_activation = self.activation_config.random_init(in_channels, key=keys[key_idx])
        key_idx += 1
        post_conv = self.post_conv_config.random_init(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=out_kernel_size,
            key=keys[key_idx],
        )

        return CausalHiFiGANDecoder(
            config=self,
            pre_conv=pre_conv,
            activations=tuple(activations),
            upsample_convs=tuple(upsample_convs),
            res_layers=tuple(res_layers),
            post_activation=post_activation,
            post_conv=post_conv,
            up_sample_rates=up_sample_rates,
        )


class CausalHiFiGANDecoder(LalamoModule[CausalHiFiGANDecoderConfig]):
    """Causal HiFi-GAN decoder for audio generation.

    Converts quantized representations to audio waveforms using:
    1. Pre-conv to expand to base_channels
    2. Multiple stages of: activation -> upsample -> res_layer
    3. Post-conv to produce single-channel audio
    4. Tanh output activation

    Input format: (batch, sequence, channels) - NSC format (JAX convention)
    Output format: (batch, audio_length) - audio waveform

    Reference: NVIDIA NeMo CausalHiFiGANDecoder
    """

    pre_conv: CausalConv1d
    activations: tuple[HalfSnake, ...]
    upsample_convs: tuple[CausalTransposeConv1d, ...]
    res_layers: tuple[HiFiGANResLayer, ...]
    post_activation: HalfSnake
    post_conv: CausalConv1d
    up_sample_rates: tuple[int, ...] = eqx.field(static=True)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        x: Float[Array, "batch sequence channels"],
    ) -> Float[Array, "batch audio_length"]:
        # Pre-conv: [B, S, input_dim] -> [B, S, base_channels]
        out = self.pre_conv(x)

        # Upsample stages
        for activation, upsample_conv, res_layer in zip(
            self.activations,
            self.upsample_convs,
            self.res_layers,
            strict=True,
        ):
            out = activation(out)
            out = upsample_conv(out)
            out = res_layer(out)

        # Post processing
        out = self.post_activation(out)
        out = self.post_conv(out)  # [B, T_audio, 1]

        # Tanh and squeeze channel dimension
        audio = jnp.tanh(out)
        audio = audio[:, :, 0]  # [B, T_audio]

        return audio

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "pre_conv": self.pre_conv.export_weights(),
            "activations": [act.export_weights() for act in self.activations],
            "upsample_convs": [conv.export_weights() for conv in self.upsample_convs],
            "res_layers": [layer.export_weights() for layer in self.res_layers],
            "post_activation": self.post_activation.export_weights(),
            "post_conv": self.post_conv.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)

        pre_conv_weights = require_tree(weights["pre_conv"])
        activations_weights = weights["activations"]
        upsample_convs_weights = weights["upsample_convs"]
        res_layers_weights = weights["res_layers"]
        post_activation_weights = require_tree(weights["post_activation"])
        post_conv_weights = require_tree(weights["post_conv"])

        assert isinstance(activations_weights, Sequence)
        assert isinstance(upsample_convs_weights, Sequence)
        assert isinstance(res_layers_weights, Sequence)

        new_activations = tuple(
            act.import_weights(require_tree(w)) for act, w in zip(self.activations, activations_weights, strict=True)
        )
        new_upsample_convs = tuple(
            conv.import_weights(require_tree(w))
            for conv, w in zip(self.upsample_convs, upsample_convs_weights, strict=True)
        )
        new_res_layers = tuple(
            layer.import_weights(require_tree(w)) for layer, w in zip(self.res_layers, res_layers_weights, strict=True)
        )

        return replace(
            self,
            pre_conv=self.pre_conv.import_weights(pre_conv_weights),
            activations=new_activations,
            upsample_convs=new_upsample_convs,
            res_layers=new_res_layers,
            post_activation=self.post_activation.import_weights(post_activation_weights),
            post_conv=self.post_conv.import_weights(post_conv_weights),
        )
