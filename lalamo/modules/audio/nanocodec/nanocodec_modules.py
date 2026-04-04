"""Lalamo implementation of NanoCodec modules from NVIDIA NeMo.

Reference: https://github.com/NVIDIA-NeMo/NeMo/blob/v2.3.0/nemo/collections/tts/modules/audio_codec_modules.py
Reference: Mentzer et al., Finite Scalar Quantization: VQ-VAE Made Simple (https://arxiv.org/abs/2309.15505v1)
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.modules.audio.common_modules import (
    CausalConv1d,
    CausalConv1dConfig,
    CausalTransposeConv1d,
    CausalTransposeConv1dConfig,
    Snake1d,
    Snake1dConfig,
)
from lalamo.modules.common import Initializer, LalamoModule

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


@dataclass(frozen=True)
class FiniteScalarQuantizerConfig:
    num_levels: tuple[int, ...]
    eps: float

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

    def init(self, initializer: Initializer) -> "FiniteScalarQuantizer":  # noqa: ARG002
        """Create quantizer with buffer arrays.

        Note: FSQ has no learnable weights, only precomputed buffers.
        """
        num_levels = jnp.array(self.num_levels, dtype=jnp.int32)
        dim_base_index = jnp.cumprod(jnp.concatenate([jnp.array([1], dtype=jnp.int32), num_levels[:-1]]))
        return FiniteScalarQuantizer(
            config=self,
            num_levels_buffer=num_levels,
            dim_base_index=dim_base_index,
        )


class FiniteScalarQuantizer(LalamoModule[FiniteScalarQuantizerConfig]):
    """Finite Scalar Quantizer.

    Quantizes each element of the input vector independently into a number of levels.
    Uses tanh compression and rounding with straight-through estimator.

    Reference: Mentzer et al., Finite Scalar Quantization: VQ-VAE Made Simple
    """

    num_levels_buffer: Int[Array, " dim"]
    dim_base_index: Int[Array, " dim"]

    @property
    def activation_precision(self) -> DTypeLike:
        return jnp.float32

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
        num_levels = self.num_levels_buffer.astype(inputs.dtype)

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
        """Round to nearest integer with straight-through estimator."""
        rounded = jnp.round(inputs)
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
        scale = (self.num_levels_buffer // 2).astype(codes.dtype)
        codes = codes / scale[None, :, None]
        return codes

    def _codes_to_nonnegative(self, codes: Float[Array, "batch dim seq"]) -> Float[Array, "batch dim seq"]:
        """Convert codes centered around zero to nonnegative integer indices."""
        scale = (self.num_levels_buffer // 2).astype(codes.dtype)
        offset = scale
        return scale[None, :, None] * codes + offset[None, :, None]

    def _nonnegative_to_codes(self, codes_nonneg: Float[Array, "dim"]) -> Float[Array, "dim"]:
        """Convert nonnegative indices back to codes centered around zero."""
        scale = (self.num_levels_buffer // 2).astype(codes_nonneg.dtype)
        offset = scale
        return (codes_nonneg - offset) / scale

    def _codes_to_indices(self, codes: Float[Array, "batch dim seq"]) -> Int[Array, "batch seq"]:
        """Convert per-dimension code vectors to single indices.

        Uses the dim_base_index to compute: sum(code_d * base_d) for d in dimensions
        """
        nonneg = self._codes_to_nonnegative(codes)
        # Sum over dimensions weighted by base index
        indices = jnp.sum(nonneg * self.dim_base_index[None, :, None].astype(nonneg.dtype), axis=1)
        return indices.astype(jnp.int32)

    def _indices_to_codes(self, index: Int[Array, " 1"]) -> Float[Array, " dim"]:
        """Convert single indices to per-dimension code vectors.
        Reverses the indexing: code_d = (index // base_d) % levels_d
        """

        codes_nonnegative = (index // self.dim_base_index) % self.num_levels_buffer
        return self._nonnegative_to_codes(codes_nonnegative.astype(jnp.float32))

    def encode(
        self,
        inputs: Float[Array, "batch dim seq"],
    ) -> Int[Array, "batch seq"]:
        """Encode continuous inputs to discrete indices."""
        codes = self._inputs_to_codes(inputs)
        indices = self._codes_to_indices(codes)
        # Add codebook dimension for compatibility with RVQ API
        return indices

    def decode(
        self,
        indices: Int[Array, " seq"],
    ) -> Float[Array, "seq dim"]:
        """Decode discrete indices back to continuous code vectors."""
        return jax.vmap(self._indices_to_codes)(indices)

    def __call__(
        self,
        inputs: Float[Array, "batch seq"],
    ) -> Float[Array, "batch seq dim"]:
        """
        Forward pass: dequantize batch of input indices vectors to continuous representation.
        """
        return jax.vmap(self.decode, in_axes=0)(inputs)


@dataclass(frozen=True)
class GroupFiniteScalarQuantizerConfig:
    num_groups: int
    quantizer_config: FiniteScalarQuantizerConfig

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

    def init(self, initializer: Initializer) -> "GroupFiniteScalarQuantizer":
        quantizers = tuple(self.quantizer_config.init(initializer) for _ in range(self.num_groups))
        return GroupFiniteScalarQuantizer(
            config=self,
            quantizers=quantizers,
        )


class GroupFiniteScalarQuantizer(LalamoModule[GroupFiniteScalarQuantizerConfig]):
    """Group Finite Scalar Quantizer.

    Splits the input vector into groups and applies FSQ on each group separately.
    This enables hierarchical/group-wise quantization where each group produces
    independent indices.
    """

    quantizers: tuple[FiniteScalarQuantizer, ...]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.quantizers[0].activation_precision

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
        """Encode inputs to indices for each group."""
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


@dataclass(frozen=True)
class HalfSnakeConfig:
    snake_config: Snake1dConfig
    leaky_relu_negative_slope: float

    def init(self, initializer: Initializer, channels: int) -> "HalfSnake":
        snake_channels = channels // 2
        snake = self.snake_config.init(initializer, snake_channels)
        return HalfSnake(
            config=self,
            snake=snake,
            total_channels=channels,
        )


class HalfSnake(LalamoModule[HalfSnakeConfig]):
    """HalfSnake activation function.

    Applies Snake activation to the first half of channels and
    LeakyReLU to the second half.
    """

    snake: Snake1d
    total_channels: int = eqx.field(static=True)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.snake.activation_precision

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
        lrelu_out = jnp.where(lrelu_input >= 0, lrelu_input, self.config.leaky_relu_negative_slope * lrelu_input)

        # Concatenate along channel dimension
        return jnp.concatenate([snake_out, lrelu_out], axis=2)


@dataclass(frozen=True)
class ResidualBlockConfig:
    """Configuration for ResidualBlock.

    A residual block that applies activation -> conv -> activation -> conv + residual.
    Used in HiFi-GAN decoder architecture.
    """

    activation_config: HalfSnakeConfig
    conv_config: CausalConv1dConfig

    def init(
        self,
        initializer: Initializer,
        channels: int,
        kernel_size: int,
        dilation: int,
    ) -> "ResidualBlock":
        input_activation = self.activation_config.init(initializer, channels)
        skip_activation = self.activation_config.init(initializer, channels)

        input_conv = self.conv_config.init(
            initializer,
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        skip_conv = self.conv_config.init(
            initializer,
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


class ResidualBlock(LalamoModule[ResidualBlockConfig]):
    """Residual block for HiFi-GAN decoder.
    Applies activation -> conv -> activation -> conv + residual
    """

    input_activation: HalfSnake
    skip_activation: HalfSnake
    input_conv: CausalConv1d
    skip_conv: CausalConv1d

    @property
    def activation_precision(self) -> DTypeLike:
        return self.input_conv.activation_precision

    @property
    def channels(self) -> int:
        return self.input_conv.in_channels

    def __call__(
        self,
        x: Float[Array, "batch sequence channels"],
    ) -> Float[Array, "batch sequence channels"]:
        # activation -> conv -> activation -> conv
        conv_input = self.input_activation(x)
        skip_input = self.input_conv(conv_input)
        skip_input = self.skip_activation(skip_input)
        res = self.skip_conv(skip_input)

        # Residual connection
        return x + res


@dataclass(frozen=True)
class HiFiGANResBlockConfig:
    """Configuration for HiFiGANResBlock.

    Wraps multiple ResidualBlocks with different dilations, applied sequentially.
    Used in HiFi-GAN decoder architecture.
    """

    residual_block_config: ResidualBlockConfig

    def init(
        self,
        initializer: Initializer,
        channels: int,
        kernel_size: int,
        dilations: tuple[int, ...],
    ) -> "HiFiGANResBlock":
        res_blocks = tuple(
            self.residual_block_config.init(
                initializer,
                channels=channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            for dilation in dilations
        )
        return HiFiGANResBlock(config=self, res_blocks=res_blocks)


class HiFiGANResBlock(LalamoModule[HiFiGANResBlockConfig]):
    """HiFiGAN residual block wrapper.
    Creates multiple ResidualBlocks with different dilations and applies them sequentially.
    """

    res_blocks: tuple[ResidualBlock, ...]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.res_blocks[0].activation_precision

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


@dataclass(frozen=True)
class HiFiGANResLayerConfig:
    """Configuration for HiFiGANResLayer.

    Creates multiple HiFiGANResBlocks with different kernel sizes.
    The outputs of all blocks are averaged.
    """

    hifigan_res_block_config: HiFiGANResBlockConfig

    def init(
        self,
        initializer: Initializer,
        channels: int,
        kernel_sizes: tuple[int, ...],
        dilations: tuple[int, ...],
    ) -> "HiFiGANResLayer":
        res_blocks = tuple(
            self.hifigan_res_block_config.init(
                initializer,
                channels=channels,
                kernel_size=kernel_size,
                dilations=dilations,
            )
            for kernel_size in kernel_sizes
        )
        return HiFiGANResLayer(config=self, res_blocks=res_blocks)


class HiFiGANResLayer(LalamoModule[HiFiGANResLayerConfig]):
    """HiFiGAN residual layer.

    Creates multiple HiFiGANResBlocks with different kernel sizes.
    Each block processes the same input, and their outputs are averaged.
    """

    res_blocks: tuple[HiFiGANResBlock, ...]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.res_blocks[0].activation_precision

    @property
    def channels(self) -> int:
        return self.res_blocks[0].channels

    def __call__(
        self,
        x: Float[Array, "batch sequence channels"],
    ) -> Float[Array, "batch sequence channels"]:
        residuals = jnp.stack([res_block(x) for res_block in self.res_blocks], axis=0)
        return jnp.mean(residuals, axis=0)


@dataclass(frozen=True)
class CausalHiFiGANDecoderConfig:
    activation_config: HalfSnakeConfig
    pre_conv_config: CausalConv1dConfig
    transpose_conv_config: CausalTransposeConv1dConfig
    res_layer_config: HiFiGANResLayerConfig
    post_conv_config: CausalConv1dConfig

    def init(
        self,
        initializer: Initializer,
        input_dim: int,
        base_channels: int,
        up_sample_rates: tuple[int, ...],
        in_kernel_size: int,
        out_kernel_size: int,
        resblock_kernel_sizes: tuple[int, ...],
        resblock_dilations: tuple[int, ...],
    ) -> "CausalHiFiGANDecoder":
        # Pre-conv: input_dim -> base_channels
        pre_conv = self.pre_conv_config.init(
            initializer,
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
            activation = self.activation_config.init(initializer, in_channels)
            activations.append(activation)

            # Upsample conv (groups=out_channels for depthwise-like operation)
            upsample_conv = self.transpose_conv_config.init(
                initializer,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=up_sample_rate,
                groups=out_channels,
            )
            upsample_convs.append(upsample_conv)

            # Res layer (at reduced channel count)
            res_layer = self.res_layer_config.init(
                initializer,
                channels=out_channels,
                kernel_sizes=resblock_kernel_sizes,
                dilations=resblock_dilations,
            )
            res_layers.append(res_layer)

            in_channels = out_channels

        # Post activation and conv
        post_activation = self.activation_config.init(initializer, in_channels)
        post_conv = self.post_conv_config.init(
            initializer,
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


class CausalHiFiGANDecoder(LalamoModule[CausalHiFiGANDecoderConfig]):
    """Causal HiFi-GAN decoder for audio generation.

    Converts quantized representations to audio waveforms using:
    1. Pre-conv to expand to base_channels
    2. Multiple stages of: activation -> upsample -> res_layer
    3. Post-conv to produce single-channel audio
    4. Tanh over result as substitute for clipping

    Returns audio waveform in (batch, audio_length) format
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
        return self.pre_conv.activation_precision

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

        out = self.post_activation(out)
        out = self.post_conv(out)  # [B, T_audio, 1]

        audio = jnp.tanh(out)
        audio = audio[:, :, 0]  # [B, T_audio]

        return audio
