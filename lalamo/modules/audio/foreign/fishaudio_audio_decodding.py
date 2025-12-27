import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
from fish_speech.models.dac.modded_dac import ModelArgs
from jax import lax, vmap
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, dummy_array
from lalamo.modules import (
    GELU,
    Activation,
    AttentionConfig,
    DenseMLPConfig,
    ForwardPassMode,
    FullPrecisionLinear,
    FullPrecisionLinearConfig,
    LalamoModule,
    LayerScale,
    LayerScaleConfig,
    Normalization,
    NormalizationConfig,
    SiLU,
    TiedEmbedding,
    TiedEmbeddingConfig,
    Transformer,
    TransformerConfig,
    TransformerLayerConfig,
    UpcastMode,
)

from .fishaudio_common import RoPEConfigFishAudio


def lalamo_transformer_cfg_from_fish_audio_codec_cfg(
    config: ModelArgs, precision: DTypeLike, window_size: int, input_dim: int
) -> TransformerConfig:
    global_rope_config = RoPEConfigFishAudio(
        precision=precision,
        base=config.rope_base,
        max_sequence_length=config.block_size,
    )
    local_rope_config = None

    norm_config_pre = NormalizationConfig(
        scale_precision=precision,
        accumulation_precision=precision,
        epsilon=config.norm_eps,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
    )
    norm_config_post = LayerScaleConfig(scale_precision=precision)

    qkv_projection_config = FullPrecisionLinearConfig(precision=precision)
    out_projection_config = FullPrecisionLinearConfig(precision=precision)
    mixer_config = AttentionConfig(
        qkv_projection_config=qkv_projection_config,
        out_projection_config=out_projection_config,
        query_norm_config=None,
        key_norm_config=None,
        num_heads=config.n_head,
        num_groups=config.n_local_heads,
        head_dim=config.head_dim,
        is_causal=True,
        scale=None,
        sliding_window_size=window_size,
        logit_soft_cap=None,
        has_sinks=False,
        has_qkv_biases=False,
        has_out_biases=False,
    )

    mlp_linear_config = FullPrecisionLinearConfig(precision=precision)
    mlp_use_up_biases = False
    mlp_use_down_biases = False
    mlp_config = DenseMLPConfig(
        linear_config=mlp_linear_config,
        activation=SiLU(),
        has_up_biases=mlp_use_up_biases,
        has_down_biases=mlp_use_down_biases,
        gate_clipping=None,
        up_clipping=None,
    )

    pre_mixer_norm_config = norm_config_pre
    post_mixer_norm_config = norm_config_post
    pre_mlp_norm_config = norm_config_pre
    post_mlp_norm_config = norm_config_post

    layer_config = TransformerLayerConfig(
        pre_mixer_norm_config=pre_mixer_norm_config,
        mixer_config=mixer_config,
        post_mixer_norm_config=post_mixer_norm_config,
        pre_mlp_norm_config=pre_mlp_norm_config,
        mlp_config=mlp_config,
        post_mlp_norm_config=post_mlp_norm_config,
    )
    hidden_dim = config.intermediate_size
    context_length = config.block_size

    transformer_cfg = TransformerConfig(
        global_rope_config=global_rope_config,
        local_rope_config=local_rope_config,
        layer_configs=tuple([layer_config] * config.n_layer),
        output_norm_config=norm_config_pre,
        model_dim=input_dim,
        hidden_dim=hidden_dim,
        context_length=context_length,
    )

    return transformer_cfg


def lalamo_residual_vector_quantize_cfg_from_fish_rvq_cfq() -> "ResidualVectorQuantizeConfig":
    pass


def lalamo_upsampling_block_cfg_from_fish_audio_dac_cfg() -> "UpsamplingBlockConfig":
    return UpsamplingBlockConfig(jnp.float32)


def _get_extra_padding_for_conv1d(length: int, kernel_size: int, stride: int, padding_total: int = 0) -> int:
    """Calculate extra padding needed to ensure output length is correct."""
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return int(ideal_length - length)


@dataclass(frozen=True)
class CausalConv1dConfig:
    """Configuration for CausalConv1d module.

    This is a causal 1D convolution that pads the input on the left side only,
    ensuring that the output at time t only depends on inputs at times <= t.
    """

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
        # Effective kernel size accounting for dilation
        effective_kernel_size = (kernel_size - 1) * dilation + 1

        # Initialize weights using Kaiming/He initialization
        fan_in = in_channels * kernel_size // groups
        scale = 1 / math.sqrt(fan_in)

        weights = jax.random.uniform(
            key,
            (out_channels, in_channels // groups, kernel_size),
            minval=-scale,
            maxval=scale,
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

    Input format: (batch, sequence, channels) - NSC format (JAX convention)
    Output format: (batch, sequence, channels) - NSC format (JAX convention)
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
        """Apply causal convolution.

        Args:
            x: Input tensor of shape (batch, sequence, in_channels)

        Returns:
            Output tensor of shape (batch, sequence_out, out_channels)
        """
        length = x.shape[1]  # sequence dimension is axis 1
        pad = self.padding
        extra_padding = _get_extra_padding_for_conv1d(length, self.effective_kernel_size, self.stride, pad)

        # Pad on the left (causal padding) and extra on the right if needed
        # Input is (batch, sequence, channels), pad the sequence dimension (axis 1)
        x_padded = jnp.pad(x, ((0, 0), (pad, extra_padding), (0, 0)), mode="constant", constant_values=0)

        # Perform convolution using jax.lax.conv_general_dilated
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

    def import_weights(self, weights: ParameterTree[Array]) -> "CausalConv1d":
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
    """Configuration for CausalTransposeConv1d module.

    This is a causal transposed 1D convolution (deconvolution) that removes
    padding from the output to maintain causality.
    """

    precision: DTypeLike
    has_biases: bool

    def random_init(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        *,
        key: PRNGKeyArray,
    ) -> "CausalTransposeConv1d":
        # Initialize weights using Kaiming/He initialization
        fan_in = in_channels * kernel_size
        scale = 1 / math.sqrt(fan_in)

        # For transposed conv, weight shape is (in_channels, out_channels, kernel_size)
        weights = jax.random.uniform(
            key,
            (in_channels, out_channels, kernel_size),
            minval=-scale,
            maxval=scale,
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
            stride=stride,
            dilation=dilation,
        )

    def empty(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> "CausalTransposeConv1d":
        weights = dummy_array(
            (in_channels, out_channels, kernel_size),
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
            stride=stride,
            dilation=dilation,
        )


class CausalTransposeConv1d(LalamoModule[CausalTransposeConv1dConfig]):
    """Causal transposed 1D convolution (deconvolution) module.

    Implements causal transposed convolution by removing appropriate padding
    from the output.

    Input format: (batch, sequence, channels) - NSC format (JAX convention)
    Output format: (batch, sequence, channels) - NSC format (JAX convention)
    """

    weights: Float[Array, "in_channels out_channels kernel_size"]
    biases: Float[Array, " out_channels"] | None

    stride: int = eqx.field(static=True)
    dilation: int = eqx.field(static=True)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def in_channels(self) -> int:
        in_channels, _, _ = self.weights.shape
        return in_channels

    @property
    def out_channels(self) -> int:
        _, out_channels, _ = self.weights.shape
        return out_channels

    @property
    def kernel_size(self) -> int:
        _, _, kernel_size = self.weights.shape
        return kernel_size

    def __call__(
        self,
        x: Float[Array, "batch sequence in_channels"],
    ) -> Float[Array, "batch sequence_out out_channels"]:
        """Apply causal transposed convolution.

        Args:
            x: Input tensor of shape (batch, sequence, in_channels)

        Returns:
            Output tensor of shape (batch, sequence_out, out_channels)
        """
        # Perform transposed convolution using jax.lax.conv_transpose
        # PyTorch ConvTranspose1d weight is (in_channels, out_channels, kernel_size)
        # Input: (N, S, C) - NSC format
        # Kernel: (I, O, K) - IOH format
        # Output: (N, S, C) - NSC format
        output = lax.conv_transpose(
            x,
            self.weights,
            strides=(self.stride,),
            padding="VALID",
            rhs_dilation=(self.dilation,),
            dimension_numbers=("NHC", "OIH", "NHC"),
            transpose_kernel=True,
        )

        if self.biases is not None:
            output = output + self.biases[None, None, :]

        # Remove padding to maintain causality
        pad = self.kernel_size - self.stride
        padding_right = math.ceil(pad)
        padding_left = pad - padding_right

        # Unpad: remove padding_left from start and padding_right from end
        # Sequence dimension is now axis 1
        if padding_left > 0 or padding_right > 0:
            end = output.shape[1] - padding_right if padding_right > 0 else output.shape[1]
            output = output[:, padding_left:end, :]

        return output

    def export_weights(self) -> ParameterTree[Array]:
        result: dict[str, Array] = {"weights": self.weights}
        if self.biases is not None:
            result["biases"] = self.biases
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> "CausalTransposeConv1d":
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
class Conv1dConfig:
    """Configuration for standard (non-causal) Conv1d module.

    This is a standard 1D convolution with symmetric padding.
    """

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
    ) -> "Conv1d":
        fan_in = in_channels * kernel_size // groups
        scale = 1 / math.sqrt(fan_in)

        weights = jax.random.uniform(
            key,
            (out_channels, in_channels // groups, kernel_size),
            minval=-scale,
            maxval=scale,
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
        )


class Conv1d(LalamoModule[Conv1dConfig]):
    """Standard (non-causal) 1D convolution module.

    Uses symmetric padding (same padding on both sides).

    Input format: (batch, sequence, channels) - NSC format (JAX convention)
    Output format: (batch, sequence, channels) - NSC format (JAX convention)
    """

    weights: Float[Array, "out_channels in_channels_per_group kernel_size"]
    biases: Float[Array, " out_channels"] | None

    stride: int = eqx.field(static=True)
    dilation: int = eqx.field(static=True)
    groups: int = eqx.field(static=True)

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

    def __call__(
        self,
        x: Float[Array, "batch sequence in_channels"],
    ) -> Float[Array, "batch sequence_out out_channels"]:
        """Apply standard convolution with symmetric padding.

        Args:
            x: Input tensor of shape (batch, sequence, in_channels)

        Returns:
            Output tensor of shape (batch, sequence_out, out_channels)
        """
        effective_kernel_size = (self.kernel_size - 1) * self.dilation + 1
        total_padding = effective_kernel_size - 1
        pad_left = total_padding // 2
        pad_right = total_padding - pad_left

        x_padded = jnp.pad(x, ((0, 0), (pad_left, pad_right), (0, 0)), mode="constant", constant_values=0)

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

    def import_weights(self, weights: ParameterTree[Array]) -> "Conv1d":
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
class ConvNeXtSpatialParams:
    """Spatial parameters for ConvNeXt blocks.

    These parameters control the spatial convolution and MLP expansion in ConvNeXt blocks.
    """

    mlp_ratio: float
    kernel_size: int
    dilation: int
    layer_scale_init_value: float


@dataclass(frozen=True)
class ConvNeXtBlockConfig:
    """Configuration for ConvNeXt block.

    ConvNeXt block consists of:
    1. Depthwise convolution (groups=channels)
    2. LayerNorm
    3. Pointwise conv (Linear) expanding to mlp_ratio * dim
    4. GELU activation
    5. Pointwise conv (Linear) projecting back to dim
    6. Optional layer scale (gamma)
    7. Residual connection
    """

    precision: DTypeLike
    activation: Activation

    def random_init(
        self,
        dim: int,
        spatial_params: ConvNeXtSpatialParams,
        *,
        key: PRNGKeyArray,
    ) -> "ConvNeXtBlock":
        key1, key2, key3 = jax.random.split(key, 3)

        dwconv_config = CausalConv1dConfig(precision=self.precision, has_biases=True)
        dwconv = dwconv_config.random_init(
            in_channels=dim,
            out_channels=dim,
            kernel_size=spatial_params.kernel_size,
            stride=1,
            dilation=spatial_params.dilation,
            groups=dim,
            key=key1,
        )

        norm_config = NormalizationConfig(
            scale_precision=self.precision,
            accumulation_precision=self.precision,
            epsilon=1e-6,
            scale_offset=None,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=True,
            use_bias=True,
        )
        norm = norm_config.empty(dim)

        hidden_dim = int(spatial_params.mlp_ratio * dim)
        pwconv1_config = FullPrecisionLinearConfig(precision=self.precision)
        pwconv1 = pwconv1_config.random_init(dim, (hidden_dim,), has_biases=True, key=key2)

        pwconv2_config = FullPrecisionLinearConfig(precision=self.precision)
        pwconv2 = pwconv2_config.random_init(hidden_dim, (dim,), has_biases=True, key=key3)

        if spatial_params.layer_scale_init_value > 0:
            gamma_config = LayerScaleConfig(scale_precision=self.precision)
            gamma = LayerScale(
                config=gamma_config,
                scales=jnp.ones((dim,), dtype=self.precision) * spatial_params.layer_scale_init_value,
            )
        else:
            gamma = None

        return ConvNeXtBlock(
            config=self,
            depthwise_conv=dwconv,
            norm=norm,
            pointwise_conv_step1=pwconv1,
            pointwise_conv_step2=pwconv2,
            scale=gamma,
        )

    def empty(
        self,
        dim: int,
        spatial_params: ConvNeXtSpatialParams,
    ) -> "ConvNeXtBlock":
        dwconv_config = CausalConv1dConfig(precision=self.precision, has_biases=True)
        dwconv = dwconv_config.empty(
            in_channels=dim,
            out_channels=dim,
            kernel_size=spatial_params.kernel_size,
            stride=1,
            dilation=spatial_params.dilation,
            groups=dim,
        )

        norm_config = NormalizationConfig(
            scale_precision=self.precision,
            accumulation_precision=self.precision,
            epsilon=10e-6,
            scale_offset=None,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=True,
            use_bias=True,
        )
        norm = norm_config.empty(dim)

        hidden_dim = int(spatial_params.mlp_ratio * dim)
        pwconv1_config = FullPrecisionLinearConfig(precision=self.precision)
        pwconv1 = pwconv1_config.empty(dim, (hidden_dim,), has_biases=True)

        pwconv2_config = FullPrecisionLinearConfig(precision=self.precision)
        pwconv2 = pwconv2_config.empty(hidden_dim, (dim,), has_biases=True)

        if spatial_params.layer_scale_init_value > 0:
            gamma_config = LayerScaleConfig(scale_precision=self.precision)
            gamma = gamma_config.empty(dim)
        else:
            gamma = None

        return ConvNeXtBlock(
            config=self,
            depthwise_conv=dwconv,
            norm=norm,
            pointwise_conv_step1=pwconv1,
            pointwise_conv_step2=pwconv2,
            scale=gamma,
        )


class ConvNeXtBlock(LalamoModule[ConvNeXtBlockConfig]):
    """ConvNeXt block implementation.

    Architecture:
    1. DwConv (depthwise causal conv)
    2. LayerNorm
    3. Pointwise conv 1 (expand)
    4. GELU
    5. Pointwise conv 2 (project)
    6. Layer scale (gamma)
    7. Residual connection

    Input format: (batch, sequence, channels) - NSC format (JAX convention)
    Output format: (batch, sequence, channels) - NSC format (JAX convention)
    """

    depthwise_conv: CausalConv1d
    norm: Normalization
    pointwise_conv_step1: FullPrecisionLinear
    pointwise_conv_step2: FullPrecisionLinear
    scale: LayerScale | None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def dim(self) -> int:
        return self.depthwise_conv.out_channels

    def __call__(
        self,
        x: Float[Array, "batch sequence channels"],
        apply_residual: bool = True,
    ) -> Float[Array, "batch sequence channels"]:
        """Apply ConvNeXt block.

        Args:
            x: Input tensor of shape (batch, sequence, channels)
            apply_residual: Whether to add residual connection

        Returns:
            Output tensor of shape (batch, sequence, channels)
        """
        residual = x

        x = self.depthwise_conv(x)
        x = jax.vmap(jax.vmap(self.norm))(x)
        (x,) = jax.vmap(jax.vmap(self.pointwise_conv_step1))(x)
        x = jax.vmap(jax.vmap(self.config.activation))(x)
        (x,) = jax.vmap(jax.vmap(self.pointwise_conv_step2))(x)
        if self.scale is not None:
            x = jax.vmap(jax.vmap(self.scale))(x)
        if apply_residual:
            x = residual + x

        return x

    def export_weights(self) -> ParameterTree[Array]:
        result: dict[str, ParameterTree[Array]] = {
            "dwconv": self.depthwise_conv.export_weights(),
            "norm": self.norm.export_weights(),
            "pwconv1": self.pointwise_conv_step1.export_weights(),
            "pwconv2": self.pointwise_conv_step2.export_weights(),
        }
        if self.scale is not None:
            result["gamma"] = self.scale.export_weights()
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> "ConvNeXtBlock":
        assert isinstance(weights, Mapping)
        dwconv_weights = weights["dwconv"]
        norm_weights = weights["norm"]
        pwconv1_weights = weights["pwconv1"]
        pwconv2_weights = weights["pwconv2"]
        assert isinstance(dwconv_weights, Mapping)
        assert isinstance(norm_weights, Mapping)
        assert isinstance(pwconv1_weights, Mapping)
        assert isinstance(pwconv2_weights, Mapping)

        if self.scale is not None:
            gamma_weights = weights.get("gamma")
            assert isinstance(gamma_weights, Mapping)
            gamma = self.scale.import_weights(gamma_weights)
        else:
            gamma = None

        return replace(
            self,
            depthwise_conv=self.depthwise_conv.import_weights(dwconv_weights),
            norm=self.norm.import_weights(norm_weights),
            pointwise_conv_step1=self.pointwise_conv_step1.import_weights(pwconv1_weights),
            pointwise_conv_step2=self.pointwise_conv_step2.import_weights(pwconv2_weights),
            scale=gamma,
        )


@dataclass(frozen=True)
class TransposeConvSpatialParams:
    """Parameters for a single upsampling block.

    These parameters define the structure of one upsampling block within the Upsampler.
    """

    in_channels: int
    out_channels: int
    upsample_kernel_size: int
    upsample_stride: int


@dataclass(frozen=True)
class UpsamplingBlockConfig:
    """Configuration for upsampling block.

    The upsampling block consists of:
    1. CausalTransposeConv1d (upsampling)
    2. ConvNeXtBlock
    """

    precision: DTypeLike

    def random_init(
        self,
        trans_conv_params: TransposeConvSpatialParams,
        convnext_spatial_params: ConvNeXtSpatialParams,
        *,
        key: PRNGKeyArray,
    ) -> "UpsamplingBlock":
        key1, key2 = jax.random.split(key)

        trans_conv_config = CausalTransposeConv1dConfig(precision=self.precision, has_biases=True)
        trans_conv = trans_conv_config.random_init(
            in_channels=trans_conv_params.in_channels,
            out_channels=trans_conv_params.out_channels,
            kernel_size=trans_conv_params.upsample_kernel_size,
            stride=trans_conv_params.upsample_stride,
            dilation=1,
            key=key1,
        )

        convnext_config = ConvNeXtBlockConfig(
            precision=self.precision,
            activation=GELU(approximate=False),
        )
        convnext = convnext_config.random_init(
            dim=trans_conv_params.out_channels,
            spatial_params=convnext_spatial_params,
            key=key2,
        )

        return UpsamplingBlock(
            config=self,
            trans_conv=trans_conv,
            convnext=convnext,
        )

    def empty(
        self,
        trans_conv_params: TransposeConvSpatialParams,
        convnext_spatial_params: ConvNeXtSpatialParams,
    ) -> "UpsamplingBlock":
        trans_conv_config = CausalTransposeConv1dConfig(precision=self.precision, has_biases=True)
        trans_conv = trans_conv_config.empty(
            in_channels=trans_conv_params.in_channels,
            out_channels=trans_conv_params.out_channels,
            kernel_size=trans_conv_params.upsample_kernel_size,
            stride=trans_conv_params.upsample_stride,
            dilation=1,
        )

        convnext_config = ConvNeXtBlockConfig(
            precision=self.precision,
            activation=GELU(approximate=False),
        )
        convnext = convnext_config.empty(
            dim=trans_conv_params.out_channels,
            spatial_params=convnext_spatial_params,
        )

        return UpsamplingBlock(
            config=self,
            trans_conv=trans_conv,
            convnext=convnext,
        )


class UpsamplingBlock(LalamoModule[UpsamplingBlockConfig]):
    """Upsampling block consisting of transposed convolution followed by ConvNeXt block.

    Architecture:
    1. CausalTransposeConv1d (upsample)
    2. ConvNeXtBlock (refine)

    Input format: (batch, sequence, channels) - NSC format (JAX convention)
    Output format: (batch, sequence_upsampled, channels) - NSC format (JAX convention)
    """

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
        """Apply upsampling block.

        Args:
            x: Input tensor of shape (batch, sequence, in_channels)

        Returns:
            Output tensor of shape (batch, sequence_upsampled, out_channels)
        """
        # Transposed conv for upsampling
        x = self.trans_conv(x)

        # ConvNeXt block for refinement
        x = self.convnext(x)

        return x

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "trans_conv": self.trans_conv.export_weights(),
            "convnext": self.convnext.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> "UpsamplingBlock":
        assert isinstance(weights, Mapping)
        trans_conv_weights = weights["trans_conv"]
        convnext_weights = weights["convnext"]
        assert isinstance(trans_conv_weights, Mapping)
        assert isinstance(convnext_weights, Mapping)

        return replace(
            self,
            trans_conv=self.trans_conv.import_weights(trans_conv_weights),
            convnext=self.convnext.import_weights(convnext_weights),
        )


@dataclass(frozen=True)
class UpsamplerConfig:
    """Configuration for the full upsampler module.

    The upsampler consists of a sequence of UpsamplingBlocks, each with its own configuration.
    """

    block_configs: tuple[UpsamplingBlockConfig, ...]

    def empty(
        self,
        trans_conv_params_per_block: tuple[TransposeConvSpatialParams, ...],
        convnext_spatial_params: ConvNeXtSpatialParams,
    ) -> "Upsampler":
        """Create an empty Upsampler with specified block parameters.

        Args:
            trans_conv_params_per_block: Tuple of TransposeConvSpatialParams for each block.
            convnext_spatial_params: Spatial parameters for ConvNeXt blocks (shared across all blocks).

        Returns:
            Empty Upsampler module.
        """
        assert len(self.block_configs) == len(trans_conv_params_per_block), (
            f"Number of block configs ({len(self.block_configs)}) must match "
            f"number of block params ({len(trans_conv_params_per_block)})"
        )

        blocks = []
        for config, trans_conv_params in zip(self.block_configs, trans_conv_params_per_block, strict=True):
            block = config.empty(
                trans_conv_params=trans_conv_params,
                convnext_spatial_params=convnext_spatial_params,
            )
            blocks.append(block)

        return Upsampler(config=self, blocks=tuple(blocks))

    def random_init(
        self,
        trans_conv_params_per_block: tuple[TransposeConvSpatialParams, ...],
        convnext_spatial_params: ConvNeXtSpatialParams,
        *,
        key: PRNGKeyArray,
    ) -> "Upsampler":
        """Create a randomly initialized Upsampler.

        Args:
            trans_conv_params_per_block: Tuple of TransposeConvSpatialParams for each block.
            convnext_spatial_params: Spatial parameters for ConvNeXt blocks (shared across all blocks).
            key: PRNG key for random initialization.

        Returns:
            Randomly initialized Upsampler module.
        """
        assert len(self.block_configs) == len(trans_conv_params_per_block)

        blocks = []
        keys = jax.random.split(key, len(self.block_configs))
        for config, trans_conv_params, k in zip(self.block_configs, trans_conv_params_per_block, keys, strict=True):
            block = config.random_init(
                trans_conv_params=trans_conv_params,
                convnext_spatial_params=convnext_spatial_params,
                key=k,
            )
            blocks.append(block)

        return Upsampler(config=self, blocks=tuple(blocks))


class Upsampler(LalamoModule[UpsamplerConfig]):
    """Full upsampler module consisting of multiple UpsamplingBlocks.

    This module sequentially applies a series of upsampling blocks to progressively
    increase the temporal resolution of the input while transforming channel dimensions.

    Input format: (batch, sequence, channels) - NSC format (JAX convention)
    Output format: (batch, sequence_upsampled, channels) - NSC format (JAX convention)
    """

    blocks: tuple[UpsamplingBlock, ...]

    @property
    def activation_precision(self) -> DTypeLike:
        if len(self.blocks) > 0:
            return self.blocks[0].activation_precision
        raise ValueError("Upsampler has no blocks")

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def __call__(
        self,
        x: Float[Array, "batch sequence in_channels"],
    ) -> Float[Array, "batch sequence_out out_channels"]:
        """Apply all upsampling blocks sequentially.

        Args:
            x: Input tensor of shape (batch, sequence, in_channels)

        Returns:
            Output tensor of shape (batch, sequence_upsampled, out_channels)
        """
        for block in self.blocks:
            x = block(x)
        return x

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "blocks": [block.export_weights() for block in self.blocks],
        }

    def import_weights(self, weights: ParameterTree[Array]) -> "Upsampler":
        assert isinstance(weights, Mapping)
        block_weights = weights["blocks"]
        assert isinstance(block_weights, list)

        new_blocks = []
        for block, w in zip(self.blocks, block_weights, strict=True):
            assert isinstance(w, Mapping)
            new_blocks.append(block.import_weights(w))

        return replace(self, blocks=tuple(new_blocks))


@dataclass(frozen=True)
class VectorQuantizeConfig:
    precision: DTypeLike

    def empty(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
    ) -> "VectorQuantize":
        codebook_config = TiedEmbeddingConfig(
            input_scale=None,
            logit_soft_cap=None,
            precision=self.precision,
        )
        codebook = codebook_config.empty(codebook_size, codebook_dim)
        assert isinstance(codebook, TiedEmbedding)

        out_proj_config = FullPrecisionLinearConfig(precision=self.precision)
        out_proj = out_proj_config.empty(
            input_dim=codebook_dim,
            output_dims=(input_dim,),
            has_biases=True,
        )
        assert isinstance(out_proj, FullPrecisionLinear)

        return VectorQuantize(
            config=self,
            codebook=codebook,
            out_proj=out_proj,
        )


class VectorQuantize(LalamoModule[VectorQuantizeConfig]):
    """Vector Quantization module (decoding path only).

    Decodes codebook indices back to input space by:
    1. Looking up codebook vectors
    2. Projecting from codebook_dim to input_dim via out_proj
    """

    codebook: TiedEmbedding
    out_proj: FullPrecisionLinear

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def codebook_size(self) -> int:
        return self.codebook.vocab_size

    @property
    def codebook_dim(self) -> int:
        return self.codebook.model_dim

    def decode_code(self, embed_id: Int[Array, " tokens"]) -> Float[Array, "tokens code_size"]:
        z_p = self.codebook.embed(embed_id)
        (z_q,) = vmap(self.out_proj)(z_p)
        return z_q

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "codebook": self.codebook.export_weights(),
            "out_proj": self.out_proj.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        codebook_weights = weights["codebook"]
        out_proj_weights = weights["out_proj"]
        assert isinstance(codebook_weights, Mapping)
        assert isinstance(out_proj_weights, Mapping)
        return replace(
            self,
            codebook=self.codebook.import_weights(codebook_weights),
            out_proj=self.out_proj.import_weights(out_proj_weights),
        )


@dataclass(frozen=True)
class ResidualVectorQuantizeConfig:
    precision: DTypeLike

    def empty(
        self,
        code_size: int,
        codebook_size: int,
        codebook_dim: int | list[int],
    ) -> "ResidualVectorQuantize":
        if isinstance(codebook_dim, int):
            codebook_dims = [codebook_dim]
        else:
            codebook_dims = list(codebook_dim)

        quantizers = []
        vq_config = VectorQuantizeConfig(precision=self.precision)
        for dim in codebook_dims:
            quantizers.append(
                vq_config.empty(
                    input_dim=code_size,
                    codebook_size=codebook_size,
                    codebook_dim=dim,
                )
            )

        return ResidualVectorQuantize(
            config=self,
            quantizers=tuple(quantizers),
        )


class ResidualVectorQuantize(LalamoModule[ResidualVectorQuantizeConfig]):
    """Residual Vector Quantization module (decoding path only).
    Decodes codes from multiple codebooks by summing their decoded outputs.
    """

    quantizers: tuple[VectorQuantize, ...]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def n_codebooks(self) -> int:
        return len(self.quantizers)

    def from_codes(self, codes: Int[Array, "n_codebooks tokens"]) -> Float[Array, "tokens code_size"]:
        n_codebooks = codes.shape[0]
        z_q = self.quantizers[0].decode_code(codes[0])
        for i in range(1, n_codebooks):
            z_q = z_q + self.quantizers[i].decode_code(codes[i])
        return z_q

    def __call__(self, codes: Int[Array, "batch n_codebooks tokens"]) -> Float[Array, "batch tokens code_size"]:
        return vmap(self.from_codes)(codes)

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "quantizers": [q.export_weights() for q in self.quantizers],
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        quantizer_weights = weights["quantizers"]
        assert isinstance(quantizer_weights, list)
        new_quantizers = []
        for q, w in zip(self.quantizers, quantizer_weights, strict=True):
            assert isinstance(w, Mapping)
            new_quantizers.append(q.import_weights(w))
        return replace(self, quantizers=tuple(new_quantizers))


@dataclass(frozen=True)
class DownsampleResidualVectorQuantizeConfig:
    """Configuration for DownsampleResidualVectorQuantize module.

    This module combines semantic and residual quantization with a post-processing
    transformer and upsampling to decode audio codes back to continuous representations.
    """

    precision: DTypeLike
    semantic_quantizer_config: ResidualVectorQuantizeConfig
    quantizer_config: ResidualVectorQuantizeConfig
    post_module_config: TransformerConfig
    upsampler_config: UpsamplerConfig

    def empty(
        self,
        upsampler_trans_conv_params: tuple[TransposeConvSpatialParams, ...],
        convnext_spatial_params: ConvNeXtSpatialParams,
        semantic_code_size: int,
        semantic_codebook_size: int,
        semantic_codebook_dim: int | list[int],
        quantizer_code_size: int,
        quantizer_codebook_size: int,
        quantizer_codebook_dim: int | list[int],
    ) -> "DownsampleResidualVectorQuantize":
        """Create module with uninitialized (dummy) weights.

        Args:
            upsampler_trans_conv_params: Tuple of TransposeConvSpatialParams for each upsampling block.
            convnext_spatial_params: Spatial parameters for ConvNeXt blocks.
            semantic_code_size: Output dimension for semantic quantizer codes.
            semantic_codebook_size: Number of entries in semantic codebook.
            semantic_codebook_dim: Dimension of semantic codebook vectors.
            quantizer_code_size: Output dimension for residual quantizer codes.
            quantizer_codebook_size: Number of entries in residual codebook.
            quantizer_codebook_dim: Dimension of residual codebook vectors.

        Returns:
            DownsampleResidualVectorQuantize module with dummy weights.
        """
        semantic_quantizer = self.semantic_quantizer_config.empty(
            code_size=semantic_code_size,
            codebook_size=semantic_codebook_size,
            codebook_dim=semantic_codebook_dim,
        )
        quantizer = self.quantizer_config.empty(
            code_size=quantizer_code_size,
            codebook_size=quantizer_codebook_size,
            codebook_dim=quantizer_codebook_dim,
        )
        post_module = self.post_module_config.empty()
        upsampler = self.upsampler_config.empty(
            trans_conv_params_per_block=upsampler_trans_conv_params,
            convnext_spatial_params=convnext_spatial_params,
        )

        return DownsampleResidualVectorQuantize(
            config=self,
            semantic_quantizer=semantic_quantizer,
            quantizer=quantizer,
            post_module=post_module,
            upsampler=upsampler,
        )

    def random_init(
        self,
        upsampler_trans_conv_params: tuple[TransposeConvSpatialParams, ...],
        convnext_spatial_params: ConvNeXtSpatialParams,
        semantic_code_size: int,
        semantic_codebook_size: int,
        semantic_codebook_dim: int | list[int],
        quantizer_code_size: int,
        quantizer_codebook_size: int,
        quantizer_codebook_dim: int | list[int],
        *,
        key: PRNGKeyArray,
    ) -> "DownsampleResidualVectorQuantize":
        """Create module with randomly initialized weights.

        Args:
            upsampler_trans_conv_params: Tuple of TransposeConvSpatialParams for each upsampling block.
            convnext_spatial_params: Spatial parameters for ConvNeXt blocks.
            semantic_code_size: Output dimension for semantic quantizer codes.
            semantic_codebook_size: Number of entries in semantic codebook.
            semantic_codebook_dim: Dimension of semantic codebook vectors.
            quantizer_code_size: Output dimension for residual quantizer codes.
            quantizer_codebook_size: Number of entries in residual codebook.
            quantizer_codebook_dim: Dimension of residual codebook vectors.
            key: PRNG key for random initialization.

        Returns:
            DownsampleResidualVectorQuantize module with random weights.
        """
        key1, key2 = jax.random.split(key)

        # ResidualVectorQuantize only has empty(), not random_init()
        semantic_quantizer = self.semantic_quantizer_config.empty(
            code_size=semantic_code_size,
            codebook_size=semantic_codebook_size,
            codebook_dim=semantic_codebook_dim,
        )
        quantizer = self.quantizer_config.empty(
            code_size=quantizer_code_size,
            codebook_size=quantizer_codebook_size,
            codebook_dim=quantizer_codebook_dim,
        )
        post_module = self.post_module_config.random_init(key=key1)
        upsampler = self.upsampler_config.random_init(
            trans_conv_params_per_block=upsampler_trans_conv_params,
            convnext_spatial_params=convnext_spatial_params,
            key=key2,
        )

        return DownsampleResidualVectorQuantize(
            config=self,
            semantic_quantizer=semantic_quantizer,
            quantizer=quantizer,
            post_module=post_module,
            upsampler=upsampler,
        )


class DownsampleResidualVectorQuantize(LalamoModule[DownsampleResidualVectorQuantizeConfig]):
    """Downsampled Residual Vector Quantization decoder module.

    This module decodes audio codes by:
    1. Decoding semantic codes through the semantic quantizer
    2. Decoding residual codes through the residual quantizer
    3. Summing the semantic and residual representations
    4. Processing through a transformer post-module
    5. Upsampling to the target temporal resolution

    Input: Integer codes with shape (batch, n_codebooks, tokens)
           where the first codebook row contains semantic codes
           and remaining rows contain residual codes.
    Output: Continuous audio features with shape (batch, upsampled_tokens, channels)
    """

    semantic_quantizer: ResidualVectorQuantize
    quantizer: ResidualVectorQuantize
    post_module: Transformer
    upsampler: Upsampler

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def semantic_codebook_size(self) -> int:
        return self.semantic_quantizer.quantizers[0].codebook_size

    @property
    def quantizer_codebook_size(self) -> int:
        return self.quantizer.quantizers[0].codebook_size

    def decode(
        self,
        indices: Int[Array, "batch n_codebooks tokens"],
    ) -> Float[Array, "batch upsampled_tokens channels"]:
        """Decode audio codes to continuous features.

        Args:
            indices: Integer codes with shape (batch, n_codebooks, tokens).
                     First row (indices[:, 0]) contains semantic codes,
                     remaining rows (indices[:, 1:]) contain residual codes.

        Returns:
            Decoded audio features with shape (batch, upsampled_tokens, channels).
        """
        # Clamp indices to valid codebook ranges
        semantic_indices = jnp.clip(indices[:, :1], 0, self.semantic_codebook_size - 1)
        residual_indices = jnp.clip(indices[:, 1:], 0, self.quantizer_codebook_size - 1)

        # Decode semantic codes: (batch, 1, tokens) -> (batch, tokens, input_dim)
        z_q_semantic = vmap(self.semantic_quantizer.from_codes)(semantic_indices)

        # Decode residual codes: (batch, n_residual_codebooks, tokens) -> (batch, tokens, input_dim)
        z_q_residual = vmap(self.quantizer.from_codes)(residual_indices)

        # Sum semantic and residual representations
        z_q = z_q_semantic + z_q_residual

        # Process through transformer post-module
        batch_size, seq_length, _ = z_q.shape
        token_positions = jnp.broadcast_to(jnp.arange(seq_length)[None, :], (batch_size, seq_length))

        post_result = self.post_module(
            inner_features=z_q,
            token_positions=token_positions,
            state=None,
            return_updated_state=False,
            return_layer_results=False,
            return_positional_embeddings=False,
            lengths_without_padding=None,
            forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
            forward_pass_config=None,
        )
        z_q = post_result.outputs

        # Upsample to target temporal resolution
        z_q = self.upsampler(z_q)

        return z_q

    def __call__(
        self,
        indices: Int[Array, "batch n_codebooks tokens"],
    ) -> Float[Array, "batch upsampled_tokens channels"]:
        """Decode audio codes to continuous features.

        This is an alias for the decode method.
        """
        return self.decode(indices)

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "semantic_quantizer": self.semantic_quantizer.export_weights(),
            "quantizer": self.quantizer.export_weights(),
            "post_module": self.post_module.export_weights(),
            "upsampler": self.upsampler.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)

        semantic_quantizer_weights = weights["semantic_quantizer"]
        quantizer_weights = weights["quantizer"]
        post_module_weights = weights["post_module"]
        upsampler_weights = weights["upsampler"]

        assert isinstance(semantic_quantizer_weights, Mapping)
        assert isinstance(quantizer_weights, Mapping)
        assert isinstance(post_module_weights, Mapping)
        assert isinstance(upsampler_weights, Mapping)

        return replace(
            self,
            semantic_quantizer=self.semantic_quantizer.import_weights(semantic_quantizer_weights),
            quantizer=self.quantizer.import_weights(quantizer_weights),
            post_module=self.post_module.import_weights(post_module_weights),
            upsampler=self.upsampler.import_weights(upsampler_weights),
        )
