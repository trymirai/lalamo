"""Loaders for NanoCodec modules from NVIDIA NeMo.

This module provides functions to load weights from PyTorch NanoCodec models
into the Lalamo JAX implementations.

Weight format conventions:
- PyTorch uses NCT format (batch, channels, time) for audio
- JAX/Lalamo uses NSC format (batch, sequence, channels)
- Weight transformations handle format differences automatically
"""

from collections.abc import Mapping

from einops import rearrange
from jax import numpy as jnp
from jaxtyping import Array, Float

from lalamo.common import ParameterPath
from lalamo.modules.audio.common_modules import Snake1d
from lalamo.modules.audio.fishaudio.fishaudio_modules import CausalConv1d
from lalamo.modules.audio.nanocodec.audio_decoding import NanoCodec
from lalamo.modules.audio.nanocodec.nanocodec_modules import (
    CausalHiFiGANDecoder,
    CausalTransposeConv1d,
    HalfSnake,
    HiFiGANResBlock,
    HiFiGANResLayer,
    ResidualBlock,
)

from .common import load_parameters
from .torch_utils import _fuse_parametrized_weight_norm_conv1d

__all__ = [
    "load_causal_conv1d",
    "load_causal_hifigan_decoder",
    "load_causal_transpose_conv1d",
    "load_half_snake",
    "load_hifigan_res_block",
    "load_hifigan_res_layer",
    "load_nanocodec",
    "load_residual_block",
    "load_snake1d",
    "transform_pytorch_transpose_conv_weights",
]


# =============================================================================
# Weight Transformation Utilities
# =============================================================================


def transform_pytorch_transpose_conv_weights(
    weights: Float[Array, "in_channels out_per_group kernel_size"],
    in_channels: int,
    out_channels: int,
    groups: int,
) -> Float[Array, "out_channels in_per_group kernel_size"]:
    """Transform PyTorch transposed conv weights to JAX format.

    PyTorch ConvTranspose1d weight shape: (in_channels, out_channels // groups, kernel_size)
    JAX expected weight shape: (out_channels, in_channels // groups, kernel_size)

    This function also flips the kernel along the spatial dimension as required
    for transposed convolution via lhs_dilation.

    Args:
        weights: PyTorch weights of shape (in_channels, out_channels // groups, kernel_size)
        in_channels: Number of input channels
        out_channels: Number of output channels
        groups: Number of groups for grouped convolution

    Returns:
        Transformed weights of shape (out_channels, in_channels // groups, kernel_size)
    """
    in_per_group = in_channels // groups
    out_per_group = out_channels // groups
    kernel_size = weights.shape[-1]

    # Reshape: (in_channels, out_per_group, K) -> (groups, in_per_group, out_per_group, K)
    kernel = weights.reshape(groups, in_per_group, out_per_group, kernel_size)
    # Swap axes: -> (groups, out_per_group, in_per_group, K)
    kernel = jnp.swapaxes(kernel, 1, 2)
    # Reshape: -> (out_channels, in_per_group, K)
    kernel = kernel.reshape(out_channels, in_per_group, kernel_size)
    # Flip kernel along spatial dimension - required for transposed convolution
    kernel = jnp.flip(kernel, axis=-1)

    return kernel


# =============================================================================
# Activation Loaders
# =============================================================================


def load_snake1d(
    module: Snake1d,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> Snake1d:
    """Load a Snake1d module from weights.

    Expected weight structure at path:
        - alpha: shape (channels,) or (1, channels, 1) (PyTorch format)

    The PyTorch Snake stores alpha as (1, channels, 1), but our module
    stores it as (channels,), so we squeeze the extra dimensions if present.

    Args:
        module: The Snake1d module to load weights into.
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Base path for this module's weights.

    Returns:
        Snake1d module with loaded weights.
    """
    alpha = weights_dict[path / "alpha"]
    # PyTorch shape: (1, channels, 1) -> (channels,)
    alpha = rearrange(alpha, "1 channels 1 -> channels")

    return load_parameters(
        lambda m: (m.alpha,),
        module,
        (alpha,),
    )


def load_half_snake(
    module: HalfSnake,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> HalfSnake:
    """Load a HalfSnake module from weights.

    Expected weight structure at path:
        - snake_act.alpha: shape (1, snake_channels, 1) in PyTorch format

    Note: In PyTorch NeMo, HalfSnake is wrapped in CodecActivation which stores
    the HalfSnake as `.activation`. The HalfSnake stores Snake as `.snake_act`.
    So the full path from a CodecActivation is: activation.snake_act.alpha

    Args:
        module: The HalfSnake module to load weights into.
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Base path for this module's weights (should point to HalfSnake, not CodecActivation).

    Returns:
        HalfSnake module with loaded weights.
    """
    snake = load_snake1d(module.snake, weights_dict, path / "snake_act")

    return load_parameters(
        lambda m: (m.snake,),
        module,
        (snake,),
    )


# =============================================================================
# Convolution Loaders
# =============================================================================


def load_causal_conv1d(
    module: CausalConv1d,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> CausalConv1d:
    """Load a CausalConv1d module from weights.

    Supports two weight formats:
        1. Fused weights (after remove_weight_norm):
            - path/weight: shape (out_channels, in_channels // groups, kernel_size)
            - path/bias: shape (out_channels,) if module has biases

        2. Parametrized weight_norm (before remove_weight_norm):
            - path/parametrizations/weight/original0 (weight_g)
            - path/parametrizations/weight/original1 (weight_v)
            - path/bias

    The function auto-detects which format is present and handles accordingly.

    Args:
        module: The CausalConv1d module to load weights into.
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Base path for this module's weights.

    Returns:
        CausalConv1d module with loaded weights.
    """
    # Check if weights are in parametrized weight_norm format
    weight_norm_path = path / "parametrizations" / "weight" / "original0"
    if weight_norm_path in weights_dict:
        # Fuse weight_norm parameters
        weight, bias = _fuse_parametrized_weight_norm_conv1d(weights_dict, path, is_transposed=False)
    else:
        # Weights are already fused
        weight = weights_dict[path / "weight"]
        bias = weights_dict.get(path / "bias") if module.biases is not None else None

    return load_parameters(
        lambda m: (m.weights, m.biases),
        module,
        (weight, bias),
    )


def load_causal_transpose_conv1d(
    module: CausalTransposeConv1d,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> CausalTransposeConv1d:
    """Load a CausalTransposeConv1d module from weights.

    Supports two weight formats:
        1. Fused weights (after remove_weight_norm):
            - path/weight: shape (in_channels, out_channels // groups, kernel_size) in PyTorch format
            - path/bias: shape (out_channels,) if module has biases

        2. Parametrized weight_norm (before remove_weight_norm):
            - path/parametrizations/weight/original0 (weight_g)
            - path/parametrizations/weight/original1 (weight_v)
            - path/bias

    The weights are automatically transformed from PyTorch format to JAX format
    using transform_pytorch_transpose_conv_weights.

    Args:
        module: The CausalTransposeConv1d module to load weights into.
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Base path for this module's weights.

    Returns:
        CausalTransposeConv1d module with loaded weights.
    """
    # Check if weights are in parametrized weight_norm format
    weight_norm_path = path / "parametrizations" / "weight" / "original0"
    if weight_norm_path in weights_dict:
        # Fuse weight_norm parameters
        weight_pytorch, bias = _fuse_parametrized_weight_norm_conv1d(weights_dict, path, is_transposed=True)
    else:
        # Weights are already fused
        weight_pytorch = weights_dict[path / "weight"]
        bias = weights_dict.get(path / "bias") if module.biases is not None else None

    # Transform PyTorch weights to JAX format
    weight_jax = transform_pytorch_transpose_conv_weights(
        weight_pytorch,
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        groups=module.groups,
    )

    return load_parameters(
        lambda m: (m.weights, m.biases),
        module,
        (weight_jax, bias),
    )


# =============================================================================
# Residual Block Loaders
# =============================================================================


def load_residual_block(
    module: ResidualBlock,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> ResidualBlock:
    """Load a ResidualBlock module from weights.

    Expected weight structure at path:
        - input_activation.snake_act.alpha: Snake alpha for input activation
        - skip_activation.snake_act.alpha: Snake alpha for skip activation
        - input_conv.weight: Input convolution weights
        - input_conv.bias: Input convolution bias
        - skip_conv.weight: Skip convolution weights
        - skip_conv.bias: Skip convolution bias

    Note: In PyTorch NeMo, activations are wrapped in CodecActivation -> HalfSnake -> Snake.
    So the path from ResidualBlock is: input_activation.activation.snake_act.alpha

    Args:
        module: The ResidualBlock module to load weights into.
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Base path for this module's weights.

    Returns:
        ResidualBlock module with loaded weights.
    """
    input_activation = load_half_snake(
        module.input_activation,
        weights_dict,
        path / "input_activation" / "activation",
    )
    skip_activation = load_half_snake(
        module.skip_activation,
        weights_dict,
        path / "skip_activation" / "activation",
    )
    input_conv = load_causal_conv1d(
        module.input_conv,
        weights_dict,
        path / "input_conv" / "conv",
    )
    skip_conv = load_causal_conv1d(
        module.skip_conv,
        weights_dict,
        path / "skip_conv" / "conv",
    )

    return load_parameters(
        lambda m: (m.input_activation, m.skip_activation, m.input_conv, m.skip_conv),
        module,
        (input_activation, skip_activation, input_conv, skip_conv),
    )


def load_hifigan_res_block(
    module: HiFiGANResBlock,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> HiFiGANResBlock:
    """Load a HiFiGANResBlock module from weights.

    Expected weight structure at path:
        - res_blocks.0.input_activation.activation.snake_act.alpha
        - res_blocks.0.skip_activation.activation.snake_act.alpha
        - res_blocks.0.input_conv.conv.weight
        - res_blocks.0.input_conv.conv.bias
        - res_blocks.0.skip_conv.conv.weight
        - res_blocks.0.skip_conv.conv.bias
        - res_blocks.1...
        - ...

    Args:
        module: The HiFiGANResBlock module to load weights into.
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Base path for this module's weights.

    Returns:
        HiFiGANResBlock module with loaded weights.
    """
    res_blocks = tuple(
        load_residual_block(block, weights_dict, path / "res_blocks" / i) for i, block in enumerate(module.res_blocks)
    )

    return load_parameters(
        lambda m: (m.res_blocks,),
        module,
        (res_blocks,),
    )


def load_hifigan_res_layer(
    module: HiFiGANResLayer,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> HiFiGANResLayer:
    """Load a HiFiGANResLayer module from weights.

    Expected weight structure at path:
        - res_blocks.0.res_blocks.0.input_activation.activation.snake_act.alpha
        - res_blocks.0.res_blocks.0.skip_activation.activation.snake_act.alpha
        - res_blocks.0.res_blocks.0.input_conv.conv.weight
        - ... (nested HiFiGANResBlock structure)
        - res_blocks.1...
        - ...

    Args:
        module: The HiFiGANResLayer module to load weights into.
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Base path for this module's weights.

    Returns:
        HiFiGANResLayer module with loaded weights.
    """
    res_blocks = tuple(
        load_hifigan_res_block(block, weights_dict, path / "res_blocks" / i)
        for i, block in enumerate(module.res_blocks)
    )

    return load_parameters(
        lambda m: (m.res_blocks,),
        module,
        (res_blocks,),
    )


# =============================================================================
# Decoder Loaders
# =============================================================================


def load_causal_hifigan_decoder(
    module: CausalHiFiGANDecoder,
    weights_dict: Mapping[str, Array],
    path: ParameterPath | None = None,
) -> CausalHiFiGANDecoder:
    """Load a CausalHiFiGANDecoder module from weights.

    This function handles the complete loading of the HiFi-GAN decoder, including:
    - pre_conv: CausalConv1d for input projection
    - activations: List of HalfSnake activations (wrapped in CodecActivation)
    - upsample_convs: List of CausalTransposeConv1d for upsampling (up_sample_conv_layers in PyTorch)
    - res_layers: List of HiFiGANResLayer blocks
    - post_activation: HalfSnake activation (wrapped in CodecActivation)
    - post_conv: CausalConv1d for output projection

    Expected weight structure:
        pre_conv:
            - pre_conv.conv.weight
            - pre_conv.conv.bias

        activations (CodecActivation -> HalfSnake -> Snake):
            - activations.0.activation.snake_act.alpha
            - activations.1.activation.snake_act.alpha
            - ...

        upsample_convs (up_sample_conv_layers in PyTorch):
            - up_sample_conv_layers.0.conv.weight
            - up_sample_conv_layers.0.conv.bias
            - up_sample_conv_layers.1.conv.weight
            - ...

        res_layers:
            - res_layers.0.res_blocks.0.res_blocks.0.input_activation.activation.snake_act.alpha
            - res_layers.0.res_blocks.0.res_blocks.0.input_conv.conv.weight
            - ... (nested structure)

        post_activation:
            - post_activation.activation.snake_act.alpha

        post_conv:
            - post_conv.conv.weight
            - post_conv.conv.bias

    Args:
        module: The CausalHiFiGANDecoder module to load weights into.
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Optional base path. If None, uses root path.

    Returns:
        CausalHiFiGANDecoder module with loaded weights.
    """
    base_path = ParameterPath() if path is None else path

    # Load pre_conv
    pre_conv = load_causal_conv1d(
        module.pre_conv,
        weights_dict,
        base_path / "pre_conv" / "conv",
    )

    # Load activations (CodecActivation -> HalfSnake)
    activations = tuple(
        load_half_snake(act, weights_dict, base_path / "activations" / i / "activation")
        for i, act in enumerate(module.activations)
    )

    # Load upsample_convs (called up_sample_conv_layers in PyTorch)
    upsample_convs = tuple(
        load_causal_transpose_conv1d(conv, weights_dict, base_path / "up_sample_conv_layers" / i / "conv")
        for i, conv in enumerate(module.upsample_convs)
    )

    # Load res_layers
    res_layers = tuple(
        load_hifigan_res_layer(layer, weights_dict, base_path / "res_layers" / i)
        for i, layer in enumerate(module.res_layers)
    )

    # Load post_activation
    post_activation = load_half_snake(
        module.post_activation,
        weights_dict,
        base_path / "post_activation" / "activation",
    )

    # Load post_conv
    post_conv = load_causal_conv1d(
        module.post_conv,
        weights_dict,
        base_path / "post_conv" / "conv",
    )

    return load_parameters(
        lambda m: (
            m.pre_conv,
            m.activations,
            m.upsample_convs,
            m.res_layers,
            m.post_activation,
            m.post_conv,
        ),
        module,
        (pre_conv, activations, upsample_convs, res_layers, post_activation, post_conv),
    )


# =============================================================================
# Full Model Loaders
# =============================================================================


def load_nanocodec(
    module: NanoCodec,
    weights_dict: Mapping[str, Array],
) -> NanoCodec:
    """Load a NanoCodec model from weights.

    Loads the CausalHiFiGANDecoder weights from the audio_decoder prefix.
    The GroupFiniteScalarQuantizer uses FSQ which has no learnable weights -
    its buffers are computed from the config.

    Expected weight structure (from NeMo AudioCodecModel state_dict):
        audio_decoder:
            - audio_decoder.pre_conv.conv.weight
            - audio_decoder.pre_conv.conv.bias
            - audio_decoder.activations.0.activation.snake_act.alpha
            - audio_decoder.up_sample_conv_layers.0.conv.weight
            - audio_decoder.res_layers.0.res_blocks.0.res_blocks.0...
            - audio_decoder.post_activation.activation.snake_act.alpha
            - audio_decoder.post_conv.conv.weight
            - audio_decoder.post_conv.conv.bias

        vector_quantizer (not loaded - FSQ has no learnable weights):
            - vector_quantizer.fsqs.0.num_levels (buffer, not weight)
            - vector_quantizer.fsqs.0.dim_base_index (buffer, not weight)

    Args:
        module: The NanoCodec module to load weights into.
        weights_dict: Dictionary mapping parameter paths to weight arrays.
            Should contain keys with "audio_decoder." prefix.

    Returns:
        NanoCodec module with loaded decoder weights.
    """
    # Load decoder with "audio_decoder" prefix
    decoder = load_causal_hifigan_decoder(
        module.decoder,
        weights_dict,
        ParameterPath("audio_decoder"),
    )

    return load_parameters(
        lambda m: (m.decoder,),
        module,
        (decoder,),
    )
