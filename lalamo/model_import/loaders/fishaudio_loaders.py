from collections.abc import Mapping
from typing import Any

import torch
from jax import numpy as jnp
from jaxtyping import Array
from torch import nn
from torch.nn.utils import remove_weight_norm, weight_norm
from torch.nn.utils.parametrizations import weight_norm as param_weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from lalamo.common import ParameterPath
from lalamo.modules import (
    Attention,
    DenseMLP,
    FullPrecisionLinear,
    LayerScale,
    MLPBase,
    Normalization,
    Transformer,
    TransformerLayer,
)
from lalamo.modules.audio.fishaudio import (
    DescriptAudioCodec,
    DescriptAudioCodecConfig,
)
from lalamo.modules.audio.fishaudio.fishaudio_common import get_default_fishaudio_dac_config
from lalamo.modules.audio.fishaudio.fishaudio_modules import (
    CausalConv1d,
    CausalTransposeConv1d,
    ConvNeXtBlock,
    DACDecoder,
    DACDecoderBlock,
    DownsampleResidualVectorQuantize,
    ResidualUnit,
    ResidualVectorQuantize,
    Snake1d,
    Upsampler,
    UpsamplingBlock,
    VectorQuantize,
)
from lalamo.modules.torch_interop import jax_to_torch

from .common import load_parameters
from .huggingface import load_linear, load_rmsnorm


def _fuse_full_precision_weights(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    sublayers_to_fuse: list[str] | None,
) -> Array:
    if sublayers_to_fuse is None:
        return weights_dict[path / "weight"]

    weights = [weights_dict[path / layer_name / "weight"] for layer_name in sublayers_to_fuse]
    return jnp.concatenate(weights, axis=0)


def _fuse_weight_norm_conv1d(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> tuple[Array, Array | None]:
    """Fuse weight normalization for a Conv1d layer using PyTorch's remove_weight_norm.

    Creates a temporary PyTorch Conv1d module, applies weight_norm, loads the weight_g
    and weight_v parameters, then calls remove_weight_norm to get the fused weight.

    Args:
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Path to the weight-normalized layer (e.g., "quantizers/0/in_proj").

    Returns:
        Tuple of (fused_weight, bias) as JAX arrays.
    """
    weight_g = weights_dict[path / "weight_g"]
    weight_v = weights_dict[path / "weight_v"]
    bias = weights_dict[path / "bias"]

    # weight_g shape: (out_channels, 1, 1) for Conv1d
    # weight_v shape: (out_channels, in_channels, kernel_size)
    out_channels, in_channels, kernel_size = weight_v.shape

    # Create a temporary Conv1d and apply weight_norm
    temp_conv = nn.Conv1d(in_channels, out_channels, kernel_size)
    temp_conv = weight_norm(temp_conv, name="weight", dim=0)

    # Load the weight_g and weight_v parameters
    with torch.no_grad():
        temp_conv.weight_g = torch.nn.Parameter(jax_to_torch(weight_g), requires_grad=False)
        temp_conv.weight_v = torch.nn.Parameter(jax_to_torch(weight_v), requires_grad=False)
        if bias is not None:
            temp_conv.bias = torch.nn.Parameter(jax_to_torch(bias), requires_grad=False)

    # Fuse with remove_weight_norm
    temp_conv = remove_weight_norm(temp_conv, name="weight")

    # Extract fused weight and convert back to JAX array
    fused_weight = jnp.array(temp_conv.weight.detach().numpy())
    fused_bias = jnp.array(temp_conv.bias.detach().numpy()) if temp_conv.bias is not None else None

    return fused_weight, fused_bias


def _fuse_parametrized_weight_norm_conv1d(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    is_transposed: bool = False,
) -> tuple[Array, Array | None]:
    """Fuse weight normalization for a Conv1d layer using PyTorch's remove_parametrizations.

    This handles the newer parametrization format where weights are stored as:
        - path/parametrizations/weight/original0 (weight_g)
        - path/parametrizations/weight/original1 (weight_v)
        - path/bias

    Args:
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Path to the weight-normalized layer.
        is_transposed: If True, creates ConvTranspose1d instead of Conv1d.

    Returns:
        Tuple of (fused_weight, bias) as JAX arrays.
    """
    weight_g = weights_dict[path / "parametrizations" / "weight" / "original0"]
    weight_v = weights_dict[path / "parametrizations" / "weight" / "original1"]
    bias = weights_dict[path / "bias"]

    if is_transposed:
        # ConvTranspose1d weight shape: (in_channels, out_channels, kernel_size)
        in_channels, out_channels, kernel_size = weight_v.shape
        temp_conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size)
    else:
        # Conv1d weight shape: (out_channels, in_channels, kernel_size)
        out_channels, in_channels, kernel_size = weight_v.shape
        temp_conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    # Apply weight_norm to match the parametrization
    temp_conv = param_weight_norm(temp_conv, name="weight", dim=0)

    # Load the weight_g and weight_v parameters
    with torch.no_grad():
        temp_conv.parametrizations.weight.original0 = torch.nn.Parameter(jax_to_torch(weight_g), requires_grad=False)
        temp_conv.parametrizations.weight.original1 = torch.nn.Parameter(jax_to_torch(weight_v), requires_grad=False)
        if bias is not None:
            temp_conv.bias = torch.nn.Parameter(jax_to_torch(bias), requires_grad=False)

    # Fuse with remove_parametrizations
    remove_parametrizations(temp_conv, "weight")

    # Extract fused weight and convert back to JAX array
    fused_weight = jnp.array(temp_conv.weight.detach().numpy())
    fused_bias = jnp.array(temp_conv.bias.detach().numpy()) if temp_conv.bias is not None else None

    return fused_weight, fused_bias


def load_layer_norm(
    module: LayerScale,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> LayerScale:
    scales = weights_dict[path / "gamma"]
    return load_parameters(lambda m: (m.scales,), module, (scales,))


def load_transformer_block(
    module: Transformer, weights_dict: Mapping[str, Array], fast: bool = False, path: ParameterPath | None = None
) -> Transformer:
    def load_attention_local(
        module: Attention,
        weights_dict: Mapping[str, Array],
        path: ParameterPath,
    ) -> Attention:
        qkv_projection = load_linear(
            module.qkv_projection,
            weights_dict,
            path / "wqkv",
            sublayers_to_fuse=None,
        )
        out_projection = load_linear(module.out_projection, weights_dict, path / "wo")

        if module.query_norm is not None:
            query_norm = load_rmsnorm(module.query_norm, weights_dict, path / "q_norm")
        else:
            query_norm = None

        if module.key_norm is not None:
            key_norm = load_rmsnorm(module.key_norm, weights_dict, path / "k_norm")
        else:
            key_norm = None

        return load_parameters(
            lambda m: (m.qkv_projection, m.out_projection, m.query_norm, m.key_norm),
            module,
            (qkv_projection, out_projection, query_norm, key_norm),
        )

    def load_mlp(
        module: MLPBase,
        weights_dict: Mapping[str, Array],
        path: ParameterPath,
        up_proj_key: str,
        gate_proj_key: str,
        down_proj_key: str,
    ) -> MLPBase:
        assert isinstance(module, DenseMLP)
        # Standard dense MLP with separate sublayers.
        up_projection = load_linear(
            module.up_projection,
            weights_dict,
            path,
            sublayers_to_fuse=[up_proj_key, gate_proj_key],
        )
        down_projection = load_linear(module.down_projection, weights_dict, path / down_proj_key)
        return load_parameters(
            lambda m: (m.up_projection, m.down_projection),
            module,
            (up_projection, down_projection),
        )

    def load_transformer_layer_local(
        module: TransformerLayer,
        weights_dict: Mapping[str, Array],
        path: ParameterPath,
    ) -> TransformerLayer:
        if module.pre_mixer_norm is not None:
            assert isinstance(module.pre_mixer_norm, Normalization)
            pre_mixer_norm = load_rmsnorm(
                module.pre_mixer_norm,
                weights_dict,
                path / "attention_norm",
            )
        else:
            pre_mixer_norm = None

        if module.post_mixer_norm is not None:
            assert isinstance(module.post_mixer_norm, LayerScale)
            post_mixer_norm = load_layer_norm(module.post_mixer_norm, weights_dict, path / "attention_layer_scale")
        else:
            post_mixer_norm = None

        assert isinstance(module.mixer, Attention)
        attention = load_attention_local(module.mixer, weights_dict, path / "attention")

        assert isinstance(module.pre_mlp_norm, Normalization)
        pre_mlp_norm = load_rmsnorm(
            module.pre_mlp_norm,
            weights_dict,
            path / "ffn_norm",
        )
        if module.post_mlp_norm is not None:
            assert isinstance(module.post_mlp_norm, LayerScale)
            post_mlp_norm = load_layer_norm(module.post_mlp_norm, weights_dict, path / "ffn_layer_scale")
        else:
            post_mlp_norm = None

        mlp = load_mlp(module.mlp, weights_dict, path / "feed_forward", "w3", "w1", "w2")

        return load_parameters(
            lambda m: (
                m.pre_mixer_norm,
                m.mixer,
                m.post_mixer_norm,
                m.pre_mlp_norm,
                m.mlp,
                m.post_mlp_norm,
            ),
            module,
            (
                pre_mixer_norm,
                attention,
                post_mixer_norm,
                pre_mlp_norm,
                mlp,
                post_mlp_norm,
            ),
        )

    base_path = ParameterPath() if path is None else path

    layers_name = "layers" if not fast else "fast_layers"
    norm_name = "norm" if not fast else "fast_norm"

    transformer_layers = tuple(
        load_transformer_layer_local(layer, weights_dict, base_path / layers_name / i)
        for i, layer in enumerate(module.layers)
    )
    output_norm = load_rmsnorm(module.output_norm, weights_dict, base_path / norm_name)

    module = load_parameters(
        lambda m: (
            m.layers,
            m.output_norm,
        ),
        module,
        (
            transformer_layers,
            output_norm,
        ),
    )

    return module


def load_fish_audio_text_decoding_modules(
    transformer: Transformer, output: FullPrecisionLinear, weights_dict: Mapping[str, Array], fast: bool = False
) -> tuple[Transformer, FullPrecisionLinear]:
    transformer = load_transformer_block(transformer, weights_dict=weights_dict, fast=fast)

    base_path = ParameterPath()
    output_linear_name = "output" if not fast else "fast_output"
    output_linear = load_linear(output, weights_dict, base_path / output_linear_name)
    output = load_parameters(
        lambda m: (m,),
        output,
        (output_linear,),
    )

    return (transformer, output)


def load_vector_quantize(
    module: VectorQuantize,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> VectorQuantize:
    """Loads a VectorQuantize module from weights.

    The in_proj and out_proj layers use weight normalization in the original PyTorch model.
    This function fuses the weight_g and weight_v parameters using PyTorch's remove_weight_norm.

    Expected weight structure at path:
        - in_proj.weight_g, in_proj.weight_v, in_proj.bias (unused in decode-only mode)
        - out_proj.weight_g, out_proj.weight_v, out_proj.bias
        - codebook.weight

    Args:
        module: The VectorQuantize module to load weights into.
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Base path for this module's weights.

    Returns:
        VectorQuantize module with loaded weights.
    """
    # Load codebook weights
    codebook_weight = weights_dict[path / "codebook" / "weight"]
    codebook = load_parameters(
        lambda m: (m.weights,),
        module.codebook,
        (codebook_weight,),
    )

    # Load out_proj with weight norm fusion
    # The original is a Conv1d with kernel_size=1, so weight shape is (out, in, 1)
    # Our FullPrecisionLinear expects (out, in), so we squeeze the last dim
    out_proj_weight, out_proj_bias = _fuse_weight_norm_conv1d(weights_dict, path / "out_proj")
    # Squeeze kernel dimension: (out_channels, in_channels, 1) -> (out_channels, in_channels)
    out_proj_weight = jnp.squeeze(out_proj_weight, axis=-1)
    out_proj = load_parameters(
        lambda m: (m.weights, m.biases),
        module.out_proj,
        (out_proj_weight, out_proj_bias),
    )

    return load_parameters(
        lambda m: (m.codebook, m.out_proj),
        module,
        (codebook, out_proj),
    )


def load_residual_vector_quantize(
    module: ResidualVectorQuantize,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> ResidualVectorQuantize:
    """Loads a ResidualVectorQuantize module from weights.

    Expected weight structure at path:
        - quantizers.0.in_proj.weight_g, quantizers.0.in_proj.weight_v, etc.
        - quantizers.0.out_proj.weight_g, quantizers.0.out_proj.weight_v, etc.
        - quantizers.0.codebook.weight
        - quantizers.1..., quantizers.2..., etc.

    Args:
        module: The ResidualVectorQuantize module to load weights into.
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Base path for this module's weights.

    Returns:
        ResidualVectorQuantize module with loaded weights.
    """
    quantizers = tuple(
        load_vector_quantize(quantizer, weights_dict, path / "quantizers" / i)
        for i, quantizer in enumerate(module.quantizers)
    )

    return load_parameters(
        lambda m: (m.quantizers,),
        module,
        (quantizers,),
    )


def load_convnext_block(
    module: ConvNeXtBlock,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> ConvNeXtBlock:
    """Loads a ConvNeXtBlock module from weights.

    Expected weight structure at path:
        - gamma (LayerScale)
        - dwconv.conv.weight
        - dwconv.conv.bias
        - norm.weight
        - norm.bias
        - pwconv1.weight
        - pwconv1.bias
        - pwconv2.weight
        - pwconv2.bias

    Args:
        module: The ConvNeXtBlock module to load weights into.
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Base path for this module's weights.

    Returns:
        ConvNeXtBlock module with loaded weights.
    """
    # Load LayerScale (gamma)
    if module.scale is not None:
        gamma_scales = weights_dict[path / "gamma"]
        scale = load_parameters(
            lambda m: (m.scales,),
            module.scale,
            (gamma_scales,),
        )
    else:
        scale = None

    # Load depthwise conv
    # PyTorch conv weights are (out_channels, in_channels/groups, kernel_size)
    dwconv_weight = weights_dict[path / "dwconv" / "conv" / "weight"]
    dwconv_bias = weights_dict[path / "dwconv" / "conv" / "bias"]
    depthwise_conv = load_parameters(
        lambda m: (m.weights, m.biases),
        module.depthwise_conv,
        (dwconv_weight, dwconv_bias),
    )

    # Load norm (LayerNorm with weight and bias)
    norm_weight = weights_dict[path / "norm" / "weight"]
    norm_bias = weights_dict[path / "norm" / "bias"]
    norm = load_parameters(
        lambda m: (m.scales, m.bias),
        module.norm,
        (norm_weight, norm_bias),
    )

    # Load pointwise conv 1 (Linear layer)
    # PyTorch Linear weight is (out_features, in_features)
    pwconv1_weight = weights_dict[path / "pwconv1" / "weight"]
    pwconv1_bias = weights_dict[path / "pwconv1" / "bias"]
    pointwise_conv_step1 = load_parameters(
        lambda m: (m.weights, m.biases),
        module.pointwise_conv_step1,
        (pwconv1_weight, pwconv1_bias),
    )

    # Load pointwise conv 2 (Linear layer)
    pwconv2_weight = weights_dict[path / "pwconv2" / "weight"]
    pwconv2_bias = weights_dict[path / "pwconv2" / "bias"]
    pointwise_conv_step2 = load_parameters(
        lambda m: (m.weights, m.biases),
        module.pointwise_conv_step2,
        (pwconv2_weight, pwconv2_bias),
    )

    return load_parameters(
        lambda m: (m.scale, m.depthwise_conv, m.norm, m.pointwise_conv_step1, m.pointwise_conv_step2),
        module,
        (scale, depthwise_conv, norm, pointwise_conv_step1, pointwise_conv_step2),
    )


def load_upsampling_block(
    module: UpsamplingBlock,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> UpsamplingBlock:
    """Loads an UpsamplingBlock module from weights.

    Expected weight structure at path:
        - 0.conv.weight (transpose conv)
        - 0.conv.bias
        - 1.gamma (ConvNeXt LayerScale)
        - 1.dwconv.conv.weight
        - 1.dwconv.conv.bias
        - 1.norm.weight
        - 1.norm.bias
        - 1.pwconv1.weight
        - 1.pwconv1.bias
        - 1.pwconv2.weight
        - 1.pwconv2.bias

    Args:
        module: The UpsamplingBlock module to load weights into.
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Base path for this module's weights.

    Returns:
        UpsamplingBlock module with loaded weights.
    """
    # Load transpose conv (at index 0)
    trans_conv_weight = weights_dict[path / "0" / "conv" / "weight"]
    trans_conv_bias = weights_dict[path / "0" / "conv" / "bias"]
    trans_conv = load_parameters(
        lambda m: (m.weights, m.biases),
        module.trans_conv,
        (trans_conv_weight, trans_conv_bias),
    )

    # Load ConvNeXt block (at index 1)
    convnext = load_convnext_block(module.convnext, weights_dict, path / "1")

    return load_parameters(
        lambda m: (m.trans_conv, m.convnext),
        module,
        (trans_conv, convnext),
    )


def load_upsampler(
    module: Upsampler,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> Upsampler:
    """Loads an Upsampler module from weights.

    Expected weight structure at path:
        - 0.0.conv.weight, 0.1.gamma, etc. (first UpsamplingBlock)
        - 1.0.conv.weight, 1.1.gamma, etc. (second UpsamplingBlock)
        - ...

    Args:
        module: The Upsampler module to load weights into.
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Base path for this module's weights.

    Returns:
        Upsampler module with loaded weights.
    """
    blocks = tuple(load_upsampling_block(block, weights_dict, path / i) for i, block in enumerate(module.blocks))

    return load_parameters(
        lambda m: (m.blocks,),
        module,
        (blocks,),
    )


def load_downsample_rvq(
    module: DownsampleResidualVectorQuantize, weights_dict: Mapping[str, Array], path: ParameterPath | None = None
) -> DownsampleResidualVectorQuantize:
    """Loads a DownsampleResidualVectorQuantize module from weights.

    This function handles the complete loading of the audio decoder module, including:
    - semantic_quantizer: ResidualVectorQuantize with weight-normalized projections
    - quantizer: ResidualVectorQuantize with weight-normalized projections
    - upsample: Upsampler with transpose convolutions and ConvNeXt blocks
    - post_module: Transformer for post-processing

    Weight normalization is fused using PyTorch's remove_weight_norm internally.

    Expected weight structure:
        semantic_quantizer:
            - semantic_quantizer.quantizers.0.in_proj.weight_g/weight_v/bias
            - semantic_quantizer.quantizers.0.out_proj.weight_g/weight_v/bias
            - semantic_quantizer.quantizers.0.codebook.weight
            - ... (for each quantizer)

        quantizer:
            - quantizer.quantizers.0.in_proj.weight_g/weight_v/bias
            - quantizer.quantizers.0.out_proj.weight_g/weight_v/bias
            - quantizer.quantizers.0.codebook.weight
            - ... (for each quantizer)

        upsample (UpsamplingBlocks):
            - upsample.0.0.conv.weight (transpose conv)
            - upsample.0.0.conv.bias
            - upsample.0.1.gamma (ConvNeXt LayerScale)
            - upsample.0.1.dwconv.conv.weight
            - upsample.0.1.dwconv.conv.bias
            - upsample.0.1.norm.weight
            - upsample.0.1.norm.bias
            - upsample.0.1.pwconv1.weight
            - upsample.0.1.pwconv1.bias
            - upsample.0.1.pwconv2.weight
            - upsample.0.1.pwconv2.bias
            - ... (for each upsampling block)

        post_module (Transformer):
            - [SKIP] post_module.freqs_cis (generated internally, not loaded)
            - [SKIP] post_module.causal_mask (generated internally, not loaded)
            - post_module.layers.0.attention.wqkv.weight
            - post_module.layers.0.attention.wo.weight
            - post_module.layers.0.feed_forward.w1.weight
            - post_module.layers.0.feed_forward.w3.weight
            - post_module.layers.0.feed_forward.w2.weight
            - post_module.layers.0.ffn_norm.weight
            - post_module.layers.0.attention_norm.weight
            - post_module.layers.0.attention_layer_scale.gamma
            - post_module.layers.0.ffn_layer_scale.gamma
            - ... (for each layer)
            - post_module.norm.weight

    Args:
        module: The DownsampleResidualVectorQuantize module to load weights into.
        weights_dict: Dictionary mapping parameter paths to weight arrays.

    Returns:
        DownsampleResidualVectorQuantize module with loaded weights.
    """
    base_path = ParameterPath() if path is None else path

    semantic_quantizer = load_residual_vector_quantize(
        module.semantic_quantizer,
        weights_dict,
        base_path / "semantic_quantizer",
    )

    quantizer = load_residual_vector_quantize(
        module.quantizer,
        weights_dict,
        base_path / "quantizer",
    )

    upsampler = load_upsampler(
        module.upsampler,
        weights_dict,
        base_path / "upsample",
    )

    post_module = load_transformer_block(module.post_module, weights_dict, fast=False, path=base_path / "post_module")

    return load_parameters(
        lambda m: (m.semantic_quantizer, m.quantizer, m.upsampler, m.post_module),
        module,
        (semantic_quantizer, quantizer, upsampler, post_module),
    )


def load_snake1d(
    module: Snake1d,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> Snake1d:
    """Loads a Snake1d module from weights.

    Expected weight structure at path:
        - alpha (shape: [1, channels, 1] in PyTorch)

    The PyTorch Snake1d stores alpha as (1, channels, 1) but our module
    stores it as (channels,), so we squeeze the extra dimensions.

    Args:
        module: The Snake1d module to load weights into.
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Base path for this module's weights.

    Returns:
        Snake1d module with loaded weights.
    """
    alpha = weights_dict[path / "alpha"]
    # PyTorch shape: (1, channels, 1) -> squeeze to (channels,)
    alpha = jnp.squeeze(alpha)

    return load_parameters(
        lambda m: (m.alpha,),
        module,
        (alpha,),
    )


def load_weight_norm_conv1d(
    module: CausalConv1d | CausalTransposeConv1d,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> CausalConv1d | CausalTransposeConv1d:
    """Loads a Conv1d or TransposeConv1d module from weight-normalized PyTorch weights.

    Weight normalization fusion is mathematically identical for both Conv1d and
    ConvTranspose1d: weight = weight_g * weight_v / ||weight_v||

    Expected weight structure at path (parametrized format):
        - conv.parametrizations.weight.original0 (weight_g)
        - conv.parametrizations.weight.original1 (weight_v)
        - conv.bias

    Args:
        module: The CausalConv1d or CausalTransposeConv1d module to load weights into.
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Base path for this module's weights.

    Returns:
        Module with loaded weights.
    """
    is_transposed = isinstance(module, CausalTransposeConv1d)
    fused_weight, fused_bias = _fuse_parametrized_weight_norm_conv1d(
        weights_dict, path / "conv", is_transposed=is_transposed
    )

    return load_parameters(
        lambda m: (m.weights, m.biases),
        module,
        (fused_weight, fused_bias),
    )


def load_residual_unit(
    module: ResidualUnit,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> ResidualUnit:
    """Loads a ResidualUnit module from weights.

    Expected weight structure at path (PyTorch block is nn.Sequential):
        - block.0.alpha (Snake1d)
        - block.1.conv.parametrizations.weight.original0/1, block.1.conv.bias (CausalWNConv1d, kernel=7)
        - block.2.alpha (Snake1d)
        - block.3.conv.parametrizations.weight.original0/1, block.3.conv.bias (CausalWNConv1d, kernel=1)

    Args:
        module: The ResidualUnit module to load weights into.
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Base path for this module's weights.

    Returns:
        ResidualUnit module with loaded weights.
    """
    snake1 = load_snake1d(module.snake1, weights_dict, path / "block" / "0")
    conv1 = load_weight_norm_conv1d(module.conv1, weights_dict, path / "block" / "1")
    snake2 = load_snake1d(module.snake2, weights_dict, path / "block" / "2")
    conv2 = load_weight_norm_conv1d(module.conv2, weights_dict, path / "block" / "3")

    return load_parameters(
        lambda m: (m.snake1, m.conv1, m.snake2, m.conv2),
        module,
        (snake1, conv1, snake2, conv2),
    )


def load_audio_decoder_block(
    module: DACDecoderBlock,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> DACDecoderBlock:
    """Loads an AudioDecoderBlock module from weights.

    Expected weight structure at path (PyTorch block is nn.Sequential):
        - block.0.alpha (Snake1d)
        - block.1.conv.parametrizations.weight.original0/1, block.1.conv.bias (CausalWNConvTranspose1d)
        - block.2.block.0..3 (ResidualUnit, dilation=1)
        - block.3.block.0..3 (ResidualUnit, dilation=3)
        - block.4.block.0..3 (ResidualUnit, dilation=9)

    Args:
        module: The AudioDecoderBlock module to load weights into.
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Base path for this module's weights.

    Returns:
        AudioDecoderBlock module with loaded weights.
    """
    snake = load_snake1d(module.snake, weights_dict, path / "block" / "0")
    trans_conv = load_weight_norm_conv1d(module.trans_conv, weights_dict, path / "block" / "1")
    res_unit1 = load_residual_unit(module.res_unit1, weights_dict, path / "block" / "2")
    res_unit2 = load_residual_unit(module.res_unit2, weights_dict, path / "block" / "3")
    res_unit3 = load_residual_unit(module.res_unit3, weights_dict, path / "block" / "4")

    return load_parameters(
        lambda m: (m.snake, m.trans_conv, m.res_unit1, m.res_unit2, m.res_unit3),
        module,
        (snake, trans_conv, res_unit1, res_unit2, res_unit3),
    )


def load_audio_decoder(
    module: DACDecoder,
    weights_dict: Mapping[str, Array],
    path: ParameterPath | None = None,
) -> DACDecoder:
    """Loads an AudioDecoder module from weights.

    This function handles the complete loading of the DAC-style audio decoder, including:
    - first_conv: CausalConv1d with weight normalization
    - decoder_blocks: Multiple AudioDecoderBlock instances
    - final_snake: Snake1d activation
    - final_conv: CausalConv1d with weight normalization

    The PyTorch Decoder stores layers in a single nn.Sequential `model`:
        - model.0: First conv (CausalWNConv1d)
        - model.1 to model.N: DecoderBlocks
        - model.N+1: Final Snake1d
        - model.N+2: Final conv (CausalWNConv1d)
        - model.N+3: Tanh (no weights)

    Expected weight structure (parametrized format):
        model.0.conv.parametrizations.weight.original0/1, model.0.conv.bias
        model.1.block.0.alpha (Snake1d in first DecoderBlock)
        model.1.block.1.conv.parametrizations.weight.original0/1, ... (TransposeConv in first DecoderBlock)
        model.1.block.2.block.0..3 (ResidualUnit in first DecoderBlock)
        ...
        model.N+1.alpha (Final Snake1d)
        model.N+2.conv.parametrizations.weight.original0/1, model.N+2.conv.bias

    Args:
        module: The AudioDecoder module to load weights into.
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Optional base path. If None, uses root path.

    Returns:
        AudioDecoder module with loaded weights.
    """
    if path is None:
        path = ParameterPath()

    # model.0 is the first conv
    first_conv = load_weight_norm_conv1d(module.first_conv, weights_dict, path / "model" / "0")

    # model.1 to model.N are decoder blocks
    num_blocks = len(module.decoder_blocks)
    decoder_blocks = tuple(
        load_audio_decoder_block(block, weights_dict, path / "model" / (i + 1))
        for i, block in enumerate(module.decoder_blocks)
    )

    # model.N+1 is final snake (where N = num_blocks)
    final_snake_idx = num_blocks + 1
    final_snake = load_snake1d(module.final_snake, weights_dict, path / "model" / final_snake_idx)

    # model.N+2 is final conv
    final_conv_idx = num_blocks + 2
    final_conv = load_weight_norm_conv1d(module.final_conv, weights_dict, path / "model" / final_conv_idx)

    return load_parameters(
        lambda m: (m.first_conv, m.decoder_blocks, m.final_snake, m.final_conv),
        module,
        (first_conv, decoder_blocks, final_snake, final_conv),
    )


def load_descript_audio_codec(state_dict: Mapping[str, Any]) -> DescriptAudioCodec:
    dac_config = DescriptAudioCodecConfig.instantiate_config_from_fishaudio_config(
        fish_dac_config=get_default_fishaudio_dac_config()
    )
    dac_module = dac_config.empty()

    loaded_quantizer = load_downsample_rvq(dac_module.quantizer, state_dict, path=ParameterPath("quantizer"))
    loaded_decoder = load_audio_decoder(dac_module.decoder, state_dict, path=ParameterPath("decoder"))

    return DescriptAudioCodec(
        config=dac_module.config,
        quantizer=loaded_quantizer,
        decoder=loaded_decoder,
    )
