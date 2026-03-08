import base64
import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from einops import rearrange
from jax import numpy as jnp
from jaxtyping import Array, Float
from tokenizers import Tokenizer

from lalamo.common import ParameterPath
from lalamo.modules import (
    Attention,
    DenseMLP,
    FullPrecisionLinear,
    Identity,
    LinearBase,
    MLPBase,
    Normalization,
    Transformer,
    TransformerLayer,
)
from lalamo.modules.audio.fishaudio import DescriptAudioCodec, FishAudioTextDecoder
from lalamo.modules.audio.fishaudio.fishaudio_common import (
    FishAudioSpecialInferenceTokens,
)
from lalamo.modules.audio.fishaudio.fishaudio_consts import (
    FISH_TIKTOKEN_PATTERN,
    IM_END_TOKEN,
)
from lalamo.modules.audio.fishaudio.fishaudio_modules import (
    ConvNeXtBlock,
    DACDecoder,
    DACDecoderBlock,
    DownsampleResidualVectorQuantize,
    ResidualUnit,
    ResidualVectorQuantize,
    Upsampler,
    UpsamplingBlock,
    VectorQuantize,
)

from .common import load_parameters
from .huggingface import load_rmsnorm, load_tied_embedding
from .nanocodec_loaders import (
    load_causal_conv1d,
    load_causal_transpose_conv1d,
    load_snake1d,
)
from .torch_utils import fuse_weight_norm_conv1d_as_linear


def _permute_for_rope_rotate_half(
    weight: Array,
    num_heads: int,
    head_dim: int,
) -> Array:
    """Permute weight matrix from interleaved RoPE format to rotate-half format.

    This converts weights trained with the interleaved RoPE format:
        interleaved: [-x1, x0, -x3, x2, -x5, x4, ...]
    to the rotate-half format used by standard RoPE:
        rotate-half: [-x_{d/2}, -x_{d/2+1}, ..., x_0, x_1, ...]

    The transformation reorders the output dimensions of Q/K projections so that
    the first half of each head's dimensions and the second half are grouped together,
    rather than being interleaved.

    Code is inspired by similar transformation from:
    https://github.com/huggingface/transformers/blob/e42587f596181396e1c4b63660abf0c736b10dae/src/transformers/models/llama/convert_llama_weights_to_hf.py
    """
    if len(weight.shape) == 1:
        # For 1D vectors: swap interleaved pairs to grouped halves
        return rearrange(weight, "(half_dim pair) -> (pair half_dim)", pair=2)

    out_features, _ = weight.shape
    assert out_features == num_heads * head_dim, (
        f"Output features {out_features} must equal num_heads * head_dim = {num_heads * head_dim}"
    )
    # For 2D matrices: swap interleaved pairs to grouped halves within each head
    return rearrange(
        weight,
        "(heads half_dim pair) in_features -> (heads pair half_dim) in_features",
        heads=num_heads,
        pair=2,
    )


def _permute_qkv_for_rope_rotate_half(
    qkv_weight: Float[Array, "q_dim+k_dim+v_dim in_features"],
    num_heads: int,
    num_groups: int,
    head_dim: int,
) -> Array:
    """Permute fused QKV weight matrix from interleaved RoPE to rotate-half format.

    For grouped query attention (GQA), Q has num_heads while K/V have num_groups.
    Only Q and K need permutation (they use RoPE). V is unchanged.

    Args:
        qkv_weight: Fused QKV weight of shape (q_dim + k_dim + v_dim, in_features)
                    where q_dim = num_heads * head_dim, k_dim = v_dim = num_groups * head_dim
        num_heads: Number of query heads.
        num_groups: Number of key/value heads (groups for GQA).
        head_dim: Dimension per head.

    Returns:
        Permuted QKV weight with Q and K converted to rotate-half format.
    """
    q_dim = num_heads * head_dim
    k_dim = num_groups * head_dim

    q_weight = qkv_weight[:q_dim, :]
    k_weight = qkv_weight[q_dim : q_dim + k_dim, :]
    v_weight = qkv_weight[q_dim + k_dim :, :]

    q_weight = _permute_for_rope_rotate_half(q_weight, num_heads, head_dim)
    k_weight = _permute_for_rope_rotate_half(k_weight, num_groups, head_dim)

    return jnp.concatenate([q_weight, k_weight, v_weight], axis=0)


def _fuse_full_precision_weights(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    sublayers_to_fuse: list[str] | None,
) -> Array:
    if sublayers_to_fuse is None:
        return weights_dict[path / "weight"]

    weights = [weights_dict[path / layer_name / "weight"] for layer_name in sublayers_to_fuse]
    return jnp.concatenate(weights, axis=0)


def load_linear_and_fuse_scaling(
    module: LinearBase,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    sublayers_to_fuse: list[str] | None = None,
    scaling_to_fuse: Array | None = None,
) -> LinearBase:
    """Load linear layer directly or fuse several sum-matrices into one linear layer.
    Additionally fuse final result with scaling weights that would follow after the layer.
    Args:
        module: target linear module into which weights will be loaded
        weights_dict: mapping with weights
        path: path to linear layer within the given weights mapping
        sublayers_to_fuse: optional list of names of matrices that we want to fuse into single linear layer
        scaling_to_fuze: optional array of scales we want to fuse into given linear module

    Returns:
        Linear layer with weights loaded into it
    """
    assert isinstance(module, FullPrecisionLinear)
    if not module.has_biases:
        if sublayers_to_fuse:
            paths_to_check = [path / proj / "bias" for proj in sublayers_to_fuse]
        else:
            paths_to_check = path / "bias"
        for p in paths_to_check:
            if p in weights_dict:
                raise ValueError(f"Bias tensor found at {p} but module does not support it.")
        bias = None
    elif sublayers_to_fuse is None:
        bias = weights_dict[path / "bias"]
    else:
        bias = jnp.concatenate(
            [weights_dict[path / proj_name / "bias"] for proj_name in sublayers_to_fuse],
            axis=0,
        )

    weights = _fuse_full_precision_weights(weights_dict, path, sublayers_to_fuse)

    if scaling_to_fuse is not None:
        weights = weights * scaling_to_fuse[:, None]
        if bias is not None:
            bias = bias * scaling_to_fuse

    return load_parameters(lambda m: (m.weights, m.biases), module, (weights, bias))


def load_transformer_block(
    module: Transformer,
    weights_dict: Mapping[str, Array],
    fast: bool = False,
    path: ParameterPath | None = None,
) -> Transformer:
    def load_attention_local(
        attn_module: Attention,
        weights_dict: Mapping[str, Array],
        path: ParameterPath,
        scaling_to_fuze: Array | None = None,
    ) -> Attention:
        qkv_projection = load_linear_and_fuse_scaling(
            attn_module.qkv_projection,
            weights_dict,
            path / "wqkv",
            sublayers_to_fuse=None,
        )
        assert isinstance(qkv_projection, FullPrecisionLinear)

        # Permute QKV weights from interleaved RoPE format to rotate-half format
        permuted_qkv_weights = _permute_qkv_for_rope_rotate_half(
            qkv_projection.weights,
            num_heads=attn_module.num_heads,
            num_groups=attn_module.num_groups,
            head_dim=attn_module.head_dim,
        )
        qkv_projection = load_parameters(
            lambda m: (m.weights,),
            qkv_projection,
            (permuted_qkv_weights,),
        )
        assert isinstance(qkv_projection, FullPrecisionLinear)

        out_projection = load_linear_and_fuse_scaling(
            attn_module.out_projection,
            weights_dict,
            path / "wo",
            scaling_to_fuse=scaling_to_fuze,
        )

        if attn_module.query_norm is not None:
            query_norm = load_rmsnorm(attn_module.query_norm, weights_dict, path / "q_norm")
            permuted_scales = _permute_for_rope_rotate_half(
                query_norm.scales,
                1,
                query_norm.scales.shape[0],
            )
            query_norm = load_parameters(
                lambda m: (m.scales,),
                query_norm,
                (permuted_scales,),
            )
        else:
            query_norm = None

        if attn_module.key_norm is not None:
            key_norm = load_rmsnorm(attn_module.key_norm, weights_dict, path / "k_norm")
            permuted_scales = _permute_for_rope_rotate_half(
                key_norm.scales,
                1,
                key_norm.scales.shape[0],
            )
            key_norm = load_parameters(
                lambda m: (m.scales,),
                key_norm,
                (permuted_scales,),
            )
        else:
            key_norm = None

        return load_parameters(
            lambda m: (m.qkv_projection, m.out_projection, m.query_norm, m.key_norm),
            attn_module,
            (qkv_projection, out_projection, query_norm, key_norm),
        )

    def load_mlp(
        module: MLPBase,
        weights_dict: Mapping[str, Array],
        path: ParameterPath,
        up_proj_key: str,
        gate_proj_key: str,
        down_proj_key: str,
        scaling_to_fuze: Array | None = None,
    ) -> MLPBase:
        assert isinstance(module, DenseMLP)
        # Standard dense MLP with separate sublayers.
        up_projection = load_linear_and_fuse_scaling(
            module.up_projection,
            weights_dict,
            path,
            sublayers_to_fuse=[up_proj_key, gate_proj_key],
        )
        down_projection = load_linear_and_fuse_scaling(
            module.down_projection,
            weights_dict,
            path / down_proj_key,
            scaling_to_fuse=scaling_to_fuze,
        )
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

        layer_scale_path = path / "attention_layer_scale" / "gamma"
        layer_scale_weights = weights_dict.get(layer_scale_path, None)
        post_mixer_norm = None

        assert isinstance(module.mixer, Attention)
        attention = load_attention_local(
            attn_module=module.mixer,
            weights_dict=weights_dict,
            path=path / "attention",
            scaling_to_fuze=layer_scale_weights,
        )

        assert isinstance(module.pre_mlp_norm, Normalization)
        pre_mlp_norm = load_rmsnorm(
            module.pre_mlp_norm,
            weights_dict,
            path / "ffn_norm",
        )

        layer_scale_path = path / "ffn_layer_scale" / "gamma"
        layer_scale_weights = weights_dict.get(layer_scale_path, None)
        post_mlp_norm = None

        mlp = load_mlp(
            module=module.mlp,
            weights_dict=weights_dict,
            path=path / "feed_forward",
            up_proj_key="w3",
            gate_proj_key="w1",
            down_proj_key="w2",
            scaling_to_fuze=layer_scale_weights,
        )

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
    transformer: Transformer,
    output: FullPrecisionLinear,
    weights_dict: Mapping[str, Array],
    fast: bool = False,
) -> tuple[Transformer, FullPrecisionLinear]:
    transformer = load_transformer_block(transformer, weights_dict=weights_dict, fast=fast)

    base_path = ParameterPath()
    output_linear_name = "output" if not fast else "fast_output"
    output_linear = load_linear_and_fuse_scaling(output, weights_dict, base_path / output_linear_name)
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
    # Our FullPrecisionLinear expects (out, in), so we remove the kernel dimension
    out_proj_weight, out_proj_bias = fuse_weight_norm_conv1d_as_linear(weights_dict, path / "out_proj")
    # Remove kernel dimension: (out_channels, in_channels, 1) -> (out_channels, in_channels)
    out_proj_weight = rearrange(out_proj_weight, "out_ch in_ch 1 -> out_ch in_ch")
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
        - gamma
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
        lambda m: (m.scales, m.biases),
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

    # Load pointwise conv 2 (Linear layer), fusing layer scaling if present
    pwconv2_weight = weights_dict[path / "pwconv2" / "weight"]
    pwconv2_bias = weights_dict[path / "pwconv2" / "bias"]
    layer_scale_path = path / "gamma"
    if layer_scale_path in weights_dict:
        layer_scale = weights_dict[layer_scale_path]
        pwconv2_weight = pwconv2_weight * layer_scale[:, None]
        pwconv2_bias = pwconv2_bias * layer_scale
    pointwise_conv_step2 = load_parameters(
        lambda m: (m.weights, m.biases),
        module.pointwise_conv_step2,
        (pwconv2_weight, pwconv2_bias),
    )

    return load_parameters(
        lambda m: (m.depthwise_conv, m.norm, m.pointwise_conv_step1, m.pointwise_conv_step2),
        module,
        (depthwise_conv, norm, pointwise_conv_step1, pointwise_conv_step2),
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
    trans_conv = load_causal_transpose_conv1d(module.trans_conv, weights_dict, path / "0" / "conv")

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
    module: DownsampleResidualVectorQuantize,
    weights_dict: Mapping[str, Array],
    path: ParameterPath | None = None,
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
    conv1 = load_causal_conv1d(module.conv1, weights_dict, path / "block" / "1" / "conv")
    snake2 = load_snake1d(module.snake2, weights_dict, path / "block" / "2")
    conv2 = load_causal_conv1d(module.conv2, weights_dict, path / "block" / "3" / "conv")

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
    trans_conv = load_causal_transpose_conv1d(module.trans_conv, weights_dict, path / "block" / "1" / "conv")
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
    first_conv = load_causal_conv1d(module.first_conv, weights_dict, path / "model" / "0" / "conv")

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
    final_conv = load_causal_conv1d(module.final_conv, weights_dict, path / "model" / final_conv_idx / "conv")

    return load_parameters(
        lambda m: (m.first_conv, m.decoder_blocks, m.final_snake, m.final_conv),
        module,
        (first_conv, decoder_blocks, final_snake, final_conv),
    )


def load_descript_audio_codec(dac_module: DescriptAudioCodec, state_dict: Mapping[str, Any]) -> DescriptAudioCodec:
    loaded_quantizer = load_downsample_rvq(dac_module.quantizer, state_dict, path=ParameterPath("quantizer"))
    loaded_decoder = load_audio_decoder(dac_module.decoder, state_dict, path=ParameterPath("decoder"))

    return DescriptAudioCodec(
        config=dac_module.config,
        quantizer=loaded_quantizer,
        decoder=loaded_decoder,
    )


def load_fishaudio_text_decoder(
    module: FishAudioTextDecoder,
    weights_dict: Mapping[str, Array],
    decoder_path: ParameterPath | None = None,
) -> FishAudioTextDecoder:
    basepath = ParameterPath() if decoder_path is None else decoder_path
    transformer_slow, readout_slow = load_fish_audio_text_decoding_modules(
        module.transformer_slow,
        module.readout_slow,
        weights_dict,
        fast=False,
    )
    transformer_fast, readout_fast = load_fish_audio_text_decoding_modules(
        module.transformer_fast,
        module.readout_fast,
        weights_dict,
        fast=True,
    )
    embeddings_slow = load_tied_embedding(
        module.embeddings_slow,
        weights_dict,
        basepath / "embeddings",
    )
    embeddings_fast = load_tied_embedding(
        module.embeddings_fast,
        weights_dict,
        basepath / "fast_embeddings",
    )

    codebook_embeddings = load_tied_embedding(
        module.codebook_embeddings,
        weights_dict,
        basepath / "codebook_embeddings",
    )

    if isinstance(module.fast_model_projection, FullPrecisionLinear):
        fast_model_projection = load_linear_and_fuse_scaling(
            module.fast_model_projection,
            weights_dict,
            basepath / "fast_project_in",
        )
        assert isinstance(fast_model_projection, FullPrecisionLinear)
    else:
        fast_model_projection = Identity()

    return load_parameters(
        lambda m: (
            m.embeddings_slow,
            m.transformer_slow,
            m.readout_slow,
            m.embeddings_fast,
            m.transformer_fast,
            m.readout_fast,
            m.codebook_embeddings,
            m.fast_model_projection,
        ),
        module,
        (
            embeddings_slow,
            transformer_slow,
            readout_slow,
            embeddings_fast,
            transformer_fast,
            readout_fast,
            codebook_embeddings,
            fast_model_projection,
        ),
    )


def load_fishaudio_audio_decoder(
    module: DescriptAudioCodec,
    weights_dict: Mapping[str, Array],
    base_path: ParameterPath,
) -> DescriptAudioCodec:
    loaded_quantizer = load_downsample_rvq(module.quantizer, weights_dict, base_path / "quantizer")
    loaded_decoder = load_audio_decoder(module.decoder, weights_dict, base_path / "decoder")

    return load_parameters(lambda m: (m.quantizer, m.decoder), module, (loaded_quantizer, loaded_decoder))


def load_tokenizer_from_fishaudio_tiktoken(
    path_to_tokenizer: Path,
    path_to_special_tokens: Path,
) -> tuple[Tokenizer, FishAudioSpecialInferenceTokens]:
    from tiktoken.core import Encoding as TiktokenEncoding
    from transformers.integrations.tiktoken import convert_tiktoken_to_fast

    def _load_fishaudio_tiktoken_data(
        tiktoken_path: Path,
        special_tokens: dict[str, int],
    ) -> tuple[TiktokenEncoding, FishAudioSpecialInferenceTokens]:
        def load_tiktoken_bpe(tiktoken_bpe_file: Path) -> dict[bytes, int]:
            data = {}
            with open(tiktoken_bpe_file) as token_file:
                for line in token_file.read().splitlines():
                    if not line:
                        continue
                    token, rank = line.split()
                    if token == "=":
                        continue
                    data[base64.b64decode(token)] = int(rank)
            return data

        mergeable_ranks = load_tiktoken_bpe(tiktoken_path)
        special_token_begin = len(mergeable_ranks)
        all_special_tokens_with_ids = {token: special_token_begin + i for i, token in enumerate(special_tokens)}

        semantic_id_to_token_id = {}
        end_idx = 0
        for token in special_tokens:
            if token.startswith("<|semantic:"):
                match_results = re.match(r"<\|semantic:(\d+)\|>", token)
                assert match_results is not None
                idx = int(match_results.group(1))
                semantic_id_to_token_id[idx] = all_special_tokens_with_ids[token]
                end_idx = max(end_idx, idx)

        semantic_begin_id = semantic_id_to_token_id[0]
        semantic_end_id = semantic_id_to_token_id[end_idx]

        tkt_model = TiktokenEncoding(
            name=Path(tiktoken_path).stem,
            pat_str=FISH_TIKTOKEN_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=all_special_tokens_with_ids,
        )

        inference_special_tokens = FishAudioSpecialInferenceTokens(
            semantic_begin_id=semantic_begin_id,
            semantic_end_id=semantic_end_id,
            im_end_token_id=all_special_tokens_with_ids[IM_END_TOKEN],
        )

        return tkt_model, inference_special_tokens

    output_temp_dir = tempfile.mkdtemp()
    try:
        if path_to_special_tokens.exists():
            with open(path_to_special_tokens) as f:
                all_special_tokens_with_ids = json.load(f)
        else:
            all_special_tokens_with_ids = {}

        tkt_model, special_inference_tokens = _load_fishaudio_tiktoken_data(
            path_to_tokenizer,
            all_special_tokens_with_ids,
        )

        convert_tiktoken_to_fast(tkt_model, output_temp_dir)
        tokenizer = Tokenizer.from_file(output_temp_dir + "/tokenizer.json")
        return tokenizer, special_inference_tokens
    finally:
        shutil.rmtree(output_temp_dir)
