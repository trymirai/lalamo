from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike

from lalamo.common import ParameterPath
from lalamo.modules import (
    Attention,
    AttentionConfig,
    Decoder,
    DenseMLP,
    FullPrecisionLinear,
    GroupQuantizedLinear,
    LinearBase,
    Mamba2,
    Mamba2Config,
    MLXQuantizedLinear,
    MLXQuantizedTiedEmbedding,
    MLXQuantizedTiedEmbeddingConfig,
    MLXSemiQuantizedUntiedEmbedding,
    Normalization,
    SeparableCausalConv,
    ShortConv,
    ShortConvConfig,
    TiedEmbedding,
    TransformerLayer,
    UntiedEmbedding,
)
from lalamo.modules.classifier import Classifier
from lalamo.modules.embedding import MLXQuantizedUntiedEmbedding
from lalamo.modules.mlp import MixtureOfExperts, MLPBase
from lalamo.quantization import QuantizationMode

from .common import load_parameters
from .utils import decode_mxfp4, deinterleave_pairwise_columns

__all__ = ["load_huggingface_decoder"]


AWQ_UINT4_REVERSE_ORDER = jnp.array([0, 4, 1, 5, 2, 6, 3, 7], dtype=jnp.int32)


def _reverse_uint4_order(array: Array, reverse_order: Array) -> Array:
    """Reverses the AWQ packing order to get the logical order of channels for INT4."""
    pack_factor = 32 // 4
    *_, last_dim = array.shape
    if last_dim % pack_factor != 0:
        return array

    array_reshaped = rearrange(
        array,
        "... (group pack_factor) -> ... group pack_factor",
        pack_factor=pack_factor,
    )
    array_reordered = array_reshaped[..., reverse_order]
    return rearrange(array_reordered, "... group pack_factor -> ... (group pack_factor)")


def unpack_int32(packed_weights: Array, mode: QuantizationMode) -> Array:
    assert packed_weights.dtype in (
        jnp.int32,
        jnp.uint32,
    ), f"Expected packed_weights to be of dtype jnp.(u)int32, got {packed_weights.dtype}"
    assert 32 % mode.bits == 0

    shifts = jnp.arange(0, 32, mode.bits)
    mask = (2**mode.bits) - 1
    unpacked = jnp.bitwise_and(jnp.right_shift(packed_weights[:, :, None], shifts[None, None, :]), mask)
    unpacked = rearrange(
        unpacked,
        "out_channels packed_groups packed_values -> out_channels (packed_groups packed_values)",
    )

    return unpacked


def _process_quantized_tensor(
    quantized: Array,
    weight_quantization: QuantizationMode,
    activation_precision: DTypeLike,
    reverse_order: Array | None = None,
) -> Array:
    unpacked = unpack_int32(quantized, weight_quantization)
    if reverse_order is not None:
        assert weight_quantization == QuantizationMode.UINT4, "reverse order only supported on uint4 quant type"
        unpacked = _reverse_uint4_order(unpacked, reverse_order)

    return unpacked.astype(activation_precision)


def _fuse_full_precision_weights(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    sublayers_to_fuse: list[str] | None,
) -> Array:
    if sublayers_to_fuse is None:
        return weights_dict[path / "weight"]

    weights = [weights_dict[path / layer_name / "weight"] for layer_name in sublayers_to_fuse]
    return jnp.concatenate(weights, axis=0)


@dataclass(frozen=True)
class QuantizedParamLayout:
    weight: str
    scale: str
    bias: str
    transposed: bool


AWQ_QUANTIZED_WEIGHT_LAYOUT = QuantizedParamLayout("qweight", "scales", "qzeros", transposed=True)
MLX_QUANTIZED_WEIGHT_LAYOUT = QuantizedParamLayout("weight", "scales", "biases", transposed=False)


def _fuse_quantized_weights(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    sublayers_to_fuse: list[str] | None,
    quantized_param_layout: QuantizedParamLayout,
) -> tuple[Array, Array, Array]:
    # Note that AWQ quantized weights are stored transposed relative to full-precision weights

    if sublayers_to_fuse is None:
        qweights = weights_dict[path / quantized_param_layout.weight]
        qzeros = weights_dict[path / quantized_param_layout.bias]
        scales = weights_dict[path / quantized_param_layout.scale]
        return qweights, qzeros, scales

    qweights = [weights_dict[path / layer_name / quantized_param_layout.weight] for layer_name in sublayers_to_fuse]
    qzeros = [weights_dict[path / layer_name / quantized_param_layout.bias] for layer_name in sublayers_to_fuse]
    scales = [weights_dict[path / layer_name / quantized_param_layout.scale] for layer_name in sublayers_to_fuse]

    fused_qweights = jnp.concatenate(qweights, axis=int(quantized_param_layout.transposed))
    fused_qzeros = jnp.concatenate(qzeros, axis=int(quantized_param_layout.transposed))
    fused_scales = jnp.concatenate(scales, axis=int(quantized_param_layout.transposed))

    return fused_qweights, fused_qzeros, fused_scales


def load_linear(
    module: LinearBase,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    sublayers_to_fuse: list[str] | None = None,
) -> LinearBase:
    """Loads a linear layer, optionally fusing weights from sublayers."""
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

    if isinstance(module, FullPrecisionLinear):
        weights = _fuse_full_precision_weights(weights_dict, path, sublayers_to_fuse)
        return load_parameters(lambda m: (m.weights, m.biases), module, (weights, bias))

    if isinstance(module, GroupQuantizedLinear):
        qweights, qzeros, scales = _fuse_quantized_weights(
            weights_dict,
            path,
            sublayers_to_fuse,
            AWQ_QUANTIZED_WEIGHT_LAYOUT,
        )
        weight_quantization = module.config.weight_quantization_mode
        activation_precision = module.activation_precision

        if weight_quantization == QuantizationMode.UINT4:
            reverse_order = AWQ_UINT4_REVERSE_ORDER
        else:
            reverse_order = None

        weights = _process_quantized_tensor(
            qweights,
            weight_quantization,
            activation_precision,
            reverse_order,
        )
        zeros = _process_quantized_tensor(
            qzeros,
            weight_quantization,
            activation_precision,
            reverse_order,
        )
        scales = scales.astype(activation_precision)

        return load_parameters(
            lambda m: (m.weights, m.scales, m.zero_points, m.biases),
            module,
            (weights.T, scales.T, zeros.T, bias),
        )

    if isinstance(module, MLXQuantizedLinear):
        qweights, deq_biases, scales = _fuse_quantized_weights(
            weights_dict,
            path,
            sublayers_to_fuse,
            MLX_QUANTIZED_WEIGHT_LAYOUT,
        )
        weight_quantization = module.config.weight_quantization_mode
        activation_precision = module.activation_precision

        weights = _process_quantized_tensor(
            qweights,
            weight_quantization,
            activation_precision,
            None,
        )
        scales = scales.astype(activation_precision)
        deq_biases = deq_biases.astype(activation_precision)

        return load_parameters(
            lambda m: (m.weights, m.scales, m.deq_biases, m.biases),
            module,
            (weights, scales, deq_biases, bias),
        )

    raise TypeError(f"Unsupported module type for loading: {type(module)}")


def load_mlp(
    module: MLPBase,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    up_proj_key: str,
    gate_proj_key: str,
    down_proj_key: str,
) -> MLPBase:
    if isinstance(module, DenseMLP):
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

    if isinstance(module, MixtureOfExperts):
        return load_moe(module, weights_dict, path)

    raise TypeError(f"Unsupported module type for loading: {type(module)}")


def load_moe(module: MixtureOfExperts, weights_dict: Mapping[str, Array], path: ParameterPath) -> MixtureOfExperts:
    # Load router via the standard linear loader
    router = load_linear(module.router, weights_dict, path / "router")

    experts_path = path / "experts"
    # Handle fused MXFP4 experts layout if present
    if (experts_path / "gate_up_proj_blocks") in weights_dict:
        # Decode fused gate/up (interleaved), split into (up, gate), and add +1.0 to up bias
        fused = decode_mxfp4(
            weights_dict[experts_path / "gate_up_proj_blocks"],
            weights_dict[experts_path / "gate_up_proj_scales"],
            dtype=module.activation_precision,
            flatten=False,
        )
        # Stored as (experts, outputs=2*hidden_dim, input_blocks, input_block_elems)
        # Merge blocks and move outputs last
        fused_eio = rearrange(fused, "e o ib ie -> e (ib ie) o")
        up_w, gate_w = deinterleave_pairwise_columns(fused_eio, first="odd")
        combined_up_gate = jnp.concatenate([up_w, gate_w], axis=-1)
        # Transpose to new layout: (experts, outputs, inputs)
        combined_up_gate_w = jnp.swapaxes(combined_up_gate, -1, -2)

        gub = weights_dict[experts_path / "gate_up_proj_bias"]
        if gub.ndim == 1:
            # Broadcast to (experts, 2*hidden_dim)
            gub = jnp.broadcast_to(gub, (combined_up_gate_w.shape[0], gub.shape[0]))
        up_b, gate_b = deinterleave_pairwise_columns(gub, first="odd")
        combined_up_gate_b = jnp.concatenate([up_b + 1.0, gate_b], axis=-1)

        up_projection = load_parameters(
            lambda m: (m.weights, m.biases),
            module.experts.up_projection,
            (combined_up_gate_w, combined_up_gate_b),
        )

        # Down projection: decode MXFP4 to dense
        down_w = decode_mxfp4(
            weights_dict[experts_path / "down_proj_blocks"],
            weights_dict[experts_path / "down_proj_scales"],
            dtype=module.activation_precision,
            flatten=False,
        )
        # Stored as (experts, outputs=model_dim, input_blocks, input_block_elems)
        # Merge blocks and move outputs last
        down_w = rearrange(down_w, "e o ib ie -> e o (ib ie)")
        down_b = weights_dict[experts_path / "down_proj_bias"]
        if down_b.ndim == 1:
            down_b = jnp.broadcast_to(down_b, (*down_w.shape[:-1], down_b.shape[0]))

        down_projection = load_parameters(
            lambda m: (m.weights, m.biases),
            module.experts.down_projection,
            (down_w, down_b),
        )

        experts = load_parameters(
            lambda m: (m.up_projection, m.down_projection),
            module.experts,
            (up_projection, down_projection),
        )
    else:
        # Fallback: recursively load a standard DenseMLP experts module
        experts = load_mlp(
            module.experts,
            weights_dict,
            experts_path,
            "up_proj",
            "gate_proj",
            "down_proj",
        )

    return load_parameters(
        lambda m: (m.router, m.experts),
        module,
        (router, experts),
    )


def load_rmsnorm(
    module: Normalization,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> Normalization:
    scales = weights_dict[path / "weight"]
    return load_parameters(lambda m: (m.scales,), module, (scales,))


def load_attention(
    module: Attention,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> Attention:
    if (path / "o_proj.weight") in weights_dict or (path / "o_proj.qweight") in weights_dict:
        o_proj_name = "o_proj"
    elif (path / "out_proj.weight") in weights_dict or (path / "out_proj.qweight") in weights_dict:
        o_proj_name = "out_proj"
    else:
        raise NotImplementedError("Can't determine attention output projection name")

    qkv_projection = load_linear(
        module.qkv_projection,
        weights_dict,
        path,
        sublayers_to_fuse=["q_proj", "k_proj", "v_proj"],
    )
    out_projection = load_linear(module.out_projection, weights_dict, path / o_proj_name)

    if module.query_norm is not None:
        if (path / "q_norm.weight") in weights_dict:
            q_norm_name = "q_norm"
        elif (path / "q_layernorm.weight") in weights_dict:
            q_norm_name = "q_layernorm"
        else:
            raise NotImplementedError("Can't determine attention query projection parameter name")

        query_norm = load_rmsnorm(module.query_norm, weights_dict, path / q_norm_name)
    else:
        query_norm = None

    if module.key_norm is not None:
        if (path / "k_norm.weight") in weights_dict:
            k_norm_name = "k_norm"
        elif (path / "k_layernorm.weight") in weights_dict:
            k_norm_name = "k_layernorm"
        else:
            raise NotImplementedError("Can't determine attention key projection parameter name")

        key_norm = load_rmsnorm(module.key_norm, weights_dict, path / k_norm_name)
    else:
        key_norm = None

    # GPT-OSS adds per-head attention sinks; load them if present.
    if (path / "sinks") in weights_dict:
        sinks = weights_dict[path / "sinks"]
    else:
        sinks = module.sinks

    return load_parameters(
        lambda m: (
            m.qkv_projection,
            m.out_projection,
            m.query_norm,
            m.key_norm,
            m.sinks,
        ),
        module,
        (qkv_projection, out_projection, query_norm, key_norm, sinks),
    )


def _load_conv(
    conv_module: SeparableCausalConv,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    permute_conv: bool,
) -> SeparableCausalConv:
    weight_path = path / "conv1d" / "weight"
    if weight_path not in weights_dict:
        weight_path = path / "conv_weight"
    if weight_path not in weights_dict:
        weight_path = path / "conv.weight"
    if weight_path not in weights_dict:
        weight_path = None

    if weight_path is not None:
        raw = weights_dict[weight_path]
        if permute_conv:
            raw = jnp.matrix_transpose(raw)
        conv_weight = raw.squeeze(1) if raw.ndim == 3 else raw
    else:
        conv_weight = conv_module.weights

    bias_path = path / "conv1d" / "bias"
    if bias_path not in weights_dict:
        bias_path = path / "conv_bias"
    if bias_path not in weights_dict:
        bias_path = path / "conv.bias"
    if bias_path not in weights_dict:
        bias_path = None

    if bias_path is not None and conv_module.biases is not None:
        conv_bias = weights_dict[bias_path]
    else:
        conv_bias = conv_module.biases

    return load_parameters(
        lambda m: (m.weights, m.biases),
        conv_module,
        (conv_weight, conv_bias),
    )


def load_mamba2(
    module: Mamba2,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    permute_conv: bool,
) -> Mamba2:
    in_projection = load_linear(module.in_projection, weights_dict, path / "in_proj")
    out_projection = load_linear(module.out_projection, weights_dict, path / "out_proj")
    conv = _load_conv(module.conv, weights_dict, path, permute_conv)

    skip_connection_weight_path = path / "D"
    if skip_connection_weight_path in weights_dict:
        skip_connection_weight = weights_dict[skip_connection_weight_path]
    else:
        skip_connection_weight = module.skip_connection_weight

    gate_bias_path = path / "z_bias"
    if gate_bias_path in weights_dict:
        gate_bias = weights_dict[gate_bias_path]
    else:
        gate_bias = module.gate_bias

    return load_parameters(
        lambda m: (
            m.in_projection,
            m.out_projection,
            m.conv,
            m.skip_connection_weight,
            m.gate_bias,
        ),
        module,
        (in_projection, out_projection, conv, skip_connection_weight, gate_bias),
    )


def load_short_conv(
    module: ShortConv,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    permute_conv: bool,
) -> ShortConv:
    in_projection = load_linear(module.in_projection, weights_dict, path / "in_proj")
    out_projection = load_linear(module.out_projection, weights_dict, path / "out_proj")
    conv = _load_conv(module.conv, weights_dict, path, permute_conv)

    return load_parameters(
        lambda m: (m.in_projection, m.out_projection, m.conv),
        module,
        (in_projection, out_projection, conv),
    )


def load_transformer_layer(
    module: TransformerLayer,
    weights_dict: Mapping[str, Array],
    mixer_path: ParameterPath,
    mlp_path: ParameterPath,
    mixer_key: str,
    mlp_key: str,
    pre_mixer_norm_key: str,
    pre_mlp_norm_key: str,
    up_proj_key: str,
    gate_proj_key: str,
    down_proj_key: str,
    permute_conv: bool,
) -> TransformerLayer:
    if module.pre_mixer_norm is not None:
        pre_attention_norm = load_rmsnorm(
            module.pre_mixer_norm,
            weights_dict,
            mixer_path / pre_mixer_norm_key,
        )

    else:
        pre_attention_norm = None
    # Load mixer (attention or mamba)
    if isinstance(module.mixer, Attention):
        mixer = load_attention(module.mixer, weights_dict, mixer_path / mixer_key)
    elif isinstance(module.mixer, Mamba2):
        mixer = load_mamba2(module.mixer, weights_dict, mixer_path / mixer_key, permute_conv)
    elif isinstance(module.mixer, ShortConv):
        mixer = load_short_conv(module.mixer, weights_dict, mixer_path / mixer_key, permute_conv)
    else:
        mixer = module.mixer

    if module.post_mixer_norm is not None:
        post_attention_norm = load_rmsnorm(
            module.post_mixer_norm,
            weights_dict,
            mixer_path / "post_attention_layernorm",
        )

        pre_mlp_norm = load_rmsnorm(
            module.pre_mlp_norm,
            weights_dict,
            mlp_path / "pre_feedforward_layernorm",
        )
    else:
        post_attention_norm = None

        pre_mlp_norm = load_rmsnorm(
            module.pre_mlp_norm,
            weights_dict,
            mlp_path / pre_mlp_norm_key,
        )

    mlp = load_mlp(
        module.mlp,
        weights_dict,
        mlp_path / mlp_key,
        up_proj_key,
        gate_proj_key,
        down_proj_key,
    )

    if module.post_mlp_norm is not None:
        post_mlp_norm = load_rmsnorm(
            module.post_mlp_norm,
            weights_dict,
            mlp_path / "post_feedforward_layernorm",
        )
    else:
        post_mlp_norm = None

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
            pre_attention_norm,
            mixer,
            post_attention_norm,
            pre_mlp_norm,
            mlp,
            post_mlp_norm,
        ),
    )


def load_tied_embedding(
    module: TiedEmbedding,
    weights_dict: Mapping[str, Array],
    embedding_path: ParameterPath,
) -> TiedEmbedding:
    weights = weights_dict[embedding_path / "weight"]
    return load_parameters(lambda m: (m.weights,), module, (weights,))


def load_mlx_quantized_tied_embedding(
    module: MLXQuantizedTiedEmbedding,
    weights_dict: Mapping[str, Array],
    embedding_path: ParameterPath,
) -> MLXQuantizedTiedEmbedding:
    qweights = weights_dict[embedding_path / "weight"]
    qscales = weights_dict[embedding_path / "scales"]
    qbiases = weights_dict[embedding_path / "biases"]

    weights = _process_quantized_tensor(
        qweights,
        module.config.embedding_quantization_mode,
        module.activation_precision,
        None,
    )
    scales = qscales.astype(module.activation_precision)
    biases = qbiases.astype(module.activation_precision)

    return load_parameters(lambda m: (m.weights, m.scales, m.biases), module, (weights, scales, biases))


def load_mlx_quantized_untied_embedding(
    module: MLXQuantizedUntiedEmbedding,
    weights_dict: Mapping[str, Array],
    embedding_path: ParameterPath,
    lm_head_path: ParameterPath,
) -> MLXQuantizedUntiedEmbedding:
    input_qweights = weights_dict[embedding_path / "weight"]
    input_qscales = weights_dict[embedding_path / "scales"]
    input_qbiases = weights_dict[embedding_path / "biases"]
    output_qweights = weights_dict[lm_head_path / "weight"]
    output_qscales = weights_dict[lm_head_path / "scales"]
    output_qbiases = weights_dict[lm_head_path / "biases"]

    input_weights = _process_quantized_tensor(
        input_qweights,
        module.config.embedding_quantization_mode,
        module.activation_precision,
        None,
    )
    input_scales = input_qscales.astype(module.activation_precision)
    input_biases = input_qbiases.astype(module.activation_precision)

    output_weights = _process_quantized_tensor(
        output_qweights,
        module.config.embedding_quantization_mode,
        module.activation_precision,
        None,
    )
    output_scales = output_qscales.astype(module.activation_precision)
    output_biases = output_qbiases.astype(module.activation_precision)

    return load_parameters(
        lambda m: (
            m.input_weights,
            m.input_scales,
            m.input_biases,
            m.output_weights,
            m.output_scales,
            m.output_biases,
        ),
        module,
        (input_weights, input_scales, input_biases, output_weights, output_scales, output_biases),
    )


def load_mlx_semi_quantized_untied_embedding(
    module: MLXSemiQuantizedUntiedEmbedding,
    weights_dict: Mapping[str, Array],
    embedding_path: ParameterPath,
    lm_head_path: ParameterPath,
) -> MLXSemiQuantizedUntiedEmbedding:
    input_weights = weights_dict[embedding_path / "weight"]

    output_qweights = weights_dict[lm_head_path / "weight"]
    output_qscales = weights_dict[lm_head_path / "scales"]
    output_qbiases = weights_dict[lm_head_path / "biases"]

    output_weights = _process_quantized_tensor(
        output_qweights,
        module.config.embedding_quantization_mode,
        module.activation_precision,
        None,
    )
    output_scales = output_qscales.astype(module.activation_precision)
    output_biases = output_qbiases.astype(module.activation_precision)

    return load_parameters(
        lambda m: (m.input_weights, m.output_weights, m.output_scales, m.output_biases),
        module,
        (input_weights, output_weights, output_scales, output_biases),
    )


def load_untied_embedding(
    module: UntiedEmbedding,
    weights_dict: Mapping[str, Array],
    embedding_path: ParameterPath,
    lm_head_path: ParameterPath,
) -> UntiedEmbedding:
    input_weights = weights_dict[embedding_path / "weight"]
    output_weights = weights_dict[lm_head_path / "weight"]
    return load_parameters(
        lambda m: (m.input_weights, m.output_weights),
        module,
        (input_weights, output_weights),
    )


def load_huggingface_decoder(
    module: Decoder,
    weights_dict: Mapping[str, Array],
) -> Decoder:
    if any(key.startswith("language_model.") for key in weights_dict):
        base_path = ParameterPath("language_model")
    else:
        base_path = ParameterPath()

    is_llamba_full_precision = any(key.startswith("backbone.") for key in weights_dict)
    is_llamba_mlx = any(key.startswith("embedding.encoder.") for key in weights_dict)
    is_lfm2 = any(key.startswith("model.layers.0.operator_norm.weight") for key in weights_dict)
    if is_llamba_full_precision:
        decoder_path = base_path / "backbone"
        embedding_path = decoder_path / "embedding"
        pre_mixer_norm_key = "input_layernorm"
        mixer_key = {Mamba2Config: "mixer"}
        permute_conv = False
        pre_mlp_norm_key = "post_attention_layernorm"
        mlp_key = "mlp"
        up_proj_key = "up_proj"
        gate_proj_key = "gate_proj"
        down_proj_key = "down_proj"
        alternating_layers = False
        norm_key = "final_layernorm"
        lm_head_path = base_path / "lm_head"
    elif is_llamba_mlx:
        decoder_path = base_path / "model"
        embedding_path = base_path / "embedding.encoder"
        pre_mixer_norm_key = "norm"
        mixer_key = {Mamba2Config: "layer"}
        permute_conv = False
        pre_mlp_norm_key = "norm"
        mlp_key = "layer"
        up_proj_key = "gate_proj"
        gate_proj_key = "in_proj"
        down_proj_key = "out_proj"
        alternating_layers = True
        norm_key = "norm"
        lm_head_path = base_path / "head.linear"
    elif is_lfm2:
        decoder_path = base_path / "model"
        embedding_path = decoder_path / "embed_tokens"
        pre_mixer_norm_key = "operator_norm"
        mixer_key = {ShortConvConfig: "conv", AttentionConfig: "self_attn"}
        permute_conv = isinstance(module.config.embedding_config, MLXQuantizedTiedEmbeddingConfig)
        pre_mlp_norm_key = "ffn_norm"
        mlp_key = "feed_forward"
        up_proj_key = "w3"
        gate_proj_key = "w1"
        down_proj_key = "w2"
        alternating_layers = False
        norm_key = "embedding_norm"
        lm_head_path = base_path / "lm_head"
    else:
        decoder_path = base_path / "model"
        embedding_path = decoder_path / "embed_tokens"
        pre_mixer_norm_key = "input_layernorm"
        mixer_key = {AttentionConfig: "self_attn"}
        permute_conv = False
        pre_mlp_norm_key = "post_attention_layernorm"
        mlp_key = "mlp"
        up_proj_key = "up_proj"
        gate_proj_key = "gate_proj"
        down_proj_key = "down_proj"
        alternating_layers = False
        norm_key = "norm"
        lm_head_path = base_path / "lm_head"

    if isinstance(module.embedding, TiedEmbedding):
        embedding = load_tied_embedding(module.embedding, weights_dict, embedding_path)
    elif isinstance(module.embedding, MLXQuantizedTiedEmbedding):
        embedding = load_mlx_quantized_tied_embedding(module.embedding, weights_dict, embedding_path)
    elif isinstance(module.embedding, MLXQuantizedUntiedEmbedding):
        embedding = load_mlx_quantized_untied_embedding(module.embedding, weights_dict, embedding_path, lm_head_path)
    elif isinstance(module.embedding, MLXSemiQuantizedUntiedEmbedding):
        embedding = load_mlx_semi_quantized_untied_embedding(
            module.embedding,
            weights_dict,
            embedding_path,
            lm_head_path,
        )
    elif isinstance(module.embedding, UntiedEmbedding):
        embedding = load_untied_embedding(module.embedding, weights_dict, embedding_path, lm_head_path)
    else:
        raise TypeError(f"Unsupported embedding type: {type(module.embedding)}")

    decoder_layers = tuple(
        load_transformer_layer(
            layer,
            weights_dict,
            decoder_path / "layers" / ((i * 2) if alternating_layers else i),
            decoder_path / "layers" / ((i * 2 + 1) if alternating_layers else i),
            mixer_key[type(layer.config.mixer_config)],
            mlp_key,
            pre_mixer_norm_key,
            pre_mlp_norm_key,
            up_proj_key,
            gate_proj_key,
            down_proj_key,
            permute_conv,
        )
        for i, layer in enumerate(module.transformer.layers)
    )
    output_norm = load_rmsnorm(module.transformer.output_norm, weights_dict, decoder_path / norm_key)
    return load_parameters(
        lambda m: (m.embedding, m.transformer.layers, m.transformer.output_norm),
        module,
        (embedding, decoder_layers, output_norm),
    )


def load_huggingface_classifier(
    module: Classifier,
    weights_dict: Mapping[str, Array],
) -> Classifier:
    def load_tied_embedding_local(
        module: TiedEmbedding,
        weights_dict: Mapping[str, Array],
        decoder_path: ParameterPath,
    ) -> TiedEmbedding:
        input_weights = weights_dict[decoder_path / "embeddings" / "tok_embeddings" / "weight"]
        return load_parameters(lambda m: (m.weights,), module, (input_weights,))

    def load_linear_with_reshufling(
        module: LinearBase,
        weights_dict: Mapping[str, Array],
        path: ParameterPath,
    ) -> LinearBase:
        """Loads a linear layer and reshufle some weights in resulting matrix to meet
        requirements of downstream 'split' in MLP layer in attention."""

        assert not module.has_biases, "Expecting no biases in FullPrecisionLinear"
        assert isinstance(module, FullPrecisionLinear), "Expecting FullPrecisionLinear module as input"

        weights = weights_dict[path / "weight"]
        rows, _ = weights.shape
        shuffled_weights = jnp.vstack((weights[rows // 2 :, :], weights[: rows // 2, :]))
        return load_parameters(lambda m: (m.weights, m.biases), module, (shuffled_weights, None))

    def load_attention_local(
        module: Attention,
        weights_dict: Mapping[str, Array],
        path: ParameterPath,
    ) -> Attention:
        qkv_projection = load_linear(
            module.qkv_projection,
            weights_dict,
            path / "Wqkv",
            sublayers_to_fuse=None,
        )
        out_projection = load_linear(module.out_projection, weights_dict, path / "Wo")

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

    def load_mlp_local(module: MLPBase, weights_dict: Mapping[str, Array], path: ParameterPath) -> MLPBase:
        assert isinstance(module, DenseMLP)
        up_projection = load_linear_with_reshufling(
            module.up_projection,
            weights_dict,
            path / "Wi",
        )
        down_projection = load_linear(module.down_projection, weights_dict, path / "Wo")
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
            pre_attention_norm = load_rmsnorm(
                module.pre_mixer_norm,
                weights_dict,
                path / "attn_norm",
            )
        else:
            pre_attention_norm = None

        assert isinstance(module.mixer, Attention)
        attention = load_attention_local(module.mixer, weights_dict, path / "attn")
        if module.post_mixer_norm is not None:
            post_attention_norm = load_rmsnorm(
                module.post_mixer_norm,
                weights_dict,
                path / "post_attention_layernorm",
            )

            pre_mlp_norm = load_rmsnorm(
                module.pre_mlp_norm,
                weights_dict,
                path / "pre_feedforward_layernorm",
            )
        else:
            post_attention_norm = None

            pre_mlp_norm = load_rmsnorm(
                module.pre_mlp_norm,
                weights_dict,
                path / "mlp_norm",
            )

        mlp = load_mlp_local(module.mlp, weights_dict, path / "mlp")
        if module.post_mlp_norm is not None:
            post_mlp_norm = load_rmsnorm(
                module.post_mlp_norm,
                weights_dict,
                path / "post_feedforward_layernorm",
            )
        else:
            post_mlp_norm = None
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
                pre_attention_norm,
                attention,
                post_attention_norm,
                pre_mlp_norm,
                mlp,
                post_mlp_norm,
            ),
        )

    base_path = ParameterPath()
    decoder_path = base_path / "model"
    head_path = base_path / "head"
    classifier_path = base_path / "classifier"
    assert isinstance(module.embedding, TiedEmbedding)
    embedding = load_tied_embedding_local(module.embedding, weights_dict, decoder_path)
    embedding_norm = load_rmsnorm(module.embedding_norm, weights_dict, base_path / "model" / "embeddings" / "norm")

    decoder_layers = tuple(
        load_transformer_layer_local(layer, weights_dict, decoder_path / "layers" / i)
        for i, layer in enumerate(module.transformer.layers)
    )
    output_norm = load_rmsnorm(module.transformer.output_norm, weights_dict, decoder_path / "final_norm")
    head_dense = load_linear(module.prediction_head.dense, weights_dict, head_path / "dense")
    head_norm = load_rmsnorm(module.prediction_head.norm, weights_dict, head_path / "norm")
    head_readout = load_linear(module.prediction_head.readout, weights_dict, classifier_path)
    return load_parameters(
        lambda m: (
            m.embedding,
            m.embedding_norm,
            m.transformer.layers,
            m.transformer.output_norm,
            m.prediction_head.dense,
            m.prediction_head.norm,
            m.prediction_head.readout,
        ),
        module,
        (
            embedding,
            embedding_norm,
            decoder_layers,
            output_norm,
            head_dense,
            head_norm,
            head_readout,
        ),
    )
