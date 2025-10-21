from collections.abc import Mapping

import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array

from lalamo.common import ParameterPath
from lalamo.modules import (
    Attention,
    Decoder,
    DecoderLayer,
    DenseMLP,
    FullPrecisionLinear,
    GroupQuantizedLinear,
    LinearBase,
    RMSNorm,
    TiedEmbedding,
    UntiedEmbedding,
)
from lalamo.modules.classifier import Classifier
from lalamo.modules.mlp import MixtureOfExperts, MLPBase
from lalamo.quantization import QuantizationMode

from .common import load_parameters
from .utils import decode_mxfp4, deinterleave_pairwise_columns

__all__ = ["load_huggingface_decoder"]


AWQ_REVERSE_ORDER = jnp.array([0, 4, 1, 5, 2, 6, 3, 7], dtype=jnp.int32)


def _reverse_uint4_awq_order(array: Array) -> Array:
    """Reverses the AWQ packing order to get the logical order of channels for INT4."""
    pack_factor = 32 // 4
    *_, last_dim = array.shape
    if last_dim % pack_factor != 0:
        return array

    array_reshaped = rearrange(array, "... (group pack_factor) -> ... group pack_factor", pack_factor=pack_factor)
    array_reordered = array_reshaped[..., AWQ_REVERSE_ORDER]
    return rearrange(array_reordered, "... group pack_factor -> ... (group pack_factor)")


def unpack_int32(packed_weights: Array, mode: QuantizationMode) -> Array:
    assert packed_weights.dtype == jnp.int32, (
        f"Expected packed_weights to be of dtype jnp.int32, got {packed_weights.dtype}"
    )
    assert 32 % mode.bits == 0

    shifts = jnp.arange(0, 32, mode.bits)
    mask = (2**mode.bits) - 1
    unpacked = jnp.bitwise_and(jnp.right_shift(packed_weights[:, :, None], shifts[None, None, :]), mask)
    unpacked = rearrange(
        unpacked,
        "out_channels packed_groups packed_values -> out_channels (packed_groups packed_values)",
    )

    return unpacked


def _process_quantized_tensors(
    qweights: Array,
    qzeros: Array,
    scales: Array,
    module: GroupQuantizedLinear,
) -> tuple[Array, Array, Array]:
    """Unpacks, recenters, transposes, and casts quantized tensors to the correct dtype."""
    mode = module.config.weight_quantization_mode
    assert qweights.dtype == jnp.int32
    unpacked_weights = unpack_int32(qweights, mode)
    if mode == QuantizationMode.UINT4:
        unpacked_weights = _reverse_uint4_awq_order(unpacked_weights)

    assert qzeros.dtype == jnp.int32
    unpacked_zero_points = unpack_int32(qzeros, mode)
    if mode == QuantizationMode.UINT4:
        unpacked_zero_points = _reverse_uint4_awq_order(unpacked_zero_points)

    weights = unpacked_weights.astype(module.config.activation_precision)
    zero_points = unpacked_zero_points.astype(module.config.activation_precision)
    processed_scales = scales.astype(module.config.activation_precision)

    return weights, zero_points, processed_scales


def _fuse_full_precision_weights(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    sublayers_to_fuse: list[str] | None,
) -> Array:
    if sublayers_to_fuse is None:
        return weights_dict[path / "weight"]

    weights = [weights_dict[path / layer_name / "weight"] for layer_name in sublayers_to_fuse]
    return jnp.concatenate(weights, axis=0)


def _fuse_quantized_weights(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    sublayers_to_fuse: list[str] | None,
) -> tuple[Array, Array, Array]:
    # Note that AWQ quantized weights are stored transposed relative to full-precision weights

    if sublayers_to_fuse is None:
        qweights = weights_dict[path / "qweight"]
        qzeros = weights_dict[path / "qzeros"]
        scales = weights_dict[path / "scales"]
        return qweights, qzeros, scales

    qweights = [weights_dict[path / layer_name / "qweight"] for layer_name in sublayers_to_fuse]
    qzeros = [weights_dict[path / layer_name / "qzeros"] for layer_name in sublayers_to_fuse]
    scales = [weights_dict[path / layer_name / "scales"] for layer_name in sublayers_to_fuse]

    fused_qweights = jnp.concatenate(qweights, axis=1)
    fused_qzeros = jnp.concatenate(qzeros, axis=1)
    fused_scales = jnp.concatenate(scales, axis=1)

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
        qweights, qzeros, scales = _fuse_quantized_weights(weights_dict, path, sublayers_to_fuse)

        weights, zero_points, scales = _process_quantized_tensors(
            qweights,
            qzeros,
            scales,
            module,
        )

        return load_parameters(
            lambda m: (m.weights, m.scales, m.zero_points, m.biases),
            module,
            (weights.T, scales.T, zero_points.T, bias),
        )

    raise TypeError(f"Unsupported module type for loading: {type(module)}")


def load_mlp(module: MLPBase, weights_dict: Mapping[str, Array], path: ParameterPath) -> MLPBase:
    if isinstance(module, DenseMLP):
        # Standard dense MLP with separate sublayers.
        up_projection = load_linear(
            module.up_projection,
            weights_dict,
            path,
            sublayers_to_fuse=["up_proj", "gate_proj"],
        )
        down_projection = load_linear(module.down_projection, weights_dict, path / "down_proj")
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
            lambda m: (m.weights, m.biases),  # type: ignore
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
            down_b = jnp.broadcast_to(down_b, down_w.shape[:-1] + (down_b.shape[0],))

        down_projection = load_parameters(
            lambda m: (m.weights, m.biases),  # type: ignore
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
        experts = load_mlp(module.experts, weights_dict, experts_path)

    return load_parameters(
        lambda m: (m.router, m.experts),
        module,
        (router, experts),
    )


def load_rmsnorm(
    module: RMSNorm,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> RMSNorm:
    scales = weights_dict[path / "weight"]
    return load_parameters(lambda m: (m.scales,), module, (scales,))


def load_attention(
    module: Attention,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> Attention:
    qkv_projection = load_linear(
        module.qkv_projection,
        weights_dict,
        path,
        sublayers_to_fuse=["q_proj", "k_proj", "v_proj"],
    )
    out_projection = load_linear(module.out_projection, weights_dict, path / "o_proj")

    if module.query_norm is not None:
        query_norm = load_rmsnorm(module.query_norm, weights_dict, path / "q_norm")
    else:
        query_norm = None

    if module.key_norm is not None:
        key_norm = load_rmsnorm(module.key_norm, weights_dict, path / "k_norm")
    else:
        key_norm = None

    # GPT-OSS adds per-head attention sinks; load them if present.
    if (path / "sinks") in weights_dict:
        sinks = weights_dict[path / "sinks"]
    else:
        sinks = module.sinks

    return load_parameters(
        lambda m: (m.qkv_projection, m.out_projection, m.query_norm, m.key_norm, m.sinks),
        module,
        (qkv_projection, out_projection, query_norm, key_norm, sinks),
    )


def load_decoder_layer(
    module: DecoderLayer,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> DecoderLayer:
    pre_attention_norm = load_rmsnorm(
        module.pre_attention_norm,
        weights_dict,
        path / "input_layernorm",
    )
    attention = load_attention(module.attention, weights_dict, path / "self_attn")
    if module.post_attention_norm is not None:
        post_attention_norm = load_rmsnorm(
            module.post_attention_norm,
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
            path / "post_attention_layernorm",
        )

    mlp = load_mlp(module.mlp, weights_dict, path / "mlp")
    if module.post_mlp_norm is not None:
        post_mlp_norm = load_rmsnorm(
            module.post_mlp_norm,
            weights_dict,
            path / "post_feedforward_layernorm",
        )
    else:
        post_mlp_norm = None
    return load_parameters(
        lambda m: (m.pre_attention_norm, m.attention, m.post_attention_norm, m.pre_mlp_norm, m.mlp, m.post_mlp_norm),
        module,
        (pre_attention_norm, attention, post_attention_norm, pre_mlp_norm, mlp, post_mlp_norm),
    )


def load_tied_embedding(
    module: TiedEmbedding,
    weights_dict: Mapping[str, Array],
    decoder_path: ParameterPath,
) -> TiedEmbedding:
    weights = weights_dict[decoder_path / "embed_tokens" / "weight"]
    return load_parameters(lambda m: (m.weights,), module, (weights,))


def load_untied_embedding(
    module: UntiedEmbedding,
    weights_dict: Mapping[str, Array],
    decoder_path: ParameterPath,
    lm_head_path: ParameterPath,
) -> UntiedEmbedding:
    input_weights = weights_dict[decoder_path / "embed_tokens" / "weight"]
    output_weights = weights_dict[lm_head_path / "weight"]
    return load_parameters(lambda m: (m.input_weights, m.output_weights), module, (input_weights, output_weights))


def load_huggingface_decoder(
    module: Decoder,
    weights_dict: Mapping[str, Array],
) -> Decoder:
    if any(key.startswith("language_model.") for key in weights_dict):
        base_path = ParameterPath("language_model")
    else:
        base_path = ParameterPath()

    decoder_path = base_path / "model"
    lm_head_path = base_path / "lm_head"

    if isinstance(module.embedding, TiedEmbedding):
        embedding = load_tied_embedding(module.embedding, weights_dict, decoder_path)
    elif isinstance(module.embedding, UntiedEmbedding):
        embedding = load_untied_embedding(module.embedding, weights_dict, decoder_path, lm_head_path)
    else:
        raise TypeError(f"Unsupported embedding type: {type(module.embedding)}")
    decoder_layers = tuple(
        load_decoder_layer(layer, weights_dict, decoder_path / "layers" / i) for i, layer in enumerate(module.transformer.layers)
    )
    output_norm = load_rmsnorm(module.transformer.output_norm, weights_dict, decoder_path / "norm")
    return load_parameters(
        lambda m: (m.embedding, m.transformer.layers, m.transformer.output_norm),
        module,
        (embedding, decoder_layers, output_norm),
    )

def load_huggingface_classifier(
    module: Classifier,
    weights_dict: Mapping[str, Array],
) -> Classifier:

    # TODO: do the actual weighs loading

    return module
