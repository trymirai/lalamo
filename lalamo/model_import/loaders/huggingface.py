import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array

from lalamo.common import ParameterPath
from lalamo.modules import (
    MLP,
    Attention,
    Decoder,
    DecoderLayer,
    FullPrecisionLinear,
    GroupQuantizedLinear,
    LinearBase,
    RMSNorm,
    TiedEmbedding,
    UntiedEmbedding,
)
from lalamo.quantization import QuantizationMode

from .common import load_parameters

__all__ = ["load_huggingface"]


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

    return weights.transpose(), zero_points.transpose(), processed_scales.transpose()


def _fuse_full_precision_weights(
    weights_dict: dict[str, Array],
    path: ParameterPath,
    sublayers_to_fuse: list[str] | None,
) -> Array:
    if sublayers_to_fuse is None:
        return weights_dict[path / "weight"]

    weights = [weights_dict[path / layer_name / "weight"] for layer_name in sublayers_to_fuse]
    return jnp.concatenate(weights, axis=0)


def _fuse_quantized_weights(
    weights_dict: dict[str, Array],
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
    weights_dict: dict[str, Array],
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
            (weights, scales, zero_points, bias),
        )

    raise TypeError(f"Unsupported module type for loading: {type(module)}")


def load_mlp(module: MLP, weights_dict: dict[str, Array], path: ParameterPath) -> MLP:
    up_projection = load_linear(module.up_projection, weights_dict, path, sublayers_to_fuse=["up_proj", "gate_proj"])
    down_projection = load_linear(module.down_projection, weights_dict, path / "down_proj")
    return load_parameters(lambda m: (m.up_projection, m.down_projection), module, (up_projection, down_projection))


def load_rmsnorm(
    module: RMSNorm,
    weights_dict: dict[str, Array],
    path: ParameterPath,
) -> RMSNorm:
    scales = weights_dict[path / "weight"]
    return load_parameters(lambda m: (m.scales,), module, (scales,))


def load_attention(
    module: Attention,
    weights_dict: dict[str, Array],
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

    return load_parameters(
        lambda m: (m.qkv_projection, m.out_projection, m.query_norm, m.key_norm),
        module,
        (qkv_projection, out_projection, query_norm, key_norm),
    )


def load_decoder_layer(
    module: DecoderLayer,
    weights_dict: dict[str, Array],
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
    weights_dict: dict[str, Array],
    decoder_path: ParameterPath,
) -> TiedEmbedding:
    weights = weights_dict[decoder_path / "embed_tokens" / "weight"]
    return load_parameters(lambda m: (m.weights,), module, (weights,))


def load_untied_embedding(
    module: UntiedEmbedding,
    weights_dict: dict[str, Array],
    decoder_path: ParameterPath,
    lm_head_path: ParameterPath,
) -> UntiedEmbedding:
    input_weights = weights_dict[decoder_path / "embed_tokens" / "weight"]
    output_weights = weights_dict[lm_head_path / "weight"]
    return load_parameters(lambda m: (m.input_weights, m.output_weights), module, (input_weights, output_weights))


def load_huggingface(
    module: Decoder,
    weights_dict: dict[str, Array],
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
        load_decoder_layer(layer, weights_dict, decoder_path / "layers" / i) for i, layer in enumerate(module.layers)
    )
    output_norm = load_rmsnorm(module.output_norm, weights_dict, decoder_path / "norm")
    return load_parameters(
        lambda m: (m.embedding, m.layers, m.output_norm),
        module,
        (embedding, decoder_layers, output_norm),
    )
