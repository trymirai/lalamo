import jax.numpy as jnp
from jaxtyping import Array

from fartsovka.common import ParameterPath
from fartsovka.modules import (
    MLP,
    Attention,
    DecoderLayer,
    FullPrecisionLinear,
    RMSNorm,
    TiedEmbedding,
    UntiedEmbedding,
)
from fartsovka.modules.decoder import Decoder

from .common import load_parameters

__all__ = ["load_huggingface"]


def load_linear(
    module: FullPrecisionLinear,
    weights_dict: dict[str, Array],
    path: ParameterPath,
) -> FullPrecisionLinear:
    if module.biases is None:
        if path / "bias" in weights_dict:
            raise ValueError(f"Bias is not supported for {path}")
        loaded_bias = None
    else:
        loaded_bias = weights_dict[path / "bias"]
    return load_parameters(
        lambda m: (m.weights, m.biases),
        module,
        (weights_dict[path / "weight"], loaded_bias),
    )


def load_mlp(module: MLP, weights_dict: dict[str, Array], path: ParameterPath) -> MLP:
    if not isinstance(module.up_projection, FullPrecisionLinear):
        raise TypeError(f"Expected up_projection to be FullPrecisionLinear, got {type(module.up_projection)}")
    if not isinstance(module.down_projection, FullPrecisionLinear):
        raise TypeError(f"Expected down_projection to be FullPrecisionLinear, got {type(module.down_projection)}")

    up_proj_weights = weights_dict[path / "up_proj" / "weight"]
    gate_proj_weights = weights_dict[path / "gate_proj" / "weight"]
    fused_up_gate_weights = jnp.concatenate([up_proj_weights, gate_proj_weights], axis=0)

    down_proj_weights = weights_dict[path / "down_proj" / "weight"]

    return load_parameters(
        lambda m: (m.up_projection.weights, m.down_projection.weights),  # type: ignore
        module,
        (fused_up_gate_weights, down_proj_weights),
    )


def load_rmsnorm(
    module: RMSNorm,
    weights_dict: dict[str, Array],
    path: ParameterPath,
    add_one: bool,
) -> RMSNorm:
    scales = weights_dict[path / "weight"]
    if add_one:
        scales = scales + 1.0
    return load_parameters(lambda m: (m.scales,), module, (scales,))


def load_attention(
    module: Attention,
    weights_dict: dict[str, Array],
    path: ParameterPath,
) -> Attention:
    if not isinstance(module.qkv_projection, FullPrecisionLinear):
        raise TypeError(f"Expected qkv_projection to be FullPrecisionLinear, got {type(module.qkv_projection)}")
    if not isinstance(module.out_projection, FullPrecisionLinear):
        raise TypeError(f"Expected out_projection to be FullPrecisionLinear, got {type(module.out_projection)}")
    out_proj = load_linear(module.out_projection, weights_dict, path / "o_proj")
    q_proj_weights = weights_dict[path / "q_proj" / "weight"]
    k_proj_weights = weights_dict[path / "k_proj" / "weight"]
    v_proj_weights = weights_dict[path / "v_proj" / "weight"]

    qkv_proj_weights = jnp.concatenate([q_proj_weights, k_proj_weights, v_proj_weights], axis=0)

    bias_paths = [path / p / "bias" for p in ["q_proj", "k_proj", "v_proj"]]
    if module.qkv_projection.biases is None:
        for bias_path in bias_paths:
            if bias_path in weights_dict:
                raise ValueError(f"Bias is not supported for {bias_path}")
        qkv_bias = None
    else:
        loaded_biases = [weights_dict[bias_path] for bias_path in bias_paths]
        qkv_bias = jnp.concatenate(loaded_biases, axis=0)

    return load_parameters(
        lambda m: (m.qkv_projection.weights, m.qkv_projection.biases, m.out_projection),  # type: ignore
        module,
        (qkv_proj_weights, qkv_bias, out_proj),
    )


def load_decoder_layer(
    module: DecoderLayer,
    weights_dict: dict[str, Array],
    path: ParameterPath,
    add_one_to_rms_norm_weights: bool,
) -> DecoderLayer:
    pre_attention_norm = load_rmsnorm(
        module.pre_attention_norm,
        weights_dict,
        path / "input_layernorm",
        add_one_to_rms_norm_weights,
    )
    attention = load_attention(module.attention, weights_dict, path / "self_attn")
    if module.post_attention_norm is not None:
        post_attention_norm = load_rmsnorm(
            module.post_attention_norm,
            weights_dict,
            path / "post_attention_layernorm",
            add_one_to_rms_norm_weights,
        )

        pre_mlp_norm = load_rmsnorm(
            module.pre_mlp_norm,
            weights_dict,
            path / "pre_feedforward_layernorm",
            add_one_to_rms_norm_weights,
        )
    else:
        post_attention_norm = None

        pre_mlp_norm = load_rmsnorm(
            module.pre_mlp_norm,
            weights_dict,
            path / "post_attention_layernorm",
            add_one_to_rms_norm_weights,
        )

    mlp = load_mlp(module.mlp, weights_dict, path / "mlp")
    if module.post_mlp_norm is not None:
        post_mlp_norm = load_rmsnorm(
            module.post_mlp_norm,
            weights_dict,
            path / "post_feedforward_layernorm",
            add_one_to_rms_norm_weights,
        )
    else:
        post_mlp_norm = None
    return load_parameters(
        lambda m: (m.pre_attention_norm, m.attention, m.post_attention_norm, m.pre_mlp_norm, m.mlp, m.post_mlp_norm),
        module,
        (pre_attention_norm, attention, post_attention_norm, pre_mlp_norm, mlp, post_mlp_norm),
    )


def load_tied_embedding(module: TiedEmbedding, weights_dict: dict[str, Array]) -> TiedEmbedding:
    weights = weights_dict[ParameterPath("model") / "embed_tokens" / "weight"]
    return load_parameters(lambda m: (m.weights,), module, (weights,))


def load_untied_embedding(
    module: UntiedEmbedding,
    weights_dict: dict[str, Array],
) -> UntiedEmbedding:
    input_weights = weights_dict[ParameterPath("model") / "embed_tokens" / "weight"]
    output_weights = weights_dict[ParameterPath("lm_head") / "weight"]
    return load_parameters(lambda m: (m.input_weights, m.output_weights), module, (input_weights, output_weights))


def load_huggingface(
    module: Decoder,
    weights_dict: dict[str, Array],
    add_one_to_rms_norm_weights: bool,
) -> Decoder:
    root_path: ParameterPath = ParameterPath("model")
    if isinstance(module.embedding, TiedEmbedding):
        embedding = load_tied_embedding(module.embedding, weights_dict)
    elif isinstance(module.embedding, UntiedEmbedding):
        embedding = load_untied_embedding(module.embedding, weights_dict)
    else:
        raise TypeError(f"Unsupported embedding type: {type(module.embedding)}")
    decoder_layers = tuple(
        load_decoder_layer(layer, weights_dict, root_path / "layers" / i, add_one_to_rms_norm_weights)
        for i, layer in enumerate(module.layers)
    )
    output_norm = load_rmsnorm(module.output_norm, weights_dict, root_path / "norm", add_one_to_rms_norm_weights)
    return load_parameters(
        lambda m: (m.embedding, m.layers, m.output_norm),
        module,
        (embedding, decoder_layers, output_norm),
    )
