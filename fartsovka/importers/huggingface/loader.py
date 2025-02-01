import jax.numpy as jnp
from jaxtyping import Array

from fartsovka.common import ParameterPath
from fartsovka.importers.common import load_parameters
from fartsovka.models.gemma2 import (
    Gemma2Attention,
    Gemma2Decoder,
    Gemma2DecoderLayer,
    Gemma2MLP,
)
from fartsovka.models.llama import (
    LlamaAttention,
    LlamaDecoder,
    LlamaDecoderLayer,
    LlamaMLP,
)
from fartsovka.models.qwen2 import (
    Qwen2Attention,
    Qwen2Decoder,
    Qwen2DecoderLayer,
    Qwen2MLP,
)
from fartsovka.modules.decoder_layer import PrePostNormDecoderLayer
from fartsovka.modules.embedding import TiedEmbedding
from fartsovka.modules.linear import FullPrecisionLinear
from fartsovka.modules.normalization import RMSNorm

__all__ = ["load_huggingface"]


def load_linear(module: FullPrecisionLinear, weights_dict: dict[str, Array], path: ParameterPath) -> FullPrecisionLinear:
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


def load_mlp[T: LlamaMLP | Qwen2MLP | Gemma2MLP](module: T, weights_dict: dict[str, Array], path: ParameterPath) -> T:
    up_proj_weights = weights_dict[path / "up_proj" / "weight"]
    gate_proj_weights = weights_dict[path / "gate_proj" / "weight"]
    fused_up_gate_weights = jnp.concatenate([up_proj_weights, gate_proj_weights], axis=0)

    down_proj_weights = weights_dict[path / "down_proj" / "weight"]

    return load_parameters(
        lambda m: (m.up_projection.weights, m.down_projection.weights),
        module,
        (fused_up_gate_weights, down_proj_weights),
    )


def load_rmsnorm(
    module: RMSNorm,
    weights_dict: dict[str, Array],
    path: ParameterPath,
    add_one: bool,
) -> RMSNorm:
    weights = weights_dict[path / "weight"]
    if add_one:
        weights = weights + 1.0
    return load_parameters(lambda m: (m.scale,), module, (weights,))


def load_attention[T: LlamaAttention | Qwen2Attention | Gemma2Attention](
    module: T,
    weights_dict: dict[str, Array],
    path: ParameterPath,
) -> T:
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
        lambda m: (m.qkv_projection.weights, m.qkv_projection.biases, m.out_projection),
        module,
        (qkv_proj_weights, qkv_bias, out_proj),
    )


def load_pre_norm_decoder_layer[T: LlamaDecoderLayer | Qwen2DecoderLayer](
    module: T,
    weights_dict: dict[str, Array],
    path: ParameterPath,
    add_one_to_rms_norm_weights: bool,
) -> T:
    attention_norm = load_rmsnorm(
        module.attention_norm,
        weights_dict,
        path / "input_layernorm",
        add_one_to_rms_norm_weights,
    )
    attention = load_attention(module.attention, weights_dict, path / "self_attn")
    mlp_norm = load_rmsnorm(
        module.mlp_norm,
        weights_dict,
        path / "post_attention_layernorm",
        add_one_to_rms_norm_weights,
    )
    mlp = load_mlp(module.mlp, weights_dict, path / "mlp")
    return load_parameters(
        lambda m: (m.attention_norm, m.attention, m.mlp_norm, m.mlp),
        module,
        (attention_norm, attention, mlp_norm, mlp),
    )


def load_pre_post_norm_decoder_layer(
    module: Gemma2DecoderLayer,
    weights_dict: dict[str, Array],
    path: ParameterPath,
    add_one_to_rms_norm_weights: bool,
) -> Gemma2DecoderLayer:
    attention_pre_norm = load_rmsnorm(
        module.attention_pre_norm,
        weights_dict,
        path / "input_layernorm",
        add_one_to_rms_norm_weights,
    )
    attention = load_attention(module.attention, weights_dict, path / "self_attn")
    attention_post_norm = load_rmsnorm(
        module.attention_post_norm,
        weights_dict,
        path / "post_attention_layernorm",
        add_one_to_rms_norm_weights,
    )
    mlp_pre_norm = load_rmsnorm(
        module.mlp_pre_norm,
        weights_dict,
        path / "pre_feedforward_layernorm",
        add_one_to_rms_norm_weights,
    )
    mlp = load_mlp(module.mlp, weights_dict, path / "mlp")
    mlp_post_norm = load_rmsnorm(
        module.mlp_post_norm,
        weights_dict,
        path / "post_feedforward_layernorm",
        add_one_to_rms_norm_weights,
    )
    return load_parameters(
        lambda m: (m.attention_pre_norm, m.attention, m.attention_post_norm, m.mlp_pre_norm, m.mlp, m.mlp_post_norm),
        module,
        (attention_pre_norm, attention, attention_post_norm, mlp_pre_norm, mlp, mlp_post_norm),
    )


def load_decoder_layer[T: LlamaDecoderLayer | Qwen2DecoderLayer | Gemma2DecoderLayer](
    module: T,
    weights_dict: dict[str, Array],
    path: ParameterPath,
    add_one_to_rms_norm_weights: bool,
) -> T:
    if isinstance(module, PrePostNormDecoderLayer):
        return load_pre_post_norm_decoder_layer(module, weights_dict, path, add_one_to_rms_norm_weights)
    return load_pre_norm_decoder_layer(module, weights_dict, path, add_one_to_rms_norm_weights)


def load_embedding(module: TiedEmbedding, weights_dict: dict[str, Array], path: ParameterPath) -> TiedEmbedding:
    weights = weights_dict[path / "weight"]
    return load_parameters(lambda m: (m.weights,), module, (weights,))


def load_huggingface[T: LlamaDecoder | Qwen2Decoder | Gemma2Decoder](
    module: T,
    weights_dict: dict[str, Array],
    add_one_to_rms_norm_weights: bool,
) -> T:
    root_path: ParameterPath = ParameterPath("model")
    embedding = load_embedding(module.embedding, weights_dict, root_path / "embed_tokens")
    decoder_layers = [
        load_decoder_layer(layer, weights_dict, root_path / "layers" / i, add_one_to_rms_norm_weights)
        for i, layer in enumerate(module.layers)
    ]
    output_norm = load_rmsnorm(module.output_norm, weights_dict, root_path / "norm", add_one_to_rms_norm_weights)
    return load_parameters(
        lambda m: (m.embedding, m.layers, m.output_norm),
        module,
        (embedding, decoder_layers, output_norm),
    )
