import jax.numpy as jnp
from jaxtyping import Array

from fartsovka.common import ParameterPath
from fartsovka.importers.common import load_parameters
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
from fartsovka.modules.embedding import Embedding
from fartsovka.modules.linear import Linear
from fartsovka.modules.normalization import RMSNorm

__all__ = ["load_huggingface"]


def load_linear(module: Linear, weights_dict: dict[str, Array], path: ParameterPath) -> Linear:
    if module.bias is None:
        if path / "bias" in weights_dict:
            raise ValueError(f"Bias is not supported for {path}")
        loaded_bias = None
    else:
        loaded_bias = weights_dict[path / "bias"]
    return load_parameters(
        lambda m: (m.weights, m.bias),
        module,
        (weights_dict[path / "weight"], loaded_bias),
    )


def load_mlp[T: LlamaMLP | Qwen2MLP](module: T, weights_dict: dict[str, Array], path: ParameterPath) -> T:
    up_proj_weights = weights_dict[path / "up_proj" / "weight"]
    gate_proj_weights = weights_dict[path / "gate_proj" / "weight"]
    fused_up_gate_weights = jnp.concatenate([up_proj_weights, gate_proj_weights], axis=0)

    down_proj_weights = weights_dict[path / "down_proj" / "weight"]

    return load_parameters(
        lambda m: (m.up_projection.weights, m.down_projection.weights),
        module,
        (fused_up_gate_weights, down_proj_weights),
    )


def load_rmsnorm(module: RMSNorm, weights_dict: dict[str, Array], path: ParameterPath) -> RMSNorm:
    return load_parameters(lambda m: (m.scale,), module, (weights_dict[path / "weight"],))


def load_attention[T: LlamaAttention | Qwen2Attention](
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
    if module.qkv_projection.bias is None:
        for bias_path in bias_paths:
            if bias_path in weights_dict:
                raise ValueError(f"Bias is not supported for {bias_path}")
        qkv_bias = None
    else:
        loaded_biases = [weights_dict[bias_path] for bias_path in bias_paths]
        qkv_bias = jnp.concatenate(loaded_biases, axis=0)

    return load_parameters(
        lambda m: (m.qkv_projection.weights, m.qkv_projection.bias, m.out_projection),
        module,
        (qkv_proj_weights, qkv_bias, out_proj),
    )


def load_decoder_layer[T: LlamaDecoderLayer | Qwen2DecoderLayer](
    module: T,
    weights_dict: dict[str, Array],
    path: ParameterPath,
) -> T:
    attention_norm = load_rmsnorm(module.attention_norm, weights_dict, path / "input_layernorm")
    attention = load_attention(module.attention, weights_dict, path / "self_attn")
    mlp_norm = load_rmsnorm(module.mlp_norm, weights_dict, path / "post_attention_layernorm")
    mlp = load_mlp(module.mlp, weights_dict, path / "mlp")
    return load_parameters(
        lambda m: (m.attention_norm, m.attention, m.mlp_norm, m.mlp),
        module,
        (attention_norm, attention, mlp_norm, mlp),
    )


def load_embedding(module: Embedding, weights_dict: dict[str, Array], path: ParameterPath) -> Embedding:
    weights = weights_dict[path / "weight"]
    return load_parameters(lambda m: (m.weights,), module, (weights,))


def load_huggingface[T: LlamaDecoder | Qwen2Decoder](
    module: T,
    weights_dict: dict[str, Array],
) -> T:
    root_path: ParameterPath = ParameterPath("model")
    embedding = load_embedding(module.embedding, weights_dict, root_path / "embed_tokens")
    decoder_layers = [
        load_decoder_layer(layer, weights_dict, root_path / "layers" / i) for i, layer in enumerate(module.layers)
    ]
    output_norm = load_rmsnorm(module.output_norm, weights_dict, root_path / "norm")
    return load_parameters(
        lambda m: (m.embedding, m.layers, m.output_norm),
        module,
        (embedding, decoder_layers, output_norm),
    )
