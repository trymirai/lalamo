import jax.numpy as jnp
from jaxtyping import Array

from fartsovka.importers.common import load_parameters
from fartsovka.models.baseline_llama import (
    BaselineAttention,
    BaselineDecoderLayer,
    BaselineLlama,
    BaselineMLP,
)
from fartsovka.modules.embedding import Embedding
from fartsovka.modules.linear import Linear
from fartsovka.modules.normalization import RMSNorm

__all__ = ["load_llama"]


def load_linear(module: Linear, weights_dict: dict[str, Array], name_prefix: str) -> Linear:
    return load_parameters(lambda m: (m.weights,), module, (weights_dict[f"{name_prefix}.weight"],))


def load_mlp(module: BaselineMLP, weights_dict: dict[str, Array], name_prefix: str) -> BaselineMLP:
    up_proj_weights = weights_dict[f"{name_prefix}.up_proj.weight"]
    gate_proj_weights = weights_dict[f"{name_prefix}.gate_proj.weight"]
    fused_up_gate_weights = jnp.concatenate([up_proj_weights, gate_proj_weights], axis=0)

    down_proj_weights = weights_dict[f"{name_prefix}.down_proj.weight"]

    return load_parameters(
        lambda m: (m.up_projection.weights, m.down_projection.weights),
        module,
        (fused_up_gate_weights, down_proj_weights),
    )


def load_rmsnorm(module: RMSNorm, weights_dict: dict[str, Array], name_prefix: str) -> RMSNorm:
    return load_parameters(lambda m: (m.scale,), module, (weights_dict[f"{name_prefix}.weight"],))


def load_attention(
    module: BaselineAttention,
    weights_dict: dict[str, Array],
    name_prefix: str,
) -> BaselineAttention:
    out_proj = load_linear(module.out_projection, weights_dict, f"{name_prefix}.o_proj")
    q_proj_weights = weights_dict[f"{name_prefix}.q_proj.weight"]
    k_proj_weights = weights_dict[f"{name_prefix}.k_proj.weight"]
    v_proj_weights = weights_dict[f"{name_prefix}.v_proj.weight"]
    qkv_proj_weights = jnp.concatenate([q_proj_weights, k_proj_weights, v_proj_weights], axis=0)
    return load_parameters(
        lambda m: (m.qkv_projection.weights, m.out_projection),
        module,
        (qkv_proj_weights, out_proj),
    )


def load_decoder_layer(
    module: BaselineDecoderLayer,
    weights_dict: dict[str, Array],
    name_prefix: str,
) -> BaselineDecoderLayer:
    attention_norm = load_rmsnorm(module.attention_norm, weights_dict, f"{name_prefix}.input_layernorm")
    attention = load_attention(module.attention, weights_dict, f"{name_prefix}.self_attn")
    mlp_norm = load_rmsnorm(module.mlp_norm, weights_dict, f"{name_prefix}.post_attention_layernorm")
    mlp = load_mlp(module.mlp, weights_dict, f"{name_prefix}.mlp")
    return load_parameters(
        lambda m: (m.attention_norm, m.attention, m.mlp_norm, m.mlp),
        module,
        (attention_norm, attention, mlp_norm, mlp),
    )


def load_embedding(module: Embedding, weights_dict: dict[str, Array], name_prefix: str) -> Embedding:
    weights = weights_dict[f"{name_prefix}.weight"]
    return load_parameters(lambda m: (m.weights,), module, (weights,))


def load_llama(module: BaselineLlama, weights_dict: dict[str, Array], name_prefix: str = "model") -> BaselineLlama:
    embedding = load_embedding(module.embedding, weights_dict, f"{name_prefix}.embed_tokens")
    decoder_layers = [
        load_decoder_layer(layer, weights_dict, f"{name_prefix}.layers.{i}") for i, layer in enumerate(module.layers)
    ]
    out_norm = load_rmsnorm(module.out_norm, weights_dict, f"{name_prefix}.norm")
    return load_parameters(
        lambda m: (m.embedding, m.layers, m.out_norm),
        module,
        (embedding, decoder_layers, out_norm),
    )
