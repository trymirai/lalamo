import jax.numpy as jnp
from jaxtyping import Array

from fartsovka.importers.common import WeightsPath, load_parameters
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


def load_linear(module: Linear, weights_dict: dict[str, Array], path: WeightsPath) -> Linear:
    return load_parameters(lambda m: (m.weights,), module, (weights_dict[path / "weight"],))


def load_mlp(module: BaselineMLP, weights_dict: dict[str, Array], path: WeightsPath) -> BaselineMLP:
    up_proj_weights = weights_dict[path / "up_proj" / "weight"]
    gate_proj_weights = weights_dict[path / "gate_proj" / "weight"]
    fused_up_gate_weights = jnp.concatenate([up_proj_weights, gate_proj_weights], axis=0)

    down_proj_weights = weights_dict[path / "down_proj" / "weight"]

    return load_parameters(
        lambda m: (m.up_projection.weights, m.down_projection.weights),
        module,
        (fused_up_gate_weights, down_proj_weights),
    )


def load_rmsnorm(module: RMSNorm, weights_dict: dict[str, Array], path: WeightsPath) -> RMSNorm:
    return load_parameters(lambda m: (m.scale,), module, (weights_dict[path / "weight"],))


def load_attention(
    module: BaselineAttention,
    weights_dict: dict[str, Array],
    path: WeightsPath,
) -> BaselineAttention:
    out_proj = load_linear(module.out_projection, weights_dict, path / "o_proj")
    q_proj_weights = weights_dict[path / "q_proj" / "weight"]
    k_proj_weights = weights_dict[path / "k_proj" / "weight"]
    v_proj_weights = weights_dict[path / "v_proj" / "weight"]
    qkv_proj_weights = jnp.concatenate([q_proj_weights, k_proj_weights, v_proj_weights], axis=0)
    return load_parameters(
        lambda m: (m.qkv_projection.weights, m.out_projection),
        module,
        (qkv_proj_weights, out_proj),
    )


def load_decoder_layer(
    module: BaselineDecoderLayer,
    weights_dict: dict[str, Array],
    path: WeightsPath,
) -> BaselineDecoderLayer:
    attention_norm = load_rmsnorm(module.attention_norm, weights_dict, path / "input_layernorm")
    attention = load_attention(module.attention, weights_dict, path / "self_attn")
    mlp_norm = load_rmsnorm(module.mlp_norm, weights_dict, path / "post_attention_layernorm")
    mlp = load_mlp(module.mlp, weights_dict, path / "mlp")
    return load_parameters(
        lambda m: (m.attention_norm, m.attention, m.mlp_norm, m.mlp),
        module,
        (attention_norm, attention, mlp_norm, mlp),
    )


def load_embedding(module: Embedding, weights_dict: dict[str, Array], path: WeightsPath) -> Embedding:
    weights = weights_dict[path / "weight"]
    return load_parameters(lambda m: (m.weights,), module, (weights,))


def load_llama(
    module: BaselineLlama,
    weights_dict: dict[str, Array],
) -> BaselineLlama:
    root_path: WeightsPath = WeightsPath("model")
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
