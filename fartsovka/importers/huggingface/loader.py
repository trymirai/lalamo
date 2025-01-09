import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from fartsovka.modules.embedding import Embedding
from fartsovka.modules.linear import Linear
from fartsovka.modules.llama import (
    BaselineAttention,
    BaselineDecoderLayer,
    BaselineLlama,
    BaselineMLP,
)
from fartsovka.modules.normalisation import RMSNorm

__all__ = ["load_llama"]


def load_linear(module: Linear, weights_dict: dict[str, Array], name_prefix: str) -> Linear:
    weights: Float[Array, "out_channels in_channels"] = weights_dict[f"{name_prefix}.weight"]
    weights = weights.astype(module.precision)
    if weights.shape != (module.output_dim, module.input_dim):
        raise ValueError(f"Expected weights shape {module.output_dim, module.input_dim}, got {weights.shape}")
    return eqx.tree_at(lambda m: m.weights, module, weights)


def load_mlp(module: BaselineMLP, weights_dict: dict[str, Array], name_prefix: str) -> BaselineMLP:
    up_proj_weights: Float[Array, "hidden_channels in_channels"] = weights_dict[f"{name_prefix}.up_proj.weight"]
    up_proj_weights = up_proj_weights.astype(module.up_projection.precision)
    if up_proj_weights.shape != (module.hidden_dim, module.model_dim):
        expected_shape = (module.hidden_dim, module.model_dim)
        raise ValueError(f"Expected up_proj weights shape {expected_shape}, got {up_proj_weights.shape}")

    gate_weights: Float[Array, "hidden_channels in_channels"] = weights_dict[f"{name_prefix}.gate_proj.weight"]
    gate_weights = gate_weights.astype(module.up_projection.precision)
    if gate_weights.shape != (module.hidden_dim, module.model_dim):
        expected_shape = (module.hidden_dim, module.model_dim)
        raise ValueError(f"Expected gate weights shape {expected_shape}, got {gate_weights.shape}")

    down_proj_weights: Float[Array, "in_channels hidden_channels"] = weights_dict[f"{name_prefix}.down_proj.weight"]
    down_proj_weights = down_proj_weights.astype(module.down_projection.precision)
    if down_proj_weights.shape != (module.model_dim, module.hidden_dim):
        expected_shape = (module.model_dim, module.hidden_dim)
        raise ValueError(f"Expected down_proj weights shape {expected_shape}, got {down_proj_weights.shape}")

    fused_hidden_gate_weights: Float[Array, "2*hidden_channels in_channels"] = jnp.concatenate(
        [up_proj_weights, gate_weights],
        axis=0,
    )
    return eqx.tree_at(
        lambda m: (m.up_projection.weights, m.down_projection.weights),
        module,
        (fused_hidden_gate_weights, down_proj_weights),
    )


def load_rmsnorm(module: RMSNorm, weights_dict: dict[str, Array], name_prefix: str) -> RMSNorm:
    weights: Float[Array, " channels"] = weights_dict[f"{name_prefix}.weight"]
    weights = weights.astype(module.precision)
    if weights.shape != (module.model_dim,):
        raise ValueError(f"Expected weights shape {module.model_dim}, got {weights.shape}")
    return eqx.tree_at(lambda m: m.scale, module, weights)


def load_attention(
    module: BaselineAttention,
    weights_dict: dict[str, Array],
    name_prefix: str,
) -> BaselineAttention:
    out_proj = load_linear(module.out_projection, weights_dict, f"{name_prefix}.o_proj")

    expected_q_proj_shape = (module.num_heads * module.head_dim, module.model_dim)
    expected_kv_proj_shape = (module.num_groups * module.head_dim, module.model_dim)
    q_proj_weights: Float[Array, "q_channels in_channels"] = weights_dict[f"{name_prefix}.q_proj.weight"]
    q_proj_weights = q_proj_weights.astype(module.qkv_projection.precision)
    if q_proj_weights.shape != expected_q_proj_shape:
        raise ValueError(f"Expected q_proj weights shape {expected_q_proj_shape}, got {q_proj_weights.shape}")

    k_proj_weights: Float[Array, "kv_channels in_channels"] = weights_dict[f"{name_prefix}.k_proj.weight"]
    k_proj_weights = k_proj_weights.astype(module.qkv_projection.precision)
    if k_proj_weights.shape != expected_kv_proj_shape:
        raise ValueError(f"Expected k_proj weights shape {expected_kv_proj_shape}, got {k_proj_weights.shape}")

    v_proj_weights: Float[Array, "kv_channels in_channels"] = weights_dict[f"{name_prefix}.v_proj.weight"]
    v_proj_weights = v_proj_weights.astype(module.qkv_projection.precision)
    if v_proj_weights.shape != expected_kv_proj_shape:
        raise ValueError(f"Expected v_proj weights shape {expected_kv_proj_shape}, got {v_proj_weights.shape}")

    qkv_proj_weights: Float[Array, "q_channels+2*kv_channels in_channels"] = jnp.concatenate(
        [q_proj_weights, k_proj_weights, v_proj_weights],
        axis=0,
    )
    return eqx.tree_at(lambda m: (m.qkv_projection.weights, m.out_projection), module, (qkv_proj_weights, out_proj))


def load_decoder_layer(
    module: BaselineDecoderLayer,
    weights_dict: dict[str, Array],
    name_prefix: str,
) -> BaselineDecoderLayer:
    attention_norm = load_rmsnorm(module.attention_norm, weights_dict, f"{name_prefix}.input_layernorm")
    attention = load_attention(module.attention, weights_dict, f"{name_prefix}.self_attn")
    mlp_norm = load_rmsnorm(module.mlp_norm, weights_dict, f"{name_prefix}.post_attention_layernorm")
    mlp = load_mlp(module.mlp, weights_dict, f"{name_prefix}.mlp")
    return eqx.tree_at(
        lambda m: (m.attention_norm, m.attention, m.mlp_norm, m.mlp),
        module,
        (attention_norm, attention, mlp_norm, mlp),
    )


def load_embedding(module: Embedding, weights_dict: dict[str, Array], name_prefix: str) -> Embedding:
    weights: Float[Array, "token_ids channels"] = weights_dict[f"{name_prefix}.weight"]
    weights = weights.astype(module.precision)
    if weights.shape != (module.vocab_dim, module.model_dim):
        raise ValueError(f"Expected weights shape {module.vocab_dim, module.model_dim}, got {weights.shape}")
    return eqx.tree_at(lambda m: m.weights, module, weights)


def load_llama(module: BaselineLlama, weights_dict: dict[str, Array], name_prefix: str = "model") -> BaselineLlama:
    embedding = load_embedding(module.embedding, weights_dict, f"{name_prefix}.embed_tokens")
    decoder_layers = [
        load_decoder_layer(layer, weights_dict, f"{name_prefix}.layers.{i}") for i, layer in enumerate(module.layers)
    ]
    out_norm = load_rmsnorm(module.out_norm, weights_dict, f"{name_prefix}.norm")
    return eqx.tree_at(lambda m: (m.embedding, m.layers, m.out_norm), module, (embedding, decoder_layers, out_norm))
