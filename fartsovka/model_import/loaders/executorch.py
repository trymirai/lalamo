from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace

import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, Int

from fartsovka.common import ParameterPath
from fartsovka.modules import MLP, Attention, Decoder, DecoderLayer, QLoRALinear, QuantizedTiedEmbedding, RMSNorm

from .common import load_parameters

__all__ = ["load_executorch"]


@dataclass
class QLoRALinearParams:
    weights: Int[Array, "out_channels in_channels"]
    scales: Float[Array, "out_channels groups"]
    lora_down_weights: Float[Array, "total_lora_channels in_channels"]
    lora_up_weights: tuple[Float[Array, "..."], ...]

    def __iter__(self) -> Iterator[Array]:
        yield self.weights
        yield self.scales
        yield self.lora_down_weights
        yield from self.lora_up_weights

    def __len__(self) -> int:
        return 3 + len(self.lora_up_weights)


def params_selector(module: QLoRALinear) -> tuple:
    return (
        module.weights,
        module.scales,
        module.lora_down_weights,
        *module.lora_up_weights,
    )


def get_qlora_linear_params(weights_dict: dict[str, Array], path: ParameterPath) -> QLoRALinearParams:
    weights = weights_dict[path / "weight"]
    scales = weights_dict[path / "scales"]
    lora_down_weights = weights_dict[path / "adaptor" / "A" / "weight"]
    lora_up_weights = (weights_dict[path / "adaptor" / "B" / "weight"],)
    return QLoRALinearParams(weights, scales, lora_down_weights, lora_up_weights)


def merge_linear_params(params_list: Iterable[QLoRALinearParams]) -> QLoRALinearParams:
    params_list = list(params_list)
    weights = jnp.concatenate([p.weights for p in params_list], axis=0)
    scales = jnp.concatenate([p.scales for p in params_list], axis=0)
    lora_down_weights = jnp.concatenate([p.lora_down_weights for p in params_list], axis=0)
    lora_up_weights = tuple(w for p in params_list for w in p.lora_up_weights)
    return QLoRALinearParams(weights, scales, lora_down_weights, lora_up_weights)


def load_linear(module: QLoRALinear, weights_dict: dict[str, Array], path: ParameterPath) -> QLoRALinear:
    params = get_qlora_linear_params(weights_dict, path)
    return load_parameters(params_selector, module, params)


def load_mlp(module: MLP, weights_dict: dict[str, Array], path: ParameterPath) -> MLP:
    if not isinstance(module.up_projection, QLoRALinear):
        raise TypeError(f"Expected up_projection to be QLoRALinear, got {type(module.up_projection)}")
    if not isinstance(module.down_projection, QLoRALinear):
        raise TypeError(f"Expected down_projection to be QLoRALinear, got {type(module.down_projection)}")

    up_proj_params = get_qlora_linear_params(weights_dict, path / "w3")
    gate_proj_params = get_qlora_linear_params(weights_dict, path / "w1")
    down_proj_params = get_qlora_linear_params(weights_dict, path / "w2")

    fused_up_gate_params = merge_linear_params([up_proj_params, gate_proj_params])

    return load_parameters(
        lambda m: (*params_selector(m.up_projection), *params_selector(m.down_projection)),  # type: ignore
        module,
        (*fused_up_gate_params, *down_proj_params),
    )


def load_rmsnorm(module: RMSNorm, weights_dict: dict[str, Array], path: ParameterPath) -> RMSNorm:
    return load_parameters(lambda m: (m.scales,), module, (weights_dict[path / "weight"],))


def permute_qk_weights(weights: Array, input_dim: int, num_heads: int, head_dim: int) -> Array:
    # Reference: https://github.com/huggingface/transformers/blob/15bd3e61f8d3680ca472c9314ad07584d20f7b81/src/transformers/models/llama/convert_llama_weights_to_hf.py#L222
    return rearrange(
        weights,
        "(heads rotors reim) input_channels -> (heads reim rotors) input_channels",
        heads=num_heads,
        rotors=head_dim // 2,
        reim=2,
        input_channels=input_dim,
    )


def permute_qk_params(
    *,
    params: QLoRALinearParams,
    model_dim: int,
    num_heads: int,
    head_dim: int,
    quantization_group_size: int,
    lora_rank: int,
) -> QLoRALinearParams:
    # Read https://github.com/huggingface/transformers/issues/25199 to understand WTF is going on here
    return replace(
        params,
        weights=permute_qk_weights(params.weights, model_dim, num_heads, head_dim),
        scales=permute_qk_weights(params.scales, model_dim // quantization_group_size, num_heads, head_dim),
        lora_up_weights=tuple(permute_qk_weights(w, lora_rank, num_heads, head_dim) for w in params.lora_up_weights),
    )


def load_attention(
    module: Attention,
    weights_dict: dict[str, Array],
    path: ParameterPath,
) -> Attention:
    if not isinstance(module.qkv_projection, QLoRALinear):
        raise TypeError(f"Expected qkv_projection to be QLoRALinear, got {type(module.qkv_projection)}")

    model_dim = module.model_dim
    num_heads = module.num_heads
    num_groups = module.num_groups
    head_dim = module.head_dim
    lora_rank = module.qkv_projection.config.lora_rank

    q_params = get_qlora_linear_params(weights_dict, path / "wq")
    q_params = permute_qk_params(
        params=q_params,
        model_dim=model_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        quantization_group_size=module.qkv_projection.config.group_size,
        lora_rank=lora_rank,
    )

    k_params = get_qlora_linear_params(weights_dict, path / "wk")
    k_params = permute_qk_params(
        params=k_params,
        model_dim=model_dim,
        num_heads=num_groups,
        head_dim=head_dim,
        quantization_group_size=module.qkv_projection.config.group_size,
        lora_rank=lora_rank,
    )

    v_params = get_qlora_linear_params(weights_dict, path / "wv")

    out_params = get_qlora_linear_params(weights_dict, path / "wo")

    qkv_params = merge_linear_params([q_params, k_params, v_params])
    return load_parameters(
        lambda m: (*params_selector(m.qkv_projection), *params_selector(m.out_projection)),  # type: ignore
        module,
        (*qkv_params, *out_params),
    )


def load_decoder_layer(
    module: DecoderLayer,
    weights_dict: dict[str, Array],
    path: ParameterPath,
) -> DecoderLayer:
    if module.post_attention_norm is not None:
        raise ValueError("Post attention normalization is not supported")
    if module.post_mlp_norm is not None:
        raise ValueError("Post MLP normalization is not supported")
    attention_norm = load_rmsnorm(module.pre_attention_norm, weights_dict, path / "attention_norm")
    attention = load_attention(module.attention, weights_dict, path / "attention")
    mlp_norm = load_rmsnorm(module.pre_mlp_norm, weights_dict, path / "ffn_norm")
    mlp = load_mlp(module.mlp, weights_dict, path / "feed_forward")
    return load_parameters(
        lambda m: (m.pre_attention_norm, m.attention, m.pre_mlp_norm, m.mlp),
        module,
        (attention_norm, attention, mlp_norm, mlp),
    )


def load_embedding(
    module: QuantizedTiedEmbedding,
    weights_dict: dict[str, Array],
    path: ParameterPath,
) -> QuantizedTiedEmbedding:
    weights = weights_dict[path / "weight"]
    scales = weights_dict[path / "scales"].squeeze(1)
    return load_parameters(lambda m: (m.weights, m.scales), module, (weights, scales))


def load_executorch(module: Decoder, weights_dict: dict[str, Array]) -> Decoder:
    root_path = ParameterPath()
    if not isinstance(module.embedding, QuantizedTiedEmbedding):
        raise TypeError(f"Expected embedding to be QuantizedTiedEmbedding, got {type(module.embedding)}")

    embedding = load_embedding(module.embedding, weights_dict, root_path / "tok_embeddings")
    decoder_layers = tuple(
        load_decoder_layer(layer, weights_dict, root_path / f"layers.{i}") for i, layer in enumerate(module.layers)
    )
    output_norm = load_rmsnorm(module.output_norm, weights_dict, root_path / "norm")
    return load_parameters(
        lambda m: (m.embedding, m.layers, m.output_norm),
        module,
        (embedding, decoder_layers, output_norm),
    )
