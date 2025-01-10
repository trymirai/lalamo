from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from fartsovka.importers.common import WeightsPath, load_parameters
from fartsovka.models.qlora_llama import (
    QLoRAAttention,
    QLoRADecoderLayer,
    QLoRALlama,
    QLoRAMLP,
)
from fartsovka.modules.embedding import QuantizedEmbedding
from fartsovka.modules.linear import QLoRALinear
from fartsovka.modules.normalization import RMSNorm

__all__ = ["load_llama"]


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
        module.quantized_linear.weights,
        module.quantized_linear.scales,
        module.lora_down_weights,
        *module.lora_up_weights,
    )


def get_qlora_linear_params(weights_dict: dict[str, Array], path: WeightsPath) -> QLoRALinearParams:
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


def load_linear(module: QLoRALinear, weights_dict: dict[str, Array], path: WeightsPath) -> QLoRALinear:
    params = get_qlora_linear_params(weights_dict, path)
    return load_parameters(params_selector, module, params)


def load_mlp(module: QLoRAMLP, weights_dict: dict[str, Array], path: WeightsPath) -> QLoRAMLP:
    up_proj_params = get_qlora_linear_params(weights_dict, path / "w3")
    gate_proj_params = get_qlora_linear_params(weights_dict, path / "w1")
    down_proj_params = get_qlora_linear_params(weights_dict, path / "w2")

    fused_up_gate_params = merge_linear_params([up_proj_params, gate_proj_params])

    return load_parameters(
        lambda m: (*params_selector(m.up_projection), *params_selector(m.down_projection)),
        module,
        (*fused_up_gate_params, *down_proj_params),
    )


def load_rmsnorm(module: RMSNorm, weights_dict: dict[str, Array], path: WeightsPath) -> RMSNorm:
    return load_parameters(lambda m: (m.scale,), module, (weights_dict[path / "weight"],))


def load_attention(
    module: QLoRAAttention,
    weights_dict: dict[str, Array],
    path: WeightsPath,
) -> QLoRAAttention:
    q_params = get_qlora_linear_params(weights_dict, path / "wq")
    k_params = get_qlora_linear_params(weights_dict, path / "wk")
    v_params = get_qlora_linear_params(weights_dict, path / "wv")
    out_params = get_qlora_linear_params(weights_dict, path / "wo")
    qkv_params = merge_linear_params([q_params, k_params, v_params])
    return load_parameters(
        lambda m: (*params_selector(m.qkv_projection), *params_selector(m.out_projection)),
        module,
        (*qkv_params, *out_params),
    )


def load_decoder_layer(
    module: QLoRADecoderLayer,
    weights_dict: dict[str, Array],
    path: WeightsPath,
) -> QLoRADecoderLayer:
    attention_norm = load_rmsnorm(module.attention_norm, weights_dict, path / "attention_norm")
    attention = load_attention(module.attention, weights_dict, path / "attention")
    mlp_norm = load_rmsnorm(module.mlp_norm, weights_dict, path / "ffn_norm")
    mlp = load_mlp(module.mlp, weights_dict, path / "feed_forward")
    return load_parameters(
        lambda m: (m.attention_norm, m.attention, m.mlp_norm, m.mlp),
        module,
        (attention_norm, attention, mlp_norm, mlp),
    )


def load_embedding(
    module: QuantizedEmbedding,
    weights_dict: dict[str, Array],
    path: WeightsPath,
) -> QuantizedEmbedding:
    weights = weights_dict[path / "weight"]
    scales = weights_dict[path / "scales"].squeeze(1)
    return load_parameters(lambda m: (m.weights, m.scales), module, (weights, scales))


def load_llama(module: QLoRALlama, weights_dict: dict[str, Array]) -> QLoRALlama:
    root_path = WeightsPath()
    embedding = load_embedding(module.embedding, weights_dict, root_path / "tok_embeddings")
    decoder_layers = [
        load_decoder_layer(layer, weights_dict, root_path / f"layers.{i}") for i, layer in enumerate(module.layers)
    ]
    out_norm = load_rmsnorm(module.out_norm, weights_dict, root_path / "norm")
    return load_parameters(
        lambda m: (m.embedding, m.layers, m.out_norm),
        module,
        (embedding, decoder_layers, out_norm),
    )
