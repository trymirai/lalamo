from collections.abc import Mapping

from einops import rearrange
from jaxtyping import Array

from lalamo.modules.linear import Linear
from lalamo.modules.mlp import UngatedMLP
from lalamo.modules.normalization import Normalization
from lalamo.modules.token_mixers.attention import Attention
from lalamo.modules.vision import (
    VisionEncoder,
    VisionLayer,
)
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.surgery import load_as_at

from .huggingface import load_linear

__all__ = ["load_qwen3_5_vision_encoder"]


def _load_layer_norm(module: Normalization, weights_dict: Mapping[str, Array], path: ParameterPath) -> Normalization:
    return load_as_at(
        lambda m: (m.scales, m.biases),
        module,
        (weights_dict[path / "weight"], weights_dict[path / "bias"]),
    )


def _load_mlp(
    module: UngatedMLP,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> UngatedMLP:
    up_projection = load_linear(module.up_projection, weights_dict, path / "linear_fc1")
    down_projection = load_linear(
        module.down_projection,
        weights_dict,
        path / "linear_fc2",
    )
    return load_as_at(lambda m: (m.up_projection, m.down_projection), module, (up_projection, down_projection))


def _load_attention(
    module: Attention,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> Attention:
    # HF's fused qkv rows are already ordered [q heads, k heads, v heads], matching our output_dims split.
    qkvg_projection = load_linear(module.qkvg_projection, weights_dict, path / "qkv")
    out_projection = load_linear(module.out_projection, weights_dict, path / "proj")
    return load_as_at(lambda m: (m.qkvg_projection, m.out_projection), module, (qkvg_projection, out_projection))


def _load_layer(
    module: VisionLayer,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> VisionLayer:
    return load_as_at(
        lambda m: (m.pre_attention_norm, m.attention, m.pre_mlp_norm, m.mlp),
        module,
        (
            _load_layer_norm(module.pre_attention_norm, weights_dict, path / "norm1"),
            _load_attention(module.attention, weights_dict, path / "attn"),
            _load_layer_norm(module.pre_mlp_norm, weights_dict, path / "norm2"),
            _load_mlp(module.mlp, weights_dict, path / "mlp"),
        ),
    )


def _load_patch_embedding(
    module: Linear,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> Linear:
    # MLX stores Conv3d kernels as (out, t, h, w, c); patches use HF's (c, t, h, w) order.
    weights = weights_dict[path / "weight"]
    if path.startswith("vision_tower."):
        weights = rearrange(weights, "out t h w c -> out c t h w")
    flattened_weights: dict[str, Array] = {
        path / "weight": rearrange(weights, "out c t h w -> out (c t h w)"),
        path / "bias": weights_dict[path / "bias"],
    }
    return load_linear(module, flattened_weights, path)


def load_qwen3_5_vision_encoder(
    module: VisionEncoder,
    weights_dict: Mapping[str, Array],
) -> VisionEncoder:
    path = ParameterPath("model.visual")
    if "vision_tower.patch_embed.proj.weight" in weights_dict:
        path = ParameterPath("vision_tower")
    return load_as_at(
        lambda m: (m.patch_embedding, m.position_embedding.table, m.layers, m.merger_norm, m.merger_mlp),
        module,
        (
            _load_patch_embedding(module.patch_embedding, weights_dict, path / "patch_embed.proj"),
            weights_dict[path / "pos_embed.weight"],
            tuple(
                _load_layer(layer, weights_dict, path / "blocks" / index) for index, layer in enumerate(module.layers)
            ),
            _load_layer_norm(module.merger_norm, weights_dict, path / "merger.norm"),
            _load_mlp(module.merger_mlp, weights_dict, path / "merger"),
        ),
    )
