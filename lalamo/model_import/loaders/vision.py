from collections.abc import Mapping

from einops import rearrange
from jaxtyping import Array

from lalamo.modules.linear import Linear
from lalamo.modules.normalization import Normalization
from lalamo.modules.vision import PatchEmbedding, PatchMerger, VisionAttention, VisionEncoder, VisionLayer, VisionMLP
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.surgery import load_as_at
from lalamo.weight_matrix import CompressionImplementation

from .huggingface import load_linear

__all__ = ["load_qwen3_5_vision_encoder"]


def _load_layer_norm(module: Normalization, weights_dict: Mapping[str, Array], path: ParameterPath) -> Normalization:
    return load_as_at(
        lambda m: (m.scales, m.biases),
        module,
        (weights_dict[path / "weight"], weights_dict[path / "bias"]),
    )


def _load_mlp(
    module: VisionMLP,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    implementation: CompressionImplementation,
) -> VisionMLP:
    up_projection = load_linear(module.up_projection, weights_dict, path / "linear_fc1", implementation=implementation)
    down_projection = load_linear(
        module.down_projection,
        weights_dict,
        path / "linear_fc2",
        implementation=implementation,
    )
    return load_as_at(lambda m: (m.up_projection, m.down_projection), module, (up_projection, down_projection))


def _load_attention(
    module: VisionAttention,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    implementation: CompressionImplementation,
) -> VisionAttention:
    # HF's fused qkv rows are already ordered [q heads, k heads, v heads], matching our output_dims split.
    qkv_projection = load_linear(module.qkv_projection, weights_dict, path / "qkv", implementation=implementation)
    out_projection = load_linear(module.out_projection, weights_dict, path / "proj", implementation=implementation)
    return load_as_at(lambda m: (m.qkv_projection, m.out_projection), module, (qkv_projection, out_projection))


def _load_layer(
    module: VisionLayer,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    implementation: CompressionImplementation,
) -> VisionLayer:
    return load_as_at(
        lambda m: (m.pre_attention_norm, m.attention, m.pre_mlp_norm, m.mlp),
        module,
        (
            _load_layer_norm(module.pre_attention_norm, weights_dict, path / "norm1"),
            _load_attention(module.attention, weights_dict, path / "attn", implementation),
            _load_layer_norm(module.pre_mlp_norm, weights_dict, path / "norm2"),
            _load_mlp(module.mlp, weights_dict, path / "mlp", implementation),
        ),
    )


def _load_patch_embedding(
    module: PatchEmbedding,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    implementation: CompressionImplementation,
) -> PatchEmbedding:
    # The Conv3d kernel (out, c, t, h, w) flattens in the same (c, t, h, w) order as the preprocessor's patches.
    flattened_weights: dict[str, Array] = {
        path / "weight": rearrange(weights_dict[path / "weight"], "out c t h w -> out (c t h w)"),
        path / "bias": weights_dict[path / "bias"],
    }
    projection: Linear = load_linear(module.projection, flattened_weights, path, implementation=implementation)
    return load_as_at(lambda m: (m.projection,), module, (projection,))


def _load_merger(
    module: PatchMerger,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    implementation: CompressionImplementation,
) -> PatchMerger:
    return load_as_at(
        lambda m: (m.norm, m.mlp),
        module,
        (
            _load_layer_norm(module.norm, weights_dict, path / "norm"),
            _load_mlp(module.mlp, weights_dict, path, implementation),
        ),
    )


def load_qwen3_5_vision_encoder(
    module: VisionEncoder,
    weights_dict: Mapping[str, Array],
    path: ParameterPath = ParameterPath("model.visual"),
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> VisionEncoder:
    return load_as_at(
        lambda m: (m.patch_embedding, m.position_embedding.table, m.layers, m.merger),
        module,
        (
            _load_patch_embedding(module.patch_embedding, weights_dict, path / "patch_embed.proj", implementation),
            weights_dict[path / "pos_embed.weight"],
            tuple(
                _load_layer(layer, weights_dict, path / "blocks" / index, implementation)
                for index, layer in enumerate(module.layers)
            ),
            _load_merger(module.merger, weights_dict, path / "merger", implementation),
        ),
    )
