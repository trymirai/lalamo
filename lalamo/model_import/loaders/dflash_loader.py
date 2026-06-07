from collections.abc import Mapping

from jaxtyping import Array

from lalamo.modules.dflash import DFlashAttention, DFlashDraftLayer, DFlashDraftModel
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.surgery import load_as_at
from lalamo.weight_matrix import CompressionImplementation

from .huggingface import load_linear, load_mlp, load_rmsnorm

__all__ = [
    "load_dflash_attention",
    "load_dflash_draft_layer",
    "load_dflash_draft_model",
]


def load_dflash_attention(
    module: DFlashAttention,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> DFlashAttention:
    query_projection = load_linear(
        module.query_projection,
        weights_dict,
        path / "q_proj",
        implementation=implementation,
    )
    key_value_projection = load_linear(
        module.key_value_projection,
        weights_dict,
        path,
        sublayers_to_fuse=["k_proj", "v_proj"],
        implementation=implementation,
    )
    output_projection = load_linear(
        module.output_projection,
        weights_dict,
        path / "o_proj",
        implementation=implementation,
    )
    query_norm = load_rmsnorm(module.query_norm, weights_dict, path / "q_norm")
    key_norm = load_rmsnorm(module.key_norm, weights_dict, path / "k_norm")

    return load_as_at(
        lambda attention: (
            attention.query_projection,
            attention.key_value_projection,
            attention.output_projection,
            attention.query_norm,
            attention.key_norm,
        ),
        module,
        (
            query_projection,
            key_value_projection,
            output_projection,
            query_norm,
            key_norm,
        ),
    )


def load_dflash_draft_layer(
    module: DFlashDraftLayer,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> DFlashDraftLayer:
    attention = load_dflash_attention(
        module.attention,
        weights_dict,
        path / "self_attn",
        implementation=implementation,
    )
    input_norm = load_rmsnorm(module.input_norm, weights_dict, path / "input_layernorm")
    post_attention_norm = load_rmsnorm(
        module.post_attention_norm,
        weights_dict,
        path / "post_attention_layernorm",
    )
    mlp = load_mlp(
        module.mlp,
        weights_dict,
        path / "mlp",
        "up_proj",
        "gate_proj",
        "down_proj",
        implementation=implementation,
    )

    return load_as_at(
        lambda layer: (
            layer.attention,
            layer.input_norm,
            layer.post_attention_norm,
            layer.mlp,
        ),
        module,
        (
            attention,
            input_norm,
            post_attention_norm,
            mlp,
        ),
    )


def load_dflash_draft_model(
    module: DFlashDraftModel,
    weights_dict: Mapping[str, Array],
    path: ParameterPath = ParameterPath(),
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> DFlashDraftModel:
    context_projection = load_linear(
        module.context_projection,
        weights_dict,
        path / "fc",
        implementation=implementation,
    )
    context_norm = load_rmsnorm(module.context_norm, weights_dict, path / "hidden_norm")
    layers = tuple(
        load_dflash_draft_layer(
            layer,
            weights_dict,
            path / "layers" / layer_index,
            implementation=implementation,
        )
        for layer_index, layer in enumerate(module.layers)
    )
    output_norm = load_rmsnorm(module.output_norm, weights_dict, path / "norm")

    return load_as_at(
        lambda draft_model: (
            draft_model.context_projection,
            draft_model.context_norm,
            draft_model.layers,
            draft_model.output_norm,
        ),
        module,
        (
            context_projection,
            context_norm,
            layers,
            output_norm,
        ),
    )
