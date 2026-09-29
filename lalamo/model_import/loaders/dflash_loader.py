from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path

import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.common import _combine_weight_shards
from lalamo.model_import.model_configs.huggingface.dflash import HFDFlashConfig
from lalamo.model_import.origins import LocalOrigin, WeightFormat
from lalamo.modules.speculators.dflash import DFlashDraftModel
from lalamo.modules.transformer_layer import TransformerLayerConv
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.surgery import load_as_at
from lalamo.weight_matrix import CompressionImplementation

from .huggingface import load_linear, load_rmsnorm, load_transformer_layer

__all__ = [
    "load_dflash_draft_model",
    "load_dflash_sublayer_transform",
    "load_hf_dflash_draft_model",
]


def load_dflash_sublayer_transform(
    module: TransformerLayerConv | None,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> TransformerLayerConv | None:
    if module is None:
        return None
    base_kernel = rearrange(
        jnp.flip(weights_dict[path / "base_kernel"], axis=1),
        "sides kernel channels -> sides channels kernel",
        sides=2,
    )
    return load_as_at(
        lambda conv: (conv.pre_conv.weights, conv.post_conv.weights, conv.kernel_projection),
        module,
        (
            base_kernel[0],
            base_kernel[1],
            load_linear(
                module.kernel_projection,
                weights_dict,
                path / "kernel_projection",
                implementation=implementation,
            ),
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
        load_transformer_layer(
            layer,
            weights_dict,
            path / "layers" / layer_index,
            path / "layers" / layer_index,
            "self_attn",
            "mlp",
            "input_layernorm",
            "post_attention_layernorm",
            "up_proj",
            "gate_proj",
            "down_proj",
            permute_conv=False,
            implementation=implementation,
        )
        for layer_index, layer in enumerate(module.layers)
    )
    layers = tuple(
        load_as_at(
            lambda layer: (layer.mixer_conv, layer.mlp_conv),
            layer,
            (
                load_dflash_sublayer_transform(
                    layer.mixer_conv,
                    weights_dict,
                    path / "layers" / layer_index / "attention_conv",
                    implementation=implementation,
                ),
                load_dflash_sublayer_transform(
                    layer.mlp_conv,
                    weights_dict,
                    path / "layers" / layer_index / "mlp_conv",
                    implementation=implementation,
                ),
            ),
        )
        for layer_index, layer in enumerate(layers)
    )
    state_kv_projection = module.state_kv_projection_from_layers(layers)
    output_norm = load_rmsnorm(module.output_norm, weights_dict, path / "norm")

    return load_as_at(
        lambda draft_model: (
            draft_model.context_projection,
            draft_model.context_norm,
            draft_model.state_kv_projection,
            draft_model.layers,
            draft_model.output_norm,
        ),
        module,
        (
            context_projection,
            context_norm,
            state_kv_projection,
            layers,
            output_norm,
        ),
    )


def load_hf_dflash_draft_model(
    hf_model_dir: Path | str,
    *,
    sharding_config: ShardingConfig,
    dtype: DTypeLike | None = None,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> DFlashDraftModel:
    hf_model_dir = Path(hf_model_dir)
    config = HFDFlashConfig.from_json(hf_model_dir / "config.json")
    draft_config = config.to_dflash_draft_config()
    template = draft_config.init(EmptyInitializer(dtype, sharding_config))

    weight_files = tuple(path.name for path in sorted(hf_model_dir.glob(f"*{WeightFormat.SAFETENSORS.value}")))
    if not weight_files:
        raise FileNotFoundError(f"DFlash HF directory does not contain safetensors weights: {hf_model_dir}")

    origin = LocalOrigin(
        root=str(hf_model_dir),
        weight_files=weight_files,
        weight_format=WeightFormat.SAFETENSORS,
    )
    with ExitStack() as stack:
        weight_shards = tuple(stack.enter_context(weight_shard) for weight_shard in origin.get_weights())
        checkpoint = _combine_weight_shards(weight_shards)
        return load_dflash_draft_model(
            template,
            checkpoint.weights,
            implementation=implementation,
        )
