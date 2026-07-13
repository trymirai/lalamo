from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path

from jaxtyping import Array, DTypeLike

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.common import _combine_weight_shards
from lalamo.model_import.model_configs.huggingface.dflash import HFDFlashConfig
from lalamo.model_import.origins import LocalOrigin, WeightFormat
from lalamo.modules.speculators.dflash import DFlashDraftLayer, DFlashDraftModel
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.surgery import load_as_at
from lalamo.weight_matrix import CompressionImplementation

from .huggingface import load_attention, load_linear, load_mlp, load_rmsnorm

__all__ = [
    "load_dflash_draft_layer",
    "load_dflash_draft_model",
    "load_hf_dflash_draft_model",
]


def load_dflash_draft_layer(
    module: DFlashDraftLayer,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> DFlashDraftLayer:
    attention = load_attention(
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


def load_hf_dflash_draft_model(
    hf_model_dir: Path | str,
    *,
    sharding_config: ShardingConfig,
    dtype: DTypeLike | None = None,
    context_length: int | None = None,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> DFlashDraftModel:
    hf_model_dir = Path(hf_model_dir)
    config = HFDFlashConfig.from_json(hf_model_dir / "config.json")
    draft_config = config.to_dflash_draft_config(context_length=context_length)
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
