from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path

import equinox as eqx
from jaxtyping import Array, DTypeLike

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.common import _combine_weight_shards
from lalamo.model_import.model_configs.huggingface.dflash import HFDFlashConfig
from lalamo.model_import.origins import LocalOrigin, WeightFormat
from lalamo.modules.speculators.dflash import CandidateSelector, DFlashDraftModel, DFlashGroupedConvolution
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.surgery import load_as_at
from lalamo.weight_matrix import CompressionImplementation

from .huggingface import load_linear, load_rmsnorm, load_transformer_layer

__all__ = [
    "load_candidate_selector",
    "load_dflash_draft_model",
    "load_dflash_grouped_convolution",
    "load_hf_dflash_draft_model",
]


def load_dflash_grouped_convolution(
    module: DFlashGroupedConvolution,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> DFlashGroupedConvolution:
    return load_as_at(
        lambda convolution: (convolution.base_kernel, convolution.kernel_projection),
        module,
        (
            weights_dict[path / "base_kernel"],
            load_linear(
                module.kernel_projection,
                weights_dict,
                path / "kernel_projection",
                implementation=implementation,
            ),
        ),
    )


def load_candidate_selector(
    module: CandidateSelector,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> CandidateSelector:
    return load_as_at(
        lambda selector: (
            selector.predecessor_codebook,
            selector.successor_codebook,
            selector.hidden_projection,
        ),
        module,
        (
            weights_dict[path / "predecessor_codebook"],
            weights_dict[path / "successor_codebook"],
            load_linear(
                module.hidden_projection,
                weights_dict,
                path / "hidden_projection",
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
    layer_grouped_convolutions = (
        tuple(
            eqx.tree_at(
                lambda convolutions: (convolutions.attention, convolutions.mlp),
                convolutions,
                (
                    load_dflash_grouped_convolution(
                        convolutions.attention,
                        weights_dict,
                        path / "layers" / layer_index / "attention_conv",
                        implementation=implementation,
                    ),
                    load_dflash_grouped_convolution(
                        convolutions.mlp,
                        weights_dict,
                        path / "layers" / layer_index / "mlp_conv",
                        implementation=implementation,
                    ),
                ),
            )
            for layer_index, convolutions in enumerate(module.layer_grouped_convolutions)
        )
        if module.layer_grouped_convolutions is not None
        else None
    )
    candidate_selector = (
        load_candidate_selector(
            module.candidate_selector,
            weights_dict,
            path / "candidate_selector",
            implementation=implementation,
        )
        if module.candidate_selector is not None
        else None
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
            draft_model.layer_grouped_convolutions,
            draft_model.candidate_selector,
        ),
        module,
        (
            context_projection,
            context_norm,
            state_kv_projection,
            layers,
            output_norm,
            layer_grouped_convolutions,
            candidate_selector,
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
