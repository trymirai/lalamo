from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path

from jaxtyping import Array, DTypeLike

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.common import _combine_weight_shards
from lalamo.model_import.model_configs.huggingface.dflash import HFDFlashConfig
from lalamo.model_import.origins import LocalOrigin, WeightFormat
from lalamo.modules.speculators.dflash import DFlashDraftConfig, DFlashDraftModel
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.surgery import load_as_at
from lalamo.weight_matrix import CompressionImplementation

from .binding import Binder
from .huggingface import huggingface_loader, load_linear, load_rmsnorm

__all__ = [
    "load_dflash_draft_model",
    "load_hf_dflash_draft_model",
]


def _reshape_muse_glimmer_weights(weights_dict: Mapping[str, Array]) -> Mapping[str, Array]:
    path_mapping = {
        ParameterPath("encoder.fc.weight"): ParameterPath("fc.weight"),
        ParameterPath("encoder.output_norm_enc.weight"): ParameterPath("hidden_norm.weight"),
    }
    return {path_mapping.get(ParameterPath(path), path): weights for path, weights in weights_dict.items()}


def load_dflash_draft_model(
    module: DFlashDraftModel,
    config: DFlashDraftConfig,
    binder: Binder,
    weights_dict: Mapping[str, Array],
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> DFlashDraftModel:
    module = load_as_at(
        lambda draft_model: (draft_model.context_projection, draft_model.context_norm, draft_model.program),
        module,
        (
            load_linear(module.context_projection, weights_dict, ParameterPath("fc"), implementation=implementation),
            load_rmsnorm(module.context_norm, weights_dict, ParameterPath("hidden_norm")),
            binder.load(module.program, config.program, weights_dict, implementation=implementation),
        ),
    )
    return load_as_at(
        lambda draft_model: (draft_model.state_kv_projection,),
        module,
        (module.state_kv_projection_from_attentions(),),
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
    binder = Binder(huggingface_loader)
    draft_config = config.build_dflash_draft(binder)
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
            draft_config,
            binder,
            _reshape_muse_glimmer_weights(checkpoint.weights),
            implementation=implementation,
        )
