from pathlib import Path

from jaxtyping import DTypeLike

from lalamo.model_import.loaders.dflash_loader import load_hf_dflash_draft_model
from lalamo.model_import.loaders.weaver_loader import load_weaver
from lalamo.models import SpeculatorModel, SpeculatorModelConfig
from lalamo.modules import DFlashSpeculator, DFlashSpeculatorConfig
from lalamo.utils.sharding import ShardingConfig

__all__ = [
    "load_speculator_model",
]


def load_speculator_model(
    dflash_path: Path | str,
    weaver_path: Path | str | None = None,
    *,
    sharding_config: ShardingConfig,
    dtype: DTypeLike | None = None,
    context_length: int | None = None,
) -> SpeculatorModel:
    draft_model = load_hf_dflash_draft_model(
        dflash_path,
        sharding_config=sharding_config,
        dtype=dtype,
        context_length=context_length,
    )
    weaver = load_weaver(weaver_path, sharding_config, dtype=dtype) if weaver_path is not None else None

    speculator = DFlashSpeculator(
        config=DFlashSpeculatorConfig(
            draft_config=draft_model.config,
            weaver_config=weaver.config if weaver is not None else None,
        ),
        sharding_config=sharding_config,
        draft_model=draft_model,
        weaver=weaver,
    )
    return SpeculatorModel(
        config=SpeculatorModelConfig(speculator_config=speculator.config),
        sharding_config=sharding_config,
        speculator=speculator,
    )
