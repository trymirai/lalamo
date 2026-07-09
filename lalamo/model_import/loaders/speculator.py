from pathlib import Path

from huggingface_hub import snapshot_download
from jaxtyping import DTypeLike
from tokenizers import Tokenizer

from lalamo.audio.utils import dummy_char_level_tokenizer_config
from lalamo.model_import.loaders.dflash_loader import load_hf_dflash_draft_model
from lalamo.model_import.loaders.weaver_loader import load_weaver
from lalamo.models import (
    DFlashSpeculatorModel,
    DFlashSpeculatorModelConfig,
    RawTextCodecConfig,
    WeaverSpeculatorModel,
    WeaverSpeculatorModelConfig,
)
from lalamo.utils.sharding import ShardingConfig

__all__ = [
    "load_speculator_model",
]


def load_speculator_model(
    source: Path | str,
    *,
    sharding_config: ShardingConfig,
    dtype: DTypeLike | None = None,
    context_length: int | None = None,
) -> DFlashSpeculatorModel | WeaverSpeculatorModel:
    source_path = Path(source)
    artifact = (
        source_path
        if source_path.exists()
        else Path(snapshot_download(str(source), allow_patterns=["config.json", "*.safetensors"]))
    )
    token_codec_config = RawTextCodecConfig()
    token_codec = token_codec_config.init(Tokenizer.from_str(dummy_char_level_tokenizer_config()))

    if (artifact / "config.json").is_file():
        draft_model = load_hf_dflash_draft_model(
            artifact,
            sharding_config=sharding_config,
            dtype=dtype,
            context_length=context_length,
        )
        return DFlashSpeculatorModel(
            config=DFlashSpeculatorModelConfig(
                token_codec_config=token_codec_config,
                draft_config=draft_model.config,
            ),
            sharding_config=sharding_config,
            token_codec=token_codec,
            draft_model=draft_model,
        )

    weaver = load_weaver(artifact, sharding_config, dtype=dtype)
    return WeaverSpeculatorModel(
        config=WeaverSpeculatorModelConfig(
            token_codec_config=token_codec_config,
            weaver_config=weaver.config,
        ),
        sharding_config=sharding_config,
        token_codec=token_codec,
        weaver=weaver,
    )
