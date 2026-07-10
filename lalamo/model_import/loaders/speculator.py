from pathlib import Path

from huggingface_hub import snapshot_download
from jaxtyping import DTypeLike
from tokenizers import Tokenizer

from lalamo.audio.utils import dummy_char_level_tokenizer_config
from lalamo.model_import.loaders.dflash_loader import load_hf_dflash_draft_model
from lalamo.model_import.loaders.weaver_loader import load_weaver
from lalamo.models import RawTextCodecConfig, SpeculatorModel, SpeculatorModelConfig
from lalamo.utils.sharding import ShardingConfig

__all__ = [
    "load_speculator_model",
]


def load_speculator_model(
    dflash_source: Path | str,
    weaver_source: Path | str | None = None,
    *,
    sharding_config: ShardingConfig,
    dtype: DTypeLike | None = None,
    context_length: int | None = None,
) -> SpeculatorModel:
    dflash_path = Path(dflash_source)
    if not dflash_path.exists():
        dflash_path = Path(snapshot_download(str(dflash_source), allow_patterns=["config.json", "*.safetensors"]))
    draft_model = load_hf_dflash_draft_model(
        dflash_path,
        sharding_config=sharding_config,
        dtype=dtype,
        context_length=context_length,
    )
    weaver = load_weaver(weaver_source, sharding_config, dtype=dtype) if weaver_source is not None else None

    token_codec_config = RawTextCodecConfig()
    return SpeculatorModel(
        config=SpeculatorModelConfig(
            token_codec_config=token_codec_config,
            draft_config=draft_model.config,
            weaver_config=weaver.config if weaver is not None else None,
        ),
        sharding_config=sharding_config,
        token_codec=token_codec_config.init(Tokenizer.from_str(dummy_char_level_tokenizer_config())),
        draft_model=draft_model,
        weaver=weaver,
    )
