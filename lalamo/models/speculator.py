from dataclasses import dataclass

from tokenizers import Tokenizer

from lalamo.initializer import Initializer
from lalamo.model import Model, ModelConfig
from lalamo.models.raw_text_codec import RawTextCodec, RawTextCodecConfig
from lalamo.modules import DFlashDraftConfig, DFlashDraftModel, Weaver, WeaverConfig

__all__ = [
    "DFlashSpeculatorModel",
    "DFlashSpeculatorModelConfig",
    "WeaverSpeculatorModel",
    "WeaverSpeculatorModelConfig",
]


@dataclass(frozen=True)
class DFlashSpeculatorModelConfig(ModelConfig[RawTextCodecConfig]):
    draft_config: DFlashDraftConfig

    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "DFlashSpeculatorModel":
        return DFlashSpeculatorModel(
            config=self,
            sharding_config=initializer.sharding_config,
            token_codec=self.token_codec_config.init(tokenizer),
            draft_model=self.draft_config.init(initializer),
        )


class DFlashSpeculatorModel(Model[RawTextCodecConfig, DFlashSpeculatorModelConfig, RawTextCodec]):
    draft_model: DFlashDraftModel


@dataclass(frozen=True)
class WeaverSpeculatorModelConfig(ModelConfig[RawTextCodecConfig]):
    weaver_config: WeaverConfig

    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "WeaverSpeculatorModel":
        return WeaverSpeculatorModel(
            config=self,
            sharding_config=initializer.sharding_config,
            token_codec=self.token_codec_config.init(tokenizer),
            weaver=self.weaver_config.init(initializer),
        )


class WeaverSpeculatorModel(Model[RawTextCodecConfig, WeaverSpeculatorModelConfig, RawTextCodec]):
    weaver: Weaver
