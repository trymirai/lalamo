from dataclasses import dataclass

from tokenizers import Tokenizer

from lalamo.initializer import Initializer
from lalamo.model import Model, ModelConfig
from lalamo.models.raw_text_codec import RawTextCodec, RawTextCodecConfig
from lalamo.modules import DFlashDraftConfig, DFlashDraftModel, Weaver, WeaverConfig

__all__ = [
    "SpeculatorModel",
    "SpeculatorModelConfig",
]


@dataclass(frozen=True)
class SpeculatorModelConfig(ModelConfig[RawTextCodecConfig]):
    draft_config: DFlashDraftConfig
    weaver_config: WeaverConfig | None

    def __post_init__(self) -> None:
        if self.weaver_config is None:
            return
        if self.weaver_config.d_model != self.draft_config.model_dim:
            raise ValueError(
                f"Weaver d_model {self.weaver_config.d_model} does not match"
                f" draft model_dim {self.draft_config.model_dim}.",
            )
        if self.weaver_config.k > self.draft_config.block_size - 1:
            raise ValueError(
                f"Weaver depth k={self.weaver_config.k} exceeds the draft block's"
                f" {self.draft_config.block_size - 1} proposal positions.",
            )

    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "SpeculatorModel":
        return SpeculatorModel(
            config=self,
            sharding_config=initializer.sharding_config,
            token_codec=self.token_codec_config.init(tokenizer),
            draft_model=self.draft_config.init(initializer),
            weaver=self.weaver_config.init(initializer) if self.weaver_config is not None else None,
        )


class SpeculatorModel(Model[RawTextCodecConfig, SpeculatorModelConfig, RawTextCodec]):
    draft_model: DFlashDraftModel
    weaver: Weaver | None
