from dataclasses import dataclass
from pathlib import Path

from lalamo.initializer import Initializer
from lalamo.model import BaseModel, BaseModelConfig
from lalamo.modules import DFlashDraftConfig, DFlashDraftModel, Weaver, WeaverConfig

__all__ = [
    "SpeculatorModel",
    "SpeculatorModelConfig",
]


@dataclass(frozen=True)
class SpeculatorModelConfig(BaseModelConfig):
    draft_config: DFlashDraftConfig
    weaver_config: WeaverConfig | None

    def init(self, initializer: Initializer) -> "SpeculatorModel":
        return SpeculatorModel(
            config=self,
            sharding_config=initializer.sharding_config,
            draft_model=self.draft_config.init(initializer),
            weaver=self.weaver_config.init(initializer) if self.weaver_config is not None else None,
        )

    def init_from_directory(self, directory: Path, initializer: Initializer) -> "SpeculatorModel":
        del directory
        return self.init(initializer)


class SpeculatorModel(BaseModel[SpeculatorModelConfig]):
    draft_model: DFlashDraftModel
    weaver: Weaver | None
