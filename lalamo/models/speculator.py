from dataclasses import dataclass
from pathlib import Path

from lalamo.initializer import Initializer
from lalamo.model import BaseModel, BaseModelConfig
from lalamo.modules import Speculator, SpeculatorConfig

__all__ = [
    "SpeculatorModel",
    "SpeculatorModelConfig",
]


@dataclass(frozen=True)
class SpeculatorModelConfig(BaseModelConfig):
    speculator_config: SpeculatorConfig

    def init(self, initializer: Initializer) -> "SpeculatorModel":
        return SpeculatorModel(
            config=self,
            sharding_config=initializer.sharding_config,
            speculator=self.speculator_config.init(initializer),
        )

    def init_from_directory(self, directory: Path, initializer: Initializer) -> "SpeculatorModel":
        del directory
        return self.init(initializer)


class SpeculatorModel(BaseModel[SpeculatorModelConfig]):
    speculator: Speculator
