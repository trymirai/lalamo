from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

from lalamo.initializer import Initializer
from lalamo.model import BaseModel, BaseModelConfig

__all__ = [
    "Speculator",
    "SpeculatorConfig",
]


@dataclass(frozen=True)
class SpeculatorConfig(BaseModelConfig):
    @abstractmethod
    def init(self, initializer: Initializer) -> "Speculator": ...

    def init_from_directory(self, directory: Path, initializer: Initializer) -> "Speculator":  # noqa: ARG002
        return self.init(initializer)


class Speculator[ConfigT: SpeculatorConfig](BaseModel[ConfigT]):
    pass
