from abc import abstractmethod
from dataclasses import dataclass

from lalamo.initializer import Initializer
from lalamo.module import LalamoConfig, LalamoModule
from lalamo.utils.registry_abc import RegistryABC

__all__ = [
    "Speculator",
    "SpeculatorConfig",
]


@dataclass(frozen=True)
class SpeculatorConfig(LalamoConfig, RegistryABC):
    @abstractmethod
    def init(self, initializer: Initializer) -> "Speculator": ...


class Speculator[ConfigT: SpeculatorConfig](LalamoModule[ConfigT]):
    pass
