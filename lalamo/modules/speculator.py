import json
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

from lalamo.initializer import Initializer
from lalamo.model import BaseModel
from lalamo.module import LalamoConfig
from lalamo.utils.registry_abc import RegistryABC

__all__ = [
    "Speculator",
    "SpeculatorConfig",
]


@dataclass(frozen=True)
class SpeculatorConfig(LalamoConfig, RegistryABC):
    @abstractmethod
    def init(self, initializer: Initializer) -> "Speculator": ...


class Speculator[ConfigT: SpeculatorConfig](BaseModel[ConfigT]):
    @staticmethod
    def init_from_directory(directory: Path | str, initializer: Initializer) -> "Speculator":
        with (Path(directory) / "config.json").open() as config_file:
            config = SpeculatorConfig.from_json(json.load(config_file))
        return config.init(initializer)
