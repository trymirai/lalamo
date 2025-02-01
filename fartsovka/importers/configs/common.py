import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self

import cattrs

from fartsovka.common import DType
from fartsovka.modules import (
    DecoderConfig,
)

__all__ = ["ForeignConfig"]


@dataclass
class ForeignConfig:
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()
    _converter.register_structure_hook(int | list[int], lambda v, _: v)

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        json_path = Path(json_path)
        with open(json_path) as f:
            config = json.load(f)
        return cls._converter.structure(config, cls)

    def to_json(self, json_path: Path | str) -> None:
        json_path = Path(json_path)
        with open(json_path, "w") as f:
            json.dump(self._converter.unstructure(self), f, indent=2)

    def to_decoder_config(
        self,
        context_length: int,
        activation_precision: DType,
        accumulation_precision: DType,
    ) -> DecoderConfig:
        raise NotImplementedError
