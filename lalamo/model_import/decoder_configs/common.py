import json
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self

import cattrs
from jaxtyping import Array, DTypeLike

from lalamo.modules import Decoder, DecoderConfig

__all__ = ["ForeignConfig"]


@dataclass(frozen=True)
class ForeignConfig:
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()
    _converter.register_structure_hook(int | list[int], lambda v, _: v)

    eos_token_id: int | list[int]

    @property
    def eos_token_ids(self) -> list[int]:
        return [self.eos_token_id] if isinstance(self.eos_token_id, int) else self.eos_token_id

    @property
    @abstractmethod
    def default_precision(self) -> DTypeLike: ...

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        json_path = Path(json_path)
        with open(json_path) as f:
            config = json.load(f)
        return cls._converter.structure(config, cls)

    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
    ) -> DecoderConfig:
        raise NotImplementedError

    @classmethod
    def _load_weights(
        cls,
        model: Decoder,
        weights_dict: Mapping[str, Array],
    ) -> Decoder:
        raise NotImplementedError

    def load_decoder(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        weights_dict: Mapping[str, Array],
    ) -> Decoder:
        config = self.to_decoder_config(context_length, activation_precision, accumulation_precision)
        model = config.empty()
        return self._load_weights(model, weights_dict)
