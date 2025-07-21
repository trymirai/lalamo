import json
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self

import cattrs
import jax
from jaxtyping import Array, DTypeLike

from lalamo.modules import Decoder, DecoderConfig

__all__ = ["ForeignConfig"]


@dataclass(frozen=True)
class ForeignConfig:
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()
    _converter.register_structure_hook(int | list[int], lambda v, _: v)

    @property
    @abstractmethod
    def default_precision(self) -> DTypeLike: ...

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
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
    ) -> DecoderConfig:
        raise NotImplementedError

    @classmethod
    def _load_weights(
        cls,
        model: Decoder,
        weights_dict: dict[str, Array],
    ) -> Decoder:
        raise NotImplementedError

    def load_model(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        weights_dict: dict[str, Array],
    ) -> Decoder:
        config = self.to_decoder_config(context_length, activation_precision, accumulation_precision)
        model = config.random_init(key=jax.random.PRNGKey(0))
        return self._load_weights(model, weights_dict)
