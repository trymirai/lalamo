from abc import abstractmethod
from dataclasses import dataclass

from ..common import ForeignConfig


@dataclass(frozen=True)
class AudioCodecConfig(ForeignConfig):
    @property
    @abstractmethod
    def default_precision(self) -> DTypeLike: ...

    @abstractmethod
    def to_lalamo_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],
    ) -> ConfigT: ...

    def load(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        weights_dict: Mapping[str, Array],
        metadata_dict: Mapping[str, str],
    ) -> LalamoModule[ConfigT]:
        config = self.to_lalamo_config(context_length, activation_precision, accumulation_precision, metadata_dict)
        model = config.empty()
        return self._load_weights(model, weights_dict)
