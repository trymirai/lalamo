from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from jaxtyping import Array, Float, Int, PRNGKeyArray

from lalamo.modules.common import LalamoModule, config_converter
from lalamo.registry_abc import RegistryABC


@dataclass(frozen=True)
class LatentTTSOutputs:
    waveform: Float[Array, "batch audio_samples"]
    waveform_lengths: Int[Array, " batch"]


@dataclass(frozen=True)
class LatentTTSGenerationConfig:
    n_timesteps: int = 8
    temperature: float = 1.0
    length_scale: float = 1.0
    guidance_scale: float = 0.0


class LatentTTSModel[ConfigT](LalamoModule[ConfigT]):
    @property
    @abstractmethod
    def samplerate(self) -> int: ...

    @abstractmethod
    def encode(self, *args: Any, **kwargs: Any) -> Any: ...

    @abstractmethod
    def predict(
        self,
        encoded: Any,
        *,
        key: PRNGKeyArray,
        generation_config: LatentTTSGenerationConfig,
    ) -> Any: ...

    @abstractmethod
    def decode(self, predicted: Any) -> LatentTTSOutputs: ...


class LatentTTSConfig(RegistryABC):
    """Base for latent TTS model configs.

    Subclasses must be frozen dataclasses that implement ``empty()`` and ``samplerate``.
    They are automatically discovered via the ``RegistryABC`` mechanism, allowing
    plugins to register new latent TTS model types.
    """

    @abstractmethod
    def empty(self) -> LatentTTSModel: ...

    @property
    @abstractmethod
    def samplerate(self) -> int: ...


def _unstructure_latent_tts_config(obj: LatentTTSConfig) -> dict:
    return {
        "type": obj.__class__.__name__,
        **config_converter.unstructure(obj, type(obj)),
    }


def _structure_latent_tts_config(data: dict, _type: type) -> LatentTTSConfig:
    new_data = dict(data)
    type_name = new_data.pop("type")
    name_to_type = {t.__name__: t for t in LatentTTSConfig.__descendants__()}
    target_type = name_to_type[type_name]
    return config_converter.structure(new_data, target_type)


config_converter.register_structure_hook(LatentTTSConfig, _structure_latent_tts_config)
config_converter.register_unstructure_hook(LatentTTSConfig, _unstructure_latent_tts_config)
