from dataclasses import dataclass

from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float

from lalamo.modules.common import LalamoModule


@dataclass(frozen=True)
class NoopVocoderConfig:
    def empty(self) -> "NoopVocoder":
        return NoopVocoder(config=self)


class NoopVocoder(LalamoModule[NoopVocoderConfig]):
    # Used in models where vocoder is absent and there is no additional
    # conversion from audio-features space to audio waveform space

    @property
    def activation_precision(self) -> DTypeLike:
        return jnp.float32

    def __call__(
        self,
        audio_features: Float[Array, "batch audio_channels samples"],
    ) -> Float[Array, "batch audio_channels samples"]:
        return audio_features


VocoderConfig = NoopVocoderConfig
Vocoder = NoopVocoder
