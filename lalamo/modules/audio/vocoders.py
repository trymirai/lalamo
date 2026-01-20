from dataclasses import dataclass
from typing import Self

from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float

from lalamo.common import ParameterTree
from lalamo.modules.common import LalamoModule


@dataclass(frozen=True)
class NoopVocoderConfig:
    def empty(self) -> "NoopVocoder":
        return NoopVocoder(config=self)


class NoopVocoder(LalamoModule[NoopVocoderConfig]):
    # Used in models where vocoder is absent and ther is no additional
    # conversion from audio-features space to audio waveform space

    @property
    def activation_precision(self) -> DTypeLike:
        return jnp.float32

    def export_weights(self) -> ParameterTree[Array]:
        return {}

    def import_weights(
        self,
        weights: ParameterTree[Array],  # noqa: ARG002
    ) -> Self:
        return self

    def __call__(
        self,
        audio_features: Float[Array, "batch audio_channels samples"],
    ) -> Float[Array, "batch audio_channels samples"]:
        return audio_features


VocoderConfig = NoopVocoderConfig
Vocoder = NoopVocoder
