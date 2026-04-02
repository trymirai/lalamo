from dataclasses import dataclass

from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float

from lalamo.common import ParameterTree
from lalamo.modules.common import Initializer, LalamoModule


@dataclass(frozen=True)
class NoopVocoderConfig:
    def init(self, initializer: Initializer) -> "NoopVocoder":  # noqa: ARG002
        return NoopVocoder(config=self)


class NoopVocoder(LalamoModule[NoopVocoderConfig]):
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
