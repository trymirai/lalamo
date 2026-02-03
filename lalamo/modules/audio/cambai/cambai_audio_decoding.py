from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import jax
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_tree
from lalamo.modules.audio.audio_decoder import TTSAudioDecoder

from .nanocodec_modules import (
    CausalHiFiGANDecoder,
    CausalHiFiGANDecoderConfig,
    GroupFiniteScalarQuantizer,
    GroupFiniteScalarQuantizerConfig,
)


@dataclass(frozen=True)
class NanoCodecConfig:
    precision: DTypeLike
    quantizer_config: GroupFiniteScalarQuantizerConfig
    decoder_config: CausalHiFiGANDecoderConfig
    samplerate: int

    def empty(
        self,
    ) -> "NanoCodec":
        quantizer = self.quantizer_config.empty()
        decoder = self.decoder_config.empty()

        return NanoCodec(
            config=self,
            quantizer=quantizer,
            decoder=decoder,
        )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,
    ) -> "NanoCodec":
        key_quantizer, key_decoder = jax.random.split(key)

        quantizer = self.quantizer_config.random_init(key=key_quantizer)
        decoder = self.decoder_config.random_init(key=key_decoder)

        return NanoCodec(
            config=self,
            quantizer=quantizer,
            decoder=decoder,
        )


class NanoCodec(TTSAudioDecoder[NanoCodecConfig]):
    """Lalamo implementation of decoder part of NanoCodec from NVidia
    Original code: https://github.com/NVIDIA-NeMo/NeMo/blob/v2.3.0/nemo/collections/tts/modules/audio_codec_modules.py
    The decoding pipeline:
    1. GroupFiniteScalarQuantizer decodes integer codes to continuous latent representations
    2. CausalHiFiGANDecoder upsamples and converts latents to audio waveform
    """

    quantizer: GroupFiniteScalarQuantizer
    decoder: CausalHiFiGANDecoder

    @property
    def samplerate(self) -> int:
        # return self.config.samplerate
        return 123

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def n_codebooks(self) -> int:
        return -1

    def __call__(
        self,
        indices: Int[Array, "batch n_codebooks tokens"],
    ) -> Float[Array, "batch audio_samples 1"]:
        z = self.quantizer.decode(indices)
        audio = self.decoder(z)

        return audio

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "quantizer": self.quantizer.export_weights(),
            "decoder": self.decoder.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)

        quantizer_weights = weights["quantizer"]
        decoder_weights = weights["decoder"]

        assert isinstance(quantizer_weights, Mapping)
        assert isinstance(decoder_weights, Mapping)

        return replace(
            self,
            quantizer=self.quantizer.import_weights(require_tree(quantizer_weights)),
            decoder=self.decoder.import_weights(require_tree(decoder_weights)),
        )

    def audio_from_codes(self, indices: Array) -> Array:
        if len(indices.shape) == 2:
            indices = indices[None, :]
        return self(indices)[0, :, 0]
