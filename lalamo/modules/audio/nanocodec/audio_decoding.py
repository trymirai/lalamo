from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import jax
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_tree
from lalamo.modules.audio.audio_decoder import TTSAudioDecoder, TTSAudioDecoderConfigBase

from .nanocodec_modules import (
    CausalHiFiGANDecoder,
    CausalHiFiGANDecoderConfig,
    GroupFiniteScalarQuantizer,
    GroupFiniteScalarQuantizerConfig,
)


@dataclass(frozen=True)
class NanoCodecConfig(TTSAudioDecoderConfigBase):
    precision: DTypeLike
    quantizer_config: GroupFiniteScalarQuantizerConfig
    decoder_config: CausalHiFiGANDecoderConfig
    samplerate: int
    base_channels: int
    up_sample_rates: tuple[int, ...]
    in_kernel_size: int
    out_kernel_size: int
    resblock_kernel_sizes: tuple[int, ...]
    resblock_dilations: tuple[int, ...]

    def empty(self) -> "NanoCodec":
        """Create NanoCodec with placeholder weights."""
        quantizer = self.quantizer_config.empty()
        decoder = self.decoder_config.empty(
            input_dim=self.quantizer_config.codebook_dim,
            base_channels=self.base_channels,
            up_sample_rates=self.up_sample_rates,
            in_kernel_size=self.in_kernel_size,
            out_kernel_size=self.out_kernel_size,
            resblock_kernel_sizes=self.resblock_kernel_sizes,
            resblock_dilations=self.resblock_dilations,
        )

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
        """Create NanoCodec with randomly initialized weights."""
        key_quantizer, key_decoder = jax.random.split(key)

        quantizer = self.quantizer_config.random_init(key=key_quantizer)
        decoder = self.decoder_config.random_init(
            input_dim=self.quantizer_config.codebook_dim,
            base_channels=self.base_channels,
            up_sample_rates=self.up_sample_rates,
            in_kernel_size=self.in_kernel_size,
            out_kernel_size=self.out_kernel_size,
            resblock_kernel_sizes=self.resblock_kernel_sizes,
            resblock_dilations=self.resblock_dilations,
            key=key_decoder,
        )

        return NanoCodec(
            config=self,
            quantizer=quantizer,
            decoder=decoder,
        )


class NanoCodec(TTSAudioDecoder[NanoCodecConfig]):
    """Lalamo implementation of decoder part of NanoCodec from NVidia.

    Original code: https://github.com/NVIDIA-NeMo/NeMo/blob/v2.3.0/nemo/collections/tts/modules/audio_codec_modules.py

    The decoding pipeline:
    1. GroupFiniteScalarQuantizer decodes integer codes to continuous latent representations
    2. CausalHiFiGANDecoder upsamples and converts latents to audio waveform
    """

    quantizer: GroupFiniteScalarQuantizer
    decoder: CausalHiFiGANDecoder

    @property
    def samplerate(self) -> int:
        return self.config.samplerate

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def n_codebooks(self) -> int:
        return self.quantizer.num_groups

    def __call__(
        self,
        indices: Int[Array, "batch n_codebooks tokens"],
    ) -> Float[Array, "batch audio_samples"]:
        """Decode discrete tokens to audio waveform."""
        # Transpose from [B, C, T] to [B, T, C] for Lalamo quantizer (NSC format)
        indices_nsc = indices.transpose((0, 2, 1))

        # Decode indices to continuous representation [B, T, D]
        z = self.quantizer.decode(indices_nsc)

        # Decode to audio [B, T_audio]
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
        """Convenience method to decode a single sequence of codes to audio.

        Args:
            indices: Token indices of shape [num_codebooks, tokens] or [batch, num_codebooks, tokens].

        Returns:
            Audio waveform of shape [audio_samples].
        """
        if len(indices.shape) == 2:
            indices = indices[None, :, :]
        return self(indices)[0, :]
