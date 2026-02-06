from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Self

import jax
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_tree
from lalamo.modules.audio.audio_decoder import TTSAudioDecoder, TTSAudioDecoderConfigBase

from .fishaudio_modules import (
    ConvNeXtSpatialParams,
    DACDecoder,
    DACDecoderConfig,
    DACDecoderSpatialParams,
    DownsampleResidualVectorQuantize,
    DownsampleResidualVectorQuantizeConfig,
    TransposeConvSpatialParams,
    VectorQuantizerParams,
)


@dataclass(frozen=True)
class DescriptAudioCodecConfig(TTSAudioDecoderConfigBase):
    precision: DTypeLike
    quantizer_config: DownsampleResidualVectorQuantizeConfig
    decoder_config: DACDecoderConfig
    samplerate: int

    # NOTE: these fields are retireved from DAC audio-codec config which is
    # currently baked into code of fish-speech package as a separate file
    encoder_dim: int
    encoder_rates: tuple[int, ...]
    decoder_dim: int
    decoder_rates: tuple[int, ...]
    input_dim: int
    n_codebooks: int
    codebook_dim: int
    downsample_factor: tuple[int, ...]
    codebook_size: int
    semantic_codebook_size: int

    def empty(
        self,
    ) -> "DescriptAudioCodec":
        (
            semantic_quantizer_params,
            quantizer_params,
            convnext_params,
            upsample_conv_params,
            decoder_spatial_params,
        ) = DescriptAudioCodecConfig.create_spatial_parameter_objects(
            encoder_dim=self.encoder_dim,
            encoder_rates=self.encoder_rates,
            decoder_dim=self.decoder_dim,
            decoder_rates=self.decoder_rates,
            input_dim=self.input_dim,
            n_codebooks=self.n_codebooks,
            codebook_dim=self.codebook_dim,
            downsample_factor=self.downsample_factor,
            codebook_size=self.codebook_size,
            semantic_codebook_size=self.semantic_codebook_size,
        )
        quantizer = self.quantizer_config.empty(
            upsampler_trans_conv_params=upsample_conv_params,
            convnext_spatial_params=convnext_params,
            semantic_quantizer_params=semantic_quantizer_params,
            quantizer_params=quantizer_params,
        )

        decoder = self.decoder_config.empty(spatial_params=decoder_spatial_params)

        return DescriptAudioCodec(
            config=self,
            quantizer=quantizer,
            decoder=decoder,
        )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,
    ) -> "DescriptAudioCodec":
        key1, key2 = jax.random.split(key)

        (
            semantic_quantizer_params,
            quantizer_params,
            convnext_params,
            upsample_conv_params,
            decoder_spatial_params,
        ) = DescriptAudioCodecConfig.create_spatial_parameter_objects(
            encoder_dim=self.encoder_dim,
            encoder_rates=self.encoder_rates,
            decoder_dim=self.decoder_dim,
            decoder_rates=self.decoder_rates,
            input_dim=self.input_dim,
            n_codebooks=self.n_codebooks,
            codebook_dim=self.codebook_dim,
            downsample_factor=self.downsample_factor,
            codebook_size=self.codebook_size,
            semantic_codebook_size=self.semantic_codebook_size,
        )

        quantizer = self.quantizer_config.random_init(
            upsampler_trans_conv_params=upsample_conv_params,
            convnext_spatial_params=convnext_params,
            semantic_quantizer_params=semantic_quantizer_params,
            quantizer_params=quantizer_params,
            key=key1,
        )

        decoder = self.decoder_config.random_init(spatial_params=decoder_spatial_params, key=key2)

        return DescriptAudioCodec(
            config=self,
            quantizer=quantizer,
            decoder=decoder,
        )

    @staticmethod
    def create_spatial_parameter_objects(
        encoder_dim: int,
        encoder_rates: Sequence[int],
        decoder_dim: int,
        decoder_rates: Sequence[int],
        input_dim: int,
        n_codebooks: int,
        codebook_dim: int,
        downsample_factor: Sequence[int],
        codebook_size: int,
        semantic_codebook_size: int,
    ) -> tuple[
        VectorQuantizerParams,
        VectorQuantizerParams,
        ConvNeXtSpatialParams,
        tuple[TransposeConvSpatialParams, ...],
        DACDecoderSpatialParams,
    ]:
        latent_dim = encoder_dim * (2 ** len(encoder_rates))
        downsample_dims = [input_dim for _ in range(len(downsample_factor))]
        all_dims = (input_dim, *downsample_dims)

        semantic_quantizer_params = VectorQuantizerParams(
            input_dim=input_dim,
            codebook_size=semantic_codebook_size,
            codebook_dim=codebook_dim,
        )
        quantizer_params = VectorQuantizerParams(
            input_dim=input_dim,
            codebook_size=codebook_size,
            codebook_dim=[codebook_dim] * n_codebooks,
        )
        convnext_params = ConvNeXtSpatialParams()
        upsample_conv_params = tuple(
            [
                TransposeConvSpatialParams(
                    in_channels=all_dims[idx + 1],
                    out_channels=all_dims[idx],
                    upsample_kernel_size=factor,
                    upsample_stride=factor,
                )
                for idx, factor in reversed(list(enumerate(downsample_factor)))
            ],
        )
        decoder_spatial_params = DACDecoderSpatialParams(
            input_channel=latent_dim,
            channels=decoder_dim,
            rates=tuple(decoder_rates),
            d_out=1,
        )

        return (
            semantic_quantizer_params,
            quantizer_params,
            convnext_params,
            upsample_conv_params,
            decoder_spatial_params,
        )


class DescriptAudioCodec(TTSAudioDecoder[DescriptAudioCodecConfig]):
    """Lalamo implementation of DAC (Descript Audio Codec)
    Original code: https://github.com/descriptinc/descript-audio-codec
    The decoding pipeline:
    1. Quantizer decodes integer codes to continuous latent representations
    2. Decoder upsamples and converts latents to audio waveform
    """

    quantizer: DownsampleResidualVectorQuantize
    decoder: DACDecoder

    @property
    def samplerate(self) -> int:
        return self.config.samplerate

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def semantic_codebook_size(self) -> int:
        return self.quantizer.semantic_codebook_size

    @property
    def quantizer_codebook_size(self) -> int:
        return self.quantizer.quantizer_codebook_size

    @property
    def n_codebooks(self) -> int:
        # NOTE: 1 semantic + n residuals
        return 1 + self.quantizer.quantizer.n_codebooks

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
