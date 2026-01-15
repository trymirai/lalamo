from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Self

import jax
from hydra.utils import instantiate
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree
from lalamo.modules import TTSAudioDecoder

from .fishaudio_modules import (
    ConfigMapping,
    ConvNeXtSpatialParams,
    DACDecoder,
    DACDecoderConfig,
    DACDecoderSpatialParams,
    DownsampleResidualVectorQuantize,
    DownsampleResidualVectorQuantizeConfig,
    ResidualVectorQuantizeConfig,
    TransposeConvSpatialParams,
    UpsamplerConfig,
    UpsamplingBlockConfig,
    VectorQuantizerParams,
)


@dataclass(frozen=True)
class DescriptAudioCodecConfig:
    """Configuration for DAC (Discrete Audio Codec) module.

    DAC combines the quantizer (DownsampleResidualVectorQuantize) and decoder (DACDecoder)
    into a single module that decodes audio codes to waveform.

    The decoding pipeline:
    1. Quantizer decodes integer codes to continuous latent representations
    2. Decoder upsamples and converts latents to audio waveform
    """

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
        """Create module with uninitialized (dummy) weights.

        Args:
            quantizer_upsampler_trans_conv_params: TransposeConv params for quantizer upsampler blocks.
            quantizer_convnext_spatial_params: ConvNeXt params for quantizer upsampler.
            semantic_quantizer_params: Parameters for semantic quantizer.
            quantizer_params: Parameters for residual quantizer.
            decoder_spatial_params: Spatial parameters for the audio decoder.

        Returns:
            DAC module with dummy weights.
        """

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
        """Create module with randomly initialized weights.

        Args:
            quantizer_upsampler_trans_conv_params: TransposeConv params for quantizer upsampler blocks.
            quantizer_convnext_spatial_params: ConvNeXt params for quantizer upsampler.
            semantic_quantizer_params: Parameters for semantic quantizer.
            quantizer_params: Parameters for residual quantizer.
            decoder_spatial_params: Spatial parameters for the audio decoder.
            key: PRNG key for random initialization.

        Returns:
            DAC module with random weights.
        """
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
    def instantiate_config_from_fishaudio_config(
        fish_dac_config: Mapping[Any, Any],
    ) -> "DescriptAudioCodecConfig":
        precision = jnp.float32

        # Extract config parameters
        samplerate = fish_dac_config["sample_rate"]

        fish_quantizer_config = fish_dac_config["quantizer"]

        # Build quantizer config using existing helper
        input_dim = fish_quantizer_config["input_dim"]
        downsample_factor = fish_quantizer_config["downsample_factor"]
        post_module_config_dict = fish_quantizer_config["post_module"]

        encoder_dim = fish_dac_config["encoder_dim"]
        encoder_rates = fish_dac_config["encoder_rates"]
        decoder_dim = fish_dac_config["decoder_dim"]
        decoder_rates = fish_dac_config["decoder_rates"]
        fish_quantizer_config = fish_dac_config["quantizer"]
        input_dim = fish_quantizer_config["input_dim"]
        n_codebooks = fish_quantizer_config["n_codebooks"]
        codebook_dim = fish_quantizer_config["codebook_dim"]
        downsample_factor = fish_quantizer_config["downsample_factor"]
        codebook_size = fish_quantizer_config["codebook_size"]
        semantic_codebook_size = fish_quantizer_config["semantic_codebook_size"]

        # === Create full DAC config ===
        upsampler_config = UpsamplerConfig(
            block_configs=tuple([UpsamplingBlockConfig(precision)] * len(downsample_factor))
        )
        post_module_transformer_foreign = instantiate(post_module_config_dict["config"])
        post_module_config = ConfigMapping.lalamo_transformer_cfg_from_fish_audio_codec_cfg(
            post_module_transformer_foreign,
            precision,
            window_size=post_module_config_dict["window_size"],
            input_dim=post_module_config_dict["input_dim"],
        )
        semantic_quantizer_config = ResidualVectorQuantizeConfig(precision)
        residual_quantizer_config = ResidualVectorQuantizeConfig(precision)

        quantizer_full_config = DownsampleResidualVectorQuantizeConfig(
            precision=precision,
            semantic_quantizer_config=semantic_quantizer_config,
            quantizer_config=residual_quantizer_config,
            post_module_config=post_module_config,
            upsampler_config=upsampler_config,
        )
        decoder_config = DACDecoderConfig(precision=precision, causal=True)
        return DescriptAudioCodecConfig(
            precision=precision,
            quantizer_config=quantizer_full_config,
            decoder_config=decoder_config,
            samplerate=samplerate,
            encoder_dim=encoder_dim,
            encoder_rates=encoder_rates,
            decoder_dim=decoder_dim,
            decoder_rates=decoder_rates,
            input_dim=input_dim,
            n_codebooks=n_codebooks,
            codebook_dim=codebook_dim,
            downsample_factor=downsample_factor,
            codebook_size=codebook_size,
            semantic_codebook_size=semantic_codebook_size,
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
        latent_dim = encoder_dim * (2 ** len(encoder_rates))  # 64 * 16 = 1024
        downsample_dims = [input_dim for _ in range(len(downsample_factor))]
        all_dims = (input_dim, *downsample_dims)

        semantic_quantizer_params = VectorQuantizerParams(
            input_dim=input_dim, codebook_size=semantic_codebook_size, codebook_dim=codebook_dim
        )
        quantizer_params = VectorQuantizerParams(
            input_dim=input_dim, codebook_size=codebook_size, codebook_dim=[codebook_dim] * n_codebooks
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
            ]
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

    @staticmethod
    def from_fishaudio_config(
        fish_dac_config: Mapping[Any, Any],
        # key: PRNGKeyArray = jax.random.PRNGKey(123),
    ) -> "DescriptAudioCodec":
        """Create DAC module from FishAudio DAC config.

        Args:
            fish_dac_config: Config dict from FishAudio DAC model.
            key: PRNG key for random initialization.

        Returns:
            DAC module configured to match FishAudio DAC structure.
        """
        dac_config = DescriptAudioCodecConfig.instantiate_config_from_fishaudio_config(fish_dac_config=fish_dac_config)
        return dac_config.empty(
            # key=key,
        )


class DescriptAudioCodec(TTSAudioDecoder[DescriptAudioCodecConfig]):
    """Lalamo implementation of DAC (Descript Audio Codec)
    Original code: https://github.com/descriptinc/descript-audio-codec
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
        return 1 + self.quantizer.quantizer.n_codebooks  # 1 semantic + n residual

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
            quantizer=self.quantizer.import_weights(quantizer_weights),
            decoder=self.decoder.import_weights(decoder_weights),
        )

    def audio_from_codes(self, indices: Array) -> Array:
        if len(indices.shape) == 2:
            indices = indices[None, :]
        return self(indices)[0, :, 0]
