"""FishAudio DescriptAudioCodec (DAC) implementation.

This module provides the high-level DescriptAudioCodec class for decoding
audio codes to waveforms. Internal building blocks are in fishaudio_modules.py.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

import jax
from fish_speech.models.dac import inference as fish_dac_inference
from fish_speech.models.dac.modded_dac import DAC
from hydra.utils import instantiate
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree
from lalamo.model_import.model_specs.common import cast_if_float
from lalamo.modules import AudioDecoder
from lalamo.modules.torch_interop import torch_to_jax
from lalamo.utils import MapDictValues

from .fishaudio_common import get_default_fishaudio_dac_config
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

    def empty(
        self,
        quantizer_upsampler_trans_conv_params: tuple[TransposeConvSpatialParams, ...],
        quantizer_convnext_spatial_params: ConvNeXtSpatialParams,
        semantic_quantizer_params: VectorQuantizerParams,
        quantizer_params: VectorQuantizerParams,
        decoder_spatial_params: DACDecoderSpatialParams,
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
        quantizer = self.quantizer_config.empty(
            upsampler_trans_conv_params=quantizer_upsampler_trans_conv_params,
            convnext_spatial_params=quantizer_convnext_spatial_params,
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
        quantizer_upsampler_trans_conv_params: tuple[TransposeConvSpatialParams, ...],
        quantizer_convnext_spatial_params: ConvNeXtSpatialParams,
        semantic_quantizer_params: VectorQuantizerParams,
        quantizer_params: VectorQuantizerParams,
        decoder_spatial_params: DACDecoderSpatialParams,
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

        quantizer = self.quantizer_config.random_init(
            upsampler_trans_conv_params=quantizer_upsampler_trans_conv_params,
            convnext_spatial_params=quantizer_convnext_spatial_params,
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
    def from_fishaudio_config(
        fish_dac_config: Mapping[Any, Any],
        key: PRNGKeyArray = jax.random.PRNGKey(123),
    ) -> "DescriptAudioCodec":
        """Create DAC module from FishAudio DAC config.

        Args:
            fish_dac_config: Config dict from FishAudio DAC model.
            key: PRNG key for random initialization.

        Returns:
            DAC module configured to match FishAudio DAC structure.
        """
        precision = jnp.float32

        # Extract config parameters
        encoder_dim = fish_dac_config["encoder_dim"]
        encoder_rates = fish_dac_config["encoder_rates"]
        decoder_dim = fish_dac_config["decoder_dim"]
        decoder_rates = fish_dac_config["decoder_rates"]
        latent_dim = encoder_dim * (2 ** len(encoder_rates))  # 64 * 16 = 1024

        fish_quantizer_config = fish_dac_config["quantizer"]

        # Build quantizer config using existing helper
        input_dim = fish_quantizer_config["input_dim"]
        n_codebooks = fish_quantizer_config["n_codebooks"]
        codebook_dim = fish_quantizer_config["codebook_dim"]
        downsample_factor = fish_quantizer_config["downsample_factor"]
        codebook_size = fish_quantizer_config["codebook_size"]
        semantic_codebook_size = fish_quantizer_config["semantic_codebook_size"]
        post_module_config_dict = fish_quantizer_config["post_module"]
        downsample_dims = [input_dim for _ in range(len(downsample_factor))]
        all_dims = (input_dim, *downsample_dims)

        semantic_quantizer_params = VectorQuantizerParams(
            input_dim=input_dim, codebook_size=semantic_codebook_size, codebook_dim=codebook_dim
        )
        quantizer_params = VectorQuantizerParams(
            input_dim=input_dim, codebook_size=codebook_size, codebook_dim=[codebook_dim] * n_codebooks
        )

        upsampler_config = UpsamplerConfig(
            block_configs=tuple([UpsamplingBlockConfig(precision)] * len(downsample_factor))
        )
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
        convnext_params = ConvNeXtSpatialParams()
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

        # Build decoder config
        decoder_config = DACDecoderConfig(precision=precision, causal=True)
        decoder_spatial_params = DACDecoderSpatialParams(
            input_channel=latent_dim,
            channels=decoder_dim,
            rates=tuple(decoder_rates),
            d_out=1,
        )

        # Create full DAC config
        dac_config = DescriptAudioCodecConfig(
            precision=precision,
            quantizer_config=quantizer_full_config,
            decoder_config=decoder_config,
        )

        return dac_config.random_init(
            quantizer_upsampler_trans_conv_params=upsample_conv_params,
            quantizer_convnext_spatial_params=convnext_params,
            semantic_quantizer_params=semantic_quantizer_params,
            quantizer_params=quantizer_params,
            decoder_spatial_params=decoder_spatial_params,
            key=key,
        )

    @classmethod
    def from_foreign_model(cls, audio_chkpt_path: Path, precision: DTypeLike) -> "DescriptAudioCodec":
        """Load a DescriptAudioCodec from a FishAudio DAC checkpoint.

        Args:
            audio_chkpt_path: Path to the FishAudio DAC checkpoint file.
            precision: Data type precision for the model weights.

        Returns:
            DescriptAudioCodec module with loaded weights.
        """
        from lalamo.model_import.loaders.fishaudio_loaders import (
            load_audio_decoder,
            load_downsample_rvq,
        )

        # Load FishAudio model
        fish_dac = fish_dac_inference.load_model("modded_dac_vq", audio_chkpt_path, device="cpu")
        assert isinstance(fish_dac, DAC)

        # Create empty module structure from config
        dac_module = DescriptAudioCodecConfig.from_fishaudio_config(get_default_fishaudio_dac_config())

        # Extract and convert quantizer weights
        quantizer_weights = dict(
            MapDictValues(
                lambda v: cast_if_float(torch_to_jax(v), precision),
                fish_dac.quantizer.state_dict(),
            )
        )
        loaded_quantizer = load_downsample_rvq(dac_module.quantizer, quantizer_weights)

        # Extract and convert decoder weights
        decoder_weights = dict(
            MapDictValues(
                lambda v: cast_if_float(torch_to_jax(v), precision),
                fish_dac.decoder.state_dict(),
            )
        )
        loaded_decoder = load_audio_decoder(dac_module.decoder, decoder_weights)

        # Create the final module with loaded components
        return DescriptAudioCodec(
            config=dac_module.config,
            quantizer=loaded_quantizer,
            decoder=loaded_decoder,
        )


class DescriptAudioCodec(AudioDecoder[DescriptAudioCodecConfig]):
    """Lalamo implementation of DAC (Descript Audio Codec)
    Original code: https://github.com/descriptinc/descript-audio-codec
    Input: Integer codes with shape (batch, n_codebooks, tokens)
    Output: Audio waveform with shape (batch, audio_samples, 1) in range [-1, 1]
    """

    quantizer: DownsampleResidualVectorQuantize
    decoder: DACDecoder

    @property
    def samplerate(self) -> int:
        return 44100  # TODO: retrieve this one from torch DAC when loading

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

    def decode(
        self,
        indices: Int[Array, "batch n_codebooks tokens"],
    ) -> Float[Array, "batch audio_samples 1"]:
        """Decode audio codes to waveform.

        Args:
            indices: Integer codes with shape (batch, n_codebooks, tokens).
                     First row (indices[:, 0]) contains semantic codes,
                     remaining rows (indices[:, 1:]) contain residual codes.

        Returns:
            Audio waveform with shape (batch, audio_samples, 1) in range [-1, 1].
        """
        # Decode codes to continuous latent representation
        z = self.quantizer.decode(indices)

        # Decode latent to audio waveform
        audio = self.decoder(z)

        return audio

    def __call__(
        self,
        indices: Int[Array, "batch n_codebooks tokens"],
    ) -> Float[Array, "batch audio_samples 1"]:
        """Decode audio codes to waveform.

        This is an alias for the decode method.
        """
        return self.decode(indices)

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
        return self(indices)[0, :]
