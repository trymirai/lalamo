"""Foreign config for NanoCodec TTS models from NVIDIA NeMo."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
from jaxtyping import Array

from lalamo.model import Model
from lalamo.model_import.loaders.nanocodec_loaders import load_nanocodec
from lalamo.model_import.model_configs import ForeignTTSConfig
from lalamo.models import TTSConfig, TTSModel
from lalamo.modules.audio.common_modules import CausalConv1dConfig
from lalamo.modules.audio.fishaudio.fishaudio_modules import Snake1dConfig
from lalamo.modules.audio.nanocodec.audio_decoding import NanoCodec, NanoCodecConfig
from lalamo.modules.audio.nanocodec.nanocodec_consts import (
    DEFAULT_AUDIO_DECODER_INPUT_CONV_SIZE,
    DEFAULT_AUDIO_DECODER_OUTPUT_CONV_SIZE,
    DEFAULT_AUDIO_DECODER_RESBLOCK_DILATIONS,
    DEFAULT_AUDIO_DECODER_RESBLOCK_KERNEL_SIZES,
    DEFAULT_FSQ_EPS,
    DEFAULT_LEAKY_RELU_NEGATIVE_SLOPE,
)
from lalamo.modules.audio.nanocodec.nanocodec_modules import (
    CausalHiFiGANDecoderConfig,
    CausalTransposeConv1dConfig,
    FiniteScalarQuantizerConfig,
    GroupFiniteScalarQuantizerConfig,
    HalfSnakeConfig,
    HiFiGANResBlockConfig,
    HiFiGANResLayerConfig,
    ResidualBlockConfig,
)
from lalamo.modules.audio.nanocodec.stub_text_decoder import StubTextDecoder, StubTextDecoderConfig
from lalamo.modules.audio.vocoders import NoopVocoderConfig
from lalamo.utils.surgery import load_as_at
from lalamo.weight_matrix import CompressionImplementation

__all__ = ["NanoCodecForeignConfig"]


def _tts_audio_decoder(model: TTSModel) -> tuple[object]:
    return (model.audio_decoder,)


def _require_key(config: Mapping, key: str, context: str) -> Any:  # noqa: ANN401
    """Get a required key from config or raise a clear error."""
    if key not in config:
        raise ValueError(f"Required key '{key}' not found in {context} config")
    return config[key]


@dataclass(frozen=True)
class NanoCodecForeignConfig(ForeignTTSConfig):
    """Foreign config for NanoCodec models from NVIDIA NeMo.

    Fields are flattened from the NeMo YAML config sections:
    - sample_rate (top-level)
    - vector_quantizer.num_groups, vector_quantizer.num_levels_per_group
    - audio_decoder.base_channels, audio_decoder.up_sample_rates, etc.
    """

    # Top-level
    samplerate: int

    # vector_quantizer section
    num_groups: int
    num_levels_per_group: tuple[int, ...]

    # audio_decoder section
    base_channels: int
    up_sample_rates: tuple[int, ...]
    in_kernel_size: int
    out_kernel_size: int
    resblock_kernel_sizes: tuple[int, ...]
    resblock_dilation_sizes: tuple[int, ...]

    def to_tts_config(
        self,
        context_length: int | None,  # noqa: ARG002
    ) -> TTSConfig:
        fsq_config = FiniteScalarQuantizerConfig(
            num_levels=self.num_levels_per_group,
            eps=DEFAULT_FSQ_EPS,
        )

        quantizer_config = GroupFiniteScalarQuantizerConfig(
            num_groups=self.num_groups,
            quantizer_config=fsq_config,
        )

        snake_config = Snake1dConfig()
        activation_config = HalfSnakeConfig(
            snake_config=snake_config,
            leaky_relu_negative_slope=DEFAULT_LEAKY_RELU_NEGATIVE_SLOPE,
        )
        conv_config = CausalConv1dConfig(has_biases=True)
        transpose_conv_config = CausalTransposeConv1dConfig(has_biases=True)

        residual_block_config = ResidualBlockConfig(
            activation_config=activation_config,
            conv_config=conv_config,
        )
        hifigan_res_block_config = HiFiGANResBlockConfig(residual_block_config=residual_block_config)
        res_layer_config = HiFiGANResLayerConfig(hifigan_res_block_config=hifigan_res_block_config)

        decoder_config = CausalHiFiGANDecoderConfig(
            activation_config=activation_config,
            pre_conv_config=conv_config,
            transpose_conv_config=transpose_conv_config,
            res_layer_config=res_layer_config,
            post_conv_config=conv_config,
        )

        num_codebooks = self.num_groups
        codebook_size = int(np.prod(self.num_levels_per_group))

        audio_decoder_config = NanoCodecConfig(
            quantizer_config=quantizer_config,
            decoder_config=decoder_config,
            samplerate=self.samplerate,
            base_channels=self.base_channels,
            up_sample_rates=self.up_sample_rates,
            in_kernel_size=self.in_kernel_size,
            out_kernel_size=self.out_kernel_size,
            resblock_kernel_sizes=self.resblock_kernel_sizes,
            resblock_dilations=self.resblock_dilation_sizes,
        )

        text_decoder_config = StubTextDecoderConfig(
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
        )

        return TTSConfig(
            text_decoder_config=text_decoder_config,
            audio_decoder_config=audio_decoder_config,
            vocoder_config=NoopVocoderConfig(),
        )

    def _load_weights(
        self,
        model: Model,
        weights_dict: Mapping[str, Array],
        *,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,  # noqa: ARG002
    ) -> Model:
        assert isinstance(model, TTSModel)
        tts_model: TTSModel = model
        assert isinstance(tts_model.text_decoder, StubTextDecoder)
        assert isinstance(tts_model.audio_decoder, NanoCodec)

        loaded_audio_decoder = load_nanocodec(tts_model.audio_decoder, weights_dict)

        return load_as_at(
            _tts_audio_decoder,
            tts_model,
            (loaded_audio_decoder,),
        )

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        json_path = Path(json_path)
        with open(json_path) as f:
            config = json.load(f)
        return cls.from_nemo_config(config)

    @classmethod
    def from_nemo_config(cls, nemo_config: Mapping) -> Self:
        """Create from a NeMo YAML config dict (extracted from .nemo archive)."""
        quantizer_config = _require_key(nemo_config, "vector_quantizer", "nemo")
        audio_decoder_config = _require_key(nemo_config, "audio_decoder", "nemo")
        assert isinstance(quantizer_config, Mapping)
        assert isinstance(audio_decoder_config, Mapping)

        return cls(
            samplerate=int(_require_key(nemo_config, "sample_rate", "nemo")),
            num_groups=int(_require_key(quantizer_config, "num_groups", "vector_quantizer")),
            num_levels_per_group=tuple(_require_key(quantizer_config, "num_levels_per_group", "vector_quantizer")),
            base_channels=int(_require_key(audio_decoder_config, "base_channels", "audio_decoder")),
            up_sample_rates=tuple(_require_key(audio_decoder_config, "up_sample_rates", "audio_decoder")),
            in_kernel_size=int(audio_decoder_config.get("in_kernel_size", DEFAULT_AUDIO_DECODER_INPUT_CONV_SIZE)),
            out_kernel_size=int(audio_decoder_config.get("out_kernel_size", DEFAULT_AUDIO_DECODER_OUTPUT_CONV_SIZE)),
            resblock_kernel_sizes=audio_decoder_config.get(
                "resblock_kernel_sizes",
                DEFAULT_AUDIO_DECODER_RESBLOCK_KERNEL_SIZES,
            ),
            resblock_dilation_sizes=audio_decoder_config.get(
                "resblock_dilation_sizes",
                DEFAULT_AUDIO_DECODER_RESBLOCK_DILATIONS,
            ),
        )
