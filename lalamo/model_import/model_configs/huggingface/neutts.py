from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Self

from jaxtyping import Array, DTypeLike

from lalamo.model_import.loaders.common import load_parameters
from lalamo.model_import.loaders.huggingface import load_huggingface_decoder
from lalamo.model_import.loaders.neutts_codec_loaders import load_neucodec_audio_decoder_from_huggingface
from lalamo.model_import.model_configs import ForeignTTSConfig
from lalamo.modules import LalamoModule, TTSConfig, TTSModel, VocoderConfig
from lalamo.modules.audio.neutts import (
    NeuCodecAudioDecoder,
    NeuCodecAudioDecoderConfig,
    NeuTTSTextDecoder,
    NeuTTSTextDecoderConfig,
)

from .llama import HFLlamaConfig

__all__ = ["HFNeuTTSConfig"]

DEFAULT_NEUTTS_LANGUAGE_CODE = "en-us"
NEUTTS_CODEC_REPO = "neuphonic/neucodec"
NEUTTS_CODEC_CHECKPOINT_FILENAME = "pytorch_model.bin"


@dataclass(frozen=True)
class HFNeuTTSConfig(ForeignTTSConfig):
    llama_config: HFLlamaConfig

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        return cls(llama_config=HFLlamaConfig.from_json(json_path))

    @property
    def default_precision(self) -> DTypeLike:
        return self.llama_config.default_precision

    def to_tts_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
    ) -> TTSConfig:
        (speech_generation_end_token_id,) = self.llama_config.eos_token_ids
        decoder_config = self.llama_config.to_decoder_config(
            context_length=context_length,
            activation_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            metadata_dict={},
        )
        text_decoder_config = NeuTTSTextDecoderConfig(
            decoder_config=decoder_config,
            speech_generation_end_token_id=speech_generation_end_token_id,
            max_context_length=decoder_config.transformer_config.context_length,
            language_code=DEFAULT_NEUTTS_LANGUAGE_CODE,
        )
        audio_decoder_config = NeuCodecAudioDecoderConfig(
            precision=activation_precision,
        )
        return TTSConfig(
            text_decoder_config=text_decoder_config,
            audio_decoder_config=audio_decoder_config,
            vocoder_config=VocoderConfig(),
            activation_precision=activation_precision,
        )

    def _load_weights(
        self,
        model: LalamoModule,
        weights_dict: Mapping[str, Array],
    ) -> LalamoModule:
        assert isinstance(model, TTSModel)
        assert isinstance(model.text_decoder, NeuTTSTextDecoder)
        assert isinstance(model.audio_decoder, NeuCodecAudioDecoder)

        loaded_decoder = load_huggingface_decoder(model.text_decoder.decoder, weights_dict)
        loaded_text_decoder = replace(model.text_decoder, decoder=loaded_decoder)
        loaded_audio_decoder = load_neucodec_audio_decoder_from_huggingface(
            model.audio_decoder,
            repo_id=NEUTTS_CODEC_REPO,
            filename=NEUTTS_CODEC_CHECKPOINT_FILENAME,
        )
        return load_parameters(
            lambda m: (m.text_decoder, m.audio_decoder),
            model,
            (loaded_text_decoder, loaded_audio_decoder),
        )
