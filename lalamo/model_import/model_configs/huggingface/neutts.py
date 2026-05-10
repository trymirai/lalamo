from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Self

from jaxtyping import Array, DTypeLike

from lalamo.model_import.loaders.huggingface import load_huggingface_decoder
from lalamo.model_import.model_configs import ForeignTTSConfig
from lalamo.modules import LalamoModule, TTSConfig, TTSModel, VocoderConfig
from lalamo.modules.audio.neutts import (
    NeuCodecAudioDecoderConfig,
    NeuTTSTextDecoder,
    NeuTTSTextDecoderConfig,
)

from .llama import HFLlamaConfig

__all__ = ["HFNeuTTSConfig"]


@dataclass(frozen=True)
class HFNeuTTSConfig(ForeignTTSConfig):
    llama_config: HFLlamaConfig
    codec_repo: str = "neuphonic/neucodec"
    codec_device: str = "cpu"
    language_code: str = "en-us"

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
            language_code=self.language_code,
        )
        audio_decoder_config = NeuCodecAudioDecoderConfig(
            precision=activation_precision,
            codec_repo=self.codec_repo,
            device=self.codec_device,
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

        loaded_decoder = load_huggingface_decoder(model.text_decoder.decoder, weights_dict)
        return replace(
            model,
            text_decoder=replace(model.text_decoder, decoder=loaded_decoder),
        )
