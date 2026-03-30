from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
from jaxtyping import Array, DTypeLike

from lalamo.common import ParameterTree, require_tree
from lalamo.modules.common import Initializer, LalamoModule, register_config_union
from lalamo.sampling import SamplingPolicy, make_policy

from .audio_decoder import TTSAudioDecoder
from .fishaudio.fishaudio_audio_decoding import DescriptAudioCodecConfig
from .fishaudio.fishaudio_text_decoding import FishAudioTextDecoderConfig
from .nanocodec.audio_decoding import NanoCodecConfig
from .nanocodec.stub_text_decoder import StubTextDecoderConfig
from .text_decoder import TTSTextDecoder
from .vocoders import Vocoder, VocoderConfig

DEFAULT_TTS_SAMPLING_POLICY: SamplingPolicy = make_policy(temperature=0.3, top_p=0.9)
DEFAULT_TTS_REPETITION_PENALTY: float = 1.1


TTSAudioDecoderConfig = DescriptAudioCodecConfig | NanoCodecConfig
register_config_union(TTSAudioDecoderConfig)

TTSTextDecoderConfig = FishAudioTextDecoderConfig | StubTextDecoderConfig
register_config_union(TTSTextDecoderConfig)


@dataclass(frozen=True)
class TTSConfig:
    text_decoder_config: TTSTextDecoderConfig
    audio_decoder_config: TTSAudioDecoderConfig
    vocoder_config: VocoderConfig

    activation_precision: DTypeLike

    def init(self, initializer: Initializer) -> "TTSModel":
        text_decoder = self.text_decoder_config.init(initializer)
        audio_decoder = self.audio_decoder_config.init(initializer)
        vocoder = self.vocoder_config.init(initializer)
        return TTSModel(
            text_decoder=text_decoder,
            audio_decoder=audio_decoder,
            vocoder=vocoder,
            activation_precision=self.activation_precision,
        )


class TTSModel(LalamoModule):
    text_decoder: TTSTextDecoder
    audio_decoder: TTSAudioDecoder
    vocoder: Vocoder

    activation_precision: DTypeLike = eqx.field(static=True)

    def export_weights(self) -> ParameterTree[Array]:
        return {}

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            text_decoder=self.text_decoder.import_weights(require_tree(weights["text_decoder"])),
            audio_decoder=self.audio_decoder.import_weights(require_tree(weights["audio_decoder"])),
        )
