from dataclasses import dataclass

from lalamo.initializer import Initializer
from lalamo.module import LalamoConfig, LalamoModule
from lalamo.sampling import SamplingPolicy

from .audio_decoder import TTSAudioDecoder, TTSAudioDecoderConfig
from .text_decoder import TTSTextDecoder, TTSTextDecoderConfig
from .vocoders import Vocoder, VocoderConfig

DEFAULT_TTS_SAMPLING_POLICY: SamplingPolicy = SamplingPolicy.init(temperature=0.3, top_p=0.9)
DEFAULT_TTS_REPETITION_PENALTY: float = 1.1


@dataclass(frozen=True)
class TTSConfig(LalamoConfig):
    text_decoder_config: TTSTextDecoderConfig
    audio_decoder_config: TTSAudioDecoderConfig
    vocoder_config: VocoderConfig

    def init(self, initializer: Initializer) -> "TTSModel":
        text_decoder = self.text_decoder_config.init(initializer)
        audio_decoder = self.audio_decoder_config.init(initializer)
        vocoder = self.vocoder_config.init(initializer)
        return TTSModel(
            config=self,
            text_decoder=text_decoder,
            audio_decoder=audio_decoder,
            vocoder=vocoder,
        )


class TTSModel(LalamoModule[TTSConfig]):
    text_decoder: TTSTextDecoder
    audio_decoder: TTSAudioDecoder
    vocoder: Vocoder
