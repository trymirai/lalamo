from lalamo.modules.audio.common_modules import (
    DecoderBlock,
    DecoderBlockConfig,
    ResidualUnit,
    ResidualUnitConfig,
    SnakeBeta,
    SnakeBetaConfig,
    UpsamplingBlock,
    UpsamplingBlockConfig,
)

from .qwen3_tts_audio_decoding import (
    QWEN3_TTS_AUDIO_DECODER_CHUNK_SIZE_DEFAULT,
    QWEN3_TTS_AUDIO_DECODER_LEFT_CONTEXT_SIZE_DEFAULT,
    Qwen3TTSAudioDecoder,
    Qwen3TTSAudioDecoderConfig,
)
from .qwen3_tts_modules import (
    EuclideanCodebook,
    EuclideanCodebookConfig,
    ResidualVectorQuantizer,
    ResidualVectorQuantizerConfig,
    VectorQuantization,
    VectorQuantizationConfig,
)
from .qwen3_tts_text_decoding import (
    Qwen3TTSTextDecoder,
    Qwen3TTSTextDecoderConfig,
)

__all__ = [
    "QWEN3_TTS_AUDIO_DECODER_CHUNK_SIZE_DEFAULT",
    "QWEN3_TTS_AUDIO_DECODER_LEFT_CONTEXT_SIZE_DEFAULT",
    "DecoderBlock",
    "DecoderBlockConfig",
    "EuclideanCodebook",
    "EuclideanCodebookConfig",
    "Qwen3TTSAudioDecoder",
    "Qwen3TTSAudioDecoderConfig",
    "Qwen3TTSTextDecoder",
    "Qwen3TTSTextDecoderConfig",
    "ResidualUnit",
    "ResidualUnitConfig",
    "ResidualVectorQuantizer",
    "ResidualVectorQuantizerConfig",
    "SnakeBeta",
    "SnakeBetaConfig",
    "UpsamplingBlock",
    "UpsamplingBlockConfig",
    "VectorQuantization",
    "VectorQuantizationConfig",
]
