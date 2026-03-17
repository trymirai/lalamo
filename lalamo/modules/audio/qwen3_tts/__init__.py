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
    Qwen3TTSEuclideanCodebook,
    Qwen3TTSEuclideanCodebookConfig,
    Qwen3TTSResidualVectorQuantization,
    Qwen3TTSResidualVectorQuantizationConfig,
    Qwen3TTSResidualVectorQuantizer,
    Qwen3TTSResidualVectorQuantizerConfig,
    Qwen3TTSSplitResidualVectorQuantizer,
    Qwen3TTSSplitResidualVectorQuantizerConfig,
    Qwen3TTSVectorQuantization,
    Qwen3TTSVectorQuantizationConfig,
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
    "Qwen3TTSAudioDecoder",
    "Qwen3TTSAudioDecoderConfig",
    "Qwen3TTSEuclideanCodebook",
    "Qwen3TTSEuclideanCodebookConfig",
    "Qwen3TTSResidualVectorQuantization",
    "Qwen3TTSResidualVectorQuantizationConfig",
    "Qwen3TTSResidualVectorQuantizer",
    "Qwen3TTSResidualVectorQuantizerConfig",
    "Qwen3TTSSplitResidualVectorQuantizer",
    "Qwen3TTSSplitResidualVectorQuantizerConfig",
    "Qwen3TTSTextDecoder",
    "Qwen3TTSTextDecoderConfig",
    "Qwen3TTSVectorQuantization",
    "Qwen3TTSVectorQuantizationConfig",
    "ResidualUnit",
    "ResidualUnitConfig",
    "SnakeBeta",
    "SnakeBetaConfig",
    "UpsamplingBlock",
    "UpsamplingBlockConfig",
]
