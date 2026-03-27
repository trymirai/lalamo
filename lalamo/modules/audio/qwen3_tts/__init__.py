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
