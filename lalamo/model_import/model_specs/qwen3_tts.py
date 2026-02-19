from lalamo.audio.utils import dummy_char_level_tokenizer_config
from lalamo.model_import.model_configs.huggingface.qwen3_tts import (
    Qwen3TTSTokenizer12HzConfig,
)

from .common import ConfigMap, FileSpec, ModelSpec, ModelType, WeightsType

__all__ = ["QWEN3_TTS_MODELS"]

QWEN3_TTS_CHAT_TEMPLATE = "{{ messages[0].content }}"
QWEN3_TTS_FULL_CHAT_TEMPLATE = (
    "{{ '<|im_start|>assistant\\n' ~ messages[0].content ~ '<|im_end|>\\n<|im_start|>assistant\\n' }}"
)


def _qwen3_tts_model(*, name: str, size: str, repo: str) -> ModelSpec:
    return ModelSpec(
        vendor="Qwen",
        family="qwen3-tts",
        name=name,
        size=size,
        quantization=None,
        repo=repo,
        config_type=Qwen3TTSTokenizer12HzConfig,
        weights_type=WeightsType.SAFETENSORS,
        configs=ConfigMap(
            model_config=FileSpec("config.json"),
            tokenizer=FileSpec("tokenizer.json", "Qwen/Qwen3-0.6B"),
            chat_template=QWEN3_TTS_FULL_CHAT_TEMPLATE,
        ),
        model_type=ModelType.TTS_MODEL,
    )


QWEN3_TTS_MODELS = [
    ModelSpec(
        vendor="Qwen",
        family="qwen3-tts",
        name="tokenizer-12hz",
        size="0.5B",
        quantization=None,
        repo="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        config_type=Qwen3TTSTokenizer12HzConfig,
        weights_type=WeightsType.SAFETENSORS,
        configs=ConfigMap(
            tokenizer=dummy_char_level_tokenizer_config(),
            generation_config=None,
            chat_template=QWEN3_TTS_CHAT_TEMPLATE,
        ),
        model_type=ModelType.TTS_MODEL,
    ),
    _qwen3_tts_model(
        name="Qwen3-TTS-12Hz-0.6B-Base",
        size="0.6B",
        repo="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ),
    _qwen3_tts_model(
        name="Qwen3-TTS-12Hz-0.6B-CustomVoice",
        size="0.6B",
        repo="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    ),
    _qwen3_tts_model(
        name="Qwen3-TTS-12Hz-1.7B-Base",
        size="1.7B",
        repo="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    ),
    _qwen3_tts_model(
        name="Qwen3-TTS-12Hz-1.7B-CustomVoice",
        size="1.7B",
        repo="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    ),
    _qwen3_tts_model(
        name="Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        size="1.7B",
        repo="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    ),
]
