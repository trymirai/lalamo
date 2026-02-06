from lalamo.model_import.model_configs.huggingface.fishaudio import FishAudioConfig

from .common import ConfigMap, FileSpec, ModelSpec, ModelType, WeightsType

__all__ = ["FISHAUDIO_TTS_MODELS"]

DIRECT_CHAT_TEMPLATE = """
{% for message in messages %}<|{{message.style}}|><|{{message.speaker_id}}|>{{message.content}}{% endfor %}
"""

FISHAUDIO_TTS_MODELS = [
    ModelSpec(
        vendor="FishAudio",
        family="openaudio",
        name="s1-mini",
        size="0.8B",
        quantization=None,
        repo="fishaudio/s1-mini",
        config_type=FishAudioConfig,
        weights_type=WeightsType.TORCH,
        configs=ConfigMap(
            chat_template=DIRECT_CHAT_TEMPLATE,
            generation_config=None,
            tokenizer=FileSpec("tokenizer.tiktoken"),
        ),
        model_type=ModelType.TTS_MODEL,
    ),
]
