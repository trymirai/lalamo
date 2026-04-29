from lalamo.model_import.model_configs.huggingface.fishaudio import FishAudioConfig
from lalamo.model_import.model_spec import ConfigMap, FileSpec, TTSModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin, WeightFormat

__all__ = ["FISHAUDIO_TTS_MODELS"]

DIRECT_CHAT_TEMPLATE = """
{% for message in messages %}<|{{message.style}}|><|{{message.speaker_id}}|>{{message.content}}{% endfor %}
"""

FISHAUDIO_TTS_MODELS = [
    TTSModelSpec(
        vendor="FishAudio",
        family="openaudio",
        name="s1-mini",
        size="0.8B",
        origin=HuggingFaceOrigin(repo="fishaudio/s1-mini", weight_format=WeightFormat.TORCH),
        config_type=FishAudioConfig,
        configs=ConfigMap(
            chat_template=DIRECT_CHAT_TEMPLATE,
            generation_config=None,
            tokenizer=FileSpec("tokenizer.tiktoken"),
        ),
    ),
]
