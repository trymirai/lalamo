from lalamo.model_import.model_configs.huggingface.neutts import HFNeuTTSConfig

from .common import ConfigMap, ModelSpec, ModelType

__all__ = ["NEUTTS_TTS_MODELS"]

# NeuTTS builds its real prompt at runtime because it needs phonemized text and
# reference audio codes; this template only satisfies lalamo's message processor config.
NEUTTS_TTS_CHAT_TEMPLATE = "{% for message in messages %}{{ message.content }}{% endfor %}"

NEUTTS_TTS_MODELS = [
    ModelSpec(
        vendor="Neuphonic",
        family="NeuTTS Nano",
        name="NeuTTS Nano",
        size="0.23B",
        quantization=None,
        repo="neuphonic/neutts-nano",
        config_type=HFNeuTTSConfig,
        configs=ConfigMap(
            chat_template=NEUTTS_TTS_CHAT_TEMPLATE,
            generation_config=None,
        ),
        model_type=ModelType.TTS_MODEL,
    ),
]
