from lalamo.model_import.model_configs.nanocodec import NanoCodecForeignConfig
from lalamo.model_import.model_spec import ConfigMap, FileSpec, TTSModelSpec
from lalamo.model_import.origins import NemoOrigin

__all__ = ["NANOCODEC_TTS_MODELS"]

NEMO_NANOCODEC_MODEL_ID = "nemo-nano-codec-22khz-1.78kbps-12.5fps"

# Trivial chat template for stub tokenization.
NANOCODEC_CHAT_TEMPLATE = """
{% for message in messages %}{{message.content}}{% endfor %}
"""

NANOCODEC_TTS_MODELS = [
    TTSModelSpec(
        vendor="NVIDIA",
        family="nanocodec",
        name=NEMO_NANOCODEC_MODEL_ID,
        size="0.1B",
        origin=NemoOrigin(repo=f"nvidia/{NEMO_NANOCODEC_MODEL_ID}"),
        config_type=NanoCodecForeignConfig,
        configs=ConfigMap(
            model_config=FileSpec(f"{NEMO_NANOCODEC_MODEL_ID}.nemo"),
            chat_template=NANOCODEC_CHAT_TEMPLATE,
            generation_config=None,
            tokenizer=None,
        ),
    ),
]
