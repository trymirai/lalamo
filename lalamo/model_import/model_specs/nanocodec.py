from lalamo.audio.utils import dummy_char_level_tokenizer_config
from lalamo.model_import.model_configs.nanocodec import NanoCodecForeignConfig

from .common import ConfigMap, FileSpec, ModelSpec, ModelType, WeightsType

__all__ = ["NANOCODEC_TTS_MODELS"]

NEMO_MODEL_ID = "nemo-nano-codec-22khz-1.78kbps-12.5fps"

# Trivial chat template for stub tokenization.
NANOCODEC_CHAT_TEMPLATE = "{{ messages[0].content }}"

NANOCODEC_TTS_MODELS = [
    ModelSpec(
        vendor="NVIDIA",
        family="nanocodec",
        name=NEMO_MODEL_ID,
        size="0.1B",
        quantization=None,
        repo=f"nvidia/{NEMO_MODEL_ID}",
        config_type=NanoCodecForeignConfig,
        weights_type=WeightsType.NEMO,
        configs=ConfigMap(
            model_config=FileSpec(f"{NEMO_MODEL_ID}.nemo"),
            chat_template=NANOCODEC_CHAT_TEMPLATE,
            generation_config=None,
            tokenizer=dummy_char_level_tokenizer_config(),
        ),
        model_type=ModelType.TTS_MODEL,
    ),
]
