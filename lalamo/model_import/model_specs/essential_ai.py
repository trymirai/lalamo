from lalamo.model_import.decoder_configs.huggingface import HFGemma3TextConfig

from .common import ModelSpec

__all__ = ["RNJ_MODELS"]

RNJ_MODELS = [
    ModelSpec(
        vendor="EssentialAI",
        family="Rnj-1",
        name="Rnj-1-Instruct",
        size="8B",
        quantization=None,
        repo="EssentialAI/rnj-1-instruct",
        config_type=HFGemma3TextConfig,
    ),
]
