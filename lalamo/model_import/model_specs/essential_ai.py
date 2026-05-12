from lalamo.model_import.model_configs.huggingface import HFGemma3TextConfig

from .common import ModelSpec

__all__ = ["RNJ_MODELS"]

RNJ_MODELS = [
    ModelSpec(
        vendor="EssentialAI",
        family="rnj-1",
        name="rnj-1-instruct",
        size="8B",
        quantization=None,
        repo="EssentialAI/rnj-1-instruct",
        config_type=HFGemma3TextConfig,
    ),
]
