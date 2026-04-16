from lalamo.model_import.model_configs.huggingface import HFGemma3TextConfig

from .common import ModelSpec
from .origins import HuggingFaceOrigin

__all__ = ["RNJ_MODELS"]

RNJ_MODELS = [
    ModelSpec(
        vendor="EssentialAI",
        family="Rnj-1",
        name="Rnj-1-Instruct",
        size="8B",
        quantization=None,
        origin=HuggingFaceOrigin(repo="EssentialAI/rnj-1-instruct"),
        config_type=HFGemma3TextConfig,
    ),
]
