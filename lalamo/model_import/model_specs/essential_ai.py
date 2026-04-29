from lalamo.model_import.model_configs.huggingface import HFGemma3TextConfig
from lalamo.model_import.origins import HuggingFaceOrigin

from .common import LanguageModelSpec

__all__ = ["RNJ_MODELS"]

RNJ_MODELS = [
    LanguageModelSpec(
        vendor="EssentialAI",
        family="Rnj-1",
        name="Rnj-1-Instruct",
        size="8B",
        origin=HuggingFaceOrigin(repo="EssentialAI/rnj-1-instruct"),
        config_type=HFGemma3TextConfig,
    ),
]
