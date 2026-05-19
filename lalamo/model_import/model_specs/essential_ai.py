from lalamo.model_import.model_configs.huggingface import HFGemma3TextConfig
from lalamo.model_import.model_spec import LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin

__all__ = ["RNJ_MODELS"]

RNJ_MODELS = [
    LanguageModelSpec(
        vendor="EssentialAI",
        family="rnj-1",
        name="rnj-1-instruct",
        size="8B",
        origin=HuggingFaceOrigin(repo="EssentialAI/rnj-1-instruct"),
        config_type=HFGemma3TextConfig,
    ),
]
