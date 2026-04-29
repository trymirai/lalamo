from lalamo.model_import.model_configs import HFLlamaConfig
from lalamo.model_import.origins import HuggingFaceOrigin

from .common import LanguageModelSpec

__all__ = ["REKA_MODELS"]

REKA_MODELS = [
    LanguageModelSpec(
        vendor="Reka",
        family="Reka-Flash",
        name="Reka-Flash-3.1",
        size="21B",
        origin=HuggingFaceOrigin(repo="RekaAI/reka-flash-3.1"),
        config_type=HFLlamaConfig,
        user_role_name="human",
    ),
]
