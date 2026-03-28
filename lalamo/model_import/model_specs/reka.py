from lalamo.model_import.model_configs import HFLlamaConfig

from .common import ModelSpec
from .origins import HuggingFaceOrigin

__all__ = ["REKA_MODELS"]

REKA_MODELS = [
    ModelSpec(
        vendor="Reka",
        family="Reka-Flash",
        name="Reka-Flash-3.1",
        size="21B",
        quantization=None,
        origin=HuggingFaceOrigin(repo="RekaAI/reka-flash-3.1"),
        config_type=HFLlamaConfig,
        user_role_name="human",
        use_cases=tuple(),
    ),
]
