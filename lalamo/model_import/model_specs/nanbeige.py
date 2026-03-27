from lalamo.model_import.model_configs import HFLlamaConfig

from .common import ModelSpec
from .origins import HuggingFaceOrigin

__all__ = ["NANBEIGE_MODELS"]

NANBEIGE41 = [
    ModelSpec(
        vendor="Nanbeige",
        family="Nanbeige-4.1",
        name="Nanbeige4.1-3B",
        size="3B",
        source=HuggingFaceOrigin(repo="Nanbeige/Nanbeige4.1-3B"),
        config_type=HFLlamaConfig,
        use_cases=tuple(),
    ),
]

NANBEIGE_MODELS = NANBEIGE41
