from lalamo.model_import.model_configs import HFLlamaConfig
from lalamo.model_import.model_spec import LanguageModelSpec
from lalamo.model_import.model_specs.qwen import QWEN_RESPONSE_FORMAT
from lalamo.model_import.model_specs.reasoning_configs import BOOLEAN_REASONING_DEFAULT_ON_CONFIG
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.models.chat_codec import THINKING_RESPONSE_FORMAT

__all__ = ["NANBEIGE_MODELS"]

NANBEIGE_MODELS = [
    LanguageModelSpec(
        vendor="Nanbeige",
        family="Nanbeige-4.1",
        name="Nanbeige4.1-3B",
        size="3B",
        origin=HuggingFaceOrigin(repo="Nanbeige/Nanbeige4.1-3B"),
        config_type=HFLlamaConfig,
        response_format=THINKING_RESPONSE_FORMAT,
    ),
    LanguageModelSpec(
        vendor="Nanbeige",
        family="Nanbeige-4.3",
        name="Nanbeige4.3-33B",
        size="33B",
        origin=HuggingFaceOrigin(repo="Nanbeige/Nanbeige4.3-33B"),
        config_type=HFLlamaConfig,
        response_format=QWEN_RESPONSE_FORMAT,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
    ),
]
