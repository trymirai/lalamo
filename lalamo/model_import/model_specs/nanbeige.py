from lalamo.model_import.model_configs import HFLlamaConfig
from lalamo.model_import.model_spec import LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.models.chat_codec import THINKING_RESPONSE_FORMAT

__all__ = ["NANBEIGE_MODELS"]

NANBEIGE41 = [
    LanguageModelSpec(
        vendor="Nanbeige",
        family="Nanbeige-4.1",
        name="Nanbeige4.1-3B",
        size="3B",
        origin=HuggingFaceOrigin(repo="Nanbeige/Nanbeige4.1-3B"),
        config_type=HFLlamaConfig,
        response_format=THINKING_RESPONSE_FORMAT,
    ),
]

NANBEIGE_MODELS = NANBEIGE41
