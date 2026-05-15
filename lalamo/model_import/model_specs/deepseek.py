from lalamo.model_import.model_configs import HFQwen2Config
from lalamo.model_import.model_spec import (
    LanguageModelSpec,
)
from lalamo.model_import.model_specs.qwen import QWEN_THINKING_PARSER_REGEX
from lalamo.model_import.origins import HuggingFaceOrigin

__all__ = ["DEEPSEEK_MODELS"]

DEEPSEEK_MODELS = [
    LanguageModelSpec(
        vendor="DeepSeek",
        family="R1-Distill-Qwen",
        name="DeepSeek-R1-Distill-Qwen-1.5B",
        size="1.5B",
        origin=HuggingFaceOrigin(repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
        config_type=HFQwen2Config,
        output_parser_regex=QWEN_THINKING_PARSER_REGEX,
    ),
    LanguageModelSpec(
        vendor="DeepSeek",
        family="R1-Distill-Qwen",
        name="DeepSeek-R1-Distill-Qwen-1.5B-AWQ",
        size="1.5B",
        origin=HuggingFaceOrigin(repo="trymirai/DeepSeek-R1-Distill-Qwen-1.5B-AWQ"),
        config_type=HFQwen2Config,
        output_parser_regex=QWEN_THINKING_PARSER_REGEX,
    ),
]
