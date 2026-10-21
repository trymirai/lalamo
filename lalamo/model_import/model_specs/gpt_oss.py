from frozendict import frozendict

from lalamo.model_import.model_configs import HFGPTOssConfig
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.models.chat_codec import ReasoningConfig, ReasoningEffort, ResponseFormat

__all__ = ["GPT_OSS_MODELS"]


GPT_OSS_RESPONSE_FORMAT = ResponseFormat(
    reasoning_pattern=(
        r"(?:<\|channel\|>analysis<\|message\|>(?P<chain_of_thought>.*?))?"
        r"(?:(?:<\|end\|><\|start\|>assistant)?<\|channel\|>final<\|message\|>(?P<response>.*?))?"
        r"(?:<\|return\|>|<\|end\|>)?\Z"
    ),
    prefixes=("<|channel|>analysis<|message|>", "<|channel|>final<|message|>"),
)


GPT_OSS_MODELS = [
    LanguageModelSpec(
        vendor="OpenAI",
        family="GPT-OSS",
        name="GPT-OSS-20B",
        size="20B",
        origin=HuggingFaceOrigin(repo="openai/gpt-oss-20b"),
        config_type=HFGPTOssConfig,
        configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
        response_format=GPT_OSS_RESPONSE_FORMAT,
        reasoning_config=ReasoningConfig(
            default_reasoning_effort=ReasoningEffort.MEDIUM,
            field_name="reasoning_effort",
            reasoning_effort_to_field_value=frozendict(
                {
                    ReasoningEffort.LOW: "low",
                    ReasoningEffort.MEDIUM: "medium",
                    ReasoningEffort.HIGH: "high",
                }
            ),
        ),
    ),
]
