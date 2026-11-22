from frozendict import frozendict

from lalamo.model_import.model_configs import HFInkling2LoopConfig
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.models.chat_codec import ReasoningConfig, ReasoningEffort, ResponseFormat

__all__ = ["THINKING_MACHINES_MODELS"]


INKLING_RESPONSE_FORMAT = ResponseFormat(
    turn_pattern=(
        r"<\|message_model\|>(?:<\|content_thinking\|>(?P<chain_of_thought>.*?)"
        r"|<\|content_text\|>(?P<response>.*?))"
        r"(?:<\|end_message\|>|<\|content_model_end_sampling\|>|\Z)"
    ),
)

THINKING_MACHINES_MODELS = [
    LanguageModelSpec(
        vendor="Thinking Machines",
        family="Inkling-2",
        name="Inkling-2-Mini",
        size="26B",
        origin=HuggingFaceOrigin(repo="mlx-community/Inkling-2-Mini-4bit"),
        config_type=HFInkling2LoopConfig,
        configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
        response_format=INKLING_RESPONSE_FORMAT,
        reasoning_config=ReasoningConfig(
            default_reasoning_effort=ReasoningEffort.HIGH,
            field_name="reasoning_effort",
            reasoning_effort_to_field_value=frozendict(
                {
                    ReasoningEffort.XHIGH: "xhigh",
                    ReasoningEffort.HIGH: "high",
                    ReasoningEffort.MEDIUM: "medium",
                    ReasoningEffort.LOW: "low",
                    ReasoningEffort.NO_REASONING: "none",
                }
            ),
        ),
    ),
]
