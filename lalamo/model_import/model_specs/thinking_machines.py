from frozendict import frozendict

from lalamo.model_import.model_configs import HFInkling2LoopConfig
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.models.chat_codec import ReasoningConfig, ReasoningEffort, ResponseFormat
from lalamo.models.language_model import GenerationConfig

__all__ = ["THINKING_MACHINES_MODELS"]


THINKING_MACHINES_MODELS = [
    LanguageModelSpec(
        vendor="Thinking Machines",
        family="Inkling-2",
        name="Inkling-2-Mini",
        size="26B",
        origin=HuggingFaceOrigin(repo="thinkingmachines/Inkling-2-Mini"),
        config_type=HFInkling2LoopConfig,
        configs=ConfigMap(
            chat_template=FileSpec("chat_template.jinja"),
            generation_params_overrides=GenerationConfig(banned_tokens=tuple(range(200058, 200064))),
        ),
        response_format=ResponseFormat(
            turn_pattern=(
                r"\s*<\|message_model\|>[^<]*"
                r"(?:<\|content_thinking\|>(?P<chain_of_thought>.*?)"
                r"|<\|content_(?:text|tool_error|invoke_tool_text)\|>(?P<response>.*?))"
                r"(?:<\|end_message\|>)?(?:<\|content_model_end_sampling\|>|(?=<\|message_model\|>)|\Z)"
            ),
        ),
        reasoning_config=ReasoningConfig(
            default_reasoning_effort=ReasoningEffort.HIGH,
            field_name="reasoning_effort",
            reasoning_effort_to_field_value=frozendict(
                {
                    ReasoningEffort.NO_REASONING: 0.0,
                    ReasoningEffort.LOW: 0.2,
                    ReasoningEffort.MEDIUM: 0.7,
                    ReasoningEffort.HIGH: 0.9,
                    ReasoningEffort.XHIGH: 0.99,
                },
            ),
        ),
    ),
]
