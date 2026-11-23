from frozendict import frozendict

from lalamo.model_import.model_configs import HFMuseGlimmerConfig
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.models.chat_codec import (
    ReasoningConfig,
    ReasoningEffort,
    ResponseFormat,
    TaggedToolCalls,
)
from lalamo.models.language_model import GenerationConfig

__all__ = ["MUSE_GLIMMER_MODELS"]


MUSE_RESPONSE_FORMAT = ResponseFormat(
    turn_pattern=(
        r"\s*to=(?:self<\|message\|>(?P<chain_of_thought>.*?)"
        r"|user<\|message\|>(?P<response>.*?)|[^\s<]+<\|message\|>(?P<tool_calls>.*?))"
        r"(?:<\|eom\|><\|start\|>assistant|<\|eot\|>|<\|end_of_text\|>|\Z)"
    ),
    tool_calls=TaggedToolCalls(
        tags=("<atem:function_calls>", "</atem:function_calls>"),
        function_pattern=r'<atem:invoke name="([^"]+)">(.*?)</atem:invoke>',
        parameter_pattern=r'<atem:parameter name="([^"]+)">(.*?)</atem:parameter>',
    ),
)


MUSE_GLIMMER_MODELS = [
    LanguageModelSpec(
        vendor="Meta",
        family="Muse-Glimmer",
        name="Muse-Glimmer-30B",
        size="30B",
        origin=HuggingFaceOrigin(repo="meta-models/Muse-Glimmer-30B"),
        config_type=HFMuseGlimmerConfig,
        configs=ConfigMap(
            chat_template=FileSpec("chat_template.jinja"),
            generation_params_overrides=GenerationConfig(
                temperature=1.0,
                top_k=64,
                top_p=0.95,
            ),
        ),
        response_format=MUSE_RESPONSE_FORMAT,
        reasoning_config=ReasoningConfig(
            default_reasoning_effort=ReasoningEffort.HIGH,
            field_name="reasoning_strength",
            reasoning_effort_to_field_value=frozendict(
                {
                    ReasoningEffort.LOW: "low",
                    ReasoningEffort.MEDIUM: "medium",
                    ReasoningEffort.HIGH: "high",
                    ReasoningEffort.XHIGH: "xhigh",
                }
            ),
        ),
    ),
]
