from frozendict import frozendict

from lalamo.model_import.model_configs import HFGraniteConfig
from lalamo.model_import.model_spec import LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.models.chat_codec import ReasoningConfig, ReasoningEffort, ResponseFormat

__all__ = ["GRANITE_MODELS"]


GRANITE_RESPONSE_FORMAT = ResponseFormat(
    reasoning_pattern=(
        r"<think>(?P<chain_of_thought>.*?)"
        r"(?:</think>\s*(?:<response>\s*)?(?P<response>.*?)(?:</response>\s*)?)?\Z"
    ),
    prefixes=("<think>",),
)


GRANITE_MODELS = [
    LanguageModelSpec(
        vendor="IBM",
        family="Granite",
        name=f"granite-{version}-{model_size}-instruct",
        size=model_size.upper(),
        origin=HuggingFaceOrigin(repo=f"ibm-granite/granite-{version}-{model_size}-instruct"),
        config_type=HFGraniteConfig,
        response_format=response_format,
        reasoning_config=reasoning_config,
    )
    for version, response_format, reasoning_config in (
        (
            "3.3",
            GRANITE_RESPONSE_FORMAT,
            ReasoningConfig(
                default_reasoning_effort=ReasoningEffort.NO_REASONING,
                field_name="thinking",
                reasoning_effort_to_field_value=frozendict(
                    {
                        ReasoningEffort.MEDIUM: True,
                        ReasoningEffort.NO_REASONING: False,
                    }
                ),
            ),
        ),
        ("3.1", None, None),
    )
    for model_size in ("2b", "8b")
]
