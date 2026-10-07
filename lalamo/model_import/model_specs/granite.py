import re

from frozendict import frozendict

from lalamo.model_import.model_configs import HFGraniteConfig
from lalamo.model_import.model_spec import LanguageModelSpec
from lalamo.model_import.model_specs.output_parsers import parse_regex_response
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.models.chat_codec import AssistantMessage, ReasoningConfig, ReasoningEffort, ResponseParser

__all__ = ["GRANITE_MODELS"]


class GraniteResponseParser(ResponseParser):
    pattern = re.compile(
        r"(?s)<think>(?P<chain_of_thought>.*?)"
        r"(?:</think>\s*(?:<response>\s*)?(?P<response>.*?)(?:</response>\s*)?)?\Z"
    )

    @classmethod
    def parse_reasoning(cls, response: str, prompt: str) -> AssistantMessage:
        return parse_regex_response(cls.pattern, response, prompt, ("<think>",))


GRANITE_MODELS = [
    LanguageModelSpec(
        vendor="IBM",
        family="Granite",
        name=f"granite-{version}-{model_size}-instruct",
        size=model_size.upper(),
        origin=HuggingFaceOrigin(repo=f"ibm-granite/granite-{version}-{model_size}-instruct"),
        config_type=HFGraniteConfig,
        response_parser=response_parser,
        reasoning_config=reasoning_config,
    )
    for version, response_parser, reasoning_config in (
        (
            "3.3",
            GraniteResponseParser,
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
