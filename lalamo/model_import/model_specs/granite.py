from lalamo.model_import.model_configs import HFGraniteConfig
from lalamo.model_import.model_spec import ConfigMap, LanguageModelSpec
from lalamo.model_import.model_specs.output_parser_regexes import GRANITE_THINKING_OUTPUT_PARSER_REGEX
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.models.chat_codec import ReasoningEffort, ReasoningEffortMapping
from lalamo.models.language_model import GenerationConfig

__all__ = ["GRANITE_MODELS"]


GRANITE33_REASONING_EFFORT_MAPPINGS = (
    ReasoningEffortMapping(effort=ReasoningEffort.XHIGH, parameter="thinking", value=True),
)

GRANITE_MODELS = [
    *(
        LanguageModelSpec(
            vendor="IBM",
            family="Granite",
            name=name,
            size=size,
            origin=HuggingFaceOrigin(repo=f"ibm-granite/{name}"),
            config_type=HFGraniteConfig,
            configs=ConfigMap(
                generation_params_overrides=GenerationConfig(reasoning_effort=ReasoningEffort.NONE),
            ),
            output_parser_regex=GRANITE_THINKING_OUTPUT_PARSER_REGEX,
            end_of_thinking_tag="</think>",
            reasoning_effort_mappings=GRANITE33_REASONING_EFFORT_MAPPINGS,
        )
        for name, size in (
            ("granite-3.3-2b-instruct", "2B"),
            ("granite-3.3-8b-instruct", "8B"),
        )
    ),
    *(
        LanguageModelSpec(
            vendor="IBM",
            family="Granite",
            name=name,
            size=size,
            origin=HuggingFaceOrigin(repo=f"ibm-granite/{name}"),
            config_type=HFGraniteConfig,
            configs=ConfigMap(),
        )
        for name, size in (
            ("granite-3.1-2b-instruct", "2B"),
            ("granite-3.1-8b-instruct", "8B"),
        )
    ),
]
