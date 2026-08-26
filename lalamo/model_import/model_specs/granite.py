from lalamo.model_import.model_configs import HFGraniteConfig
from lalamo.model_import.model_spec import LanguageModelSpec
from lalamo.model_import.model_specs.output_parser_regexes import GRANITE_THINKING_OUTPUT_PARSER_REGEX
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.models.chat_codec import ReasoningConfig, ReasoningEffort

__all__ = ["GRANITE_MODELS"]


GRANITE_MODELS = [
    LanguageModelSpec(
        vendor="IBM",
        family="Granite",
        name=f"granite-{version}-{model_size}-instruct",
        size=model_size.upper(),
        origin=HuggingFaceOrigin(repo=f"ibm-granite/granite-{version}-{model_size}-instruct"),
        config_type=HFGraniteConfig,
        output_parser_regex=output_parser_regex,
        end_of_thinking_tag="</think>" if output_parser_regex is not None else None,
        reasoning_config=reasoning_config,
    )
    for version, output_parser_regex, reasoning_config in (
        (
            "3.3",
            GRANITE_THINKING_OUTPUT_PARSER_REGEX,
            ReasoningConfig(
                default_reasoning_effort=ReasoningEffort.NO_REASONING,
                field_name="thinking",
                reasoning_effort_to_field_value={
                    ReasoningEffort.MEDIUM: True,
                    ReasoningEffort.NO_REASONING: False,
                },
            ),
        ),
        ("3.1", None, None),
    )
    for model_size in ("2b", "8b")
]
