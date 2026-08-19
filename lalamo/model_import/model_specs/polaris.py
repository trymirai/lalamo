from lalamo.model_import.model_configs import HFQwen3Config
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.model_specs.output_parser_regexes import OPTIONAL_THINKING_OUTPUT_PARSER_REGEX
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.models.chat_codec import ENABLE_THINKING_DEFAULT_ON_REASONING_EFFORT_MAPPINGS

__all__ = ["POLARIS_MODELS"]

POLARIS_MODELS = [
    LanguageModelSpec(
        vendor="POLARIS-Project",
        family="Polaris-Preview",
        name="Polaris-4B-Preview",
        size="4B",
        origin=HuggingFaceOrigin(repo="POLARIS-Project/Polaris-4B-Preview"),
        config_type=HFQwen3Config,
        configs=ConfigMap(
            chat_template=FileSpec("chat_template.jinja"),
        ),
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag="\n\n</think>",
        reasoning_effort_mappings=ENABLE_THINKING_DEFAULT_ON_REASONING_EFFORT_MAPPINGS,
    ),
]
