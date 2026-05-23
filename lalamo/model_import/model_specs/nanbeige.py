from lalamo.model_import.model_configs import HFLlamaConfig
from lalamo.model_import.model_spec import LanguageModelSpec
from lalamo.model_import.model_specs.output_parser_regexes import OPTIONAL_THINKING_OUTPUT_PARSER_REGEX
from lalamo.model_import.origins import HuggingFaceOrigin

__all__ = ["NANBEIGE_MODELS"]

NANBEIGE41 = [
    LanguageModelSpec(
        vendor="Nanbeige",
        family="Nanbeige-4.1",
        name="Nanbeige4.1-3B",
        size="3B",
        origin=HuggingFaceOrigin(repo="Nanbeige/Nanbeige4.1-3B"),
        config_type=HFLlamaConfig,
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
    ),
]

NANBEIGE_MODELS = NANBEIGE41
