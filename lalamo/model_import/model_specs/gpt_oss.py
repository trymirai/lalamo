from lalamo.model_import.model_configs import HFGPTOssConfig
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.model_specs.output_parser_regexes import GPT_OSS_OUTPUT_PARSER_REGEX
from lalamo.model_import.origins import HuggingFaceOrigin

__all__ = ["GPT_OSS_MODELS"]

GPT_OSS_MODELS = [
    LanguageModelSpec(
        vendor="OpenAI",
        family="GPT-OSS",
        name="GPT-OSS-20B",
        size="20B",
        origin=HuggingFaceOrigin(repo="openai/gpt-oss-20b"),
        config_type=HFGPTOssConfig,
        configs=ConfigMap(
            chat_template=FileSpec("chat_template.jinja"),
        ),
        output_parser_regex=GPT_OSS_OUTPUT_PARSER_REGEX,
    ),
]
