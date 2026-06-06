from lalamo.model_import.model_configs import HFLlamaConfig, HFSmolLM3Config
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.model_specs.output_parser_regexes import OPTIONAL_THINKING_OUTPUT_PARSER_REGEX
from lalamo.model_import.origins import HuggingFaceOrigin

__all__ = ["HUGGINGFACE_MODELS"]

HUGGINGFACE_MODELS = [
    LanguageModelSpec(
        vendor="HuggingFace",
        family="SmolLM2",
        name="SmolLM2-1.7B-Instruct",
        size="1.7B",
        origin=HuggingFaceOrigin(repo="HuggingFaceTB/SmolLM2-1.7B-Instruct"),
        config_type=HFLlamaConfig,
    ),
    LanguageModelSpec(
        vendor="HuggingFace",
        family="SmolLM3",
        name="SmolLM3-3B",
        size="3B",
        origin=HuggingFaceOrigin(repo="HuggingFaceTB/SmolLM3-3B"),
        config_type=HFSmolLM3Config,
        configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag="\n</think>",
    ),
    LanguageModelSpec(
        vendor="HuggingFace",
        family="SmolLM3",
        name="SmolLM3-3B-4bit",
        size="3B",
        origin=HuggingFaceOrigin(repo="mlx-community/SmolLM3-3B-4bit"),
        config_type=HFSmolLM3Config,
        configs=ConfigMap(
            generation_config=FileSpec("generation_config.json", "HuggingFaceTB/SmolLM3-3B"),
            chat_template=FileSpec("chat_template.jinja", "HuggingFaceTB/SmolLM3-3B"),
        ),
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag="\n</think>",
    ),
    LanguageModelSpec(
        vendor="HuggingFace",
        family="SmolLM3",
        name="SmolLM3-3B-8bit",
        size="3B",
        origin=HuggingFaceOrigin(repo="mlx-community/SmolLM3-3B-8bit"),
        config_type=HFSmolLM3Config,
        configs=ConfigMap(
            generation_config=FileSpec("generation_config.json", "HuggingFaceTB/SmolLM3-3B"),
            chat_template=FileSpec("chat_template.jinja", "HuggingFaceTB/SmolLM3-3B"),
        ),
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag="\n</think>",
    ),
]
