from lalamo.model_import.model_configs import (
    HFGemma3Config,
    HFGemma3TextConfig,
    HFGemma4Config,
)
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.model_specs.reasoning_configs import BOOLEAN_REASONING_DEFAULT_OFF_CONFIG
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.models.chat_codec import ResponseFormat

__all__ = ["GEMMA_MODELS"]


GEMMA4_RESPONSE_FORMAT = ResponseFormat(
    reasoning_pattern=(
        r"(?:<\|channel>thought\n(?P<chain_of_thought>.*?)(?:<channel\|>|\Z))?"
        r"(?P<response>(?:(?!<turn\|>)(?!<\|tool_response>).)+)?"
        r"(?:<turn\|>|<\|tool_response>)?\Z"
    ),
    prefixes=("<|channel>thought\n",),
)


GEMMA4_BASE_CHAT_TEMPLATE = "{{ bos_token or '' }}{% for message in messages %}{{ message.content }}{% endfor %}"

GEMMA3 = [
    LanguageModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="gemma-3-1b-it",
        size="1B",
        origin=HuggingFaceOrigin(repo="google/gemma-3-1b-it"),
        config_type=HFGemma3TextConfig,
    ),
    LanguageModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="gemma-3-1b-it-4bit",
        size="1B",
        origin=HuggingFaceOrigin(repo="mlx-community/gemma-3-1b-it-4bit"),
        config_type=HFGemma3TextConfig,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "google/gemma-3-1b-it")),
    ),
    LanguageModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="gemma-3-1b-it-8bit",
        size="1B",
        origin=HuggingFaceOrigin(repo="mlx-community/gemma-3-1b-it-8bit"),
        config_type=HFGemma3TextConfig,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "google/gemma-3-1b-it")),
    ),
    LanguageModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="gemma-3-4b-it",
        size="4B",
        origin=HuggingFaceOrigin(repo="google/gemma-3-4b-it"),
        config_type=HFGemma3Config,
    ),
    LanguageModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="gemma-3-4b-it-4bit",
        size="4B",
        origin=HuggingFaceOrigin(repo="mlx-community/gemma-3-4b-it-4bit"),
        config_type=HFGemma3Config,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "google/gemma-3-4b-it")),
    ),
    LanguageModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="gemma-3-4b-it-8bit",
        size="4B",
        origin=HuggingFaceOrigin(repo="mlx-community/gemma-3-4b-it-8bit"),
        config_type=HFGemma3Config,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "google/gemma-3-4b-it")),
    ),
    LanguageModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="gemma-3-27b-it",
        size="27B",
        origin=HuggingFaceOrigin(repo="google/gemma-3-27b-it"),
        config_type=HFGemma3Config,
    ),
    LanguageModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="gemma-3-27b-it-4bit",
        size="27B",
        origin=HuggingFaceOrigin(repo="mlx-community/gemma-3-27b-it-4bit"),
        config_type=HFGemma3Config,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "google/gemma-3-27b-it")),
    ),
    LanguageModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="gemma-3-27b-it-8bit",
        size="27B",
        origin=HuggingFaceOrigin(repo="mlx-community/gemma-3-27b-it-8bit"),
        config_type=HFGemma3Config,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "google/gemma-3-27b-it")),
    ),
    LanguageModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="FunctionGemma-270M-IT",
        size="270M",
        origin=HuggingFaceOrigin(repo="google/functiongemma-270m-it"),
        config_type=HFGemma3TextConfig,
        system_role_name="developer",
        configs=ConfigMap(
            chat_template=FileSpec("chat_template.jinja"),
            system_prompt="You are a model that can do function calling with the following functions",
        ),
    ),
]

GEMMA4 = [
    LanguageModelSpec(
        vendor="Google",
        family="Gemma-4",
        name=f"gemma-4-{variant.lower()}{suffix}",
        size=size,
        origin=HuggingFaceOrigin(repo=f"google/gemma-4-{variant}{suffix}"),
        config_type=HFGemma4Config,
        configs=configs,
        response_format=response_format,
        reasoning_config=reasoning_config,
    )
    for variant, size in (("E2B", "5B"), ("E4B", "8B"), ("26B-A4B", "26B"), ("31B", "31B"))
    for suffix, configs, response_format, reasoning_config in (
        ("", ConfigMap(chat_template=GEMMA4_BASE_CHAT_TEMPLATE), None, None),
        (
            "-it",
            ConfigMap(chat_template=FileSpec("chat_template.jinja")),
            GEMMA4_RESPONSE_FORMAT,
            BOOLEAN_REASONING_DEFAULT_OFF_CONFIG,
        ),
    )
]

GEMMA_MODELS = GEMMA3 + GEMMA4
