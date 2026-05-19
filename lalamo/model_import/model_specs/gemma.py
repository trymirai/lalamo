from lalamo.model_import.model_configs import (
    HFGemma3Config,
    HFGemma3TextConfig,
    HFGemma4Config,
)
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin

__all__ = ["GEMMA_MODELS"]

GEMMA4_BASE_CHAT_TEMPLATE = "{{ bos_token or '' }}{% for message in messages %}{{ message.content }}{% endfor %}"


def _gemma4_model_spec(name: str, size: str) -> LanguageModelSpec:
    chat_template = FileSpec("chat_template.jinja") if name.endswith("-it") else GEMMA4_BASE_CHAT_TEMPLATE
    return LanguageModelSpec(
        vendor="Google",
        family="Gemma-4",
        name=name,
        size=size,
        origin=HuggingFaceOrigin(repo=f"google/{name}"),
        config_type=HFGemma4Config,
        configs=ConfigMap(chat_template=chat_template),
    )


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
    _gemma4_model_spec("gemma-4-E2B", "2B"),
    _gemma4_model_spec("gemma-4-E2B-it", "2B"),
    _gemma4_model_spec("gemma-4-E4B", "4B"),
    _gemma4_model_spec("gemma-4-E4B-it", "4B"),
]

GEMMA_MODELS = GEMMA3 + GEMMA4
