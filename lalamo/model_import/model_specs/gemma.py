from lalamo.model_import.model_configs import (
    HFGemma2Config,
    HFGemma3Config,
    HFGemma3TextConfig,
    HFGemma4Config,
)
from lalamo.model_import.origins import HuggingFaceOrigin

from .common import ConfigMap, FileSpec, LanguageModelSpec

__all__ = ["GEMMA_MODELS"]

GEMMA2 = [
    LanguageModelSpec(
        vendor="Google",
        family="Gemma-2",
        name="Gemma-2-2B-Instruct",
        size="2B",
        origin=HuggingFaceOrigin(repo="google/gemma-2-2b-it"),
        config_type=HFGemma2Config,
    ),
]

GEMMA3 = [
    LanguageModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="Gemma-3-1B-Instruct",
        size="1B",
        origin=HuggingFaceOrigin(repo="google/gemma-3-1b-it"),
        config_type=HFGemma3TextConfig,
    ),
    LanguageModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="Gemma-3-1B-Instruct-4bit",
        size="1B",
        origin=HuggingFaceOrigin(repo="mlx-community/gemma-3-1b-it-4bit"),
        config_type=HFGemma3TextConfig,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "google/gemma-3-1b-it")),
    ),
    LanguageModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="Gemma-3-1B-Instruct-8bit",
        size="1B",
        origin=HuggingFaceOrigin(repo="mlx-community/gemma-3-1b-it-8bit"),
        config_type=HFGemma3TextConfig,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "google/gemma-3-1b-it")),
    ),
    LanguageModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="Gemma-3-4B-Instruct",
        size="4B",
        origin=HuggingFaceOrigin(repo="google/gemma-3-4b-it"),
        config_type=HFGemma3Config,
    ),
    LanguageModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="Gemma-3-4B-Instruct-4bit",
        size="4B",
        origin=HuggingFaceOrigin(repo="mlx-community/gemma-3-4b-it-4bit"),
        config_type=HFGemma3Config,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "google/gemma-3-4b-it")),
    ),
    LanguageModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="Gemma-3-4B-Instruct-8bit",
        size="4B",
        origin=HuggingFaceOrigin(repo="mlx-community/gemma-3-4b-it-8bit"),
        config_type=HFGemma3Config,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "google/gemma-3-4b-it")),
    ),
    LanguageModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="Gemma-3-27B-Instruct",
        size="27B",
        origin=HuggingFaceOrigin(repo="google/gemma-3-27b-it"),
        config_type=HFGemma3Config,
    ),
    LanguageModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="Gemma-3-27B-Instruct-4bit",
        size="27B",
        origin=HuggingFaceOrigin(repo="mlx-community/gemma-3-27b-it-4bit"),
        config_type=HFGemma3Config,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "google/gemma-3-27b-it")),
    ),
    LanguageModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="Gemma-3-27B-Instruct-8bit",
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
    ModelSpec(
        vendor="Google",
        family="Gemma-4",
        name="Gemma-4-E2B",
        size="5B",
        quantization=None,
        repo="google/gemma-4-E2B",
        config_type=HFGemma4Config,
        weights_type=WeightsType.SAFETENSORS,
        configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
    ),
    ModelSpec(
        vendor="Google",
        family="Gemma-4",
        name="Gemma-4-E2B-Instruct",
        size="5B",
        quantization=None,
        repo="google/gemma-4-E2B-it",
        config_type=HFGemma4Config,
        weights_type=WeightsType.SAFETENSORS,
        configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
    ),
    ModelSpec(
        vendor="Google",
        family="Gemma-4",
        name="Gemma-4-E4B",
        size="8B",
        quantization=None,
        repo="google/gemma-4-E4B",
        config_type=HFGemma4Config,
        weights_type=WeightsType.SAFETENSORS,
        configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
    ),
    ModelSpec(
        vendor="Google",
        family="Gemma-4",
        name="Gemma-4-E4B-Instruct",
        size="8B",
        quantization=None,
        repo="google/gemma-4-E4B-it",
        config_type=HFGemma4Config,
        weights_type=WeightsType.SAFETENSORS,
        configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
    ),
]
"""
TODO(interview problem):

ModelSpec(
    vendor="Google",
    family="Gemma-4",
    name="Gemma-4-31B",
    size="33B",
    quantization=None,
    repo="google/gemma-4-31B",
    config_type=HFGemma4Config,
    weights_type=WeightsType.SAFETENSORS,
    configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
),
ModelSpec(
    vendor="Google",
    family="Gemma-4",
    name="Gemma-4-31B-Instruct",
    size="33B",
    quantization=None,
    repo="google/gemma-4-31B-it",
    config_type=HFGemma4Config,
    weights_type=WeightsType.SAFETENSORS,
    configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
),
ModelSpec(
    vendor="Google",
    family="Gemma-4",
    name="Gemma-4-26B-A4B",
    size="27B",
    quantization=None,
    repo="google/gemma-4-26B-A4B",
    config_type=HFGemma4Config,
    weights_type=WeightsType.SAFETENSORS,
    configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
),
ModelSpec(
    vendor="Google",
    family="Gemma-4",
    name="Gemma-4-26B-A4B-Instruct",
    size="27B",
    quantization=None,
    repo="google/gemma-4-26B-A4B-it",
    config_type=HFGemma4Config,
    weights_type=WeightsType.SAFETENSORS,
    configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
),
"""

GEMMA_MODELS = GEMMA2 + GEMMA3 + GEMMA4
