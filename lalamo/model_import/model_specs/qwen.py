from lalamo.model_import.model_configs import (
    HFBonsaiConfig,
    HFQwen2Config,
    HFQwen3Config,
    HFQwen3NextConfig,
    HFQwen35Config,
)
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin

__all__ = ["QWEN_MODELS"]


QWEN25 = [
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen2.5",
        name="Qwen2.5-0.5B-Instruct",
        size="0.5B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen2.5-0.5B-Instruct"),
        config_type=HFQwen2Config,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen2.5",
        name="Qwen2.5-1.5B-Instruct",
        size="1.5B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen2.5-1.5B-Instruct"),
        config_type=HFQwen2Config,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen2.5",
        name="Qwen2.5-3B-Instruct",
        size="3B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen2.5-3B-Instruct"),
        config_type=HFQwen2Config,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen2.5",
        name="Qwen2.5-7B-Instruct",
        size="7B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen2.5-7B-Instruct"),
        config_type=HFQwen2Config,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen2.5",
        name="Qwen2.5-14B-Instruct",
        size="14B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen2.5-14B-Instruct"),
        config_type=HFQwen2Config,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen2.5",
        name="Qwen2.5-32B-Instruct",
        size="32B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen2.5-32B-Instruct"),
        config_type=HFQwen2Config,
    ),
]


QWEN25_CODER = [
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen2.5-Coder",
        name="Qwen2.5-Coder-0.5B-Instruct",
        size="0.5B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen2.5-Coder-0.5B-Instruct"),
        config_type=HFQwen2Config,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen2.5-Coder",
        name="Qwen2.5-Coder-1.5B-Instruct",
        size="1.5B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen2.5-Coder-1.5B-Instruct"),
        config_type=HFQwen2Config,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen2.5-Coder",
        name="Qwen2.5-Coder-3B-Instruct",
        size="3B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen2.5-Coder-3B-Instruct"),
        config_type=HFQwen2Config,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen2.5-Coder",
        name="Qwen2.5-Coder-7B-Instruct",
        size="7B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen2.5-Coder-7B-Instruct"),
        config_type=HFQwen2Config,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen2.5-Coder",
        name="Qwen2.5-Coder-14B-Instruct",
        size="14B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen2.5-Coder-14B-Instruct"),
        config_type=HFQwen2Config,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen2.5-Coder",
        name="Qwen2.5-Coder-32B-Instruct",
        size="32B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen2.5-Coder-32B-Instruct"),
        config_type=HFQwen2Config,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen2.5-Coder",
        name="Qwen2.5-Coder-32B-Instruct",
        size="32B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen2.5-Coder-32B-Instruct"),
        config_type=HFQwen2Config,
    ),
]


QWEN3 = [
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-0.6B",
        size="0.6B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3-0.6B"),
        config_type=HFQwen3Config,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-0.6B-MLX-4bit",
        size="0.6B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3-0.6B-MLX-4bit"),
        config_type=HFQwen3Config,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", "Qwen/Qwen3-0.6B"),
            tokenizer_config=FileSpec("tokenizer_config.json", "Qwen/Qwen3-0.6B"),
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3-0.6B"),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-1.7B",
        size="1.7B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3-1.7B"),
        config_type=HFQwen3Config,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-4B",
        size="4B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3-4B"),
        config_type=HFQwen3Config,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-4B-AWQ",
        size="4B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3-4B-AWQ"),
        config_type=HFQwen3Config,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-4B-MLX-4bit",
        size="4B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3-4B-MLX-4bit"),
        config_type=HFQwen3Config,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", "Qwen/Qwen3-4B"),
            tokenizer_config=FileSpec("tokenizer_config.json", "Qwen/Qwen3-4B"),
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3-4B"),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-8B",
        size="8B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3-8B"),
        config_type=HFQwen3Config,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-8B-AWQ",
        size="8B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3-8B-AWQ"),
        config_type=HFQwen3Config,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-8B-MLX-4bit",
        size="8B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3-8B-MLX-4bit"),
        config_type=HFQwen3Config,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", "Qwen/Qwen3-8B"),
            tokenizer_config=FileSpec("tokenizer_config.json", "Qwen/Qwen3-8B"),
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3-8B"),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-14B",
        size="14B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3-14B"),
        config_type=HFQwen3Config,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-14B-AWQ",
        size="14B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3-14B-AWQ"),
        config_type=HFQwen3Config,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-32B",
        size="32B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3-32B"),
        config_type=HFQwen3Config,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-32B-AWQ",
        size="32B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3-32B-AWQ"),
        config_type=HFQwen3Config,
    ),
]

QWEN3_NEXT = [
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3-Next",
        name="Qwen3-Next-80B-A3B-Instruct",
        size="80B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3-Next-80B-A3B-Instruct"),
        config_type=HFQwen3NextConfig,
    ),
]

QWEN35 = [
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-0.8B",
        size="0.8B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3.5-0.8B"),
        config_type=HFQwen35Config,
        configs=ConfigMap(
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-0.8B-MLX-4bit",
        size="0.8B",
        origin=HuggingFaceOrigin(repo="mlx-community/Qwen3.5-0.8B-MLX-4bit"),
        config_type=HFQwen35Config,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", "Qwen/Qwen3.5-0.8B"),
            tokenizer_config=FileSpec("tokenizer_config.json", "Qwen/Qwen3.5-0.8B"),
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-0.8B-MLX-8bit",
        size="0.8B",
        origin=HuggingFaceOrigin(repo="mlx-community/Qwen3.5-0.8B-MLX-8bit"),
        config_type=HFQwen35Config,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", "Qwen/Qwen3.5-0.8B"),
            tokenizer_config=FileSpec("tokenizer_config.json", "Qwen/Qwen3.5-0.8B"),
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-2B",
        size="2B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3.5-2B"),
        config_type=HFQwen35Config,
        configs=ConfigMap(
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-2B-MLX-4bit",
        size="2B",
        origin=HuggingFaceOrigin(repo="mlx-community/Qwen3.5-2B-MLX-4bit"),
        config_type=HFQwen35Config,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", "Qwen/Qwen3.5-2B"),
            tokenizer_config=FileSpec("tokenizer_config.json", "Qwen/Qwen3.5-2B"),
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-2B-MLX-8bit",
        size="2B",
        origin=HuggingFaceOrigin(repo="mlx-community/Qwen3.5-2B-MLX-8bit"),
        config_type=HFQwen35Config,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", "Qwen/Qwen3.5-2B"),
            tokenizer_config=FileSpec("tokenizer_config.json", "Qwen/Qwen3.5-2B"),
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-4B",
        size="4B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3.5-4B"),
        config_type=HFQwen35Config,
        configs=ConfigMap(
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-4B-MLX-4bit",
        size="4B",
        origin=HuggingFaceOrigin(repo="mlx-community/Qwen3.5-4B-MLX-4bit"),
        config_type=HFQwen35Config,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", "Qwen/Qwen3.5-4B"),
            tokenizer_config=FileSpec("tokenizer_config.json", "Qwen/Qwen3.5-4B"),
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-4B-MLX-8bit",
        size="4B",
        origin=HuggingFaceOrigin(repo="mlx-community/Qwen3.5-4B-MLX-8bit"),
        config_type=HFQwen35Config,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", "Qwen/Qwen3.5-4B"),
            tokenizer_config=FileSpec("tokenizer_config.json", "Qwen/Qwen3.5-4B"),
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-9B",
        size="9B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3.5-9B"),
        config_type=HFQwen35Config,
        configs=ConfigMap(
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-9B-MLX-4bit",
        size="9B",
        origin=HuggingFaceOrigin(repo="mlx-community/Qwen3.5-9B-MLX-4bit"),
        config_type=HFQwen35Config,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", "Qwen/Qwen3.5-9B"),
            tokenizer_config=FileSpec("tokenizer_config.json", "Qwen/Qwen3.5-9B"),
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-9B-MLX-8bit",
        size="9B",
        origin=HuggingFaceOrigin(repo="mlx-community/Qwen3.5-9B-MLX-8bit"),
        config_type=HFQwen35Config,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", "Qwen/Qwen3.5-9B"),
            tokenizer_config=FileSpec("tokenizer_config.json", "Qwen/Qwen3.5-9B"),
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-27B",
        size="27B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3.5-27B"),
        config_type=HFQwen35Config,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-27B-MLX-4bit",
        size="27B",
        origin=HuggingFaceOrigin(repo="mlx-community/Qwen3.5-27B-4bit"),
        config_type=HFQwen35Config,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", "Qwen/Qwen3.5-27B"),
            tokenizer_config=FileSpec("tokenizer_config.json", "Qwen/Qwen3.5-27B"),
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-27B-MLX-8bit",
        size="27B",
        origin=HuggingFaceOrigin(repo="mlx-community/Qwen3.5-27B-8bit"),
        config_type=HFQwen35Config,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", "Qwen/Qwen3.5-27B"),
            tokenizer_config=FileSpec("tokenizer_config.json", "Qwen/Qwen3.5-27B"),
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
        ),
    ),
    ModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-35B-A3B",
        size="35B",
        quantization=None,
        repo="Qwen/Qwen3.5-35B-A3B",
        config_type=HFQwen35Config,
    ),
]

QWEN36 = [
    ModelSpec(
        vendor="Alibaba",
        family="Qwen3.6",
        name="Qwen3.6-35B-A3B",
        size="35B",
        quantization=None,
        repo="Qwen/Qwen3.6-35B-A3B",
        config_type=HFQwen35Config,
    ),
]

QWEN3_PARD = [
    LanguageModelSpec(
        vendor="AMD",
        family="PARD",
        name="PARD-Qwen3-0.6B",
        size="0.6B",
        origin=HuggingFaceOrigin(repo="amd/PARD-Qwen3-0.6B"),
        config_type=HFQwen3Config,
    ),
]

BONSAI = [
    LanguageModelSpec(
        vendor="PrismML",
        family="Bonsai",
        name="Bonsai-4B-mlx-1bit",
        size="4B",
        origin=HuggingFaceOrigin(repo="prism-ml/Bonsai-4B-mlx-1bit"),
        config_type=HFBonsaiConfig,
        configs=ConfigMap(
            chat_template=FileSpec("chat_template.jinja"),
        ),
    ),
]

QWEN_MODELS = QWEN25 + QWEN25_CODER + QWEN3 + QWEN3_NEXT + QWEN35 + QWEN3_PARD + BONSAI
