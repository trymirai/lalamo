from lalamo.model_import.model_configs import (
    HFQwen2Config,
    HFQwen3Config,
    HFQwen35Config,
)
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin

__all__ = ["QWEN_MODELS"]

QWEN_THINKING_OUTPUT_PARSER_REGEX = r"(?s)(?:<think>)?(?P<chain_of_thought>.*?)(?:</think>\s*(?P<response>.*))?\Z"


def _qwen3_mlx_model_spec(
    *,
    name: str,
    size: str,
    base_repo: str,
) -> LanguageModelSpec:
    return LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name=name,
        size=size,
        origin=HuggingFaceOrigin(repo=f"Qwen/{name}"),
        config_type=HFQwen3Config,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", base_repo),
            tokenizer_config=FileSpec("tokenizer_config.json", base_repo),
            generation_config=FileSpec("generation_config.json", base_repo),
        ),
    )


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
    _qwen3_mlx_model_spec(
        name="Qwen3-0.6B-MLX-4bit",
        size="0.6B",
        base_repo="Qwen/Qwen3-0.6B",
    ),
    _qwen3_mlx_model_spec(
        name="Qwen3-0.6B-MLX-8bit",
        size="0.6B",
        base_repo="Qwen/Qwen3-0.6B",
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-1.7B",
        size="1.7B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3-1.7B"),
        config_type=HFQwen3Config,
    ),
    _qwen3_mlx_model_spec(
        name="Qwen3-1.7B-MLX-4bit",
        size="1.7B",
        base_repo="Qwen/Qwen3-1.7B",
    ),
    _qwen3_mlx_model_spec(
        name="Qwen3-1.7B-MLX-8bit",
        size="1.7B",
        base_repo="Qwen/Qwen3-1.7B",
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
        name="Qwen3-4B-Instruct",
        size="4B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3-4B-Instruct-2507"),
        config_type=HFQwen3Config,
    ),
    _qwen3_mlx_model_spec(
        name="Qwen3-4B-MLX-4bit",
        size="4B",
        base_repo="Qwen/Qwen3-4B",
    ),
    _qwen3_mlx_model_spec(
        name="Qwen3-4B-MLX-8bit",
        size="4B",
        base_repo="Qwen/Qwen3-4B",
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-4B-Thinking",
        size="4B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3-4B-Thinking-2507"),
        config_type=HFQwen3Config,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-8B",
        size="8B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3-8B"),
        config_type=HFQwen3Config,
    ),
    _qwen3_mlx_model_spec(
        name="Qwen3-8B-MLX-4bit",
        size="8B",
        base_repo="Qwen/Qwen3-8B",
    ),
    _qwen3_mlx_model_spec(
        name="Qwen3-8B-MLX-8bit",
        size="8B",
        base_repo="Qwen/Qwen3-8B",
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-14B",
        size="14B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3-14B"),
        config_type=HFQwen3Config,
    ),
    _qwen3_mlx_model_spec(
        name="Qwen3-14B-MLX-4bit",
        size="14B",
        base_repo="Qwen/Qwen3-14B",
    ),
    _qwen3_mlx_model_spec(
        name="Qwen3-14B-MLX-8bit",
        size="14B",
        base_repo="Qwen/Qwen3-14B",
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-32B",
        size="32B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3-32B"),
        config_type=HFQwen3Config,
    ),
    _qwen3_mlx_model_spec(
        name="Qwen3-32B-MLX-4bit",
        size="32B",
        base_repo="Qwen/Qwen3-32B",
    ),
    _qwen3_mlx_model_spec(
        name="Qwen3-32B-MLX-8bit",
        size="32B",
        base_repo="Qwen/Qwen3-32B",
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
        output_parser_regex=QWEN_THINKING_OUTPUT_PARSER_REGEX,
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
        output_parser_regex=QWEN_THINKING_OUTPUT_PARSER_REGEX,
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
        output_parser_regex=QWEN_THINKING_OUTPUT_PARSER_REGEX,
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
        output_parser_regex=QWEN_THINKING_OUTPUT_PARSER_REGEX,
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
        output_parser_regex=QWEN_THINKING_OUTPUT_PARSER_REGEX,
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
        output_parser_regex=QWEN_THINKING_OUTPUT_PARSER_REGEX,
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
        output_parser_regex=QWEN_THINKING_OUTPUT_PARSER_REGEX,
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
        output_parser_regex=QWEN_THINKING_OUTPUT_PARSER_REGEX,
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
        output_parser_regex=QWEN_THINKING_OUTPUT_PARSER_REGEX,
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
        output_parser_regex=QWEN_THINKING_OUTPUT_PARSER_REGEX,
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
        output_parser_regex=QWEN_THINKING_OUTPUT_PARSER_REGEX,
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
        output_parser_regex=QWEN_THINKING_OUTPUT_PARSER_REGEX,
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
        output_parser_regex=QWEN_THINKING_OUTPUT_PARSER_REGEX,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-27B-MLX-4bit",
        size="27B",
        origin=HuggingFaceOrigin(repo="mlx-community/Qwen3.5-27B-4bit"),
        config_type=HFQwen35Config,
        output_parser_regex=QWEN_THINKING_OUTPUT_PARSER_REGEX,
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
        output_parser_regex=QWEN_THINKING_OUTPUT_PARSER_REGEX,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", "Qwen/Qwen3.5-27B"),
            tokenizer_config=FileSpec("tokenizer_config.json", "Qwen/Qwen3.5-27B"),
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-35B-A3B",
        size="35B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3.5-35B-A3B"),
        config_type=HFQwen35Config,
        output_parser_regex=QWEN_THINKING_OUTPUT_PARSER_REGEX,
    ),
]

QWEN36 = [
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.6",
        name="Qwen3.6-27B",
        size="27B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3.6-27B"),
        config_type=HFQwen35Config,
        output_parser_regex=QWEN_THINKING_OUTPUT_PARSER_REGEX,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.6",
        name="Qwen3.6-35B-A3B",
        size="35B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3.6-35B-A3B"),
        config_type=HFQwen35Config,
        output_parser_regex=QWEN_THINKING_OUTPUT_PARSER_REGEX,
    ),
]


QWEN_MODELS = QWEN25_CODER + QWEN3 + QWEN35 + QWEN36
