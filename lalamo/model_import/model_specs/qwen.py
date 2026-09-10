from frozendict import frozendict

from lalamo.model_import.model_configs import (
    HFQwen2Config,
    HFQwen3Config,
    HFQwen35Config,
)
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.model_specs.output_parser_regexes import OPTIONAL_THINKING_OUTPUT_PARSER_REGEX
from lalamo.model_import.model_specs.reasoning_configs import (
    BOOLEAN_REASONING_DEFAULT_OFF_CONFIG,
    BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
)
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.models.chat_codec import ReasoningConfig, ReasoningEffort
from lalamo.models.language_model import GenerationConfig

__all__ = ["QWEN_MODELS"]

QWEN_END_OF_THINKING_TAG = "\n</think>"

# The MLX build of this model carries neither a tokenizer pair nor a generation config of its own.
QWEN36_A3B_REPO = "Qwen/Qwen3.6-35B-A3B"

QWEN38_REASONING_CONFIG = ReasoningConfig(
    default_reasoning_effort=ReasoningEffort.XHIGH,
    field_name="reasoning_effort",
    reasoning_effort_to_field_value=frozendict(
        {
            ReasoningEffort.XHIGH: "xhigh",
            ReasoningEffort.MEDIUM: "medium",
            ReasoningEffort.LOW: "low",
        }
    ),
)


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
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
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
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
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
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
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
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
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
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-8B",
        size="8B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3-8B"),
        config_type=HFQwen3Config,
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
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
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
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
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
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
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_OFF_CONFIG,
        configs=ConfigMap(
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
            generation_params_overrides=GenerationConfig(
                temperature=0.8,
                top_k=40,
                repetition_penalty=1.15,
                suffix_repetition_length=1024,
            ),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-0.8B-MLX-4bit",
        size="0.8B",
        origin=HuggingFaceOrigin(repo="mlx-community/Qwen3.5-0.8B-MLX-4bit"),
        config_type=HFQwen35Config,
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_OFF_CONFIG,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", "Qwen/Qwen3.5-0.8B"),
            tokenizer_config=FileSpec("tokenizer_config.json", "Qwen/Qwen3.5-0.8B"),
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
            generation_params_overrides=GenerationConfig(
                temperature=0.8,
                top_k=40,
                repetition_penalty=1.15,
                suffix_repetition_length=1024,
            ),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-0.8B-MLX-8bit",
        size="0.8B",
        origin=HuggingFaceOrigin(repo="mlx-community/Qwen3.5-0.8B-MLX-8bit"),
        config_type=HFQwen35Config,
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_OFF_CONFIG,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", "Qwen/Qwen3.5-0.8B"),
            tokenizer_config=FileSpec("tokenizer_config.json", "Qwen/Qwen3.5-0.8B"),
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
            generation_params_overrides=GenerationConfig(
                temperature=0.8,
                top_k=40,
                repetition_penalty=1.15,
                suffix_repetition_length=1024,
            ),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-2B",
        size="2B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3.5-2B"),
        config_type=HFQwen35Config,
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_OFF_CONFIG,
        configs=ConfigMap(
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
            generation_params_overrides=GenerationConfig(
                top_k=40,
                repetition_penalty=1.15,
                suffix_repetition_length=1024,
            ),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-2B-MLX-4bit",
        size="2B",
        origin=HuggingFaceOrigin(repo="mlx-community/Qwen3.5-2B-MLX-4bit"),
        config_type=HFQwen35Config,
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_OFF_CONFIG,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", "Qwen/Qwen3.5-2B"),
            tokenizer_config=FileSpec("tokenizer_config.json", "Qwen/Qwen3.5-2B"),
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
            generation_params_overrides=GenerationConfig(
                temperature=0.8,
                top_k=40,
                repetition_penalty=1.15,
                suffix_repetition_length=1024,
            ),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-2B-MLX-8bit",
        size="2B",
        origin=HuggingFaceOrigin(repo="mlx-community/Qwen3.5-2B-MLX-8bit"),
        config_type=HFQwen35Config,
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_OFF_CONFIG,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", "Qwen/Qwen3.5-2B"),
            tokenizer_config=FileSpec("tokenizer_config.json", "Qwen/Qwen3.5-2B"),
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
            generation_params_overrides=GenerationConfig(
                top_k=40,
                repetition_penalty=1.15,
                suffix_repetition_length=1024,
            ),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-4B",
        size="4B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3.5-4B"),
        config_type=HFQwen35Config,
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
        configs=ConfigMap(
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
            generation_params_overrides=GenerationConfig(
                top_k=40,
                repetition_penalty=1.10,
                suffix_repetition_length=1024,
            ),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-4B-MLX-4bit",
        size="4B",
        origin=HuggingFaceOrigin(repo="mlx-community/Qwen3.5-4B-MLX-4bit"),
        config_type=HFQwen35Config,
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", "Qwen/Qwen3.5-4B"),
            tokenizer_config=FileSpec("tokenizer_config.json", "Qwen/Qwen3.5-4B"),
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
            generation_params_overrides=GenerationConfig(
                top_k=40,
                repetition_penalty=1.10,
                suffix_repetition_length=1024,
            ),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-4B-MLX-8bit",
        size="4B",
        origin=HuggingFaceOrigin(repo="mlx-community/Qwen3.5-4B-MLX-8bit"),
        config_type=HFQwen35Config,
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", "Qwen/Qwen3.5-4B"),
            tokenizer_config=FileSpec("tokenizer_config.json", "Qwen/Qwen3.5-4B"),
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
            generation_params_overrides=GenerationConfig(
                top_k=40,
                repetition_penalty=1.10,
                suffix_repetition_length=1024,
            ),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-9B",
        size="9B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3.5-9B"),
        config_type=HFQwen35Config,
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
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
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
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
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
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
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.5",
        name="Qwen3.5-27B-MLX-4bit",
        size="27B",
        origin=HuggingFaceOrigin(repo="mlx-community/Qwen3.5-27B-4bit"),
        config_type=HFQwen35Config,
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
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
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
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
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
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
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.6",
        name="Qwen3.6-35B-A3B",
        size="35B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3.6-35B-A3B"),
        config_type=HFQwen35Config,
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
        configs=ConfigMap(
            # The model card recommends 1.5, but generation_config.json omits presence_penalty.
            generation_params_overrides=GenerationConfig(presence_penalty=1.5),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.6",
        name="Qwen3.6-35B-A3B-UD-MLX-4bit",
        size="35B",
        origin=HuggingFaceOrigin(repo="unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit"),
        # Same config type as the unquantized model: it already handles `qwen3_5_moe`, and it reads
        # the `quantization` section to drop the RMSNorm offset MLX bakes into the weights.
        config_type=HFQwen35Config,
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
        configs=ConfigMap(
            # The MLX repo ships no generation_config.json at all, so sampling parameters -- and with
            # them the deployed sampler -- have to come from the source repo.
            tokenizer=FileSpec("tokenizer.json", QWEN36_A3B_REPO),
            tokenizer_config=FileSpec("tokenizer_config.json", QWEN36_A3B_REPO),
            generation_config=FileSpec("generation_config.json", QWEN36_A3B_REPO),
            generation_params_overrides=GenerationConfig(presence_penalty=1.5),
        ),
    ),
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.6",
        name="Qwen3.6-35B-A3B-MLX-8bit",
        size="35B",
        # Same vendor as the UD-MLX-4bit build above, so the two differ in bit width and nothing else.
        # Uniform 8 bits with group 64 across all 512 quantized tensors, the router included -- the
        # near-lossless counterpart used to separate the cost of quantization noise in the experts from
        # the cost of loading a quantized checkpoint at all.
        origin=HuggingFaceOrigin(repo="unsloth/Qwen3.6-35B-A3B-MLX-8bit"),
        config_type=HFQwen35Config,
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
        configs=ConfigMap(
            # This repo keeps the chat template in a separate chat_template.jinja and ships no
            # sampling parameters, so both come from the source repo, as for the 4-bit build.
            tokenizer=FileSpec("tokenizer.json", QWEN36_A3B_REPO),
            tokenizer_config=FileSpec("tokenizer_config.json", QWEN36_A3B_REPO),
            generation_config=FileSpec("generation_config.json", QWEN36_A3B_REPO),
            generation_params_overrides=GenerationConfig(presence_penalty=1.5),
        ),
    ),
]

QWEN38 = [
    LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.8",
        name="Qwen3.8-27B",
        size="27B",
        origin=HuggingFaceOrigin(repo="Qwen/Qwen3.8-27B"),
        config_type=HFQwen35Config,
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=QWEN38_REASONING_CONFIG,
    ),
]


QWEN_MODELS = QWEN25_CODER + QWEN3 + QWEN35 + QWEN36 + QWEN38
