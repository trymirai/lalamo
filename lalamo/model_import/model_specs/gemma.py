from lalamo.model_import.configs import HFGemma2Config, HFGemma3Config, HFGemma3TextConfig

from .common import (
    HUGGINFACE_GENERATION_CONFIG_FILE,
    HUGGINGFACE_TOKENIZER_FILES,
    ModelSpec,
    WeightsType,
    huggingface_weight_files,
)

__all__ = ["GEMMA_MODELS"]

GEMMA2 = [
    ModelSpec(
        vendor="Google",
        family="Gemma-2",
        name="Gemma-2-2B-Instruct",
        size="2B",
        quantization=None,
        repo="google/gemma-2-2b-it",
        config_type=HFGemma2Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(2),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
]

GEMMA3 = [
    ModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="Gemma-3-1B-Instruct",
        size="1B",
        quantization=None,
        repo="google/gemma-3-1b-it",
        config_type=HFGemma3TextConfig,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(1),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="Gemma-3-4B-Instruct",
        size="4B",
        quantization=None,
        repo="google/gemma-3-4b-it",
        config_type=HFGemma3Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(2),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="Gemma-3-27B-Instruct",
        size="27B",
        quantization=None,
        repo="google/gemma-3-27b-it",
        config_type=HFGemma3Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(12),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
]


GEMMA_MODELS = GEMMA2 + GEMMA3
