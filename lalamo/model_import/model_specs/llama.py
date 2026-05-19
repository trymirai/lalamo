from lalamo.model_import.model_configs import HFLlamaConfig
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin

__all__ = ["LLAMA_MODELS"]

LLAMA31 = [
    LanguageModelSpec(
        vendor="Meta",
        family="Llama-3.1",
        name="Llama-3.1-8B-Instruct",
        size="8B",
        origin=HuggingFaceOrigin(repo="meta-llama/Llama-3.1-8B-Instruct"),
        config_type=HFLlamaConfig,
    ),
    LanguageModelSpec(
        vendor="Meta",
        family="Llama-3.1",
        name="Llama-3.1-8B-Instruct-4bit",
        size="8B",
        origin=HuggingFaceOrigin(repo="mlx-community/Llama-3.1-8B-Instruct-4bit"),
        config_type=HFLlamaConfig,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "meta-llama/Llama-3.1-8B-Instruct")),
    ),
]


LLAMA32 = [
    LanguageModelSpec(
        vendor="Meta",
        family="Llama-3.2",
        name="Llama-3.2-1B-Instruct",
        size="1B",
        origin=HuggingFaceOrigin(repo="meta-llama/Llama-3.2-1B-Instruct"),
        config_type=HFLlamaConfig,
    ),
    LanguageModelSpec(
        vendor="Meta",
        family="Llama-3.2",
        name="Llama-3.2-1B-Instruct-4bit",
        size="1B",
        origin=HuggingFaceOrigin(repo="mlx-community/Llama-3.2-1B-Instruct-4bit"),
        config_type=HFLlamaConfig,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "meta-llama/Llama-3.2-1B-Instruct")),
    ),
    LanguageModelSpec(
        vendor="Meta",
        family="Llama-3.2",
        name="Llama-3.2-1B-Instruct-8bit",
        size="1B",
        origin=HuggingFaceOrigin(repo="mlx-community/Llama-3.2-1B-Instruct-8bit"),
        config_type=HFLlamaConfig,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "meta-llama/Llama-3.2-1B-Instruct")),
    ),
    LanguageModelSpec(
        vendor="Meta",
        family="Llama-3.2",
        name="Llama-3.2-3B-Instruct",
        size="3B",
        origin=HuggingFaceOrigin(repo="meta-llama/Llama-3.2-3B-Instruct"),
        config_type=HFLlamaConfig,
    ),
    LanguageModelSpec(
        vendor="Meta",
        family="Llama-3.2",
        name="Llama-3.2-3B-Instruct-AWQ",
        size="3B",
        origin=HuggingFaceOrigin(repo="trymirai/Llama-3.2-3B-Instruct-AWQ"),
        config_type=HFLlamaConfig,
    ),
    LanguageModelSpec(
        vendor="Meta",
        family="Llama-3.2",
        name="Llama-3.2-3B-Instruct-4bit",
        size="3B",
        origin=HuggingFaceOrigin(repo="mlx-community/Llama-3.2-3B-Instruct-4bit"),
        config_type=HFLlamaConfig,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "meta-llama/Llama-3.2-3B-Instruct")),
    ),
    LanguageModelSpec(
        vendor="Meta",
        family="Llama-3.2",
        name="Llama-3.2-3B-Instruct-8bit",
        size="3B",
        origin=HuggingFaceOrigin(repo="mlx-community/Llama-3.2-3B-Instruct-8bit"),
        config_type=HFLlamaConfig,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "meta-llama/Llama-3.2-3B-Instruct")),
    ),
]

LLAMA_MODELS = LLAMA31 + LLAMA32
