from lalamo.model_import.model_configs import HFLlambaConfig
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin

__all__ = ["LLAMBA_MODELS"]

LLAMBA_MODELS = [
    LanguageModelSpec(
        vendor="Cartesia",
        family="Llamba",
        name="Llamba-1B",
        size="1B",
        origin=HuggingFaceOrigin(repo="cartesia-ai/Llamba-1B"),
        config_type=HFLlambaConfig,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", "meta-llama/Llama-3.2-1B-Instruct"),
            tokenizer_config=FileSpec("tokenizer_config.json", "meta-llama/Llama-3.2-1B-Instruct"),
            generation_config=FileSpec("generation_config.json", "meta-llama/Llama-3.2-1B-Instruct"),
        ),
    ),
    LanguageModelSpec(
        vendor="Cartesia",
        family="Llamba",
        name="Llamba-1B-4bit-mlx",
        size="1B",
        origin=HuggingFaceOrigin(repo="cartesia-ai/Llamba-1B-4bit-mlx"),
        config_type=HFLlambaConfig,
        configs=ConfigMap(
            model_config=FileSpec("config.json", "cartesia-ai/Llamba-1B"),
            tokenizer=FileSpec("tokenizer.json", "meta-llama/Llama-3.2-1B-Instruct"),
            tokenizer_config=FileSpec("tokenizer_config.json", "meta-llama/Llama-3.2-1B-Instruct"),
            generation_config=FileSpec("generation_config.json", "meta-llama/Llama-3.2-1B-Instruct"),
        ),
    ),
]
