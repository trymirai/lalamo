from lalamo.model_import.model_configs import HFLlambaConfig
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin

__all__ = ["LLAMBA_MODELS"]


def _llama_tokenizer_repo(size: str) -> str:
    return {
        "1B": "meta-llama/Llama-3.2-1B-Instruct",
        "3B": "meta-llama/Llama-3.2-3B-Instruct",
        "8B": "meta-llama/Llama-3.1-8B-Instruct",
    }[size]


def _llamba_configs(size: str, quantization_bits: int | None) -> ConfigMap:
    tokenizer_repo = _llama_tokenizer_repo(size)
    model_config = (
        FileSpec("config.json", f"cartesia-ai/Llamba-{size}")
        if quantization_bits is not None
        else FileSpec("config.json")
    )
    return ConfigMap(
        model_config=model_config,
        tokenizer=FileSpec("tokenizer.json", tokenizer_repo),
        tokenizer_config=FileSpec("tokenizer_config.json", tokenizer_repo),
        generation_config=FileSpec("generation_config.json", tokenizer_repo),
    )


def _llamba_model_spec(size: str, suffix: str, quantization_bits: int | None) -> LanguageModelSpec:
    name = f"Llamba-{size}{suffix}"
    return LanguageModelSpec(
        vendor="Cartesia",
        family="Llamba",
        name=name,
        size=size,
        origin=HuggingFaceOrigin(repo=f"cartesia-ai/{name}"),
        config_type=HFLlambaConfig,
        configs=_llamba_configs(size, quantization_bits),
    )


LLAMBA_MODELS = [
    _llamba_model_spec("1B", "", None),
    _llamba_model_spec("1B", "-4bit-mlx", 4),
    _llamba_model_spec("1B", "-8bit-mlx", 8),
    _llamba_model_spec("3B", "", None),
    _llamba_model_spec("3B", "-4bit-mlx", 4),
    _llamba_model_spec("8B", "", None),
    _llamba_model_spec("8B", "-8bit-mlx", 8),
]
