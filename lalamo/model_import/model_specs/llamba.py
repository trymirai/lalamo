from lalamo.model_import.model_configs import HFLlambaConfig
from lalamo.quantization import QuantizationMode

from .common import ConfigMap, FileSpec, ModelSpec

__all__ = ["LLAMBA_MODELS"]


def _llama_tokenizer_repo(size: str) -> str:
    return {
        "1B": "meta-llama/Llama-3.2-1B-Instruct",
        "3B": "meta-llama/Llama-3.2-3B-Instruct",
        "8B": "meta-llama/Llama-3.1-8B-Instruct",
    }[size]


def _llamba_configs(size: str, quantization: QuantizationMode | None) -> ConfigMap:
    tokenizer_repo = _llama_tokenizer_repo(size)
    model_config = (
        FileSpec("config.json", f"cartesia-ai/Llamba-{size}") if quantization is not None else FileSpec("config.json")
    )
    return ConfigMap(
        model_config=model_config,
        tokenizer=FileSpec("tokenizer.json", tokenizer_repo),
        tokenizer_config=FileSpec("tokenizer_config.json", tokenizer_repo),
        generation_config=FileSpec("generation_config.json", tokenizer_repo),
    )


def _llamba_model_spec(size: str, suffix: str, quantization: QuantizationMode | None) -> ModelSpec:
    name = f"Llamba-{size}{suffix}"
    return ModelSpec(
        vendor="Cartesia",
        family="Llamba",
        name=name,
        size=size,
        quantization=quantization,
        repo=f"cartesia-ai/{name}",
        config_type=HFLlambaConfig,
        configs=_llamba_configs(size, quantization),
        use_cases=tuple(),
    )


LLAMBA_MODELS = [
    _llamba_model_spec("1B", "", None),
    _llamba_model_spec("1B", "-4bit-mlx", QuantizationMode.UINT4),
    _llamba_model_spec("1B", "-8bit-mlx", QuantizationMode.UINT8),
    _llamba_model_spec("3B", "", None),
    _llamba_model_spec("3B", "-4bit-mlx", QuantizationMode.UINT4),
    _llamba_model_spec("8B", "", None),
    _llamba_model_spec("8B", "-8bit-mlx", QuantizationMode.UINT8),
]
