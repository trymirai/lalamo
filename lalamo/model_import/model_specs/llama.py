from lalamo.model_import.decoder_configs import HFLlamaConfig
from lalamo.quantization import QuantizationMode

from .common import ConfigMap, FileSpec, ModelSpec

__all__ = ["LLAMA_MODELS"]

LLAMA31 = [
    ModelSpec(
        vendor="Meta",
        family="Llama-3.1",
        name="Llama-3.1-8B-Instruct",
        size="8B",
        quantization=None,
        repo="meta-llama/Llama-3.1-8B-Instruct",
        config_type=HFLlamaConfig,
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Meta",
        family="Llama-3.1",
        name="Llama-3.1-8B-Instruct-4bit",
        size="8B",
        quantization=QuantizationMode.UINT4,
        repo="mlx-community/Llama-3.1-8B-Instruct-4bit",
        config_type=HFLlamaConfig,
        use_cases=tuple(),
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "meta-llama/Llama-3.1-8B-Instruct")),
    ),
]


LLAMA32 = [
    ModelSpec(
        vendor="Meta",
        family="Llama-3.2",
        name="Llama-3.2-1B-Instruct",
        size="1B",
        quantization=None,
        repo="meta-llama/Llama-3.2-1B-Instruct",
        config_type=HFLlamaConfig,
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Meta",
        family="Llama-3.2",
        name="Llama-3.2-1B-Instruct-4bit",
        size="1B",
        quantization=QuantizationMode.UINT4,
        repo="mlx-community/Llama-3.2-1B-Instruct-4bit",
        config_type=HFLlamaConfig,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "meta-llama/Llama-3.2-1B-Instruct")),
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Meta",
        family="Llama-3.2",
        name="Llama-3.2-1B-Instruct-8bit",
        size="1B",
        quantization=QuantizationMode.UINT8,
        repo="mlx-community/Llama-3.2-1B-Instruct-8bit",
        config_type=HFLlamaConfig,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "meta-llama/Llama-3.2-1B-Instruct")),
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Meta",
        family="Llama-3.2",
        name="Llama-3.2-3B-Instruct",
        size="3B",
        quantization=None,
        repo="meta-llama/Llama-3.2-3B-Instruct",
        config_type=HFLlamaConfig,
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Meta",
        family="Llama-3.2",
        name="Llama-3.2-3B-Instruct-4bit",
        size="3B",
        quantization=QuantizationMode.UINT4,
        repo="mlx-community/Llama-3.2-3B-Instruct-4bit",
        config_type=HFLlamaConfig,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "meta-llama/Llama-3.2-3B-Instruct")),
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Meta",
        family="Llama-3.2",
        name="Llama-3.2-3B-Instruct-8bit",
        size="3B",
        quantization=QuantizationMode.UINT8,
        repo="mlx-community/Llama-3.2-3B-Instruct-8bit",
        config_type=HFLlamaConfig,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "meta-llama/Llama-3.2-3B-Instruct")),
        use_cases=tuple(),
    ),
]

LLAMA_MODELS = LLAMA31 + LLAMA32
