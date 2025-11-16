from lalamo.model_import.decoder_configs import HFLlambaConfig
from lalamo.quantization import QuantizationMode

from .common import ConfigMap, FileSpec, ModelSpec

__all__ = ["LLAMBA_MODELS"]

LLAMBA_MODELS = [
    ModelSpec(
        vendor="Cartesia",
        family="Llamba",
        name="Llamba-1B",
        size="1B",
        quantization=None,
        repo="cartesia-ai/Llamba-1B",
        config_type=HFLlambaConfig,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json", "meta-llama/Llama-3.2-1B-Instruct"),
            tokenizer_config=FileSpec("tokenizer_config.json", "meta-llama/Llama-3.2-1B-Instruct"),
            generation_config=FileSpec("generation_config.json", "meta-llama/Llama-3.2-1B-Instruct"),
        ),
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Cartesia",
        family="Llamba",
        name="Llamba-1B-4bit-mlx",
        size="1B",
        quantization=QuantizationMode.UINT4,
        repo="cartesia-ai/Llamba-1B-4bit-mlx",
        config_type=HFLlambaConfig,
        configs=ConfigMap(
            model_config=FileSpec("config.json", "cartesia-ai/Llamba-1B"),
            tokenizer=FileSpec("tokenizer.json", "meta-llama/Llama-3.2-1B-Instruct"),
            tokenizer_config=FileSpec("tokenizer_config.json", "meta-llama/Llama-3.2-1B-Instruct"),
            generation_config=FileSpec("generation_config.json", "meta-llama/Llama-3.2-1B-Instruct"),
        ),
        use_cases=tuple(),
    ),
]
