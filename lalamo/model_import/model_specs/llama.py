from lalamo.model_import.decoder_configs import HFLlamaConfig

from .common import ModelSpec

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
        name="Llama-3.2-3B-Instruct",
        size="3B",
        quantization=None,
        repo="meta-llama/Llama-3.2-3B-Instruct",
        config_type=HFLlamaConfig,
        use_cases=tuple(),
    ),
]

LLAMA_MODELS = LLAMA31 + LLAMA32
