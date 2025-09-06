from lalamo.model_import.decoder_configs import HFLlamaConfig

from .common import ModelSpec

__all__ = ["HUGGINGFACE_MODELS"]

HUGGINGFACE_MODELS = [
    ModelSpec(
        vendor="HuggingFace",
        family="SmolLM2",
        name="SmolLM2-1.7B-Instruct",
        size="1.7B",
        quantization=None,
        repo="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        config_type=HFLlamaConfig,
        use_cases=tuple(),
    ),
]
