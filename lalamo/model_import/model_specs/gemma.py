from lalamo.model_import.decoder_configs import HFGemma2Config, HFGemma3Config, HFGemma3TextConfig

from .common import ModelSpec, WeightsType

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
        weights_type=WeightsType.SAFETENSORS,
    ),
    ModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="Gemma-3-4B-Instruct",
        size="4B",
        quantization=None,
        repo="google/gemma-3-4b-it",
        config_type=HFGemma3Config,
        weights_type=WeightsType.SAFETENSORS,
    ),
    ModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="Gemma-3-27B-Instruct",
        size="27B",
        quantization=None,
        repo="google/gemma-3-27b-it",
        config_type=HFGemma3Config,
        weights_type=WeightsType.SAFETENSORS,
    ),
]


GEMMA_MODELS = GEMMA2 + GEMMA3
