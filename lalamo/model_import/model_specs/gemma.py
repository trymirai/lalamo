from lalamo.model_import.decoder_configs import (
    HFGemma2Config,
    HFGemma3Config,
    HFGemma3TextConfig,
)
from lalamo.quantization import QuantizationMode

from .common import ConfigMap, FileSpec, ModelSpec, WeightsType

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
        name="Gemma-3-1B-Instruct-4bit",
        size="1B",
        quantization=QuantizationMode.UINT4,
        repo="mlx-community/gemma-3-1b-it-4bit",
        config_type=HFGemma3TextConfig,
        weights_type=WeightsType.SAFETENSORS,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "google/gemma-3-1b-it")),
    ),
    ModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="Gemma-3-1B-Instruct-8bit",
        size="1B",
        quantization=QuantizationMode.UINT8,
        repo="mlx-community/gemma-3-1b-it-8bit",
        config_type=HFGemma3TextConfig,
        weights_type=WeightsType.SAFETENSORS,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "google/gemma-3-1b-it")),
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
        name="Gemma-3-4B-Instruct-4bit",
        size="4B",
        quantization=QuantizationMode.UINT4,
        repo="mlx-community/gemma-3-4b-it-4bit",
        config_type=HFGemma3Config,
        weights_type=WeightsType.SAFETENSORS,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "google/gemma-3-4b-it")),
    ),
    ModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="Gemma-3-4B-Instruct-8bit",
        size="4B",
        quantization=QuantizationMode.UINT8,
        repo="mlx-community/gemma-3-4b-it-8bit",
        config_type=HFGemma3Config,
        weights_type=WeightsType.SAFETENSORS,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "google/gemma-3-4b-it")),
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
    ModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="Gemma-3-27B-Instruct-4bit",
        size="27B",
        quantization=QuantizationMode.UINT4,
        repo="mlx-community/gemma-3-27b-it-4bit",
        config_type=HFGemma3Config,
        weights_type=WeightsType.SAFETENSORS,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "google/gemma-3-27b-it")),
    ),
    ModelSpec(
        vendor="Google",
        family="Gemma-3",
        name="Gemma-3-27B-Instruct-8bit",
        size="27B",
        quantization=QuantizationMode.UINT8,
        repo="mlx-community/gemma-3-27b-it-8bit",
        config_type=HFGemma3Config,
        weights_type=WeightsType.SAFETENSORS,
        configs=ConfigMap(generation_config=FileSpec("generation_config.json", "google/gemma-3-27b-it")),
    ),
]


GEMMA_MODELS = GEMMA2 + GEMMA3
