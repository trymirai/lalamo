from lalamo.model_import.configs import HFLlamaConfig

from .common import (
    HUGGINFACE_GENERATION_CONFIG_FILE,
    HUGGINGFACE_TOKENIZER_FILES,
    ModelSpec,
    WeightsType,
    huggingface_weight_files,
)

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
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(1),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
]
