from lalamo.model_import.configs import HFLlamaConfig

from .common import (
    HUGGINFACE_GENERATION_CONFIG_FILE,
    HUGGINGFACE_TOKENIZER_FILES,
    ModelSpec,
    WeightsType,
    huggingface_weight_files,
)

__all__ = ["REKA_MODELS"]

REKA_MODELS = [
    ModelSpec(
        vendor="Reka",
        family="Reka-Flash",
        name="Reka-Flash-3.1",
        size="21B",
        quantization=None,
        repo="RekaAI/reka-flash-3.1",
        config_type=HFLlamaConfig,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(9),  # Model has 9 shards
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
] 