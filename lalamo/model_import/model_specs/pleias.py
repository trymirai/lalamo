from lalamo.model_import.configs import HFLlamaConfig

from .common import (
    HUGGINFACE_GENERATION_CONFIG_FILE,
    HUGGINGFACE_TOKENIZER_FILES,
    ModelSpec,
    WeightsType,
    huggingface_weight_files,
)

__all__ = ["PLEIAS_MODELS"]

PLEIAS_MODELS = [
    ModelSpec(
        vendor="PleIAs",
        family="Pleias-RAG",
        name="Pleias-RAG-1B",
        size="1B",
        quantization=None,
        repo="PleIAs/Pleias-RAG-1B",
        config_type=HFLlamaConfig,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(1),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
]
