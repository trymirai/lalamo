from fartsovka.model_import.configs import HFLlamaConfig

from .common import HUGGINGFACE_TOKENIZER_FILES, ModelSpec, WeightsType, huggingface_weight_files

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
        tokenizer_file_names=HUGGINGFACE_TOKENIZER_FILES,
    ),
]
