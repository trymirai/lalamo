from lalamo.model_import.configs import HFMistralConfig

from .common import HUGGINGFACE_TOKENIZER_FILES, ModelSpec, WeightsType, huggingface_weight_files

__all__ = ["MISTRAL_MODELS"]

CODESTRAL = [
    ModelSpec(
        vendor="Mistral",
        family="Codestral",
        name="Codestral-22B-v0.1",
        size="22B",
        quantization=None,
        repo="mistral-community/Codestral-22B-v0.1",
        config_type=HFMistralConfig,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(9),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_file_names=HUGGINGFACE_TOKENIZER_FILES,
    ),
]

DEVSTRAL = [
    ModelSpec(
        vendor="Mistral",
        family="Devstral",
        name="Devstral-Small-2505",
        size="24B",
        quantization=None,
        repo="mistralai/Devstral-Small-2505",
        config_type=HFMistralConfig,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(10),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_file_names=tuple(),
    ),
]


MISTRAL_MODELS = CODESTRAL + DEVSTRAL
