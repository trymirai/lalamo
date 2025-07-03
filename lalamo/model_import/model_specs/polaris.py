from lalamo.model_import.configs import HFQwen3Config

from .common import HUGGINGFACE_TOKENIZER_FILES, ModelSpec, TokenizerFileSpec, WeightsType, huggingface_weight_files

__all__ = ["POLARIS_MODELS"]

POLARIS_MODELS = [
    ModelSpec(
        vendor="POLARIS-Project",
        family="Polaris-Preview",
        name="Polaris-4B-Preview",
        size="4B",
        quantization=None,
        repo="POLARIS-Project/Polaris-4B-Preview",
        config_type=HFQwen3Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(2),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, TokenizerFileSpec(repo=None, filename="chat_template.jinja")),
        use_cases=tuple(),
    ),
]
