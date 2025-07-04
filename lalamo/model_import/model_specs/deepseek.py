from lalamo.model_import.configs import HFQwen2Config

from .common import (
    HUGGINFACE_GENERATION_CONFIG_FILE,
    HUGGINGFACE_TOKENIZER_FILES,
    ModelSpec,
    WeightsType,
    huggingface_weight_files,
)

__all__ = ["DEEPSEEK_MODELS"]

DEEPSEEK_MODELS = [
    ModelSpec(
        vendor="DeepSeek",
        family="R1-Distill-Qwen",
        name="R1-Distill-Qwen-1.5B",
        size="1.5B",
        quantization=None,
        repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(1),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
]
