from lalamo.model_import.configs import HFQwen2Config, HFQwen3Config
from lalamo.quantization import QuantizationMode

from .common import (
    HUGGINFACE_GENERATION_CONFIG_FILE,
    HUGGINGFACE_TOKENIZER_FILES,
    ModelSpec,
    UseCase,
    WeightsType,
    huggingface_weight_files,
)

__all__ = ["QWEN_MODELS"]


QWEN25 = [
    ModelSpec(
        vendor="Alibaba",
        family="Qwen2.5",
        name="Qwen2.5-0.5B-Instruct",
        size="0.5B",
        quantization=None,
        repo="Qwen/Qwen2.5-0.5B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(1),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Alibaba",
        family="Qwen2.5",
        name="Qwen2.5-1.5B-Instruct",
        size="1.5B",
        quantization=None,
        repo="Qwen/Qwen2.5-1.5B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(1),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Alibaba",
        family="Qwen2.5",
        name="Qwen2.5-3B-Instruct",
        size="3B",
        quantization=None,
        repo="Qwen/Qwen2.5-3B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(2),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Alibaba",
        family="Qwen2.5",
        name="Qwen2.5-7B-Instruct",
        size="7B",
        quantization=None,
        repo="Qwen/Qwen2.5-7B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(4),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Alibaba",
        family="Qwen2.5",
        name="Qwen2.5-14B-Instruct",
        size="14B",
        quantization=None,
        repo="Qwen/Qwen2.5-14B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(8),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Alibaba",
        family="Qwen2.5",
        name="Qwen2.5-32B-Instruct",
        size="32B",
        quantization=None,
        repo="Qwen/Qwen2.5-32B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(17),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
]


QWEN25_CODER = [
    ModelSpec(
        vendor="Alibaba",
        family="Qwen2.5-Coder",
        name="Qwen2.5-Coder-0.5B-Instruct",
        size="0.5B",
        quantization=None,
        repo="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(1),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=(UseCase.CODE,),
    ),
    ModelSpec(
        vendor="Alibaba",
        family="Qwen2.5-Coder",
        name="Qwen2.5-Coder-1.5B-Instruct",
        size="1.5B",
        quantization=None,
        repo="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(1),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=(UseCase.CODE,),
    ),
    ModelSpec(
        vendor="Alibaba",
        family="Qwen2.5-Coder",
        name="Qwen2.5-Coder-3B-Instruct",
        size="3B",
        quantization=None,
        repo="Qwen/Qwen2.5-Coder-3B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(2),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=(UseCase.CODE,),
    ),
    ModelSpec(
        vendor="Alibaba",
        family="Qwen2.5-Coder",
        name="Qwen2.5-Coder-7B-Instruct",
        size="7B",
        quantization=None,
        repo="Qwen/Qwen2.5-Coder-7B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(4),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=(UseCase.CODE,),
    ),
    ModelSpec(
        vendor="Alibaba",
        family="Qwen2.5-Coder",
        name="Qwen2.5-Coder-14B-Instruct",
        size="14B",
        quantization=None,
        repo="Qwen/Qwen2.5-Coder-14B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(6),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=(UseCase.CODE,),
    ),
    ModelSpec(
        vendor="Alibaba",
        family="Qwen2.5-Coder",
        name="Qwen2.5-Coder-32B-Instruct",
        size="32B",
        quantization=None,
        repo="Qwen/Qwen2.5-Coder-32B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(14),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=(UseCase.CODE,),
    ),
]


QWEN3 = [
    ModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-0.6B",
        size="0.6B",
        quantization=None,
        repo="Qwen/Qwen3-0.6B",
        config_type=HFQwen3Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(1),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-1.7B",
        size="1.7B",
        quantization=None,
        repo="Qwen/Qwen3-1.7B",
        config_type=HFQwen3Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(2),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-4B",
        size="4B",
        quantization=None,
        repo="Qwen/Qwen3-4B",
        config_type=HFQwen3Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(3),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-4B-AWQ",
        size="4B",
        quantization=QuantizationMode.UINT4,
        repo="Qwen/Qwen3-4B-AWQ",
        config_type=HFQwen3Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(1),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-8B",
        size="8B",
        quantization=None,
        repo="Qwen/Qwen3-8B",
        config_type=HFQwen3Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(5),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-8B-AWQ",
        size="8B",
        quantization=QuantizationMode.UINT4,
        repo="Qwen/Qwen3-8B-AWQ",
        config_type=HFQwen3Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(2),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-14B",
        size="14B",
        quantization=None,
        repo="Qwen/Qwen3-14B",
        config_type=HFQwen3Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(8),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-14B-AWQ",
        size="14B",
        quantization=None,
        repo="Qwen/Qwen3-14B-AWQ",
        config_type=HFQwen3Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(2),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-32B",
        size="32B",
        quantization=None,
        repo="Qwen/Qwen3-32B",
        config_type=HFQwen3Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(17),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="Alibaba",
        family="Qwen3",
        name="Qwen3-32B-AWQ",
        size="32B",
        quantization=QuantizationMode.UINT4,
        repo="Qwen/Qwen3-32B-AWQ",
        config_type=HFQwen3Config,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(4),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
]


QWEN_MODELS = QWEN25 + QWEN25_CODER + QWEN3
