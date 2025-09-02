from dataclasses import replace

from lalamo.model_import.configs import HFLlamaConfig

from .common import (
    HUGGINFACE_GENERATION_CONFIG_FILE,
    HUGGINGFACE_TOKENIZER_FILES,
    ModelSpec,
    TokenizerFileSpec,
    WeightsType,
    huggingface_weight_files,
)

__all__ = ["LLAMA_MODELS"]

LLAMA31 = [
    ModelSpec(
        vendor="Meta",
        family="Llama-3.1",
        name="Llama-3.1-8B-Instruct",
        size="8B",
        quantization=None,
        repo="meta-llama/Llama-3.1-8B-Instruct",
        config_type=HFLlamaConfig,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(4),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
]


def _tokenizer_files_from_another_repo(repo: str) -> tuple[TokenizerFileSpec, ...]:
    return tuple(
        replace(spec, repo=repo) for spec in (*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE)
    )


LLAMA32 = [
    # LLAMA
    ModelSpec(
        vendor="Meta",
        family="Llama-3.2",
        name="Llama-3.2-1B-Instruct",
        size="1B",
        quantization=None,
        repo="meta-llama/Llama-3.2-1B-Instruct",
        config_type=HFLlamaConfig,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(1),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
    # ModelSpec(
    #     vendor="Meta",
    #     family="Llama-3.2",
    #     name="Llama-3.2-1B-Instruct-QLoRA",
    #     size="1B",
    #     quantization=QuantizationMode.UINT4,
    #     repo="meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8",
    #     config_type=ETLlamaConfig,
    #     config_file_name="params.json",
    #     weights_file_names=("consolidated.00.pth",),
    #     weights_type=WeightsType.TORCH,
    #     tokenizer_files=_tokenizer_files_from_another_repo("meta-llama/Llama-3.2-1B-Instruct"),
    #     use_cases=tuple(),
    # ),
    ModelSpec(
        vendor="Meta",
        family="Llama-3.2",
        name="Llama-3.2-3B-Instruct",
        size="3B",
        quantization=None,
        repo="meta-llama/Llama-3.2-3B-Instruct",
        config_type=HFLlamaConfig,
        config_file_name="config.json",
        weights_file_names=huggingface_weight_files(2),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=tuple(),
    ),
    # ModelSpec(
    #     vendor="Meta",
    #     family="Llama-3.2",
    #     name="Llama-3.2-3B-Instruct-QLoRA",
    #     size="3B",
    #     quantization=QuantizationMode.UINT4,
    #     repo="meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8",
    #     config_type=ETLlamaConfig,
    #     config_file_name="params.json",
    #     weights_file_names=("consolidated.00.pth",),
    #     tokenizer_files=_tokenizer_files_from_another_repo("meta-llama/Llama-3.2-3B-Instruct"),
    #     weights_type=WeightsType.TORCH,
    #     use_cases=tuple(),
    # ),
]

LLAMA_MODELS = LLAMA31 + LLAMA32
