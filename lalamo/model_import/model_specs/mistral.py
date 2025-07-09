from dataclasses import replace

from lalamo.model_import.configs import HFMistralConfig

from .common import (
    HUGGINFACE_GENERATION_CONFIG_FILE,
    HUGGINGFACE_TOKENIZER_FILES,
    ModelSpec,
    TokenizerFileSpec,
    UseCase,
    WeightsType,
    huggingface_weight_files,
)

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
        tokenizer_files=(*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE),
        use_cases=(UseCase.CODE,),
    ),
]


def _tokenizer_files_from_another_repo(repo: str) -> tuple[TokenizerFileSpec, ...]:
    return tuple(
        replace(spec, repo=repo) for spec in (*HUGGINGFACE_TOKENIZER_FILES, HUGGINFACE_GENERATION_CONFIG_FILE)
    )


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
        tokenizer_files=_tokenizer_files_from_another_repo("mistralai/Mistral-Small-3.1-24B-Base-2503"),
        use_cases=(UseCase.CODE,),
    ),
]


MISTRAL_MODELS = CODESTRAL + DEVSTRAL
