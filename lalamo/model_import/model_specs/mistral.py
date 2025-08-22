from lalamo.model_import.decoder_configs import HFMistralConfig

from .common import (
    ConfigMap,
    FileSpec,
    ModelSpec,
    UseCase,
    WeightsType,
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
        weights_type=WeightsType.SAFETENSORS,
        use_cases=(UseCase.CODE,),
    ),
]


DEVSTRAL_TOKENIZER_REPO = "mistralai/Mistral-Small-3.1-24B-Base-2503"

DEVSTRAL = [
    ModelSpec(
        vendor="Mistral",
        family="Devstral",
        name="Devstral-Small-2505",
        size="24B",
        quantization=None,
        repo="mistralai/Devstral-Small-2505",
        config_type=HFMistralConfig,
        configs=ConfigMap(
            tokenizer=FileSpec(repo=DEVSTRAL_TOKENIZER_REPO, filename="tokenizer.json"),
            tokenizer_config=FileSpec(repo=DEVSTRAL_TOKENIZER_REPO, filename="tokenizer_config.json"),
            generation_config=FileSpec(repo=DEVSTRAL_TOKENIZER_REPO, filename="generation_config.json"),
        ),
        use_cases=(UseCase.CODE,),
    ),
]


MISTRAL_MODELS = CODESTRAL + DEVSTRAL
