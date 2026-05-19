from lalamo.model_import.model_configs import HFMistralConfig
from lalamo.model_import.model_spec import ConfigMap, FileSpec, JSONFieldSpec, LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin

__all__ = ["MISTRAL_MODELS"]

CODESTRAL_TOKENIZER_REPO = "mistralai/Codestral-22B-v0.1"

CODESTRAL = [
    LanguageModelSpec(
        vendor="Mistral",
        family="Codestral",
        name="Codestral-22B-v0.1",
        size="22B",
        origin=HuggingFaceOrigin(repo="mistralai/Codestral-22B-v0.1"),
        config_type=HFMistralConfig,
        configs=ConfigMap(
            tokenizer_config=FileSpec(repo=CODESTRAL_TOKENIZER_REPO, filename="tokenizer_config.json"),
        ),
    ),
]


DEVSTRAL_TOKENIZER_REPO = "mistralai/Mistral-Small-3.1-24B-Base-2503"
DEVSTRAL_CHAT_REPO = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

DEVSTRAL = [
    LanguageModelSpec(
        vendor="Mistral",
        family="Devstral",
        name="Devstral-Small-2505",
        size="24B",
        origin=HuggingFaceOrigin(repo="mistralai/Devstral-Small-2505"),
        config_type=HFMistralConfig,
        configs=ConfigMap(
            tokenizer=FileSpec(repo=DEVSTRAL_TOKENIZER_REPO, filename="tokenizer.json"),
            tokenizer_config=FileSpec(repo=DEVSTRAL_TOKENIZER_REPO, filename="tokenizer_config.json"),
            generation_config=FileSpec(repo=DEVSTRAL_TOKENIZER_REPO, filename="generation_config.json"),
            chat_template=JSONFieldSpec(
                FileSpec(repo=DEVSTRAL_CHAT_REPO, filename="chat_template.json"),
                "chat_template",
            ),
            system_prompt=FileSpec("SYSTEM_PROMPT.txt"),
        ),
    ),
]


MISTRAL_MODELS = CODESTRAL + DEVSTRAL
