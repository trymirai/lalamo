from lalamo.model_import.model_configs import HFPhi3Config
from lalamo.model_import.model_spec import LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin

__all__ = ["PHI_MODELS"]

PHI_MODELS = [
    LanguageModelSpec(
        vendor="Microsoft",
        family="Phi-4",
        name="Phi-4-mini-instruct",
        size="3.8B",
        origin=HuggingFaceOrigin(repo="microsoft/Phi-4-mini-instruct"),
        config_type=HFPhi3Config,
    ),
]
