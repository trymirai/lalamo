from lalamo.model_import.model_configs import HFMobileMoEConfig
from lalamo.model_import.model_spec import LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin

__all__ = ["MOBILEMOE_MODELS"]


MOBILEMOE_MODELS = [
    LanguageModelSpec(
        vendor="Meta",
        family="MobileMoE",
        name="MobileMoE-S-QAT",
        size="1.3B",
        origin=HuggingFaceOrigin(repo="facebook/MobileMoE-S-QAT"),
        config_type=HFMobileMoEConfig,
    ),
    LanguageModelSpec(
        vendor="Meta",
        family="MobileMoE",
        name="MobileMoE-M-QAT",
        size="2.8B",
        origin=HuggingFaceOrigin(repo="facebook/MobileMoE-M-QAT"),
        config_type=HFMobileMoEConfig,
    ),
    LanguageModelSpec(
        vendor="Meta",
        family="MobileMoE",
        name="MobileMoE-L-QAT",
        size="5.3B",
        origin=HuggingFaceOrigin(repo="facebook/MobileMoE-L-QAT"),
        config_type=HFMobileMoEConfig,
    ),
]
