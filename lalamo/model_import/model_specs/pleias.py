from lalamo.model_import.model_configs import HFLlamaConfig
from lalamo.model_import.model_spec import LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin

__all__ = ["PLEIAS_MODELS"]


PLEIAS_MODELS = [
    LanguageModelSpec(
        vendor="PleIAs",
        family="Pleias-RAG",
        name="Pleias-RAG-1B",
        size="1B",
        origin=HuggingFaceOrigin(repo="PleIAs/Pleias-RAG-1B"),
        config_type=HFLlamaConfig,
    ),
]
