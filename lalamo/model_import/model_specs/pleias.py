from lalamo.model_import.model_configs import HFLlamaConfig

from .common import ModelSpec
from .origins import HuggingFaceOrigin

__all__ = ["PLEIAS_MODELS"]


PLEIAS_MODELS = [
    ModelSpec(
        vendor="PleIAs",
        family="Pleias-RAG",
        name="Pleias-RAG-1B",
        size="1B",
        quantization=None,
        origin=HuggingFaceOrigin(repo="PleIAs/Pleias-RAG-1B"),
        config_type=HFLlamaConfig,
    ),
]
