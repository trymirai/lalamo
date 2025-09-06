from lalamo.model_import.decoder_configs import HFLlamaConfig

from .common import ModelSpec

__all__ = ["PLEIAS_MODELS"]


PLEIAS_MODELS = [
    ModelSpec(
        vendor="PleIAs",
        family="Pleias-RAG",
        name="Pleias-RAG-1B",
        size="1B",
        quantization=None,
        repo="PleIAs/Pleias-RAG-1B",
        config_type=HFLlamaConfig,
    ),
]
