from lalamo.model_import.model_configs import HFLlamaConfig, HFSmolLM3Config

from .common import ConfigMap, FileSpec, ModelSpec

__all__ = ["HUGGINGFACE_MODELS"]

HUGGINGFACE_MODELS = [
    ModelSpec(
        vendor="HuggingFace",
        family="SmolLM2",
        name="SmolLM2-1.7B-Instruct",
        size="1.7B",
        quantization=None,
        repo="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        config_type=HFLlamaConfig,
        use_cases=tuple(),
    ),
    ModelSpec(
        vendor="HuggingFace",
        family="SmolLM3",
        name="SmolLM3-3B",
        size="3B",
        quantization=None,
        repo="HuggingFaceTB/SmolLM3-3B",
        config_type=HFSmolLM3Config,
        configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
        use_cases=tuple(),
    ),
]
