from lalamo.model_import.model_configs import HFGPTOssConfig

from .common import ConfigMap, FileSpec, ModelSpec
from .origins import HuggingFaceOrigin

__all__ = ["GPT_OSS_MODELS"]

GPT_OSS_MODELS = [
    ModelSpec(
        vendor="OpenAI",
        family="GPT-OSS",
        name="GPT-OSS-20B",
        size="20B",
        quantization=None,
        origin=HuggingFaceOrigin(repo="openai/gpt-oss-20b"),
        config_type=HFGPTOssConfig,
        configs=ConfigMap(
            chat_template=FileSpec("chat_template.jinja"),
        ),
    ),
]
