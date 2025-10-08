from lalamo.model_import.decoder_configs import HFGPTOssConfig

from .common import ConfigMap, FileSpec, ModelSpec, WeightsType

__all__ = ["GPT_OSS_MODELS"]

GPT_OSS_MODELS = [
    ModelSpec(
        vendor="OpenAI",
        family="GPT-OSS",
        name="GPT-OSS-20B",
        size="20B",
        quantization=None,
        repo="openai/gpt-oss-20b",
        config_type=HFGPTOssConfig,
        weights_type=WeightsType.SAFETENSORS,
        configs=ConfigMap(
            chat_template=FileSpec("chat_template.jinja"),
        ),
    ),
]
